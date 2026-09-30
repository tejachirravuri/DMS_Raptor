"""
Offline training for contextual bandit agents (LinUCB, Thompson Sampling).

Training loop:
  1. Load pre-collected dataset (features + both models' outputs)
  2. For each frame, let the agent choose an action
  3. Compute oracle reward using both models' data
  4. Update the agent
  5. Track cumulative reward, regret, and action distribution

Supports:
  - Multiple epochs over the dataset (with shuffling)
  - Per-epoch evaluation on held-out videos
  - Logging of training curves
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import RLConfig, FEATURE_DIM, ACTION_N, ACTION_S
from ..agents.base import BaseAgent
from ..data.dataset import SwitchingDataset
from ..rewards import (
    reward_oracle_iou,
    reward_confidence,
    compute_set_iou,
)


class BanditTrainer:
    """Train a contextual bandit agent on offline data.

    Parameters
    ----------
    agent : BaseAgent
        The bandit agent to train (LinUCB or ThompsonSampling).
    cfg : RLConfig
        Configuration.
    train_dataset : SwitchingDataset
        Training data (pre-collected dual-model outputs).
    eval_dataset : SwitchingDataset, optional
        Held-out evaluation data.
    """

    def __init__(
        self,
        agent: BaseAgent,
        cfg: RLConfig,
        train_dataset: SwitchingDataset,
        eval_dataset: Optional[SwitchingDataset] = None,
    ):
        self.agent = agent
        self.cfg = cfg
        self.train_data = train_dataset
        self.eval_data = eval_dataset

        # Training logs
        self.train_log: List[Dict] = []
        self.eval_log: List[Dict] = []

    def train(
        self,
        num_epochs: int = 10,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict:
        """Run offline bandit training.

        Returns
        -------
        dict
            Training summary with per-epoch metrics.
        """
        cfg = self.cfg
        best_eval_reward = -np.inf
        best_epoch = 0

        for epoch in range(num_epochs):
            t0 = time.time()

            # Shuffle data each epoch
            indices = self.train_data.shuffled_indices(seed=seed + epoch)

            epoch_rewards = []
            epoch_actions = []
            epoch_correct = 0  # frames where agent matches optimal action
            prev_action = ACTION_N

            for idx in indices:
                sample = self.train_data[int(idx)]
                features = sample["features"]
                iou = sample["iou"]
                t_n = sample["t_infer_n"]
                t_s = sample["t_infer_s"]

                # Agent chooses
                action = self.agent.choose(features)

                # Compute reward (oracle: we have both models' outputs)
                n_sufficient = iou >= cfg.iou_threshold
                if action == ACTION_N:
                    if n_sufficient:
                        reward = cfg.reward_correct_n + (t_s - t_n) * cfg.lambda_latency
                    else:
                        reward = cfg.reward_missed_n
                else:
                    if n_sufficient:
                        reward = cfg.reward_wasteful_s
                    else:
                        reward = cfg.reward_needed_s

                # Switch penalty
                if action != prev_action:
                    reward -= cfg.switch_penalty

                # Update agent
                self.agent.update(features, action, reward)

                epoch_rewards.append(reward)
                epoch_actions.append(action)

                # Track optimality
                optimal = ACTION_N if n_sufficient else ACTION_S
                if action == optimal:
                    epoch_correct += 1

                prev_action = action

            # Epoch stats
            epoch_time = time.time() - t0
            n_s = sum(1 for a in epoch_actions if a == ACTION_S)
            log_entry = {
                "epoch": epoch,
                "mean_reward": float(np.mean(epoch_rewards)),
                "total_reward": float(np.sum(epoch_rewards)),
                "accuracy": epoch_correct / len(indices),
                "s_usage": n_s / len(indices),
                "n_usage": 1.0 - n_s / len(indices),
                "time_s": epoch_time,
            }
            self.train_log.append(log_entry)

            # Evaluation
            eval_entry = None
            if self.eval_data is not None and len(self.eval_data) > 0:
                eval_entry = self._evaluate(epoch)
                self.eval_log.append(eval_entry)

                if eval_entry["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_entry["mean_reward"]
                    best_epoch = epoch

            if verbose:
                msg = (
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"reward={log_entry['mean_reward']:.4f} | "
                    f"acc={log_entry['accuracy']:.3f} | "
                    f"s%={log_entry['s_usage']:.3f} | "
                    f"time={epoch_time:.1f}s"
                )
                if eval_entry:
                    msg += f" | eval_reward={eval_entry['mean_reward']:.4f}"
                print(msg)

        return {
            "agent": self.agent.name,
            "num_epochs": num_epochs,
            "train_frames": len(self.train_data),
            "eval_frames": len(self.eval_data) if self.eval_data else 0,
            "best_epoch": best_epoch,
            "best_eval_reward": float(best_eval_reward),
            "train_log": self.train_log,
            "eval_log": self.eval_log,
            "final_params": self.agent.get_params(),
        }

    def _evaluate(self, epoch: int) -> Dict:
        """Evaluate the agent on held-out data (greedy, no exploration)."""
        cfg = self.cfg
        rewards = []
        actions = []
        correct = 0
        prev_action = ACTION_N

        for idx in range(len(self.eval_data)):
            sample = self.eval_data[idx]
            features = sample["features"]
            iou = sample["iou"]
            t_n = sample["t_infer_n"]
            t_s = sample["t_infer_s"]

            action = self.agent.greedy_choose(features)

            n_sufficient = iou >= cfg.iou_threshold
            if action == ACTION_N:
                if n_sufficient:
                    reward = cfg.reward_correct_n + (t_s - t_n) * cfg.lambda_latency
                else:
                    reward = cfg.reward_missed_n
            else:
                if n_sufficient:
                    reward = cfg.reward_wasteful_s
                else:
                    reward = cfg.reward_needed_s

            if action != prev_action:
                reward -= cfg.switch_penalty

            rewards.append(reward)
            actions.append(action)
            optimal = ACTION_N if n_sufficient else ACTION_S
            if action == optimal:
                correct += 1
            prev_action = action

        n_s = sum(1 for a in actions if a == ACTION_S)
        return {
            "epoch": epoch,
            "mean_reward": float(np.mean(rewards)),
            "accuracy": correct / max(1, len(self.eval_data)),
            "s_usage": n_s / max(1, len(self.eval_data)),
        }

    def save_results(self, output_dir: str) -> str:
        """Save training logs and agent checkpoint."""
        os.makedirs(output_dir, exist_ok=True)

        # Save agent
        agent_path = os.path.join(output_dir, f"{self.agent.name}_agent.npz")
        self.agent.save(agent_path)

        # Save training log
        log_path = os.path.join(output_dir, f"{self.agent.name}_training_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "train_log": self.train_log,
                "eval_log": self.eval_log,
                "agent_params": self.agent.get_params(),
                "config": self.cfg.to_dict(),
            }, f, indent=2)

        # Save feature importance (if available)
        if hasattr(self.agent, "get_feature_importance"):
            imp = self.agent.get_feature_importance()
            imp_path = os.path.join(output_dir, f"{self.agent.name}_feature_importance.json")
            with open(imp_path, "w") as f:
                json.dump(imp, f, indent=2)

        return output_dir
