"""
DQN training loop for temporal model switching.

Trains the DQN agent on video episodes:
  - Each video is one episode
  - State = feature vector (with temporal features like dwell, prev_action)
  - Training uses experience replay with target network
  - Evaluation runs greedy policy on held-out videos
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from ..config import RLConfig, FEATURE_DIM, ACTION_N, ACTION_S
from ..agents.dqn import DQNAgent
from ..data.dataset import SwitchingDataset
from ..data.replay_buffer import ReplayBuffer


class DQNTrainer:
    """Train a DQN agent on video episode data.

    Parameters
    ----------
    agent : DQNAgent
        The DQN agent to train.
    cfg : RLConfig
        Configuration.
    train_dataset : SwitchingDataset
        Training data (pre-collected).
    eval_dataset : SwitchingDataset, optional
        Held-out evaluation data.
    """

    def __init__(
        self,
        agent: DQNAgent,
        cfg: RLConfig,
        train_dataset: SwitchingDataset,
        eval_dataset: Optional[SwitchingDataset] = None,
    ):
        self.agent = agent
        self.cfg = cfg
        self.train_data = train_dataset
        self.eval_data = eval_dataset

        self.buffer = ReplayBuffer(
            capacity=cfg.dqn_buffer_size,
            state_dim=cfg.dqn_state_dim,
        )

        self.train_log: List[Dict] = []
        self.eval_log: List[Dict] = []
        self.loss_history: List[float] = []

    def _build_state(
        self,
        features: np.ndarray,
        prev_action: int,
        dwell: int,
    ) -> np.ndarray:
        """Build DQN state from features + temporal context.

        The feature vector already contains prev_conf, prev_n_dets,
        prev_action, and dwell_time in its last 4 positions.
        We update them with the actual current temporal state.
        """
        state = features.copy()
        # Overwrite temporal features with actual values
        state[-2] = float(prev_action)   # prev_action slot
        state[-1] = float(min(dwell, 100)) / 100.0  # normalized dwell
        return state

    def _compute_reward(
        self,
        action: int,
        iou: float,
        t_n: float,
        t_s: float,
        prev_action: int,
    ) -> float:
        """Compute reward from pre-collected data."""
        cfg = self.cfg
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

        return reward

    def train(
        self,
        num_episodes: int = 50,
        verbose: bool = True,
    ) -> Dict:
        """Run DQN training over video episodes.

        Each episode processes one video sequentially.
        Training steps happen after each frame (once buffer is warm).

        Returns
        -------
        dict
            Training summary.
        """
        cfg = self.cfg
        self.agent.train_mode()

        train_video_ids = list(self.train_data.video_data.keys())
        if not train_video_ids:
            raise ValueError("No training videos loaded")

        best_eval_reward = -np.inf
        best_episode = 0
        total_steps = 0

        for episode in range(num_episodes):
            t0 = time.time()

            # Sample a random video for this episode
            vid_id = train_video_ids[episode % len(train_video_ids)]
            ep_data = self.train_data.get_video_episode(vid_id)

            episode_rewards = []
            episode_actions = []
            episode_losses = []
            prev_action = ACTION_N
            dwell = 0

            n_frames = ep_data["n_frames"]
            for i in range(n_frames):
                features = ep_data["features"][i]
                iou = float(ep_data["iou"][i])
                t_n = float(ep_data["t_infer_n"][i])
                t_s = float(ep_data["t_infer_s"][i])

                state = self._build_state(features, prev_action, dwell)

                # Agent chooses
                action = self.agent.choose(state)

                # Compute reward
                reward = self._compute_reward(action, iou, t_n, t_s, prev_action)

                # Next state
                done = (i == n_frames - 1)
                if not done:
                    next_features = ep_data["features"][i + 1]
                    new_dwell = dwell + 1 if action == prev_action else 1
                    next_state = self._build_state(next_features, action, new_dwell)
                else:
                    next_state = state  # terminal; won't be used

                # Store transition
                self.buffer.add(state, action, reward, next_state, done)

                # Train step
                if self.buffer.is_ready(cfg.dqn_batch_size):
                    batch = self.buffer.sample(cfg.dqn_batch_size)
                    loss = self.agent.train_step(*batch)
                    episode_losses.append(loss)
                    self.loss_history.append(loss)

                episode_rewards.append(reward)
                episode_actions.append(action)

                # Update temporal state
                if action == prev_action:
                    dwell += 1
                else:
                    dwell = 1
                prev_action = action
                total_steps += 1

            # Episode stats
            n_s = sum(1 for a in episode_actions if a == ACTION_S)
            n_switches = sum(
                1 for i in range(1, len(episode_actions))
                if episode_actions[i] != episode_actions[i-1]
            )
            log_entry = {
                "episode": episode,
                "video_id": vid_id,
                "mean_reward": float(np.mean(episode_rewards)),
                "total_reward": float(np.sum(episode_rewards)),
                "s_usage": n_s / max(1, n_frames),
                "switches": n_switches,
                "sw_per_100": n_switches / max(1, n_frames) * 100,
                "mean_loss": float(np.mean(episode_losses)) if episode_losses else 0.0,
                "epsilon": self.agent.epsilon,
                "buffer_size": len(self.buffer),
                "time_s": time.time() - t0,
            }
            self.train_log.append(log_entry)

            # Evaluate every 5 episodes
            eval_entry = None
            if self.eval_data and (episode + 1) % 5 == 0:
                eval_entry = self._evaluate(episode)
                self.eval_log.append(eval_entry)
                if eval_entry["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_entry["mean_reward"]
                    best_episode = episode

            if verbose:
                msg = (
                    f"Ep {episode+1}/{num_episodes} [{vid_id}] | "
                    f"reward={log_entry['mean_reward']:.4f} | "
                    f"s%={log_entry['s_usage']:.3f} | "
                    f"sw/100={log_entry['sw_per_100']:.1f} | "
                    f"loss={log_entry['mean_loss']:.4f} | "
                    f"eps={log_entry['epsilon']:.3f}"
                )
                if eval_entry:
                    msg += f" | eval={eval_entry['mean_reward']:.4f}"
                print(msg)

        return {
            "agent": "dqn",
            "num_episodes": num_episodes,
            "total_steps": total_steps,
            "best_episode": best_episode,
            "best_eval_reward": float(best_eval_reward),
            "train_log": self.train_log,
            "eval_log": self.eval_log,
            "final_params": self.agent.get_params(),
        }

    def _evaluate(self, episode: int) -> Dict:
        """Evaluate DQN agent on held-out videos (greedy)."""
        self.agent.eval_mode()

        all_rewards = []
        all_actions = []
        all_correct = 0
        total_frames = 0

        for vid_id in self.eval_data.video_data:
            ep_data = self.eval_data.get_video_episode(vid_id)
            prev_action = ACTION_N
            dwell = 0

            for i in range(ep_data["n_frames"]):
                features = ep_data["features"][i]
                iou = float(ep_data["iou"][i])
                t_n = float(ep_data["t_infer_n"][i])
                t_s = float(ep_data["t_infer_s"][i])

                state = self._build_state(features, prev_action, dwell)
                action = self.agent.greedy_choose(state)
                reward = self._compute_reward(action, iou, t_n, t_s, prev_action)

                all_rewards.append(reward)
                all_actions.append(action)

                n_sufficient = iou >= self.cfg.iou_threshold
                optimal = ACTION_N if n_sufficient else ACTION_S
                if action == optimal:
                    all_correct += 1
                total_frames += 1

                if action == prev_action:
                    dwell += 1
                else:
                    dwell = 1
                prev_action = action

        self.agent.train_mode()

        n_s = sum(1 for a in all_actions if a == ACTION_S)
        return {
            "episode": episode,
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "accuracy": all_correct / max(1, total_frames),
            "s_usage": n_s / max(1, total_frames),
            "total_frames": total_frames,
        }

    def save_results(self, output_dir: str) -> str:
        """Save DQN checkpoint and training logs."""
        os.makedirs(output_dir, exist_ok=True)

        # Save agent
        agent_path = os.path.join(output_dir, "dqn_agent.pt")
        self.agent.save(agent_path)

        # Save training log
        log_path = os.path.join(output_dir, "dqn_training_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "train_log": self.train_log,
                "eval_log": self.eval_log,
                "loss_history_sample": self.loss_history[::100],  # every 100th
                "agent_params": self.agent.get_params(),
                "config": self.cfg.to_dict(),
            }, f, indent=2)

        return output_dir
