"""
Policy evaluator: compare RL agents against heuristic baselines.

Runs each policy (RL agents + heuristic simulations) on the same
pre-collected dataset and produces a unified comparison table.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from ..config import RLConfig, FEATURE_DIM, ACTION_N, ACTION_S
from ..agents.base import BaseAgent
from ..data.dataset import SwitchingDataset
from .metrics import SwitchingMetrics, compute_metrics


class HeuristicPolicy:
    """Simulate one of the 8 existing DMS-Raptor heuristic policies
    on pre-collected data (for comparison with RL agents).

    Only n_only, s_only, and conf_ema are simulated here because
    the others require rolling window state that is already baked into
    the features. For the full heuristic comparison, use the real
    StreamingEngine results from research scripts 02/03/09.
    """

    def __init__(self, name: str):
        self.name = name

    def evaluate(
        self,
        dataset: SwitchingDataset,
        video_id: str,
        cfg: RLConfig,
    ) -> SwitchingMetrics:
        """Run the heuristic on one video's data."""
        ep = dataset.get_video_episode(video_id)
        n_frames = ep["n_frames"]
        actions = np.zeros(n_frames, dtype=np.int32)
        rewards = np.zeros(n_frames, dtype=np.float32)

        if self.name == "n_only":
            actions[:] = ACTION_N
        elif self.name == "s_only":
            actions[:] = ACTION_S
        elif self.name == "oracle":
            # Perfect hindsight: pick n when sufficient, s when needed
            actions = np.where(
                ep["iou"] >= cfg.iou_threshold, ACTION_N, ACTION_S
            ).astype(np.int32)
        elif self.name == "conf_ema":
            # Simulate conf_ema with default parameters
            fast_beta, slow_beta = 0.30, 0.02
            c_high, c_low = 0.12, 0.04
            fast_ema = 0.0
            slow_ema = 0.0
            last_choice = ACTION_N
            for i in range(n_frames):
                conf = float(ep["conf_n"][i])
                fast_ema = fast_beta * conf + (1 - fast_beta) * fast_ema
                slow_ema = slow_beta * conf + (1 - slow_beta) * slow_ema
                drop = max(0, (slow_ema - fast_ema) / (slow_ema + 1e-9))

                if i < 5:
                    choice = ACTION_N
                elif last_choice == ACTION_N:
                    choice = ACTION_S if drop >= c_high else ACTION_N
                else:
                    choice = ACTION_N if drop <= c_low else ACTION_S

                actions[i] = choice
                last_choice = choice
        else:
            raise ValueError(f"Unknown heuristic: {self.name}")

        # Compute rewards
        prev_action = ACTION_N
        for i in range(n_frames):
            iou = float(ep["iou"][i])
            t_n = float(ep["t_infer_n"][i])
            t_s = float(ep["t_infer_s"][i])
            n_sufficient = iou >= cfg.iou_threshold

            a = actions[i]
            if a == ACTION_N:
                if n_sufficient:
                    r = cfg.reward_correct_n + (t_s - t_n) * cfg.lambda_latency
                else:
                    r = cfg.reward_missed_n
            else:
                if n_sufficient:
                    r = cfg.reward_wasteful_s
                else:
                    r = cfg.reward_needed_s

            if a != prev_action:
                r -= cfg.switch_penalty
            rewards[i] = r
            prev_action = a

        return compute_metrics(
            actions=actions,
            ious=ep["iou"],
            t_infer_n=ep["t_infer_n"],
            t_infer_s=ep["t_infer_s"],
            rewards=rewards,
            iou_threshold=cfg.iou_threshold,
            video_id=video_id,
            policy_name=self.name,
        )


class PolicyEvaluator:
    """Compare RL agents against heuristic baselines.

    Usage::

        evaluator = PolicyEvaluator(cfg, dataset)
        evaluator.add_agent("linucb", linucb_agent)
        evaluator.add_agent("dqn", dqn_agent)
        results = evaluator.evaluate_all()
        evaluator.print_comparison_table(results)
    """

    def __init__(self, cfg: RLConfig, dataset: SwitchingDataset):
        self.cfg = cfg
        self.dataset = dataset
        self.agents: Dict[str, BaseAgent] = {}
        self.heuristics = ["n_only", "s_only", "conf_ema", "oracle"]

    def add_agent(self, name: str, agent: BaseAgent) -> None:
        self.agents[name] = agent

    def evaluate_all(
        self,
        video_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[SwitchingMetrics]]:
        """Evaluate all policies on all videos.

        Returns
        -------
        dict
            policy_name -> list of SwitchingMetrics (one per video).
        """
        if video_ids is None:
            video_ids = list(self.dataset.video_data.keys())

        results: Dict[str, List[SwitchingMetrics]] = {}

        # Heuristic baselines
        for h_name in self.heuristics:
            hp = HeuristicPolicy(h_name)
            metrics_list = []
            for vid_id in video_ids:
                m = hp.evaluate(self.dataset, vid_id, self.cfg)
                metrics_list.append(m)
            results[h_name] = metrics_list

        # RL agents
        for agent_name, agent in self.agents.items():
            metrics_list = []
            for vid_id in video_ids:
                m = self._evaluate_agent(agent, agent_name, vid_id)
                metrics_list.append(m)
            results[agent_name] = metrics_list

        return results

    def _evaluate_agent(
        self,
        agent: BaseAgent,
        agent_name: str,
        video_id: str,
    ) -> SwitchingMetrics:
        """Evaluate a single RL agent on one video."""
        cfg = self.cfg
        ep = self.dataset.get_video_episode(video_id)
        n_frames = ep["n_frames"]

        actions = np.zeros(n_frames, dtype=np.int32)
        rewards = np.zeros(n_frames, dtype=np.float32)
        prev_action = ACTION_N
        dwell = 0

        for i in range(n_frames):
            features = ep["features"][i]

            # For DQN: update temporal features in the state
            state = features.copy()
            state[-2] = float(prev_action)
            state[-1] = float(min(dwell, 100)) / 100.0

            action = agent.greedy_choose(state)
            actions[i] = action

            # Reward
            iou = float(ep["iou"][i])
            t_n = float(ep["t_infer_n"][i])
            t_s = float(ep["t_infer_s"][i])
            n_sufficient = iou >= cfg.iou_threshold

            if action == ACTION_N:
                if n_sufficient:
                    r = cfg.reward_correct_n + (t_s - t_n) * cfg.lambda_latency
                else:
                    r = cfg.reward_missed_n
            else:
                if n_sufficient:
                    r = cfg.reward_wasteful_s
                else:
                    r = cfg.reward_needed_s

            if action != prev_action:
                r -= cfg.switch_penalty
            rewards[i] = r

            if action == prev_action:
                dwell += 1
            else:
                dwell = 1
            prev_action = action

        return compute_metrics(
            actions=actions,
            ious=ep["iou"],
            t_infer_n=ep["t_infer_n"],
            t_infer_s=ep["t_infer_s"],
            rewards=rewards,
            iou_threshold=cfg.iou_threshold,
            video_id=video_id,
            policy_name=agent_name,
            t_scene_overhead=0.1,  # ~0.1ms for feature extraction + agent
        )

    @staticmethod
    def aggregate_metrics(
        metrics_list: List[SwitchingMetrics],
    ) -> Dict[str, float]:
        """Aggregate per-video metrics into summary statistics."""
        if not metrics_list:
            return {}
        return {
            "mean_routing_accuracy": float(np.mean([m.routing_accuracy for m in metrics_list])),
            "mean_s_usage": float(np.mean([m.s_usage for m in metrics_list])),
            "mean_latency_ms": float(np.mean([m.mean_latency_ms for m in metrics_list])),
            "mean_latency_reduction": float(np.mean([m.latency_reduction for m in metrics_list])),
            "mean_sw_per_100": float(np.mean([m.sw_per_100 for m in metrics_list])),
            "mean_reward": float(np.mean([m.mean_reward for m in metrics_list])),
            "std_reward": float(np.std([m.mean_reward for m in metrics_list])),
            "mean_dwell": float(np.mean([m.mean_dwell for m in metrics_list])),
            "n_videos": len(metrics_list),
        }

    def print_comparison_table(
        self,
        results: Dict[str, List[SwitchingMetrics]],
    ) -> str:
        """Format a comparison table as a string."""
        header = (
            f"{'Policy':<16} {'Accuracy':>8} {'s%':>6} "
            f"{'Latency':>10} {'Reduction':>10} {'sw/100':>7} "
            f"{'Reward':>8} {'Dwell':>7}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for policy_name, metrics_list in results.items():
            agg = self.aggregate_metrics(metrics_list)
            if not agg:
                continue
            line = (
                f"{policy_name:<16} "
                f"{agg['mean_routing_accuracy']:>8.3f} "
                f"{agg['mean_s_usage']:>5.1%} "
                f"{agg['mean_latency_ms']:>8.1f}ms "
                f"{agg['mean_latency_reduction']:>9.1%} "
                f"{agg['mean_sw_per_100']:>7.2f} "
                f"{agg['mean_reward']:>8.4f} "
                f"{agg['mean_dwell']:>7.1f}"
            )
            lines.append(line)

        table = "\n".join(lines)
        print(table)
        return table

    def save_results(
        self,
        results: Dict[str, List[SwitchingMetrics]],
        output_dir: str,
    ) -> str:
        """Save full evaluation results to JSON."""
        os.makedirs(output_dir, exist_ok=True)

        output = {}
        for policy_name, metrics_list in results.items():
            output[policy_name] = {
                "per_video": [m.to_dict() for m in metrics_list],
                "aggregate": self.aggregate_metrics(metrics_list),
            }

        path = os.path.join(output_dir, "rl_vs_heuristic_comparison.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        return path
