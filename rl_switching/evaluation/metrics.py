"""
Metrics for evaluating model switching policies.

Computes coverage, latency, switching stability, and composite scores
that directly compare with the existing 8 heuristic policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..config import ACTION_N, ACTION_S


@dataclass
class SwitchingMetrics:
    """Comprehensive metrics for a switching policy evaluated on one video.

    Matches the metrics computed by DMS-Raptor's research scripts
    so results are directly comparable.
    """
    video_id: str = ""
    policy_name: str = ""
    n_frames: int = 0

    # ── Detection coverage (pseudo-oracle: s_only as reference) ──────
    # These require running both models, so they come from the
    # pre-collected dataset where we have both n and s detections.
    coverage: float = 0.0       # fraction of s-detections matched by policy
    precision: float = 0.0      # of policy's detections, how many match s
    f1: float = 0.0

    # ── IoU-based accuracy ───────────────────────────────────────────
    routing_accuracy: float = 0.0  # fraction of frames optimally routed
    n_correct_n: int = 0           # n chosen correctly (IoU >= threshold)
    n_missed_n: int = 0            # n chosen wrongly (IoU < threshold)
    n_needed_s: int = 0            # s chosen correctly (IoU < threshold)
    n_wasteful_s: int = 0          # s chosen but n was sufficient

    # ── Timing ───────────────────────────────────────────────────────
    mean_latency_ms: float = 0.0   # estimated mean T_total
    p95_latency_ms: float = 0.0
    latency_reduction: float = 0.0 # vs s_only baseline

    # ── Model usage ──────────────────────────────────────────────────
    s_usage: float = 0.0          # fraction of frames using s-model
    n_usage: float = 0.0

    # ── Switching stability ──────────────────────────────────────────
    total_switches: int = 0
    sw_per_100: float = 0.0
    mean_dwell: float = 0.0       # average frames between switches

    # ── Reward ───────────────────────────────────────────────────────
    mean_reward: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


def compute_metrics(
    actions: np.ndarray,
    ious: np.ndarray,
    t_infer_n: np.ndarray,
    t_infer_s: np.ndarray,
    rewards: np.ndarray,
    iou_threshold: float = 0.5,
    video_id: str = "",
    policy_name: str = "",
    t_scene_overhead: float = 0.0,
) -> SwitchingMetrics:
    """Compute all switching metrics from action/outcome arrays.

    Parameters
    ----------
    actions : (N,) int
        Action taken per frame (0=n, 1=s).
    ious : (N,)
        IoU agreement between n and s models per frame.
    t_infer_n, t_infer_s : (N,)
        Inference times for each model per frame.
    rewards : (N,)
        Reward per frame.
    iou_threshold : float
        IoU threshold for n-sufficiency.
    t_scene_overhead : float
        Per-frame overhead in ms for the agent's own computation
        (feature extraction for the RL agent). Added to latency.
    """
    n_frames = len(actions)
    if n_frames == 0:
        return SwitchingMetrics(video_id=video_id, policy_name=policy_name)

    n_sufficient = ious >= iou_threshold
    optimal = np.where(n_sufficient, ACTION_N, ACTION_S)

    # Routing accuracy
    correct = (actions == optimal).sum()

    # Breakdown
    is_n = actions == ACTION_N
    is_s = actions == ACTION_S
    n_correct_n = int((is_n & n_sufficient).sum())
    n_missed_n = int((is_n & ~n_sufficient).sum())
    n_needed_s = int((is_s & ~n_sufficient).sum())
    n_wasteful_s = int((is_s & n_sufficient).sum())

    # Latency estimation
    latencies = np.where(
        is_n, t_infer_n, t_infer_s
    ) + t_scene_overhead
    mean_lat = float(latencies.mean())
    p95_lat = float(np.percentile(latencies, 95))
    s_only_lat = float(t_infer_s.mean())
    lat_reduction = 1.0 - mean_lat / max(1e-6, s_only_lat)

    # Switching
    switches = np.sum(actions[1:] != actions[:-1])
    sw_per_100 = float(switches) / max(1, n_frames) * 100

    # Mean dwell
    dwells = []
    current_dwell = 1
    for i in range(1, n_frames):
        if actions[i] == actions[i-1]:
            current_dwell += 1
        else:
            dwells.append(current_dwell)
            current_dwell = 1
    dwells.append(current_dwell)
    mean_dwell = float(np.mean(dwells))

    # S-usage
    s_count = int(is_s.sum())

    return SwitchingMetrics(
        video_id=video_id,
        policy_name=policy_name,
        n_frames=n_frames,
        routing_accuracy=int(correct) / n_frames,
        n_correct_n=n_correct_n,
        n_missed_n=n_missed_n,
        n_needed_s=n_needed_s,
        n_wasteful_s=n_wasteful_s,
        mean_latency_ms=mean_lat,
        p95_latency_ms=p95_lat,
        latency_reduction=lat_reduction,
        s_usage=s_count / n_frames,
        n_usage=1.0 - s_count / n_frames,
        total_switches=int(switches),
        sw_per_100=sw_per_100,
        mean_dwell=mean_dwell,
        mean_reward=float(rewards.mean()),
        total_reward=float(rewards.sum()),
    )
