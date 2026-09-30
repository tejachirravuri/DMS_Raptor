"""
Reward functions for RL-based model switching.

Three reward strategies:
  1. oracle_iou   — uses both models' outputs (training time only)
  2. confidence   — uses chosen model's confidence (deployable)
  3. hybrid       — weighted combination of IoU + confidence + latency
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import RLConfig, ACTION_N, ACTION_S


def compute_box_iou_matrix(
    dets_a: List[np.ndarray],
    dets_b: List[np.ndarray],
) -> np.ndarray:
    """Compute pairwise IoU between two detection lists.

    Each detection is [x1, y1, x2, y2, conf].
    Returns shape (len(dets_a), len(dets_b)).
    """
    if not dets_a or not dets_b:
        return np.zeros((len(dets_a), len(dets_b)), dtype=np.float32)

    a = np.array([d[:4] for d in dets_a], dtype=np.float32)  # (M, 4)
    b = np.array([d[:4] for d in dets_b], dtype=np.float32)  # (N, 4)

    # Intersection
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Areas
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def compute_set_iou(
    dets_n: List[np.ndarray],
    dets_s: List[np.ndarray],
    conf_threshold: float = 0.25,
) -> float:
    """Compute set-level agreement IoU between n-model and s-model detections.

    Filters detections by conf_threshold, then uses greedy matching
    to compute the average best-match IoU.  Returns 1.0 if both empty.
    """
    dets_n = [d for d in dets_n if float(d[4]) >= conf_threshold]
    dets_s = [d for d in dets_s if float(d[4]) >= conf_threshold]

    if not dets_n and not dets_s:
        return 1.0  # both empty → perfect agreement
    if not dets_n or not dets_s:
        return 0.0  # one has detections, other doesn't → total disagreement

    iou_mat = compute_box_iou_matrix(dets_n, dets_s)

    # Greedy match: for each s-detection, find best-matching n-detection
    matched_ious = []
    used_n = set()
    # Sort s-detections by confidence (highest first)
    s_order = sorted(range(len(dets_s)),
                     key=lambda i: float(dets_s[i][4]), reverse=True)
    for si in s_order:
        best_iou = 0.0
        best_ni = -1
        for ni in range(len(dets_n)):
            if ni not in used_n and iou_mat[ni, si] > best_iou:
                best_iou = iou_mat[ni, si]
                best_ni = ni
        if best_ni >= 0:
            used_n.add(best_ni)
        matched_ious.append(best_iou)

    return float(np.mean(matched_ious)) if matched_ious else 0.0


def reward_oracle_iou(
    action: int,
    dets_n: List[np.ndarray],
    dets_s: List[np.ndarray],
    t_infer_n: float,
    t_infer_s: float,
    prev_action: int,
    cfg: RLConfig,
) -> float:
    """Oracle reward: requires both models' outputs (training only).

    Measures whether the action was correct by comparing n vs s
    detection agreement.
    """
    iou = compute_set_iou(dets_n, dets_s, conf_threshold=0.25)
    n_sufficient = iou >= cfg.iou_threshold

    # Base quality reward
    if action == ACTION_N:
        if n_sufficient:
            quality = cfg.reward_correct_n
            latency_bonus = (t_infer_s - t_infer_n) * cfg.lambda_latency
        else:
            quality = cfg.reward_missed_n
            latency_bonus = 0.0
    else:  # ACTION_S
        if n_sufficient:
            quality = cfg.reward_wasteful_s
        else:
            quality = cfg.reward_needed_s
        latency_bonus = 0.0

    # Switch penalty
    sw_penalty = cfg.switch_penalty if action != prev_action else 0.0

    return quality + latency_bonus - sw_penalty


def reward_confidence(
    action: int,
    dets_chosen: List[np.ndarray],
    t_infer: float,
    t_infer_s_baseline: float,
    prev_action: int,
    cfg: RLConfig,
) -> float:
    """Confidence-based reward: uses only the chosen model's output.

    Deployable at inference time (doesn't need both models).
    """
    # Mean confidence of chosen model
    if dets_chosen:
        confs = [float(d[4]) for d in dets_chosen if float(d[4]) >= 0.25]
        quality = float(np.mean(confs)) if confs else 0.0
    else:
        quality = 0.0

    # Latency reward: bonus for using n (saved time)
    if action == ACTION_N:
        latency_bonus = (t_infer_s_baseline - t_infer) * cfg.lambda_latency
    else:
        latency_bonus = 0.0

    sw_penalty = cfg.switch_penalty if action != prev_action else 0.0

    return quality + latency_bonus - sw_penalty


def reward_hybrid(
    action: int,
    dets_n: List[np.ndarray],
    dets_s: List[np.ndarray],
    t_infer_n: float,
    t_infer_s: float,
    prev_action: int,
    cfg: RLConfig,
    alpha_iou: float = 0.6,
    alpha_conf: float = 0.4,
) -> float:
    """Hybrid reward: weighted combination of IoU-oracle and confidence."""
    r_iou = reward_oracle_iou(
        action, dets_n, dets_s, t_infer_n, t_infer_s, prev_action, cfg
    )
    dets_chosen = dets_n if action == ACTION_N else dets_s
    t_infer = t_infer_n if action == ACTION_N else t_infer_s
    r_conf = reward_confidence(
        action, dets_chosen, t_infer, t_infer_s, prev_action, cfg
    )
    return alpha_iou * r_iou + alpha_conf * r_conf


def compute_reward(
    action: int,
    dets_n: List[np.ndarray],
    dets_s: List[np.ndarray],
    t_infer_n: float,
    t_infer_s: float,
    prev_action: int,
    cfg: RLConfig,
) -> float:
    """Dispatch to the configured reward function."""
    if cfg.reward_type == "oracle_iou":
        return reward_oracle_iou(
            action, dets_n, dets_s, t_infer_n, t_infer_s, prev_action, cfg
        )
    elif cfg.reward_type == "confidence":
        dets_chosen = dets_n if action == ACTION_N else dets_s
        t_infer = t_infer_n if action == ACTION_N else t_infer_s
        return reward_confidence(
            action, dets_chosen, t_infer, t_infer_s, prev_action, cfg
        )
    elif cfg.reward_type == "hybrid":
        return reward_hybrid(
            action, dets_n, dets_s, t_infer_n, t_infer_s, prev_action, cfg
        )
    else:
        raise ValueError(f"Unknown reward type: {cfg.reward_type}")
