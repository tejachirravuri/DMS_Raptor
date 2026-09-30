"""Per-frame and per-run thesis metrics — pure numpy, no I/O.

Primary metric: iou_match_frame (strict 1:1 match)
Relaxed metrics: det_coverage_frame, det_recall_frame
Count metric: count_agree_frame
Trigger target: benefit_positive_frame (LOCKED definition)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from .matching import greedy_match


# ===========================================================================
# Per-frame binary metrics
# ===========================================================================
def iou_match_frame(
    policy_boxes: np.ndarray,
    ref_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    policy_classes: Optional[np.ndarray] = None,
    ref_classes: Optional[np.ndarray] = None,
) -> bool:
    """1 iff perfect 1:1 match of policy and reference at the threshold."""
    p = np.asarray(policy_boxes, dtype=float).reshape(-1, 4)
    r = np.asarray(ref_boxes, dtype=float).reshape(-1, 4)
    if p.shape[0] == 0 and r.shape[0] == 0:
        return True
    if p.shape[0] != r.shape[0]:
        return False
    res = greedy_match(
        p, r, iou_threshold,
        classes_a=policy_classes, classes_b=ref_classes,
    )
    return len(res.unmatched_a) == 0 and len(res.unmatched_b) == 0


def det_recall_frame(
    policy_boxes: np.ndarray,
    ref_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    policy_classes: Optional[np.ndarray] = None,
    ref_classes: Optional[np.ndarray] = None,
) -> float:
    """Fraction of reference detections matched by policy at the threshold."""
    p = np.asarray(policy_boxes, dtype=float).reshape(-1, 4)
    r = np.asarray(ref_boxes, dtype=float).reshape(-1, 4)
    if r.shape[0] == 0:
        return 1.0
    if p.shape[0] == 0:
        return 0.0
    res = greedy_match(
        p, r, iou_threshold,
        classes_a=policy_classes, classes_b=ref_classes,
    )
    matched = int(r.shape[0]) - len(res.unmatched_b)
    return float(matched) / float(r.shape[0])


def det_coverage_frame(
    policy_boxes: np.ndarray,
    ref_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    policy_classes: Optional[np.ndarray] = None,
    ref_classes: Optional[np.ndarray] = None,
) -> bool:
    """1 iff every reference detection is matched by some policy detection."""
    return det_recall_frame(
        policy_boxes, ref_boxes, iou_threshold,
        policy_classes=policy_classes, ref_classes=ref_classes,
    ) >= 1.0


def count_agree_frame(
    policy_boxes: np.ndarray,
    ref_boxes: np.ndarray,
) -> bool:
    """1 iff |policy| == |reference|. Spatial agreement ignored."""
    p = np.asarray(policy_boxes, dtype=float).reshape(-1, 4)
    r = np.asarray(ref_boxes, dtype=float).reshape(-1, 4)
    return int(p.shape[0]) == int(r.shape[0])


def benefit_positive_frame(
    fast_boxes: np.ndarray,
    accurate_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    fast_classes: Optional[np.ndarray] = None,
    accurate_classes: Optional[np.ndarray] = None,
) -> bool:
    """LOCKED benefit-positive definition.

    1 iff the accurate model produces at least one detection unmatched to
    any fast-model detection at IoU >= threshold. Empty accurate -> False.
    """
    acc = np.asarray(accurate_boxes, dtype=float).reshape(-1, 4)
    if acc.shape[0] == 0:
        return False
    fast = np.asarray(fast_boxes, dtype=float).reshape(-1, 4)
    res = greedy_match(
        acc, fast, iou_threshold,
        classes_a=accurate_classes, classes_b=fast_classes,
    )
    return len(res.unmatched_a) > 0


# ===========================================================================
# Latency summary
# ===========================================================================
def latency_summary(times_ms: Iterable[float]) -> Dict[str, float]:
    """Mean / p50 / p95 / p99 / fps over per-frame times (ms)."""
    arr = np.asarray(list(times_ms), dtype=float)
    if arr.size == 0:
        return dict(n=0, mean=0.0, p50=0.0, p95=0.0, p99=0.0, fps=0.0)
    mean = float(arr.mean())
    return dict(
        n=int(arr.size),
        mean=mean,
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        fps=float(1000.0 / max(mean, 1e-9)),
    )


# ===========================================================================
# Run aggregation
# ===========================================================================
def summarise_run(per_frame: List[Dict]) -> Dict:
    """Aggregate a list of per-frame records to a single run summary."""
    n = len(per_frame)
    if n == 0:
        return dict(n_frames=0)
    iou_rate = sum(int(bool(r.get("iou_match", 0))) for r in per_frame) / n
    cov_rate = sum(int(bool(r.get("det_coverage", 0))) for r in per_frame) / n
    rec_mean = sum(float(r.get("det_recall", 0.0)) for r in per_frame) / n
    cnt_rate = sum(int(bool(r.get("count_agree", 0))) for r in per_frame) / n
    ben_rate = sum(int(bool(r.get("benefit", 0))) for r in per_frame) / n
    s_rate = sum(1 for r in per_frame if r.get("choice") == "s") / n
    times = [r["time_ms"] for r in per_frame if "time_ms" in r]
    lat = latency_summary(times)
    return dict(
        n_frames=int(n),
        iou_match_rate=float(iou_rate),
        det_coverage_rate=float(cov_rate),
        det_recall_mean=float(rec_mean),
        count_agree_rate=float(cnt_rate),
        benefit_rate=float(ben_rate),
        s_choice_rate=float(s_rate),
        latency=lat,
    )


__all__ = [
    "iou_match_frame",
    "det_coverage_frame",
    "det_recall_frame",
    "count_agree_frame",
    "benefit_positive_frame",
    "latency_summary",
    "summarise_run",
]
