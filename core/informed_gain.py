"""Informed-gain computation — same-usage random baseline.

The informed gain measures how much a switching policy improves over a
hypothetical random switcher that uses the same fraction of accurate-model
frames (``u = s_choice_rate``).

    expected_random_{metric} = mean((1 - u) * fast_metric + u * accurate_metric)
    informed_gain = A_policy - expected_random_{metric}

``fast_metric`` and ``accurate_metric`` are the per-frame metric arrays
from the n_only and s_only baselines respectively (evaluated against the
same accurate-reference agreement definition).

A positive gain means the policy's trigger is better than random at the
same usage budget. A gain near zero means the trigger adds no value
beyond what random switching achieves at that rate.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


GAIN_METRICS = ("iou_match", "det_coverage", "det_recall", "count_agree")


def expected_random_metric(
    fast_values: np.ndarray,
    accurate_values: np.ndarray,
    s_choice_rate: float,
) -> float:
    """Expected metric value of a random switcher at usage rate ``u``.

    Args:
        fast_values:     Per-frame metric array from n_only baseline.
        accurate_values: Per-frame metric array from s_only baseline.
        s_choice_rate:   Fraction of frames the policy chose the accurate
                         model (``u``). Must be in ``[0, 1]``.

    Returns:
        ``mean((1 - u) * fast + u * accurate)`` over aligned frames.
    """
    f = np.asarray(fast_values, dtype=float).ravel()
    a = np.asarray(accurate_values, dtype=float).ravel()
    u = float(s_choice_rate)
    n = min(len(f), len(a))
    if n == 0 or not np.isfinite(u):
        return float("nan")
    return float(np.nanmean((1.0 - u) * f[:n] + u * a[:n]))


def informed_gain(
    policy_metric: float,
    expected_random: float,
) -> float:
    """``A_policy - A_random(u)``. Positive = better than random switching."""
    pm = float(policy_metric)
    er = float(expected_random)
    if not (np.isfinite(pm) and np.isfinite(er)):
        return float("nan")
    return pm - er


def compute_informed_gains(
    fast_per_frame: Dict[str, np.ndarray],
    accurate_per_frame: Dict[str, np.ndarray],
    policy_aggregated: Dict[str, float],
    s_choice_rate: float,
    metrics: Sequence[str] = GAIN_METRICS,
) -> Dict[str, float]:
    """Compute expected-random baselines and informed gains for all metrics.

    Args:
        fast_per_frame:      ``{metric_name: array}`` from n_only baseline.
        accurate_per_frame:  ``{metric_name: array}`` from s_only baseline.
        policy_aggregated:   ``{metric_name: float}`` for the policy under test.
        s_choice_rate:       The policy's accurate-model usage fraction.
        metrics:             Which metrics to compute (default: the 4 thesis metrics).

    Returns:
        Dict with keys ``expected_random_{m}`` and ``informed_gain_{m}``
        for each metric ``m``.
    """
    out: Dict[str, float] = {}
    u = float(s_choice_rate)
    for m in metrics:
        f_arr = fast_per_frame.get(m, np.array([]))
        a_arr = accurate_per_frame.get(m, np.array([]))
        er = expected_random_metric(f_arr, a_arr, u)
        out[f"expected_random_{m}"] = er
        pm = float(policy_aggregated.get(m, float("nan")))
        out[f"informed_gain_{m}"] = informed_gain(pm, er)
    return out


__all__ = [
    "GAIN_METRICS",
    "expected_random_metric",
    "informed_gain",
    "compute_informed_gains",
]
