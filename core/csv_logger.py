"""Thesis-compatible CSV logger for per-frame analysis results.

Writes a thesis-style per-frame CSV containing the required analysis
fields. Some downstream thesis scripts may require column mapping.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .analysis_engine import FrameResult


_DIAG_KEYS = (
    "policy_score",
    "policy_thresh_low",
    "policy_thresh_high",
    "policy_thresh_mid",
    "fast_ema",
    "slow_ema",
    "conf_drop",
)

_PROXY_KEYS = (
    "L", "H", "color_entropy", "tenengrad",
    "edge_density", "local_contrast",
    "bright_fraction", "hue_std", "brightness_mean",
)

_FIXED_COLUMNS = [
    "frame_idx",
    "policy",
    "choice",
    "selected_model",
    "accurate_selected",
    "accurate_reference_ran",
    "fast_reference_ran",
    "fast_mean_conf",
]

_DIAG_COLUMNS = list(_DIAG_KEYS)

_PROXY_COLUMNS = list(_PROXY_KEYS)

_TIMING_COLUMNS = [
    "t_fast_ms",
    "t_accurate_ms",
    "t_ctrl_ms",
    "t_proxy_ms",
    "t_total_deployed_ms",
]

_METRIC_COLUMNS = [
    "iou_match",
    "det_coverage",
    "det_recall",
    "count_agree",
    "benefit_positive",
]

COLUMNS = (
    _FIXED_COLUMNS
    + _DIAG_COLUMNS
    + _PROXY_COLUMNS
    + _TIMING_COLUMNS
    + _METRIC_COLUMNS
)


def _frame_to_row(r: FrameResult) -> Dict[str, object]:
    row: Dict[str, object] = {
        "frame_idx": r.frame_idx,
        "policy": r.policy,
        "choice": r.choice,
        "selected_model": r.selected_model,
        "accurate_selected": int(r.accurate_selected),
        "accurate_reference_ran": int(r.accurate_reference_ran),
        "fast_reference_ran": int(r.fast_reference_ran),
        "fast_mean_conf": round(r.fast_mean_conf, 6),
    }
    for k in _DIAG_KEYS:
        row[k] = round(r.diag.get(k, float("nan")), 6)
    for k in _PROXY_KEYS:
        row[k] = round(r.proxy_features.get(k, float("nan")), 6)
    row["t_fast_ms"] = round(r.t_fast_ms, 3)
    row["t_accurate_ms"] = round(r.t_accurate_ms, 3)
    row["t_ctrl_ms"] = round(r.t_ctrl_ms, 3)
    row["t_proxy_ms"] = round(r.t_proxy_ms, 3)
    row["t_total_deployed_ms"] = round(r.t_total_deployed_ms, 3)
    row["iou_match"] = int(r.iou_match)
    row["det_coverage"] = int(r.det_coverage)
    row["det_recall"] = round(r.det_recall, 6)
    row["count_agree"] = int(r.count_agree)
    row["benefit_positive"] = int(r.benefit_positive)
    return row


class AnalysisCSVLogger:
    """Streaming CSV writer for thesis-analysis per-frame results."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._count = 0

    def log(self, result: FrameResult) -> None:
        self._writer.writerow(_frame_to_row(result))
        self._count += 1
        if self._count % 50 == 0:
            self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def row_count(self) -> int:
        return self._count


def write_results_csv(
    results: Sequence[FrameResult],
    path: str | Path,
) -> int:
    """Write a complete list of FrameResults to a CSV file.

    Returns the number of rows written.
    """
    with AnalysisCSVLogger(path) as logger:
        for r in results:
            logger.log(r)
    return logger.row_count


__all__ = [
    "COLUMNS",
    "AnalysisCSVLogger",
    "write_results_csv",
]
