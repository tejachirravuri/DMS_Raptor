"""Multi-policy thesis-analysis runner.

Runs multiple policies on the same video, writes per-policy CSVs and
summary JSONs, then produces a compact ``analysis_policy_summary.csv``
with informed-gain columns when both baselines (n_only, s_only) are
included.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .analysis_engine import FrameResult, run_analysis, summarise_analysis
from .csv_logger import write_results_csv
from .inference_backend import InferenceBackend
from .informed_gain import GAIN_METRICS, compute_informed_gains
from .policies import POLICY_REGISTRY, PolicyConfig
from .proxies import ProxyConfig


_STATIC_POLICIES = frozenset({"n_only", "s_only"})

_SUMMARY_COLUMNS = [
    "policy",
    "n_frames",
    "accurate_usage_rate",
    "mean_iou_match",
    "mean_det_coverage",
    "mean_det_recall",
    "mean_count_agree",
    "benefit_rate",
    "mean_t_total_deployed_ms",
    "mean_t_fast_ms",
    "mean_t_accurate_ms",
    "mean_t_ctrl_ms",
    "mean_t_proxy_ms",
]

_GAIN_COLUMNS = [
    "expected_random_iou_match",
    "informed_gain_iou_match",
    "expected_random_det_coverage",
    "informed_gain_det_coverage",
    "expected_random_det_recall",
    "informed_gain_det_recall",
    "expected_random_count_agree",
    "informed_gain_count_agree",
]

SUMMARY_ALL_COLUMNS = _SUMMARY_COLUMNS + _GAIN_COLUMNS


def _collect_per_frame_metrics(
    results: List[FrameResult],
) -> Dict[str, np.ndarray]:
    return {
        "iou_match": np.array([int(r.iou_match) for r in results], dtype=float),
        "det_coverage": np.array([int(r.det_coverage) for r in results], dtype=float),
        "det_recall": np.array([r.det_recall for r in results], dtype=float),
        "count_agree": np.array([int(r.count_agree) for r in results], dtype=float),
    }


def _summary_to_gain_input(summary: Dict) -> Dict[str, float]:
    return {
        "iou_match": summary.get("mean_iou_match", float("nan")),
        "det_coverage": summary.get("mean_det_coverage", float("nan")),
        "det_recall": summary.get("mean_det_recall", float("nan")),
        "count_agree": summary.get("mean_count_agree", float("nan")),
    }


def _write_summary_csv(
    summaries: Dict[str, Dict],
    path: Path,
    policy_order: Sequence[str],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_ALL_COLUMNS)
        writer.writeheader()
        for pname in policy_order:
            s = summaries[pname]
            row: Dict[str, object] = {}
            for col in _SUMMARY_COLUMNS:
                if col == "policy":
                    row[col] = pname
                elif col == "n_frames":
                    row[col] = s.get(col, 0)
                else:
                    val = s.get(col, float("nan"))
                    row[col] = round(val, 6) if isinstance(val, float) else val
            for col in _GAIN_COLUMNS:
                val = s.get(col, float("nan"))
                row[col] = round(val, 6) if isinstance(val, float) and math.isfinite(val) else ""
            writer.writerow(row)


def run_analysis_policies(
    video_path: str,
    fast_backend: InferenceBackend,
    accurate_backend: InferenceBackend,
    policies: Sequence[str],
    out_dir: str | Path,
    policy_config: Optional[PolicyConfig] = None,
    proxy_config: Optional[ProxyConfig] = None,
    iou_threshold: float = 0.5,
    max_frames: int = 0,
    start_frame: int = 0,
    progress_callback=None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Dict]:
    """Run thesis-analysis for multiple policies on the same video.

    Args:
        video_path:        Path to the input video.
        fast_backend:      Fast (n) model backend.
        accurate_backend:  Accurate (s) model backend.
        policies:          List of policy names to evaluate.
        out_dir:           Output directory for CSVs and JSONs.
        policy_config:     Base config (name field is overridden per policy).
        proxy_config:      Scene-proxy settings.
        iou_threshold:     IoU threshold for matching metrics.
        max_frames:        0 = all frames.
        start_frame:       First video frame to process (0 = beginning).
        progress_callback: Optional ``fn(policy_name, status)`` for progress.
        should_stop:       Optional callable returning True to cancel between
                           policies.  Completed policies are still written.

    Returns:
        Dict mapping policy name to its summary dict (including informed
        gain columns when baselines are present).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict[str, Dict] = {}
    per_frame_metrics: Dict[str, Dict[str, np.ndarray]] = {}
    policy_order: List[str] = []

    for pname in policies:
        if pname not in POLICY_REGISTRY:
            raise ValueError(f"unknown policy: {pname!r}")

    for pname in policies:
        if should_stop is not None and should_stop():
            break

        if progress_callback:
            progress_callback(pname, "running")

        cfg = PolicyConfig() if policy_config is None else PolicyConfig(
            **{k: v for k, v in policy_config.__dict__.items()}
        )
        cfg.name = pname

        results = list(run_analysis(
            video_path=str(video_path),
            fast_backend=fast_backend,
            accurate_backend=accurate_backend,
            policy_name=pname,
            policy_config=cfg,
            proxy_config=proxy_config,
            iou_threshold=iou_threshold,
            max_frames=max_frames,
            start_frame=start_frame,
        ))

        csv_path = out / f"{pname}_per_frame.csv"
        write_results_csv(results, csv_path)

        summary = summarise_analysis(results)
        json_path = out / f"{pname}_summary.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        all_summaries[pname] = summary
        per_frame_metrics[pname] = _collect_per_frame_metrics(results)
        policy_order.append(pname)

        if progress_callback:
            progress_callback(pname, "done")

        if should_stop is not None and should_stop():
            break

    # ----------------------------------------------------------------
    # Informed gain (requires both baselines)
    # ----------------------------------------------------------------
    has_baselines = "n_only" in all_summaries and "s_only" in all_summaries

    for pname in policy_order:
        summary = all_summaries[pname]
        if has_baselines and pname not in _STATIC_POLICIES:
            u = summary.get("accurate_usage_rate", float("nan"))
            gains = compute_informed_gains(
                fast_per_frame=per_frame_metrics["n_only"],
                accurate_per_frame=per_frame_metrics["s_only"],
                policy_aggregated=_summary_to_gain_input(summary),
                s_choice_rate=u,
            )
            summary.update(gains)
        else:
            for m in GAIN_METRICS:
                summary[f"expected_random_{m}"] = float("nan")
                summary[f"informed_gain_{m}"] = float("nan")

    # ----------------------------------------------------------------
    # Write compact summary CSV
    # ----------------------------------------------------------------
    _write_summary_csv(all_summaries, out / "analysis_policy_summary.csv", policy_order)

    return all_summaries


__all__ = [
    "SUMMARY_ALL_COLUMNS",
    "run_analysis_policies",
]
