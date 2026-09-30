#!/usr/bin/env python3
"""
Oracle Ceiling Check — MUST RUN BEFORE RL WORK.

Answers: "How many frames genuinely NEED the s-model?"

Uses existing n_only and s_only trace CSVs (from the overnight pipeline)
to estimate the oracle ceiling.  No new model runs needed.

Logic:
  For each frame, compare n_only and s_only traces:
    - If n detects similar to s (within tolerance) → n was sufficient
    - If n misses significantly more than s → s was needed

  Oracle policy = use n when sufficient, s when needed.
  Oracle coverage = upper bound any switching policy can achieve.

  If oracle - conf_ema < 5%, RL won't help much.
  If oracle - conf_ema > 10%, significant room for improvement.

Usage:
    python research/oracle_ceiling_check.py
    python research/oracle_ceiling_check.py --results-dir G:/DMS_Experiment_Results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def load_trace(csv_path: str) -> List[Dict]:
    """Load a per-frame trace CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def analyze_video(
    n_trace_path: str,
    s_trace_path: str,
    video_id: str,
    det_count_tolerance: int = 2,
    conf_ratio_threshold: float = 0.7,
) -> Dict:
    """Analyze one video's n_only vs s_only traces.

    For each frame, determines if n was "sufficient" using multiple
    heuristics (since we don't have per-frame bounding boxes):

    Heuristic 1 (Detection Count): n is sufficient if:
      |n_dets - s_dets| <= tolerance  OR  (n_dets >= s_dets)

    Heuristic 2 (Confidence Ratio): n is sufficient if:
      mean_conf_n / mean_conf_s >= threshold  (when both have dets)

    Heuristic 3 (Combined): n is sufficient if BOTH heuristics agree,
      OR if neither model detects anything (empty frame → n is fine).

    Returns per-frame labels and aggregate statistics.
    """
    n_trace = load_trace(n_trace_path)
    s_trace = load_trace(s_trace_path)

    n_frames = min(len(n_trace), len(s_trace))
    if n_frames == 0:
        return {"video_id": video_id, "error": "empty traces"}

    # Per-frame analysis
    n_sufficient_count = np.zeros(n_frames, dtype=bool)
    n_sufficient_conf = np.zeros(n_frames, dtype=bool)
    n_sufficient_combined = np.zeros(n_frames, dtype=bool)

    n_dets_arr = np.zeros(n_frames)
    s_dets_arr = np.zeros(n_frames)
    n_conf_arr = np.zeros(n_frames)
    s_conf_arr = np.zeros(n_frames)
    latency_n = np.zeros(n_frames)
    latency_s = np.zeros(n_frames)

    for i in range(n_frames):
        n_row = n_trace[i]
        s_row = s_trace[i]

        nd = int(float(n_row.get("num_dets", 0)))
        sd = int(float(s_row.get("num_dets", 0)))
        nc = float(n_row.get("mean_conf", 0))
        sc = float(s_row.get("mean_conf", 0))
        tn = float(n_row.get("T_infer_n_ms", 0)) or float(n_row.get("T_total_ms", 0))
        ts = float(s_row.get("T_infer_s_ms", 0)) or float(s_row.get("T_total_ms", 0))

        n_dets_arr[i] = nd
        s_dets_arr[i] = sd
        n_conf_arr[i] = nc
        s_conf_arr[i] = sc
        latency_n[i] = tn
        latency_s[i] = ts

        # Heuristic 1: Detection count similarity
        if sd == 0 and nd == 0:
            n_sufficient_count[i] = True  # both empty
        elif sd == 0:
            n_sufficient_count[i] = True  # s sees nothing, n doesn't matter
        elif nd >= sd:
            n_sufficient_count[i] = True  # n detects as much or more
        elif abs(nd - sd) <= det_count_tolerance:
            n_sufficient_count[i] = True  # within tolerance
        else:
            n_sufficient_count[i] = False  # n misses significantly more

        # Heuristic 2: Confidence ratio
        if sd == 0 and nd == 0:
            n_sufficient_conf[i] = True
        elif sc <= 0:
            n_sufficient_conf[i] = True
        elif nc / (sc + 1e-9) >= conf_ratio_threshold:
            n_sufficient_conf[i] = True
        else:
            n_sufficient_conf[i] = False

        # Heuristic 3: Combined (conservative — both must agree)
        n_sufficient_combined[i] = n_sufficient_count[i] and n_sufficient_conf[i]

    # ── Aggregate results ────────────────────────────────────────────
    # Oracle coverage = fraction of frames where we'd use s
    # (= frames where n is NOT sufficient)
    oracle_s_needed_count = (~n_sufficient_count).sum()
    oracle_s_needed_conf = (~n_sufficient_conf).sum()
    oracle_s_needed_combined = (~n_sufficient_combined).sum()

    oracle_s_pct_count = oracle_s_needed_count / n_frames
    oracle_s_pct_conf = oracle_s_needed_conf / n_frames
    oracle_s_pct_combined = oracle_s_needed_combined / n_frames

    # Oracle latency estimate
    oracle_latency_count = np.where(
        n_sufficient_count, latency_n, latency_s
    ).mean()
    oracle_latency_combined = np.where(
        n_sufficient_combined, latency_n, latency_s
    ).mean()

    # Detection count disagreement statistics
    det_diff = s_dets_arr - n_dets_arr
    has_dets = s_dets_arr > 0

    return {
        "video_id": video_id,
        "n_frames": int(n_frames),

        # Real latencies from this machine
        "mean_latency_n_ms": float(latency_n[latency_n > 0].mean()) if (latency_n > 0).any() else 0,
        "mean_latency_s_ms": float(latency_s[latency_s > 0].mean()) if (latency_s > 0).any() else 0,
        "latency_gap_ms": float(latency_s[latency_s > 0].mean() - latency_n[latency_n > 0].mean()) if (latency_n > 0).any() and (latency_s > 0).any() else 0,

        # Oracle: what % of frames genuinely need s?
        "oracle_s_needed_pct_by_count": float(oracle_s_pct_count),
        "oracle_s_needed_pct_by_conf": float(oracle_s_pct_conf),
        "oracle_s_needed_pct_combined": float(oracle_s_pct_combined),

        # Oracle latency
        "oracle_latency_count_ms": float(oracle_latency_count),
        "oracle_latency_combined_ms": float(oracle_latency_combined),

        # Detection disagreement stats
        "mean_det_diff_s_minus_n": float(det_diff.mean()),
        "median_det_diff": float(np.median(det_diff)),
        "pct_frames_s_has_more_dets": float((det_diff > 0).mean()),
        "pct_frames_equal_dets": float((det_diff == 0).mean()),
        "pct_frames_n_has_more_dets": float((det_diff < 0).mean()),
        "pct_frames_both_empty": float((~has_dets & (n_dets_arr == 0)).mean()),

        # Confidence analysis
        "mean_conf_n": float(n_conf_arr[has_dets].mean()) if has_dets.any() else 0,
        "mean_conf_s": float(s_conf_arr[has_dets].mean()) if has_dets.any() else 0,
        "conf_ratio_mean": float((n_conf_arr[has_dets] / (s_conf_arr[has_dets] + 1e-9)).mean()) if has_dets.any() else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle Ceiling Check")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with overnight results")
    parser.add_argument("--det-tolerance", type=int, default=2,
                        help="Detection count tolerance for n-sufficiency")
    parser.add_argument("--conf-ratio", type=float, default=0.7,
                        help="Min conf_n/conf_s ratio for n-sufficiency")
    args = parser.parse_args()

    # Find results directory
    candidates = [
        args.results_dir,
        "G:/DMS_Experiment_Results/overnight/glass",
        os.path.join(PROJECT_ROOT, "research_results", "overnight_full"),
        "G:/DMS_Experiment_Results/overnight",
    ]
    results_dir = None
    for c in candidates:
        if c and os.path.isdir(c):
            results_dir = c
            break

    if not results_dir:
        print("ERROR: Cannot find overnight results directory.")
        print("Try: python research/oracle_ceiling_check.py --results-dir <path>")
        sys.exit(1)

    print("=" * 70)
    print("ORACLE CEILING CHECK")
    print("Determines if RL-based switching can improve over conf_ema")
    print("=" * 70)
    print(f"  Results dir: {results_dir}")
    print(f"  Det count tolerance: ±{args.det_tolerance}")
    print(f"  Confidence ratio threshold: {args.conf_ratio}")
    print()

    # ── Find all videos with both n_only and s_only traces ───────────
    video_results = []

    # Check direct subdirectories
    for entry in sorted(os.listdir(results_dir)):
        video_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(video_dir):
            continue

        n_path = os.path.join(video_dir, "n_only_trace.csv")
        s_path = os.path.join(video_dir, "s_only_trace.csv")

        if not (os.path.exists(n_path) and os.path.exists(s_path)):
            # Try subdirectory structure
            for sub in os.listdir(video_dir):
                sub_dir = os.path.join(video_dir, sub)
                if os.path.isdir(sub_dir):
                    n_path = os.path.join(sub_dir, "n_only_trace.csv")
                    s_path = os.path.join(sub_dir, "s_only_trace.csv")
                    if os.path.exists(n_path) and os.path.exists(s_path):
                        break
            else:
                continue

        print(f"Analyzing: {entry}")
        result = analyze_video(
            n_path, s_path, entry,
            det_count_tolerance=args.det_tolerance,
            conf_ratio_threshold=args.conf_ratio,
        )
        video_results.append(result)

    if not video_results:
        print("ERROR: No videos found with both n_only and s_only traces.")
        sys.exit(1)

    # ── Print per-video results ──────────────────────────────────────
    print()
    print("PER-VIDEO RESULTS")
    print("-" * 70)
    print(f"{'Video':<25s} {'Frames':>7s} {'Gap(ms)':>8s} "
          f"{'s_need%':>8s} {'s_need%(c)':>10s} {'det_diff':>9s}")
    print("-" * 70)

    for r in video_results:
        print(f"{r['video_id']:<25s} {r['n_frames']:>7d} "
              f"{r['latency_gap_ms']:>7.1f}ms "
              f"{r['oracle_s_needed_pct_combined']:>7.1%} "
              f"{r['oracle_s_needed_pct_by_count']:>9.1%} "
              f"{r['mean_det_diff_s_minus_n']:>+8.1f}")

    # ── Aggregate across all videos ──────────────────────────────────
    total_frames = sum(r["n_frames"] for r in video_results)
    weighted_s_needed = sum(
        r["oracle_s_needed_pct_combined"] * r["n_frames"]
        for r in video_results
    ) / total_frames
    weighted_s_by_count = sum(
        r["oracle_s_needed_pct_by_count"] * r["n_frames"]
        for r in video_results
    ) / total_frames
    avg_gap = np.mean([r["latency_gap_ms"] for r in video_results])
    avg_conf_ratio = np.mean([r["conf_ratio_mean"] for r in video_results])
    avg_n_lat = np.mean([r["mean_latency_n_ms"] for r in video_results])
    avg_s_lat = np.mean([r["mean_latency_s_ms"] for r in video_results])

    print()
    print("=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  Total frames analyzed:        {total_frames:,}")
    print(f"  Videos analyzed:              {len(video_results)}")
    print()
    print(f"  Mean latency n-model:         {avg_n_lat:.1f}ms")
    print(f"  Mean latency s-model:         {avg_s_lat:.1f}ms")
    print(f"  Mean latency GAP:             {avg_gap:.1f}ms")
    print()
    print(f"  Frames where s is needed:")
    print(f"    By detection count:         {weighted_s_by_count:.1%}")
    print(f"    By combined heuristic:      {weighted_s_needed:.1%}")
    print()

    # ── THE VERDICT ──────────────────────────────────────────────────
    oracle_coverage = 1.0  # oracle always gets 100% by definition
    conf_ema_coverage = 0.86  # from thesis results

    # Oracle s-usage tells us the theoretical minimum s-usage
    oracle_s_usage = weighted_s_needed
    conf_ema_s_usage = 0.27  # from thesis results

    print("ORACLE vs CONF_EMA COMPARISON")
    print("-" * 70)
    print(f"  {'Metric':<35s} {'conf_ema':>12s} {'Oracle':>12s} {'Gap':>10s}")
    print(f"  {'Coverage':.<35s} {'86%':>12s} {'100%':>12s} {'14%':>10s}")
    print(f"  {'s-model usage':.<35s} {conf_ema_s_usage:>11.1%} {oracle_s_usage:>11.1%} "
          f"{'':>10s}")
    print()

    # Can RL help?
    headroom = oracle_coverage - conf_ema_coverage
    print("VERDICT")
    print("-" * 70)

    if weighted_s_needed < 0.10:
        print(f"  Oracle needs s on only {weighted_s_needed:.1%} of frames.")
        print(f"  conf_ema uses s on 27% of frames — ALREADY OVERSHOOTING.")
        print(f"  RL CAN help by REDUCING s-usage while maintaining coverage.")
        print(f"  >> Focus: efficiency (use less s, same quality)")
        verdict = "GO -- focus on efficiency"
    elif weighted_s_needed < 0.20:
        print(f"  Oracle needs s on {weighted_s_needed:.1%} of frames.")
        print(f"  conf_ema coverage gap is {headroom:.0%} (86% vs 100%).")
        print(f"  RL can potentially improve both coverage AND efficiency.")
        print(f"  >> Good potential for improvement")
        verdict = "GO -- moderate headroom"
    elif weighted_s_needed < 0.40:
        print(f"  Oracle needs s on {weighted_s_needed:.1%} of frames.")
        print(f"  Significant number of frames genuinely need s.")
        print(f"  RL has STRONG potential to improve routing accuracy.")
        print(f"  >> Best case for RL")
        verdict = "STRONG GO -- large headroom"
    else:
        print(f"  Oracle needs s on {weighted_s_needed:.1%} of frames.")
        print(f"  Many frames need s based on count+confidence heuristic.")
        print(f"  BUT: this heuristic is CONSERVATIVE (see note below).")
        print(f"  >> Check: is this because conf_min=0.001 inflates s-model")
        print(f"     detection counts with low-confidence noise?")
        verdict = "NEEDS DEEPER ANALYSIS"

    print()
    print(f"  Latency gap: {avg_gap:.1f}ms")
    if avg_gap < 20:
        print(f"  WARNING: Gap < 20ms — switching overhead may exceed savings!")
        print(f"  DMS only makes sense if gap > feature extraction cost (~5ms)")
    elif avg_gap < 50:
        print(f"  MODERATE gap — RL reward function needs careful calibration.")
        print(f"  Each correct n→s routing saves only {avg_gap:.0f}ms.")
    else:
        print(f"  GOOD gap — clear latency benefit from correct routing.")

    print()
    print(f"  FINAL VERDICT: {verdict}")
    print()
    print(f"  Mean conf_n / conf_s ratio: {avg_conf_ratio:.3f}")
    print(f"  (Closer to 1.0 = n-model is almost as good as s-model)")
    print("=" * 70)

    # ── Save results ─────────────────────────────────────────────────
    output_dir = os.path.join(PROJECT_ROOT, "research_results", "rl_switching")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "oracle_ceiling_check.json")

    output = {
        "per_video": video_results,
        "aggregate": {
            "total_frames": total_frames,
            "n_videos": len(video_results),
            "mean_latency_n_ms": float(avg_n_lat),
            "mean_latency_s_ms": float(avg_s_lat),
            "mean_latency_gap_ms": float(avg_gap),
            "oracle_s_needed_pct_by_count": float(weighted_s_by_count),
            "oracle_s_needed_pct_combined": float(weighted_s_needed),
            "mean_conf_ratio": float(avg_conf_ratio),
            "verdict": verdict,
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
