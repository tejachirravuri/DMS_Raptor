"""
Phase 2: Statistical rigor — repeated timing trials.

Runs each policy N times on each video and reports mean ± std for timing.
Also performs Wilcoxon signed-rank test between policy pairs.

Usage:
    python research/03_repeated_trials.py --video <path> --material glass --repeats 3
    python research/03_repeated_trials.py --all --repeats 3 --stride 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RunConfig, InferenceParams, RunSummary, POLICIES
from core.engine import StreamingEngine
from core.research_paths import MODELS, VIDEOS

# ── Configuration ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "research_results" / "03_repeated_trials"


def run_single_trial(video_path: str, material: str, policy: str,
                     model_n, model_s, inf: InferenceParams) -> RunSummary:
    """Run one policy on one video and return summary."""
    cfg = RunConfig(policy=policy, name=f"{policy}_trial")
    engine = StreamingEngine(video_path, cfg, model_n, model_s, inf, "video")
    for _ in engine.run():
        pass
    return engine.get_summary()


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--video", type=str)
    parser.add_argument("--material", type=str, default="glass",
                        choices=["glass", "porcelain"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--policies", type=str, default=None,
                        help="Comma-separated policy list (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    policies = args.policies.split(",") if args.policies else POLICIES
    inf = InferenceParams(imgsz=640, device="cpu", conf_min=0.001,
                          iou_nms=0.45, stride=args.stride,
                          max_frames=args.max_frames)

    if args.all:
        video_list = []
        for mat, vids in VIDEOS.items():
            for v in vids:
                video_list.append((v, mat))
    elif args.video:
        video_list = [(args.video, args.material)]
    else:
        parser.print_help()
        return

    all_stats = {}

    for vpath, material in video_list:
        if not os.path.isfile(vpath):
            print(f"WARNING: {vpath} not found, skipping")
            continue

        vname = Path(vpath).stem
        print(f"\n{'='*60}")
        print(f"  Video: {vname} ({material}) | Repeats: {args.repeats}")
        print(f"{'='*60}")

        # Load models once per material
        model_n = YOLO(MODELS[material]["y8n"])
        model_s = YOLO(MODELS[material]["y8s"])

        video_stats = {}

        for policy in policies:
            trial_means = []
            trial_p95s = []
            trial_p99s = []
            trial_sw = []

            for trial in range(args.repeats):
                print(f"  {policy} trial {trial+1}/{args.repeats}...", end="", flush=True)
                t0 = time.time()
                summary = run_single_trial(vpath, material, policy, model_n, model_s, inf)
                elapsed = time.time() - t0
                trial_means.append(summary.T_total_ms_mean)
                trial_p95s.append(summary.T_total_ms_p95)
                trial_p99s.append(summary.T_total_ms_p99)
                trial_sw.append(summary.sw_per_100)
                print(f" mean={summary.T_total_ms_mean:.1f}ms "
                      f"p95={summary.T_total_ms_p95:.1f}ms "
                      f"({elapsed:.0f}s)")

            stats = {
                "mean_T_total": {
                    "mean": float(np.mean(trial_means)),
                    "std": float(np.std(trial_means)),
                    "values": trial_means,
                },
                "p95_T_total": {
                    "mean": float(np.mean(trial_p95s)),
                    "std": float(np.std(trial_p95s)),
                    "values": trial_p95s,
                },
                "p99_T_total": {
                    "mean": float(np.mean(trial_p99s)),
                    "std": float(np.std(trial_p99s)),
                    "values": trial_p99s,
                },
                "sw_per_100": {
                    "mean": float(np.mean(trial_sw)),
                    "std": float(np.std(trial_sw)),
                    "values": trial_sw,
                },
            }
            video_stats[policy] = stats

        all_stats[f"{material}/{vname}"] = video_stats

        # Print table for this video
        print(f"\n  {'Policy':<16} {'Mean T ± std':>16} {'P95 T ± std':>16} {'Sw/100 ± std':>16}")
        print(f"  {'-'*68}")
        for p in policies:
            s = video_stats[p]
            print(f"  {p:<16} "
                  f"{s['mean_T_total']['mean']:>7.1f} ± {s['mean_T_total']['std']:>5.1f} "
                  f"{s['p95_T_total']['mean']:>7.1f} ± {s['p95_T_total']['std']:>5.1f} "
                  f"{s['sw_per_100']['mean']:>7.2f} ± {s['sw_per_100']['std']:>5.2f}")

    # Save all results
    with open(OUTPUT_DIR / "repeated_trials.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_DIR / 'repeated_trials.json'}")

    # Wilcoxon test between key policy pairs
    print(f"\n{'='*60}")
    print("  Wilcoxon Signed-Rank Tests (paired by video)")
    print(f"{'='*60}")
    try:
        from scipy.stats import wilcoxon
        pairs = [
            ("s_only", "combined_hyst"),
            ("s_only", "conf_ema"),
            ("s_only", "multi_proxy"),
            ("combined_hyst", "conf_ema"),
            ("combined_hyst", "multi_proxy"),
        ]
        for p1, p2 in pairs:
            vals_1, vals_2 = [], []
            for vkey, vdata in all_stats.items():
                if p1 in vdata and p2 in vdata:
                    vals_1.append(vdata[p1]["mean_T_total"]["mean"])
                    vals_2.append(vdata[p2]["mean_T_total"]["mean"])
            if len(vals_1) >= 5:
                stat, pval = wilcoxon(vals_1, vals_2)
                print(f"  {p1} vs {p2}: W={stat:.1f}, p={pval:.4f} "
                      f"{'*' if pval < 0.05 else 'ns'}")
            else:
                print(f"  {p1} vs {p2}: insufficient paired samples ({len(vals_1)})")
    except ImportError:
        print("  scipy not available — skipping Wilcoxon tests")


if __name__ == "__main__":
    main()
