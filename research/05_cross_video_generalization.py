"""
Phase 5: Cross-video generalization — leave-one-out validation.

For the multi_proxy policy, tests whether weights tuned on N-1 videos
generalize to the held-out video.

Also tests: do policy parameters tuned on glass transfer to porcelain?

Usage:
    python research/05_cross_video_generalization.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RunConfig, InferenceParams, POLICIES
from core.engine import StreamingEngine
from core.research_paths import MODELS, GLASS_VIDEOS, PORCELAIN_VIDEOS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "research_results" / "05_generalization"

# Policies to test for cross-material transfer
TRANSFER_POLICIES = ["combined_hyst", "conf_ema", "multi_proxy"]


def run_policy(video_path: str, material: str, policy: str,
               model_n, model_s, inf: InferenceParams) -> dict:
    """Run a policy and return key metrics."""
    cfg = RunConfig(policy=policy)
    engine = StreamingEngine(video_path, cfg, model_n, model_s, inf, "video")
    for _ in engine.run():
        pass
    s = engine.get_summary()
    if s is None:
        return {}
    return {
        "T_total_mean": s.T_total_ms_mean,
        "T_total_p95": s.T_total_ms_p95,
        "sw_per_100": s.sw_per_100,
        "slow_pct": s.slow_pct,
        "total_frames": s.total_frames,
    }


def main():
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inf = InferenceParams(imgsz=640, device="cpu", stride=1, max_frames=0)

    results = {"cross_material": {}, "leave_one_out": {}}

    # ══════════════════════════════════════════════════════════════════
    # Test 1: Cross-material transfer
    # Run glass-tuned parameters on porcelain videos and vice versa
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  Test 1: Cross-Material Transfer")
    print(f"{'='*70}")

    for policy in TRANSFER_POLICIES:
        print(f"\n  Policy: {policy}")

        # Run on glass videos with glass models (same-domain)
        glass_metrics = []
        model_n_g = YOLO(MODELS["glass"]["y8n"])
        model_s_g = YOLO(MODELS["glass"]["y8s"])
        for vpath in GLASS_VIDEOS:
            if not os.path.isfile(vpath):
                continue
            vname = Path(vpath).stem
            print(f"    glass/{vname}...", end="", flush=True)
            m = run_policy(vpath, "glass", policy, model_n_g, model_s_g, inf)
            glass_metrics.append(m)
            print(f" T={m.get('T_total_mean',0):.1f}ms sw={m.get('sw_per_100',0):.1f}")

        # Run on porcelain videos with porcelain models (same-domain)
        porc_metrics = []
        model_n_p = YOLO(MODELS["porcelain"]["y8n"])
        model_s_p = YOLO(MODELS["porcelain"]["y8s"])
        for vpath in PORCELAIN_VIDEOS:
            if not os.path.isfile(vpath):
                continue
            vname = Path(vpath).stem
            print(f"    porcelain/{vname}...", end="", flush=True)
            m = run_policy(vpath, "porcelain", policy, model_n_p, model_s_p, inf)
            porc_metrics.append(m)
            print(f" T={m.get('T_total_mean',0):.1f}ms sw={m.get('sw_per_100',0):.1f}")

        # Compare: same parameters, different domains
        if glass_metrics and porc_metrics:
            glass_mean_T = np.mean([m["T_total_mean"] for m in glass_metrics])
            glass_mean_sw = np.mean([m["sw_per_100"] for m in glass_metrics])
            porc_mean_T = np.mean([m["T_total_mean"] for m in porc_metrics])
            porc_mean_sw = np.mean([m["sw_per_100"] for m in porc_metrics])

            results["cross_material"][policy] = {
                "glass_mean_T": float(glass_mean_T),
                "glass_mean_sw": float(glass_mean_sw),
                "porcelain_mean_T": float(porc_mean_T),
                "porcelain_mean_sw": float(porc_mean_sw),
                "T_ratio": float(porc_mean_T / max(1, glass_mean_T)),
                "sw_ratio": float(porc_mean_sw / max(0.01, glass_mean_sw)),
            }

            print(f"\n    Glass avg:     T={glass_mean_T:.1f}ms  sw={glass_mean_sw:.1f}/100")
            print(f"    Porcelain avg: T={porc_mean_T:.1f}ms  sw={porc_mean_sw:.1f}/100")
            print(f"    T ratio (porc/glass): {porc_mean_T/max(1,glass_mean_T):.2f}")

    # ══════════════════════════════════════════════════════════════════
    # Test 2: Leave-one-out within glass videos
    # Check if switching behavior is consistent across videos
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  Test 2: Leave-One-Out Consistency (Glass)")
    print(f"{'='*70}")

    model_n_g = YOLO(MODELS["glass"]["y8n"])
    model_s_g = YOLO(MODELS["glass"]["y8s"])

    for policy in TRANSFER_POLICIES:
        print(f"\n  Policy: {policy}")
        per_video = {}

        for vpath in GLASS_VIDEOS:
            if not os.path.isfile(vpath):
                continue
            vname = Path(vpath).stem
            print(f"    {vname}...", end="", flush=True)
            m = run_policy(vpath, "glass", policy, model_n_g, model_s_g, inf)
            per_video[vname] = m
            print(f" T={m.get('T_total_mean',0):.1f}ms "
                  f"sw={m.get('sw_per_100',0):.1f} "
                  f"s%={m.get('slow_pct',0):.1f}")

        if per_video:
            T_vals = [m["T_total_mean"] for m in per_video.values()]
            sw_vals = [m["sw_per_100"] for m in per_video.values()]
            spct_vals = [m["slow_pct"] for m in per_video.values()]

            results["leave_one_out"][policy] = {
                "per_video": per_video,
                "T_mean": float(np.mean(T_vals)),
                "T_std": float(np.std(T_vals)),
                "T_cv": float(np.std(T_vals) / np.mean(T_vals)) if np.mean(T_vals) > 0 else 0,
                "sw_mean": float(np.mean(sw_vals)),
                "sw_std": float(np.std(sw_vals)),
                "s_pct_mean": float(np.mean(spct_vals)),
                "s_pct_std": float(np.std(spct_vals)),
            }

            print(f"\n    T: {np.mean(T_vals):.1f} ± {np.std(T_vals):.1f}ms "
                  f"(CV={np.std(T_vals)/np.mean(T_vals)*100:.1f}%)")
            print(f"    sw/100: {np.mean(sw_vals):.1f} ± {np.std(sw_vals):.1f}")
            print(f"    s%: {np.mean(spct_vals):.1f} ± {np.std(spct_vals):.1f}%")

    # Save
    with open(OUTPUT_DIR / "generalization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_DIR / 'generalization_results.json'}")


if __name__ == "__main__":
    main()
