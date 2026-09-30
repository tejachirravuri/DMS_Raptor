"""
Phase 4: Ablation study — decompose contribution of each DMS component.

Tests three orthogonal dimensions:
  A) Proxy signal: L-only, H-only, C(L+H), conf_ema, NR-IQA, multi_proxy
  B) Controller: single-threshold vs hysteresis
  C) Stability: with vs without dwell/rate limits

Outputs a comprehensive ablation table.

Usage:
    python research/04_ablation_study.py --video <path> --material glass
    python research/04_ablation_study.py --all --stride 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RunConfig, InferenceParams, POLICIES
from core.engine import StreamingEngine
from core.research_paths import MODELS, VIDEOS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "research_results" / "04_ablation"


# ── Ablation configurations ───────────────────────────────────────────
def get_ablation_configs() -> List[dict]:
    """Return list of {name, config} for each ablation variant."""
    configs = []

    # ── Baselines ──
    configs.append({
        "name": "n_only (baseline)",
        "category": "baseline",
        "config": RunConfig(policy="n_only"),
    })
    configs.append({
        "name": "s_only (baseline)",
        "category": "baseline",
        "config": RunConfig(policy="s_only"),
    })

    # ── A) Proxy signal ablation (all use hysteresis + stability) ──
    # L-only via entropy_only won't work directly, so we use combined with alpha=1.0
    cfg_L_only = RunConfig(policy="combined_hyst", alpha=1.0)  # pure L
    configs.append({"name": "L-only (hyst)", "category": "proxy", "config": cfg_L_only})

    cfg_H_only = RunConfig(policy="combined_hyst", alpha=0.0)  # pure H
    configs.append({"name": "H-only (hyst)", "category": "proxy", "config": cfg_H_only})

    cfg_C_default = RunConfig(policy="combined_hyst", alpha=0.6)  # default C
    configs.append({"name": "C(L+H) α=0.6 (hyst)", "category": "proxy", "config": cfg_C_default})

    cfg_conf = RunConfig(policy="conf_ema")
    configs.append({"name": "conf_ema (hyst)", "category": "proxy", "config": cfg_conf})

    cfg_niqe = RunConfig(policy="niqe_switch")
    configs.append({"name": "niqe_switch (hyst)", "category": "proxy", "config": cfg_niqe})

    cfg_multi = RunConfig(policy="multi_proxy")
    configs.append({"name": "multi_proxy (hyst)", "category": "proxy", "config": cfg_multi})

    # ── B) Controller ablation (using C(L+H) proxy) ──
    cfg_single_thr = RunConfig(policy="combined", alpha=0.6)  # single threshold, no hyst
    configs.append({"name": "C(L+H) single-thr", "category": "controller", "config": cfg_single_thr})

    cfg_hyst = RunConfig(policy="combined_hyst", alpha=0.6)  # with hysteresis
    configs.append({"name": "C(L+H) hysteresis", "category": "controller", "config": cfg_hyst})

    cfg_entropy = RunConfig(policy="entropy_only")  # median-threshold
    configs.append({"name": "H median-thr", "category": "controller", "config": cfg_entropy})

    # ── C) Stability ablation (using combined_hyst) ──
    cfg_no_stab = RunConfig(policy="combined_hyst", alpha=0.6,
                            min_dwell_frames=0, max_switches_per_100=10000)
    configs.append({"name": "C(L+H) hyst NO-stab", "category": "stability", "config": cfg_no_stab})

    cfg_dwell_only = RunConfig(policy="combined_hyst", alpha=0.6,
                               min_dwell_frames=10, max_switches_per_100=10000)
    configs.append({"name": "C(L+H) hyst dwell-only", "category": "stability", "config": cfg_dwell_only})

    cfg_rate_only = RunConfig(policy="combined_hyst", alpha=0.6,
                              min_dwell_frames=0, max_switches_per_100=12)
    configs.append({"name": "C(L+H) hyst rate-only", "category": "stability", "config": cfg_rate_only})

    cfg_full_stab = RunConfig(policy="combined_hyst", alpha=0.6,
                              min_dwell_frames=10, max_switches_per_100=12)
    configs.append({"name": "C(L+H) hyst FULL-stab", "category": "stability", "config": cfg_full_stab})

    return configs


def run_ablation_variant(video_path: str, material: str, cfg: RunConfig,
                         model_n, model_s, inf: InferenceParams) -> dict:
    """Run one variant and return metrics dict."""
    engine = StreamingEngine(video_path, cfg, model_n, model_s, inf, "video")
    for _ in engine.run():
        pass
    s = engine.get_summary()
    if s is None:
        return {}
    return {
        "T_total_mean": s.T_total_ms_mean,
        "T_total_p95": s.T_total_ms_p95,
        "T_total_p99": s.T_total_ms_p99,
        "T_scene_mean": s.T_scene_ms_mean,
        "sw_per_100": s.sw_per_100,
        "switches": s.switches,
        "slow_pct": s.slow_pct,
        "total_frames": s.total_frames,
    }


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--video", type=str)
    parser.add_argument("--material", type=str, default="glass")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inf = InferenceParams(imgsz=640, device="cpu", stride=args.stride,
                          max_frames=args.max_frames)

    ablation_configs = get_ablation_configs()

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

    all_results = {}

    for vpath, material in video_list:
        if not os.path.isfile(vpath):
            print(f"WARNING: {vpath} not found")
            continue

        vname = Path(vpath).stem
        print(f"\n{'='*70}")
        print(f"  Ablation Study: {vname} ({material})")
        print(f"{'='*70}")

        model_n = YOLO(MODELS[material]["y8n"])
        model_s = YOLO(MODELS[material]["y8s"])

        video_results = {}
        for ac in ablation_configs:
            name = ac["name"]
            cfg = ac["config"]
            cat = ac["category"]
            print(f"  [{cat}] {name}...", end="", flush=True)
            t0 = time.time()
            metrics = run_ablation_variant(vpath, material, cfg, model_n, model_s, inf)
            elapsed = time.time() - t0
            metrics["category"] = cat
            metrics["name"] = name
            video_results[name] = metrics
            print(f" T={metrics.get('T_total_mean',0):.1f}ms "
                  f"sw={metrics.get('sw_per_100',0):.1f} ({elapsed:.0f}s)")

        all_results[f"{material}/{vname}"] = video_results

        # Print formatted ablation table
        print(f"\n  {'Variant':<28} {'Cat':<12} {'MeanT':>7} {'P95T':>7} "
              f"{'Sw/100':>7} {'s%':>6}")
        print(f"  {'-'*72}")
        for name, m in video_results.items():
            print(f"  {name:<28} {m.get('category',''):<12} "
                  f"{m.get('T_total_mean',0):>7.1f} {m.get('T_total_p95',0):>7.1f} "
                  f"{m.get('sw_per_100',0):>7.2f} {m.get('slow_pct',0):>5.1f}%")

    # Save
    with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_DIR / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
