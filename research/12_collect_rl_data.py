#!/usr/bin/env python3
"""
Phase 12: Collect dual-model RL training data.

Runs BOTH YOLOv8n and YOLOv8s on every frame of every video, saving
features, detections, IoU agreement, and inference times to HDF5 files.

This creates the offline dataset used by scripts 13 (bandits) and 14 (DQN).

On GPU: ~40ms per frame (both models). 78K frames → ~52 minutes.
On CPU: ~400ms per frame. 78K frames → ~8.7 hours.

Usage:
    # All videos, GPU
    python research/12_collect_rl_data.py --device cuda

    # Quick test (500 frames from one video)
    python research/12_collect_rl_data.py --videos 633-01 --max-frames 500

    # Specific material
    python research/12_collect_rl_data.py --material glass --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

from rl_switching.config import RLConfig
from rl_switching.data.collector import DualModelCollector


def discover_videos(base_dir: str) -> dict:
    """Auto-discover input videos.  Returns {video_id: path}."""
    videos = {}
    for material in ["glass_insulator_videos", "porcelain_insulator_videos"]:
        mat_dir = os.path.join(base_dir, material)
        if not os.path.isdir(mat_dir):
            continue
        for fname in sorted(os.listdir(mat_dir)):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                vid_id = os.path.splitext(fname)[0]
                videos[vid_id] = os.path.join(mat_dir, fname)
    return videos


def main():
    parser = argparse.ArgumentParser(description="Phase 12: Collect RL training data")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Specific video IDs to process (default: all)")
    parser.add_argument("--material", choices=["glass", "porcelain", "all"],
                        default="all", help="Which material to process")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video (0=all)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame")
    parser.add_argument("--model-n", default=None,
                        help="Path to n-model .pt (auto-detected if not set)")
    parser.add_argument("--model-s", default=None,
                        help="Path to s-model .pt (auto-detected if not set)")
    parser.add_argument("--video-dir", default=None,
                        help="Base directory containing input videos")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for HDF5 files")
    args = parser.parse_args()

    # ── Auto-detect paths ────────────────────────────────────────────
    # Works on both local (G:\Teja_Master_Thesis\DMS_Raptor) and
    # remote (DMS_Clean) setups.
    thesis_root = os.path.dirname(PROJECT_ROOT)  # G:\Teja_Master_Thesis

    if args.video_dir:
        video_base = args.video_dir
    else:
        # Try common locations
        candidates = [
            os.path.join(thesis_root, "MT_Chirravuri", "input_videos"),
            os.path.join(PROJECT_ROOT, "input_videos"),
            os.path.join(thesis_root, "input_videos"),
        ]
        video_base = next((c for c in candidates if os.path.isdir(c)), candidates[0])

    if args.model_n:
        model_n_path = args.model_n
    else:
        candidates = [
            os.path.join(thesis_root, "MT_Chirravuri", "models", "glass_y8n.pt"),
            os.path.join(PROJECT_ROOT, "models", "glass_y8n.pt"),
        ]
        model_n_path = next((c for c in candidates if os.path.isfile(c)), candidates[0])

    if args.model_s:
        model_s_path = args.model_s
    else:
        candidates = [
            os.path.join(thesis_root, "MT_Chirravuri", "models", "glass_y8s.pt"),
            os.path.join(PROJECT_ROOT, "models", "glass_y8s.pt"),
        ]
        model_s_path = next((c for c in candidates if os.path.isfile(c)), candidates[0])

    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "research_results", "rl_switching", "data_cache"
    )

    print("=" * 70)
    print("Phase 12: Collect RL Training Data")
    print("=" * 70)
    print(f"  Device:    {args.device}")
    print(f"  Video dir: {video_base}")
    print(f"  Model n:   {model_n_path}")
    print(f"  Model s:   {model_s_path}")
    print(f"  Output:    {output_dir}")
    print(f"  Stride:    {args.stride}")
    print(f"  Max frames:{args.max_frames or 'all'}")
    print()

    # ── Discover videos ──────────────────────────────────────────────
    all_videos = discover_videos(video_base)

    if args.videos:
        videos = {k: v for k, v in all_videos.items()
                  if any(sel in k for sel in args.videos)}
    elif args.material != "all":
        mat_key = f"{args.material}_insulator"
        videos = {k: v for k, v in all_videos.items() if mat_key in v}
    else:
        videos = all_videos

    if not videos:
        print(f"ERROR: No videos found in {video_base}")
        print(f"  Available: {list(all_videos.keys())}")
        sys.exit(1)

    print(f"Videos to process ({len(videos)}):")
    for vid_id, vid_path in videos.items():
        print(f"  {vid_id}: {vid_path}")
    print()

    # ── Load models ──────────────────────────────────────────────────
    print("Loading models...")
    model_n = YOLO(model_n_path)
    model_s = YOLO(model_s_path)
    print(f"  n-model: {model_n_path}")
    print(f"  s-model: {model_s_path}")

    # Warmup
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model_n.predict(source=dummy, imgsz=640, device=args.device, verbose=False)
    model_s.predict(source=dummy, imgsz=640, device=args.device, verbose=False)
    print("  Models warmed up.\n")

    # ── Configure ────────────────────────────────────────────────────
    cfg = RLConfig(
        device=args.device,
        data_cache_dir=output_dir,
        collect_stride=args.stride,
        max_frames=args.max_frames,
    )

    # ── Collect ──────────────────────────────────────────────────────
    collector = DualModelCollector(cfg, model_n, model_s)

    t_start = time.time()
    collector.collect_all(videos)
    elapsed = time.time() - t_start

    manifest_path = collector.save_manifest()

    print()
    print("=" * 70)
    print(f"Collection complete in {elapsed/60:.1f} minutes")
    print(f"Manifest: {manifest_path}")
    print()
    for vid_id, info in collector.manifest.items():
        print(f"  {vid_id}: {info['n_frames']} frames, "
              f"IoU={info['mean_iou']:.3f}, "
              f"t_n={info['mean_t_n']:.1f}ms, t_s={info['mean_t_s']:.1f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
