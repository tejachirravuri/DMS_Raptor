"""
Phase 8: Generate annotated output videos for all videos × selected policies.

Runs each video through the StreamingEngine with save_annotated_video=True,
producing MP4 files with detection overlays, policy badges, and metadata headers.

NOTE: This is CPU-intensive and takes several hours for all 12 videos × 8 policies.
      Use --policies to select a subset (e.g., --policies n_only conf_ema combined_hyst).
      Use --videos to filter specific videos.

Outputs to: research_results/annotated_videos/
  └── {video_name}/
      ├── annotated_{video_name}_n_only.mp4
      ├── annotated_{video_name}_conf_ema.mp4
      └── ...

Usage:
    python research/08_generate_annotated_videos.py
    python research/08_generate_annotated_videos.py --policies n_only conf_ema combined_hyst
    python research/08_generate_annotated_videos.py --policies conf_ema --videos 633-01 porce
    python research/08_generate_annotated_videos.py --max-frames 500  # quick preview
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InferenceParams, RunConfig, POLICIES
from core.engine import StreamingEngine
from core.research_paths import MODELS, VIDEOS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "research_results" / "annotated_videos"


def run_video_policy(video_path: str, material: str, policy: str,
                     inf: InferenceParams, model_n, model_s,
                     output_dir: Path) -> dict:
    """Run one video with one policy and save annotated output."""
    video_name = Path(video_path).stem
    out_path = output_dir / f"annotated_{video_name}_{policy}.mp4"

    if out_path.exists():
        print(f"    SKIP (exists): {out_path.name}")
        return {"video": video_name, "policy": policy, "status": "skipped"}

    cfg = RunConfig(
        name=f"{video_name}_{policy}",
        policy=policy,
        save_annotated_video=True,
        output_video_path=str(out_path),
    )

    engine = StreamingEngine(
        source_path=video_path,
        cfg=cfg,
        model_n=model_n,
        model_s=model_s,
        inf=inf,
        source_type="video",
    )

    t0 = time.time()
    frame_count = 0
    total_frames = engine.total_frames or 0
    for fr in engine.run():
        frame_count += 1
        if frame_count % 200 == 0:
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            pct = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"      Frame {frame_count}/{total_frames} "
                  f"({pct:.0f}%) — {fps:.1f} fps", end="\r")

    elapsed = time.time() - t0
    print(f"    Done: {out_path.name} — {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps)")

    return {
        "video": video_name, "policy": policy, "status": "done",
        "frames": frame_count, "elapsed_s": elapsed,
        "output": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate annotated videos for all experiments")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Policies to run (default: n_only s_only conf_ema combined_hyst)")
    parser.add_argument("--all-policies", action="store_true",
                        help="Run all 8 policies (very slow)")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Filter videos by substring match (e.g., 633-01 porce)")
    parser.add_argument("--materials", nargs="+", default=["glass", "porcelain"],
                        help="Material types to process")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video (0=all)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Default to the 4 most important policies
    if args.all_policies:
        policies = POLICIES
    elif args.policies:
        policies = args.policies
    else:
        policies = ["n_only", "s_only", "conf_ema", "combined_hyst"]

    # Validate policies
    for p in policies:
        if p not in POLICIES:
            print(f"ERROR: Unknown policy '{p}'. Valid: {POLICIES}")
            return

    inf = InferenceParams(imgsz=640, device=args.device, max_frames=args.max_frames)

    print("=" * 65)
    print("  DMS-Raptor: Annotated Video Generation")
    print("=" * 65)
    print(f"  Policies: {policies}")
    print(f"  Materials: {args.materials}")
    print(f"  Max frames: {'all' if args.max_frames == 0 else args.max_frames}")
    print(f"  Device: {args.device}")

    # Count total jobs
    total_jobs = 0
    for mat in args.materials:
        for vpath in VIDEOS.get(mat, []):
            vname = Path(vpath).stem
            if args.videos and not any(f in vname for f in args.videos):
                continue
            total_jobs += len(policies)
    print(f"  Total jobs: {total_jobs}")
    print("=" * 65)

    results = []
    job_idx = 0

    for mat in args.materials:
        model_paths = MODELS.get(mat)
        if not model_paths:
            continue

        # Load models once per material
        print(f"\n  Loading {mat} models...")
        from ultralytics import YOLO
        model_n = YOLO(model_paths["y8n"])
        model_s = YOLO(model_paths["y8s"])

        # Warmup
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model_n(dummy, imgsz=640, verbose=False)
        model_s(dummy, imgsz=640, verbose=False)
        print(f"  Models loaded and warmed up.")

        for vpath in VIDEOS.get(mat, []):
            vname = Path(vpath).stem

            # Filter
            if args.videos and not any(f in vname for f in args.videos):
                continue

            if not os.path.exists(vpath):
                print(f"\n  WARNING: Video not found: {vpath}")
                continue

            vid_out_dir = OUTPUT_DIR / vname
            vid_out_dir.mkdir(parents=True, exist_ok=True)

            for policy in policies:
                job_idx += 1
                print(f"\n  [{job_idx}/{total_jobs}] {mat}/{vname} — {policy}")

                try:
                    result = run_video_policy(
                        vpath, mat, policy, inf, model_n, model_s, vid_out_dir
                    )
                    results.append(result)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({
                        "video": vname, "policy": policy,
                        "status": "error", "error": str(e),
                    })

    # Summary
    done = sum(1 for r in results if r["status"] == "done")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'=' * 65}")
    print(f"  Complete! {done} generated, {skipped} skipped, {errors} errors")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 65}")

    # List all output files
    if OUTPUT_DIR.exists():
        total_size = 0
        count = 0
        for f in OUTPUT_DIR.rglob("*.mp4"):
            total_size += f.stat().st_size
            count += 1
        print(f"  Total annotated videos: {count}")
        print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
