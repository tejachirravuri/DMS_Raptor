"""
Extract sample frames routed to n-model vs s-model for visual evidence.

For each image-proxy switching policy, randomly selects:
  - 30 n-routed frames (5 with 0 detections + 25 with detections)
  - 30 s-routed frames (5 with 0 detections + 25 with detections)

Each frame is saved as a PNG with a metadata bar at the bottom showing:
  Frame#, L, H, C, NIQE, Dets, MeanConf, Choice

Usage:
    python research/12_extract_routed_frames.py \
        --video "path/to/20190916-633-01.mp4" \
        --results-dir "path/to/overnight_full/20190916-633-01" \
        --output-dir "path/to/routing_evidence"

    # Specific policies only:
    python research/12_extract_routed_frames.py \
        --video ... --results-dir ... \
        --policies entropy_only conf_ema

    # Change sample count:
    python research/12_extract_routed_frames.py \
        --video ... --results-dir ... \
        --n-frames 30 --n-zero-det 5
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np

# Proxy-based switching policies (skip baselines)
PROXY_POLICIES = [
    "entropy_only", "combined", "combined_hyst",
    "conf_ema", "niqe_switch", "multi_proxy",
]


def read_trace(csv_path: str) -> list[dict]:
    """Read per-frame trace CSV into list of dicts."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def sample_frames(rows: list[dict], choice: str,
                  n_total: int, n_zero_det: int) -> list[dict]:
    """Pick n_total frames with given choice: n_zero_det with 0 dets + rest with dets."""
    zero_det = [r for r in rows if r["choice"] == choice
                and int(r["num_detections"]) == 0]
    with_det = [r for r in rows if r["choice"] == choice
                and int(r["num_detections"]) > 0]

    n_with_det = n_total - n_zero_det

    # Sample what we can
    picked_zero = random.sample(zero_det, min(n_zero_det, len(zero_det)))
    picked_det = random.sample(with_det, min(n_with_det, len(with_det)))

    # If one pool is short, fill from the other
    shortfall = n_total - len(picked_zero) - len(picked_det)
    if shortfall > 0 and len(zero_det) > len(picked_zero):
        extra = [r for r in zero_det if r not in picked_zero]
        picked_zero.extend(random.sample(extra, min(shortfall, len(extra))))
        shortfall = n_total - len(picked_zero) - len(picked_det)
    if shortfall > 0 and len(with_det) > len(picked_det):
        extra = [r for r in with_det if r not in picked_det]
        picked_det.extend(random.sample(extra, min(shortfall, len(extra))))

    result = picked_zero + picked_det
    result.sort(key=lambda r: int(r["frame_idx"]))
    return result


def extract_and_save(video_path: str, sampled: list[dict],
                     out_dir: Path, policy: str):
    """Read frames from video and save as annotated PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build frame_idx -> row mapping
    frame_map = {int(r["frame_idx"]): r for r in sampled}
    target_frames = sorted(frame_map.keys())

    if not target_frames:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: Cannot open video {video_path}")
        return

    saved = 0
    for fidx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        r = frame_map[fidx]
        h, w = frame.shape[:2]

        # Build metadata text
        choice = r["choice"]
        dets = r["num_detections"]
        L_val = r.get("L", "")
        H_val = r.get("H", "")
        C_val = r.get("C_score", "")
        niqe_val = r.get("niqe_score", "")
        conf_val = r.get("mean_conf", "")

        line1 = f"Frame {fidx}  |  Choice: {'YOLOv8n' if choice == 'n' else 'YOLOv8s'}  |  Detections: {dets}"

        parts = []
        if L_val and L_val != "":
            parts.append(f"L={float(L_val):.1f}")
        if H_val and H_val != "":
            parts.append(f"H={float(H_val):.4f}")
        if C_val and C_val != "":
            parts.append(f"C={float(C_val):.4f}")
        if niqe_val and niqe_val != "" and float(niqe_val) != 0:
            parts.append(f"NIQE={float(niqe_val):.4f}")
        if conf_val and conf_val != "":
            parts.append(f"MeanConf={float(conf_val):.4f}")
        line2 = "  |  ".join(parts) if parts else ""

        # Draw metadata bar at bottom
        bar_h = 80
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        cv2.putText(bar, line1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        if line2:
            cv2.putText(bar, line2, (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

        # Add colored border (green=n, red=s)
        border = 6
        color = (0, 200, 0) if choice == "n" else (0, 0, 200)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, border)

        combined = np.vstack([frame, bar])

        fname = f"frame_{fidx:05d}_dets{dets}.png"
        cv2.imwrite(str(out_dir / fname), combined)
        saved += 1

    cap.release()
    print(f"    Saved {saved} frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract routed frame samples for supervisor presentation"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to source video")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to overnight results for this video "
                             "(contains policy subdirs with per_frame_trace.csv)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results-dir/routing_evidence)")
    parser.add_argument("--policies", nargs="+", default=None,
                        help=f"Policies to extract (default: all proxy policies). "
                             f"Valid: {PROXY_POLICIES}")
    parser.add_argument("--n-frames", type=int, default=30,
                        help="Frames per group (default: 30)")
    parser.add_argument("--n-zero-det", type=int, default=5,
                        help="Zero-detection frames per group (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "routing_evidence"
    policies = args.policies or PROXY_POLICIES

    print(f"Video: {args.video}")
    print(f"Results: {results_dir}")
    print(f"Output: {output_dir}")
    print(f"Policies: {policies}")
    print(f"Frames per group: {args.n_frames} ({args.n_zero_det} zero-det + "
          f"{args.n_frames - args.n_zero_det} with-det)")
    print()

    for policy in policies:
        trace_csv = results_dir / policy / "per_frame_trace.csv"
        if not trace_csv.exists():
            print(f"  SKIP {policy}: no per_frame_trace.csv")
            continue

        print(f"  {policy}:")
        rows = read_trace(str(trace_csv))

        n_rows = [r for r in rows if r["choice"] == "n"]
        s_rows = [r for r in rows if r["choice"] == "s"]
        print(f"    Total: {len(rows)} frames | n-routed: {len(n_rows)} | s-routed: {len(s_rows)}")

        # Sample n-routed
        n_sampled = sample_frames(rows, "n", args.n_frames, args.n_zero_det)
        print(f"    n-routed sample: {len(n_sampled)} "
              f"({sum(1 for r in n_sampled if int(r['num_detections']) == 0)} zero-det)")
        extract_and_save(args.video, n_sampled,
                         output_dir / policy / "n_routed", policy)

        # Sample s-routed
        s_sampled = sample_frames(rows, "s", args.n_frames, args.n_zero_det)
        print(f"    s-routed sample: {len(s_sampled)} "
              f"({sum(1 for r in s_sampled if int(r['num_detections']) == 0)} zero-det)")
        extract_and_save(args.video, s_sampled,
                         output_dir / policy / "s_routed", policy)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
