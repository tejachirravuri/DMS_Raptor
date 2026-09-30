"""CLI for multi-policy thesis-analysis on a single video.

Usage:
    python scripts/run_thesis_analysis_policies.py ^
        --video path/to/video.mp4 ^
        --fast-weights path/to/yolov8n.pt ^
        --accurate-weights path/to/yolov8s.pt ^
        --out-dir results/analysis ^
        --max-frames 50

Produces per-policy CSVs, per-policy summary JSONs, and a compact
analysis_policy_summary.csv with informed-gain columns.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.analysis_runner import run_analysis_policies, SUMMARY_ALL_COLUMNS
from core.inference_backend import UltralyticsYOLOBackend
from core.policies import POLICY_REGISTRY

_DEFAULT_POLICIES = (
    "n_only,s_only,conf_ema,local_contrast_hyst,"
    "combined_hyst,entropy_only,combined,multi_proxy"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-policy thesis-analysis runner",
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--fast-weights", required=True)
    parser.add_argument("--accurate-weights", required=True)
    parser.add_argument(
        "--policies", default=_DEFAULT_POLICIES,
        help=f"Comma-separated policy names (default: {_DEFAULT_POLICIES})",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        if p not in POLICY_REGISTRY:
            parser.error(f"unknown policy: {p!r}")

    print(f"Loading fast model:     {args.fast_weights}")
    fast = UltralyticsYOLOBackend(
        args.fast_weights, device=args.device,
        imgsz=args.imgsz, conf=args.conf,
    )
    print(f"Loading accurate model: {args.accurate_weights}")
    accurate = UltralyticsYOLOBackend(
        args.accurate_weights, device=args.device,
        imgsz=args.imgsz, conf=args.conf,
    )

    print(f"Video:    {args.video}")
    print(f"Policies: {', '.join(policies)}")
    if args.start_frame > 0:
        print(f"Start frame: {args.start_frame}")
    if args.max_frames > 0:
        print(f"Max frames: {args.max_frames}")
    print()

    def _progress(policy, status):
        if status == "running":
            print(f"  [{policies.index(policy)+1}/{len(policies)}] {policy} ...")
        elif status == "done":
            print(f"  [{policies.index(policy)+1}/{len(policies)}] {policy} done")

    summaries = run_analysis_policies(
        video_path=args.video,
        fast_backend=fast,
        accurate_backend=accurate,
        policies=policies,
        out_dir=args.out_dir,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        progress_callback=_progress,
    )

    print(f"\nDone. Output: {args.out_dir}")
    print()

    summary_csv = Path(args.out_dir) / "analysis_policy_summary.csv"
    with open(summary_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    fmt = "{:<22s} {:>6s} {:>7s} {:>9s} {:>9s} {:>8s} {:>10s}"
    print(fmt.format(
        "policy", "frames", "s_rate", "iou_match", "det_cov",
        "benefit", "gain_iou",
    ))
    print("-" * 80)
    for r in rows:
        gain = r.get("informed_gain_iou_match", "")
        gain_str = f"{float(gain):.4f}" if gain else ""
        print(fmt.format(
            r["policy"],
            r["n_frames"],
            f"{float(r['accurate_usage_rate']):.3f}",
            f"{float(r['mean_iou_match']):.4f}",
            f"{float(r['mean_det_coverage']):.4f}",
            f"{float(r['benefit_rate']):.4f}",
            gain_str,
        ))


if __name__ == "__main__":
    main()
