"""CLI smoke test for thesis-analysis engine.

Usage:
    python scripts/run_thesis_analysis_smoke.py \\
        --video path/to/video.mp4 \\
        --fast-weights path/to/yolov8n.pt \\
        --accurate-weights path/to/yolov8s.pt \\
        --policy conf_ema \\
        --out-dir results/ \\
        --max-frames 30

Produces:
    results/analysis_per_frame.csv
    results/analysis_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.analysis_engine import FrameResult, run_analysis, summarise_analysis
from core.csv_logger import AnalysisCSVLogger
from core.inference_backend import UltralyticsYOLOBackend
from core.policies import POLICY_REGISTRY, PolicyConfig
from core.proxies import ProxyConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thesis-analysis engine smoke test",
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--fast-weights", required=True, help="Fast model weights (.pt/.onnx/.engine)")
    parser.add_argument("--accurate-weights", required=True, help="Accurate model weights (.pt/.onnx/.engine)")
    parser.add_argument("--policy", required=True, choices=sorted(POLICY_REGISTRY), help="Policy name")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu", help="Inference device (default: cpu)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--start-frame", type=int, default=0, help="First video frame to process (default: 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for metrics (default: 0.5)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "analysis_per_frame.csv"
    json_path = out_dir / "analysis_summary.json"

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

    print(f"Policy: {args.policy}")
    print(f"Video:  {args.video}")
    if args.start_frame > 0:
        print(f"Start frame: {args.start_frame}")
    if args.max_frames > 0:
        print(f"Max frames: {args.max_frames}")

    results = []
    with AnalysisCSVLogger(csv_path) as csv_log:
        for fr in run_analysis(
            video_path=args.video,
            fast_backend=fast,
            accurate_backend=accurate,
            policy_name=args.policy,
            iou_threshold=args.iou_threshold,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
        ):
            csv_log.log(fr)
            results.append(fr)
            if fr.frame_idx % 10 == 0:
                print(
                    f"  frame {fr.frame_idx:>5d}  "
                    f"choice={fr.choice}  "
                    f"iou={int(fr.iou_match)}  "
                    f"t_deployed={fr.t_total_deployed_ms:.1f}ms"
                )

    summary = summarise_analysis(results)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\nDone: {len(results)} frames processed.")
    print(f"CSV:     {csv_path}")
    print(f"Summary: {json_path}")
    print()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
