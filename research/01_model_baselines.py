"""
Phase 1A: Authoritative model baselines on MPID val sets.

Runs YOLO val on each model × dataset combination and reports:
  - mAP@0.5, mAP@0.5:0.95
  - Precision, Recall, F1
  - Per-class breakdown (single class for glass/porcelain)
  - Inference speed (preprocess + inference + NMS)

Usage:
    conda activate uav-thesis
    python research/01_model_baselines.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from core.research_paths import get_datasets

# ── Configuration ──────────────────────────────────────────────────────
DATASETS = get_datasets()
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "research_results" / "01_model_baselines"
IMGSZ = 640
CONF_THRESHOLD = 0.001   # standard for mAP evaluation
IOU_THRESHOLD = 0.5      # for mAP@0.5


def run_validation(dataset_name: str, model_tag: str, model_path: str,
                   data_yaml: str) -> dict:
    """Run YOLO validation and return metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Validating: {dataset_name} / {model_tag}")
    print(f"  Model:      {model_path}")
    print(f"  Data:       {data_yaml}")
    print(f"{'='*60}")

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device="cpu",
        verbose=True,
        save_json=False,
    )

    # Extract metrics
    metrics = {
        "dataset": dataset_name,
        "model": model_tag,
        "model_path": model_path,
        "imgsz": IMGSZ,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "device": "cpu",
        # Detection metrics
        "precision": float(results.box.mp),       # mean precision
        "recall": float(results.box.mr),           # mean recall
        "mAP50": float(results.box.map50),         # mAP @ IoU=0.5
        "mAP50_95": float(results.box.map),        # mAP @ IoU=0.5:0.95
        "f1": float(2 * results.box.mp * results.box.mr /
                     (results.box.mp + results.box.mr + 1e-9)),
        # Speed (ms per image)
        "speed_preprocess_ms": float(results.speed.get("preprocess", 0)),
        "speed_inference_ms": float(results.speed.get("inference", 0)),
        "speed_postprocess_ms": float(results.speed.get("postprocess", 0)),
        # Per-class (for single-class, this is the same as mean)
        "num_images": int(results.box.n) if hasattr(results.box, 'n') else 0,
        "timestamp": datetime.now().isoformat(),
    }

    # Try to get per-class AP if available
    try:
        if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
            metrics["per_class_ap50"] = results.box.ap50.tolist()
        if hasattr(results.box, 'ap') and results.box.ap is not None:
            metrics["per_class_ap50_95"] = results.box.ap.tolist()
    except Exception:
        pass

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_lines = []

    summary_lines.append(f"DMS-Raptor Model Baseline Validation")
    summary_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    summary_lines.append(f"Device: CPU | imgsz: {IMGSZ}")
    summary_lines.append("")
    summary_lines.append(f"{'Dataset':<12} {'Model':<6} {'Prec':>6} {'Recall':>7} "
                         f"{'F1':>6} {'mAP50':>7} {'mAP50-95':>9} {'Infer(ms)':>10}")
    summary_lines.append("-" * 72)

    for ds_name, ds_cfg in DATASETS.items():
        data_yaml = ds_cfg["data_yaml"]
        if not os.path.isfile(data_yaml):
            print(f"WARNING: data yaml not found: {data_yaml}, skipping {ds_name}")
            continue

        for model_tag, model_path in ds_cfg["models"].items():
            if not os.path.isfile(model_path):
                print(f"WARNING: model not found: {model_path}, skipping")
                continue

            metrics = run_validation(ds_name, model_tag, model_path, data_yaml)
            all_results.append(metrics)

            # Save individual result
            out_file = OUTPUT_DIR / f"{ds_name}_{model_tag}_val.json"
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=2)

            # Summary line
            line = (f"{ds_name:<12} {model_tag:<6} "
                    f"{metrics['precision']:>6.3f} {metrics['recall']:>7.3f} "
                    f"{metrics['f1']:>6.3f} {metrics['mAP50']:>7.3f} "
                    f"{metrics['mAP50_95']:>9.3f} "
                    f"{metrics['speed_inference_ms']:>10.1f}")
            summary_lines.append(line)
            print(f"\n>> {line}")

    # Save combined results
    with open(OUTPUT_DIR / "all_baselines.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save summary
    summary_text = "\n".join(summary_lines)
    with open(OUTPUT_DIR / "baseline_summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\n\n{'='*72}")
    print("BASELINE SUMMARY")
    print("=" * 72)
    print(summary_text)
    print(f"\nResults saved to: {OUTPUT_DIR}")

    # Also print the key comparison for the paper
    print(f"\n{'='*72}")
    print("KEY FINDING: Model accuracy gap (justifies DMS)")
    print("=" * 72)
    for ds_name in DATASETS:
        y8n = next((r for r in all_results
                    if r["dataset"] == ds_name and r["model"] == "y8n"), None)
        y8s = next((r for r in all_results
                    if r["dataset"] == ds_name and r["model"] == "y8s"), None)
        if y8n and y8s:
            gap_map50 = y8s["mAP50"] - y8n["mAP50"]
            gap_map = y8s["mAP50_95"] - y8n["mAP50_95"]
            speedup = y8s["speed_inference_ms"] / max(0.1, y8n["speed_inference_ms"])
            print(f"  {ds_name}: y8s mAP50 +{gap_map50:+.3f} over y8n | "
                  f"y8s is {speedup:.1f}x slower")


if __name__ == "__main__":
    main()
