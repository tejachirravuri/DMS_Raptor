#!/usr/bin/env python3
"""
11_frame_level_validation.py — Frame-Level Proxy Routing Validation

For each proxy-based policy, samples frames routed to n-model and s-model,
runs BOTH models on each sampled frame, and produces:

  Per-policy output:
    n_routed_samples/        30 annotated PNGs (5 zero-det + 25 with-det)
    s_routed_samples/        30 annotated PNGs (5 zero-det + 25 with-det)
    frame_grid_n_routed.png  5x6 thumbnail grid
    frame_grid_s_routed.png  5x6 thumbnail grid
    validation_stats.csv     Per-frame: proxy values, n-dets, s-dets, IoU
    validation_summary.json  Aggregate metrics

  Grand output:
    proxy_vs_iou_scatter.png       Proxy value vs IoU(n,s)
    iou_distribution.png           Histogram: IoU for n-routed vs s-routed
    detection_count_comparison.png Grouped bar: n-dets vs s-dets
    proxy_routing_accuracy.png     Bar chart: routing correctness %
    grand_validation_summary.json  Cross-policy summary

Usage:
    python research/11_frame_level_validation.py --device cuda
    python research/11_frame_level_validation.py --device cuda --videos 633-01
    python research/11_frame_level_validation.py --device cuda --policies combined_hyst conf_ema
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from core.engine import run_model_single, complexity_proxies_fast
from core.config import InferenceParams
from core.research_paths import MODELS, VIDEOS

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
# Proxy-based policies (skip n_only and s_only baselines)
PROXY_POLICIES = [
    "entropy_only", "combined", "combined_hyst",
    "conf_ema", "niqe_switch", "multi_proxy",
]

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "research_results" / "11_frame_validation"

N_SAMPLES_ZERO_DET = 5
N_SAMPLES_WITH_DET = 25
N_TOTAL = N_SAMPLES_ZERO_DET + N_SAMPLES_WITH_DET

CONF_SHOW = 0.25   # confidence threshold for counting visible detections
IMGSZ = 640
IOU_NMS = 0.45

# Colors
COLOR_N = (0, 200, 0)       # Green (BGR) for n-model
COLOR_S = (0, 0, 230)       # Red (BGR) for s-model
COLOR_TEXT_BG = (40, 40, 40)


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2,...]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_set_iou(dets_n: List[np.ndarray], dets_s: List[np.ndarray],
                    conf_thresh: float = 0.25) -> float:
    """Compute average best-match IoU between two detection sets.

    For each detection in s (reference), find best matching detection in n.
    Returns mean of best IoUs. If both empty, returns 1.0 (perfect agreement).
    If one is empty and other isn't, returns 0.0 (complete disagreement).
    """
    dets_n_f = [d for d in dets_n if d[4] >= conf_thresh]
    dets_s_f = [d for d in dets_s if d[4] >= conf_thresh]

    if len(dets_s_f) == 0 and len(dets_n_f) == 0:
        return 1.0  # both see nothing — agreement
    if len(dets_s_f) == 0 or len(dets_n_f) == 0:
        return 0.0  # one sees, other doesn't — disagreement

    ious = []
    for ds in dets_s_f:
        best = max(box_iou(ds, dn) for dn in dets_n_f)
        ious.append(best)
    for dn in dets_n_f:
        best = max(box_iou(dn, ds) for ds in dets_s_f)
        ious.append(best)
    return float(np.mean(ious))


def draw_detections(frame: np.ndarray, dets: List[np.ndarray],
                    color: Tuple[int, int, int], label: str,
                    conf_thresh: float = 0.25, side: str = "left") -> np.ndarray:
    """Draw bounding boxes on frame with model label."""
    vis = frame.copy()
    count = 0
    for d in dets:
        if d[4] >= conf_thresh:
            x1, y1, x2, y2, conf = int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            count += 1

    # Model label in top corner
    h, w = vis.shape[:2]
    x_pos = 10 if side == "left" else w // 2 + 10
    cv2.putText(vis, f"{label}: {count} dets", (x_pos, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return vis


def create_side_by_side(frame: np.ndarray, dets_n: List[np.ndarray],
                        dets_s: List[np.ndarray], frame_idx: int,
                        proxy_values: Dict[str, float], iou: float,
                        choice: str, policy: str) -> np.ndarray:
    """Create annotated side-by-side image: n-model (green) | s-model (red)."""
    h, w = frame.shape[:2]

    # Draw detections
    vis_n = draw_detections(frame, dets_n, COLOR_N, "YOLOv8n", CONF_SHOW, "left")
    vis_s = draw_detections(frame, dets_s, COLOR_S, "YOLOv8s", CONF_SHOW, "left")

    # Side by side
    combined = np.hstack([vis_n, vis_s])
    ch, cw = combined.shape[:2]

    # Add border showing routing decision
    border_color = COLOR_N if choice == "n" else COLOR_S
    cv2.rectangle(combined, (0, 0), (cw - 1, ch - 1), border_color, 4)

    # Divider line
    cv2.line(combined, (w, 0), (w, ch), (255, 255, 255), 2)

    # Bottom info bar
    n_count = sum(1 for d in dets_n if d[4] >= CONF_SHOW)
    s_count = sum(1 for d in dets_s if d[4] >= CONF_SHOW)

    bar_h = 45
    info_bar = np.full((bar_h, cw, 3), COLOR_TEXT_BG, dtype=np.uint8)

    # Proxy info string
    proxy_str = " | ".join(f"{k}={v:.3f}" for k, v in proxy_values.items() if v != 0)

    line1 = f"Frame #{frame_idx} | Policy: {policy} | Routed to: {'n (fast)' if choice == 'n' else 's (accurate)'}"
    line2 = f"n-dets: {n_count} | s-dets: {s_count} | IoU(n,s): {iou:.3f} | {proxy_str}"

    cv2.putText(info_bar, line1, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info_bar, line2, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([combined, info_bar])


def create_thumbnail_grid(images: List[np.ndarray], cols: int = 6,
                          thumb_w: int = 320, thumb_h: int = 180) -> np.ndarray:
    """Create a grid of thumbnail images."""
    rows_needed = (len(images) + cols - 1) // cols
    grid = np.full((rows_needed * thumb_h, cols * thumb_w, 3), 30, dtype=np.uint8)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        grid[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

    return grid


# ═══════════════════════════════════════════════════════════════════════
# Core validation
# ═══════════════════════════════════════════════════════════════════════
def load_trace_csv(csv_path: Path) -> List[Dict]:
    """Load per_frame_trace.csv into list of dicts."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def sample_frames(trace: List[Dict], choice_filter: str,
                  n_zero: int, n_with: int) -> Tuple[List[int], List[int]]:
    """Sample frame indices: n_zero with 0 detections + n_with with detections.

    Returns (zero_det_indices, with_det_indices) from trace rows.
    """
    zero_det = []
    with_det = []

    for i, row in enumerate(trace):
        if row.get("choice", "") != choice_filter:
            continue
        n_dets = int(row.get("num_detections", 0))
        if n_dets == 0:
            zero_det.append(i)
        else:
            with_det.append(i)

    # Sample
    random.seed(42)  # reproducible
    sampled_zero = random.sample(zero_det, min(n_zero, len(zero_det)))
    sampled_with = random.sample(with_det, min(n_with, len(with_det)))

    # If not enough zero-det frames, fill from with-det (and vice versa)
    total_needed = n_zero + n_with
    total_got = len(sampled_zero) + len(sampled_with)
    if total_got < total_needed:
        remaining_zero = [i for i in zero_det if i not in sampled_zero]
        remaining_with = [i for i in with_det if i not in sampled_with]
        shortfall = total_needed - total_got
        extra = random.sample(remaining_with + remaining_zero,
                              min(shortfall, len(remaining_with) + len(remaining_zero)))
        # Distribute extras to whichever category was short
        if len(sampled_zero) < n_zero:
            zero_needed = n_zero - len(sampled_zero)
            sampled_zero.extend(extra[:zero_needed])
            extra = extra[zero_needed:]
        sampled_with.extend(extra)

    return sampled_zero, sampled_with


def validate_policy(video_path: str, material: str, policy: str,
                    trace_csv: Path, model_n, model_s,
                    device: str, out_dir: Path) -> Dict:
    """Run frame-level validation for one video-policy combination."""
    trace = load_trace_csv(trace_csv)
    if not trace:
        print(f"    [SKIP] Empty trace CSV: {trace_csv}")
        return {}

    # Sample frames
    n_zero_idx, n_with_idx = sample_frames(trace, "n", N_SAMPLES_ZERO_DET, N_SAMPLES_WITH_DET)
    s_zero_idx, s_with_idx = sample_frames(trace, "s", N_SAMPLES_ZERO_DET, N_SAMPLES_WITH_DET)

    all_n_idx = sorted(n_zero_idx + n_with_idx)
    all_s_idx = sorted(s_zero_idx + s_with_idx)
    all_indices = sorted(set(all_n_idx + all_s_idx))

    if not all_indices:
        print(f"    [SKIP] No frames to sample for {policy}")
        return {}

    # Map trace index to raw frame index
    trace_to_raw = {}
    for i in all_indices:
        trace_to_raw[i] = int(trace[i].get("frame_idx", i))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [ERROR] Cannot open video: {video_path}")
        return {}

    # Process sampled frames
    n_dir = out_dir / "n_routed_samples"
    s_dir = out_dir / "s_routed_samples"
    n_dir.mkdir(parents=True, exist_ok=True)
    s_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    n_images = []
    s_images = []

    for trace_idx in all_indices:
        row = trace[trace_idx]
        raw_frame = trace_to_raw[trace_idx]
        choice = row.get("choice", "n")

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        # Run BOTH models
        dets_n = run_model_single(model_n, frame, IMGSZ, device, 0.001, IOU_NMS)
        dets_s = run_model_single(model_s, frame, IMGSZ, device, 0.001, IOU_NMS)

        # Compute IoU between detection sets
        iou = compute_set_iou(dets_n, dets_s, CONF_SHOW)

        # Proxy values from trace
        proxy_values = {}
        for key in ["C_score", "L", "H", "mean_conf", "conf_drop", "niqe_score"]:
            val = row.get(key, "")
            proxy_values[key] = float(val) if val else 0.0

        n_count = sum(1 for d in dets_n if d[4] >= CONF_SHOW)
        s_count = sum(1 for d in dets_s if d[4] >= CONF_SHOW)

        # Create annotated image
        vis = create_side_by_side(frame, dets_n, dets_s, raw_frame,
                                  proxy_values, iou, choice, policy)

        # Save
        if trace_idx in all_n_idx:
            cat = "zero_det" if trace_idx in n_zero_idx else "with_det"
            fname = f"n_{cat}_frame{raw_frame:06d}.png"
            cv2.imwrite(str(n_dir / fname), vis)
            n_images.append(vis)
        if trace_idx in all_s_idx:
            cat = "zero_det" if trace_idx in s_zero_idx else "with_det"
            fname = f"s_{cat}_frame{raw_frame:06d}.png"
            cv2.imwrite(str(s_dir / fname), vis)
            s_images.append(vis)

        # Stats
        stats_rows.append({
            "frame_idx": raw_frame,
            "choice": choice,
            "n_detections": n_count,
            "s_detections": s_count,
            "iou_n_s": f"{iou:.4f}",
            "C_score": proxy_values.get("C_score", 0),
            "L": proxy_values.get("L", 0),
            "H": proxy_values.get("H", 0),
            "mean_conf": proxy_values.get("mean_conf", 0),
            "conf_drop": proxy_values.get("conf_drop", 0),
            "niqe_score": proxy_values.get("niqe_score", 0),
            "category": "zero_det" if (trace_idx in n_zero_idx or trace_idx in s_zero_idx) else "with_det",
        })

    cap.release()

    # Save stats CSV
    if stats_rows:
        with open(out_dir / "validation_stats.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats_rows[0].keys())
            writer.writeheader()
            writer.writerows(stats_rows)

    # Create thumbnail grids
    if n_images:
        grid_n = create_thumbnail_grid(n_images)
        cv2.imwrite(str(out_dir / "frame_grid_n_routed.png"), grid_n)
    if s_images:
        grid_s = create_thumbnail_grid(s_images)
        cv2.imwrite(str(out_dir / "frame_grid_s_routed.png"), grid_s)

    # Compute summary metrics
    n_rows = [r for r in stats_rows if r["choice"] == "n"]
    s_rows = [r for r in stats_rows if r["choice"] == "s"]

    n_ious = [float(r["iou_n_s"]) for r in n_rows]
    s_ious = [float(r["iou_n_s"]) for r in s_rows]

    # Routing accuracy: n-routed should have high IoU (n sufficient)
    # s-routed should have low IoU (s needed)
    n_correct = sum(1 for iou_val in n_ious if iou_val >= 0.5) if n_ious else 0
    s_correct = sum(1 for iou_val in s_ious if iou_val < 0.5) if s_ious else 0

    summary = {
        "policy": policy,
        "video": Path(video_path).stem,
        "material": material,
        "n_routed_count": len(n_rows),
        "s_routed_count": len(s_rows),
        "n_mean_iou": float(np.mean(n_ious)) if n_ious else None,
        "s_mean_iou": float(np.mean(s_ious)) if s_ious else None,
        "n_routing_accuracy": n_correct / len(n_ious) if n_ious else None,
        "s_routing_accuracy": s_correct / len(s_ious) if s_ious else None,
        "n_mean_dets_n": float(np.mean([int(r["n_detections"]) for r in n_rows])) if n_rows else None,
        "n_mean_dets_s": float(np.mean([int(r["s_detections"]) for r in n_rows])) if n_rows else None,
        "s_mean_dets_n": float(np.mean([int(r["n_detections"]) for r in s_rows])) if s_rows else None,
        "s_mean_dets_s": float(np.mean([int(r["s_detections"]) for r in s_rows])) if s_rows else None,
    }

    with open(out_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ═══════════════════════════════════════════════════════════════════════
# Grand summary plots
# ═══════════════════════════════════════════════════════════════════════
def plot_grand_summary(all_summaries: List[Dict], all_stats: List[Dict],
                       out_dir: Path):
    """Generate cross-policy validation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    if not all_stats:
        return

    # ── 1. IoU Distribution: n-routed vs s-routed ──
    n_ious = [float(r["iou_n_s"]) for r in all_stats if r["choice"] == "n"]
    s_ious = [float(r["iou_n_s"]) for r in all_stats if r["choice"] == "s"]

    if n_ious and s_ious:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 25)
        ax.hist(n_ious, bins=bins, alpha=0.7, color="green", label=f"n-routed (n={len(n_ious)})", edgecolor="darkgreen")
        ax.hist(s_ious, bins=bins, alpha=0.7, color="red", label=f"s-routed (n={len(s_ious)})", edgecolor="darkred")
        ax.set_xlabel("IoU(n-model, s-model)")
        ax.set_ylabel("Frame count")
        ax.set_title("Detection Agreement: n-routed vs s-routed frames")
        ax.legend()
        ax.axvline(np.mean(n_ious), color="darkgreen", linestyle="--", alpha=0.8,
                    label=f"n-routed mean={np.mean(n_ious):.3f}")
        ax.axvline(np.mean(s_ious), color="darkred", linestyle="--", alpha=0.8,
                    label=f"s-routed mean={np.mean(s_ious):.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(out_dir / "iou_distribution.png"))
        plt.close(fig)

    # ── 2. Proxy value vs IoU scatter ──
    c_scores = [float(r.get("C_score", 0)) for r in all_stats]
    ious_all = [float(r["iou_n_s"]) for r in all_stats]
    choices = [r["choice"] for r in all_stats]

    if any(c != 0 for c in c_scores):
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["green" if ch == "n" else "red" for ch in choices]
        ax.scatter(c_scores, ious_all, c=colors, alpha=0.5, s=20, edgecolors="none")
        ax.set_xlabel("C-score (proxy value)")
        ax.set_ylabel("IoU(n-model, s-model)")
        ax.set_title("Proxy Signal vs Detection Agreement")

        # Correlation
        valid = [(c, iou) for c, iou in zip(c_scores, ious_all) if c != 0]
        if len(valid) > 5:
            try:
                from scipy import stats as sp_stats
                r, p = sp_stats.pearsonr([v[0] for v in valid], [v[1] for v in valid])
            except ImportError:
                x = np.array([v[0] for v in valid], dtype=float)
                y = np.array([v[1] for v in valid], dtype=float)
                r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0
                p = float("nan")
            ax.annotate(f"Pearson r={r:.3f} (p={p:.2e})", xy=(0.02, 0.02),
                        xycoords="axes fraction", fontsize=10,
                        bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='n-routed'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='s-routed'),
        ]
        ax.legend(handles=legend_elements)
        fig.tight_layout()
        fig.savefig(str(out_dir / "proxy_vs_iou_scatter.png"))
        plt.close(fig)

    # ── 3. Detection count comparison ──
    n_rows = [r for r in all_stats if r["choice"] == "n"]
    s_rows = [r for r in all_stats if r["choice"] == "s"]

    if n_rows and s_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["n-routed frames", "s-routed frames"]
        n_model_means = [
            np.mean([int(r["n_detections"]) for r in n_rows]),
            np.mean([int(r["n_detections"]) for r in s_rows]),
        ]
        s_model_means = [
            np.mean([int(r["s_detections"]) for r in n_rows]),
            np.mean([int(r["s_detections"]) for r in s_rows]),
        ]

        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width / 2, n_model_means, width, color="green", alpha=0.8, label="n-model dets")
        ax.bar(x + width / 2, s_model_means, width, color="red", alpha=0.8, label="s-model dets")
        ax.set_ylabel("Mean detections per frame")
        ax.set_title("Detection Counts: n-model vs s-model")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        # Annotate values
        for i, (nv, sv) in enumerate(zip(n_model_means, s_model_means)):
            ax.text(i - width / 2, nv + 0.1, f"{nv:.1f}", ha="center", fontsize=9)
            ax.text(i + width / 2, sv + 0.1, f"{sv:.1f}", ha="center", fontsize=9)

        fig.tight_layout()
        fig.savefig(str(out_dir / "detection_count_comparison.png"))
        plt.close(fig)

    # ── 4. Routing accuracy per policy ──
    if all_summaries:
        fig, ax = plt.subplots(figsize=(10, 5))
        policies = [s["policy"] for s in all_summaries]
        n_acc = [s.get("n_routing_accuracy", 0) or 0 for s in all_summaries]
        s_acc = [s.get("s_routing_accuracy", 0) or 0 for s in all_summaries]

        x = np.arange(len(policies))
        width = 0.35
        ax.bar(x - width / 2, [a * 100 for a in n_acc], width, color="green", alpha=0.8,
               label="n-routed correct (IoU≥0.5)")
        ax.bar(x + width / 2, [a * 100 for a in s_acc], width, color="red", alpha=0.8,
               label="s-routed correct (IoU<0.5)")
        ax.set_ylabel("Routing accuracy (%)")
        ax.set_title("Proxy Routing Accuracy by Policy")
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=30, ha="right")
        ax.legend()
        ax.set_ylim(0, 105)

        fig.tight_layout()
        fig.savefig(str(out_dir / "proxy_routing_accuracy.png"))
        plt.close(fig)

    # Save grand summary
    with open(out_dir / "grand_validation_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n  Grand validation plots saved to: {out_dir}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Frame-level proxy routing validation")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Video name substrings to filter (default: all available)")
    parser.add_argument("--policies", nargs="*", default=None,
                        help="Policies to validate (default: all proxy policies)")
    parser.add_argument("--overnight-dir", default=None,
                        help="Path to overnight_full results (default: auto-detect)")
    args = parser.parse_args()

    device = args.device
    policies = args.policies or PROXY_POLICIES

    # Locate overnight results
    overnight_dir = Path(args.overnight_dir) if args.overnight_dir else \
        Path(__file__).resolve().parent.parent / "research_results" / "overnight_full"

    if not overnight_dir.exists():
        print(f"ERROR: Overnight results not found at {overnight_dir}")
        print("Run 09_overnight_full_pipeline.py first.")
        sys.exit(1)

    # Find available videos
    available_videos = []
    for material, vids in VIDEOS.items():
        for vp in vids:
            vname = Path(vp).stem
            # Check if overnight results exist for this video
            vid_dir = overnight_dir / vname
            if vid_dir.exists():
                if args.videos is None or any(v in vname for v in args.videos):
                    available_videos.append((vp, vname, material))

    if not available_videos:
        print("ERROR: No matching videos with overnight results found.")
        print(f"  Overnight dir: {overnight_dir}")
        print(f"  Video filter: {args.videos}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DMS-Raptor: Frame-Level Proxy Routing Validation")
    print("=" * 70)
    print(f"  Videos ({len(available_videos)}): {[v[1] for v in available_videos]}")
    print(f"  Policies ({len(policies)}): {policies}")
    print(f"  Device: {device}")
    print(f"  Samples per routing: {N_SAMPLES_ZERO_DET} zero-det + {N_SAMPLES_WITH_DET} with-det = {N_TOTAL}")
    print(f"  Output: {OUTPUT_ROOT}")
    print("=" * 70)

    all_summaries = []
    all_stats = []
    t0 = time.time()

    for video_path, vname, material in available_videos:
        print(f"\n{'─' * 70}")
        print(f"  VIDEO: {vname} ({material})")
        print(f"{'─' * 70}")

        # Load models
        print(f"  Loading {material} models...")
        model_n = YOLO(MODELS[material]["y8n"])
        model_s = YOLO(MODELS[material]["y8s"])

        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        run_model_single(model_n, dummy, IMGSZ, device, 0.001, IOU_NMS)
        run_model_single(model_s, dummy, IMGSZ, device, 0.001, IOU_NMS)
        print("  Models loaded and warmed up.")

        for policy in policies:
            policy_dir = overnight_dir / vname / policy
            trace_csv = policy_dir / "per_frame_trace.csv"

            if not trace_csv.exists():
                print(f"\n    [{policy}] SKIP — no trace CSV")
                continue

            out_dir = OUTPUT_ROOT / vname / policy
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n    [{policy}] Validating...")
            summary = validate_policy(video_path, material, policy, trace_csv,
                                      model_n, model_s, device, out_dir)

            if summary:
                all_summaries.append(summary)
                # Load stats for grand plots
                stats_csv = out_dir / "validation_stats.csv"
                if stats_csv.exists():
                    with open(stats_csv, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row["policy"] = policy
                            row["video"] = vname
                            all_stats.append(row)

                n_iou = summary.get("n_mean_iou")
                s_iou = summary.get("s_mean_iou")
                print(f"      n-routed: {summary['n_routed_count']} frames, "
                      f"mean IoU={n_iou:.3f}" if n_iou else "      n-routed: 0 frames")
                print(f"      s-routed: {summary['s_routed_count']} frames, "
                      f"mean IoU={s_iou:.3f}" if s_iou else "      s-routed: 0 frames")

    # Grand summary plots
    print(f"\n{'=' * 70}")
    print("  Generating grand validation summary...")
    print(f"{'=' * 70}")
    plot_grand_summary(all_summaries, all_stats, OUTPUT_ROOT)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
