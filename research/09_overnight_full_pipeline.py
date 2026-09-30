"""
Phase 9: Full Overnight Pipeline — All 8 Policies × All 12 Videos

Comprehensive experiment runner that uses StreamingEngine for correct timing
and produces rich per-video outputs matching the old dms_overnight_pipeline.py
structure, but with ALL 8 policies including conf_ema, niqe_switch, multi_proxy.

Per video × policy outputs:
  ├── per_frame_trace.csv        (every metric per frame)
  ├── annotated_{video}_{policy}.mp4  (with thick green/red model border)
  ├── exemplar_easy.png          (lowest C-score frame = easiest scene)
  ├── exemplar_hard.png          (highest C-score frame = hardest scene)
  ├── summary.json               (aggregate RunSummary metrics)
  └── T_total_timeseries.png     (T_total over time, colored by model choice)

Per video cross-policy outputs:
  ├── timing_breakdown_bar.png   (stacked bar: T_scene + T_ctrl + T_infer)
  ├── policy_comparison.csv      (aggregate metrics for all policies)
  └── choice_timeline.png        (binary heatmap strip per policy)

Grand summary outputs:
  ├── grand_summary.csv          (all videos × all policies)
  └── grand_summary.json

Usage:
    python research/09_overnight_full_pipeline.py
    python research/09_overnight_full_pipeline.py --policies n_only conf_ema combined_hyst
    python research/09_overnight_full_pipeline.py --videos 633-01 porce --max-frames 500
    python research/09_overnight_full_pipeline.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InferenceParams, RunConfig, RunSummary, POLICIES
from core.engine import StreamingEngine
from core.research_paths import MODELS, VIDEOS

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "research_results" / "overnight_full"

# Plot style
POLICY_COLORS = {
    "n_only": "#4CAF50",
    "s_only": "#E94560",
    "entropy_only": "#FFA726",
    "combined": "#42A5F5",
    "combined_hyst": "#AB47BC",
    "conf_ema": "#26C6DA",
    "niqe_switch": "#EC407A",
    "multi_proxy": "#66BB6A",
}

POLICY_SHORT = {
    "n_only": "n-only",
    "s_only": "s-only",
    "entropy_only": "entropy",
    "combined": "combined",
    "combined_hyst": "comb-hyst",
    "conf_ema": "conf-ema",
    "niqe_switch": "niqe-sw",
    "multi_proxy": "multi-px",
}


# ═══════════════════════════════════════════════════════════════════════
# Per-frame CSV export
# ═══════════════════════════════════════════════════════════════════════
def save_per_frame_csv(summary: RunSummary, out_path: Path):
    """Write per-frame trace data to CSV."""
    n = summary.total_frames
    if n == 0:
        return

    fields = [
        "frame_idx", "T_total_ms", "T_scene_ms", "T_ctrl_ms",
        "T_infer_n_ms", "T_infer_s_ms", "choice", "C_score",
        "L", "H", "c_low", "c_high", "dwell", "num_detections",
        "mean_conf", "conf_drop", "niqe_score", "zero_det_gated",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "frame_idx": summary.frame_indices[i] if i < len(summary.frame_indices) else i,
                "T_total_ms": f"{summary.T_total_trace[i]:.2f}" if i < len(summary.T_total_trace) else "",
                "T_scene_ms": f"{summary.T_scene_trace[i]:.2f}" if i < len(summary.T_scene_trace) else "",
                "T_ctrl_ms": f"{summary.T_ctrl_trace[i]:.2f}" if i < len(summary.T_ctrl_trace) else "",
                "T_infer_n_ms": f"{summary.T_infer_n_trace[i]:.2f}" if i < len(summary.T_infer_n_trace) else "",
                "T_infer_s_ms": f"{summary.T_infer_s_trace[i]:.2f}" if i < len(summary.T_infer_s_trace) else "",
                "choice": summary.choice_trace[i] if i < len(summary.choice_trace) else "",
                "C_score": f"{summary.C_trace[i]:.4f}" if i < len(summary.C_trace) else "",
                "L": f"{summary.L_trace[i]:.2f}" if i < len(summary.L_trace) else "",
                "H": f"{summary.H_trace[i]:.4f}" if i < len(summary.H_trace) else "",
                "c_low": f"{summary.c_low_trace[i]:.4f}" if i < len(summary.c_low_trace) else "",
                "c_high": f"{summary.c_high_trace[i]:.4f}" if i < len(summary.c_high_trace) else "",
                "dwell": summary.dwell_trace[i] if i < len(summary.dwell_trace) else "",
                "num_detections": summary.num_detections_trace[i] if i < len(summary.num_detections_trace) else "",
                "mean_conf": f"{summary.mean_conf_trace[i]:.4f}" if i < len(summary.mean_conf_trace) else "",
                "conf_drop": f"{summary.conf_drop_trace[i]:.4f}" if i < len(summary.conf_drop_trace) else "",
                "niqe_score": f"{summary.niqe_trace[i]:.4f}" if i < len(summary.niqe_trace) else "",
                "zero_det_gated": summary.zero_det_gated_trace[i] if i < len(summary.zero_det_gated_trace) else "",
            })


# ═══════════════════════════════════════════════════════════════════════
# Exemplar frame extraction
# ═══════════════════════════════════════════════════════════════════════
def extract_exemplar_frames(video_path: str, summary: RunSummary, out_dir: Path):
    """Extract low-routing-risk (lowest C) and high-routing-risk (highest C) frames as PNGs.

    Uses thesis-safe terminology: frames are NOT labeled as 'easy' or 'hard'
    because image proxies do not establish true scene complexity.
    """
    c_trace = np.array(summary.C_trace, dtype=np.float32)
    if len(c_trace) == 0:
        return

    low_risk_idx = int(np.argmin(c_trace))
    high_risk_idx = int(np.argmax(c_trace))

    # Map back to raw frame indices
    frame_indices = summary.frame_indices
    low_risk_raw = frame_indices[low_risk_idx] if low_risk_idx < len(frame_indices) else low_risk_idx
    high_risk_raw = frame_indices[high_risk_idx] if high_risk_idx < len(frame_indices) else high_risk_idx

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    for label, raw_idx, kept_idx in [("low_routing_risk", low_risk_raw, low_risk_idx),
                                      ("high_routing_risk", high_risk_raw, high_risk_idx)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ret, frame = cap.read()
        if ret:
            c_val = c_trace[kept_idx]
            choice = summary.choice_trace[kept_idx] if kept_idx < len(summary.choice_trace) else "?"
            dets = summary.num_detections_trace[kept_idx] if kept_idx < len(summary.num_detections_trace) else 0

            # Add annotation text
            h, w = frame.shape[:2]
            info = f"Frame {raw_idx} | C={c_val:.3f} | Model={'n' if choice == 'n' else 's'} | Dets={dets}"
            cv2.putText(frame, info, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, info, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            out_path = out_dir / f"exemplar_{label}.png"
            cv2.imwrite(str(out_path), frame)

    cap.release()


# ═══════════════════════════════════════════════════════════════════════
# Per-policy plots
# ═══════════════════════════════════════════════════════════════════════
def plot_T_total_timeseries(summary: RunSummary, out_path: Path):
    """T_total over time, colored green (n) / red (s) by model choice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = summary.total_frames
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    fig.patch.set_facecolor("#0A0E1A")
    ax.set_facecolor("#0A0E1A")

    t_total = np.array(summary.T_total_trace[:n])
    frames = np.arange(n)
    choices = summary.choice_trace[:n]

    # Separate into n and s segments
    n_mask = np.array([c == "n" for c in choices])
    s_mask = ~n_mask

    ax.scatter(frames[n_mask], t_total[n_mask], c="#4CAF50", s=1, alpha=0.6, label="YOLOv8n")
    ax.scatter(frames[s_mask], t_total[s_mask], c="#E94560", s=1, alpha=0.6, label="YOLOv8s")

    # Rolling average
    if n > 50:
        kernel = np.ones(50) / 50
        rolling = np.convolve(t_total, kernel, mode="same")
        ax.plot(frames, rolling, color="#FFD700", linewidth=1.2, alpha=0.8, label="50-frame avg")

    # P95 line
    p95 = np.percentile(t_total, 95)
    ax.axhline(p95, color="#FF6B6B", linestyle="--", alpha=0.5, label=f"P95={p95:.0f}ms")

    ax.set_xlabel("Frame", color="white", fontsize=10)
    ax.set_ylabel("T_total (ms)", color="white", fontsize=10)
    ax.set_title(f"Latency: {summary.policy} — {summary.video}",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="white")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, facecolor="#0A0E1A", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Cross-policy comparison plots (per video)
# ═══════════════════════════════════════════════════════════════════════
def plot_timing_breakdown(summaries: dict, video_name: str, out_path: Path):
    """Stacked bar chart: T_scene + T_ctrl + T_infer per policy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = list(summaries.keys())
    if not policies:
        return

    t_scene = [summaries[p].T_scene_ms_mean for p in policies]
    t_ctrl = [summaries[p].T_ctrl_ms_mean for p in policies]
    t_infer = [
        summaries[p].T_infer_n_ms_mean + summaries[p].T_infer_s_ms_mean
        for p in policies
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#0A0E1A")
    ax.set_facecolor("#0A0E1A")

    x = np.arange(len(policies))
    labels = [POLICY_SHORT.get(p, p) for p in policies]

    bar_scene = ax.bar(x, t_scene, 0.6, label="T_scene", color="#26C6DA")
    bar_ctrl = ax.bar(x, t_ctrl, 0.6, bottom=t_scene, label="T_ctrl", color="#FFA726")
    bar_infer = ax.bar(x, t_infer, 0.6,
                       bottom=[s + c for s, c in zip(t_scene, t_ctrl)],
                       label="T_infer", color="#AB47BC")

    # Total labels on top
    for i, p in enumerate(policies):
        total = t_scene[i] + t_ctrl[i] + t_infer[i]
        ax.text(i, total + 3, f"{total:.0f}", ha="center", color="white", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms)", color="white", fontsize=10)
    ax.set_title(f"Timing Breakdown — {video_name}", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, facecolor="#0A0E1A", bbox_inches="tight")
    plt.close(fig)


def plot_metric_subplots(summaries: dict, metric_attr: str, ylabel: str,
                         video_name: str, out_path: Path):
    """Colored scatter subplots (green=n, red=s) for a timing metric, one panel per policy.

    Replicates the old pipeline's plot_T_scene_subplots / plot_T_ctrl_subplots /
    plot_T_total_subplots output that was useful for visual per-frame comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    policies = [p for p in summaries if summaries[p].total_frames > 0]
    if not policies:
        return

    cols = 2 if len(policies) > 1 else 1
    rows = math.ceil(len(policies) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), dpi=150,
                             squeeze=False)
    fig.patch.set_facecolor("#0A0E1A")

    for idx, p in enumerate(policies):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor("#0A0E1A")

        s = summaries[p]
        n = s.total_frames
        trace = np.array(getattr(s, metric_attr, [])[:n], dtype=np.float32)
        frames = np.arange(len(trace))
        choices = s.choice_trace[:len(trace)]

        n_mask = np.array([c == "n" for c in choices])
        s_mask = ~n_mask

        if np.any(n_mask):
            ax.scatter(frames[n_mask], trace[n_mask], s=4, c="#4CAF50",
                       alpha=0.6, label="YOLOv8n", rasterized=True)
        if np.any(s_mask):
            ax.scatter(frames[s_mask], trace[s_mask], s=4, c="#E94560",
                       alpha=0.6, label="YOLOv8s", rasterized=True)

        ax.set_title(f"{POLICY_SHORT.get(p, p)}: {ylabel}",
                     color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Frame", color="white", fontsize=8)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, alpha=0.15, color="white")
        ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
                  edgecolor="#333", labelcolor="white")

    # Hide unused subplots
    for idx in range(len(policies), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"{video_name} — {ylabel} per policy (green=n, red=s)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, facecolor="#0A0E1A", bbox_inches="tight")
    plt.close(fig)


def plot_choice_timeline(summaries: dict, video_name: str, out_path: Path):
    """Binary heatmap strip: one row per policy, green=n, red=s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    policies = [p for p in summaries if summaries[p].total_frames > 0]
    if not policies:
        return

    max_frames = max(summaries[p].total_frames for p in policies)

    fig, ax = plt.subplots(figsize=(14, max(3, len(policies) * 0.5 + 1)), dpi=150)
    fig.patch.set_facecolor("#0A0E1A")
    ax.set_facecolor("#0A0E1A")

    cmap = ListedColormap(["#4CAF50", "#E94560"])  # 0=n(green), 1=s(red)

    data = []
    labels = []
    for p in policies:
        s = summaries[p]
        row = [0 if c == "n" else 1 for c in s.choice_trace]
        # Pad to max_frames
        if len(row) < max_frames:
            row.extend([0] * (max_frames - len(row)))
        data.append(row[:max_frames])
        sw = s.sw_per_100
        labels.append(f"{POLICY_SHORT.get(p, p)} (sw={sw:.1f}/100)")

    data_arr = np.array(data)
    ax.imshow(data_arr, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel("Frame", color="white", fontsize=10)
    ax.set_title(f"Model Choice Timeline — {video_name}\n(green=YOLOv8n, red=YOLOv8s)",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, facecolor="#0A0E1A", bbox_inches="tight")
    plt.close(fig)


def save_policy_comparison_csv(summaries: dict, out_path: Path):
    """Write aggregate metrics for all policies to a comparison CSV."""
    if not summaries:
        return

    fields = [
        "policy", "total_frames", "T_total_mean", "T_total_p95", "T_total_p99",
        "T_scene_mean", "T_ctrl_mean",
        "T_infer_n_mean_uncond", "T_infer_s_mean_uncond",
        "T_infer_n_mean_cond", "T_infer_s_mean_cond",
        "slow_pct", "switches", "sw_per_100", "zero_det_gate_activations",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for policy, s in summaries.items():
            writer.writerow({
                "policy": policy,
                "total_frames": s.total_frames,
                "T_total_mean": f"{s.T_total_ms_mean:.2f}",
                "T_total_p95": f"{s.T_total_ms_p95:.2f}",
                "T_total_p99": f"{s.T_total_ms_p99:.2f}",
                "T_scene_mean": f"{s.T_scene_ms_mean:.2f}",
                "T_ctrl_mean": f"{s.T_ctrl_ms_mean:.2f}",
                "T_infer_n_mean_uncond": f"{s.T_infer_n_ms_mean:.2f}",
                "T_infer_s_mean_uncond": f"{s.T_infer_s_ms_mean:.2f}",
                "T_infer_n_mean_cond": f"{s.T_infer_n_ms_cond_mean:.2f}",
                "T_infer_s_mean_cond": f"{s.T_infer_s_ms_cond_mean:.2f}",
                "slow_pct": f"{s.slow_pct:.2f}",
                "switches": s.switches,
                "sw_per_100": f"{s.sw_per_100:.2f}",
                "zero_det_gate_activations": getattr(s, 'zero_det_gate_activations', 0),
            })


# ═══════════════════════════════════════════════════════════════════════
# Grand summary plots
# ═══════════════════════════════════════════════════════════════════════
def plot_grand_summary(all_results: dict, out_dir: Path):
    """Generate grand cross-video summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect per-policy averages across all videos
    policy_metrics = {}
    for video_name, summaries in all_results.items():
        for policy, s in summaries.items():
            if policy not in policy_metrics:
                policy_metrics[policy] = {
                    "T_total": [], "T_p95": [], "slow_pct": [],
                    "sw_per_100": [], "T_scene": [], "T_ctrl": [],
                }
            policy_metrics[policy]["T_total"].append(s.T_total_ms_mean)
            policy_metrics[policy]["T_p95"].append(s.T_total_ms_p95)
            policy_metrics[policy]["slow_pct"].append(s.slow_pct)
            policy_metrics[policy]["sw_per_100"].append(s.sw_per_100)
            policy_metrics[policy]["T_scene"].append(s.T_scene_ms_mean)
            policy_metrics[policy]["T_ctrl"].append(s.T_ctrl_ms_mean)

    if not policy_metrics:
        return

    policies = sorted(policy_metrics.keys(), key=lambda p: POLICIES.index(p) if p in POLICIES else 99)

    # --- Plot 1: Mean T_total bar chart with error bars ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#0A0E1A")

    for ax in axes.flat:
        ax.set_facecolor("#0A0E1A")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    x = np.arange(len(policies))
    labels = [POLICY_SHORT.get(p, p) for p in policies]
    colors = [POLICY_COLORS.get(p, "#888") for p in policies]

    # (a) Mean T_total
    ax = axes[0, 0]
    means = [np.mean(policy_metrics[p]["T_total"]) for p in policies]
    stds = [np.std(policy_metrics[p]["T_total"]) for p in policies]
    ax.bar(x, means, 0.6, yerr=stds, color=colors, capsize=3, error_kw={"ecolor": "white", "alpha": 0.5})
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 3, f"{m:.0f}", ha="center", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("T_total (ms)", color="white")
    ax.set_title("(a) Mean Latency ± std", color="white", fontweight="bold")

    # (b) Slow %
    ax = axes[0, 1]
    slow_means = [np.mean(policy_metrics[p]["slow_pct"]) for p in policies]
    ax.bar(x, slow_means, 0.6, color=colors)
    for i, m in enumerate(slow_means):
        ax.text(i, m + 1, f"{m:.1f}%", ha="center", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("s-model usage (%)", color="white")
    ax.set_title("(b) YOLOv8s Usage", color="white", fontweight="bold")

    # (c) Switching rate
    ax = axes[1, 0]
    sw_means = [np.mean(policy_metrics[p]["sw_per_100"]) for p in policies]
    ax.bar(x, sw_means, 0.6, color=colors)
    for i, m in enumerate(sw_means):
        ax.text(i, m + 0.1, f"{m:.1f}", ha="center", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Switches / 100 frames", color="white")
    ax.set_title("(c) Switching Rate", color="white", fontweight="bold")

    # (d) Pareto: T_total vs slow_pct
    ax = axes[1, 1]
    for i, p in enumerate(policies):
        ax.scatter(slow_means[i], means[i], c=colors[i], s=80, zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(POLICY_SHORT.get(p, p), (slow_means[i], means[i]),
                    textcoords="offset points", xytext=(5, 5), color="white", fontsize=8)
    ax.set_xlabel("s-model usage (%)", color="white")
    ax.set_ylabel("Mean T_total (ms)", color="white")
    ax.set_title("(d) Latency vs Accuracy Trade-off", color="white", fontweight="bold")

    fig.suptitle("Grand Summary -- All Videos x All Policies",
                 color="white", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3, bottom=0.1)
    fig.savefig(str(out_dir / "grand_summary_panel.png"), dpi=150,
                facecolor="#0A0E1A", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════
def run_single(video_path: str, material: str, policy: str,
               model_n, model_s, inf: InferenceParams,
               out_dir: Path, save_video: bool = True,
               zero_det_gate: bool = False) -> RunSummary | None:
    """Run one video x one policy through StreamingEngine."""
    video_name = Path(video_path).stem
    policy_dir = out_dir / policy
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    summary_file = policy_dir / "summary.json"
    if summary_file.exists():
        print(f"      SKIP (summary.json exists)")
        try:
            with open(summary_file, "r") as f:
                return RunSummary.from_dict(json.load(f))
        except Exception:
            pass  # Re-run if corrupt

    annotated_path = policy_dir / f"annotated_{video_name}_{policy}.mp4"

    cfg = RunConfig(
        name=f"{video_name}_{policy}",
        policy=policy,
        save_annotated_video=save_video,
        output_video_path=str(annotated_path) if save_video else "",
        zero_det_gate=zero_det_gate,
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
    total_est = engine.total_frames or 0

    for fr in engine.run():
        frame_count += 1
        if frame_count % 300 == 0:
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            pct = frame_count / total_est * 100 if total_est > 0 else 0
            print(f"      Frame {frame_count}/{total_est} "
                  f"({pct:.0f}%) — {fps:.1f} fps", end="\r")

    elapsed = time.time() - t0
    summary = engine.get_summary()
    if summary is None:
        return None

    fps_actual = frame_count / elapsed if elapsed > 0 else 0
    print(f"      Done: {frame_count} frames in {elapsed:.1f}s ({fps_actual:.1f} fps)    ")

    # --- Save outputs ---
    # 1. Summary JSON
    with open(summary_file, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    # 2. Per-frame CSV
    save_per_frame_csv(summary, policy_dir / "per_frame_trace.csv")

    # 3. Exemplar frames
    extract_exemplar_frames(video_path, summary, policy_dir)

    # 4. T_total timeseries plot
    plot_T_total_timeseries(summary, policy_dir / "T_total_timeseries.png")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Full overnight pipeline: all 8 policies × all 12 videos"
    )
    parser.add_argument("--policies", nargs="+", default=None,
                        help=f"Policies to run (default: all 8). Valid: {POLICIES}")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Filter videos by substring (e.g., 633-01 porce)")
    parser.add_argument("--materials", nargs="+", default=["glass", "porcelain"])
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--annotate-videos", nargs="*", default=None,
                        help="Only save annotated MP4s for these videos (substring match). "
                             "All videos still get CSVs/JSONs/plots. "
                             "e.g. --annotate-videos 633-01 porce glass_ins  "
                             "Omit flag entirely = annotate ALL videos.")
    parser.add_argument("--annotate-policies", nargs="*", default=None,
                        help="Only save annotated MP4s for these policies. "
                             "All policies still get CSVs/JSONs/plots. "
                             "e.g. --annotate-policies conf_ema niqe_switch multi_proxy  "
                             "Omit flag entirely = annotate ALL policies.")
    parser.add_argument("--zero-det-gate", action="store_true", default=False,
                        help="Enable zero-detection gate: force n-model when recent "
                             "frames have 0 detections (saves s-model cost on empty frames)")
    args = parser.parse_args()

    policies = args.policies or POLICIES
    for p in policies:
        if p not in POLICIES:
            print(f"ERROR: Unknown policy '{p}'. Valid: {POLICIES}")
            return

    output_root = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    inf = InferenceParams(imgsz=640, device=args.device, max_frames=args.max_frames)

    # Build job list
    jobs = []
    for mat in args.materials:
        for vpath in VIDEOS.get(mat, []):
            vname = Path(vpath).stem
            if args.videos and not any(f in vname for f in args.videos):
                continue
            if not os.path.exists(vpath):
                print(f"  WARNING: Video not found: {vpath}")
                continue
            jobs.append((vpath, mat, vname))

    total_jobs = len(jobs) * len(policies)

    # Determine which (video, policy) combos get annotated MP4s.
    # Both filters are AND-ed: a job gets an MP4 only if BOTH video AND policy pass.
    # --annotate-videos  not passed (None) => all videos pass
    # --annotate-policies not passed (None) => all policies pass
    vid_filter = args.annotate_videos    # None = all, list = selective
    pol_filter = args.annotate_policies  # None = all, list = selective

    def should_annotate(vname: str, policy: str) -> bool:
        vid_ok = (vid_filter is None) or any(f in vname for f in vid_filter)
        pol_ok = (pol_filter is None) or (policy in pol_filter)
        return vid_ok and pol_ok

    # Summary for header
    n_mp4_jobs = sum(
        1 for j in jobs for p in policies if should_annotate(j[2], p)
    )
    n_csv_only = total_jobs - n_mp4_jobs

    print("=" * 70)
    print("  DMS-Raptor: Full Overnight Pipeline v2")
    print("=" * 70)
    print(f"  Policies ({len(policies)}): {policies}")
    print(f"  Videos ({len(jobs)}): {[j[2] for j in jobs]}")
    if pol_filter is not None:
        print(f"  Annotated MP4 policies: {pol_filter}")
    if vid_filter is not None:
        print(f"  Annotated MP4 videos: {vid_filter}")
    print(f"  Jobs with MP4: {n_mp4_jobs} | CSV-only: {n_csv_only}")
    print(f"  Zero-det gate: {'ON' if args.zero_det_gate else 'OFF'}")
    print(f"  Max frames: {'all' if args.max_frames == 0 else args.max_frames}")
    print(f"  Device: {args.device}")
    print(f"  Output: {output_root}")
    print(f"  Total jobs: {total_jobs}")
    print("=" * 70)

    pipeline_t0 = time.time()
    all_results = {}  # {video_name: {policy: RunSummary}}
    grand_rows = []
    job_idx = 0

    for vpath, mat, vname in jobs:
        print(f"\n{'-' * 70}")
        print(f"  VIDEO: {vname} ({mat})")
        print(f"{'-' * 70}")

        # Load models for this material
        model_paths = MODELS[mat]
        from ultralytics import YOLO
        print(f"  Loading {mat} models...")
        model_n = YOLO(model_paths["y8n"])
        model_s = YOLO(model_paths["y8s"])

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model_n(dummy, imgsz=640, verbose=False)
        model_s(dummy, imgsz=640, verbose=False)
        print(f"  Models loaded and warmed up.")

        vid_dir = output_root / vname
        vid_dir.mkdir(parents=True, exist_ok=True)
        video_summaries = {}

        for policy in policies:
            job_idx += 1
            do_annotate = should_annotate(vname, policy)
            tag = "" if do_annotate else " [no MP4]"
            print(f"\n    [{job_idx}/{total_jobs}] {policy}{tag}")

            try:
                summary = run_single(
                    vpath, mat, policy, model_n, model_s, inf, vid_dir,
                    save_video=do_annotate,
                    zero_det_gate=args.zero_det_gate,
                )
                if summary:
                    video_summaries[policy] = summary
                    grand_rows.append({
                        "video": vname,
                        "material": mat,
                        "policy": policy,
                        "total_frames": summary.total_frames,
                        "T_total_mean": summary.T_total_ms_mean,
                        "T_total_p95": summary.T_total_ms_p95,
                        "T_scene_mean": summary.T_scene_ms_mean,
                        "T_ctrl_mean": summary.T_ctrl_ms_mean,
                        "slow_pct": summary.slow_pct,
                        "switches": summary.switches,
                        "sw_per_100": summary.sw_per_100,
                    })
            except Exception as e:
                print(f"      ERROR: {e}")
                traceback.print_exc()

        all_results[vname] = video_summaries

        # Per-video cross-policy outputs
        if len(video_summaries) >= 2:
            print(f"\n    Generating cross-policy plots for {vname}...")
            try:
                plot_timing_breakdown(video_summaries, vname,
                                     vid_dir / "timing_breakdown_bar.png")
                plot_choice_timeline(video_summaries, vname,
                                    vid_dir / "choice_timeline.png")
                save_policy_comparison_csv(video_summaries,
                                          vid_dir / "policy_comparison.csv")
                # Colored scatter subplots (ported from old pipeline)
                plot_metric_subplots(video_summaries, "T_scene_trace",
                                     "T_scene (ms)", vname,
                                     vid_dir / "plot_T_scene_subplots.png")
                plot_metric_subplots(video_summaries, "T_ctrl_trace",
                                     "T_ctrl (ms)", vname,
                                     vid_dir / "plot_T_ctrl_subplots.png")
                plot_metric_subplots(video_summaries, "T_total_trace",
                                     "T_total (ms)", vname,
                                     vid_dir / "plot_T_total_subplots.png")
            except Exception as e:
                print(f"      Plot error: {e}")

    # Grand summary
    print(f"\n{'=' * 70}")
    print("  Generating grand summary...")
    print(f"{'=' * 70}")

    # Grand CSV
    if grand_rows:
        grand_csv = output_root / "grand_summary.csv"
        with open(grand_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=grand_rows[0].keys())
            writer.writeheader()
            writer.writerows(grand_rows)
        print(f"  Saved: {grand_csv}")

    # Grand JSON
    grand_json_data = {}
    for vname, summaries in all_results.items():
        grand_json_data[vname] = {}
        for policy, s in summaries.items():
            grand_json_data[vname][policy] = {
                "total_frames": s.total_frames,
                "T_total_mean": round(s.T_total_ms_mean, 2),
                "T_total_p95": round(s.T_total_ms_p95, 2),
                "T_total_p99": round(s.T_total_ms_p99, 2),
                "T_scene_mean": round(s.T_scene_ms_mean, 2),
                "T_ctrl_mean": round(s.T_ctrl_ms_mean, 2),
                "T_infer_n_mean_uncond": round(s.T_infer_n_ms_mean, 2),
                "T_infer_s_mean_uncond": round(s.T_infer_s_ms_mean, 2),
                "T_infer_n_mean_cond": round(getattr(s, 'T_infer_n_ms_cond_mean', 0.0), 2),
                "T_infer_s_mean_cond": round(getattr(s, 'T_infer_s_ms_cond_mean', 0.0), 2),
                "slow_pct": round(s.slow_pct, 2),
                "switches": s.switches,
                "sw_per_100": round(s.sw_per_100, 2),
                "zero_det_gate_activations": getattr(s, 'zero_det_gate_activations', 0),
            }
    grand_json = output_root / "grand_summary.json"
    with open(grand_json, "w") as f:
        json.dump(grand_json_data, f, indent=2)
    print(f"  Saved: {grand_json}")

    # Grand plots
    if len(all_results) >= 1:
        try:
            plot_grand_summary(all_results, output_root)
            print(f"  Saved: grand_summary_panel.png")
        except Exception as e:
            print(f"  Grand plot error: {e}")

    # Final stats
    elapsed_total = time.time() - pipeline_t0
    hours = int(elapsed_total // 3600)
    mins = int((elapsed_total % 3600) // 60)
    secs = int(elapsed_total % 60)

    done = sum(
        1 for vname in all_results
        for p in all_results[vname]
    )

    # Count output files
    file_counts = {"mp4": 0, "csv": 0, "json": 0, "png": 0}
    total_size = 0
    if output_root.exists():
        for f in output_root.rglob("*"):
            if f.is_file():
                ext = f.suffix.lower().lstrip(".")
                if ext in file_counts:
                    file_counts[ext] += 1
                total_size += f.stat().st_size

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Jobs completed: {done}/{total_jobs}")
    print(f"  Total time: {hours}h {mins}m {secs}s")
    print(f"  Output directory: {output_root}")
    print(f"  Files: {file_counts['mp4']} MP4, {file_counts['csv']} CSV, "
          f"{file_counts['json']} JSON, {file_counts['png']} PNG")
    print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
