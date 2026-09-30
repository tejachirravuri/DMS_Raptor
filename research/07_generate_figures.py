"""
Phase 7: Generate all publication-quality figures from research results.

Reads JSON outputs from phases 01–06 and produces matplotlib figures saved as
high-resolution PNGs, organized into subdirectories by category.

Outputs to: research_results/figures/
  ├── 01_baselines/         — model baseline comparison
  ├── 02_detection/         — detection quality per-video and aggregate
  ├── 03_timing/            — repeated-trial timing analysis
  ├── 04_ablation/          — ablation study decomposition
  ├── 05_generalization/    — cross-video and cross-material transfer
  ├── 06_cscore/            — C-score failure analysis
  └── 07_summary/           — aggregate comparison / Pareto / paper-ready

Usage:
    python research/07_generate_figures.py [--dpi 300] [--light]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "research_results"
OUT = RESULTS / "figures"

POLICIES_ORDERED = [
    "n_only", "s_only", "entropy_only", "combined",
    "combined_hyst", "conf_ema", "niqe_switch", "multi_proxy",
]

POLICY_LABELS = {
    "n_only": "n_only", "s_only": "s_only",
    "entropy_only": "entropy", "combined": "combined",
    "combined_hyst": "comb_hyst", "conf_ema": "conf_ema",
    "niqe_switch": "niqe_sw", "multi_proxy": "multi_prx",
}

POLICY_COLORS = {
    "n_only":        "#4caf50",
    "s_only":        "#e94560",
    "entropy_only":  "#9c27b0",
    "combined":      "#ff9800",
    "combined_hyst": "#00bcd4",
    "conf_ema":      "#ffeb3b",
    "niqe_switch":   "#2196f3",
    "multi_proxy":   "#e91e63",
}

# Theme
DARK_BG = "#181c24"
DARK_FG = "#cccccc"
DARK_GRID = "#2a2e38"
LIGHT_BG = "#ffffff"
LIGHT_FG = "#333333"
LIGHT_GRID = "#e0e0e0"

_use_dark = True


def set_theme(dark: bool):
    global _use_dark
    _use_dark = dark


def apply_style(fig, axes):
    bg = DARK_BG if _use_dark else LIGHT_BG
    fg = DARK_FG if _use_dark else LIGHT_FG
    grid = DARK_GRID if _use_dark else LIGHT_GRID
    fig.patch.set_facecolor(bg)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    else:
        axes = list(np.array(axes).flat) if hasattr(axes, '__iter__') else [axes]
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=8)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.grid(True, color=grid, linewidth=0.5, alpha=0.6)


def policy_color(p):
    return POLICY_COLORS.get(p, "#888888")


def policy_label(p):
    return POLICY_LABELS.get(p, p)


def short_video(name):
    """Shorten video path to a readable label."""
    return name.replace("glass/", "g/").replace("porcelain/", "p/")


def save(fig, path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(ROOT)}")


# ── Data Loaders ──────────────────────────────────────────────────────────────

def load_json(relpath):
    p = RESULTS / relpath
    if not p.exists():
        print(f"  WARNING: {p} not found, skipping")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── 01. Model Baselines ──────────────────────────────────────────────────────

def gen_baselines(dpi):
    data = load_json("01_model_baselines/all_baselines.json")
    if not data:
        return
    subdir = OUT / "01_baselines"

    # Group by dataset
    by_ds = {}
    for entry in data:
        ds = entry["dataset"]
        by_ds.setdefault(ds, []).append(entry)

    # --- Figure 1: mAP50 and F1 comparison bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    apply_style(fig, axes)

    datasets = list(by_ds.keys())
    x = np.arange(len(datasets))
    width = 0.35

    for idx, metric in enumerate(["mAP50", "f1"]):
        ax = axes[idx]
        n_vals, s_vals = [], []
        for ds in datasets:
            entries = sorted(by_ds[ds], key=lambda e: e["model"])
            for e in entries:
                if "n" in e["model"]:
                    n_vals.append(e[metric])
                else:
                    s_vals.append(e[metric])

        bars_n = ax.bar(x - width / 2, n_vals, width, label="YOLOv8n",
                        color="#4caf50", edgecolor="white", linewidth=0.5)
        bars_s = ax.bar(x + width / 2, s_vals, width, label="YOLOv8s",
                        color="#e94560", edgecolor="white", linewidth=0.5)

        # Value labels
        for bar in list(bars_n) + list(bars_s):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7, color=DARK_FG if _use_dark else LIGHT_FG)

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in datasets])
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Dataset and Model")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Model Baseline Performance", fontsize=13,
                 color=DARK_FG if _use_dark else LIGHT_FG, y=1.02)
    save(fig, subdir / "baselines_map_f1.png", dpi)

    # --- Figure 2: Inference speed comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_style(fig, ax)

    labels = []
    n_speeds, s_speeds = [], []
    for ds in datasets:
        entries = sorted(by_ds[ds], key=lambda e: e["model"])
        for e in entries:
            if "n" in e["model"]:
                n_speeds.append(e["speed_inference_ms"])
            else:
                s_speeds.append(e["speed_inference_ms"])
        labels.append(ds.capitalize())

    x = np.arange(len(labels))
    ax.bar(x - width / 2, n_speeds, width, label="YOLOv8n", color="#4caf50")
    ax.bar(x + width / 2, s_speeds, width, label="YOLOv8s", color="#e94560")

    for i, (n, s) in enumerate(zip(n_speeds, s_speeds)):
        ax.text(i - width / 2, n + 2, f"{n:.0f}ms", ha="center", fontsize=7,
                color=DARK_FG if _use_dark else LIGHT_FG)
        ax.text(i + width / 2, s + 2, f"{s:.0f}ms", ha="center", fontsize=7,
                color=DARK_FG if _use_dark else LIGHT_FG)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Model Inference Speed by Dataset")
    ax.legend(fontsize=8)
    save(fig, subdir / "baselines_speed.png", dpi)


# ── 02. Detection Quality ─────────────────────────────────────────────────────

def _load_correct_timing():
    """Load timing from repeated_trials (StreamingEngine) instead of detection_quality."""
    trial_data = load_json("03_repeated_trials/repeated_trials.json")
    if not trial_data:
        return {}
    timing = {}
    for vid in trial_data.values():
        for pol, metrics in vid.items():
            if pol not in timing:
                timing[pol] = []
            timing[pol].append(metrics["mean_T_total"]["mean"])
    return {pol: np.mean(vals) for pol, vals in timing.items()}


def gen_detection_quality(dpi):
    data = load_json("02_detection_quality/all_detection_quality.json")
    if not data:
        return
    subdir = OUT / "02_detection"

    videos = list(data.keys())
    correct_timing = _load_correct_timing()

    # --- Aggregate bar chart ---
    agg = {}
    for pol in POLICIES_ORDERED:
        total_oracle = 0
        total_matched = 0
        total_cw_num = 0.0
        total_cw_den = 0.0
        sw_sum = 0.0
        s_pct_sum = 0.0
        count = 0
        for vid in videos:
            d = data[vid].get(pol)
            if not d:
                continue
            total_oracle += d["total_s_oracle_dets"]
            total_matched += d["matched_with_oracle"]
            sw_sum += d["sw_per_100"]
            s_pct_sum += d["s_usage_pct"]
            count += 1
        if count == 0:
            continue
        agg[pol] = {
            "coverage": total_matched / total_oracle if total_oracle > 0 else 0,
            "mean_T": correct_timing.get(pol, 0),  # Use StreamingEngine timing
            "sw_per_100": sw_sum / count,
            "s_pct": s_pct_sum / count,
        }

    # Figure: Aggregate detection coverage bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)

    pols = [p for p in POLICIES_ORDERED if p in agg]
    covs = [agg[p]["coverage"] * 100 for p in pols]
    colors = [policy_color(p) for p in pols]

    bars = ax.bar(range(len(pols)), covs, color=colors, edgecolor="white", linewidth=0.5)
    for bar, cov in zip(bars, covs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{cov:.1f}%", ha="center", va="bottom", fontsize=9,
                color=DARK_FG if _use_dark else LIGHT_FG, fontweight="bold")

    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([policy_label(p) for p in pols], rotation=30, ha="right")
    ax.set_ylabel("Detection Coverage (%)")
    ax.set_title(f"Aggregate Detection Coverage ({len(videos)} Videos, {sum(data[v][POLICIES_ORDERED[0]]['total_frames'] for v in videos):,} Frames)")
    ax.set_ylim(70, 105)
    ax.axhline(100, color="#e94560", linestyle="--", alpha=0.5, label="s_only (oracle)")
    ax.legend(fontsize=8)
    save(fig, subdir / "aggregate_coverage_bar.png", dpi)

    # Figure: Coverage-Latency scatter (Pareto front)
    fig, ax = plt.subplots(figsize=(10, 7))
    apply_style(fig, ax)

    for p in pols:
        ax.scatter(agg[p]["mean_T"], agg[p]["coverage"] * 100,
                   s=150, color=policy_color(p), edgecolors="white",
                   linewidth=1, zorder=5)
        offset_x = 5
        offset_y = 0.5
        if p == "niqe_switch":
            offset_y = -1.5
        if p == "multi_proxy":
            offset_y = -1.5
        ax.annotate(policy_label(p),
                    (agg[p]["mean_T"], agg[p]["coverage"] * 100),
                    textcoords="offset points", xytext=(offset_x, offset_y),
                    fontsize=8, color=DARK_FG if _use_dark else LIGHT_FG)

    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Detection Coverage (%)")
    ax.set_title("Coverage vs. Latency Pareto Front")
    ax.axhline(100, color="#e94560", linestyle=":", alpha=0.3)
    save(fig, subdir / "pareto_coverage_latency.png", dpi)

    # Figure: Per-video coverage heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    apply_style(fig, ax)

    matrix = []
    for pol in pols:
        row = []
        for vid in videos:
            d = data[vid].get(pol)
            row.append(d["detection_coverage"] * 100 if d else 0)
        matrix.append(row)

    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=70, vmax=100)
    ax.set_xticks(range(len(videos)))
    ax.set_xticklabels([short_video(v) for v in videos], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels([policy_label(p) for p in pols], fontsize=8)

    # Annotate cells
    for i in range(len(pols)):
        for j in range(len(videos)):
            val = matrix[i, j]
            color = "black" if val > 85 else "white"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coverage (%)", color=DARK_FG if _use_dark else LIGHT_FG)
    cbar.ax.tick_params(colors=DARK_FG if _use_dark else LIGHT_FG)
    ax.set_title("Detection Coverage per Video per Policy")
    save(fig, subdir / "per_video_coverage_heatmap.png", dpi)

    # Figure: S-model usage vs coverage scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    apply_style(fig, ax)

    for p in pols:
        ax.scatter(agg[p]["s_pct"], agg[p]["coverage"] * 100,
                   s=150, color=policy_color(p), edgecolors="white",
                   linewidth=1, zorder=5)
        ax.annotate(policy_label(p),
                    (agg[p]["s_pct"], agg[p]["coverage"] * 100),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=8, color=DARK_FG if _use_dark else LIGHT_FG)

    ax.set_xlabel("S-Model Usage (%)")
    ax.set_ylabel("Detection Coverage (%)")
    ax.set_title("Coverage vs. S-Model Usage (Switching Efficiency)")
    # Diagonal reference: if coverage scaled linearly with s-usage
    x_ref = np.linspace(0, 100, 100)
    n_cov = agg.get("n_only", {}).get("coverage", 0.8) * 100
    y_ref = n_cov + (100 - n_cov) * x_ref / 100
    ax.plot(x_ref, y_ref, "--", color="#888888", alpha=0.4, label="Linear baseline")
    ax.legend(fontsize=8)
    save(fig, subdir / "coverage_vs_s_usage.png", dpi)

    # Figure: Switching rate bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_style(fig, ax)

    adaptive = [p for p in pols if p not in ("n_only", "s_only")]
    sw_rates = [agg[p]["sw_per_100"] for p in adaptive]
    colors_a = [policy_color(p) for p in adaptive]
    bars = ax.bar(range(len(adaptive)), sw_rates, color=colors_a, edgecolor="white")
    for bar, sw in zip(bars, sw_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{sw:.2f}", ha="center", va="bottom", fontsize=8,
                color=DARK_FG if _use_dark else LIGHT_FG)
    ax.set_xticks(range(len(adaptive)))
    ax.set_xticklabels([policy_label(p) for p in adaptive], rotation=30, ha="right")
    ax.set_ylabel("Switches per 100 Frames")
    ax.set_title("Switching Rate by Policy (Aggregate)")
    save(fig, subdir / "switching_rate_bar.png", dpi)

    # Figure: Per-video detection coverage grouped bar
    for vid in videos:
        fig, ax = plt.subplots(figsize=(10, 5))
        apply_style(fig, ax)

        vdata = data[vid]
        pols_v = [p for p in POLICIES_ORDERED if p in vdata]
        covs_v = [vdata[p]["detection_coverage"] * 100 for p in pols_v]
        colors_v = [policy_color(p) for p in pols_v]
        bars = ax.bar(range(len(pols_v)), covs_v, color=colors_v, edgecolor="white")
        for bar, cov in zip(bars, covs_v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{cov:.1f}%", ha="center", va="bottom", fontsize=7,
                    color=DARK_FG if _use_dark else LIGHT_FG)
        ax.set_xticks(range(len(pols_v)))
        ax.set_xticklabels([policy_label(p) for p in pols_v], rotation=30, ha="right")
        ax.set_ylabel("Coverage (%)")
        nframes = vdata[pols_v[0]]["total_frames"]
        ax.set_title(f"Detection Coverage — {vid} ({nframes} frames)")
        ax.set_ylim(60, 105)

        safe_name = vid.replace("/", "_").replace("\\", "_")
        save(fig, subdir / f"coverage_{safe_name}.png", dpi)

    # Figure: Per-video latency grouped bar
    for vid in videos:
        fig, ax = plt.subplots(figsize=(10, 5))
        apply_style(fig, ax)

        vdata = data[vid]
        pols_v = [p for p in POLICIES_ORDERED if p in vdata]
        latencies = [vdata[p]["mean_T_total_ms"] for p in pols_v]
        colors_v = [policy_color(p) for p in pols_v]
        bars = ax.bar(range(len(pols_v)), latencies, color=colors_v, edgecolor="white")
        for bar, t in zip(bars, latencies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{t:.0f}", ha="center", va="bottom", fontsize=7,
                    color=DARK_FG if _use_dark else LIGHT_FG)
        ax.set_xticks(range(len(pols_v)))
        ax.set_xticklabels([policy_label(p) for p in pols_v], rotation=30, ha="right")
        ax.set_ylabel("Mean T_total (ms)")
        ax.set_title(f"Mean Latency — {vid}")

        safe_name = vid.replace("/", "_").replace("\\", "_")
        save(fig, subdir / f"latency_{safe_name}.png", dpi)


# ── 03. Repeated Trials ───────────────────────────────────────────────────────

def gen_repeated_trials(dpi):
    data = load_json("03_repeated_trials/repeated_trials.json")
    if not data:
        return
    subdir = OUT / "03_timing"

    videos = list(data.keys())

    # Figure: Mean T_total with error bars, per video
    for vid in videos:
        fig, ax = plt.subplots(figsize=(10, 5))
        apply_style(fig, ax)

        vdata = data[vid]
        pols = [p for p in POLICIES_ORDERED if p in vdata]
        means = [vdata[p]["mean_T_total"]["mean"] for p in pols]
        stds = [vdata[p]["mean_T_total"]["std"] for p in pols]
        colors = [policy_color(p) for p in pols]

        bars = ax.bar(range(len(pols)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth": 1.5, "capthick": 1.5,
                                "ecolor": DARK_FG if _use_dark else LIGHT_FG})
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 2,
                    f"{m:.0f}±{s:.0f}", ha="center", va="bottom", fontsize=7,
                    color=DARK_FG if _use_dark else LIGHT_FG)

        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([policy_label(p) for p in pols], rotation=30, ha="right")
        ax.set_ylabel("Mean T_total (ms)")
        ax.set_title(f"Repeated Trials (3×) — {vid}")

        safe_name = vid.replace("/", "_").replace("\\", "_")
        save(fig, subdir / f"timing_trials_{safe_name}.png", dpi)

    # Figure: P95 comparison across videos (grouped bar)
    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(fig, ax)

    pols = [p for p in POLICIES_ORDERED if all(p in data[v] for v in videos)]
    n_vids = len(videos)
    n_pols = len(pols)
    bar_w = 0.8 / n_vids
    x = np.arange(n_pols)

    for vi, vid in enumerate(videos):
        p95s = [data[vid][p]["p95_T_total"]["mean"] for p in pols]
        offset = (vi - n_vids / 2 + 0.5) * bar_w
        ax.bar(x + offset, p95s, bar_w, label=short_video(vid), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([policy_label(p) for p in pols], rotation=30, ha="right")
    ax.set_ylabel("P95 T_total (ms)")
    ax.set_title("P95 Latency Across Videos (3-Trial Mean)")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    save(fig, subdir / "p95_comparison_all_videos.png", dpi)

    # Figure: Timing consistency (std as % of mean) heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)

    matrix = []
    for pol in pols:
        row = []
        for vid in videos:
            m = data[vid][pol]["mean_T_total"]["mean"]
            s = data[vid][pol]["mean_T_total"]["std"]
            row.append(abs(s / m * 100) if m > 0 else 0)
        matrix.append(row)

    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)
    ax.set_xticks(range(len(videos)))
    ax.set_xticklabels([short_video(v) for v in videos], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels([policy_label(p) for p in pols], fontsize=8)

    for i in range(len(pols)):
        for j in range(len(videos)):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center",
                    fontsize=6, color="black" if matrix[i, j] < 5 else "white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("CoV (%)", color=DARK_FG if _use_dark else LIGHT_FG)
    cbar.ax.tick_params(colors=DARK_FG if _use_dark else LIGHT_FG)
    ax.set_title("Timing Consistency (Std/Mean %) Across Trials")
    save(fig, subdir / "timing_consistency_heatmap.png", dpi)


# ── 04. Ablation Study ────────────────────────────────────────────────────────

def gen_ablation(dpi):
    data = load_json("04_ablation/ablation_results.json")
    if not data:
        return
    subdir = OUT / "04_ablation"

    videos = list(data.keys())

    # Category colors
    CAT_COLORS = {
        "baseline": "#78909c",
        "proxy": "#4caf50",
        "controller": "#ff9800",
        "stability": "#2196f3",
    }

    for vid in videos:
        vdata = data[vid]
        variants = list(vdata.keys())
        categories = [vdata[v].get("category", "unknown") for v in variants]

        # Figure: T_total_mean bar chart per variant
        fig, ax = plt.subplots(figsize=(14, 6))
        apply_style(fig, ax)

        t_means = [vdata[v]["T_total_mean"] for v in variants]
        colors = [CAT_COLORS.get(c, "#888") for c in categories]

        bars = ax.barh(range(len(variants)), t_means, color=colors, edgecolor="white")
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants, fontsize=7)
        ax.set_xlabel("Mean T_total (ms)")
        ax.set_title(f"Ablation — {vid}")
        ax.invert_yaxis()

        for bar, t in zip(bars, t_means):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{t:.0f}", va="center", fontsize=7,
                    color=DARK_FG if _use_dark else LIGHT_FG)

        # Legend for categories
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=CAT_COLORS[c], label=c.capitalize())
                   for c in ["baseline", "proxy", "controller", "stability"]
                   if c in set(categories)]
        ax.legend(handles=handles, fontsize=8, loc="lower right")

        safe_name = vid.replace("/", "_").replace("\\", "_")
        save(fig, subdir / f"ablation_{safe_name}.png", dpi)

    # Figure: Switching rate reduction (hysteresis effect)
    fig, axes = plt.subplots(1, min(len(videos), 4), figsize=(6 * min(len(videos), 4), 7),
                             squeeze=False)
    apply_style(fig, axes[0])

    for vi, vid in enumerate(list(videos)[:4]):
        ax = axes[0][vi]
        vdata = data[vid]

        y_labels, sw_vals = [], []
        for v in vdata:
            if vdata[v].get("sw_per_100") is not None:
                y_labels.append(v)
                sw_vals.append(vdata[v]["sw_per_100"])

        colors = [CAT_COLORS.get(vdata[v].get("category", ""), "#888") for v in y_labels]
        ax.barh(range(len(y_labels)), sw_vals, color=colors, edgecolor="white")
        ax.set_yticks(range(len(y_labels)))
        # Shorten long labels
        short_labels = [v.replace(" (baseline)", "").replace(" (hyst)", "")[:18] for v in y_labels]
        ax.set_yticklabels(short_labels, fontsize=5)
        ax.set_xlabel("sw/100", fontsize=7)
        ax.set_title(short_video(vid), fontsize=9)
        ax.invert_yaxis()

    fig.suptitle("Switching Rate Across Ablation Variants",
                 color=DARK_FG if _use_dark else LIGHT_FG, fontsize=12)
    save(fig, subdir / "ablation_switching_rates.png", dpi)


# ── 05. Generalization ────────────────────────────────────────────────────────

def gen_generalization(dpi):
    data = load_json("05_generalization/generalization_results.json")
    if not data:
        return
    subdir = OUT / "05_generalization"

    # --- Cross-material transfer ---
    cross = data.get("cross_material", {})
    if cross:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        apply_style(fig, axes)

        pols = list(cross.keys())
        t_ratios = [cross[p]["T_ratio"] for p in pols]
        sw_ratios = [cross[p]["sw_ratio"] for p in pols]
        colors = [policy_color(p) for p in pols]

        # T_ratio
        ax = axes[0]
        bars = ax.bar(range(len(pols)), t_ratios, color=colors, edgecolor="white")
        ax.axhline(1.0, color="#e94560", linestyle="--", alpha=0.5, label="No degradation")
        for bar, r in zip(bars, t_ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{r:.2f}", ha="center", va="bottom", fontsize=9,
                    color=DARK_FG if _use_dark else LIGHT_FG)
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([policy_label(p) for p in pols], rotation=30, ha="right")
        ax.set_ylabel("T_ratio (cross / native)")
        ax.set_title("Cross-Material Transfer: Timing")
        ax.set_ylim(0.7, 1.2)
        ax.legend(fontsize=8)

        # sw_ratio
        ax = axes[1]
        bars = ax.bar(range(len(pols)), sw_ratios, color=colors, edgecolor="white")
        ax.axhline(1.0, color="#e94560", linestyle="--", alpha=0.5)
        for bar, r in zip(bars, sw_ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{r:.2f}", ha="center", va="bottom", fontsize=9,
                    color=DARK_FG if _use_dark else LIGHT_FG)
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([policy_label(p) for p in pols], rotation=30, ha="right")
        ax.set_ylabel("sw_ratio (cross / native)")
        ax.set_title("Cross-Material Transfer: Switching Rate")
        ax.set_ylim(0.5, 1.7)

        fig.suptitle("Glass → Porcelain Transfer",
                     color=DARK_FG if _use_dark else LIGHT_FG, fontsize=13)
        save(fig, subdir / "cross_material_transfer.png", dpi)

    # --- Leave-one-out ---
    loo = data.get("leave_one_out", {})
    if loo:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        apply_style(fig, axes)

        pols = list(loo.keys())
        metrics = [
            ("T_cv", "Timing CoV", "Coefficient of Variation (T_total)"),
            ("sw_mean", "Mean sw/100", "Switches per 100 Frames"),
            ("s_pct_mean", "Mean S-Usage (%)", "S-Model Usage Percentage"),
        ]

        for ax_idx, (key, ylabel, title) in enumerate(metrics):
            ax = axes[ax_idx]
            vals = [loo[p].get(key, 0) for p in pols]
            if key == "T_cv":
                vals = [v * 100 for v in vals]  # Convert to %
            colors = [policy_color(p) for p in pols]

            bars = ax.bar(range(len(pols)), vals, color=colors, edgecolor="white")
            for bar, v in zip(bars, vals):
                fmt = f"{v:.1f}%" if key == "T_cv" else f"{v:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        fmt, ha="center", va="bottom", fontsize=8,
                        color=DARK_FG if _use_dark else LIGHT_FG)

            ax.set_xticks(range(len(pols)))
            ax.set_xticklabels([policy_label(p) for p in pols], rotation=30, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        fig.suptitle("Leave-One-Out Generalization (Glass Videos)",
                     color=DARK_FG if _use_dark else LIGHT_FG, fontsize=13)
        save(fig, subdir / "leave_one_out_summary.png", dpi)


# ── 06. C-Score Analysis ─────────────────────────────────────────────────────

def gen_cscore(dpi):
    subdir = OUT / "06_cscore"

    # Load all cscore analysis files
    cscore_dir = RESULTS / "06_cscore_analysis"
    if not cscore_dir.exists():
        return

    analyses = {}
    for f in cscore_dir.glob("cscore_analysis_*.json"):
        name = f.stem.replace("cscore_analysis_", "")
        if name == "unknown":
            continue
        with open(f, encoding="utf-8") as fh:
            analyses[name] = json.load(fh)

    if not analyses:
        # Try the generic one
        generic = load_json("06_cscore_analysis/cscore_analysis.json")
        if generic:
            analyses["default"] = generic

    if not analyses:
        return

    for name, analysis in analyses.items():
        # Figure: Raw proxy correlations bar chart
        raw = analysis.get("raw_correlations", {})
        if raw:
            fig, ax = plt.subplots(figsize=(12, 6))
            apply_style(fig, ax)

            proxies = list(raw.keys())
            rs = [raw[p]["r"] for p in proxies]
            abs_rs = [abs(r) for r in rs]

            # Sort by |r|
            order = np.argsort(abs_rs)[::-1]
            proxies_s = [proxies[i] for i in order]
            rs_s = [rs[i] for i in order]

            colors = ["#4caf50" if r > 0 else "#e94560" for r in rs_s]
            bars = ax.barh(range(len(proxies_s)), [abs(r) for r in rs_s],
                           color=colors, edgecolor="white")
            ax.set_yticks(range(len(proxies_s)))
            ax.set_yticklabels(proxies_s, fontsize=9)
            ax.set_xlabel("|Pearson r| with Model Disagreement")
            ax.set_title(f"Proxy Correlation with Model Disagreement — {name}")
            ax.invert_yaxis()

            # Annotate values
            for bar, r in zip(bars, rs_s):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"r={r:+.3f}", va="center", fontsize=8,
                        color=DARK_FG if _use_dark else LIGHT_FG)

            # Highlight C-score
            for i, p in enumerate(proxies_s):
                if "C-score" in p:
                    ax.get_children()[i].set_edgecolor("#ffeb3b")
                    ax.get_children()[i].set_linewidth(2)

            save(fig, subdir / f"proxy_correlations_{name}.png", dpi)

        # Figure: Alpha sweep
        alpha_sweep = analysis.get("alpha_sweep", {})
        if alpha_sweep:
            fig, ax = plt.subplots(figsize=(8, 5))
            apply_style(fig, ax)

            alphas = []
            r_vals = []
            for k, v in sorted(alpha_sweep.items()):
                a = float(k.replace("alpha_", ""))
                alphas.append(a)
                r_vals.append(v)

            ax.plot(alphas, r_vals, "o-", color="#00bcd4", linewidth=2, markersize=8)
            ax.axhline(0, color="#888888", linestyle=":", alpha=0.5)

            # Mark best
            best = analysis.get("best_alpha", {})
            if best:
                ax.scatter([best["alpha"]], [best["r"]], s=200, color="#ffeb3b",
                           edgecolors="white", zorder=10, label=f"Best: α={best['alpha']}")
                ax.legend(fontsize=8)

            ax.set_xlabel("α (weight on Laplacian)")
            ax.set_ylabel("Pearson r (C vs. Disagreement)")
            ax.set_title(f"C-score α Sweep — {name}")
            save(fig, subdir / f"alpha_sweep_{name}.png", dpi)

        # Figure: Normalization impact
        norm = analysis.get("normalization_impact", {})
        if norm:
            fig, ax = plt.subplots(figsize=(8, 5))
            apply_style(fig, ax)

            labels = ["L raw", "L norm", "H raw", "H norm"]
            vals = [norm.get("L_raw_r", 0), norm.get("L_norm_r", 0),
                    norm.get("H_raw_r", 0), norm.get("H_norm_r", 0)]
            colors = ["#4caf50", "#4caf50", "#e94560", "#e94560"]
            alphas_v = [1.0, 0.5, 1.0, 0.5]

            bars = ax.bar(range(len(labels)), [abs(v) for v in vals],
                          color=colors, edgecolor="white")
            for bar, a in zip(bars, alphas_v):
                bar.set_alpha(a)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:+.3f}", ha="center", va="bottom", fontsize=9,
                        color=DARK_FG if _use_dark else LIGHT_FG)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("|Pearson r|")
            ax.set_title(f"Normalization Destroys Signal — {name}")

            # Arrow showing degradation
            deg = norm.get("normalization_degradation_L", 0)
            ax.annotate(f"delta = {deg:+.3f}",
                        xy=(1, abs(vals[1])), xytext=(1.5, abs(vals[0]) - 0.02),
                        fontsize=9, color="#ffeb3b",
                        arrowprops=dict(arrowstyle="->", color="#ffeb3b"))

            save(fig, subdir / f"normalization_impact_{name}.png", dpi)


# ── 07. Summary / Paper-Ready ─────────────────────────────────────────────────

def gen_summary(dpi):
    det_data = load_json("02_detection_quality/all_detection_quality.json")
    trial_data = load_json("03_repeated_trials/repeated_trials.json")
    if not det_data:
        return
    subdir = OUT / "07_summary"

    videos = list(det_data.keys())
    correct_timing = _load_correct_timing()

    # Compute aggregates
    agg = {}
    for pol in POLICIES_ORDERED:
        total_oracle = 0
        total_matched = 0
        sw_sum = 0.0
        s_pct_sum = 0.0
        count = 0
        for vid in videos:
            d = det_data[vid].get(pol)
            if not d:
                continue
            total_oracle += d["total_s_oracle_dets"]
            total_matched += d["matched_with_oracle"]
            sw_sum += d["sw_per_100"]
            s_pct_sum += d["s_usage_pct"]
            count += 1
        if count == 0:
            continue
        agg[pol] = {
            "coverage": total_matched / total_oracle if total_oracle > 0 else 0,
            "mean_T": correct_timing.get(pol, 0),  # Use StreamingEngine timing
            "sw_per_100": sw_sum / count,
            "s_pct": s_pct_sum / count,
        }

    # --- Grand summary figure (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    apply_style(fig, axes)

    pols = [p for p in POLICIES_ORDERED if p in agg]

    # (a) Coverage bar
    ax = axes[0, 0]
    covs = [agg[p]["coverage"] * 100 for p in pols]
    colors = [policy_color(p) for p in pols]
    bars = ax.bar(range(len(pols)), covs, color=colors, edgecolor="white")
    for bar, cov in zip(bars, covs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{cov:.1f}%", ha="center", va="bottom", fontsize=8,
                color=DARK_FG if _use_dark else LIGHT_FG, fontweight="bold")
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([policy_label(p) for p in pols], rotation=30, ha="right")
    ax.set_ylabel("Detection Coverage (%)")
    ax.set_title("(a) Detection Coverage")
    ax.set_ylim(70, 105)

    # (b) Latency bar
    ax = axes[0, 1]
    lats = [agg[p]["mean_T"] for p in pols]
    bars = ax.bar(range(len(pols)), lats, color=colors, edgecolor="white")
    for bar, t in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{t:.0f}", ha="center", va="bottom", fontsize=8,
                color=DARK_FG if _use_dark else LIGHT_FG)
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([policy_label(p) for p in pols], rotation=30, ha="right")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("(b) Mean Latency")

    # (c) Pareto scatter
    ax = axes[1, 0]
    for p in pols:
        ax.scatter(agg[p]["mean_T"], agg[p]["coverage"] * 100,
                   s=180, color=policy_color(p), edgecolors="white",
                   linewidth=1.5, zorder=5)
        ax.annotate(policy_label(p),
                    (agg[p]["mean_T"], agg[p]["coverage"] * 100),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=DARK_FG if _use_dark else LIGHT_FG)
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Detection Coverage (%)")
    ax.set_title("(c) Coverage–Latency Pareto Front")

    # (d) S-usage vs coverage with efficiency annotation
    ax = axes[1, 1]
    for p in pols:
        cov = agg[p]["coverage"] * 100
        s_pct = agg[p]["s_pct"]
        ax.scatter(s_pct, cov, s=180, color=policy_color(p),
                   edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(policy_label(p),
                    (s_pct, cov), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=DARK_FG if _use_dark else LIGHT_FG)
    ax.set_xlabel("S-Model Usage (%)")
    ax.set_ylabel("Detection Coverage (%)")
    ax.set_title("(d) Switching Efficiency")

    fig.suptitle(f"DMS-Raptor: Full Experiment Summary ({len(videos)} Videos)",
                 color=DARK_FG if _use_dark else LIGHT_FG, fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, subdir / "grand_summary_4panel.png", dpi)

    # --- Paper-ready: Table as figure ---
    fig, ax = plt.subplots(figsize=(14, 4))
    apply_style(fig, ax)
    ax.axis("off")

    headers = ["Policy", "Coverage", "Miss%", "T̄ (ms)", "sw/100", "s%"]
    rows = []
    for p in pols:
        a = agg[p]
        rows.append([
            policy_label(p),
            f"{a['coverage']*100:.1f}%",
            f"{(1-a['coverage'])*100:.1f}%",
            f"{a['mean_T']:.0f}",
            f"{a['sw_per_100']:.2f}",
            f"{a['s_pct']:.1f}%",
        ])

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#2a3a5c")
        cell.set_text_props(color="white", fontweight="bold")

    # Style data rows
    for i in range(len(rows)):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_facecolor("#1e2530" if i % 2 == 0 else "#232a36")
            cell.set_text_props(color=DARK_FG if _use_dark else LIGHT_FG)

    ax.set_title("Aggregate Detection Quality Summary",
                 color=DARK_FG if _use_dark else LIGHT_FG, fontsize=12, pad=20)
    save(fig, subdir / "summary_table.png", dpi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate all thesis figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--light", action="store_true", help="Use light theme")
    args = parser.parse_args()

    if args.light:
        set_theme(False)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    print("=" * 60)
    print("  DMS-Raptor: Generating All Research Figures")
    print("=" * 60)

    print("\n[1/7] Model Baselines...")
    gen_baselines(args.dpi)

    print("\n[2/7] Detection Quality...")
    gen_detection_quality(args.dpi)

    print("\n[3/7] Repeated Trials / Timing...")
    gen_repeated_trials(args.dpi)

    print("\n[4/7] Ablation Study...")
    gen_ablation(args.dpi)

    print("\n[5/7] Generalization...")
    gen_generalization(args.dpi)

    print("\n[6/7] C-Score Analysis...")
    gen_cscore(args.dpi)

    print("\n[7/7] Summary / Paper-Ready...")
    gen_summary(args.dpi)

    # Count outputs
    total = sum(1 for _ in OUT.rglob("*.png")) if OUT.exists() else 0
    print(f"\n{'=' * 60}")
    print(f"  Done! Generated {total} figures in:")
    print(f"  {OUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
