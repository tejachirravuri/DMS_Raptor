"""Publication-quality thesis plot functions for DMS-Raptor."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.ticker as mticker

from gui.plot_utils import apply_dark, color_for, theme_colors, POLICY_COLORS, TIMING_COLORS

try:
    from scipy.stats import pearsonr as _pearsonr

    def pearsonr(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0, 1.0
        return _pearsonr(x, y)
except ImportError:
    def pearsonr(x, y):
        """Fallback Pearson r using numpy; p-value approximated via t-test."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        n = len(x)
        if n < 3:
            return 0.0, 1.0
        r = float(np.corrcoef(x, y)[0, 1])
        # Two-tailed p-value via t approximation
        if abs(r) >= 1.0:
            return r, 0.0
        t_stat = r * np.sqrt((n - 2) / (1.0 - r * r))
        # Rough p-value from normal approximation for large n
        p = float(2.0 * np.exp(-0.717 * t_stat * t_stat - 0.416 * abs(t_stat)))
        p = max(0.0, min(1.0, p))
        return r, p


def _gaussian_kde_manual(data: np.ndarray, x_grid: np.ndarray,
                         bandwidth: Optional[float] = None) -> np.ndarray:
    """Simple Gaussian KDE without scipy dependency."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]  # drop NaN / Inf values
    n = len(data)
    if n == 0:
        return np.zeros_like(x_grid)
    if bandwidth is None:
        std = np.std(data, ddof=1) if n > 1 else 1.0
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        h = 0.9 * min(std, iqr / 1.34) * n ** (-0.2) if iqr > 0 else 1.06 * std * n ** (-0.2)
        bandwidth = max(h, 1e-6)
    density = np.zeros_like(x_grid, dtype=float)
    for xi in data:
        density += np.exp(-0.5 * ((x_grid - xi) / bandwidth) ** 2)
    density /= (n * bandwidth * np.sqrt(2.0 * np.pi))
    return density


def _legend(ax, tc: dict, **kwargs) -> None:
    """Add a styled legend to an axes."""
    leg = ax.legend(
        facecolor=tc["legend_bg"], edgecolor=tc["legend_edge"],
        labelcolor=tc["fg"], fontsize=8, **kwargs,
    )
    if leg is not None:
        leg.get_frame().set_alpha(0.9)


def _filter_summaries(summaries, policies_filter):
    """Return summaries matching the policy filter, or all if None."""
    if policies_filter is None:
        return list(summaries)
    return [s for s in summaries if s.policy in policies_filter]


def _choice_to_binary(choice_trace: List[str]) -> np.ndarray:
    """Convert choice_trace ('n'/'s') to binary array (0=n, 1=s)."""
    return np.array([0 if c == "n" else 1 for c in choice_trace], dtype=float)


def _run_lengths(choice_trace: List[str]) -> List[int]:
    """Compute consecutive run lengths from choice_trace."""
    if not choice_trace:
        return []
    lengths = []
    current_len = 1
    for i in range(1, len(choice_trace)):
        if choice_trace[i] == choice_trace[i - 1]:
            current_len += 1
        else:
            lengths.append(current_len)
            current_len = 1
    lengths.append(current_len)
    return lengths


def _simulate_hysteresis(C_trace: List[float], c_low: float, c_high: float):
    """Simulate hysteresis switching on a C_trace.

    Returns (switches, s_usage_pct).
    """
    if not C_trace:
        return 0, 0.0
    use_s = False
    switches = 0
    s_count = 0
    for c in C_trace:
        if use_s and c < c_low:
            use_s = False
            switches += 1
        elif not use_s and c > c_high:
            use_s = True
            switches += 1
        if use_s:
            s_count += 1
    n = len(C_trace)
    sw_per_100 = switches / n * 100.0 if n > 0 else 0.0
    s_pct = s_count / n * 100.0 if n > 0 else 0.0
    return sw_per_100, s_pct


# =========================================================================
# 1. Latency CDF
# =========================================================================
def plot_latency_cdf(fig: Figure, ax, summaries, policies_filter=None):
    """CDF of T_total per policy for deadline compliance analysis.

    Plots the empirical cumulative distribution function of per-frame
    total latency for each policy, with vertical reference lines at
    P95 and P99 percentiles.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])
    sums = _filter_summaries(summaries, policies_filter)

    for idx, s in enumerate(sums):
        if not s.T_total_trace:
            continue
        data = np.sort(s.T_total_trace)
        cdf = np.arange(1, len(data) + 1) / len(data)
        clr = color_for(s.policy, idx)
        label = s.policy
        ax.plot(data, cdf, color=clr, linewidth=1.5, label=label)
        # P95 / P99 vertical lines
        p95 = np.percentile(data, 95)
        p99 = np.percentile(data, 99)
        ax.axvline(p95, color=clr, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(p99, color=clr, linestyle=":", linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r"$T_{\mathrm{total}}$ (ms)", color=tc["fg"], fontsize=10)
    ax.set_ylabel("Cumulative Probability", color=tc["fg"], fontsize=10)
    ax.set_title("Latency CDF", color=tc["fg"], fontsize=12)
    ax.set_ylim(0, 1.05)
    _legend(ax, tc, loc="lower right")


# =========================================================================
# 2. Latency box plot
# =========================================================================
def plot_latency_boxplot(fig: Figure, ax, summaries, policies_filter=None):
    """Box plot of T_total per policy showing median, IQR, and outliers."""
    tc = theme_colors()
    apply_dark(fig, [ax])
    sums = _filter_summaries(summaries, policies_filter)

    data_list = []
    labels = []
    colors = []
    for idx, s in enumerate(sums):
        if not s.T_total_trace:
            continue
        data_list.append(s.T_total_trace)
        labels.append(s.policy)
        colors.append(color_for(s.policy, idx))

    if not data_list:
        ax.set_title("No data", color=tc["fg"])
        return

    bp = ax.boxplot(
        data_list, patch_artist=True, notch=True,
        medianprops=dict(color=tc["fg"], linewidth=1.5),
        whiskerprops=dict(color=tc["fg"]),
        capprops=dict(color=tc["fg"]),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, clr in zip(bp["boxes"], colors):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    for flier, clr in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(clr)
        flier.set_markeredgecolor(clr)

    ax.set_xticklabels(labels, rotation=30, ha="right", color=tc["fg"], fontsize=8)
    ax.set_ylabel(r"$T_{\mathrm{total}}$ (ms)", color=tc["fg"], fontsize=10)
    ax.set_title("Latency Distribution per Policy", color=tc["fg"], fontsize=12)


# =========================================================================
# 3. Timing breakdown stacked bar
# =========================================================================
def plot_timing_breakdown(fig: Figure, ax, summaries):
    """Stacked bar chart: T_scene + T_ctrl + T_infer_n + T_infer_s per policy."""
    tc = theme_colors()
    apply_dark(fig, [ax])

    policies = []
    t_scene, t_ctrl, t_infer_n, t_infer_s = [], [], [], []
    for s in summaries:
        policies.append(s.policy)
        t_scene.append(s.T_scene_ms_mean)
        t_ctrl.append(s.T_ctrl_ms_mean)
        t_infer_n.append(s.T_infer_n_ms_mean)
        t_infer_s.append(s.T_infer_s_ms_mean)

    if not policies:
        ax.set_title("No data", color=tc["fg"])
        return

    x = np.arange(len(policies))
    width = 0.6
    t_scene = np.array(t_scene)
    t_ctrl = np.array(t_ctrl)
    t_infer_n = np.array(t_infer_n)
    t_infer_s = np.array(t_infer_s)

    bars_scene = ax.bar(x, t_scene, width, label=r"$T_{\mathrm{scene}}$",
                        color=TIMING_COLORS["T_scene"], alpha=0.85)
    bars_ctrl = ax.bar(x, t_ctrl, width, bottom=t_scene,
                       label=r"$T_{\mathrm{ctrl}}$", color=TIMING_COLORS["T_ctrl"], alpha=0.85)
    bars_n = ax.bar(x, t_infer_n, width, bottom=t_scene + t_ctrl,
                    label=r"$T_{\mathrm{infer\_n}}$", color=TIMING_COLORS["T_infer_n"], alpha=0.85)
    bars_s = ax.bar(x, t_infer_s, width, bottom=t_scene + t_ctrl + t_infer_n,
                    label=r"$T_{\mathrm{infer\_s}}$", color=TIMING_COLORS["T_infer_s"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=30, ha="right", color=tc["fg"], fontsize=8)
    ax.set_ylabel("Mean Latency (ms)", color=tc["fg"], fontsize=10)
    ax.set_title("Timing Breakdown per Policy", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="upper right")


# =========================================================================
# 4. Latency time series overlay with P95 band
# =========================================================================
def plot_latency_timeseries(fig: Figure, ax, summaries, policies_filter=None):
    """T_total over time for all policies, with shaded P95 band."""
    tc = theme_colors()
    apply_dark(fig, [ax])
    sums = _filter_summaries(summaries, policies_filter)

    for idx, s in enumerate(sums):
        if not s.T_total_trace:
            continue
        frames = s.frame_indices if s.frame_indices else list(range(len(s.T_total_trace)))
        data = np.array(s.T_total_trace)
        clr = color_for(s.policy, idx)
        ax.plot(frames, data, color=clr, linewidth=0.8, alpha=0.85, label=s.policy)
        p95 = np.percentile(data, 95)
        ax.axhline(p95, color=clr, linestyle="--", linewidth=0.7, alpha=0.5)
        ax.fill_between(frames, p95, data.max(), where=data >= p95,
                        color=clr, alpha=0.10)

    ax.set_xlabel("Frame", color=tc["fg"], fontsize=10)
    ax.set_ylabel(r"$T_{\mathrm{total}}$ (ms)", color=tc["fg"], fontsize=10)
    ax.set_title("Latency Time Series", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="upper right")


# =========================================================================
# 5. Choice heatmap
# =========================================================================
def plot_choice_heatmap(fig: Figure, axes_list, summaries):
    """Binary strip heatmap per policy: green=n, red=s.

    axes_list should have len(summaries) axes vertically stacked.
    """
    tc = theme_colors()
    apply_dark(fig, axes_list)

    for i, (ax, s) in enumerate(zip(axes_list, summaries)):
        if not s.choice_trace:
            ax.set_ylabel(s.policy, color=tc["fg"], fontsize=8, rotation=0,
                          labelpad=60, va="center")
            continue
        binary = _choice_to_binary(s.choice_trace)
        # Create a 2D array for imshow (1 row)
        strip = binary.reshape(1, -1)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#4caf50", "#e94560"])
        ax.imshow(strip, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(s.policy, color=tc["fg"], fontsize=8, rotation=0,
                      labelpad=60, va="center")
        if i < len(summaries) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Frame", color=tc["fg"], fontsize=10)

    if axes_list:
        axes_list[0].set_title("Model Choice Heatmap (green=n, red=s)",
                               color=tc["fg"], fontsize=12)


# =========================================================================
# 6. Rolling switching rate
# =========================================================================
def plot_switching_rate(fig: Figure, ax, summaries, window: int = 100):
    """Rolling switches per 100 frames over time.

    Computes the number of model-choice transitions within a rolling
    window and normalizes to switches per 100 frames.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    for idx, s in enumerate(summaries):
        if not s.choice_trace or len(s.choice_trace) < 2:
            continue
        binary = _choice_to_binary(s.choice_trace)
        transitions = np.abs(np.diff(binary))  # 1 where switch occurs
        # Rolling sum
        kernel = np.ones(window)
        if len(transitions) < window:
            rolling = np.cumsum(transitions) / (np.arange(1, len(transitions) + 1)) * 100.0
        else:
            rolling = np.convolve(transitions, kernel, mode="valid") / window * 100.0
        frames = np.arange(len(rolling))
        clr = color_for(s.policy, idx)
        ax.plot(frames, rolling, color=clr, linewidth=1.2, label=s.policy)

    ax.set_xlabel("Frame", color=tc["fg"], fontsize=10)
    ax.set_ylabel("Switches / 100 frames", color=tc["fg"], fontsize=10)
    ax.set_title(f"Rolling Switching Rate (window={window})", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="upper right")


# =========================================================================
# 7. Dwell histogram
# =========================================================================
def plot_dwell_histogram(fig: Figure, ax, summaries):
    """Distribution of consecutive dwell lengths (run lengths) per policy."""
    tc = theme_colors()
    apply_dark(fig, [ax])

    for idx, s in enumerate(summaries):
        if not s.choice_trace:
            continue
        lengths = _run_lengths(s.choice_trace)
        if not lengths:
            continue
        clr = color_for(s.policy, idx)
        max_len = max(lengths)
        bins = np.arange(1, min(max_len + 2, 202))
        ax.hist(lengths, bins=bins, color=clr, alpha=0.55, edgecolor=clr,
                linewidth=0.5, label=s.policy, density=True)

    ax.set_xlabel("Dwell Length (frames)", color=tc["fg"], fontsize=10)
    ax.set_ylabel("Density", color=tc["fg"], fontsize=10)
    ax.set_title("Dwell Duration Distribution", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="upper right")


# =========================================================================
# 8. Proxy scatter with regression
# =========================================================================
def plot_proxy_scatter(fig: Figure, ax, summaries, x_attr: str, y_attr: str,
                       x_label: str, y_label: str):
    """Scatter of any two traces with regression line, Pearson r, and p-value.

    x_attr / y_attr are attribute names on RunSummary (e.g. 'C_trace').
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    for idx, s in enumerate(summaries):
        x_data = np.array(getattr(s, x_attr, []), dtype=float)
        y_data = np.array(getattr(s, y_attr, []), dtype=float)
        n = min(len(x_data), len(y_data))
        if n < 3:
            continue
        x_data, y_data = x_data[:n], y_data[:n]
        clr = color_for(s.policy, idx)
        ax.scatter(x_data, y_data, s=4, alpha=0.35, color=clr, label=s.policy)

        # Regression line
        r, p = pearsonr(x_data, y_data)
        if np.std(x_data) > 0 and np.std(y_data) > 0 and np.isfinite(r):
            coeffs = np.polyfit(x_data, y_data, 1)
            x_fit = np.linspace(x_data.min(), x_data.max(), 100)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, color=clr, linewidth=1.5, linestyle="--", alpha=0.9)
        ax.annotate(f"r={r:.3f}, p={p:.2e}", xy=(0.02, 0.95 - idx * 0.07),
                    xycoords="axes fraction", fontsize=8, color=clr)

    ax.set_xlabel(x_label, color=tc["fg"], fontsize=10)
    ax.set_ylabel(y_label, color=tc["fg"], fontsize=10)
    ax.set_title(f"{y_label} vs {x_label}", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="lower right")


# =========================================================================
# 9. Detection scatter (n vs s)
# =========================================================================
def plot_detection_scatter(fig: Figure, ax, val_summary):
    """n_det vs s_det per frame with y=x reference line.

    Takes a ValidationSummary object.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    n_dets = np.array(getattr(val_summary, "n_det_counts", []), dtype=float)
    s_dets = np.array(getattr(val_summary, "s_det_counts", []), dtype=float)
    if len(n_dets) == 0 or len(s_dets) == 0:
        ax.set_title("No validation data", color=tc["fg"])
        return
    # Truncate to common length
    n = min(len(n_dets), len(s_dets))
    n_dets, s_dets = n_dets[:n], s_dets[:n]

    ax.scatter(n_dets, s_dets, s=8, alpha=0.4, color="#2196f3", edgecolors="none")

    # y=x reference line
    lim_max = max(n_dets.max(), s_dets.max(), 1)
    ax.plot([0, lim_max], [0, lim_max], color=tc["fg"], linestyle="--",
            linewidth=1.0, alpha=0.6, label="y = x")

    r, p = pearsonr(n_dets, s_dets)
    ax.annotate(f"r = {r:.3f}, p = {p:.2e}", xy=(0.02, 0.95),
                xycoords="axes fraction", fontsize=9, color=tc["fg"])

    ax.set_xlabel("YOLOv8n Detections", color=tc["fg"], fontsize=10)
    ax.set_ylabel("YOLOv8s Detections", color=tc["fg"], fontsize=10)
    ax.set_title("Detection Count: n vs s", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="lower right")


# =========================================================================
# 10. Confidence comparison
# =========================================================================
def plot_confidence_comparison(fig: Figure, ax, val_summary):
    """n_mean_conf vs s_mean_conf scatter with y=x reference."""
    tc = theme_colors()
    apply_dark(fig, [ax])

    n_conf = np.array(getattr(val_summary, "n_mean_confs", []), dtype=float)
    s_conf = np.array(getattr(val_summary, "s_mean_confs", []), dtype=float)
    if len(n_conf) == 0 or len(s_conf) == 0:
        ax.set_title("No validation data", color=tc["fg"])
        return
    # Truncate to common length
    n = min(len(n_conf), len(s_conf))
    n_conf, s_conf = n_conf[:n], s_conf[:n]

    # Filter out frames where both are zero (no detections)
    mask = (n_conf > 0) | (s_conf > 0)
    n_conf, s_conf = n_conf[mask], s_conf[mask]
    if len(n_conf) == 0:
        ax.set_title("No detections in validation data", color=tc["fg"])
        return

    ax.scatter(n_conf, s_conf, s=8, alpha=0.4, color="#ff9800", edgecolors="none")
    ax.plot([0, 1], [0, 1], color=tc["fg"], linestyle="--", linewidth=1.0,
            alpha=0.6, label="y = x")

    r, p = pearsonr(n_conf, s_conf)
    ax.annotate(f"r = {r:.3f}, p = {p:.2e}", xy=(0.02, 0.95),
                xycoords="axes fraction", fontsize=9, color=tc["fg"])

    ax.set_xlabel("YOLOv8n Mean Confidence", color=tc["fg"], fontsize=10)
    ax.set_ylabel("YOLOv8s Mean Confidence", color=tc["fg"], fontsize=10)
    ax.set_title("Confidence Comparison: n vs s", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="lower right")


# =========================================================================
# 11. Threshold sweep heatmap
# =========================================================================
def plot_threshold_heatmap(fig: Figure, ax, summary,
                           metric: str = "sw_per_100",
                           c_low_range: Tuple[float, float, float] = (0.1, 0.8, 0.05),
                           c_high_range: Tuple[float, float, float] = (0.2, 0.9, 0.05)):
    """Simulate switching at different c_low/c_high pairs on recorded C_trace.

    metric: 'sw_per_100' or 's_usage_pct'.
    Produces a heatmap where each cell is the result of simulated
    hysteresis switching with that (c_low, c_high) pair.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    if not getattr(summary, "C_trace", None):
        ax.set_title("No C_trace data", color=tc["fg"])
        return

    c_lows = np.arange(c_low_range[0], c_low_range[1] + 1e-9, c_low_range[2])
    c_highs = np.arange(c_high_range[0], c_high_range[1] + 1e-9, c_high_range[2])

    grid = np.full((len(c_highs), len(c_lows)), np.nan)

    for i, ch in enumerate(c_highs):
        for j, cl in enumerate(c_lows):
            if cl >= ch:
                continue
            sw, s_pct = _simulate_hysteresis(summary.C_trace, cl, ch)
            grid[i, j] = sw if metric == "sw_per_100" else s_pct

    # Plot with origin at lower-left
    cmap_name = "viridis" if metric == "sw_per_100" else "magma"
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap_name,
                   extent=[c_lows[0], c_lows[-1], c_highs[0], c_highs[-1]])
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar_label = "Switches / 100 frames" if metric == "sw_per_100" else "s-model Usage (%)"
    cbar.set_label(cbar_label, color=tc["fg"], fontsize=9)
    cbar.ax.tick_params(colors=tc["fg"], labelsize=8)

    ax.set_xlabel(r"$c_{\mathrm{low}}$", color=tc["fg"], fontsize=10)
    ax.set_ylabel(r"$c_{\mathrm{high}}$", color=tc["fg"], fontsize=10)
    title_metric = "Switching Rate" if metric == "sw_per_100" else "s-Model Usage"
    ax.set_title(f"Threshold Sweep: {title_metric}", color=tc["fg"], fontsize=12)


# =========================================================================
# 12. Pareto front
# =========================================================================
def plot_pareto_front(fig: Figure, ax, summary,
                      c_low_range: Tuple[float, float, float] = (0.1, 0.8, 0.05),
                      c_high_range: Tuple[float, float, float] = (0.2, 0.9, 0.05)):
    """sw_per_100 vs s_model_usage_pct trade-off scatter from threshold sweep.

    Highlights the Pareto-optimal configurations.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    if not getattr(summary, "C_trace", None):
        ax.set_title("No C_trace data", color=tc["fg"])
        return

    c_lows = np.arange(c_low_range[0], c_low_range[1] + 1e-9, c_low_range[2])
    c_highs = np.arange(c_high_range[0], c_high_range[1] + 1e-9, c_high_range[2])

    points = []  # (sw, s_pct, c_low, c_high)
    for ch in c_highs:
        for cl in c_lows:
            if cl >= ch:
                continue
            sw, s_pct = _simulate_hysteresis(summary.C_trace, cl, ch)
            points.append((sw, s_pct, cl, ch))

    if not points:
        ax.set_title("No valid threshold pairs", color=tc["fg"])
        return

    pts = np.array(points)
    sw_vals = pts[:, 0]
    s_pct_vals = pts[:, 1]

    ax.scatter(s_pct_vals, sw_vals, s=12, alpha=0.35, color="#78909c",
               edgecolors="none", label="All configs")

    # Find Pareto front: minimize both sw and s_pct
    # Sort by s_pct ascending, then keep those with decreasing sw
    order = np.argsort(s_pct_vals)
    pareto_mask = np.zeros(len(pts), dtype=bool)
    min_sw = np.inf
    for i in order:
        if sw_vals[i] < min_sw:
            pareto_mask[i] = True
            min_sw = sw_vals[i]

    pareto_s = s_pct_vals[pareto_mask]
    pareto_sw = sw_vals[pareto_mask]
    sort_idx = np.argsort(pareto_s)
    pareto_s = pareto_s[sort_idx]
    pareto_sw = pareto_sw[sort_idx]

    ax.plot(pareto_s, pareto_sw, color="#e94560", linewidth=2.0, marker="o",
            markersize=5, label="Pareto front", zorder=5)

    ax.set_xlabel("s-Model Usage (%)", color=tc["fg"], fontsize=10)
    ax.set_ylabel("Switches / 100 frames", color=tc["fg"], fontsize=10)
    ax.set_title("Pareto Front: Switching vs s-Usage", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="upper right")


# =========================================================================
# 13. Distribution with KDE
# =========================================================================
def plot_distribution_kde(fig: Figure, ax, summaries, policies_filter=None):
    """Histogram + KDE of T_total per policy, overlaid."""
    tc = theme_colors()
    apply_dark(fig, [ax])
    sums = _filter_summaries(summaries, policies_filter)

    for idx, s in enumerate(sums):
        if not s.T_total_trace:
            continue
        data = np.array(s.T_total_trace, dtype=float)
        clr = color_for(s.policy, idx)
        if np.std(data) < 1e-9:
            continue  # skip constant traces (e.g. all-zero)

        # Histogram
        ax.hist(data, bins=60, color=clr, alpha=0.25, density=True, edgecolor=clr,
                linewidth=0.3)

        # KDE overlay
        x_min, x_max = data.min(), data.max()
        margin = max((x_max - x_min) * 0.1, 1e-6)
        x_grid = np.linspace(x_min - margin, x_max + margin, 300)
        kde = _gaussian_kde_manual(data, x_grid)
        ax.plot(x_grid, kde, color=clr, linewidth=1.8, label=s.policy)

    ax.set_xlabel(r"$T_{\mathrm{total}}$ (ms)", color=tc["fg"], fontsize=10)
    ax.set_ylabel("Density", color=tc["fg"], fontsize=10)
    ax.set_title("Latency Distribution with KDE", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="upper right")


# =========================================================================
# 14. Outlier detection
# =========================================================================
def plot_outlier_detection(fig: Figure, ax, summary):
    """T_total time series with IQR-based outliers highlighted in red."""
    tc = theme_colors()
    apply_dark(fig, [ax])

    if not summary.T_total_trace:
        ax.set_title("No data", color=tc["fg"])
        return

    data = np.array(summary.T_total_trace, dtype=float)
    frames = np.array(summary.frame_indices if summary.frame_indices
                      else list(range(len(data))))

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    normal_mask = (data >= lower) & (data <= upper)
    outlier_mask = ~normal_mask

    clr = color_for(summary.policy, 0)
    ax.plot(frames, data, color=clr, linewidth=0.6, alpha=0.7, label=summary.policy)
    ax.scatter(frames[outlier_mask], data[outlier_mask], s=12, color="#e94560",
               zorder=5, label=f"Outliers ({outlier_mask.sum()})", edgecolors="none")
    ax.axhline(upper, color="#ff9800", linestyle="--", linewidth=0.8, alpha=0.7,
               label=f"Upper fence ({upper:.1f} ms)")
    ax.axhline(lower, color="#ff9800", linestyle=":", linewidth=0.8, alpha=0.7,
               label=f"Lower fence ({lower:.1f} ms)")
    ax.fill_between(frames, lower, upper, color=clr, alpha=0.05)

    ax.set_xlabel("Frame", color=tc["fg"], fontsize=10)
    ax.set_ylabel(r"$T_{\mathrm{total}}$ (ms)", color=tc["fg"], fontsize=10)
    ax.set_title("Outlier Detection (IQR Method)", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="upper right")


# =========================================================================
# 15. Full correlation matrix
# =========================================================================
def plot_full_correlation_matrix(fig: Figure, ax, summary):
    """Correlation heatmap of all numeric traces against each other."""
    tc = theme_colors()
    apply_dark(fig, [ax])

    trace_attrs = [
        ("T_total", "T_total_trace"),
        ("T_scene", "T_scene_trace"),
        ("T_ctrl", "T_ctrl_trace"),
        ("T_infer_n", "T_infer_n_trace"),
        ("T_infer_s", "T_infer_s_trace"),
        ("C", "C_trace"),
        ("L", "L_trace"),
        ("H", "H_trace"),
        ("Dwell", "dwell_trace"),
        ("Detections", "num_detections_trace"),
        ("Penalty", "penalty_trace"),
        ("Mean Conf", "mean_conf_trace"),
        ("Conf Drop", "conf_drop_trace"),
        ("NIQE", "niqe_trace"),
    ]

    # Collect traces that have data and are not constant
    labels = []
    traces = []
    for name, attr in trace_attrs:
        vals = getattr(summary, attr, [])
        if vals and len(vals) > 2:
            arr = np.array(vals, dtype=float)
            if np.std(arr) > 1e-12:
                traces.append(arr)
                labels.append(name)

    if len(traces) < 2:
        ax.set_title("Insufficient trace data", color=tc["fg"])
        return

    # Truncate to common length
    min_len = min(len(t) for t in traces)
    matrix = np.column_stack([t[:min_len] for t in traces])
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", color=tc["fg"], fontsize=9)
    cbar.ax.tick_params(colors=tc["fg"], labelsize=7)

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=tc["fg"], fontsize=7)
    ax.set_yticklabels(labels, color=tc["fg"], fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            text_color = "white" if abs(val) > 0.6 else tc["fg"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)

    ax.set_title("Correlation Matrix", color=tc["fg"], fontsize=12)


# =========================================================================
# 16. Feature importance
# =========================================================================
def plot_feature_importance(fig: Figure, ax, summaries):
    """Bar chart of |Pearson r| for each proxy signal vs binary choice.

    Proxy signals: L, H, C, mean_conf, conf_drop, niqe.
    Target: choice (0=n, 1=s).
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    signal_attrs = [
        ("L", "L_trace"),
        ("H", "H_trace"),
        ("C", "C_trace"),
        ("Mean Conf", "mean_conf_trace"),
        ("Conf Drop", "conf_drop_trace"),
        ("NIQE", "niqe_trace"),
    ]

    # Aggregate across summaries
    all_signals = {name: [] for name, _ in signal_attrs}
    all_choice = []

    for s in summaries:
        if not s.choice_trace:
            continue
        binary = _choice_to_binary(s.choice_trace)
        n = len(binary)
        all_choice.extend(binary[:n])
        for name, attr in signal_attrs:
            vals = getattr(s, attr, [])
            all_signals[name].extend(list(vals[:n]))

    if len(all_choice) < 3:
        ax.set_title("Insufficient data", color=tc["fg"])
        return

    choice_arr = np.array(all_choice, dtype=float)
    labels = []
    importances = []
    for name, _ in signal_attrs:
        sig = np.array(all_signals[name], dtype=float)
        n = min(len(sig), len(choice_arr))
        if n < 3:
            continue
        r, _ = pearsonr(sig[:n], choice_arr[:n])
        labels.append(name)
        importances.append(abs(r))

    if not labels:
        ax.set_title("Insufficient data", color=tc["fg"])
        return

    # Sort by importance
    order = np.argsort(importances)[::-1]
    labels = [labels[i] for i in order]
    importances = [importances[i] for i in order]

    x = np.arange(len(labels))
    max_imp = max(importances) if importances else 1.0
    colors = [cm.viridis(v / max_imp) if max_imp > 0
              else cm.viridis(0.5) for v in importances]
    ax.barh(x, importances, color=colors, edgecolor=tc["grid"], linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, color=tc["fg"], fontsize=9)
    ax.set_xlabel("|Pearson r| with Model Choice", color=tc["fg"], fontsize=10)
    ax.set_title("Feature Importance for Switching Decision", color=tc["fg"], fontsize=12)
    ax.invert_yaxis()


# =========================================================================
# 17. Decision boundary (L vs H colored by choice)
# =========================================================================
def plot_decision_boundary(fig: Figure, ax, summary):
    """Scatter of L vs H colored by model choice (green=n, red=s)."""
    tc = theme_colors()
    apply_dark(fig, [ax])

    if not summary.choice_trace or not summary.L_trace or not summary.H_trace:
        ax.set_title("No data", color=tc["fg"])
        return

    n = min(len(summary.L_trace), len(summary.H_trace), len(summary.choice_trace))
    L = np.array(summary.L_trace[:n], dtype=float)
    H = np.array(summary.H_trace[:n], dtype=float)
    binary = _choice_to_binary(summary.choice_trace[:n])

    mask_n = binary == 0
    mask_s = binary == 1

    ax.scatter(L[mask_n], H[mask_n], s=6, alpha=0.35, color="#4caf50",
               edgecolors="none", label="n (nano)")
    ax.scatter(L[mask_s], H[mask_s], s=6, alpha=0.35, color="#e94560",
               edgecolors="none", label="s (small)")

    ax.set_xlabel("Laplacian (L)", color=tc["fg"], fontsize=10)
    ax.set_ylabel("Entropy (H)", color=tc["fg"], fontsize=10)
    ax.set_title("Decision Boundary: L vs H", color=tc["fg"], fontsize=12)
    _legend(ax, tc, loc="upper right")


# =========================================================================
# 18. Signal lag cross-correlation
# =========================================================================
def plot_signal_lag_xcorr(fig: Figure, ax, summary,
                          signal_attr: str = "C_trace", max_lag: int = 50):
    """Cross-correlation of a proxy signal vs choice at different lag offsets.

    Positive lag means signal leads choice; negative means signal lags choice.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    signal = np.array(getattr(summary, signal_attr, []), dtype=float)
    if not summary.choice_trace or len(signal) < max_lag * 2:
        ax.set_title("Insufficient data for cross-correlation", color=tc["fg"])
        return

    n = min(len(signal), len(summary.choice_trace))
    signal = signal[:n]
    choice = _choice_to_binary(summary.choice_trace[:n])

    # Normalize
    signal = (signal - signal.mean()) / (signal.std() + 1e-9)
    choice = (choice - choice.mean()) / (choice.std() + 1e-9)

    lags = np.arange(-max_lag, max_lag + 1)
    xcorr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            xcorr[i] = np.mean(signal[:n - lag] * choice[lag:n])
        else:
            xcorr[i] = np.mean(signal[-lag:n] * choice[:n + lag])

    clr = color_for(summary.policy, 0)
    ax.bar(lags, xcorr, width=1.0, color=clr, alpha=0.7, edgecolor=clr, linewidth=0.3)
    ax.axhline(0, color=tc["fg"], linewidth=0.5, alpha=0.5)
    ax.axvline(0, color=tc["fg"], linewidth=0.5, alpha=0.5, linestyle="--")

    # Mark peak
    peak_idx = np.argmax(np.abs(xcorr))
    peak_lag = lags[peak_idx]
    peak_val = xcorr[peak_idx]
    ax.annotate(f"Peak at lag={peak_lag}\nr={peak_val:.3f}",
                xy=(peak_lag, peak_val),
                xytext=(peak_lag + max_lag * 0.2, peak_val),
                fontsize=8, color=tc["fg"],
                arrowprops=dict(arrowstyle="->", color=tc["fg"], lw=0.8))

    signal_name = signal_attr.replace("_trace", "").replace("_", " ").title()
    ax.set_xlabel("Lag (frames)", color=tc["fg"], fontsize=10)
    ax.set_ylabel("Cross-correlation", color=tc["fg"], fontsize=10)
    ax.set_title(f"Signal-Choice Cross-Correlation: {signal_name}",
                 color=tc["fg"], fontsize=12)


# =========================================================================
# Export helper
# =========================================================================
# =========================================================================
# PROXY VALIDATION — Core thesis question:
# "When L/H/C say switch to s, does s actually outperform n?"
# Uses ValidationSummary (both models on every frame) as ground truth.
# =========================================================================

def plot_proxy_vs_s_benefit(fig: Figure, ax, val_summary,
                            proxy_attr: str = "C_trace",
                            benefit_attr: str = "det_count_gaps",
                            proxy_label: str = "C-score",
                            benefit_label: str = "Extra detections (s − n)"):
    """Scatter: proxy value vs s-model benefit, with regression line.

    Shows whether high proxy values correlate with frames where s outperforms n.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    proxy = np.asarray(getattr(val_summary, proxy_attr, []), dtype=float)
    benefit = np.asarray(getattr(val_summary, benefit_attr, []), dtype=float)

    n = min(len(proxy), len(benefit))
    if n < 10:
        ax.set_title(f"Insufficient data ({n} frames)", color=tc["fg"])
        return

    proxy, benefit = proxy[:n], benefit[:n]

    # Scatter with transparency
    ax.scatter(proxy, benefit, s=6, alpha=0.25, color="#26c6da", edgecolors="none")

    # Regression line + stats
    r, p = pearsonr(proxy, benefit)
    coeffs = np.polyfit(proxy, benefit, 1)
    x_line = np.linspace(np.min(proxy), np.max(proxy), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), color="#ff9800",
            linewidth=2, label=f"r = {r:.3f}  (p = {p:.2e})")

    # Zero line
    ax.axhline(0, color=tc["fg"], linewidth=0.5, alpha=0.3, linestyle="--")

    ax.set_xlabel(proxy_label, color=tc["fg"])
    ax.set_ylabel(benefit_label, color=tc["fg"])
    ax.set_title(f"{proxy_label} vs {benefit_label}", color=tc["fg"], fontsize=10)
    _legend(ax, tc, loc="best")


def plot_conditional_s_benefit(fig: Figure, ax, val_summary, n_bins: int = 5):
    """Bar chart: bin frames by C-score, show mean s-benefit per bin.

    Proves that higher complexity → larger gap between s and n performance.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    C = np.asarray(getattr(val_summary, "C_trace", []), dtype=float)
    det_gap = np.asarray(getattr(val_summary, "det_count_gaps", []), dtype=float)
    conf_gap = np.asarray(getattr(val_summary, "mean_conf_gaps", []), dtype=float)

    n = min(len(C), len(det_gap), len(conf_gap))
    if n < 30:
        ax.set_title("Insufficient data for binned analysis", color=tc["fg"])
        return

    C, det_gap, conf_gap = C[:n], det_gap[:n], conf_gap[:n]

    # Create equal-frequency bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(C, percentiles)
    # Make edges unique
    edges = np.unique(edges)
    actual_bins = len(edges) - 1
    if actual_bins < 2:
        ax.set_title("C-score too uniform for binning", color=tc["fg"])
        return

    bin_labels = []
    det_means = []
    det_stds = []
    conf_means = []
    frames_per_bin = []

    for i in range(actual_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < actual_bins - 1:
            mask = (C >= lo) & (C < hi)
        else:
            mask = (C >= lo) & (C <= hi)
        if np.sum(mask) < 3:
            continue
        bin_labels.append(f"{lo:.2f}–{hi:.2f}")
        det_means.append(np.mean(det_gap[mask]))
        det_stds.append(np.std(det_gap[mask]) / np.sqrt(np.sum(mask)))
        conf_means.append(np.mean(conf_gap[mask]))
        frames_per_bin.append(int(np.sum(mask)))

    if not bin_labels:
        ax.set_title("No valid bins", color=tc["fg"])
        return

    x = np.arange(len(bin_labels))
    w = 0.35
    bars_det = ax.bar(x - w / 2, det_means, w, yerr=det_stds,
                       color="#4caf50", alpha=0.85, label="Extra dets (s−n)",
                       capsize=3, edgecolor="#16213e")
    bars_conf = ax.bar(x + w / 2, conf_means, w,
                        color="#26c6da", alpha=0.85, label="Conf gap (s−n)",
                        edgecolor="#16213e")

    ax.axhline(0, color=tc["fg"], linewidth=0.5, alpha=0.3, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=7)
    ax.set_xlabel("C-score bin (low → high complexity)", color=tc["fg"])
    ax.set_ylabel("Mean s − n advantage", color=tc["fg"])
    ax.set_title("Conditional s-model Benefit by Complexity Bin", color=tc["fg"],
                 fontsize=10)

    # Annotate frame counts
    for i, cnt in enumerate(frames_per_bin):
        ax.annotate(f"n={cnt}", (i, 0), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=6, color=tc["fg"],
                    alpha=0.6)

    _legend(ax, tc, loc="upper left")


def plot_proxy_correlation_table(fig: Figure, ax, val_summary):
    """Heatmap: Pearson r of each proxy vs each disagreement metric.

    The 'thesis defense table' — shows which proxies best predict s-benefit.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    proxy_names = ["L", "H", "C", "NIQE",
                   "Tenengrad", "Edge dens.", "Local contr.", "Brenner", "Color H"]
    proxy_attrs = ["L_trace", "H_trace", "C_trace", "niqe_scores",
                   "tenengrad_trace", "edge_density_trace",
                   "local_contrast_trace", "brenner_trace",
                   "color_entropy_trace"]
    # Gracefully shrink to available proxies (backward compat)
    available = [(n, a) for n, a in zip(proxy_names, proxy_attrs)
                 if len(getattr(val_summary, a, [])) >= 10]
    if available:
        proxy_names, proxy_attrs = zip(*available)
        proxy_names = list(proxy_names)
        proxy_attrs = list(proxy_attrs)
    metric_names = ["Det gap\n(s−n)", "Conf gap\n(s−n)", "Disagree\nscore",
                    "IoU\nagreement", "Extra\ndets s"]
    metric_attrs = ["det_count_gaps", "mean_conf_gaps", "disagreement_scores",
                    "iou_agreements", "extra_dets_s_trace"]

    # Build correlation matrix
    r_matrix = np.full((len(proxy_names), len(metric_names)), np.nan)
    p_matrix = np.full_like(r_matrix, np.nan)

    for i, pa in enumerate(proxy_attrs):
        pdata = np.asarray(getattr(val_summary, pa, []), dtype=float)
        if len(pdata) < 10:
            continue
        for j, ma in enumerate(metric_attrs):
            mdata = np.asarray(getattr(val_summary, ma, []), dtype=float)
            n = min(len(pdata), len(mdata))
            if n < 10:
                continue
            r, p = pearsonr(pdata[:n], mdata[:n])
            r_matrix[i, j] = r
            p_matrix[i, j] = p

    # Draw heatmap
    cmap = cm.get_cmap("RdYlGn")
    norm = Normalize(vmin=-0.5, vmax=0.5)
    im = ax.imshow(r_matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=7, color=tc["fg"])
    ax.set_yticks(range(len(proxy_names)))
    ax.set_yticklabels(proxy_names, fontsize=9, color=tc["fg"])

    # Annotate cells with r value and significance stars
    for i in range(len(proxy_names)):
        for j in range(len(metric_names)):
            r_val = r_matrix[i, j]
            p_val = p_matrix[i, j]
            if np.isnan(r_val):
                txt = "—"
            else:
                stars = ""
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                txt = f"{r_val:.3f}{stars}"
            text_color = "white" if abs(r_val) > 0.25 else tc["fg"]
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Proxy–Performance Correlation Matrix", color=tc["fg"],
                 fontsize=10)


def plot_proxy_roc_analysis(fig: Figure, ax, val_summary,
                            threshold_attr: str = "C_trace",
                            threshold_label: str = "C-score"):
    """ROC-like curve: how well does 'proxy > threshold' predict 's is better'.

    Defines 's is better' as frames where s detects more objects OR has higher
    confidence than n. Sweeps threshold from min to max and plots TPR vs FPR.
    """
    tc = theme_colors()
    apply_dark(fig, [ax])

    proxy = np.asarray(getattr(val_summary, threshold_attr, []), dtype=float)
    det_gap = np.asarray(getattr(val_summary, "det_count_gaps", []), dtype=float)
    conf_gap = np.asarray(getattr(val_summary, "mean_conf_gaps", []), dtype=float)

    n = min(len(proxy), len(det_gap), len(conf_gap))
    if n < 30:
        ax.set_title("Insufficient data for ROC", color=tc["fg"])
        return

    proxy, det_gap, conf_gap = proxy[:n], det_gap[:n], conf_gap[:n]

    # Ground truth: s is genuinely better on this frame
    s_better = ((det_gap > 0) | (conf_gap > 0.02)).astype(float)
    n_positive = np.sum(s_better)
    n_negative = n - n_positive

    if n_positive < 5 or n_negative < 5:
        ax.set_title("Not enough variation for ROC analysis", color=tc["fg"])
        return

    # Sweep thresholds
    thresholds = np.linspace(np.min(proxy), np.max(proxy), 200)
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred_positive = (proxy >= t).astype(float)
        tp = np.sum(pred_positive * s_better)
        fp = np.sum(pred_positive * (1 - s_better))
        tpr_list.append(tp / n_positive)
        fpr_list.append(fp / n_negative)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    # AUC via trapezoidal rule (sort by FPR)
    order = np.argsort(fpr)
    auc = float(np.trapz(tpr[order], fpr[order]))

    ax.plot(fpr, tpr, color="#ff9800", linewidth=2,
            label=f"{threshold_label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=tc["fg"], linewidth=0.8, alpha=0.4,
            label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", color=tc["fg"])
    ax.set_ylabel("True Positive Rate", color=tc["fg"])
    ax.set_title(f"Can '{threshold_label}' predict when s outperforms n?",
                 color=tc["fg"], fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _legend(ax, tc, loc="lower right")

    # Annotate AUC interpretation
    if auc > 0.7:
        interp = "Good predictor"
        color = "#4caf50"
    elif auc > 0.55:
        interp = "Moderate predictor"
        color = "#ff9800"
    else:
        interp = "Weak predictor"
        color = "#e94560"
    ax.text(0.55, 0.15, interp, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=color,
            bbox=dict(facecolor=tc["bg"], edgecolor=color, alpha=0.9, pad=4))


def plot_proxy_validation_dashboard(fig, val_summary):
    """Draw the full 2×2 proxy validation dashboard into *fig*.

    Returns the list of axes used.
    """
    fig.clf()
    fig.set_constrained_layout(True)
    axes = fig.subplots(2, 2)
    apply_dark(fig, list(axes.flat))

    # (a) C-score vs detection count gap
    plot_proxy_vs_s_benefit(
        fig, axes[0, 0], val_summary,
        proxy_attr="C_trace", benefit_attr="det_count_gaps",
        proxy_label="C-score", benefit_label="Extra detections (s − n)",
    )

    # (b) Conditional s-benefit by C-score bin
    plot_conditional_s_benefit(fig, axes[0, 1], val_summary)

    # (c) Proxy–performance correlation matrix
    plot_proxy_correlation_table(fig, axes[1, 0], val_summary)

    # (d) ROC analysis — can C predict when s is better?
    plot_proxy_roc_analysis(
        fig, axes[1, 1], val_summary,
        threshold_attr="C_trace", threshold_label="C-score",
    )

    return list(axes.flat)


def export_all_thesis_plots(summaries, val_summary, output_dir: str,
                            video_name: str = "", dpi: int = 300):
    """Save all thesis plots as PNGs with video_name in filenames.

    Parameters
    ----------
    summaries : list of RunSummary
        One or more run summaries to plot.
    val_summary : ValidationSummary or None
        Validation summary for detection/confidence plots.
    output_dir : str
        Directory to save PNG files into.
    video_name : str
        Prefix for filenames (sanitized automatically).
    dpi : int
        Resolution for saved figures.
    """
    import matplotlib.pyplot as plt
    # Do NOT switch backend — GUI's QtAgg backend can savefig just fine.

    os.makedirs(output_dir, exist_ok=True)
    prefix = video_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if prefix:
        prefix += "_"

    def _save(name: str, fig: Figure):
        path = os.path.join(output_dir, f"{prefix}{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    saved = []

    # 1. Latency CDF
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_latency_cdf(fig, ax, summaries)
    saved.append(_save("latency_cdf", fig))

    # 2. Latency box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_latency_boxplot(fig, ax, summaries)
    saved.append(_save("latency_boxplot", fig))

    # 3. Timing breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_timing_breakdown(fig, ax, summaries)
    saved.append(_save("timing_breakdown", fig))

    # 4. Latency time series
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_latency_timeseries(fig, ax, summaries)
    saved.append(_save("latency_timeseries", fig))

    # 5. Choice heatmap
    n_policies = max(len(summaries), 1)
    fig, axes = plt.subplots(n_policies, 1, figsize=(10, 1.2 * n_policies + 0.5),
                             squeeze=False)
    axes_flat = [axes[i, 0] for i in range(n_policies)]
    plot_choice_heatmap(fig, axes_flat, summaries)
    saved.append(_save("choice_heatmap", fig))

    # 6. Rolling switching rate
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_switching_rate(fig, ax, summaries)
    saved.append(_save("switching_rate", fig))

    # 7. Dwell histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_dwell_histogram(fig, ax, summaries)
    saved.append(_save("dwell_histogram", fig))

    # 8. Proxy scatter (C vs T_total)
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_proxy_scatter(fig, ax, summaries, "C_trace", "T_total_trace",
                       "C-score", r"$T_{\mathrm{total}}$ (ms)")
    saved.append(_save("proxy_scatter_C_vs_Ttotal", fig))

    # 9-10. Validation plots (only if val_summary available)
    if val_summary is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_detection_scatter(fig, ax, val_summary)
        saved.append(_save("detection_scatter", fig))

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_confidence_comparison(fig, ax, val_summary)
        saved.append(_save("confidence_comparison", fig))

    # 11. Threshold heatmap (use first summary)
    if summaries:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_threshold_heatmap(fig, ax, summaries[0], metric="sw_per_100")
        saved.append(_save("threshold_heatmap_sw", fig))

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_threshold_heatmap(fig, ax, summaries[0], metric="s_usage_pct")
        saved.append(_save("threshold_heatmap_usage", fig))

        # 12. Pareto front
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_pareto_front(fig, ax, summaries[0])
        saved.append(_save("pareto_front", fig))

    # 13. Distribution KDE
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_distribution_kde(fig, ax, summaries)
    saved.append(_save("distribution_kde", fig))

    # 14. Outlier detection (first summary)
    if summaries:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_outlier_detection(fig, ax, summaries[0])
        saved.append(_save("outlier_detection", fig))

        # 15. Correlation matrix
        fig, ax = plt.subplots(figsize=(9, 8))
        plot_full_correlation_matrix(fig, ax, summaries[0])
        saved.append(_save("correlation_matrix", fig))

    # 16. Feature importance
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_feature_importance(fig, ax, summaries)
    saved.append(_save("feature_importance", fig))

    # 19-22. Proxy validation plots (if val_summary available)
    if val_summary is not None:
        fig = Figure(figsize=(14, 10))
        plot_proxy_validation_dashboard(fig, val_summary)
        saved.append(_save("proxy_validation_dashboard", fig))

        # Individual plots for thesis figures
        for proxy_attr, proxy_lbl in [("L_trace", "Laplacian (L)"),
                                       ("H_trace", "Entropy (H)"),
                                       ("C_trace", "C-score")]:
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_proxy_vs_s_benefit(fig, ax, val_summary,
                                    proxy_attr=proxy_attr,
                                    benefit_attr="det_count_gaps",
                                    proxy_label=proxy_lbl,
                                    benefit_label="Extra detections (s − n)")
            safe_name = proxy_lbl.replace(" ", "_").replace("(", "").replace(")", "")
            saved.append(_save(f"proxy_vs_sbenefit_{safe_name}", fig))

        # ROC for each proxy
        for proxy_attr, proxy_lbl in [("L_trace", "L"),
                                       ("H_trace", "H"),
                                       ("C_trace", "C-score")]:
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_proxy_roc_analysis(fig, ax, val_summary,
                                    threshold_attr=proxy_attr,
                                    threshold_label=proxy_lbl)
            saved.append(_save(f"proxy_roc_{proxy_lbl.replace('-', '_')}", fig))

    # 17. Decision boundary (first summary)
    if summaries:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_decision_boundary(fig, ax, summaries[0])
        saved.append(_save("decision_boundary", fig))

        # 18. Signal lag cross-correlation
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_signal_lag_xcorr(fig, ax, summaries[0], signal_attr="C_trace")
        saved.append(_save("signal_lag_xcorr_C", fig))

    return saved
