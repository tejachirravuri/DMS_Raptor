"""Shared plotting helpers for DMS-Raptor GUI tabs."""
from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Policy colour palette
# ---------------------------------------------------------------------------
POLICY_COLORS = {
    # Live demonstrator policies (core/engine.py)
    "n_only":              "#4caf50",
    "s_only":              "#e94560",
    "entropy_only":        "#9c27b0",
    "combined":            "#ff9800",
    "combined_hyst":       "#00bcd4",
    "conf_ema":            "#ffeb3b",
    "niqe_switch":         "#2196f3",
    "multi_proxy":         "#66bb6a",
    # Thesis analysis only (core/analysis_engine.py)
    "local_contrast_hyst": "#42a5f5",
    # Legacy (kept for loading old history)
    "combined_inv":        "#78909c",
    "combined_hyst_inv":   "#8d6e63",
    "ema_switch":          "#26c6da",
}

_FALLBACK_COLORS = [
    "#e91e63", "#00e676", "#ff6d00", "#aa00ff",
    "#76ff03", "#d500f9", "#ffd600", "#00b0ff",
]

# Consistent timing component colours across all tabs.
# Rule: T_infer_n = GREEN (n model), T_infer_s = RED (s model).
# T_scene and T_ctrl use neutral colours that do NOT clash with n/s.
TIMING_COLORS = {
    "T_scene":   "#26c6da",   # teal  — scene proxy computation
    "T_ctrl":    "#ff9800",   # orange — controller overhead
    "T_infer_n": "#4caf50",   # green — YOLOv8n inference  (n = green)
    "T_infer_s": "#e94560",   # red   — YOLOv8s inference  (s = red)
}


def color_for(policy: str, index: int = 0) -> str:
    """Return a hex colour for *policy*, falling back to index-based palette."""
    if policy in POLICY_COLORS:
        return POLICY_COLORS[policy]
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


# ---------------------------------------------------------------------------
# Theme state and colours for matplotlib figures
# ---------------------------------------------------------------------------
_current_theme = "dark"

_DARK = {
    "bg": "#181c24", "fg": "#cccccc", "grid": "#2a2e38",
    "legend_bg": "#16213e", "legend_edge": "#2a3a5c",
}
_LIGHT = {
    "bg": "#ffffff", "fg": "#333333", "grid": "#e0e0e0",
    "legend_bg": "#f5f5f5", "legend_edge": "#cccccc",
}


def set_theme(name: str) -> None:
    """Set the current matplotlib plot theme ('dark' or 'light')."""
    global _current_theme
    _current_theme = name


def get_theme() -> str:
    """Return the current theme name."""
    return _current_theme


def theme_colors() -> dict:
    """Return the colour dict for the active theme."""
    return _DARK if _current_theme == "dark" else _LIGHT


def apply_dark(fig: Figure, axes=None) -> None:
    """Apply current theme colours to a matplotlib *fig* and its axes."""
    tc = theme_colors()
    fig.patch.set_facecolor(tc["bg"])
    if axes is None:
        axes = fig.get_axes()
    if hasattr(axes, "flat"):  # numpy ndarray / flatiter
        axes = list(axes.flat) if hasattr(axes, "shape") else list(axes)
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(tc["bg"])
        ax.tick_params(colors=tc["fg"], labelsize=8)
        ax.xaxis.label.set_color(tc["fg"])
        ax.yaxis.label.set_color(tc["fg"])
        ax.title.set_color(tc["fg"])
        for spine in ax.spines.values():
            spine.set_color(tc["grid"])
        ax.grid(True, color=tc["grid"], linewidth=0.5, alpha=0.6)
