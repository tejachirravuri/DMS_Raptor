"""DMS-Raptor configuration dataclasses and constants."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field, fields as dc_fields, asdict
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "DMS-Raptor"
APP_VERSION = "1.8.0"
APP_SUBTITLE = "Dynamic Model Switching for Real-Time UAV Inspection"

# Live demonstrator policies — must match core/engine.py decision branches.
# Do NOT add policies here that engine.py cannot execute.
POLICIES = [
    "n_only", "s_only", "entropy_only", "combined", "combined_hyst",
    "conf_ema", "niqe_switch", "multi_proxy",
]
ADAPTIVE_POLICY = "combined_hyst"

# Thesis analysis policies — must match core/analysis_engine.py + core/policies.py.
# These are the 8 thesis-canonical policies evaluated in the offline pipeline.
THESIS_ANALYSIS_POLICIES = [
    "n_only", "s_only", "conf_ema", "local_contrast_hyst",
    "combined_hyst", "entropy_only", "combined", "multi_proxy",
]

INPUT_MODES = ["Video File", "Live Stream", "Image Folder"]

HISTORY_DIR_NAME = "history"


# ---------------------------------------------------------------------------
# Inference hardware / model parameters
# ---------------------------------------------------------------------------
@dataclass
class InferenceParams:
    imgsz: int = 640
    device: str = "cpu"
    conf_min: float = 0.001
    iou_nms: float = 0.45
    conf_show: float = 0.25
    max_frames: int = 0       # 0 = all
    stride: int = 1


# ---------------------------------------------------------------------------
# Controller configuration for a single policy run
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    name: str = ""
    policy: str = "combined_hyst"
    mode: str = "fixed"           # "fixed" or "adaptive"

    # scene proxy weighting
    alpha: float = 0.6

    # fixed thresholds
    c_low: float = 0.45
    c_high: float = 0.55
    combined_mid: float = 0.50

    # rolling window
    history_window_size: int = 200
    probe_every_k: int = 1
    latency_smooth_window: int = 15

    # adaptive percentile bounds
    norm_lo: float = 10.0
    norm_hi: float = 90.0
    thr_lo: float = 35.0
    thr_hi: float = 65.0
    thr_mid: float = 50.0

    # latency budget (adaptive mode)
    latency_budget_ms: float = 150.0
    latency_penalty_factor: float = 0.01
    budget_guard_margin: float = 0.10
    budget_guard_frames: int = 8

    # actuator stabilisers
    c_ema_beta: float = 0.25
    min_dwell_frames: int = 10
    max_switches_per_100: int = 12

    # proxy computation
    proxy_size: int = 160
    hist_bins: int = 64
    downsample: int = 8

    # EMA-Relative switching (LEGACY — ema_switch policy removed from public set)
    # Retained for backward compatibility with saved history JSON files.
    ema_fast_beta: float = 0.30
    ema_slow_beta: float = 0.02
    ema_c_high: float = 0.12
    ema_c_low: float = 0.04

    # Confidence-EMA switching (conf_ema policy)
    conf_ema_fast_beta: float = 0.30
    conf_ema_slow_beta: float = 0.02
    conf_ema_c_high: float = 0.12   # switch to s when confidence drops 12%
    conf_ema_c_low: float = 0.04    # switch back to n when within 4%

    # NIQE-Switch (niqe_switch policy)
    # NOTE: NIQE scores have very low variation in typical UAV inspection
    # footage (CoV ~0.2-0.5%). Thresholds must be much tighter than conf_ema.
    niqe_fast_beta: float = 0.30      # fast EMA smoothing (~3 frame response)
    niqe_slow_beta: float = 0.02      # slow EMA baseline (~50 frame response)
    niqe_c_high: float = 0.003        # switch to s when quality degrades 0.3%
    niqe_c_low: float = 0.001         # switch back to n when within 0.1%

    # Multi-Proxy weighted composite (multi_proxy policy)
    # Uses EMA-relative normalization: each proxy's deviation from its slow
    # baseline is measured, then combined with configurable weights.
    # drop_i = |fast_i - slow_i| / (slow_i + ε)
    # C = Σ(w_i * drop_i) / Σ(w_i)
    # This detects scene TRANSITIONS without destroying the absolute signal
    # like percentile normalization does (which caused C-score sign flips).
    #
    # Weights: 3-video cross-validated average Pearson |r| with Disagreement.
    # Set to 0 to disable any proxy.  Auto-normalized to sum to 1.
    mp_w_laplacian: float = 0.37      # Laplacian — avg r=0.37, consistent V1+V2
    mp_w_entropy: float = 0.29        # Shannon entropy — avg r=0.29
    mp_w_tenengrad: float = 0.38      # Sobel gradient — avg r=0.38, best cross-video
    mp_w_edge_density: float = 0.0    # Canny edges — inconsistent across videos
    mp_w_local_contrast: float = 0.0  # RMS std — V1 fluke (0.60→0.07→0.08)
    mp_w_brenner: float = 0.0         # Adjacent pixel gradient — redundant with L
    mp_w_color_entropy: float = 0.28  # HSV entropy — stable (0.31, 0.21, 0.31)
    mp_fast_beta: float = 0.30        # fast EMA smoothing (~3 frame τ)
    mp_slow_beta: float = 0.02        # slow EMA baseline (~50 frame τ)
    mp_c_high: float = 0.10           # switch to s when composite drop >= 10%
    mp_c_low: float = 0.03            # switch back to n when drop <= 3%

    # Zero-detection gate: force n-model when recent frames have no detections.
    # Rationale: if no object is visible, s-model adds ~100ms for zero benefit.
    # Uses temporal locality: empty frames tend to cluster (sky, ground, transitions).
    zero_det_gate: bool = False        # enable/disable
    zero_det_lookback: int = 3         # force n if last N frames all had 0 detections
    zero_det_conf_thresh: float = 0.25 # count detections above this confidence

    # output
    save_annotated_video: bool = False
    output_video_path: str = ""


# ---------------------------------------------------------------------------
# Per-frame result emitted by the streaming engine
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    frame_idx: int = 0
    annotated_frame: Optional[np.ndarray] = None   # BGR w/ overlay
    choice: str = "n"
    num_detections: int = 0
    L: float = 0.0
    H: float = 0.0
    C: float = 0.0
    c_low: float = 0.0
    c_high: float = 0.0
    penalty: float = 0.0
    avg_T_total: float = 0.0
    T_scene_ms: float = 0.0
    T_ctrl_ms: float = 0.0
    T_infer_n_ms: float = 0.0
    T_infer_s_ms: float = 0.0
    T_total_ms: float = 0.0
    dwell: int = 0
    budget_guard_left: int = 0
    mean_conf: float = 0.0
    conf_drop: float = 0.0
    niqe_score: float = 0.0
    zero_det_gated: bool = False


# ---------------------------------------------------------------------------
# Aggregated summary returned after a full run
# ---------------------------------------------------------------------------
@dataclass
class RunSummary:
    video: str = ""
    run_name: str = ""
    policy: str = ""
    mode: str = ""
    total_frames: int = 0

    T_scene_ms_mean: float = 0.0
    T_ctrl_ms_mean: float = 0.0
    T_infer_n_ms_mean: float = 0.0         # unconditional: mean over ALL frames (includes 0s)
    T_infer_s_ms_mean: float = 0.0         # unconditional: mean over ALL frames (includes 0s)
    T_infer_n_ms_cond_mean: float = 0.0    # conditional: mean over frames where n actually ran
    T_infer_s_ms_cond_mean: float = 0.0    # conditional: mean over frames where s actually ran
    T_total_ms_mean: float = 0.0
    T_total_ms_p95: float = 0.0
    T_total_ms_p99: float = 0.0
    slow_pct: float = 0.0
    switches: int = 0
    sw_per_100: float = 0.0
    zero_det_gate_activations: int = 0     # how many frames the gate overrode s→n

    # Metadata
    run_timestamp: str = ""     # ISO format
    output_video_path: str = ""
    output_video_size_mb: float = 0.0

    # Full per-frame traces for plotting / data exploration
    frame_indices: List[int] = field(default_factory=list)
    T_total_trace: List[float] = field(default_factory=list)
    T_scene_trace: List[float] = field(default_factory=list)
    T_ctrl_trace: List[float] = field(default_factory=list)
    T_infer_n_trace: List[float] = field(default_factory=list)
    T_infer_s_trace: List[float] = field(default_factory=list)
    C_trace: List[float] = field(default_factory=list)
    L_trace: List[float] = field(default_factory=list)
    H_trace: List[float] = field(default_factory=list)
    choice_trace: List[str] = field(default_factory=list)
    c_low_trace: List[float] = field(default_factory=list)
    c_high_trace: List[float] = field(default_factory=list)
    dwell_trace: List[int] = field(default_factory=list)
    num_detections_trace: List[int] = field(default_factory=list)
    penalty_trace: List[float] = field(default_factory=list)
    avg_T_total_trace: List[float] = field(default_factory=list)
    mean_conf_trace: List[float] = field(default_factory=list)
    conf_drop_trace: List[float] = field(default_factory=list)
    niqe_trace: List[float] = field(default_factory=list)
    zero_det_gated_trace: List[bool] = field(default_factory=list)

    # session tracking
    is_history: bool = False  # True if loaded from disk

    # ----- serialization helpers -----
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        d: Dict[str, Any] = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray):
                d[f.name] = val.tolist()
            elif isinstance(val, (np.floating, np.integer)):
                d[f.name] = float(val)
            else:
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSummary":
        """Reconstruct a RunSummary from a dict (loaded from JSON)."""
        valid = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Trace metric definitions for the Data Explorer
# ---------------------------------------------------------------------------
TRACE_METRICS = {
    "Frame Index": "frame_indices",
    "T_total (ms)": "T_total_trace",
    "T_scene (ms)": "T_scene_trace",
    "T_ctrl (ms)": "T_ctrl_trace",
    "T_infer_n (ms)": "T_infer_n_trace",
    "T_infer_s (ms)": "T_infer_s_trace",
    "C-score": "C_trace",
    "Laplacian (L)": "L_trace",
    "Entropy (H)": "H_trace",
    "c_low threshold": "c_low_trace",
    "c_high threshold": "c_high_trace",
    "Dwell": "dwell_trace",
    "Detections": "num_detections_trace",
    "Penalty": "penalty_trace",
    "Avg T_total (ms)": "avg_T_total_trace",
    "Choice (0=n, 1=s)": "choice_trace",
    "Mean Confidence": "mean_conf_trace",
    "Conf Drop (EMA)": "conf_drop_trace",
    "NR-IQA Score": "niqe_trace",
    "Zero-Det Gate Active": "zero_det_gated_trace",
}

AGGREGATE_METRICS = {
    "Mean T_total (ms)": "T_total_ms_mean",
    "P95 T_total (ms)": "T_total_ms_p95",
    "P99 T_total (ms)": "T_total_ms_p99",
    "Slow %": "slow_pct",
    "Switches / 100": "sw_per_100",
    "Total Switches": "switches",
    "Mean T_scene (ms)": "T_scene_ms_mean",
    "Mean T_ctrl (ms)": "T_ctrl_ms_mean",
    "Mean T_infer_n (ms) [unconditional]": "T_infer_n_ms_mean",
    "Mean T_infer_s (ms) [unconditional]": "T_infer_s_ms_mean",
    "Mean T_infer_n (ms) [conditional]": "T_infer_n_ms_cond_mean",
    "Mean T_infer_s (ms) [conditional]": "T_infer_s_ms_cond_mean",
    "Total Frames": "total_frames",
    "Zero-Det Gate Activations": "zero_det_gate_activations",
}
