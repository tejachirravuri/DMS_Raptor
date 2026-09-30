"""
RL switching configuration.

All hyperparameters for bandit agents, DQN, reward shaping,
feature extraction, and training live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields, asdict
from typing import Any, Dict, List, Optional


# ── Feature set ──────────────────────────────────────────────────────
# Canonical order of features in the context vector.  Every agent,
# dataset, and evaluation script references this list so that
# feature indices are always consistent.
FEATURE_NAMES: List[str] = [
    "laplacian",          # Laplacian variance (sharpness)
    "entropy",            # Shannon entropy (texture complexity)
    "tenengrad",          # Sobel gradient energy
    "color_entropy",      # HSV joint color entropy
    "nriqa",              # NR-IQA quality score
    "brightness",         # Mean pixel intensity
    "edge_density",       # Fraction of Canny edge pixels
    "local_contrast",     # Pixel intensity std dev
    "prev_conf",          # Previous frame's mean detection confidence
    "prev_n_dets",        # Previous frame's detection count
    "prev_action",        # Previous action (0=n, 1=s)
    "dwell_time",         # Frames since last switch
]

FEATURE_DIM: int = len(FEATURE_NAMES)

# Actions
ACTION_N: int = 0   # use YOLOv8n (fast)
ACTION_S: int = 1   # use YOLOv8s (accurate)
NUM_ACTIONS: int = 2


@dataclass
class RLConfig:
    """Master configuration for RL-based switching experiments."""

    # ── Paths ────────────────────────────────────────────────────────
    # These are set at runtime by the research scripts.
    video_dir: str = ""
    model_n_path: str = ""
    model_s_path: str = ""
    output_dir: str = "research_results/rl_switching"
    data_cache_dir: str = "research_results/rl_switching/data_cache"

    # ── Inference ────────────────────────────────────────────────────
    imgsz: int = 640
    device: str = "cpu"       # "cpu", "cuda", "cuda:0", etc.
    conf_min: float = 0.001
    iou_nms: float = 0.45
    proxy_size: int = 160
    hist_bins: int = 64

    # ── Data collection ──────────────────────────────────────────────
    # Both models run on every frame during collection.
    collect_stride: int = 1   # process every Nth frame (1=all)
    max_frames: int = 0       # 0 = all frames

    # ── Reward function ──────────────────────────────────────────────
    # CALIBRATED from oracle ceiling check (batch 1, 4 videos, 27K frames):
    #   Real latency gap: ~34ms (not 140ms as in thesis theory section)
    #   n_only recall: 64-91% (oracle needs s on 9-36% of frames)
    #   conf_ema uses s on ~49% but only achieves 88% recall
    #   RL target: same/better recall with 20-30% s-usage (efficiency win)
    reward_type: str = "oracle_iou"   # "oracle_iou" | "confidence" | "hybrid"
    lambda_latency: float = 0.01      # calibrated for ~34ms gap (0.01 * 34 = 0.34 bonus)
    switch_penalty: float = 0.05      # cost for switching models
    iou_threshold: float = 0.5        # IoU threshold for n-sufficiency
    # oracle_iou rewards:
    reward_correct_n: float = 1.0     # n chosen, n was sufficient (IoU >= thr)
    reward_wasteful_s: float = 0.1    # LOWERED: penalize s-waste more (was 0.3)
    reward_needed_s: float = 1.0      # s chosen, s genuinely needed
    reward_missed_n: float = -1.0     # n chosen, s was needed (IoU < thr)

    # ── Feature normalization ────────────────────────────────────────
    normalize_features: bool = True   # z-score normalization during training
    clip_features: float = 5.0        # clip z-scored features to [-clip, clip]

    # ── LinUCB ───────────────────────────────────────────────────────
    linucb_alpha: float = 1.0         # exploration parameter
    linucb_epochs: int = 10           # passes over training data

    # ── Thompson Sampling ────────────────────────────────────────────
    thompson_v: float = 1.0           # observation noise variance
    thompson_epochs: int = 10

    # ── DQN ──────────────────────────────────────────────────────────
    dqn_hidden: int = 128             # hidden layer size
    dqn_layers: int = 2               # number of hidden layers
    dqn_lr: float = 1e-3
    dqn_gamma: float = 0.95           # discount factor
    dqn_batch_size: int = 256
    dqn_buffer_size: int = 100_000
    dqn_target_update: int = 500      # steps between target network sync
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.01
    dqn_epsilon_decay: int = 50_000   # linear decay steps
    dqn_num_episodes: int = 50        # training episodes (full video passes)
    dqn_warmup_steps: int = 1000      # fill buffer before training

    # State includes temporal features beyond the base feature vector.
    dqn_state_dim: int = FEATURE_DIM  # auto-set; don't change manually

    # ── Evaluation ───────────────────────────────────────────────────
    eval_train_videos: List[str] = field(default_factory=list)
    eval_test_videos: List[str] = field(default_factory=list)
    eval_num_seeds: int = 3           # repeated evaluation seeds

    # ── Serialization ────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for f in dc_fields(self):
            d[f.name] = getattr(self, f.name)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RLConfig":
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
