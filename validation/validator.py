"""
Dual-model proxy validation engine for DMS-Raptor.

Runs BOTH YOLOv8n and YOLOv8s on every frame and computes
inter-model disagreement metrics alongside scene complexity proxies.
This provides an independent ground-truth for whether L, H, C
actually predict when the heavier model is needed.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from core.config import InferenceParams
from core.engine import (
    complexity_proxies_fast,
    run_model_single,
    nms_xyxy,
    box_iou_xyxy,
    FastRollingPercentile,
    IMAGE_EXTS,
)
from core.niqe import compute_niqe_score


# ======================================================================
# Extended proxy computation (for validation / proxy selection study)
# ======================================================================
def compute_extended_proxies(gray: np.ndarray) -> Dict[str, float]:
    """Compute additional scene complexity proxies on a grayscale image.

    These are used in the validation framework to evaluate which proxies
    best correlate with model detection performance gaps.  Only L and H
    are used in real-time deployment; the others are for ablation study.

    Returns dict with keys: tenengrad, edge_density, local_contrast,
    color_entropy (color_entropy requires the BGR image, handled separately).
    """
    # Tenengrad: mean of squared Sobel gradients (focus/sharpness)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float((gx ** 2 + gy ** 2).mean())

    # Edge density: fraction of pixels that are Canny edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.astype(np.float32).mean() / 255.0)

    # Local contrast (RMS): standard deviation of pixel intensities
    local_contrast = float(gray.astype(np.float32).std())

    # Brenner gradient: mean of squared difference between adjacent pixels
    shifted = np.zeros_like(gray, dtype=np.float32)
    shifted[:, :-2] = gray[:, 2:].astype(np.float32)
    brenner = float(((shifted - gray.astype(np.float32)) ** 2).mean())

    return {
        "tenengrad": tenengrad,
        "edge_density": edge_density,
        "local_contrast": local_contrast,
        "brenner": brenner,
    }


def compute_color_entropy(img_bgr_small: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy on the joint color histogram (H, S channels of HSV)."""
    hsv = cv2.cvtColor(img_bgr_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins],
                        [0, 180, 0, 256]).astype(np.float32).ravel()
    s = hist.sum()
    if s <= 0:
        return 0.0
    p = hist[hist > 1e-12] / s
    return float(-(p * np.log2(p)).sum())


# ======================================================================
# Per-frame result
# ======================================================================
@dataclass
class ValidationFrameResult:
    """Metrics for a single frame from dual-model validation."""
    frame_idx: int = 0

    # Scene proxies (computed at multiple settings)
    L: float = 0.0
    H: float = 0.0
    C: float = 0.0  # alpha=0.6 default

    # Extended proxies (for proxy selection ablation study)
    tenengrad: float = 0.0
    edge_density: float = 0.0
    local_contrast: float = 0.0
    brenner: float = 0.0
    color_entropy: float = 0.0

    # Multi-resolution proxy values for sensitivity analysis
    L_settings: Dict[str, float] = field(default_factory=dict)
    H_settings: Dict[str, float] = field(default_factory=dict)
    C_settings: Dict[str, float] = field(default_factory=dict)

    # Model-n outputs
    n_det_count: int = 0
    n_mean_conf: float = 0.0
    n_max_conf: float = 0.0
    n_latency_ms: float = 0.0

    # Model-s outputs
    s_det_count: int = 0
    s_mean_conf: float = 0.0
    s_max_conf: float = 0.0
    s_latency_ms: float = 0.0

    # Disagreement metrics
    det_count_gap: int = 0           # s_det - n_det
    mean_conf_gap: float = 0.0       # s_mean_conf - n_mean_conf
    max_conf_gap: float = 0.0        # s_max_conf - n_max_conf
    iou_agreement: float = 0.0       # mean IoU of matched boxes
    extra_dets_s: int = 0            # detections only s found
    extra_dets_n: int = 0            # detections only n found (potential FPs)
    disagreement_score: float = 0.0  # composite disagreement metric

    # Timing
    T_proxy_ms: float = 0.0          # time to compute all proxy settings


# ======================================================================
# Validation summary
# ======================================================================
@dataclass
class ValidationSummary:
    """Aggregated validation results across all frames."""
    video: str = ""
    total_frames: int = 0
    alpha: float = 0.6

    # Proxy settings tested
    proxy_settings: List[str] = field(default_factory=list)

    # Per-frame data (for correlation analysis)
    frame_indices: List[int] = field(default_factory=list)
    L_trace: List[float] = field(default_factory=list)
    H_trace: List[float] = field(default_factory=list)
    C_trace: List[float] = field(default_factory=list)

    # Multi-setting proxy traces
    L_traces: Dict[str, List[float]] = field(default_factory=dict)
    H_traces: Dict[str, List[float]] = field(default_factory=dict)
    C_traces: Dict[str, List[float]] = field(default_factory=dict)

    # Model outputs
    n_det_counts: List[int] = field(default_factory=list)
    s_det_counts: List[int] = field(default_factory=list)
    n_mean_confs: List[float] = field(default_factory=list)
    s_mean_confs: List[float] = field(default_factory=list)
    n_max_confs: List[float] = field(default_factory=list)
    s_max_confs: List[float] = field(default_factory=list)
    n_latencies: List[float] = field(default_factory=list)
    s_latencies: List[float] = field(default_factory=list)

    # NIQE scores
    niqe_scores: List[float] = field(default_factory=list)

    # Extended proxy traces (for proxy selection ablation)
    tenengrad_trace: List[float] = field(default_factory=list)
    edge_density_trace: List[float] = field(default_factory=list)
    local_contrast_trace: List[float] = field(default_factory=list)
    brenner_trace: List[float] = field(default_factory=list)
    color_entropy_trace: List[float] = field(default_factory=list)

    # Disagreement traces
    det_count_gaps: List[int] = field(default_factory=list)
    mean_conf_gaps: List[float] = field(default_factory=list)
    max_conf_gaps: List[float] = field(default_factory=list)
    iou_agreements: List[float] = field(default_factory=list)
    extra_dets_s_trace: List[int] = field(default_factory=list)
    extra_dets_n_trace: List[int] = field(default_factory=list)
    disagreement_scores: List[float] = field(default_factory=list)


# ======================================================================
# Hungarian-style box matching
# ======================================================================
def _match_detections(
    dets_a: List[np.ndarray],
    dets_b: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy bipartite matching of detections by IoU.

    Returns:
        matched: list of (idx_a, idx_b, iou) tuples
        unmatched_a: indices in dets_a with no match
        unmatched_b: indices in dets_b with no match
    """
    if not dets_a or not dets_b:
        return [], list(range(len(dets_a))), list(range(len(dets_b)))

    # Build IoU matrix
    n_a, n_b = len(dets_a), len(dets_b)
    iou_matrix = np.zeros((n_a, n_b), dtype=np.float32)
    for i in range(n_a):
        for j in range(n_b):
            iou_matrix[i, j] = box_iou_xyxy(dets_a[i][:4], dets_b[j][:4])

    matched: List[Tuple[int, int, float]] = []
    used_a = set()
    used_b = set()

    # Greedy: pick highest IoU pairs first
    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        i, j = int(idx[0]), int(idx[1])
        matched.append((i, j, float(max_iou)))
        used_a.add(i)
        used_b.add(j)
        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    unmatched_a = [i for i in range(n_a) if i not in used_a]
    unmatched_b = [j for j in range(n_b) if j not in used_b]
    return matched, unmatched_a, unmatched_b


# ======================================================================
# Proxy Validator
# ======================================================================
# Default sensitivity settings
DEFAULT_PROXY_SETTINGS = [
    {"proxy_size": 160, "hist_bins": 64},
    {"proxy_size": 320, "hist_bins": 64},
    {"proxy_size": 160, "hist_bins": 128},
    {"proxy_size": 320, "hist_bins": 128},
    {"proxy_size": 640, "hist_bins": 256},   # "gold standard"
]


class ProxyValidator:
    """
    Dual-model validation engine.

    Runs both YOLOv8n and YOLOv8s on every frame and computes
    inter-model disagreement alongside scene complexity proxies
    at multiple resolution/bin settings.
    """

    def __init__(
        self,
        source_path: str,
        model_n,
        model_s,
        inf: InferenceParams,
        source_type: str = "video",
        alpha: float = 0.6,
        proxy_settings: Optional[List[Dict]] = None,
        iou_match_threshold: float = 0.5,
        conf_threshold: float = 0.25,
    ):
        self.source_path = source_path
        self.model_n = model_n
        self.model_s = model_s
        self.inf = inf
        self.source_type = source_type
        self.alpha = alpha
        self.proxy_settings = proxy_settings or DEFAULT_PROXY_SETTINGS
        self.iou_match_threshold = iou_match_threshold
        self.conf_threshold = conf_threshold
        self._stop_flag = False

        # Pre-compute total frames
        if source_type == "images":
            self._image_paths = self._list_images(source_path)
            self._n_total = len(self._image_paths)
        elif source_type == "stream":
            self._image_paths: List[str] = []
            self._n_total = 0
        else:
            self._image_paths = []
            cap = cv2.VideoCapture(source_path)
            self._n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

        if inf.max_frames > 0 and self._n_total > 0:
            self._n_total = min(self._n_total, inf.max_frames * max(1, inf.stride))
        if inf.stride > 1 and self._n_total > 0:
            self._n_total = self._n_total // inf.stride

        # Accumulators
        self._summary: Optional[ValidationSummary] = None

    @property
    def total_frames(self) -> int:
        return max(1, self._n_total) if self._n_total > 0 else 0

    def stop(self):
        self._stop_flag = True

    @staticmethod
    def _list_images(folder: str) -> List[str]:
        paths = []
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(folder, f))
        return paths

    def _frame_source(self):
        """Yield (raw_index, frame_bgr)."""
        stride = self.inf.stride
        if self.source_type == "images":
            for idx, path in enumerate(self._image_paths):
                if stride > 1 and (idx % stride) != 0:
                    continue
                img = cv2.imread(path)
                if img is not None:
                    yield idx, img
        else:
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source: {self.source_path}")
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if stride > 1 and (idx % stride) != 0:
                    idx += 1
                    continue
                yield idx, frame
                idx += 1
            cap.release()

    # ------------------------------------------------------------------
    def run(self) -> Generator[ValidationFrameResult, None, None]:
        """
        Yield one ValidationFrameResult per frame.

        For each frame:
        1. Compute scene proxies at all configured settings
        2. Run model-n inference
        3. Run model-s inference
        4. Compute disagreement metrics
        """
        summary = ValidationSummary(
            video=os.path.basename(self.source_path),
            alpha=self.alpha,
        )

        # Init per-setting trace containers and rolling percentiles
        setting_keys = []
        rolling_L: Dict[str, FastRollingPercentile] = {}
        rolling_H: Dict[str, FastRollingPercentile] = {}
        for s in self.proxy_settings:
            key = f"ps{s['proxy_size']}_hb{s['hist_bins']}"
            setting_keys.append(key)
            summary.L_traces[key] = []
            summary.H_traces[key] = []
            summary.C_traces[key] = []
            rolling_L[key] = FastRollingPercentile(200)
            rolling_H[key] = FastRollingPercentile(200)
        summary.proxy_settings = list(setting_keys)

        kept = 0
        for _raw_i, frame in self._frame_source():
            if self._stop_flag:
                break

            fr = ValidationFrameResult(frame_idx=kept)

            # --- 1) SCENE PROXIES (all settings) ---
            t0 = time.perf_counter()
            for s, key in zip(self.proxy_settings, setting_keys):
                L, H = complexity_proxies_fast(
                    frame, downsample=8,
                    proxy_size=s["proxy_size"],
                    hist_bins=s["hist_bins"],
                )

                # Rolling percentile normalization (same as real engine)
                rolling_L[key].add(L)
                rolling_H[key].add(H)

                L_lo = rolling_L[key].percentile(0)
                L_hi = rolling_L[key].percentile(100)
                H_lo = rolling_H[key].percentile(0)
                H_hi = rolling_H[key].percentile(100)

                Ln = float(np.clip((L - L_lo) / (L_hi - L_lo + 1e-9), 0.0, 1.0))
                Hn = float(np.clip((H - H_lo) / (H_hi - H_lo + 1e-9), 0.0, 1.0))
                C = float(self.alpha * Ln + (1.0 - self.alpha) * Hn)

                fr.L_settings[key] = L
                fr.H_settings[key] = H
                fr.C_settings[key] = C
                summary.L_traces[key].append(L)
                summary.H_traces[key].append(H)
                summary.C_traces[key].append(C)

            # Extended proxies (ablation study — same 160x160 grayscale)
            proxy_size_primary = self.proxy_settings[0]["proxy_size"]
            if proxy_size_primary > 0:
                img_ext = cv2.resize(frame, (proxy_size_primary, proxy_size_primary),
                                     interpolation=cv2.INTER_AREA)
            else:
                img_ext = frame
            gray_ext = cv2.cvtColor(img_ext, cv2.COLOR_BGR2GRAY)
            ext = compute_extended_proxies(gray_ext)
            fr.tenengrad = ext["tenengrad"]
            fr.edge_density = ext["edge_density"]
            fr.local_contrast = ext["local_contrast"]
            fr.brenner = ext["brenner"]
            fr.color_entropy = compute_color_entropy(img_ext)

            # NIQE score
            niqe_val = compute_niqe_score(frame, proxy_size=160)

            # Primary proxy (first setting — typically 160/64)
            primary_key = setting_keys[0]
            fr.L = fr.L_settings[primary_key]
            fr.H = fr.H_settings[primary_key]
            fr.C = fr.C_settings[primary_key]
            fr.T_proxy_ms = (time.perf_counter() - t0) * 1000.0

            # --- 2) MODEL-N INFERENCE ---
            t0 = time.perf_counter()
            dets_n = run_model_single(
                self.model_n, frame, self.inf.imgsz,
                self.inf.device, self.inf.conf_min, self.inf.iou_nms,
            )
            fr.n_latency_ms = (time.perf_counter() - t0) * 1000.0

            # Filter by confidence threshold for metric computation
            dets_n_filtered = [d for d in dets_n if float(d[4]) >= self.conf_threshold]
            fr.n_det_count = len(dets_n_filtered)
            confs_n = [float(d[4]) for d in dets_n_filtered]
            fr.n_mean_conf = float(np.mean(confs_n)) if confs_n else 0.0
            fr.n_max_conf = float(max(confs_n)) if confs_n else 0.0

            # --- 3) MODEL-S INFERENCE ---
            t0 = time.perf_counter()
            dets_s = run_model_single(
                self.model_s, frame, self.inf.imgsz,
                self.inf.device, self.inf.conf_min, self.inf.iou_nms,
            )
            fr.s_latency_ms = (time.perf_counter() - t0) * 1000.0

            dets_s_filtered = [d for d in dets_s if float(d[4]) >= self.conf_threshold]
            fr.s_det_count = len(dets_s_filtered)
            confs_s = [float(d[4]) for d in dets_s_filtered]
            fr.s_mean_conf = float(np.mean(confs_s)) if confs_s else 0.0
            fr.s_max_conf = float(max(confs_s)) if confs_s else 0.0

            # --- 4) DISAGREEMENT METRICS ---
            fr.det_count_gap = fr.s_det_count - fr.n_det_count
            fr.mean_conf_gap = fr.s_mean_conf - fr.n_mean_conf
            fr.max_conf_gap = fr.s_max_conf - fr.n_max_conf

            # Box matching
            matched, unmatched_n, unmatched_s = _match_detections(
                dets_n_filtered, dets_s_filtered, self.iou_match_threshold
            )
            fr.extra_dets_s = len(unmatched_s)
            fr.extra_dets_n = len(unmatched_n)

            if matched:
                fr.iou_agreement = float(np.mean([m[2] for m in matched]))
            else:
                fr.iou_agreement = 0.0 if (dets_n_filtered or dets_s_filtered) else 1.0

            # Composite disagreement: normalize and combine
            # Higher = more disagreement = scene was harder
            fr.disagreement_score = self._compute_disagreement(fr)

            # --- 5) ACCUMULATE ---
            summary.frame_indices.append(kept)
            summary.L_trace.append(fr.L)
            summary.H_trace.append(fr.H)
            summary.C_trace.append(fr.C)
            summary.niqe_scores.append(niqe_val)
            summary.tenengrad_trace.append(fr.tenengrad)
            summary.edge_density_trace.append(fr.edge_density)
            summary.local_contrast_trace.append(fr.local_contrast)
            summary.brenner_trace.append(fr.brenner)
            summary.color_entropy_trace.append(fr.color_entropy)
            summary.n_det_counts.append(fr.n_det_count)
            summary.s_det_counts.append(fr.s_det_count)
            summary.n_mean_confs.append(fr.n_mean_conf)
            summary.s_mean_confs.append(fr.s_mean_conf)
            summary.n_max_confs.append(fr.n_max_conf)
            summary.s_max_confs.append(fr.s_max_conf)
            summary.n_latencies.append(fr.n_latency_ms)
            summary.s_latencies.append(fr.s_latency_ms)
            summary.det_count_gaps.append(fr.det_count_gap)
            summary.mean_conf_gaps.append(fr.mean_conf_gap)
            summary.max_conf_gaps.append(fr.max_conf_gap)
            summary.iou_agreements.append(fr.iou_agreement)
            summary.extra_dets_s_trace.append(fr.extra_dets_s)
            summary.extra_dets_n_trace.append(fr.extra_dets_n)
            summary.disagreement_scores.append(fr.disagreement_score)

            yield fr

            kept += 1
            if self.inf.max_frames > 0 and kept >= self.inf.max_frames:
                break

        summary.total_frames = kept
        self._summary = summary

    def get_summary(self) -> Optional[ValidationSummary]:
        return self._summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_disagreement(fr: ValidationFrameResult) -> float:
        """
        Composite disagreement score ∈ [0, 1].

        Combines:
        - Detection count gap (normalized)
        - Confidence gap
        - 1 - IoU agreement
        - Extra detections by s
        """
        # Detection gap component (cap at 10 for normalization)
        det_gap_norm = min(abs(fr.det_count_gap), 10) / 10.0

        # Confidence gap component (already in [0,1] range roughly)
        conf_gap_norm = float(np.clip(abs(fr.mean_conf_gap), 0, 1))

        # IoU disagreement (1 - agreement)
        iou_disagree = 1.0 - fr.iou_agreement

        # Extra s detections (cap at 5)
        extra_s_norm = min(fr.extra_dets_s, 5) / 5.0

        # Weighted combination
        score = (
            0.30 * det_gap_norm
            + 0.25 * conf_gap_norm
            + 0.25 * iou_disagree
            + 0.20 * extra_s_norm
        )
        return float(np.clip(score, 0.0, 1.0))
