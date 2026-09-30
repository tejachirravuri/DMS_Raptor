"""
Streaming inference engine for DMS-Raptor.

Refactored from dms_overnight_pipeline.py to yield per-frame
FrameResult objects for live GUI consumption.
"""
from __future__ import annotations

import datetime
import os
import time
import bisect
from collections import deque
from typing import List, Tuple, Optional, Generator

import cv2
import numpy as np

from .config import RunConfig, InferenceParams, FrameResult, RunSummary
from .niqe import compute_nriqa_score


# =====================================================================
# Extended proxy computation (for multi_proxy policy)
# =====================================================================
def _compute_extended_proxies(gray: np.ndarray) -> dict:
    """Compute additional proxies: Tenengrad, EdgeDensity, LocalContrast, Brenner."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float((gx ** 2 + gy ** 2).mean())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.astype(np.float32).mean() / 255.0)
    local_contrast = float(gray.astype(np.float32).std())
    shifted = np.zeros_like(gray, dtype=np.float32)
    shifted[:, :-2] = gray[:, 2:].astype(np.float32)
    brenner = float(((shifted - gray.astype(np.float32)) ** 2).mean())
    return {
        "tenengrad": tenengrad,
        "edge_density": edge_density,
        "local_contrast": local_contrast,
        "brenner": brenner,
    }


def _compute_color_entropy(img_bgr_small: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy on the joint color histogram (H, S channels of HSV)."""
    hsv = cv2.cvtColor(img_bgr_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins],
                        [0, 180, 0, 256]).astype(np.float32).ravel()
    s = hist.sum()
    if s <= 0:
        return 0.0
    p = hist[hist > 1e-12] / s
    return float(-(p * np.log2(p)).sum())


# =====================================================================
# Rolling percentile (bounded window)
# =====================================================================
class FastRollingPercentile:
    """O(N) insert, O(1) query rolling percentile over a fixed window."""

    def __init__(self, maxlen: int):
        self.maxlen = int(maxlen)
        self.history: deque = deque(maxlen=self.maxlen)
        self.sorted_list: List[float] = []

    def add(self, val: float):
        val = float(val)
        if len(self.history) == self.maxlen:
            oldest = self.history.popleft()
            idx = bisect.bisect_left(self.sorted_list, oldest)
            if 0 <= idx < len(self.sorted_list) and self.sorted_list[idx] == oldest:
                self.sorted_list.pop(idx)
            else:
                try:
                    self.sorted_list.remove(oldest)
                except ValueError:
                    pass
        self.history.append(val)
        bisect.insort(self.sorted_list, val)

    def percentile(self, p: float) -> float:
        if not self.sorted_list:
            return 0.0
        p = float(np.clip(p, 0.0, 100.0))
        idx = int((p / 100.0) * (len(self.sorted_list) - 1))
        return float(self.sorted_list[idx])

    def __len__(self):
        return len(self.history)


# =====================================================================
# Detection helpers (used by research scripts, NOT by the main pipeline)
# =====================================================================
def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes. Used by 02_detection_quality.py."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms_xyxy(dets: List[np.ndarray], iou_th: float) -> List[np.ndarray]:
    """Python NMS utility — NOT used by the main inference pipeline.

    YOLO's .predict() applies C++/CUDA NMS internally. This function
    exists only as a utility for external analysis scripts.
    """
    if not dets:
        return []
    dets = sorted(dets, key=lambda x: float(x[4]), reverse=True)
    keep: List[np.ndarray] = []
    for d in dets:
        if all(box_iou_xyxy(d[:4], k[:4]) <= iou_th for k in keep):
            keep.append(d)
    return keep


def run_model_single(model, img_bgr: np.ndarray, imgsz: int,
                     device: str, conf: float, iou_nms: float) -> List[np.ndarray]:
    """Run a single YOLO model and return detections as list of [x1,y1,x2,y2,conf].

    NOTE: YOLO's .predict() already applies NMS internally (C++/CUDA).
    We do NOT run a second Python NMS — that was O(n^2) and made n-model
    slower than s-model due to the higher number of low-conf detections.
    """
    res = model.predict(source=img_bgr, imgsz=imgsz, device=device,
                        conf=conf, iou=iou_nms, verbose=False)[0]
    dets: List[np.ndarray] = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, cf):
            dets.append(np.array([b[0], b[1], b[2], b[3], c], dtype=np.float32))
    return dets


# =====================================================================
# Scene complexity proxies
# =====================================================================
def complexity_proxies_fast(img_bgr: np.ndarray, downsample: int,
                            proxy_size: int, hist_bins: int) -> Tuple[float, float]:
    h, w = img_bgr.shape[:2]
    if proxy_size > 0:
        img_small = cv2.resize(img_bgr, (proxy_size, proxy_size),
                               interpolation=cv2.INTER_AREA)
    else:
        img_small = cv2.resize(
            img_bgr,
            (max(32, w // max(1, downsample)), max(32, h // max(1, downsample))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    L = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    bins = int(max(16, hist_bins))
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).astype(np.float32).ravel()
    s = float(hist.sum())
    if s <= 0:
        H = 0.0
    else:
        p = hist[hist > 1e-12] / s
        H = float(-(p * np.log2(p)).sum())
    return L, H


# =====================================================================
# Overlay drawing
# =====================================================================
def draw_overlay(frame: np.ndarray, dets: List[np.ndarray], choice: str,
                 header_lines: List[str], conf_show: float,
                 mode: str = "full") -> np.ndarray:
    """Draw overlay on frame.

    mode="full"    : saved-video overlay (header panel, badge, boxes, border)
    mode="monitor" : live-feed overlay (thick border + boxes only, no text)
    """
    out = frame.copy()
    fh, fw = out.shape[:2]

    # -- Color scheme --
    if choice == "n":
        accent = (80, 175, 76)     # #4CAF50 green BGR
        model_label = "YOLOv8n"
    else:
        accent = (96, 69, 233)     # #E94560 red BGR
        model_label = "YOLOv8s"

    panel_bg = (26, 14, 10)  # #0A0E1A dark navy BGR

    # -- 1) Model border --
    if mode == "monitor":
        # Thick border for live feed (~5% of smaller dimension)
        border = max(12, min(fh, fw) // 20)
        overlay_border = out.copy()
        cv2.rectangle(overlay_border, (0, 0), (fw - 1, fh - 1), accent, border)
        cv2.addWeighted(overlay_border, 0.65, out, 0.35, 0, out)
    else:
        # Thick visible border for saved video (~2% of smaller dimension, min 8px)
        border = max(8, min(fh, fw) // 50)
        overlay_border = out.copy()
        cv2.rectangle(overlay_border, (0, 0), (fw - 1, fh - 1), accent, border)
        cv2.addWeighted(overlay_border, 0.75, out, 0.25, 0, out)

    # -- 2) Bounding boxes with filled label backgrounds --
    for d in dets:
        x1, y1, x2, y2, c = map(float, d)
        if c < conf_show:
            continue
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (ix1, iy1), (ix2, iy2), accent, 2, cv2.LINE_AA)
        if mode == "full":
            # Filled label background (only in saved video)
            label = f"{c:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            lbl_y1 = max(0, iy1 - th - baseline - 4)
            lbl_y2 = max(0, iy1)
            overlay_lbl = out.copy()
            cv2.rectangle(overlay_lbl, (ix1, lbl_y1), (ix1 + tw + 6, lbl_y2), accent, -1)
            cv2.addWeighted(overlay_lbl, 0.85, out, 0.15, 0, out)
            cv2.putText(out, label, (ix1 + 3, lbl_y2 - baseline - 1),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Monitor mode: done (border + boxes only)
    if mode == "monitor":
        return out

    # -- 3) Header panel (semi-transparent dark panel) --
    x0, y0 = border + 6, border + 6
    line_h = 24
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    box_w = min(700, fw - 2 * x0)
    box_h = line_h * (len(header_lines) + 1) + 8
    overlay_hdr = out.copy()
    cv2.rectangle(overlay_hdr, (x0, y0), (x0 + box_w, y0 + box_h), panel_bg, -1)
    cv2.addWeighted(overlay_hdr, 0.75, out, 0.25, 0, out)
    # Text lines
    for i, s in enumerate(header_lines):
        cv2.putText(out, s, (x0 + 6, y0 + 14 + (i + 1) * line_h),
                    font, font_scale, (230, 230, 230), 1, cv2.LINE_AA)

    # -- 4) Status badge (colored pill in top-right) --
    badge_font = cv2.FONT_HERSHEY_DUPLEX
    badge_scale = 0.7
    badge_thickness = 2
    (btw, bth), _ = cv2.getTextSize(model_label, badge_font, badge_scale, badge_thickness)
    badge_w = btw + 20
    badge_h = bth + 16
    bx1 = fw - badge_w - border - 8
    by1 = border + 8
    bx2 = fw - border - 8
    by2 = by1 + badge_h
    overlay_badge = out.copy()
    cv2.rectangle(overlay_badge, (bx1, by1), (bx2, by2), accent, -1)
    cv2.addWeighted(overlay_badge, 0.85, out, 0.15, 0, out)
    cv2.putText(out, model_label, (bx1 + 10, by2 - 8),
                badge_font, badge_scale, (255, 255, 255), badge_thickness, cv2.LINE_AA)

    return out


# =====================================================================
# Video / image utilities
# =====================================================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def count_video_frames(path: str) -> Tuple[int, float, int, int]:
    """Return (frame_count, fps, width, height)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return n, fps, w, h


def _list_images(folder: str) -> List[str]:
    paths = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.join(folder, f))
    return paths


def _format_file_size(size_bytes: int) -> float:
    """Return size in MB."""
    return round(size_bytes / (1024 * 1024), 2)


# =====================================================================
# Streaming engine
# =====================================================================
class StreamingEngine:
    """
    Generator-based inference engine.

    Usage::

        engine = StreamingEngine(path, cfg, model_n, model_s, inf, "video")
        for frame_result in engine.run():
            ...  # display / emit
        summary = engine.get_summary()

    Supports source_type: "video" | "stream" | "images"
    """

    def __init__(self, source_path: str, cfg: RunConfig,
                 model_n, model_s,
                 inf: InferenceParams,
                 source_type: str = "video"):
        self.source_path = source_path
        self.cfg = cfg
        self.model_n = model_n
        self.model_s = model_s
        self.inf = inf
        self.source_type = source_type
        self._stop_flag = False

        # pre-compute total frames (0 = unknown / streaming)
        if source_type == "images":
            self._image_paths = _list_images(source_path)
            self._n_total = len(self._image_paths)
        elif source_type == "stream":
            self._image_paths: List[str] = []
            self._n_total = 0
        else:  # video
            self._image_paths = []
            n, _, _, _ = count_video_frames(source_path)
            self._n_total = n

        if inf.max_frames > 0 and self._n_total > 0:
            self._n_total = min(self._n_total, inf.max_frames * max(1, inf.stride))
        if inf.stride > 1 and self._n_total > 0:
            self._n_total = self._n_total // inf.stride

        # accumulators
        self._totals: List[float] = []
        self._t_scene: List[float] = []
        self._t_ctrl: List[float] = []
        self._t_in_n: List[float] = []
        self._t_in_s: List[float] = []
        self._L_trace: List[float] = []
        self._H_trace: List[float] = []
        self._dwell_trace: List[int] = []
        self._det_trace: List[int] = []
        self._penalty_trace: List[float] = []
        self._avg_total_trace: List[float] = []
        self._slow_frames = 0
        self._switches = 0
        self._frame_indices: List[int] = []
        self._C_trace: List[float] = []
        self._choice_trace: List[str] = []
        self._c_low_trace: List[float] = []
        self._c_high_trace: List[float] = []
        self._mean_conf_trace: List[float] = []
        self._conf_drop_trace: List[float] = []
        self._niqe_trace: List[float] = []
        self._zero_det_gated_trace: List[bool] = []
        self._summary: Optional[RunSummary] = None

        # Video writer (optional)
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._output_video_path: str = ""

    # ------------------------------------------------------------------
    @property
    def total_frames(self) -> int:
        return max(1, self._n_total) if self._n_total > 0 else 0

    def stop(self):
        self._stop_flag = True

    # ------------------------------------------------------------------
    def _init_video_writer(self, frame_shape: Tuple[int, ...], fps: float) -> None:
        """Initialize the video writer for annotated output if configured."""
        if not self.cfg.save_annotated_video:
            return
        if not self.cfg.output_video_path:
            return

        h, w = frame_shape[:2]
        out_path = self.cfg.output_video_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if self._video_writer.isOpened():
            self._output_video_path = out_path
        else:
            self._video_writer = None

    def _release_video_writer(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    # ------------------------------------------------------------------
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
    def _get_source_fps(self) -> float:
        """Get FPS from video source, default 25 for images/stream."""
        if self.source_type == "video":
            cap = cv2.VideoCapture(self.source_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            return fps
        return 25.0

    # ------------------------------------------------------------------
    def run(self) -> Generator[FrameResult, None, None]:  # noqa: C901
        cfg = self.cfg
        inf = self.inf

        rolling_L = FastRollingPercentile(cfg.history_window_size)
        rolling_H = FastRollingPercentile(cfg.history_window_size)
        rolling_C = FastRollingPercentile(cfg.history_window_size)

        latency_hist: deque = deque(maxlen=cfg.latency_smooth_window)
        switch_events: deque = deque()

        last_LH: Optional[Tuple[float, float]] = None
        last_LH_i: int = -(10 ** 9)
        last_choice: Optional[str] = None
        dwell = 0
        budget_guard_left = 0
        c_ema_state: Optional[float] = None

        # conf_ema state
        last_mean_conf: float = 0.0
        conf_ema_fast: Optional[float] = None
        conf_ema_slow: Optional[float] = None

        # niqe_switch state
        niqe_ema_fast: Optional[float] = None
        niqe_ema_slow: Optional[float] = None

        # multi_proxy state — EMA pairs for each proxy (fast + slow)
        mp_ema: dict = {}  # {proxy_name: (fast_ema, slow_ema)} — initialized on first frame

        # Zero-detection gate state
        recent_det_counts: deque = deque(maxlen=max(1, cfg.zero_det_lookback))
        zero_det_gated = False  # track if current frame was gated

        writer_initialized = False
        source_fps = self._get_source_fps()

        kept = 0
        for _raw_i, frame in self._frame_source():
            if self._stop_flag:
                break

            # Init video writer on first frame (we need frame shape)
            if not writer_initialized:
                self._init_video_writer(frame.shape, source_fps)
                writer_initialized = True

            # -- 1) SCENE ANALYSIS --
            T_scene_ms = 0.0
            L: Optional[float] = None
            H: Optional[float] = None
            C: Optional[float] = None
            Hmed = 0.0

            needs_scene = cfg.policy in (
                "entropy_only", "combined", "combined_hyst",
            )
            needs_conf_ema = cfg.policy == "conf_ema"
            needs_niqe = cfg.policy == "niqe_switch"
            needs_multi = cfg.policy == "multi_proxy"

            # conf_ema: signal computed from PREVIOUS frame's detection
            # confidence.  NO image processing → T_scene stays 0.
            # EMA arithmetic is timed inside T_ctrl (below).
            conf_drop_val = 0.0

            if needs_scene:
                # ── T_scene: ONLY proxy image processing (benchmarkable) ──
                t0 = time.perf_counter()
                if (kept - last_LH_i) >= cfg.probe_every_k or last_LH is None:
                    L, H = complexity_proxies_fast(
                        frame, cfg.downsample, cfg.proxy_size, cfg.hist_bins
                    )
                    last_LH = (L, H)
                    last_LH_i = kept
                else:
                    L, H = last_LH
                T_scene_ms = (time.perf_counter() - t0) * 1000.0

            # niqe_switch: compute NR-IQA score (own block, no L/H needed)
            niqe_val_raw = 0.0
            if needs_niqe:
                # ── T_scene: ONLY NIQE image quality computation ──
                t0 = time.perf_counter()
                niqe_val_raw = compute_nriqa_score(frame, cfg.proxy_size)
                T_scene_ms = (time.perf_counter() - t0) * 1000.0

            # multi_proxy: compute ALL image-based proxies
            mp_raw: Optional[dict] = None
            if needs_multi:
                t0 = time.perf_counter()
                # Resize once, share for all proxies
                if cfg.proxy_size > 0:
                    mp_small = cv2.resize(frame, (cfg.proxy_size, cfg.proxy_size),
                                          interpolation=cv2.INTER_AREA)
                else:
                    h_, w_ = frame.shape[:2]
                    mp_small = cv2.resize(
                        frame,
                        (max(32, w_ // max(1, cfg.downsample)),
                         max(32, h_ // max(1, cfg.downsample))),
                        interpolation=cv2.INTER_AREA,
                    )
                mp_gray = cv2.cvtColor(mp_small, cv2.COLOR_BGR2GRAY)
                # L and H (same as complexity_proxies_fast)
                L = float(cv2.Laplacian(mp_gray, cv2.CV_32F).var())
                bins_ = int(max(16, cfg.hist_bins))
                hist_ = cv2.calcHist([mp_gray], [0], None, [bins_], [0, 256]).astype(np.float32).ravel()
                s_ = float(hist_.sum())
                if s_ <= 0:
                    H = 0.0
                else:
                    p_ = hist_[hist_ > 1e-12] / s_
                    H = float(-(p_ * np.log2(p_)).sum())
                # Extended proxies
                ext = _compute_extended_proxies(mp_gray)
                col_ent = _compute_color_entropy(mp_small)
                mp_raw = {
                    "L": L, "H": H,
                    "tenengrad": ext["tenengrad"],
                    "edge_density": ext["edge_density"],
                    "local_contrast": ext["local_contrast"],
                    "brenner": ext["brenner"],
                    "color_entropy": col_ent,
                }
                T_scene_ms = (time.perf_counter() - t0) * 1000.0

            # -- 2) CONTROLLER --
            # T_ctrl captures: rolling windows, percentile normalization,
            # EMA smoothing, C-score computation, threshold adaptation,
            # hysteresis, dwell guard, and policy decision.
            # T_scene (above) ONLY measures raw image processing
            # (Laplacian + Entropy, or NIQE computation).
            t0_ctrl = time.perf_counter()

            # ── conf_ema: reactive EMA on previous frame's chosen-model confidence ──
            # Placed inside T_ctrl so the arithmetic is properly timed.
            if needs_conf_ema:
                bf_c = cfg.conf_ema_fast_beta
                bs_c = cfg.conf_ema_slow_beta
                if conf_ema_fast is None:
                    conf_ema_fast = last_mean_conf
                    conf_ema_slow = last_mean_conf
                else:
                    conf_ema_fast = bf_c * last_mean_conf + (1.0 - bf_c) * conf_ema_fast
                    conf_ema_slow = bs_c * last_mean_conf + (1.0 - bs_c) * conf_ema_slow
                conf_drop_val = max(0.0, (conf_ema_slow - conf_ema_fast) / (conf_ema_slow + 1e-9))
                C = float(conf_drop_val)

            # ── Normalization & C-score for scene proxies ──
            if needs_scene:
                rolling_L.add(L)
                rolling_H.add(H)

                # ── PERCENTILE NORMALIZATION ──
                if cfg.mode == "fixed":
                    L_lo = rolling_L.percentile(0)
                    L_hi = rolling_L.percentile(100)
                    H_lo = rolling_H.percentile(0)
                    H_hi = rolling_H.percentile(100)
                else:
                    L_lo = rolling_L.percentile(cfg.norm_lo)
                    L_hi = rolling_L.percentile(cfg.norm_hi)
                    H_lo = rolling_H.percentile(cfg.norm_lo)
                    H_hi = rolling_H.percentile(cfg.norm_hi)

                Ln = float(np.clip((L - L_lo) / (L_hi - L_lo + 1e-9), 0.0, 1.0))
                Hn = float(np.clip((H - H_lo) / (H_hi - H_lo + 1e-9), 0.0, 1.0))

                C_raw = float(cfg.alpha * Ln + (1.0 - cfg.alpha) * Hn)
                if cfg.c_ema_beta <= 0.0:
                    C = C_raw
                else:
                    if c_ema_state is None:
                        c_ema_state = C_raw
                    else:
                        c_ema_state = (
                            (1.0 - cfg.c_ema_beta) * c_ema_state
                            + cfg.c_ema_beta * C_raw
                        )
                    C = float(c_ema_state)

                rolling_C.add(C)
                Hmed = rolling_H.percentile(50)

            # ── EMA smoothing for NIQE ──
            if needs_niqe:
                bf_n = cfg.niqe_fast_beta
                bs_n = cfg.niqe_slow_beta
                if niqe_ema_fast is None:
                    niqe_ema_fast = niqe_val_raw
                    niqe_ema_slow = niqe_val_raw
                else:
                    niqe_ema_fast = bf_n * niqe_val_raw + (1.0 - bf_n) * niqe_ema_fast
                    niqe_ema_slow = bs_n * niqe_val_raw + (1.0 - bs_n) * niqe_ema_slow
                niqe_rise = max(0.0, (niqe_ema_fast - niqe_ema_slow) / (niqe_ema_slow + 1e-9))
                C = float(niqe_rise)

            # ── Multi-proxy EMA-relative composite ──
            # For each proxy: track fast EMA and slow EMA.
            # drop_i = max(0, (slow_i - fast_i) / (slow_i + ε))
            #   → positive when proxy drops below baseline (complexity spike)
            # C = Σ(w_i * drop_i) / Σ(w_i)
            # This detects transitions without destroying the absolute signal.
            if needs_multi and mp_raw is not None:
                bf_mp = cfg.mp_fast_beta
                bs_mp = cfg.mp_slow_beta

                # Active proxies and their weights
                proxy_weights = {
                    "L": cfg.mp_w_laplacian,
                    "H": cfg.mp_w_entropy,
                    "tenengrad": cfg.mp_w_tenengrad,
                    "edge_density": cfg.mp_w_edge_density,
                    "local_contrast": cfg.mp_w_local_contrast,
                    "brenner": cfg.mp_w_brenner,
                    "color_entropy": cfg.mp_w_color_entropy,
                }

                weighted_drop_sum = 0.0
                w_sum = 0.0

                for pname, pval in mp_raw.items():
                    w = proxy_weights.get(pname, 0.0)
                    if w < 1e-12:
                        continue  # proxy disabled

                    pval_f = float(pval)
                    if pname not in mp_ema:
                        # Initialize both EMAs to first value
                        mp_ema[pname] = (pval_f, pval_f)
                    else:
                        fast_prev, slow_prev = mp_ema[pname]
                        fast_new = bf_mp * pval_f + (1.0 - bf_mp) * fast_prev
                        slow_new = bs_mp * pval_f + (1.0 - bs_mp) * slow_prev
                        mp_ema[pname] = (fast_new, slow_new)

                    fast_v, slow_v = mp_ema[pname]
                    # drop > 0 when fast deviates from slow (either direction)
                    # Use absolute deviation — both blur (L drops) and
                    # complexity spikes (H rises) should trigger switching
                    drop = abs(fast_v - slow_v) / (slow_v + 1e-9)
                    weighted_drop_sum += w * drop
                    w_sum += w

                if w_sum > 1e-12:
                    C = float(weighted_drop_sum / w_sum)
                else:
                    C = 0.0

                rolling_C.add(C)

            avg_T_total = (
                float(sum(latency_hist) / max(1, len(latency_hist)))
                if latency_hist
                else 0.0
            )

            penalty = 0.0
            if cfg.mode != "fixed":
                deficit = max(0.0, avg_T_total - cfg.latency_budget_ms)
                penalty = deficit * cfg.latency_penalty_factor
                if avg_T_total > cfg.latency_budget_ms * (1.0 + cfg.budget_guard_margin):
                    budget_guard_left = max(budget_guard_left, cfg.budget_guard_frames)

            have_enough = len(rolling_C) >= max(10, cfg.history_window_size // 2)

            if cfg.mode == "fixed" or not have_enough:
                c_low = cfg.c_low
                c_high = cfg.c_high
                c_mid = cfg.combined_mid
            else:
                c_mid = float(np.clip(
                    rolling_C.percentile(cfg.thr_mid) + penalty, 0.0, 1.0
                ))
                c_low = float(np.clip(
                    rolling_C.percentile(cfg.thr_lo) + penalty, 0.0, 1.0
                ))
                c_high = float(np.clip(
                    rolling_C.percentile(cfg.thr_hi) + penalty, 0.0, 1.0
                ))
                if c_low > c_high:
                    c_low, c_high = c_high, c_low

            # policy decision
            if cfg.policy == "n_only":
                choice = "n"
            elif cfg.policy == "s_only":
                choice = "s"
            elif cfg.policy == "entropy_only":
                choice = "s" if float(H or 0) >= Hmed else "n"
            elif cfg.policy == "combined":
                choice = "s" if float(C or 0) >= c_mid else "n"
            elif cfg.policy == "combined_hyst":
                if last_choice is None:
                    last_choice = "n"
                choice = (
                    ("s" if float(C or 0) >= c_high else "n")
                    if last_choice == "n"
                    else ("n" if float(C or 0) <= c_low else "s")
                )
            elif cfg.policy == "conf_ema":
                # Confidence-EMA: hysteresis on confidence drop
                if last_choice is None:
                    last_choice = "n"
                c_val = float(C or 0)
                # Warmup: first 5 frames force n (EMA needs to settle)
                if kept < 5:
                    choice = "n"
                else:
                    choice = (
                        ("s" if c_val >= cfg.conf_ema_c_high else "n")
                        if last_choice == "n"
                        else ("n" if c_val <= cfg.conf_ema_c_low else "s")
                    )
            elif cfg.policy == "niqe_switch":
                # NIQE-Switch: hysteresis on quality degradation
                if last_choice is None:
                    last_choice = "n"
                c_val = float(C or 0)
                choice = (
                    ("s" if c_val >= cfg.niqe_c_high else "n")
                    if last_choice == "n"
                    else ("n" if c_val <= cfg.niqe_c_low else "s")
                )
            elif cfg.policy == "multi_proxy":
                # Multi-proxy weighted composite: hysteresis on composite
                if last_choice is None:
                    last_choice = "n"
                c_val = float(C or 0)
                choice = (
                    ("s" if c_val >= cfg.mp_c_high else "n")
                    if last_choice == "n"
                    else ("n" if c_val <= cfg.mp_c_low else "s")
                )
            else:
                choice = "n"

            # budget guard override
            if cfg.mode != "fixed" and cfg.policy in (
                "combined_hyst",
                "conf_ema", "niqe_switch", "multi_proxy",
            ):
                if budget_guard_left > 0:
                    choice = "n"

            # -- Zero-detection gate --
            # If enabled and recent frames all had 0 detections, force n
            # to avoid wasting s-model inference on empty frames.
            # Placed BEFORE actuator bookkeeping so that switch counts,
            # dwell, and last_choice all reflect the actual emitted choice.
            zero_det_gated = False
            if (cfg.zero_det_gate
                    and cfg.policy not in ("n_only", "s_only")
                    and choice == "s"
                    and len(recent_det_counts) >= cfg.zero_det_lookback
                    and all(d == 0 for d in recent_det_counts)):
                choice = "n"
                zero_det_gated = True

            # actuator protections (switching policies only)
            if cfg.policy in (
                "entropy_only", "combined", "combined_hyst",
                "conf_ema", "niqe_switch", "multi_proxy",
            ):
                if last_choice is None:
                    last_choice = choice

                while switch_events and switch_events[0] <= kept - 100:
                    switch_events.popleft()

                requested_switch = choice != last_choice
                if requested_switch and dwell < cfg.min_dwell_frames:
                    choice = last_choice
                    requested_switch = False
                if requested_switch and len(switch_events) >= cfg.max_switches_per_100:
                    choice = last_choice
                    requested_switch = False

                if choice == last_choice:
                    dwell += 1
                else:
                    self._switches += 1
                    switch_events.append(kept)
                    dwell = 1
                    last_choice = choice
            else:
                dwell += 1

            T_ctrl_ms = (time.perf_counter() - t0_ctrl) * 1000.0

            # -- 3) INFERENCE --
            T_infer_n_ms = 0.0
            T_infer_s_ms = 0.0
            t0 = time.perf_counter()
            if choice == "n":
                dets = run_model_single(
                    self.model_n, frame, inf.imgsz, inf.device,
                    inf.conf_min, inf.iou_nms,
                )
                T_infer_n_ms = (time.perf_counter() - t0) * 1000.0
            else:
                dets = run_model_single(
                    self.model_s, frame, inf.imgsz, inf.device,
                    inf.conf_min, inf.iou_nms,
                )
                T_infer_s_ms = (time.perf_counter() - t0) * 1000.0
                self._slow_frames += 1

            T_total_ms = T_scene_ms + T_ctrl_ms + T_infer_n_ms + T_infer_s_ms
            latency_hist.append(T_total_ms)
            if budget_guard_left > 0:
                budget_guard_left -= 1

            # Post-inference: update last_mean_conf for conf_ema policy
            cur_mean_conf = 0.0
            if dets:
                cur_mean_conf = float(np.mean([float(d[4]) for d in dets]))
            if needs_conf_ema:
                last_mean_conf = cur_mean_conf

            # Update zero-detection gate history
            if cfg.zero_det_gate:
                n_above_thresh = sum(1 for d in dets if float(d[4]) >= cfg.zero_det_conf_thresh)
                recent_det_counts.append(n_above_thresh)

            # NIQE trace: raw score for trace, EMA for display
            niqe_val_trace = 0.0
            if needs_niqe:
                niqe_val_trace = float(niqe_val_raw)

            # -- 4) ANNOTATE --
            fps_inst = 1000.0 / max(1e-6, T_total_ms)
            header = [
                f"Policy={cfg.policy} | Mode={cfg.mode} | "
                f"Model={'YOLOv8n' if choice == 'n' else 'YOLOv8s'} | dwell={dwell}",
                f"FPS={fps_inst:.1f} | T_total={T_total_ms:.1f}ms | "
                f"avg={avg_T_total:.1f}ms | budget={cfg.latency_budget_ms:.1f}ms",
            ]
            if C is not None:
                if cfg.policy == "conf_ema":
                    header.append(
                        f"MeanConf={cur_mean_conf:.3f} | ConfDrop={conf_drop_val:.4f} | "
                        f"thr=[{cfg.conf_ema_c_low:.3f},{cfg.conf_ema_c_high:.3f}]"
                    )
                elif cfg.policy == "niqe_switch":
                    header.append(
                        f"NIQE={niqe_val_trace:.2f} | NiqeRise={float(C):.4f} | "
                        f"thr=[{cfg.niqe_c_low:.3f},{cfg.niqe_c_high:.3f}]"
                    )
                elif cfg.policy == "multi_proxy":
                    header.append(
                        f"C_multi={float(C):.3f} | "
                        f"thr=[{cfg.mp_c_low:.3f},{cfg.mp_c_high:.3f}] | "
                        f"T_scene={T_scene_ms:.1f}ms"
                    )
                else:
                    header.append(
                        f"L={L:.0f} H={H:.2f} C={C:.2f} | "
                        f"thr=[{c_low:.2f},{c_high:.2f}] | penalty={penalty:.2f}"
                    )
            # Monitor overlay: thick border + boxes only (no text clutter)
            vis_monitor = draw_overlay(frame, dets, choice, header, inf.conf_show, mode="monitor")

            # Write full overlay to video if enabled
            if self._video_writer is not None and self._video_writer.isOpened():
                vis_full = draw_overlay(frame, dets, choice, header, inf.conf_show, mode="full")
                self._video_writer.write(vis_full)

            # -- 5) EMIT --
            fr = FrameResult(
                frame_idx=kept,
                annotated_frame=vis_monitor,
                choice=choice,
                num_detections=len(dets),
                L=float(L) if L is not None else 0.0,
                H=float(H) if H is not None else 0.0,
                C=float(C) if C is not None else 0.0,
                c_low=float(c_low),
                c_high=float(c_high),
                penalty=float(penalty),
                avg_T_total=float(avg_T_total),
                T_scene_ms=T_scene_ms,
                T_ctrl_ms=T_ctrl_ms,
                T_infer_n_ms=T_infer_n_ms,
                T_infer_s_ms=T_infer_s_ms,
                T_total_ms=T_total_ms,
                dwell=dwell,
                budget_guard_left=budget_guard_left,
                mean_conf=cur_mean_conf,
                conf_drop=conf_drop_val,
                niqe_score=niqe_val_trace,
                zero_det_gated=zero_det_gated,
            )

            self._totals.append(T_total_ms)
            self._t_scene.append(T_scene_ms)
            self._t_ctrl.append(T_ctrl_ms)
            self._t_in_n.append(T_infer_n_ms)
            self._t_in_s.append(T_infer_s_ms)
            self._L_trace.append(float(L) if L is not None else 0.0)
            self._H_trace.append(float(H) if H is not None else 0.0)
            self._dwell_trace.append(dwell)
            self._det_trace.append(len(dets))
            self._penalty_trace.append(float(penalty))
            self._avg_total_trace.append(float(avg_T_total))
            self._frame_indices.append(kept)
            self._C_trace.append(float(C) if C is not None else 0.0)
            self._choice_trace.append(choice)
            self._c_low_trace.append(float(c_low))
            self._c_high_trace.append(float(c_high))
            self._mean_conf_trace.append(cur_mean_conf)
            self._conf_drop_trace.append(conf_drop_val)
            self._niqe_trace.append(niqe_val_trace)
            self._zero_det_gated_trace.append(zero_det_gated)

            yield fr

            kept += 1
            if inf.max_frames > 0 and kept >= inf.max_frames:
                break

        self._release_video_writer()
        self._build_summary()

    # ------------------------------------------------------------------
    def _build_summary(self):
        totals = np.asarray(self._totals, dtype=np.float32)
        n = len(totals)

        # Conditional means: mean over only frames where that model actually ran
        t_in_n_arr = np.asarray(self._t_in_n, dtype=np.float32)
        t_in_s_arr = np.asarray(self._t_in_s, dtype=np.float32)
        n_ran_mask = t_in_n_arr > 0
        s_ran_mask = t_in_s_arr > 0
        T_infer_n_cond = float(t_in_n_arr[n_ran_mask].mean()) if n_ran_mask.any() else 0.0
        T_infer_s_cond = float(t_in_s_arr[s_ran_mask].mean()) if s_ran_mask.any() else 0.0

        # Zero-detection gate activation count
        zdg_count = sum(1 for g in self._zero_det_gated_trace if g)

        # Get output video file size
        out_video_size = 0.0
        if self._output_video_path and os.path.isfile(self._output_video_path):
            out_video_size = _format_file_size(os.path.getsize(self._output_video_path))

        self._summary = RunSummary(
            video=os.path.basename(self.source_path),
            run_name=self.cfg.name,
            policy=self.cfg.policy,
            mode=self.cfg.mode,
            total_frames=n,
            T_scene_ms_mean=float(np.mean(self._t_scene)) if n else 0.0,
            T_ctrl_ms_mean=float(np.mean(self._t_ctrl)) if n else 0.0,
            T_infer_n_ms_mean=float(np.mean(self._t_in_n)) if n else 0.0,
            T_infer_s_ms_mean=float(np.mean(self._t_in_s)) if n else 0.0,
            T_infer_n_ms_cond_mean=T_infer_n_cond,
            T_infer_s_ms_cond_mean=T_infer_s_cond,
            T_total_ms_mean=float(np.mean(totals)) if n else 0.0,
            T_total_ms_p95=float(np.quantile(totals, 0.95)) if n else 0.0,
            T_total_ms_p99=float(np.quantile(totals, 0.99)) if n else 0.0,
            slow_pct=100.0 * (self._slow_frames / max(1, n)),
            switches=self._switches,
            sw_per_100=100.0 * (self._switches / max(1, n)),
            zero_det_gate_activations=zdg_count,
            run_timestamp=datetime.datetime.now().isoformat(),
            output_video_path=self._output_video_path,
            output_video_size_mb=out_video_size,
            frame_indices=list(self._frame_indices),
            T_total_trace=list(self._totals),
            T_scene_trace=list(self._t_scene),
            T_ctrl_trace=list(self._t_ctrl),
            T_infer_n_trace=list(self._t_in_n),
            T_infer_s_trace=list(self._t_in_s),
            C_trace=list(self._C_trace),
            L_trace=list(self._L_trace),
            H_trace=list(self._H_trace),
            choice_trace=list(self._choice_trace),
            c_low_trace=list(self._c_low_trace),
            c_high_trace=list(self._c_high_trace),
            dwell_trace=list(self._dwell_trace),
            num_detections_trace=list(self._det_trace),
            penalty_trace=list(self._penalty_trace),
            avg_T_total_trace=list(self._avg_total_trace),
            mean_conf_trace=list(self._mean_conf_trace),
            conf_drop_trace=list(self._conf_drop_trace),
            niqe_trace=list(self._niqe_trace),
            zero_det_gated_trace=list(self._zero_det_gated_trace),
        )

    def get_summary(self) -> Optional[RunSummary]:
        return self._summary
