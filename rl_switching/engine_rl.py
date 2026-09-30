"""
RL-based StreamingEngine — drop-in replacement for core.engine.StreamingEngine.

Uses a trained RL/bandit agent for the switching decision instead of
hand-crafted threshold policies.  Compatible with the existing GUI
and research scripts.
"""
from __future__ import annotations

import os
import time
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from core.config import RunConfig, InferenceParams, FrameResult, RunSummary
from core.engine import (
    run_model_single,
    draw_overlay,
    count_video_frames,
    _list_images,
    IMAGE_EXTS,
)
from .config import FEATURE_DIM, ACTION_N, ACTION_S
from .features import compute_all_features, FeatureNormalizer
from .agents.base import BaseAgent


class RLStreamingEngine:
    """Streaming engine that uses an RL agent for model switching.

    Drop-in replacement for StreamingEngine.  The agent replaces the
    policy decision logic; everything else (inference, overlay, timing,
    summary) is identical.

    Usage::

        from rl_switching.agents.linucb import LinUCBAgent
        agent = LinUCBAgent().load("checkpoints/linucb_agent.npz")
        normalizer = FeatureNormalizer().load("checkpoints/normalizer.npz")

        engine = RLStreamingEngine(
            source_path="video.mp4",
            cfg=cfg, model_n=model_n, model_s=model_s,
            inf=inf, agent=agent, normalizer=normalizer,
        )
        for result in engine.run():
            ...
        summary = engine.get_summary()
    """

    def __init__(
        self,
        source_path: str,
        cfg: RunConfig,
        model_n,
        model_s,
        inf: InferenceParams,
        agent: BaseAgent,
        normalizer: Optional[FeatureNormalizer] = None,
        source_type: str = "video",
    ):
        self.source_path = source_path
        self.cfg = cfg
        self.model_n = model_n
        self.model_s = model_s
        self.inf = inf
        self.agent = agent
        self.normalizer = normalizer
        self.source_type = source_type
        self._stop_flag = False

        # Frame count
        if source_type == "images":
            self._image_paths = _list_images(source_path)
            self._n_total = len(self._image_paths)
        elif source_type == "stream":
            self._image_paths: List[str] = []
            self._n_total = 0
        else:
            self._image_paths = []
            n, _, _, _ = count_video_frames(source_path)
            self._n_total = n

        if inf.max_frames > 0 and self._n_total > 0:
            self._n_total = min(self._n_total, inf.max_frames * max(1, inf.stride))
        if inf.stride > 1 and self._n_total > 0:
            self._n_total = self._n_total // inf.stride

        # Accumulators (same as StreamingEngine)
        self._totals: List[float] = []
        self._t_scene: List[float] = []
        self._t_ctrl: List[float] = []
        self._t_in_n: List[float] = []
        self._t_in_s: List[float] = []
        self._frame_indices: List[int] = []
        self._choice_trace: List[str] = []
        self._det_trace: List[int] = []
        self._mean_conf_trace: List[float] = []
        self._slow_frames = 0
        self._switches = 0
        self._summary: Optional[RunSummary] = None

        # Video writer
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._output_video_path = ""

    @property
    def total_frames(self) -> int:
        return max(1, self._n_total) if self._n_total > 0 else 0

    def stop(self):
        self._stop_flag = True

    def _frame_source(self):
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
                raise RuntimeError(f"Cannot open: {self.source_path}")
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

    def _get_source_fps(self) -> float:
        if self.source_type == "video":
            cap = cv2.VideoCapture(self.source_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            return fps
        return 25.0

    def run(self) -> Generator[FrameResult, None, None]:
        inf = self.inf
        cfg = self.cfg

        prev_conf = 0.0
        prev_n_dets = 0
        prev_action = ACTION_N
        dwell = 0
        last_choice_str: Optional[str] = None

        source_fps = self._get_source_fps()
        writer_initialized = False
        kept = 0

        for _raw_i, frame in self._frame_source():
            if self._stop_flag:
                break

            if not writer_initialized:
                if cfg.save_annotated_video and cfg.output_video_path:
                    h, w = frame.shape[:2]
                    os.makedirs(os.path.dirname(cfg.output_video_path) or ".", exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._video_writer = cv2.VideoWriter(
                        cfg.output_video_path, fourcc, source_fps, (w, h)
                    )
                    if self._video_writer.isOpened():
                        self._output_video_path = cfg.output_video_path
                    else:
                        self._video_writer = None
                writer_initialized = True

            # ── 1) Feature extraction (T_scene) ──────────────────────
            t0 = time.perf_counter()
            features = compute_all_features(
                frame,
                proxy_size=cfg.proxy_size,
                hist_bins=cfg.hist_bins,
                prev_conf=prev_conf,
                prev_n_dets=prev_n_dets,
                prev_action=prev_action,
                dwell_time=dwell,
            )
            if self.normalizer:
                features = self.normalizer.transform(features)
            T_scene_ms = (time.perf_counter() - t0) * 1000.0

            # ── 2) Agent decision (T_ctrl) ───────────────────────────
            t0_ctrl = time.perf_counter()
            action = self.agent.greedy_choose(features)
            choice = "n" if action == ACTION_N else "s"
            T_ctrl_ms = (time.perf_counter() - t0_ctrl) * 1000.0

            # Track switches
            if last_choice_str is not None and choice != last_choice_str:
                self._switches += 1
                dwell = 1
            else:
                dwell += 1
            last_choice_str = choice

            # ── 3) Inference ─────────────────────────────────────────
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

            # Update temporal features for next frame
            if dets:
                high_conf = [float(d[4]) for d in dets if float(d[4]) >= 0.25]
                prev_conf = float(np.mean(high_conf)) if high_conf else 0.0
                prev_n_dets = len(high_conf)
            else:
                prev_conf = 0.0
                prev_n_dets = 0
            prev_action = action

            # ── 4) Annotate ──────────────────────────────────────────
            header = [
                f"Policy=rl_{self.agent.name} | "
                f"Model={'YOLOv8n' if choice == 'n' else 'YOLOv8s'} | dwell={dwell}",
                f"T_total={T_total_ms:.1f}ms | T_scene={T_scene_ms:.1f}ms | "
                f"T_ctrl={T_ctrl_ms:.2f}ms",
            ]
            annotated = draw_overlay(
                frame, dets, choice, header, inf.conf_show
            )

            if self._video_writer is not None:
                self._video_writer.write(annotated)

            # Accumulate
            self._totals.append(T_total_ms)
            self._t_scene.append(T_scene_ms)
            self._t_ctrl.append(T_ctrl_ms)
            self._t_in_n.append(T_infer_n_ms)
            self._t_in_s.append(T_infer_s_ms)
            self._frame_indices.append(_raw_i)
            self._choice_trace.append(choice)
            self._det_trace.append(len(dets))
            self._mean_conf_trace.append(prev_conf)

            kept += 1

            if inf.max_frames > 0 and kept >= inf.max_frames:
                break

            yield FrameResult(
                frame_idx=_raw_i,
                annotated_frame=annotated,
                choice=choice,
                num_detections=len(dets),
                T_scene_ms=T_scene_ms,
                T_ctrl_ms=T_ctrl_ms,
                T_infer_n_ms=T_infer_n_ms,
                T_infer_s_ms=T_infer_s_ms,
                T_total_ms=T_total_ms,
                dwell=dwell,
                mean_conf=prev_conf,
            )

        if self._video_writer is not None:
            self._video_writer.release()

    def get_summary(self) -> RunSummary:
        """Build summary compatible with existing RunSummary format."""
        if self._summary is not None:
            return self._summary

        n = len(self._totals)
        totals = np.array(self._totals) if n > 0 else np.zeros(1)

        # Conditional means
        t_in_n = np.array(self._t_in_n)
        t_in_s = np.array(self._t_in_s)
        n_ran = t_in_n > 0
        s_ran = t_in_s > 0

        self._summary = RunSummary(
            video=self.source_path,
            run_name=f"rl_{self.agent.name}",
            policy=f"rl_{self.agent.name}",
            mode="fixed",
            total_frames=n,
            T_scene_ms_mean=float(np.mean(self._t_scene)) if n > 0 else 0,
            T_ctrl_ms_mean=float(np.mean(self._t_ctrl)) if n > 0 else 0,
            T_infer_n_ms_mean=float(t_in_n.mean()) if n > 0 else 0,
            T_infer_s_ms_mean=float(t_in_s.mean()) if n > 0 else 0,
            T_infer_n_ms_cond_mean=float(t_in_n[n_ran].mean()) if n_ran.any() else 0,
            T_infer_s_ms_cond_mean=float(t_in_s[s_ran].mean()) if s_ran.any() else 0,
            T_total_ms_mean=float(totals.mean()),
            T_total_ms_p95=float(np.percentile(totals, 95)),
            T_total_ms_p99=float(np.percentile(totals, 99)),
            slow_pct=self._slow_frames / max(1, n) * 100,
            switches=self._switches,
            sw_per_100=self._switches / max(1, n) * 100,
            frame_indices=self._frame_indices,
            T_total_trace=self._totals,
            T_scene_trace=self._t_scene,
            T_ctrl_trace=self._t_ctrl,
            T_infer_n_trace=self._t_in_n,
            T_infer_s_trace=self._t_in_s,
            choice_trace=self._choice_trace,
            num_detections_trace=self._det_trace,
            mean_conf_trace=self._mean_conf_trace,
        )
        return self._summary
