"""Background inference worker thread for DMS-Raptor GUI."""
from __future__ import annotations

import dataclasses
import logging
from typing import List

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.config import RunConfig, InferenceParams, FrameResult, RunSummary

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Runs one or more policy evaluations in a background thread.

    For each policy the worker creates a :class:`StreamingEngine`, iterates
    its generator, and emits per-frame / per-run signals that the GUI can
    connect to for live display and final summary collection.
    """

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    frame_ready = pyqtSignal(object)          # FrameResult
    policy_finished = pyqtSignal(object)      # RunSummary
    all_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)   # (current_frame, total_frames)
    stopped_early = pyqtSignal()              # Emitted when user stops the run

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        source_path: str,
        source_type: str,
        policies: List[str],
        base_config: RunConfig,
        inf_params: InferenceParams,
        model_n_path: str,
        model_s_path: str,
        trt_export: bool = False,
        trt_precision: str = "fp16",
        save_video: bool = False,
        output_video_dir: str = "",
    ) -> None:
        super().__init__()
        self.source_path = source_path
        self.source_type = source_type
        self.policies = list(policies)
        self.base_config = base_config
        self.inf_params = inf_params
        self.model_n_path = model_n_path
        self.model_s_path = model_s_path
        self.trt_export = trt_export
        self.trt_precision = trt_precision
        self.save_video = save_video
        self.output_video_dir = output_video_dir

        self._stop_requested = False
        self._engine = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def request_stop(self) -> None:
        """Request a graceful stop of the current run."""
        self._stop_requested = True
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 – Qt override
        try:
            self._run_impl()
        except Exception as exc:
            if self._stop_requested:
                # Error caused by stop — this is expected, not an error
                logger.info("Worker stopped by user (exception during stop: %s)", exc)
                self.stopped_early.emit()
            else:
                logger.exception("InferenceWorker failed")
                self.error_occurred.emit(str(exc))
            return

        if self._stop_requested:
            self.stopped_early.emit()
        else:
            self.all_finished.emit()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------
    def _run_impl(self) -> None:
        import os as _os
        from ultralytics import YOLO
        from core.engine import StreamingEngine

        # ---- resolve device string ----------------------------------------
        dev = self.inf_params.device
        if dev.startswith("jetson"):
            # Map jetson labels to cuda:0 (standard for Jetson boards)
            idx = dev.split(":")[-1] if ":" in dev else "0"
            try:
                idx = int(idx)
            except ValueError:
                idx = 0
            self.inf_params = dataclasses.replace(self.inf_params, device=f"cuda:{idx}")
            logger.info("Jetson device mapped to %s", self.inf_params.device)

        # ---- load models (with optional TensorRT export) -----------------
        model_n = self._load_model("n", self.model_n_path)
        if self._stop_requested:
            return
        model_s = self._load_model("s", self.model_s_path)
        if self._stop_requested:
            return

        # ---- warmup both models on a dummy frame -----------------------
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for tag, mdl in [("n", model_n), ("s", model_s)]:
            if self._stop_requested:
                return
            try:
                mdl.predict(
                    dummy,
                    imgsz=self.inf_params.imgsz,
                    device=self.inf_params.device,
                    verbose=False,
                )
                logger.info("Warmup complete for model-%s", tag)
            except Exception as exc:
                logger.warning("Warmup failed for model-%s: %s", tag, exc)

        # ---- iterate over requested policies ----------------------------
        for policy in self.policies:
            if self._stop_requested:
                break

            cfg = dataclasses.replace(
                self.base_config,
                policy=policy,
                name=f"run_{policy}",
            )

            # Adaptive mode for hysteresis policies when a budget is set
            if policy == "combined_hyst" and self.base_config.latency_budget_ms > 0:
                cfg = dataclasses.replace(cfg, mode="adaptive")
            else:
                cfg = dataclasses.replace(cfg, mode="fixed")

            # Video output path per policy (includes input video name)
            if self.save_video and self.output_video_dir:
                video_stem = _os.path.splitext(
                    _os.path.basename(self.source_path)
                )[0]
                video_name = f"annotated_{video_stem}_{policy}.mp4"
                out_path = _os.path.join(self.output_video_dir, video_name)
                cfg = dataclasses.replace(
                    cfg,
                    save_annotated_video=True,
                    output_video_path=out_path,
                )

            engine = StreamingEngine(
                self.source_path,
                cfg,
                model_n,
                model_s,
                self.inf_params,
                self.source_type,
            )
            self._engine = engine

            total = engine.total_frames

            for fr in engine.run():
                if self._stop_requested:
                    break
                self.frame_ready.emit(fr)
                self.progress_updated.emit(fr.frame_idx, total)

            # Emit summary even if stopped early
            try:
                summary = engine.get_summary()
                if summary is not None:
                    self.policy_finished.emit(summary)
            except Exception as exc:
                logger.warning("Could not get summary for %s: %s", policy, exc)

            self._engine = None

    # ------------------------------------------------------------------
    # TensorRT / model loading helpers
    # ------------------------------------------------------------------
    def _load_model(self, tag: str, pt_path: str):
        """Load a YOLO model, optionally exporting to TensorRT first."""
        import os as _os
        from ultralytics import YOLO

        logger.info("Loading YOLO-%s model: %s", tag, pt_path)
        model = YOLO(pt_path)

        if not self.trt_export:
            return model

        dev = self.inf_params.device
        if dev == "cpu":
            logger.info("TRT export skipped for CPU device")
            return model

        # Derive .engine path
        base, _ = _os.path.splitext(pt_path)
        engine_path = f"{base}.engine"

        if _os.path.isfile(engine_path):
            logger.info("Reusing existing TRT engine: %s", engine_path)
            return YOLO(engine_path)

        # Export to TensorRT
        logger.info(
            "Exporting model-%s to TensorRT (%s) — this may take a few minutes...",
            tag, self.trt_precision,
        )
        try:
            half = self.trt_precision in ("fp16",)
            int8 = self.trt_precision == "int8"
            model.export(
                format="engine",
                imgsz=self.inf_params.imgsz,
                half=half,
                int8=int8,
                device=dev,
            )
            if _os.path.isfile(engine_path):
                logger.info("TRT export complete: %s", engine_path)
                return YOLO(engine_path)
            else:
                logger.warning("TRT export did not produce %s, using .pt", engine_path)
                return model
        except Exception as exc:
            logger.warning("TRT export failed for model-%s: %s — falling back to .pt", tag, exc)
            return model
