"""Background worker for running the full thesis experiment suite."""
from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.config import RunConfig, InferenceParams, RunSummary, POLICIES, ADAPTIVE_POLICY

logger = logging.getLogger(__name__)


class ThesisExperimentWorker(QThread):
    """Run every selected policy on a single source, then optionally validate."""

    # Signals
    status_update = pyqtSignal(str)
    progress = pyqtSignal(int, int)           # current_step, total_steps
    policy_done = pyqtSignal(object)          # RunSummary
    validation_done = pyqtSignal(object)      # ValidationSummary
    all_done = pyqtSignal(list, object)       # List[RunSummary], ValidationSummary|None
    error_occurred = pyqtSignal(str)

    def __init__(self, source_path, source_type, model_n_path, model_s_path,
                 inf_params, policies, base_config, output_dir,
                 run_validation=True, save_video=False):
        super().__init__()
        self.source_path = source_path
        self.source_type = source_type
        self.model_n_path = model_n_path
        self.model_s_path = model_s_path
        self.inf_params: InferenceParams = inf_params
        self.policies: List[str] = list(policies)
        self.base_config: RunConfig = base_config
        self.output_dir = output_dir
        self.run_validation = run_validation
        self.save_video = save_video
        self._stop_requested = False

    # ------------------------------------------------------------------
    def request_stop(self):
        self._stop_requested = True

    # ------------------------------------------------------------------
    stopped = pyqtSignal()  # emitted when user-requested stop completes

    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            if not self._stop_requested:
                logger.exception("ThesisExperimentWorker failed")
                self.error_occurred.emit(str(exc))
            else:
                self.stopped.emit()
            return
        if self._stop_requested:
            self.stopped.emit()

    # ------------------------------------------------------------------
    def _run_impl(self):
        from ultralytics import YOLO
        from core.engine import StreamingEngine

        # Total steps: one per policy + optionally one for validation
        total_steps = len(self.policies) + (1 if self.run_validation else 0)
        current_step = 0

        os.makedirs(self.output_dir, exist_ok=True)

        # ---- Load models ----
        self.status_update.emit("Loading YOLO-n model...")
        model_n = YOLO(self.model_n_path)

        self.status_update.emit("Loading YOLO-s model...")
        model_s = YOLO(self.model_s_path)

        # ---- Warmup ----
        self.status_update.emit("Warming up models...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        dev = self.inf_params.device
        for mdl in (model_n, model_s):
            try:
                mdl.predict(dummy, imgsz=self.inf_params.imgsz, device=dev, verbose=False)
            except Exception:
                pass

        # ---- Derive video stem for output filenames ----
        video_stem = Path(self.source_path).stem

        # ---- Run each policy ----
        summaries: List[RunSummary] = []

        for policy in self.policies:
            if self._stop_requested:
                break

            current_step += 1
            self.status_update.emit(f"Running policy: {policy} ({current_step}/{total_steps})")
            self.progress.emit(current_step, total_steps)

            try:
                # Determine mode: adaptive only for combined_hyst with a
                # positive latency budget, otherwise fixed.
                mode = "fixed"
                if policy == ADAPTIVE_POLICY and self.base_config.latency_budget_ms > 0:
                    mode = "adaptive"

                cfg = dataclasses.replace(
                    self.base_config,
                    policy=policy,
                    name=f"thesis_{policy}",
                    mode=mode,
                )

                if self.save_video:
                    vid_out = os.path.join(
                        self.output_dir,
                        f"annotated_{video_stem}_{policy}.mp4",
                    )
                    cfg = dataclasses.replace(
                        cfg,
                        save_annotated_video=True,
                        output_video_path=vid_out,
                    )

                engine = StreamingEngine(
                    self.source_path, cfg, model_n, model_s,
                    self.inf_params, self.source_type,
                )

                gen = engine.run()
                try:
                    for fr in gen:
                        if self._stop_requested:
                            break
                finally:
                    gen.close()          # release video capture immediately

                if self._stop_requested:
                    break

                summary = engine.get_summary()
                if summary is not None:
                    summaries.append(summary)
                    self.policy_done.emit(summary)

            except Exception as exc:
                logger.warning("Policy %s failed: %s", policy, exc, exc_info=True)
                self.status_update.emit(f"Policy {policy} failed: {exc}")

        # ---- Proxy validation ----
        val_summary = None

        if self.run_validation and not self._stop_requested:
            current_step += 1
            self.status_update.emit(f"Running proxy validation ({current_step}/{total_steps})")
            self.progress.emit(current_step, total_steps)

            try:
                from validation.validator import ProxyValidator

                validator = ProxyValidator(
                    self.source_path, model_n, model_s,
                    self.inf_params, self.source_type,
                )

                vgen = validator.run()
                try:
                    for vfr in vgen:
                        if self._stop_requested:
                            break
                finally:
                    vgen.close()

                if not self._stop_requested:
                    val_summary = validator.get_summary()
                    if val_summary is not None:
                        self.validation_done.emit(val_summary)

            except Exception as exc:
                logger.warning("Proxy validation failed: %s", exc, exc_info=True)
                self.status_update.emit(f"Validation failed: {exc}")

        # ---- Done ----
        if not self._stop_requested:
            self.all_done.emit(summaries, val_summary)
