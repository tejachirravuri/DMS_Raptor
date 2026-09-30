"""Background worker for thesis-analysis mode (offline multi-policy evaluation)."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Run the thesis-analysis pipeline in a background thread.

    Wraps ``core.analysis_runner.run_analysis_policies`` and emits Qt
    signals so the GUI can update progress and display results.
    """

    status_update = pyqtSignal(str)
    policy_progress = pyqtSignal(str, str)   # policy_name, "running"|"done"
    all_done = pyqtSignal(dict)              # {policy_name: summary_dict}
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        fast_weights: str,
        accurate_weights: str,
        policies: List[str],
        out_dir: str,
        device: str = "cpu",
        conf: float = 0.25,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        start_frame: int = 0,
        max_frames: int = 0,
    ):
        super().__init__()
        self.video_path = video_path
        self.fast_weights = fast_weights
        self.accurate_weights = accurate_weights
        self.policies = list(policies)
        self.out_dir = out_dir
        self.device = device
        self.conf = conf
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.start_frame = start_frame
        self.max_frames = max_frames
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            logger.exception("AnalysisWorker failed")
            self.error_occurred.emit(str(exc))

    def _run_impl(self):
        from core.inference_backend import UltralyticsYOLOBackend
        from core.analysis_runner import run_analysis_policies

        self.status_update.emit("Loading fast model...")
        fast = UltralyticsYOLOBackend(
            self.fast_weights, device=self.device,
            imgsz=self.imgsz, conf=self.conf,
        )

        self.status_update.emit("Loading accurate model...")
        accurate = UltralyticsYOLOBackend(
            self.accurate_weights, device=self.device,
            imgsz=self.imgsz, conf=self.conf,
        )

        self.status_update.emit("Running thesis analysis...")

        def _progress(policy_name, status):
            self.policy_progress.emit(policy_name, status)
            if status == "running":
                self.status_update.emit(f"Running policy: {policy_name}")
            elif status == "done":
                self.status_update.emit(f"Completed: {policy_name}")

        summaries = run_analysis_policies(
            video_path=self.video_path,
            fast_backend=fast,
            accurate_backend=accurate,
            policies=self.policies,
            out_dir=self.out_dir,
            iou_threshold=self.iou_threshold,
            max_frames=self.max_frames,
            start_frame=self.start_frame,
            progress_callback=_progress,
            should_stop=lambda: self._stop_requested,
        )

        self.all_done.emit(summaries)
