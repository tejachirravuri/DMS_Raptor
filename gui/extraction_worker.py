"""Background worker for extracting frames and YOLO-format labels."""
from __future__ import annotations
import logging
import os
from typing import List
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class ExtractionWorker(QThread):
    status_update = pyqtSignal(str)
    frame_progress = pyqtSignal(int, int)   # current, total
    all_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, datasets, model_n_path, model_s_path,
                 output_root, imgsz=640, device="cpu",
                 conf_threshold=0.25, img_format="jpg",
                 max_frames=0, stride=1):
        super().__init__()
        # datasets: list of dicts with keys: subfolder, videos (list of paths)
        self.datasets = datasets
        self.model_n_path = model_n_path
        self.model_s_path = model_s_path
        self.output_root = output_root
        self.imgsz = imgsz
        self.device = device
        self.conf_threshold = conf_threshold
        self.img_format = img_format
        self.max_frames = max_frames
        self.stride = stride
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            if not self._stop_requested:
                logger.exception("ExtractionWorker failed")
                self.error_occurred.emit(str(exc))
            return
        if not self._stop_requested:
            self.all_complete.emit()

    def _run_impl(self):
        from ultralytics import YOLO

        self.status_update.emit("Loading models...")
        model_n = YOLO(self.model_n_path)
        model_s = YOLO(self.model_s_path)

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for mdl in (model_n, model_s):
            try:
                mdl.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)
            except Exception:
                pass

        for ds in self.datasets:
            if self._stop_requested:
                break
            subfolder = ds["subfolder"]

            for video_path in ds["videos"]:
                if self._stop_requested:
                    break
                video_name = os.path.splitext(os.path.basename(video_path))[0]

                # Process with both models
                for tag, model in [("n_only", model_n), ("s_only", model_s)]:
                    if self._stop_requested:
                        break
                    self.status_update.emit(f"{subfolder}/{video_name} — {tag}")

                    img_dir = os.path.join(self.output_root, subfolder, tag, "images")
                    lbl_dir = os.path.join(self.output_root, subfolder, tag, "labels")
                    os.makedirs(img_dir, exist_ok=True)
                    os.makedirs(lbl_dir, exist_ok=True)

                    cap = cv2.VideoCapture(video_path)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if self.max_frames > 0:
                        total = min(total, self.max_frames * self.stride)

                    frame_idx = 0
                    saved_count = 0
                    while cap.isOpened():
                        if self._stop_requested:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if self.max_frames > 0 and saved_count >= self.max_frames:
                            break

                        frame_idx += 1
                        if frame_idx % self.stride != 0:
                            continue

                        # Run inference
                        results = model.predict(
                            frame, imgsz=self.imgsz, device=self.device,
                            conf=self.conf_threshold, verbose=False,
                        )

                        # Save frame
                        fname = f"{video_name}_{saved_count:06d}"
                        img_path = os.path.join(img_dir, f"{fname}.{self.img_format}")
                        cv2.imwrite(img_path, frame)

                        # Save YOLO-format labels
                        lbl_path = os.path.join(lbl_dir, f"{fname}.txt")
                        h, w = frame.shape[:2]
                        with open(lbl_path, "w", encoding="utf-8") as f:
                            if results and len(results) > 0 and results[0].boxes is not None:
                                boxes = results[0].boxes
                                for box in boxes:
                                    cls_id = int(box.cls.item())
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    # Convert to YOLO format (normalized center x, y, w, h)
                                    cx = ((x1 + x2) / 2.0) / w
                                    cy = ((y1 + y2) / 2.0) / h
                                    bw = (x2 - x1) / w
                                    bh = (y2 - y1) / h
                                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                        saved_count += 1
                        self.frame_progress.emit(saved_count, total // max(self.stride, 1))

                    cap.release()

        self.status_update.emit("Extraction complete!")
