"""Frame & Label Extraction tab — extract frames + YOLO labels from both models."""
from __future__ import annotations
import logging
import os
from typing import List
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QMessageBox, QFileDialog,
    QAbstractItemView, QComboBox, QSpinBox, QDoubleSpinBox,
)

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class ExtractionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._datasets = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # Input group
        grp_in = QGroupBox("Input")
        g = QGridLayout(grp_in)

        g.addWidget(QLabel("Source Folder (with subfolders):"), 0, 0)
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText("Folder containing subfolders with videos")
        g.addWidget(self._source_edit, 0, 1)
        btn_src = QPushButton("Browse")
        btn_src.clicked.connect(lambda: self._browse_folder(self._source_edit))
        g.addWidget(btn_src, 0, 2)

        g.addWidget(QLabel("Model-n Path (.pt):"), 1, 0)
        self._model_n_edit = QLineEdit()
        self._model_n_edit.setPlaceholderText("path/to/yolov8n.pt")
        g.addWidget(self._model_n_edit, 1, 1)
        btn_n = QPushButton("Browse")
        btn_n.clicked.connect(lambda: self._browse_file(self._model_n_edit))
        g.addWidget(btn_n, 1, 2)

        g.addWidget(QLabel("Model-s Path (.pt):"), 2, 0)
        self._model_s_edit = QLineEdit()
        self._model_s_edit.setPlaceholderText("path/to/yolov8s.pt")
        g.addWidget(self._model_s_edit, 2, 1)
        btn_s = QPushButton("Browse")
        btn_s.clicked.connect(lambda: self._browse_file(self._model_s_edit))
        g.addWidget(btn_s, 2, 2)

        g.addWidget(QLabel("Output Folder:"), 3, 0)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Where to save extracted frames & labels")
        g.addWidget(self._output_edit, 3, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(lambda: self._browse_folder(self._output_edit))
        g.addWidget(btn_out, 3, 2)

        scan_btn = QPushButton("Scan Source Folder")
        scan_btn.setStyleSheet("background-color: #00bcd4; color: white; font-weight: bold; padding: 6px;")
        scan_btn.clicked.connect(self._scan_folders)
        g.addWidget(scan_btn, 4, 0, 1, 3)

        root.addWidget(grp_in)

        # Options
        grp_opt = QGroupBox("Options")
        opt_grid = QGridLayout(grp_opt)

        opt_grid.addWidget(QLabel("Image Format:"), 0, 0)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["jpg", "png"])
        opt_grid.addWidget(self._format_combo, 0, 1)

        opt_grid.addWidget(QLabel("Confidence Threshold:"), 0, 2)
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        opt_grid.addWidget(self._conf_spin, 0, 3)

        opt_grid.addWidget(QLabel("Device:"), 1, 0)
        self._device_combo = QComboBox()
        devices = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    devices.append(f"cuda:{i}")
        except Exception:
            pass
        self._device_combo.addItems(devices)
        opt_grid.addWidget(self._device_combo, 1, 1)

        opt_grid.addWidget(QLabel("Max Frames (0=all):"), 1, 2)
        self._max_frames_spin = QSpinBox()
        self._max_frames_spin.setRange(0, 999999)
        self._max_frames_spin.setValue(0)
        opt_grid.addWidget(self._max_frames_spin, 1, 3)

        opt_grid.addWidget(QLabel("Frame Stride:"), 2, 0)
        self._stride_spin = QSpinBox()
        self._stride_spin.setRange(1, 100)
        self._stride_spin.setValue(1)
        opt_grid.addWidget(self._stride_spin, 2, 1)

        opt_grid.addWidget(QLabel("Image Size:"), 2, 2)
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(320, 1280)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        opt_grid.addWidget(self._imgsz_spin, 2, 3)

        root.addWidget(grp_opt)

        # Discovered datasets table
        grp_ds = QGroupBox("Discovered Datasets")
        ds_layout = QVBoxLayout(grp_ds)
        self._ds_table = QTableWidget(0, 4)
        self._ds_table.setHorizontalHeaderLabels(["Subfolder", "Videos", "Est. Frames", "Status"])
        self._ds_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._ds_table.horizontalHeader().setStretchLastSection(True)
        self._ds_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        ds_layout.addWidget(self._ds_table)
        root.addWidget(grp_ds)

        # Actions
        act_row = QHBoxLayout()
        self._run_btn = QPushButton("Extract All")
        self._run_btn.setStyleSheet("background-color: #4caf50; color: white; font-weight: bold; padding: 10px;")
        self._run_btn.clicked.connect(self._on_run)
        act_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        act_row.addWidget(self._stop_btn)

        self._open_btn = QPushButton("Open Output")
        self._open_btn.clicked.connect(self._open_output)
        act_row.addWidget(self._open_btn)
        root.addLayout(act_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("")
        root.addWidget(self._progress)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #8899aa; font-size: 12px;")
        root.addWidget(self._status_label)

        root.addStretch()

    def _browse_folder(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            line_edit.setText(path)

    def _browse_file(self, line_edit):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "YOLO weights (*.pt)")
        if path:
            line_edit.setText(path)

    def _scan_folders(self):
        src = self._source_edit.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Error", "Please select a valid source folder.")
            return

        self._datasets.clear()
        self._ds_table.setRowCount(0)

        for sub in sorted(os.listdir(src)):
            sub_path = os.path.join(src, sub)
            if not os.path.isdir(sub_path):
                continue

            videos = [os.path.join(sub_path, f) for f in sorted(os.listdir(sub_path))
                      if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
            if not videos:
                continue

            # Estimate frame count
            est_frames = 0
            import cv2
            for v in videos:
                cap = cv2.VideoCapture(v)
                est_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

            ds = {"subfolder": sub, "videos": videos}
            self._datasets.append(ds)

            r = self._ds_table.rowCount()
            self._ds_table.insertRow(r)
            self._ds_table.setItem(r, 0, QTableWidgetItem(sub))
            self._ds_table.setItem(r, 1, QTableWidgetItem(str(len(videos))))
            self._ds_table.setItem(r, 2, QTableWidgetItem(str(est_frames)))
            self._ds_table.setItem(r, 3, QTableWidgetItem("Ready"))

        if not self._datasets:
            QMessageBox.information(self, "Scan Results", "No video subfolders found.")
        else:
            self._status_label.setText(f"Found {len(self._datasets)} dataset(s)")

    def _on_run(self):
        if self._worker and self._worker.isRunning():
            return

        model_n = self._model_n_edit.text().strip()
        model_s = self._model_s_edit.text().strip()
        output = self._output_edit.text().strip()

        if not model_n or not os.path.isfile(model_n):
            QMessageBox.warning(self, "Error", "Please select a valid Model-n path.")
            return
        if not model_s or not os.path.isfile(model_s):
            QMessageBox.warning(self, "Error", "Please select a valid Model-s path.")
            return
        if not output:
            QMessageBox.warning(self, "Error", "Please select an output folder.")
            return
        if not self._datasets:
            QMessageBox.warning(self, "Error", "No datasets found. Scan first.")
            return

        os.makedirs(output, exist_ok=True)

        from gui.extraction_worker import ExtractionWorker
        self._worker = ExtractionWorker(
            datasets=self._datasets,
            model_n_path=model_n,
            model_s_path=model_s,
            output_root=output,
            imgsz=self._imgsz_spin.value(),
            device=self._device_combo.currentText(),
            conf_threshold=self._conf_spin.value(),
            img_format=self._format_combo.currentText(),
            max_frames=self._max_frames_spin.value(),
            stride=self._stride_spin.value(),
        )
        self._worker.status_update.connect(self._on_status)
        self._worker.frame_progress.connect(self._on_progress)
        self._worker.all_complete.connect(self._on_complete)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _on_stop(self):
        if self._worker:
            self._worker.request_stop()
            self._status_label.setText("Stopping...")

    def _on_status(self, msg):
        self._status_label.setText(msg)

    def _on_progress(self, current, total):
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(current)
            pct = int(100 * current / total) if total > 0 else 0
            self._progress.setFormat(f"Frame {current}/{total} ({pct}%)")

    def _on_complete(self):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Extraction complete!")
        self._progress.setFormat("Done!")
        QMessageBox.information(self, "Complete", "Frame & label extraction finished.")

    def _on_error(self, msg):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _open_output(self):
        out = self._output_edit.text().strip()
        if out and os.path.isdir(out):
            import subprocess, sys
            if sys.platform == "win32":
                os.startfile(out)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", out])
            else:
                subprocess.Popen(["xdg-open", out])
