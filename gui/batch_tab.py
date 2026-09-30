"""Batch Run tab — one-click processing of multiple datasets."""
from __future__ import annotations
import logging
import os
import re
from typing import Dict, List, Optional
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QMessageBox, QFileDialog,
    QAbstractItemView, QComboBox, QSpinBox, QDoubleSpinBox,
)
from core.config import POLICIES, RunConfig, InferenceParams

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MODEL_EXT = ".pt"


class BatchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._datasets = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # Input config
        grp_in = QGroupBox("Input Configuration")
        g = QGridLayout(grp_in)

        g.addWidget(QLabel("Source Videos Folder:"), 0, 0)
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText("Folder with subfolders (glass/, composite/, ...)")
        g.addWidget(self._source_edit, 0, 1)
        btn_src = QPushButton("Browse")
        btn_src.clicked.connect(lambda: self._browse_folder(self._source_edit))
        g.addWidget(btn_src, 0, 2)

        g.addWidget(QLabel("Models Folder:"), 1, 0)
        self._models_edit = QLineEdit()
        self._models_edit.setPlaceholderText("Folder with matching subfolders containing .pt files")
        g.addWidget(self._models_edit, 1, 1)
        btn_mdl = QPushButton("Browse")
        btn_mdl.clicked.connect(lambda: self._browse_folder(self._models_edit))
        g.addWidget(btn_mdl, 1, 2)

        g.addWidget(QLabel("Output Folder:"), 2, 0)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Where to save all results")
        g.addWidget(self._output_edit, 2, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(lambda: self._browse_folder(self._output_edit))
        g.addWidget(btn_out, 2, 2)

        scan_btn = QPushButton("Scan Folders")
        scan_btn.setStyleSheet("background-color: #00bcd4; color: white; font-weight: bold; padding: 6px;")
        scan_btn.clicked.connect(self._scan_folders)
        g.addWidget(scan_btn, 3, 0, 1, 3)

        root.addWidget(grp_in)

        # Discovered datasets table
        grp_ds = QGroupBox("Discovered Datasets")
        ds_layout = QVBoxLayout(grp_ds)
        self._ds_table = QTableWidget(0, 5)
        self._ds_table.setHorizontalHeaderLabels(["Subfolder", "Videos", "Model-n", "Model-s", "Status"])
        self._ds_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._ds_table.horizontalHeader().setStretchLastSection(True)
        self._ds_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        ds_layout.addWidget(self._ds_table)
        root.addWidget(grp_ds)

        # Run options
        grp_opt = QGroupBox("Run Options")
        opt_layout = QVBoxLayout(grp_opt)

        # Policy checkboxes
        pol_row = QHBoxLayout()
        self._policy_checks = {}
        for pol in POLICIES:
            cb = QCheckBox(pol)
            cb.setChecked(True)
            self._policy_checks[pol] = cb
            pol_row.addWidget(cb)
        opt_layout.addLayout(pol_row)

        # Options
        chk_row = QHBoxLayout()
        self._chk_videos = QCheckBox("Save annotated videos")
        self._chk_videos.setChecked(True)
        self._chk_plots = QCheckBox("Generate comparison plots")
        self._chk_plots.setChecked(True)
        self._chk_csv = QCheckBox("Export CSV summaries")
        self._chk_csv.setChecked(True)
        self._chk_validation = QCheckBox("Run Proxy Validation")
        self._chk_validation.setChecked(True)
        for c in (self._chk_videos, self._chk_plots, self._chk_csv, self._chk_validation):
            chk_row.addWidget(c)
        opt_layout.addLayout(chk_row)

        # Device
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
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
        dev_row.addWidget(self._device_combo)
        dev_row.addWidget(QLabel("Image size:"))
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(320, 1280)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        dev_row.addWidget(self._imgsz_spin)
        dev_row.addStretch()
        opt_layout.addLayout(dev_row)

        root.addWidget(grp_opt)

        # Actions
        act_row = QHBoxLayout()
        self._run_btn = QPushButton("Run All")
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

    def _scan_folders(self):
        src = self._source_edit.text().strip()
        mdl = self._models_edit.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Error", "Please select a valid source videos folder.")
            return
        if not mdl or not os.path.isdir(mdl):
            QMessageBox.warning(self, "Error", "Please select a valid models folder.")
            return

        self._datasets.clear()
        self._ds_table.setRowCount(0)

        for sub in sorted(os.listdir(src)):
            sub_path = os.path.join(src, sub)
            if not os.path.isdir(sub_path):
                continue

            # Find videos
            videos = [os.path.join(sub_path, f) for f in sorted(os.listdir(sub_path))
                      if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
            if not videos:
                continue

            # Find models in matching subfolder
            mdl_sub = os.path.join(mdl, sub)
            model_n, model_s = None, None
            if os.path.isdir(mdl_sub):
                pt_files = sorted([f for f in os.listdir(mdl_sub) if f.endswith(MODEL_EXT)])
                for f in pt_files:
                    fl = f.lower()
                    if any(k in fl for k in ("nano", "_n.", "_n_", "yolov8n", "best_n")):
                        model_n = os.path.join(mdl_sub, f)
                    elif any(k in fl for k in ("small", "_s.", "_s_", "yolov8s", "best_s")):
                        model_s = os.path.join(mdl_sub, f)
                # Fallback: alphabetical (first=n, second=s)
                if not model_n and not model_s and len(pt_files) >= 2:
                    model_n = os.path.join(mdl_sub, pt_files[0])
                    model_s = os.path.join(mdl_sub, pt_files[1])
                elif not model_n and not model_s and len(pt_files) == 1:
                    # Single model — use it for both (baseline comparison)
                    model_n = os.path.join(mdl_sub, pt_files[0])
                    model_s = model_n

            ds = {
                "subfolder": sub,
                "videos": videos,
                "model_n": model_n,
                "model_s": model_s,
            }
            self._datasets.append(ds)

            # Add table row
            r = self._ds_table.rowCount()
            self._ds_table.insertRow(r)
            self._ds_table.setItem(r, 0, QTableWidgetItem(sub))
            self._ds_table.setItem(r, 1, QTableWidgetItem(str(len(videos))))
            self._ds_table.setItem(r, 2, QTableWidgetItem("\u2713" if model_n else "\u2717"))
            self._ds_table.setItem(r, 3, QTableWidgetItem("\u2713" if model_s else "\u2717"))
            status = "Ready" if (model_n and model_s) else "Missing model(s)"
            self._ds_table.setItem(r, 4, QTableWidgetItem(status))

        if not self._datasets:
            QMessageBox.information(self, "Scan Results", "No datasets found. Check folder structure.")
        else:
            self._status_label.setText(f"Found {len(self._datasets)} dataset(s)")

    def _on_run(self):
        if self._worker and self._worker.isRunning():
            return

        out = self._output_edit.text().strip()
        if not out:
            QMessageBox.warning(self, "Error", "Please select an output folder.")
            return

        # Filter datasets with models
        valid = [ds for ds in self._datasets if ds["model_n"] and ds["model_s"]]
        if not valid:
            QMessageBox.warning(self, "Error", "No valid datasets found (missing models).")
            return

        policies = [p for p, cb in self._policy_checks.items() if cb.isChecked()]
        if not policies:
            QMessageBox.warning(self, "Error", "Select at least one policy.")
            return

        os.makedirs(out, exist_ok=True)

        inf = InferenceParams(
            imgsz=self._imgsz_spin.value(),
            device=self._device_combo.currentText(),
        )

        from gui.batch_worker import BatchWorker
        self._worker = BatchWorker(
            datasets=valid,
            policies=policies,
            base_config=RunConfig(),
            inf_params=inf,
            output_root=out,
            save_videos=self._chk_videos.isChecked(),
            generate_plots=self._chk_plots.isChecked(),
            export_csv=self._chk_csv.isChecked(),
            run_validation=self._chk_validation.isChecked(),
        )
        self._worker.status_update.connect(self._on_status)
        self._worker.frame_progress.connect(self._on_progress)
        self._worker.run_finished.connect(self._on_run_finished)
        self._worker.dataset_complete.connect(self._on_ds_complete)
        self._worker.all_complete.connect(self._on_all_complete)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _on_stop(self):
        if self._worker:
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._status_label.setText("Stopping...")

    def _on_status(self, msg):
        self._status_label.setText(msg)

    def _on_progress(self, current, total):
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(current)
            pct = int(100 * current / total)
            self._progress.setFormat(f"Frame {current}/{total} ({pct}%)")

    def _on_run_finished(self, subfolder, summary):
        # Update table status
        for r in range(self._ds_table.rowCount()):
            item = self._ds_table.item(r, 0)
            if item and item.text() == subfolder:
                status_item = self._ds_table.item(r, 4)
                if status_item:
                    current = status_item.text()
                    if "Running" not in current:
                        status_item.setText(f"Running — {summary.policy} done")
                    else:
                        status_item.setText(f"Running — {summary.policy} done")

    def _on_ds_complete(self, subfolder):
        for r in range(self._ds_table.rowCount()):
            item = self._ds_table.item(r, 0)
            if item and item.text() == subfolder:
                self._ds_table.item(r, 4).setText("Complete \u2713")

    def _on_all_complete(self):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Batch processing complete!")
        self._progress.setFormat("Done!")
        self._disconnect_worker()
        QMessageBox.information(self, "Complete", "All batch runs finished successfully.")

    def _on_stopped(self):
        completed = sum(
            1 for r in range(self._ds_table.rowCount())
            if self._ds_table.item(r, 4) and "\u2713" in self._ds_table.item(r, 4).text()
        )
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText(
            f"Stopped by user \u2014 {completed} dataset(s) completed"
        )
        self._progress.setFormat("Stopped")
        self._disconnect_worker()

    def _on_error(self, msg):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText(f"Error: {msg}")
        self._disconnect_worker()
        QMessageBox.critical(self, "Error", msg)

    def _disconnect_worker(self):
        if self._worker is not None:
            for sig in (
                self._worker.status_update,
                self._worker.frame_progress,
                self._worker.run_finished,
                self._worker.dataset_complete,
                self._worker.all_complete,
                self._worker.stopped,
                self._worker.error_occurred,
            ):
                try:
                    sig.disconnect()
                except TypeError:
                    pass
            self._worker = None

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
