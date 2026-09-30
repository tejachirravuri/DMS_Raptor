"""Thesis Analysis tab — offline multi-policy evaluation GUI."""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QFileDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QAbstractItemView, QMessageBox,
)

from core.config import THESIS_ANALYSIS_POLICIES

logger = logging.getLogger(__name__)

_ANALYSIS_POLICIES = list(THESIS_ANALYSIS_POLICIES)

_SUMMARY_DISPLAY_COLUMNS = [
    ("policy", "Policy"),
    ("accurate_usage_rate", "Accurate Usage"),
    ("mean_iou_match", "IoU Match"),
    ("mean_det_coverage", "Det Coverage"),
    ("mean_det_recall", "Det Recall"),
    ("benefit_rate", "Benefit Rate"),
    ("mean_t_total_deployed_ms", "Deployed ms"),
    ("informed_gain_iou_match", "Gain (IoU)"),
]


class AnalysisTab(QWidget):
    """Thesis Analysis Mode — offline dual-model evaluation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # --- Note ---
        note = QLabel(
            "Thesis Analysis Mode runs offline evaluation. Both detectors "
            "execute for accurate-reference metrics. Deployed runtime columns "
            "exclude evaluation-only inference cost."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background-color: #263238; color: #b0bec5; padding: 8px; "
            "border-radius: 4px; font-size: 11px;"
        )
        root.addWidget(note)

        # --- Input configuration ---
        grp_in = QGroupBox("Input Configuration")
        g = QGridLayout(grp_in)
        row = 0

        g.addWidget(QLabel("Video:"), row, 0)
        self._video_edit = QLineEdit()
        self._video_edit.setPlaceholderText("Path to input video (.mp4, .avi, ...)")
        g.addWidget(self._video_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._browse_video)
        g.addWidget(btn, row, 2)
        row += 1

        g.addWidget(QLabel("Fast model:"), row, 0)
        self._fast_edit = QLineEdit()
        self._fast_edit.setPlaceholderText("Fast (n) model weights (.pt / .onnx / .engine)")
        g.addWidget(self._fast_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_weights(self._fast_edit))
        g.addWidget(btn, row, 2)
        row += 1

        g.addWidget(QLabel("Accurate model:"), row, 0)
        self._accurate_edit = QLineEdit()
        self._accurate_edit.setPlaceholderText("Accurate (s) model weights (.pt / .onnx / .engine)")
        g.addWidget(self._accurate_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_weights(self._accurate_edit))
        g.addWidget(btn, row, 2)
        row += 1

        g.addWidget(QLabel("Output folder:"), row, 0)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Where to write per-policy CSVs and summary")
        g.addWidget(self._out_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._browse_out)
        g.addWidget(btn, row, 2)

        root.addWidget(grp_in)

        # --- Parameters ---
        grp_params = QGroupBox("Parameters")
        p_layout = QHBoxLayout(grp_params)

        p_layout.addWidget(QLabel("Device:"))
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
        p_layout.addWidget(self._device_combo)

        p_layout.addWidget(QLabel("Conf:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setDecimals(2)
        p_layout.addWidget(self._conf_spin)

        p_layout.addWidget(QLabel("IoU thr:"))
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.1, 1.0)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(0.50)
        self._iou_spin.setDecimals(2)
        p_layout.addWidget(self._iou_spin)

        p_layout.addWidget(QLabel("ImgSz:"))
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(320, 1280)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        p_layout.addWidget(self._imgsz_spin)

        p_layout.addWidget(QLabel("Start frame:"))
        self._start_spin = QSpinBox()
        self._start_spin.setRange(0, 999999)
        self._start_spin.setValue(0)
        p_layout.addWidget(self._start_spin)

        p_layout.addWidget(QLabel("Max frames:"))
        self._max_spin = QSpinBox()
        self._max_spin.setRange(0, 999999)
        self._max_spin.setValue(0)
        self._max_spin.setSpecialValueText("all")
        p_layout.addWidget(self._max_spin)

        p_layout.addStretch()
        root.addWidget(grp_params)

        # --- Policy checklist ---
        grp_pol = QGroupBox("Policies")
        pol_layout = QHBoxLayout(grp_pol)
        self._policy_checks: Dict[str, QCheckBox] = {}
        for pol in _ANALYSIS_POLICIES:
            cb = QCheckBox(pol)
            cb.setChecked(True)
            self._policy_checks[pol] = cb
            pol_layout.addWidget(cb)
        root.addWidget(grp_pol)

        # --- Actions ---
        act_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Thesis Analysis")
        self._run_btn.setStyleSheet(
            "background-color: #4caf50; color: white; "
            "font-weight: bold; padding: 10px;"
        )
        self._run_btn.clicked.connect(self._on_run)
        act_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        act_row.addWidget(self._stop_btn)

        self._open_btn = QPushButton("Open Output Folder")
        self._open_btn.clicked.connect(self._open_output)
        act_row.addWidget(self._open_btn)

        root.addLayout(act_row)

        # --- Progress ---
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("")
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #8899aa; font-size: 12px;")
        root.addWidget(self._status_label)

        # --- Results table ---
        grp_res = QGroupBox("Summary (accurate-reference agreement)")
        res_layout = QVBoxLayout(grp_res)
        ncols = len(_SUMMARY_DISPLAY_COLUMNS)
        self._result_table = QTableWidget(0, ncols)
        self._result_table.setHorizontalHeaderLabels(
            [label for _, label in _SUMMARY_DISPLAY_COLUMNS]
        )
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._result_table.horizontalHeader().setStretchLastSection(True)
        self._result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        res_layout.addWidget(self._result_table)
        root.addWidget(grp_res)

        root.addStretch()

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------
    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video",
            "", "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)",
        )
        if path:
            self._video_edit.setText(path)

    def _browse_weights(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weights",
            "", "Model files (*.pt *.onnx *.engine);;All files (*)",
        )
        if path:
            line_edit.setText(path)

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._out_edit.setText(path)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self):
        if self._worker is not None and self._worker.isRunning():
            return

        video = self._video_edit.text().strip()
        fast = self._fast_edit.text().strip()
        accurate = self._accurate_edit.text().strip()
        out_dir = self._out_edit.text().strip()

        if not video or not os.path.isfile(video):
            QMessageBox.warning(self, "Error", "Please select a valid video file.")
            return
        if not fast or not os.path.isfile(fast):
            QMessageBox.warning(self, "Error", "Please select valid fast model weights.")
            return
        if not accurate or not os.path.isfile(accurate):
            QMessageBox.warning(self, "Error", "Please select valid accurate model weights.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Error", "Please select an output folder.")
            return

        policies = [p for p, cb in self._policy_checks.items() if cb.isChecked()]
        if not policies:
            QMessageBox.warning(self, "Error", "Select at least one policy.")
            return

        self._result_table.setRowCount(0)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        from gui.analysis_worker import AnalysisWorker

        self._worker = AnalysisWorker(
            video_path=video,
            fast_weights=fast,
            accurate_weights=accurate,
            policies=policies,
            out_dir=out_dir,
            device=self._device_combo.currentText(),
            conf=self._conf_spin.value(),
            iou_threshold=self._iou_spin.value(),
            imgsz=self._imgsz_spin.value(),
            start_frame=self._start_spin.value(),
            max_frames=self._max_spin.value(),
        )
        self._worker.status_update.connect(self._on_status)
        self._worker.policy_progress.connect(self._on_policy_progress)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._status_label.setText("Stopping after current policy...")

    def _on_status(self, msg: str):
        self._status_label.setText(msg)

    def _on_policy_progress(self, policy_name: str, status: str):
        if status == "running":
            self._status_label.setText(f"Running: {policy_name}")
        elif status == "done":
            self._status_label.setText(f"Completed: {policy_name}")

    def _on_all_done(self, summaries: dict):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        stopped = self._worker is not None and self._worker._stop_requested
        n = len(summaries)
        if stopped:
            self._status_label.setText(
                f"Stopped — {n} policy(ies) completed. "
                f"Output: {self._out_edit.text().strip()}"
            )
        else:
            self._status_label.setText(
                f"Done — {n} policies evaluated. "
                f"Output: {self._out_edit.text().strip()}"
            )
        self._populate_table(summaries)
        self._disconnect_worker()

    def _on_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {msg}")
        self._disconnect_worker()
        QMessageBox.critical(self, "Analysis Error", msg)

    def _disconnect_worker(self):
        if self._worker is not None:
            for sig in (
                self._worker.status_update,
                self._worker.policy_progress,
                self._worker.all_done,
                self._worker.error_occurred,
            ):
                try:
                    sig.disconnect()
                except TypeError:
                    pass
            self._worker = None

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    def _populate_table(self, summaries: dict):
        self._result_table.setRowCount(0)
        import math
        for pname, s in summaries.items():
            row = self._result_table.rowCount()
            self._result_table.insertRow(row)
            for col_idx, (key, _label) in enumerate(_SUMMARY_DISPLAY_COLUMNS):
                val = s.get(key, "")
                if key == "policy":
                    text = pname
                elif isinstance(val, float):
                    if math.isfinite(val):
                        text = f"{val:.4f}"
                    else:
                        text = ""
                else:
                    text = str(val) if val != "" else ""
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self._result_table.setItem(row, col_idx, item)

    # ------------------------------------------------------------------
    # Open output folder
    # ------------------------------------------------------------------
    def _open_output(self):
        out = self._out_edit.text().strip()
        if out and os.path.isdir(out):
            if sys.platform == "win32":
                os.startfile(out)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", out])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", out])
        else:
            QMessageBox.information(
                self, "Output Folder",
                "No output folder found. Run an analysis first.",
            )
