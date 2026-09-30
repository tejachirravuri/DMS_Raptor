"""DMS-Raptor Proxy Validation tab — validates L,H,C proxies against model disagreement."""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QTabWidget, QSplitter,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
    QMessageBox, QGridLayout,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Dark theme helper
# ======================================================================
def _apply_dark(fig: Figure, ax=None):
    fig.patch.set_facecolor("#0a0e1a")
    if ax is None:
        return
    ax.set_facecolor("#0a0e1a")
    ax.tick_params(colors="#8899aa")
    ax.xaxis.label.set_color("#8899aa")
    ax.yaxis.label.set_color("#8899aa")
    ax.title.set_color("#c0c0c0")
    for spine in ax.spines.values():
        spine.set_color("#2a3a5c")
    ax.grid(True, alpha=0.2, color="#1a2040")


# ======================================================================
# Background validation worker
# ======================================================================
class ValidationWorker(QThread):
    """Runs dual-model validation in a background thread."""

    frame_progress = pyqtSignal(int, int)    # (current, total)
    finished = pyqtSignal(object)            # ValidationSummary
    error_occurred = pyqtSignal(str)
    stopped_early = pyqtSignal()

    def __init__(self, source_path: str, source_type: str,
                 model_n_path: str, model_s_path: str,
                 inf_params, alpha: float = 0.6,
                 proxy_settings=None, max_frames: int = 0,
                 iou_threshold: float = 0.5,
                 conf_threshold: float = 0.25):
        super().__init__()
        self.source_path = source_path
        self.source_type = source_type
        self.model_n_path = model_n_path
        self.model_s_path = model_s_path
        self.inf_params = inf_params
        self.alpha = alpha
        self.proxy_settings = proxy_settings
        self.max_frames = max_frames
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self._stop_requested = False
        self._validator = None

    def request_stop(self):
        self._stop_requested = True
        if self._validator is not None:
            self._validator.stop()

    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            if self._stop_requested:
                logger.info("Validation stopped by user: %s", exc)
                self.stopped_early.emit()
            else:
                logger.exception("Validation failed")
                self.error_occurred.emit(str(exc))
            return

        if self._stop_requested:
            self.stopped_early.emit()

    def _run_impl(self):
        import dataclasses
        from ultralytics import YOLO
        from core.config import InferenceParams
        from validation.validator import ProxyValidator

        # Adjust inf_params for max_frames
        inf = self.inf_params
        if self.max_frames > 0:
            inf = dataclasses.replace(inf, max_frames=self.max_frames)

        # Load models
        logger.info("Loading models for validation...")
        model_n = YOLO(self.model_n_path)
        if self._stop_requested:
            return
        model_s = YOLO(self.model_s_path)
        if self._stop_requested:
            return

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for tag, mdl in [("n", model_n), ("s", model_s)]:
            if self._stop_requested:
                return
            try:
                mdl.predict(dummy, imgsz=inf.imgsz, device=inf.device, verbose=False)
            except Exception:
                pass

        # Create validator
        validator = ProxyValidator(
            source_path=self.source_path,
            model_n=model_n,
            model_s=model_s,
            inf=inf,
            source_type=self.source_type,
            alpha=self.alpha,
            proxy_settings=self.proxy_settings,
            iou_match_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold,
        )
        self._validator = validator
        total = validator.total_frames

        for fr in validator.run():
            if self._stop_requested:
                break
            self.frame_progress.emit(fr.frame_idx + 1, total)

        summary = validator.get_summary()
        if summary is not None:
            self.finished.emit(summary)


# ======================================================================
# Background grid-search worker (keeps GUI responsive)
# ======================================================================
class GridSearchWorker(QThread):
    """Runs proxy_size x hist_bins grid search off the main thread."""

    progress = pyqtSignal(int, int, str)     # (current, total, label)
    finished = pyqtSignal(object)            # dict with results
    error_occurred = pyqtSignal(str)

    def __init__(self, source_path: str, disagree: np.ndarray,
                 ps_range: list, hb_range: list, alpha: float,
                 max_sample: int = 300):
        super().__init__()
        self.source_path = source_path
        self.disagree = disagree
        self.ps_range = ps_range
        self.hb_range = hb_range
        self.alpha = alpha
        self.max_sample = max_sample

    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _run_impl(self):
        import time
        import cv2
        from core.engine import complexity_proxies_fast

        disagree = self.disagree
        # Read sampled frames
        cap = cv2.VideoCapture(self.source_path)
        frames = []
        stride = max(1, len(disagree) // self.max_sample)
        idx = 0
        while cap.isOpened() and len(frames) < self.max_sample:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frames.append(frame)
            idx += 1
        cap.release()

        if len(frames) < 10:
            self.error_occurred.emit("Could not read enough frames.")
            return

        disagree_sampled = disagree[::stride][:len(frames)]
        total_combos = len(self.ps_range) * len(self.hb_range)

        results_L = np.zeros((len(self.hb_range), len(self.ps_range)))
        results_H = np.zeros_like(results_L)
        results_C = np.zeros_like(results_L)
        timing = np.zeros_like(results_L)
        lines = []
        combo_idx = 0

        for hi, hb in enumerate(self.hb_range):
            for pi, ps in enumerate(self.ps_range):
                combo_idx += 1
                self.progress.emit(combo_idx, total_combos, f"ps={ps}, hb={hb}")

                l_vals, h_vals = [], []
                t0 = time.perf_counter()
                for frame in frames:
                    L, H = complexity_proxies_fast(frame, 8, ps, hb)
                    l_vals.append(L)
                    h_vals.append(H)
                elapsed = (time.perf_counter() - t0) * 1000.0 / len(frames)
                timing[hi, pi] = elapsed

                l_arr = np.array(l_vals[:len(disagree_sampled)])
                h_arr = np.array(h_vals[:len(disagree_sampled)])
                d_arr = disagree_sampled[:len(l_arr)]

                l_lo, l_hi = l_arr.min(), l_arr.max()
                h_lo, h_hi = h_arr.min(), h_arr.max()
                l_norm = (l_arr - l_lo) / (l_hi - l_lo + 1e-9)
                h_norm = (h_arr - h_lo) / (h_hi - h_lo + 1e-9)
                c_norm = self.alpha * l_norm + (1.0 - self.alpha) * h_norm

                def _safe_corr(x, y):
                    if len(x) > 2 and np.std(x) > 1e-9 and np.std(y) > 1e-9:
                        return float(np.corrcoef(x, y)[0, 1])
                    return 0.0

                r_L = _safe_corr(l_arr, d_arr)
                r_H = _safe_corr(h_arr, d_arr)
                r_C = _safe_corr(c_norm, d_arr)

                results_L[hi, pi] = abs(r_L)
                results_H[hi, pi] = abs(r_H)
                results_C[hi, pi] = abs(r_C)
                lines.append(f"  {ps:>4} | {hb:>4}   | {r_L:+.4f} | {r_H:+.4f} |  {r_C:+.4f}   | {elapsed:.2f}")

        self.finished.emit({
            "results_L": results_L, "results_H": results_H,
            "results_C": results_C, "timing": timing,
            "ps_range": self.ps_range, "hb_range": self.hb_range,
            "lines": lines, "alpha": self.alpha,
        })


# ======================================================================
# Validation Tab
# ======================================================================
class ValidationTab(QWidget):
    """
    Proxy Validation tab for DMS-Raptor.

    Allows the user to run dual-model validation on a video, then
    view correlation analysis, sensitivity analysis, and export results.
    """

    # Emitted when validation completes, so other tabs can use the data
    validation_complete = pyqtSignal(object)   # ValidationSummary

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary = None   # ValidationSummary
        self._report = None    # ValidationReport
        self._worker: Optional[ValidationWorker] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter)

        # ===== TOP: Config Panel =====
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(4, 4, 4, 4)

        # Title
        title = QLabel("<h3>Proxy Validation — Validate L, H, C Against Model Disagreement</h3>")
        title.setStyleSheet("color: #00bcd4;")
        top_layout.addWidget(title)

        desc = QLabel(
            "Runs <b>both</b> YOLOv8n and YOLOv8s on every frame, then measures whether "
            "the C-score (from L and H) actually predicts when the heavier model was needed. "
            "This is <b>offline validation only</b> — it does not affect the deployed switching system."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8899aa; margin-bottom: 8px;")
        top_layout.addWidget(desc)

        # Config grid
        config_group = QGroupBox("Validation Configuration")
        grid = QGridLayout()
        config_group.setLayout(grid)

        # Video source
        grid.addWidget(QLabel("Video Source:"), 0, 0)
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText("Path to video file or image folder")
        grid.addWidget(self._source_edit, 0, 1)
        self._btn_browse = QPushButton("Browse")
        self._btn_browse.clicked.connect(self._browse_source)
        grid.addWidget(self._btn_browse, 0, 2)

        # Source type
        grid.addWidget(QLabel("Source Type:"), 0, 3)
        self._source_type = QComboBox()
        self._source_type.addItems(["video", "images"])
        grid.addWidget(self._source_type, 0, 4)

        # Model paths
        grid.addWidget(QLabel("Model n (.pt):"), 1, 0)
        self._model_n_edit = QLineEdit()
        self._model_n_edit.setPlaceholderText("yolov8n.pt")
        grid.addWidget(self._model_n_edit, 1, 1)
        btn_n = QPushButton("Browse")
        btn_n.clicked.connect(lambda: self._browse_model("n"))
        grid.addWidget(btn_n, 1, 2)

        grid.addWidget(QLabel("Model s (.pt):"), 1, 3)
        self._model_s_edit = QLineEdit()
        self._model_s_edit.setPlaceholderText("yolov8s.pt")
        grid.addWidget(self._model_s_edit, 1, 4)
        btn_s = QPushButton("Browse")
        btn_s.clicked.connect(lambda: self._browse_model("s"))
        grid.addWidget(btn_s, 1, 5)

        # Parameters row
        grid.addWidget(QLabel("Device:"), 2, 0)
        self._device_combo = QComboBox()
        self._device_combo.addItems(["cpu", "cuda:0", "cuda:1"])
        self._device_combo.setEditable(True)
        grid.addWidget(self._device_combo, 2, 1)

        grid.addWidget(QLabel("Max Frames (0=all):"), 2, 2)
        self._max_frames_spin = QSpinBox()
        self._max_frames_spin.setRange(0, 999999)
        self._max_frames_spin.setValue(0)
        grid.addWidget(self._max_frames_spin, 2, 3)

        grid.addWidget(QLabel("Alpha (L weight):"), 2, 4)
        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 1.0)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setValue(0.6)
        grid.addWidget(self._alpha_spin, 2, 5)

        # Advanced row
        grid.addWidget(QLabel("Conf Threshold:"), 3, 0)
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 0.99)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        grid.addWidget(self._conf_spin, 3, 1)

        grid.addWidget(QLabel("IoU Match Thr:"), 3, 2)
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.1, 0.9)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(0.5)
        grid.addWidget(self._iou_spin, 3, 3)

        grid.addWidget(QLabel("Image Size:"), 3, 4)
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(160, 1280)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        grid.addWidget(self._imgsz_spin, 3, 5)

        top_layout.addWidget(config_group)

        # Buttons row
        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run Validation")
        self._btn_run.setStyleSheet(
            "background-color: #00bcd4; color: white; font-weight: bold; "
            "padding: 10px 24px; font-size: 14px;"
        )
        self._btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self._btn_run)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setStyleSheet(
            "background-color: #e94560; color: white; font-weight: bold; "
            "padding: 10px 24px; font-size: 14px;"
        )
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_stop)

        btn_row.addStretch()

        self._btn_export_csv = QPushButton("Export Full CSV")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(self._btn_export_csv)

        self._btn_export_corr = QPushButton("Export Correlations CSV")
        self._btn_export_corr.setEnabled(False)
        self._btn_export_corr.clicked.connect(self._export_corr_csv)
        btn_row.addWidget(self._btn_export_corr)

        self._btn_export_plots = QPushButton("Export All Plots")
        self._btn_export_plots.setEnabled(False)
        self._btn_export_plots.clicked.connect(self._export_all_plots)
        btn_row.addWidget(self._btn_export_plots)

        top_layout.addLayout(btn_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("Ready")
        top_layout.addWidget(self._progress)

        splitter.addWidget(top)

        # ===== BOTTOM: Results =====
        bottom = QTabWidget()

        # Tab 1: Text Report
        self._report_text = QTextEdit()
        self._report_text.setReadOnly(True)
        self._report_text.setStyleSheet(
            "background-color:#0a0e1a; color:#c0c0c0; "
            "font-family:Consolas,monospace; font-size:12px;"
        )
        bottom.addTab(self._report_text, "Report")

        # Tab 2: Correlation Scatter
        self._fig_scatter = Figure(figsize=(8, 5))
        self._ax_scatter = self._fig_scatter.add_subplot(111)
        _apply_dark(self._fig_scatter, self._ax_scatter)
        self._canvas_scatter = FigureCanvasQTAgg(self._fig_scatter)
        scatter_widget = QWidget()
        scatter_layout = QVBoxLayout(scatter_widget)

        scatter_ctrl = QHBoxLayout()
        scatter_ctrl.addWidget(QLabel("Proxy:"))
        self._proxy_combo = QComboBox()
        self._proxy_combo.addItems([
            "C-score", "Laplacian (L)", "Entropy (H)", "Mean Conf (n)", "NIQE Score",
            "Tenengrad", "Edge Density", "Local Contrast", "Brenner", "Color Entropy",
        ])
        scatter_ctrl.addWidget(self._proxy_combo)
        scatter_ctrl.addWidget(QLabel("vs Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([
            "Disagreement Score", "Det Count Gap", "Mean Conf Gap",
            "Max Conf Gap", "IoU Agreement", "Extra Dets (s)",
        ])
        scatter_ctrl.addWidget(self._metric_combo)
        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self._plot_scatter)
        scatter_ctrl.addWidget(btn_plot)
        scatter_ctrl.addStretch()
        scatter_layout.addLayout(scatter_ctrl)
        scatter_layout.addWidget(self._canvas_scatter)
        bottom.addTab(scatter_widget, "Correlation Scatter")

        # Tab 3: Time Series Overlay
        self._fig_ts = Figure(figsize=(8, 5))
        self._ax_ts = self._fig_ts.add_subplot(111)
        _apply_dark(self._fig_ts, self._ax_ts)
        self._canvas_ts = FigureCanvasQTAgg(self._fig_ts)
        bottom.addTab(self._canvas_ts, "Time Series Overlay")

        # Tab 4: Correlation Matrix
        self._fig_corr = Figure(figsize=(8, 6))
        self._ax_corr = self._fig_corr.add_subplot(111)
        _apply_dark(self._fig_corr, self._ax_corr)
        self._canvas_corr = FigureCanvasQTAgg(self._fig_corr)
        bottom.addTab(self._canvas_corr, "Correlation Matrix")

        # Tab 5: Sensitivity Heatmap
        self._fig_sens = Figure(figsize=(8, 5))
        self._ax_sens = self._fig_sens.add_subplot(111)
        _apply_dark(self._fig_sens, self._ax_sens)
        self._canvas_sens = FigureCanvasQTAgg(self._fig_sens)
        bottom.addTab(self._canvas_sens, "Sensitivity Analysis")

        # Tab 6: Detection Comparison
        self._fig_det = Figure(figsize=(8, 5))
        self._ax_det = self._fig_det.add_subplot(111)
        _apply_dark(self._fig_det, self._ax_det)
        self._canvas_det = FigureCanvasQTAgg(self._fig_det)
        bottom.addTab(self._canvas_det, "Detection Comparison")

        # Tab 7: Proxy Size Study
        proxy_study_widget = self._build_proxy_study_tab()
        bottom.addTab(proxy_study_widget, "Proxy Size Study")

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

    # ------------------------------------------------------------------
    # Proxy Size Study sub-tab
    # ------------------------------------------------------------------
    def _build_proxy_study_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        desc = QLabel(
            "<b>Proxy Size Study</b> — Grid search over proxy_size \u00d7 hist_bins to find "
            "the optimal combination for scene complexity analysis. Measures correlation "
            "between C-score and model disagreement, plus computation time."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8899aa; margin-bottom: 8px;")
        lay.addWidget(desc)

        grid = QGridLayout()
        grid.addWidget(QLabel("proxy_size range:"), 0, 0)
        self._ps_min = QSpinBox(); self._ps_min.setRange(32, 1280); self._ps_min.setValue(64)
        grid.addWidget(self._ps_min, 0, 1)
        grid.addWidget(QLabel("to"), 0, 2)
        self._ps_max = QSpinBox(); self._ps_max.setRange(32, 1280); self._ps_max.setValue(640)
        grid.addWidget(self._ps_max, 0, 3)
        grid.addWidget(QLabel("step"), 0, 4)
        self._ps_step = QSpinBox(); self._ps_step.setRange(16, 256); self._ps_step.setValue(64)
        grid.addWidget(self._ps_step, 0, 5)

        grid.addWidget(QLabel("hist_bins range:"), 1, 0)
        self._hb_min = QSpinBox(); self._hb_min.setRange(8, 512); self._hb_min.setValue(32)
        grid.addWidget(self._hb_min, 1, 1)
        grid.addWidget(QLabel("to"), 1, 2)
        self._hb_max = QSpinBox(); self._hb_max.setRange(8, 512); self._hb_max.setValue(256)
        grid.addWidget(self._hb_max, 1, 3)
        grid.addWidget(QLabel("step"), 1, 4)
        self._hb_step = QSpinBox(); self._hb_step.setRange(8, 128); self._hb_step.setValue(32)
        grid.addWidget(self._hb_step, 1, 5)
        lay.addLayout(grid)

        self._btn_grid_search = QPushButton("Run Grid Search")
        self._btn_grid_search.setStyleSheet(
            "background-color: #ff9800; color: white; font-weight: bold; padding: 8px;"
        )
        self._btn_grid_search.clicked.connect(self._run_proxy_grid_search)
        lay.addWidget(self._btn_grid_search)

        self._grid_progress = QProgressBar()
        self._grid_progress.setTextVisible(True)
        self._grid_progress.setFormat("Ready")
        lay.addWidget(self._grid_progress)

        # Heatmap canvas
        self._fig_grid = Figure(figsize=(10, 6))
        self._ax_grid = self._fig_grid.add_subplot(111)
        _apply_dark(self._fig_grid, self._ax_grid)
        self._canvas_grid = FigureCanvasQTAgg(self._fig_grid)
        self._canvas_grid.setMinimumHeight(350)
        lay.addWidget(self._canvas_grid, stretch=1)

        # Results text
        self._grid_text = QTextEdit()
        self._grid_text.setReadOnly(True)
        self._grid_text.setMaximumHeight(160)
        self._grid_text.setStyleSheet(
            "background-color:#0a0e1a; color:#c0c0c0; "
            "font-family:Consolas,monospace; font-size:11px;"
        )
        lay.addWidget(self._grid_text)

        return w

    def _run_proxy_grid_search(self):
        """Run grid search over proxy_size x hist_bins in a background thread."""
        if self._summary is None:
            QMessageBox.warning(
                self, "No Data",
                "Run a validation first to get frame data for the grid search.",
            )
            return

        disagree = np.array(self._summary.disagreement_scores) if hasattr(self._summary, 'disagreement_scores') else None
        if disagree is None or len(disagree) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Not enough disagreement data.")
            return

        source = self._source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "Error", "No video source specified.")
            return

        ps_range = list(range(self._ps_min.value(), self._ps_max.value() + 1, self._ps_step.value()))
        hb_range = list(range(self._hb_min.value(), self._hb_max.value() + 1, self._hb_step.value()))
        alpha = self._alpha_spin.value()

        self._btn_grid_search.setEnabled(False)
        self._btn_grid_search.setText("Running...")
        self._grid_text.setPlainText("Grid search running in background...")

        self._grid_worker = GridSearchWorker(
            source_path=source, disagree=disagree,
            ps_range=ps_range, hb_range=hb_range,
            alpha=alpha, max_sample=min(300, len(disagree)),
        )
        self._grid_worker.progress.connect(self._on_grid_progress)
        self._grid_worker.finished.connect(self._on_grid_finished)
        self._grid_worker.error_occurred.connect(self._on_grid_error)
        self._grid_worker.start()

    def _on_grid_progress(self, current, total, label):
        self._grid_progress.setMaximum(total)
        self._grid_progress.setValue(current)
        self._grid_progress.setFormat(f"{current}/{total} -- {label}")

    def _on_grid_error(self, msg):
        self._btn_grid_search.setEnabled(True)
        self._btn_grid_search.setText("Run Grid Search")
        QMessageBox.critical(self, "Grid Search Error", msg)

    def _on_grid_finished(self, result):
        self._btn_grid_search.setEnabled(True)
        self._btn_grid_search.setText("Run Grid Search")
        self._grid_progress.setFormat("Done!")

        results_L = result["results_L"]
        results_H = result["results_H"]
        results_C = result["results_C"]
        timing = result["timing"]
        ps_range = result["ps_range"]
        hb_range = result["hb_range"]
        lines = result["lines"]

        # Plot 3 side-by-side heatmaps
        self._fig_grid.clear()
        _apply_dark(self._fig_grid)

        ax_L, ax_H, ax_C = self._fig_grid.subplots(1, 3)
        for ax in (ax_L, ax_H, ax_C):
            _apply_dark(self._fig_grid, ax)

        def _draw_heatmap(ax, data, title_str, ps_r, hb_r):
            im = ax.imshow(data, cmap="YlOrRd", aspect="auto",
                           interpolation="nearest", vmin=0, vmax=0.7)
            ax.set_xticks(range(len(ps_r)))
            ax.set_xticklabels([str(v) for v in ps_r], fontsize=6, rotation=45)
            ax.set_yticks(range(len(hb_r)))
            ax.set_yticklabels([str(v) for v in hb_r], fontsize=6)
            ax.set_xlabel("proxy_size", fontsize=8, color="#cccccc")
            ax.set_ylabel("hist_bins", fontsize=8, color="#cccccc")
            for hi_ in range(len(hb_r)):
                for pi_ in range(len(ps_r)):
                    val_ = data[hi_, pi_]
                    tc = "white" if val_ > 0.3 else "black"
                    ax.text(pi_, hi_, f"{val_:.2f}", ha="center", va="center",
                            color=tc, fontsize=6)
            best = np.unravel_index(np.argmax(data), data.shape)
            ax.plot(best[1], best[0], 'w*', markersize=14,
                    markeredgecolor='black', markeredgewidth=0.5)
            best_val = data[best]
            ax.set_title(
                f"{title_str}\nBest: ps={ps_r[best[1]]}, hb={hb_r[best[0]]}, |r|={best_val:.3f}",
                color="#cccccc", fontsize=8,
            )
            return im

        _draw_heatmap(ax_L, results_L, "|r(L)| vs Disagree", ps_range, hb_range)
        _draw_heatmap(ax_H, results_H, "|r(H)| vs Disagree", ps_range, hb_range)
        im_c = _draw_heatmap(ax_C, results_C, "|r(C_norm)| vs Disagree", ps_range, hb_range)

        self._fig_grid.colorbar(im_c, ax=[ax_L, ax_H, ax_C], shrink=0.75, label="|r|")
        self._ax_grid = ax_C

        self._fig_grid.tight_layout()
        self._canvas_grid.draw_idle()

        # Show results text
        best_L = np.unravel_index(np.argmax(results_L), results_L.shape)
        best_H = np.unravel_index(np.argmax(results_H), results_H.shape)
        best_C = np.unravel_index(np.argmax(results_C), results_C.shape)
        lines.append("")
        lines.append("=== BEST COMBINATIONS ===")
        lines.append(f"  L (raw):    ps={ps_range[best_L[1]]}, hb={hb_range[best_L[0]]}, "
                     f"|r|={results_L[best_L]:.4f}, time={timing[best_L]:.2f} ms/f")
        lines.append(f"  H (raw):    ps={ps_range[best_H[1]]}, hb={hb_range[best_H[0]]}, "
                     f"|r|={results_H[best_H]:.4f}, time={timing[best_H]:.2f} ms/f")
        lines.append(f"  C (normed): ps={ps_range[best_C[1]]}, hb={hb_range[best_C[0]]}, "
                     f"|r|={results_C[best_C]:.4f}, time={timing[best_C]:.2f} ms/f")
        lines.append("")
        lines.append("NOTE: L = Laplacian variance (raw), H = histogram entropy (raw)")
        lines.append("      C = alpha*L_norm + (1-alpha)*H_norm (min-max normalized)")
        lines.append("      If |r(L)| >> |r(C)|, the normalization may be diluting the signal.")
        self._grid_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------
    def _browse_source(self):
        if self._source_type.currentText() == "images":
            path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "",
                "Video (*.mp4 *.avi *.mkv *.mov *.wmv);;All (*)",
            )
        if path:
            self._source_edit.setText(path)

    def _browse_model(self, tag: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select Model-{tag}", "",
            "YOLO models (*.pt *.engine);;All (*)",
        )
        if path:
            if tag == "n":
                self._model_n_edit.setText(path)
            else:
                self._model_s_edit.setText(path)

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------
    def _on_run(self):
        source = self._source_edit.text().strip()
        model_n = self._model_n_edit.text().strip()
        model_s = self._model_s_edit.text().strip()

        if not source or not model_n or not model_s:
            QMessageBox.warning(
                self, "Missing Input",
                "Please specify video source and both model paths.",
            )
            return

        if self._worker is not None and self._worker.isRunning():
            return

        from core.config import InferenceParams
        from validation.validator import DEFAULT_PROXY_SETTINGS

        inf = InferenceParams(
            imgsz=self._imgsz_spin.value(),
            device=self._device_combo.currentText(),
            conf_min=0.001,
            iou_nms=0.45,
            max_frames=self._max_frames_spin.value(),
        )

        self._worker = ValidationWorker(
            source_path=source,
            source_type=self._source_type.currentText(),
            model_n_path=model_n,
            model_s_path=model_s,
            inf_params=inf,
            alpha=self._alpha_spin.value(),
            proxy_settings=DEFAULT_PROXY_SETTINGS,
            max_frames=self._max_frames_spin.value(),
            iou_threshold=self._iou_spin.value(),
            conf_threshold=self._conf_spin.value(),
        )
        self._worker.frame_progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.stopped_early.connect(self._on_stopped)

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setFormat("Loading models...")
        self._progress.setValue(0)

        self._worker.start()

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._progress.setFormat("Stopping...")

    def _on_progress(self, current: int, total: int):
        if total > 0:
            pct = int(100 * current / total)
            self._progress.setMaximum(100)
            self._progress.setValue(pct)
            self._progress.setFormat(f"Frame {current}/{total} ({pct}%) — running both models")
        else:
            self._progress.setFormat(f"Frame {current} (streaming)")

    def _on_finished(self, summary):
        self._summary = summary
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_export_csv.setEnabled(True)
        self._btn_export_corr.setEnabled(True)
        self._btn_export_plots.setEnabled(True)
        self._progress.setValue(100)
        self._progress.setFormat(f"Complete — {summary.total_frames} frames analyzed")

        # Notify other tabs (e.g., Thesis Experiments)
        self.validation_complete.emit(summary)

        # Generate report
        from validation.report import ValidationReport
        self._report = ValidationReport(summary)

        # Display text report
        self._report_text.setPlainText(self._report.generate_text_report())

        # Auto-generate all plots
        self._refresh_all_plots()
        self._cleanup_worker()

    def _on_error(self, msg: str):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setFormat(f"Error: {msg}")
        QMessageBox.critical(self, "Validation Error", msg)
        self._cleanup_worker()

    def _on_stopped(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setFormat("Stopped by user")
        self._cleanup_worker()

    def _cleanup_worker(self):
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.wait(3000)
            try:
                self._worker.frame_progress.disconnect()
                self._worker.finished.disconnect()
                self._worker.error_occurred.disconnect()
                self._worker.stopped_early.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._worker = None

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _refresh_all_plots(self):
        if self._report is None:
            return

        # Scatter (default: C vs Disagreement)
        self._plot_scatter()

        # Time series overlay
        self._ax_ts.clear()
        _apply_dark(self._fig_ts, self._ax_ts)
        self._report.plot_timeseries_overlay(self._ax_ts)
        self._fig_ts.tight_layout()
        self._canvas_ts.draw_idle()

        # Correlation matrix
        self._ax_corr.clear()
        _apply_dark(self._fig_corr, self._ax_corr)
        self._report.plot_correlation_matrix(self._ax_corr, self._fig_corr)
        self._fig_corr.tight_layout()
        self._canvas_corr.draw_idle()

        # Sensitivity heatmap
        self._ax_sens.clear()
        _apply_dark(self._fig_sens, self._ax_sens)
        self._report.plot_sensitivity_heatmap(self._ax_sens)
        self._fig_sens.tight_layout()
        self._canvas_sens.draw_idle()

        # Detection comparison
        self._ax_det.clear()
        _apply_dark(self._fig_det, self._ax_det)
        self._report.plot_detection_comparison(self._ax_det)
        self._fig_det.tight_layout()
        self._canvas_det.draw_idle()

    def _plot_scatter(self):
        if self._report is None:
            return
        self._ax_scatter.clear()
        _apply_dark(self._fig_scatter, self._ax_scatter)
        proxy = self._proxy_combo.currentText()
        metric = self._metric_combo.currentText()
        self._report.plot_correlation_scatter(self._ax_scatter, proxy, metric)
        self._fig_scatter.tight_layout()
        self._canvas_scatter.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_csv(self):
        if self._report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Validation Data", "proxy_validation_data.csv",
            "CSV (*.csv)",
        )
        if path:
            self._report.export_csv(path)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")

    def _export_corr_csv(self):
        if self._report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Correlations", "proxy_correlations.csv",
            "CSV (*.csv)",
        )
        if path:
            self._report.export_correlations_csv(path)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")

    def _export_all_plots(self):
        if self._report is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
        plots = [
            (self._fig_scatter, "correlation_scatter.png"),
            (self._fig_ts, "timeseries_overlay.png"),
            (self._fig_corr, "correlation_matrix.png"),
            (self._fig_sens, "sensitivity_heatmap.png"),
            (self._fig_det, "detection_comparison.png"),
        ]
        for fig, name in plots:
            fig.savefig(
                os.path.join(folder, name), dpi=220,
                facecolor=fig.get_facecolor(), bbox_inches="tight",
            )
        QMessageBox.information(
            self, "Exported",
            f"Saved {len(plots)} plots to:\n{folder}",
        )
