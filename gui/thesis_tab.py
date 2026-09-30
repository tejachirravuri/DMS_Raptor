"""DMS-Raptor Thesis Experiments tab — research-grade analysis dashboard."""
from __future__ import annotations

import logging
import os
import warnings
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QTabWidget, QSplitter,
    QCheckBox, QSizePolicy, QMessageBox, QGridLayout,
    QDoubleSpinBox, QSpinBox, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from core.config import (
    APP_NAME, APP_VERSION, APP_SUBTITLE,
    POLICIES, RunSummary, RunConfig, InferenceParams,
    TRACE_METRICS,
)
from gui.plot_utils import apply_dark, color_for, theme_colors, POLICY_COLORS

# ---------------------------------------------------------------------------
# Helper: wrap a matplotlib canvas in a scroll area with a minimum height
# so plots never get clipped or crushed.
# ---------------------------------------------------------------------------
def _scrollable_canvas(canvas: FigureCanvasQTAgg,
                       min_h: int = 500) -> QScrollArea:
    """Return a QScrollArea wrapping *canvas* with a minimum height."""
    canvas.setMinimumHeight(min_h)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(canvas)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    return scroll

logger = logging.getLogger(__name__)

# Proxy signal names for feature engineering / scatter analysis
_PROXY_SIGNALS = {
    "C-score": "C_trace",
    "Laplacian (L)": "L_trace",
    "Entropy (H)": "H_trace",
    "Mean Confidence": "mean_conf_trace",
    "Conf Drop (EMA)": "conf_drop_trace",
    "NIQE Score": "niqe_trace",
}

_PERF_SIGNALS = {
    "T_total (ms)": "T_total_trace",
    "Detections": "num_detections_trace",
    "Choice (0=n, 1=s)": "choice_trace",
    "T_infer_n (ms)": "T_infer_n_trace",
    "T_infer_s (ms)": "T_infer_s_trace",
}


class ThesisExperimentsTab(QWidget):
    """Comprehensive research analysis dashboard for thesis work."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: List[RunSummary] = []
        self._val_summary = None  # ValidationSummary, if loaded
        self._worker = None
        self._build_ui()

    # ==================================================================
    # UI Construction
    # ==================================================================
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- Top: Control Panel (compact) ----
        ctrl_group = QGroupBox("Experiment Controls")
        ctrl_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        ctrl_layout = QGridLayout(ctrl_group)
        ctrl_layout.setContentsMargins(6, 4, 6, 4)
        ctrl_layout.setVerticalSpacing(3)
        ctrl_layout.setHorizontalSpacing(4)

        # Row 0: Video
        ctrl_layout.addWidget(QLabel("Video:"), 0, 0)
        self._video_edit = QLineEdit()
        self._video_edit.setPlaceholderText("Path to input video...")
        ctrl_layout.addWidget(self._video_edit, 0, 1, 1, 4)
        self._btn_browse_video = QPushButton("…")
        self._btn_browse_video.setFixedWidth(30)
        self._btn_browse_video.clicked.connect(self._browse_video)
        ctrl_layout.addWidget(self._btn_browse_video, 0, 5)

        # Row 1: Model-n + Model-s (on same row)
        ctrl_layout.addWidget(QLabel("Model-n:"), 1, 0)
        self._model_n_edit = QLineEdit()
        self._model_n_edit.setPlaceholderText("YOLOv8n .pt path...")
        ctrl_layout.addWidget(self._model_n_edit, 1, 1)
        btn_n = QPushButton("…")
        btn_n.setFixedWidth(30)
        btn_n.clicked.connect(lambda: self._browse_model("n"))
        ctrl_layout.addWidget(btn_n, 1, 2)

        ctrl_layout.addWidget(QLabel("Model-s:"), 1, 3)
        self._model_s_edit = QLineEdit()
        self._model_s_edit.setPlaceholderText("YOLOv8s .pt path...")
        ctrl_layout.addWidget(self._model_s_edit, 1, 4)
        btn_s = QPushButton("…")
        btn_s.setFixedWidth(30)
        btn_s.clicked.connect(lambda: self._browse_model("s"))
        ctrl_layout.addWidget(btn_s, 1, 5)

        # Row 2: Output
        ctrl_layout.addWidget(QLabel("Output:"), 2, 0)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Output folder for plots & reports...")
        ctrl_layout.addWidget(self._output_edit, 2, 1, 1, 4)
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._browse_output)
        ctrl_layout.addWidget(btn_out, 2, 5)

        # Row 3: Policy filter checkboxes (compact)
        policy_row = QHBoxLayout()
        policy_row.setSpacing(6)
        policy_row.addWidget(QLabel("Policies:"))
        self._policy_checks = {}
        for p in POLICIES:
            cb = QCheckBox(p)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_filter_changed)
            self._policy_checks[p] = cb
            policy_row.addWidget(cb)
        policy_row.addStretch()
        ctrl_layout.addLayout(policy_row, 3, 0, 1, 6)

        # Row 4: Action buttons + progress (merged into one row)
        action_row = QHBoxLayout()
        action_row.setSpacing(4)

        self._btn_load_history = QPushButton("Load History")
        self._btn_load_history.setToolTip("Load all saved run results from disk")
        self._btn_load_history.clicked.connect(self._load_history)
        action_row.addWidget(self._btn_load_history)

        self._btn_run_suite = QPushButton("▶ Run Suite")
        self._btn_run_suite.setStyleSheet(
            "background-color: #00695c; color: white; "
            "font-weight: bold; padding: 6px 14px;"
        )
        self._btn_run_suite.clicked.connect(self._run_experiment)
        action_row.addWidget(self._btn_run_suite)

        self._btn_stop = QPushButton("⏹ Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_experiment)
        action_row.addWidget(self._btn_stop)

        self._btn_export_plots = QPushButton("Export PNGs")
        self._btn_export_plots.clicked.connect(self._export_all_plots)
        action_row.addWidget(self._btn_export_plots)

        self._btn_export_docx = QPushButton("Report (.docx)")
        self._btn_export_docx.clicked.connect(self._export_docx_report)
        action_row.addWidget(self._btn_export_docx)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(140)
        action_row.addWidget(self._progress)

        self._status_label = QLabel("Ready")
        self._status_label.setMinimumWidth(120)
        action_row.addWidget(self._status_label, stretch=1)

        ctrl_layout.addLayout(action_row, 4, 0, 1, 6)

        root.addWidget(ctrl_group)

        # ---- Bottom: Sub-tabs ----
        self._sub_tabs = QTabWidget()
        self._build_latency_dashboard()      # Tab 1
        self._build_switching_analysis()     # Tab 2
        self._build_proxy_signals()          # Tab 3
        self._build_model_comparison()       # Tab 4
        self._build_hyperparam_sensitivity() # Tab 5
        self._build_eda_stats()              # Tab 6
        self._build_feature_engineering()    # Tab 7
        self._build_proxy_validation()       # Tab 8 (⭐ core thesis question)
        self._build_research_summary()       # Tab 9

        root.addWidget(self._sub_tabs, stretch=1)

    # ==================================================================
    # Sub-tab builders
    # ==================================================================

    def _make_canvas(self, fig: Figure) -> FigureCanvasQTAgg:
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return canvas

    # ---- Tab 1: Latency Dashboard ----
    def _build_latency_dashboard(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)
        self._fig_lat = Figure(figsize=(12, 9), constrained_layout=True)
        self._canvas_lat = self._make_canvas(self._fig_lat)
        layout.addWidget(_scrollable_canvas(self._canvas_lat, min_h=580))
        self._sub_tabs.addTab(w, "Latency Dashboard")

    # ---- Tab 2: Switching Analysis ----
    def _build_switching_analysis(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)
        self._fig_sw = Figure(figsize=(14, 14), constrained_layout=True)
        self._canvas_sw = self._make_canvas(self._fig_sw)
        layout.addWidget(_scrollable_canvas(self._canvas_sw, min_h=900))
        self._sub_tabs.addTab(w, "Switching Analysis")

    # ---- Tab 3: Proxy Signals ----
    def _build_proxy_signals(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)
        ctrl.addWidget(QLabel("X:"))
        self._proxy_x = QComboBox()
        self._proxy_x.addItems(list(_PROXY_SIGNALS.keys()))
        ctrl.addWidget(self._proxy_x)
        ctrl.addWidget(QLabel("Y:"))
        self._proxy_y = QComboBox()
        self._proxy_y.addItems(list(_PERF_SIGNALS.keys()))
        ctrl.addWidget(self._proxy_y)
        self._btn_proxy_plot = QPushButton("Plot")
        self._btn_proxy_plot.clicked.connect(self._refresh_proxy_scatter)
        ctrl.addWidget(self._btn_proxy_plot)
        self._btn_proxy_all = QPushButton("All Combinations")
        self._btn_proxy_all.clicked.connect(self._export_proxy_grid)
        ctrl.addWidget(self._btn_proxy_all)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._fig_proxy = Figure(figsize=(10, 7), constrained_layout=True)
        self._ax_proxy = self._fig_proxy.add_subplot(111)
        self._canvas_proxy = self._make_canvas(self._fig_proxy)
        apply_dark(self._fig_proxy, [self._ax_proxy])
        layout.addWidget(_scrollable_canvas(self._canvas_proxy, min_h=480))
        self._sub_tabs.addTab(w, "Proxy Signals")

    # ---- Tab 4: Model Comparison ----
    def _build_model_comparison(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)

        btn_row = QHBoxLayout()
        self._btn_load_val = QPushButton("Load Validation Data")
        self._btn_load_val.clicked.connect(self._load_validation_data)
        btn_row.addWidget(self._btn_load_val)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._fig_mc = Figure(figsize=(12, 9), constrained_layout=True)
        self._canvas_mc = self._make_canvas(self._fig_mc)
        layout.addWidget(_scrollable_canvas(self._canvas_mc, min_h=580))
        self._sub_tabs.addTab(w, "Model Comparison")

    # ---- Tab 5: Hyperparameter Sensitivity ----
    def _build_hyperparam_sensitivity(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)
        ctrl.addWidget(QLabel("c_low:"))
        self._hp_clow_min = QDoubleSpinBox()
        self._hp_clow_min.setRange(0.0, 1.0)
        self._hp_clow_min.setValue(0.1)
        self._hp_clow_min.setSingleStep(0.05)
        ctrl.addWidget(self._hp_clow_min)
        ctrl.addWidget(QLabel("–"))
        self._hp_clow_max = QDoubleSpinBox()
        self._hp_clow_max.setRange(0.0, 1.0)
        self._hp_clow_max.setValue(0.8)
        self._hp_clow_max.setSingleStep(0.05)
        ctrl.addWidget(self._hp_clow_max)

        ctrl.addWidget(QLabel("c_high:"))
        self._hp_chigh_min = QDoubleSpinBox()
        self._hp_chigh_min.setRange(0.0, 1.0)
        self._hp_chigh_min.setValue(0.2)
        self._hp_chigh_min.setSingleStep(0.05)
        ctrl.addWidget(self._hp_chigh_min)
        ctrl.addWidget(QLabel("–"))
        self._hp_chigh_max = QDoubleSpinBox()
        self._hp_chigh_max.setRange(0.0, 1.0)
        self._hp_chigh_max.setValue(0.9)
        self._hp_chigh_max.setSingleStep(0.05)
        ctrl.addWidget(self._hp_chigh_max)

        ctrl.addWidget(QLabel("Step:"))
        self._hp_step = QDoubleSpinBox()
        self._hp_step.setRange(0.01, 0.2)
        self._hp_step.setValue(0.05)
        self._hp_step.setSingleStep(0.01)
        ctrl.addWidget(self._hp_step)

        self._btn_hp_run = QPushButton("Run Sweep")
        self._btn_hp_run.clicked.connect(self._refresh_hyperparam)
        ctrl.addWidget(self._btn_hp_run)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._fig_hp = Figure(figsize=(14, 6), constrained_layout=True)
        self._canvas_hp = self._make_canvas(self._fig_hp)
        layout.addWidget(_scrollable_canvas(self._canvas_hp, min_h=480))
        self._sub_tabs.addTab(w, "Hyperparameter Sensitivity")

    # ---- Tab 6: EDA ----
    def _build_eda_stats(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)

        splitter = QSplitter(Qt.Vertical)

        # Top: plots in scroll area
        self._fig_eda = Figure(figsize=(12, 9), constrained_layout=True)
        self._canvas_eda = self._make_canvas(self._fig_eda)
        splitter.addWidget(_scrollable_canvas(self._canvas_eda, min_h=560))

        # Bottom: stats table
        self._eda_table = QTableWidget()
        self._eda_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._eda_table.setAlternatingRowColors(True)
        self._eda_table.setMaximumHeight(200)
        splitter.addWidget(self._eda_table)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
        self._sub_tabs.addTab(w, "EDA — Statistics")

    # ---- Tab 7: Feature Engineering ----
    def _build_feature_engineering(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)
        self._fig_feat = Figure(figsize=(14, 6), constrained_layout=True)
        self._canvas_feat = self._make_canvas(self._fig_feat)
        layout.addWidget(_scrollable_canvas(self._canvas_feat, min_h=480))
        self._sub_tabs.addTab(w, "Feature Engineering")

    # ---- Tab 9: Proxy Validation ----
    def _build_proxy_validation(self):
        """Sub-tab that answers: When L/H/C say 'switch to s', does s do better?"""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)
        self._btn_load_val_pv = QPushButton("Load Validation Data")
        self._btn_load_val_pv.clicked.connect(self._load_validation_data)
        ctrl.addWidget(self._btn_load_val_pv)

        ctrl.addWidget(QLabel("Proxy:"))
        self._pv_proxy = QComboBox()
        self._pv_proxy.addItems(["C-score", "Laplacian (L)", "Entropy (H)", "NIQE"])
        ctrl.addWidget(self._pv_proxy)

        ctrl.addWidget(QLabel("Metric:"))
        self._pv_metric = QComboBox()
        self._pv_metric.addItems([
            "Det count gap (s−n)", "Conf gap (s−n)",
            "Disagreement score", "Extra dets s",
        ])
        ctrl.addWidget(self._pv_metric)

        self._btn_pv_refresh = QPushButton("Update Scatter")
        self._btn_pv_refresh.clicked.connect(self._refresh_proxy_validation)
        ctrl.addWidget(self._btn_pv_refresh)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Info label explaining what this tab proves
        info = QLabel(
            "⚡ <b>Core thesis question</b>: When proxies say 'switch to s', "
            "does s actually detect more / with higher confidence? "
            "Positive correlation = proxy is a valid switching signal."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #26c6da; padding: 4px; font-size: 11px;")
        layout.addWidget(info)

        self._fig_pv = Figure(figsize=(14, 10), constrained_layout=True)
        self._canvas_pv = self._make_canvas(self._fig_pv)
        layout.addWidget(_scrollable_canvas(self._canvas_pv, min_h=600))

        # Interpretation panel
        self._pv_interp = QTextEdit()
        self._pv_interp.setReadOnly(True)
        self._pv_interp.setMaximumHeight(160)
        self._pv_interp.setStyleSheet(
            "font-family: 'Segoe UI', Arial; font-size: 12px; "
            "background-color: #16213e; color: #e0e0e0; border: 1px solid #2a3a5c;"
        )
        layout.addWidget(self._pv_interp)

        self._sub_tabs.addTab(w, "⭐ Proxy Validation")

    # ---- Tab 8: Research Summary ----
    def _build_research_summary(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self._summary_html = QTextEdit()
        self._summary_html.setReadOnly(True)
        self._summary_html.setStyleSheet(
            "font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px;"
        )
        layout.addWidget(self._summary_html)

        btn_row = QHBoxLayout()
        self._btn_summary_docx = QPushButton("Download Research Report (.docx)")
        self._btn_summary_docx.clicked.connect(self._export_docx_report)
        btn_row.addWidget(self._btn_summary_docx)
        self._btn_summary_export = QPushButton("Export All Plots (.png)")
        self._btn_summary_export.clicked.connect(self._export_all_plots)
        btn_row.addWidget(self._btn_summary_export)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._sub_tabs.addTab(w, "Research Summary")

    # ==================================================================
    # Browse helpers
    # ==================================================================
    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video",
            "",
            "Video files (*.mp4 *.avi *.mkv *.mov);;All files (*)",
        )
        if path:
            self._video_edit.setText(path)

    def _browse_model(self, tag):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select Model-{tag}",
            "",
            "YOLO weights (*.pt *.engine);;All files (*)",
        )
        if path:
            if tag == "n":
                self._model_n_edit.setText(path)
            else:
                self._model_s_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._output_edit.setText(path)

    # ==================================================================
    # Data management
    # ==================================================================
    def _get_filtered_results(self) -> List[RunSummary]:
        """Return results filtered by checked policies."""
        active = {p for p, cb in self._policy_checks.items() if cb.isChecked()}
        return [r for r in self._results if r.policy in active]

    def add_run_result(self, summary: RunSummary):
        """Called by MainWindow when a policy finishes."""
        self._results.append(summary)
        self.refresh_plots()

    def _load_history(self):
        """Load all historical runs from disk."""
        try:
            from core.history import load_all_runs
            app_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            runs = load_all_runs(app_root)
            if runs:
                for r in runs:
                    r.is_history = True
                    self._results.append(r)
                self._status_label.setText(
                    f"Loaded {len(runs)} historical run(s)."
                )
                self.refresh_plots()
            else:
                self._status_label.setText("No history found.")
        except Exception as exc:
            logger.exception("Failed to load history")
            QMessageBox.warning(self, "Load Error", str(exc))

    def _load_validation_data(self):
        """Load validation summary from a JSON file or run validation."""
        QMessageBox.information(
            self, "Model Comparison",
            "Run a validation from the Validation tab first, then the data "
            "will be available here.\n\nAlternatively, use 'Run Full Experiment "
            "Suite' which includes validation.",
        )

    def _on_filter_changed(self, _=None):
        self.refresh_plots()

    # ==================================================================
    # Run experiment suite
    # ==================================================================
    def _run_experiment(self):
        video = self._video_edit.text().strip()
        model_n = self._model_n_edit.text().strip()
        model_s = self._model_s_edit.text().strip()
        output = self._output_edit.text().strip()

        if not video or not os.path.isfile(video):
            QMessageBox.warning(self, "Missing Input", "Select a valid video file.")
            return
        if not model_n or not os.path.isfile(model_n):
            QMessageBox.warning(self, "Missing Model", "Select a valid Model-n path.")
            return
        if not model_s or not os.path.isfile(model_s):
            QMessageBox.warning(self, "Missing Model", "Select a valid Model-s path.")
            return

        if not output:
            output = os.path.join(os.path.dirname(video), "thesis_output")
        os.makedirs(output, exist_ok=True)

        selected_policies = [
            p for p, cb in self._policy_checks.items() if cb.isChecked()
        ]
        if not selected_policies:
            QMessageBox.warning(self, "No Policies", "Select at least one policy.")
            return

        self._btn_run_suite.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setValue(0)
        self._status_label.setText("Starting experiment suite...")

        from gui.thesis_worker import ThesisExperimentWorker

        inf_params = InferenceParams()  # defaults
        base_config = RunConfig()

        self._worker = ThesisExperimentWorker(
            source_path=video,
            source_type="video",
            model_n_path=model_n,
            model_s_path=model_s,
            inf_params=inf_params,
            policies=selected_policies,
            base_config=base_config,
            output_dir=output,
            run_validation=True,
            save_video=True,
        )
        self._worker.status_update.connect(self._on_status)
        self._worker.progress.connect(self._on_progress)
        self._worker.policy_done.connect(self._on_policy_done)
        self._worker.validation_done.connect(self._on_validation_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.start()

    def _stop_experiment(self):
        if self._worker is not None:
            self._worker.request_stop()
            self._btn_stop.setEnabled(False)
            self._status_label.setText("Stopping…")

    def _on_status(self, msg: str):
        self._status_label.setText(msg)

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self._progress.setValue(int(100 * current / total))

    def _on_policy_done(self, summary):
        self._results.append(summary)
        self._status_label.setText(
            f"Completed: {summary.policy} — "
            f"Mean T_total: {summary.T_total_ms_mean:.1f} ms"
        )
        self.refresh_plots()

    def _on_validation_done(self, val_summary):
        self._val_summary = val_summary
        self._status_label.setText("Validation complete.")
        self._refresh_model_comparison()
        try:
            self._refresh_proxy_validation()
        except Exception:
            logger.debug("Proxy validation refresh failed", exc_info=True)

    def _on_all_done(self, summaries, val_summary):
        self._val_summary = val_summary
        self._btn_run_suite.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setValue(100)
        self._status_label.setText(
            f"Experiment suite complete — {len(summaries)} policies processed."
        )
        self.refresh_plots()
        self._cleanup_worker()

    def _on_stopped(self):
        """Worker acknowledged the stop — reset UI cleanly."""
        n = len(self._results)
        self._btn_run_suite.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setValue(0)
        self._status_label.setText(
            f"Stopped by user — {n} polic{'y' if n == 1 else 'ies'} completed before stop."
        )
        if self._results:
            self.refresh_plots()
        self._cleanup_worker()

    def _on_error(self, msg: str):
        self._btn_run_suite.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Experiment Error", msg)
        self._cleanup_worker()

    def _cleanup_worker(self):
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.quit()
                if not self._worker.wait(5000):
                    logger.warning("ThesisExperimentWorker did not stop within 5 s — terminating")
                    self._worker.terminate()
                    self._worker.wait(2000)
            try:
                self._worker.status_update.disconnect()
                self._worker.progress.disconnect()
                self._worker.policy_done.disconnect()
                self._worker.validation_done.disconnect()
                self._worker.all_done.disconnect()
                self._worker.error_occurred.disconnect()
                self._worker.stopped.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._worker.deleteLater()
            self._worker = None

    # ==================================================================
    # Plot refresh — delegates to thesis_plots.py
    # ==================================================================
    def refresh_plots(self):
        """Refresh all sub-tab plots with current data."""
        filtered = self._get_filtered_results()
        if not filtered:
            return
        try:
            self._refresh_latency_dashboard(filtered)
        except Exception:
            logger.debug("Latency dashboard refresh failed", exc_info=True)
        try:
            self._refresh_switching_analysis(filtered)
        except Exception:
            logger.debug("Switching analysis refresh failed", exc_info=True)
        try:
            self._refresh_proxy_scatter()
        except Exception:
            logger.debug("Proxy scatter refresh failed", exc_info=True)
        try:
            self._refresh_model_comparison()
        except Exception:
            logger.debug("Model comparison refresh failed", exc_info=True)
        try:
            self._refresh_eda(filtered)
        except Exception:
            logger.debug("EDA refresh failed", exc_info=True)
        try:
            self._refresh_feature_engineering(filtered)
        except Exception:
            logger.debug("Feature engineering refresh failed", exc_info=True)
        try:
            self._refresh_proxy_validation()
        except Exception:
            logger.debug("Proxy validation refresh failed", exc_info=True)
        try:
            self._refresh_research_summary(filtered)
        except Exception:
            logger.debug("Research summary refresh failed", exc_info=True)

    # ---- Tab 1: Latency Dashboard ----
    def _refresh_latency_dashboard(self, filtered=None):
        if filtered is None:
            filtered = self._get_filtered_results()
        if not filtered:
            return
        from gui.thesis_plots import (
            plot_latency_timeseries, plot_latency_cdf,
            plot_timing_breakdown, plot_latency_boxplot,
        )
        self._fig_lat.clf()
        self._fig_lat.set_constrained_layout(True)
        axes = self._fig_lat.subplots(2, 2)
        apply_dark(self._fig_lat, list(axes.flat))

        plot_latency_timeseries(self._fig_lat, axes[0, 0], filtered)
        plot_latency_cdf(self._fig_lat, axes[0, 1], filtered)
        plot_timing_breakdown(self._fig_lat, axes[1, 0], filtered)
        plot_latency_boxplot(self._fig_lat, axes[1, 1], filtered)

        self._canvas_lat.draw_idle()

    # ---- Tab 2: Switching Analysis ----
    def _refresh_switching_analysis(self, filtered=None):
        if filtered is None:
            filtered = self._get_filtered_results()
        if not filtered:
            return
        from gui.thesis_plots import (
            plot_choice_heatmap, plot_switching_rate, plot_dwell_histogram,
        )
        switching = [r for r in filtered if r.policy not in ("n_only", "s_only")]
        if not switching:
            return

        n = len(switching)
        # Dynamically resize the figure height based on number of policies
        fig_h = max(12, 1.8 * n + 8)
        self._fig_sw.set_size_inches(14, fig_h)
        self._canvas_sw.setMinimumHeight(int(fig_h * 80))  # generous pixel height

        self._fig_sw.clf()
        self._fig_sw.set_constrained_layout(True)
        # Top: choice heatmaps (one row per policy)
        # Bottom: switching rate + dwell histogram
        gs = self._fig_sw.add_gridspec(
            n + 2, 1,
            height_ratios=[1] * n + [4, 4],
            hspace=0.45,
        )
        heatmap_axes = [self._fig_sw.add_subplot(gs[i]) for i in range(n)]
        ax_rate = self._fig_sw.add_subplot(gs[n])
        ax_dwell = self._fig_sw.add_subplot(gs[n + 1])

        all_axes = heatmap_axes + [ax_rate, ax_dwell]
        apply_dark(self._fig_sw, all_axes)

        plot_choice_heatmap(self._fig_sw, heatmap_axes, switching)
        plot_switching_rate(self._fig_sw, ax_rate, switching)
        plot_dwell_histogram(self._fig_sw, ax_dwell, switching)

        self._canvas_sw.draw_idle()

    # ---- Tab 3: Proxy Scatter ----
    def _refresh_proxy_scatter(self):
        filtered = self._get_filtered_results()
        if not filtered:
            return
        from gui.thesis_plots import plot_proxy_scatter

        x_name = self._proxy_x.currentText()
        y_name = self._proxy_y.currentText()
        x_attr = _PROXY_SIGNALS.get(x_name, "C_trace")
        y_attr = _PERF_SIGNALS.get(y_name, "T_total_trace")

        self._ax_proxy.clear()
        apply_dark(self._fig_proxy, [self._ax_proxy])
        plot_proxy_scatter(
            self._fig_proxy, self._ax_proxy, filtered,
            x_attr, y_attr, x_name, y_name,
        )
        self._canvas_proxy.draw_idle()

    def _export_proxy_grid(self):
        """Export a grid of all proxy-vs-metric scatter plots."""
        filtered = self._get_filtered_results()
        if not filtered:
            QMessageBox.information(self, "No Data", "Load data first.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder for proxy grid plots"
        )
        if not folder:
            return
        from gui.thesis_plots import plot_proxy_scatter

        video_stem = self._get_video_stem()
        for x_name, x_attr in _PROXY_SIGNALS.items():
            for y_name, y_attr in _PERF_SIGNALS.items():
                fig = Figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                apply_dark(fig, [ax])
                plot_proxy_scatter(fig, ax, filtered, x_attr, y_attr, x_name, y_name)
                fig.tight_layout(pad=1.2)
                safe_x = x_name.replace(" ", "_").replace("(", "").replace(")", "")
                safe_y = y_name.replace(" ", "_").replace("(", "").replace(")", "")
                fname = f"scatter_{safe_x}_vs_{safe_y}_{video_stem}.png"
                fig.savefig(
                    os.path.join(folder, fname), dpi=300,
                    facecolor=fig.get_facecolor(), bbox_inches="tight",
                )

        self._status_label.setText(f"Exported proxy scatter grid to {folder}")

    # ---- Tab 4: Model Comparison ----
    def _refresh_model_comparison(self):
        if self._val_summary is None:
            return
        from gui.thesis_plots import (
            plot_detection_scatter, plot_confidence_comparison,
        )
        self._fig_mc.clf()
        self._fig_mc.set_constrained_layout(True)
        axes = self._fig_mc.subplots(2, 2)
        apply_dark(self._fig_mc, list(axes.flat))

        plot_detection_scatter(self._fig_mc, axes[0, 0], self._val_summary)
        plot_confidence_comparison(self._fig_mc, axes[0, 1], self._val_summary)

        # Latency comparison bar (n vs s)
        tc = theme_colors()
        ax = axes[1, 0]
        filtered = self._get_filtered_results()
        n_runs = [r for r in filtered if r.policy == "n_only"]
        s_runs = [r for r in filtered if r.policy == "s_only"]
        if n_runs and s_runs:
            labels = ["Mean", "P95", "P99"]
            n_vals = [n_runs[0].T_total_ms_mean, n_runs[0].T_total_ms_p95, n_runs[0].T_total_ms_p99]
            s_vals = [s_runs[0].T_total_ms_mean, s_runs[0].T_total_ms_p95, s_runs[0].T_total_ms_p99]
            x = np.arange(len(labels))
            ax.bar(x - 0.2, n_vals, 0.35, color="#4caf50", label="YOLOv8n")
            ax.bar(x + 0.2, s_vals, 0.35, color="#e94560", label="YOLOv8s")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Latency (ms)", color=tc["fg"])
            ax.set_title("Latency: n vs s", color=tc["fg"])
            ax.legend(fontsize=8, facecolor=tc["legend_bg"],
                      edgecolor=tc["legend_edge"], labelcolor=tc["fg"])

        # IoU agreement histogram
        ax = axes[1, 1]
        try:
            iou_vals = getattr(self._val_summary, "iou_agreements", [])
            if iou_vals:
                ax.hist(iou_vals, bins=30, color="#00bcd4", alpha=0.7, edgecolor="#16213e")
                ax.set_xlabel("IoU Agreement", color=tc["fg"])
                ax.set_ylabel("Frames", color=tc["fg"])
                ax.set_title("IoU Agreement Distribution", color=tc["fg"])
        except Exception:
            ax.set_title("IoU data not available", color=tc["fg"])

        self._canvas_mc.draw_idle()

    # ---- Tab 5: Hyperparameter Sensitivity ----
    def _refresh_hyperparam(self):
        filtered = self._get_filtered_results()
        if not filtered:
            QMessageBox.information(self, "No Data", "Load data first.")
            return
        from gui.thesis_plots import plot_threshold_heatmap, plot_pareto_front

        # Find a switching policy run with C_trace
        run = None
        for r in filtered:
            if r.C_trace and any(v != 0 for v in r.C_trace):
                run = r
                break
        if run is None:
            QMessageBox.information(
                self, "No C-score Data",
                "Need a run with C-score trace (entropy_only, combined, or combined_hyst).",
            )
            return

        c_low_range = (
            self._hp_clow_min.value(),
            self._hp_clow_max.value(),
            self._hp_step.value(),
        )
        c_high_range = (
            self._hp_chigh_min.value(),
            self._hp_chigh_max.value(),
            self._hp_step.value(),
        )

        self._fig_hp.clf()
        self._fig_hp.set_constrained_layout(True)
        axes = self._fig_hp.subplots(1, 3)
        apply_dark(self._fig_hp, axes)

        plot_threshold_heatmap(
            self._fig_hp, axes[0], run, metric="sw_per_100",
            c_low_range=c_low_range, c_high_range=c_high_range,
        )
        plot_threshold_heatmap(
            self._fig_hp, axes[1], run, metric="s_usage_pct",
            c_low_range=c_low_range, c_high_range=c_high_range,
        )
        plot_pareto_front(
            self._fig_hp, axes[2], run,
            c_low_range=c_low_range, c_high_range=c_high_range,
        )

        self._canvas_hp.draw_idle()
        self._status_label.setText("Hyperparameter sweep complete.")

    # ---- Tab 6: EDA ----
    def _refresh_eda(self, filtered=None):
        if filtered is None:
            filtered = self._get_filtered_results()
        if not filtered:
            return
        from gui.thesis_plots import (
            plot_distribution_kde, plot_outlier_detection,
            plot_full_correlation_matrix,
        )
        tc = theme_colors()

        self._fig_eda.clf()
        self._fig_eda.set_constrained_layout(True)
        axes = self._fig_eda.subplots(2, 2)
        apply_dark(self._fig_eda, list(axes.flat))

        plot_distribution_kde(self._fig_eda, axes[0, 0], filtered)

        # Q-Q plot (simple: sorted data vs theoretical normal quantiles)
        ax_qq = axes[0, 1]
        for i, r in enumerate(filtered[:4]):  # max 4 policies
            if not r.T_total_trace:
                continue
            data = np.sort(r.T_total_trace)
            n = len(data)
            theoretical = np.linspace(0.5/n, 1 - 0.5/n, n)
            from_normal = np.sqrt(2) * self._erfinv_approx(2 * theoretical - 1)
            from_normal = from_normal * np.std(data) + np.mean(data)
            c = color_for(r.policy, i)
            ax_qq.scatter(from_normal, data, s=4, color=c, alpha=0.5, label=r.policy)
        lims = ax_qq.get_xlim()
        ax_qq.plot(lims, lims, "r--", linewidth=0.8, alpha=0.5)
        ax_qq.set_xlabel("Theoretical Quantiles", color=tc["fg"])
        ax_qq.set_ylabel("Sample Quantiles", color=tc["fg"])
        ax_qq.set_title("Q-Q Plot (T_total)", color=tc["fg"])
        if len(filtered) <= 6:
            ax_qq.legend(fontsize=6, facecolor=tc["legend_bg"],
                         edgecolor=tc["legend_edge"], labelcolor=tc["fg"])

        if filtered:
            plot_outlier_detection(self._fig_eda, axes[1, 0], filtered[0])
        plot_full_correlation_matrix(self._fig_eda, axes[1, 1], filtered[0] if filtered else None)

        self._canvas_eda.draw_idle()

        # Stats table
        self._fill_eda_table(filtered)

    @staticmethod
    def _erfinv_approx(x):
        """Approximate inverse error function for Q-Q plots (no scipy needed)."""
        # Winitzki approximation
        a = 0.147
        ln1mx2 = np.log(np.clip(1 - x * x, 1e-15, None))
        s = np.sign(x)
        t = 2 / (np.pi * a) + ln1mx2 / 2
        return s * np.sqrt(np.sqrt(t * t - ln1mx2 / a) - t)

    def _fill_eda_table(self, filtered):
        """Fill the descriptive statistics table."""
        stats = ["Mean", "Median", "Std", "Min", "Max", "P95", "P99", "IQR", "Skew", "Kurtosis"]
        self._eda_table.setColumnCount(len(stats))
        self._eda_table.setHorizontalHeaderLabels(stats)
        self._eda_table.setRowCount(len(filtered))
        labels = []
        for i, r in enumerate(filtered):
            labels.append(r.policy)
            if not r.T_total_trace:
                continue
            d = np.array(r.T_total_trace)
            q25, q75 = np.percentile(d, [25, 75])
            n = len(d)
            mean = np.mean(d)
            std = np.std(d)
            vals = [
                f"{mean:.2f}",
                f"{np.median(d):.2f}",
                f"{std:.2f}",
                f"{np.min(d):.2f}",
                f"{np.max(d):.2f}",
                f"{np.percentile(d, 95):.2f}",
                f"{np.percentile(d, 99):.2f}",
                f"{q75 - q25:.2f}",
                f"{np.mean(((d - mean) / (std + 1e-9)) ** 3):.3f}" if n > 2 else "N/A",
                f"{np.mean(((d - mean) / (std + 1e-9)) ** 4) - 3:.3f}" if n > 3 else "N/A",
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self._eda_table.setItem(i, c, item)

        self._eda_table.setVerticalHeaderLabels(labels)
        hdr = self._eda_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)

    # ---- Tab 7: Feature Engineering ----
    def _refresh_feature_engineering(self, filtered=None):
        if filtered is None:
            filtered = self._get_filtered_results()
        if not filtered:
            return
        from gui.thesis_plots import (
            plot_feature_importance, plot_decision_boundary,
            plot_signal_lag_xcorr,
        )
        # Find a switching policy with choice trace
        switching = [
            r for r in filtered
            if r.choice_trace and r.policy not in ("n_only", "s_only")
        ]
        if not switching:
            return

        self._fig_feat.clf()
        self._fig_feat.set_constrained_layout(True)
        axes = self._fig_feat.subplots(1, 3)
        apply_dark(self._fig_feat, axes)

        plot_feature_importance(self._fig_feat, axes[0], switching)
        plot_decision_boundary(self._fig_feat, axes[1], switching[0])
        # Auto-pick best signal attr for cross-correlation
        s0 = switching[0]
        if s0.policy == "conf_ema" and getattr(s0, "conf_drop_trace", None):
            sig_attr = "conf_drop_trace"
        elif s0.policy == "niqe_switch" and getattr(s0, "niqe_trace", None):
            sig_attr = "niqe_trace"
        elif getattr(s0, "C_trace", None) and any(v != 0 for v in s0.C_trace[:20]):
            sig_attr = "C_trace"
        else:
            sig_attr = "mean_conf_trace"  # fallback
        plot_signal_lag_xcorr(self._fig_feat, axes[2], s0, signal_attr=sig_attr)

        self._canvas_feat.draw_idle()

    # ---- Tab 9 (⭐): Proxy Validation ----
    def _refresh_proxy_validation(self):
        """Refresh the proxy validation dashboard — the core thesis defense plot."""
        if self._val_summary is None:
            return
        from gui.thesis_plots import (
            plot_proxy_validation_dashboard,
            plot_proxy_vs_s_benefit,
            plot_proxy_roc_analysis,
            pearsonr,
        )

        # Draw the full 2×2 dashboard
        plot_proxy_validation_dashboard(self._fig_pv, self._val_summary)
        self._canvas_pv.draw_idle()

        # Generate interpretation text
        self._generate_pv_interpretation()

    def _generate_pv_interpretation(self):
        """Auto-generate a thesis-ready interpretation of proxy validation."""
        if self._val_summary is None:
            return
        from gui.thesis_plots import pearsonr

        val = self._val_summary
        tc_html = "#26c6da"

        lines = []
        lines.append(f"<h3 style='color:{tc_html};'>Proxy Validation Interpretation</h3>")

        # Compute key correlations
        proxy_results = []
        for name, attr in [("L (Laplacian)", "L_trace"),
                           ("H (Entropy)", "H_trace"),
                           ("C (Combined)", "C_trace"),
                           ("NIQE", "niqe_scores")]:
            pdata = np.asarray(getattr(val, attr, []), dtype=float)
            det_gap = np.asarray(getattr(val, "det_count_gaps", []), dtype=float)
            conf_gap = np.asarray(getattr(val, "mean_conf_gaps", []), dtype=float)
            n = min(len(pdata), len(det_gap), len(conf_gap))
            if n < 10:
                continue
            r_det, p_det = pearsonr(pdata[:n], det_gap[:n])
            r_conf, p_conf = pearsonr(pdata[:n], conf_gap[:n])
            proxy_results.append((name, r_det, p_det, r_conf, p_conf))

        if not proxy_results:
            self._pv_interp.setHtml("<p>No validation data available.</p>")
            return

        # Correlation table
        lines.append("<table border='1' cellpadding='5' style='border-color:#2a3a5c; "
                     "color:#e0e0e0; border-collapse:collapse; width:100%;'>")
        lines.append("<tr style='background:#1a1a2e;'>"
                     "<th>Proxy</th><th>r (det gap)</th><th>p-value</th>"
                     "<th>r (conf gap)</th><th>p-value</th><th>Verdict</th></tr>")

        for name, r_det, p_det, r_conf, p_conf in proxy_results:
            # Determine color and verdict
            best_r = max(abs(r_det), abs(r_conf))
            if best_r > 0.3 and min(p_det, p_conf) < 0.01:
                verdict = "✅ Strong signal"
                color = "#4caf50"
            elif best_r > 0.15 and min(p_det, p_conf) < 0.05:
                verdict = "⚠️ Moderate signal"
                color = "#ff9800"
            else:
                verdict = "❌ Weak signal"
                color = "#e94560"

            lines.append(
                f"<tr><td><b>{name}</b></td>"
                f"<td>{r_det:+.3f}</td><td>{p_det:.2e}</td>"
                f"<td>{r_conf:+.3f}</td><td>{p_conf:.2e}</td>"
                f"<td style='color:{color};'><b>{verdict}</b></td></tr>"
            )

        lines.append("</table>")

        # Auto-generate thesis-safe paragraph
        strong = [name for name, r_d, p_d, r_c, p_c in proxy_results
                  if max(abs(r_d), abs(r_c)) > 0.15 and min(p_d, p_c) < 0.05]
        if strong:
            lines.append(
                f"<p style='margin-top:8px;'><b>Thesis conclusion:</b> "
                f"The proxy signals <b>{', '.join(strong)}</b> show statistically "
                f"significant correlation with the performance gap between models. "
                f"This validates their use as switching signals — when the proxy "
                f"indicates higher complexity, the heavier model genuinely provides "
                f"more detections and/or higher confidence.</p>"
            )
        else:
            lines.append(
                "<p style='margin-top:8px;'><b>Note:</b> No proxy showed a strong "
                "linear correlation on this video. This may indicate: (a) the video "
                "has uniform complexity, (b) both models perform similarly, or "
                "(c) the relationship is non-linear. Check the conditional bar chart "
                "for bin-wise analysis.</p>"
            )

        # Frame-count summary
        n_frames = getattr(val, "total_frames", 0)
        if n_frames:
            det_gap = np.asarray(val.det_count_gaps, dtype=float)
            pct_s_better = float(np.mean(det_gap > 0)) * 100
            pct_n_better = float(np.mean(det_gap < 0)) * 100
            pct_equal = float(np.mean(det_gap == 0)) * 100
            lines.append(
                f"<p style='font-size:11px; color:#aaa;'>"
                f"Frames: {n_frames} total | "
                f"s better: {pct_s_better:.1f}% | "
                f"n better: {pct_n_better:.1f}% | "
                f"equal: {pct_equal:.1f}%</p>"
            )

        self._pv_interp.setHtml("".join(lines))

    # ---- Tab 8: Research Summary ----
    def _refresh_research_summary(self, filtered=None):
        if filtered is None:
            filtered = self._get_filtered_results()
        if not filtered:
            return
        tc = theme_colors()
        bg = tc["bg"]
        fg = tc["fg"]
        accent = "#00bcd4"

        html_parts = []
        html_parts.append(f"""
        <div style="font-family: Segoe UI, Arial; color: {fg};">
        <h1 style="color: {accent};">{APP_NAME} v{APP_VERSION} — Experiment Report</h1>
        <p><em>{APP_SUBTITLE}</em></p>
        <hr style="border-color: {accent};">
        """)

        # System Overview
        html_parts.append(f"""
        <h2 style="color: {accent};">1. System Architecture</h2>
        <p>DMS-Raptor implements <b>complexity-aware dynamic model switching</b>
        for real-time object detection on UAV platforms. The system switches between
        two YOLO models at runtime:</p>
        <ul>
        <li><b>YOLOv8n</b> (nano) — fast, lower accuracy, ~4-8 ms inference</li>
        <li><b>YOLOv8s</b> (small) — slower, higher accuracy, ~10-20 ms inference</li>
        </ul>
        """)

        # Per-frame pipeline
        html_parts.append(f"""
        <h2 style="color: {accent};">2. Per-Frame Pipeline</h2>
        <p>Each video frame passes through this pipeline:</p>
        <ol>
        <li><b>Frame Capture</b> — <code>cv2.VideoCapture.read()</code></li>
        <li><b>Scene Proxy Computation</b> — <code>complexity_proxies_fast()</code> resizes
            to proxy_size, computes Laplacian variance (L) and histogram entropy (H)</li>
        <li><b>Normalization</b> — L and H are normalized to [0,1] via rolling percentiles,
            combined: <em>C = α·L_norm + (1−α)·H_norm</em></li>
        <li><b>Controller Decision</b> — policy-specific switching logic with hysteresis,
            dwell guard, and switch rate limiter</li>
        <li><b>Model Inference</b> — <code>run_model_single()</code> runs the chosen YOLO model</li>
        <li><b>Overlay & Accumulation</b> — annotate frame, accumulate traces</li>
        </ol>
        """)

        # Mathematical Foundations
        html_parts.append(f"""
        <h2 style="color: {accent};">3. Mathematical Foundations</h2>
        <table border="1" cellpadding="8" cellspacing="0" style="border-color: #2a3a5c; color: {fg};">
        <tr style="background-color: #16213e;">
            <th>Signal</th><th>Formula</th><th>Purpose</th>
        </tr>
        <tr><td>Laplacian (L)</td>
            <td><em>L = Var(∇²I)</em>, where ∇²I is the discrete Laplacian of the grayscale proxy</td>
            <td>Measures image sharpness/blur</td></tr>
        <tr><td>Entropy (H)</td>
            <td><em>H = −Σ pᵢ · log₂(pᵢ)</em>, over hist_bins intensity bins</td>
            <td>Measures scene complexity/texture</td></tr>
        <tr><td>Combined (C)</td>
            <td><em>C = α · L_norm + (1−α) · H_norm</em></td>
            <td>Unified complexity score, α=0.6</td></tr>
        <tr><td>Conf EMA rise</td>
            <td><em>EMA_fast = β_f · x + (1−β_f) · EMA_prev</em>;<br>
                <em>rise = max(0, (EMA_fast − EMA_slow) / EMA_slow)</em></td>
            <td>Detects confidence degradation</td></tr>
        <tr><td>NIQE</td>
            <td><em>μ̂(i,j) = (I(i,j) − μ(i,j)) / (σ(i,j) + 1)</em>, then GGD shape β</td>
            <td>No-reference image quality assessment</td></tr>
        <tr><td>Hysteresis</td>
            <td>if signal &gt; c_high → s; if signal &lt; c_low → n; else → hold</td>
            <td>Prevents oscillation</td></tr>
        </table>
        """)

        # Policy descriptions
        html_parts.append(f"""
        <h2 style="color: {accent};">4. Policy Descriptions</h2>
        <table border="1" cellpadding="6" cellspacing="0" style="border-color: #2a3a5c; color: {fg};">
        <tr style="background-color: #16213e;">
            <th>Policy</th><th>Switching Signal</th><th>Algorithm</th>
        </tr>
        <tr><td>n_only</td><td>None (baseline)</td><td>Always use YOLOv8n</td></tr>
        <tr><td>s_only</td><td>None (baseline)</td><td>Always use YOLOv8s</td></tr>
        <tr><td>entropy_only</td><td>H (entropy)</td><td>Switch to s if H ≥ median(H)</td></tr>
        <tr><td>combined</td><td>C-score</td><td>Switch to s if C ≥ c_mid</td></tr>
        <tr><td>combined_hyst</td><td>C-score</td><td>Hysteresis: s if C &gt; c_high, n if C &lt; c_low</td></tr>
        <tr><td>conf_ema</td><td>Confidence drop</td><td>Dual-EMA on detection confidence; switch on rise</td></tr>
        <tr><td>niqe_switch</td><td>NIQE quality</td><td>Dual-EMA on NIQE; switch when quality degrades</td></tr>
        <tr><td>multi_proxy</td><td>C-score (composite)</td><td>Weighted 7-proxy composite with EMA-relative switching</td></tr>
        </table>
        """)

        # Results summary
        html_parts.append(f"""
        <h2 style="color: {accent};">5. Experiment Results</h2>
        <table border="1" cellpadding="6" cellspacing="0" style="border-color: #2a3a5c; color: {fg};">
        <tr style="background-color: #16213e;">
            <th>Policy</th><th>Mean T_total</th><th>P95</th><th>P99</th>
            <th>Slow %</th><th>Sw/100</th><th>Frames</th>
        </tr>
        """)
        for r in filtered:
            html_parts.append(
                f"<tr><td>{r.policy}</td>"
                f"<td>{r.T_total_ms_mean:.2f} ms</td>"
                f"<td>{r.T_total_ms_p95:.2f} ms</td>"
                f"<td>{r.T_total_ms_p99:.2f} ms</td>"
                f"<td>{r.slow_pct:.1f}%</td>"
                f"<td>{r.sw_per_100:.1f}</td>"
                f"<td>{r.total_frames}</td></tr>"
            )
        html_parts.append("</table>")

        # Key findings
        switching = [r for r in filtered if r.policy not in ("n_only", "s_only")]
        if switching:
            best = min(switching, key=lambda r: r.T_total_ms_mean)
            n_runs = [r for r in filtered if r.policy == "n_only"]
            s_runs = [r for r in filtered if r.policy == "s_only"]
            html_parts.append(f"""
            <h2 style="color: {accent};">6. Key Findings</h2>
            <ul>
            <li><b>Best switching policy</b>: {best.policy}
                (mean {best.T_total_ms_mean:.2f} ms, {best.slow_pct:.1f}% slow frames)</li>
            """)
            if n_runs:
                speedup = n_runs[0].T_total_ms_mean
                html_parts.append(
                    f"<li>YOLOv8n baseline: {speedup:.2f} ms mean</li>"
                )
            if s_runs:
                html_parts.append(
                    f"<li>YOLOv8s baseline: {s_runs[0].T_total_ms_mean:.2f} ms mean</li>"
                )
            html_parts.append("</ul>")

        html_parts.append("</div>")
        self._summary_html.setHtml("".join(html_parts))

    # ==================================================================
    # Export
    # ==================================================================
    def _get_video_stem(self) -> str:
        video = self._video_edit.text().strip()
        if video:
            return os.path.splitext(os.path.basename(video))[0]
        # Try to get from results
        for r in self._results:
            if r.video:
                return os.path.splitext(r.video)[0]
        return "experiment"

    def _export_all_plots(self):
        filtered = self._get_filtered_results()
        if not filtered:
            QMessageBox.information(self, "No Data", "Load or run data first.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder for thesis plot PNGs"
        )
        if not folder:
            return

        try:
            from gui.thesis_plots import export_all_thesis_plots
            video_stem = self._get_video_stem()
            export_all_thesis_plots(
                filtered, self._val_summary, folder, video_stem,
            )
            self._status_label.setText(f"Exported all thesis plots to {folder}")
            QMessageBox.information(
                self, "Export Complete",
                f"All thesis plots saved to:\n{folder}",
            )
        except Exception as exc:
            logger.exception("Failed to export thesis plots")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_docx_report(self):
        filtered = self._get_filtered_results()
        if not filtered:
            QMessageBox.information(self, "No Data", "Load or run data first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Research Report",
            f"DMS-Raptor_Thesis_Report_{self._get_video_stem()}.docx",
            "Word Document (*.docx)",
        )
        if not path:
            return
        try:
            from scripts.generate_thesis_report import generate_thesis_report
            plot_dir = os.path.dirname(path)
            generate_thesis_report(
                summaries=filtered,
                output_path=path,
                validation_summary=self._val_summary,
                plot_dir=plot_dir,
                video_name=self._get_video_stem(),
            )
            self._status_label.setText(f"Report saved: {path}")
            QMessageBox.information(
                self, "Report Generated",
                f"Research report saved to:\n{path}",
            )
        except Exception as exc:
            logger.exception("Failed to generate DOCX report")
            QMessageBox.critical(self, "Report Error", str(exc))
