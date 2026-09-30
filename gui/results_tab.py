"""DMS-Raptor results dashboard — quick plots, data explorer, export, history."""
from __future__ import annotations

import csv
import os
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QPushButton, QHeaderView,
    QAbstractItemView, QFileDialog, QSizePolicy, QLabel,
    QComboBox, QGroupBox, QCheckBox, QGridLayout,
    QScrollArea, QListWidget, QListWidgetItem,
    QMessageBox, QSlider, QSpinBox,
)

from core.config import RunSummary, TRACE_METRICS, AGGREGATE_METRICS
from gui.plot_utils import POLICY_COLORS, TIMING_COLORS, color_for, apply_dark


def _get_trace_data(summary: RunSummary, attr_name: str) -> Optional[list]:
    """Retrieve a trace list from a RunSummary, converting choice_trace to numeric."""
    data = getattr(summary, attr_name, None)
    if data is None:
        return None
    if attr_name == "choice_trace":
        return [0 if c == "n" else 1 for c in data]
    return list(data)


class ResultsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: List[RunSummary] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # == Left: table + buttons ==
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels(
            ["Policy", "Mode", "Mean T (ms)", "P95 T (ms)",
             "Slow %", "Sw/100", "Frames", "Source", "Video"]
        )
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        hdr = self._table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        lv.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_delete_sel = QPushButton("Delete Selected")
        self._btn_clear = QPushButton("Clear All")
        self._btn_csv = QPushButton("Export CSV")
        self._btn_plots = QPushButton("Export Plots")
        self._btn_open_video = QPushButton("Open Video Folder")
        for b in (self._btn_delete_sel, self._btn_clear, self._btn_csv,
                  self._btn_plots, self._btn_open_video):
            btn_row.addWidget(b)
        lv.addLayout(btn_row)

        self._btn_delete_sel.clicked.connect(self._on_delete_selected)
        self._btn_clear.clicked.connect(self._on_clear_clicked)
        self._btn_csv.clicked.connect(self._export_csv)
        self._btn_plots.clicked.connect(self._export_plots)
        self._btn_open_video.clicked.connect(self._open_video_folder)

        splitter.addWidget(left)

        # == Right: tabbed views ==
        self._view_tabs = QTabWidget()

        # --- Tab 1: Quick Plots ---
        self._quick_plots_tab = QTabWidget()
        self._build_quick_plots()
        self._view_tabs.addTab(self._quick_plots_tab, "Quick Plots")

        # --- Tab 2: Data Explorer ---
        self._explorer_widget = self._build_data_explorer()
        self._view_tabs.addTab(self._explorer_widget, "Data Explorer")

        # --- Tab 3: Summary ---
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setStyleSheet(
            "background-color:#0a0e1a; color:#c0c0c0; "
            "font-family:Consolas,monospace; font-size:13px;"
        )
        self._view_tabs.addTab(self._summary_text, "Summary")

        splitter.addWidget(self._view_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

    # ------------------------------------------------------------------
    # Quick Plots
    # ------------------------------------------------------------------
    def _build_quick_plots(self):
        # timing breakdown
        self._fig_tb = Figure(figsize=(6, 4))
        self._ax_tb = self._fig_tb.add_subplot(111)
        self._canvas_tb = FigureCanvasQTAgg(self._fig_tb)
        apply_dark(self._fig_tb, [self._ax_tb])
        self._quick_plots_tab.addTab(self._canvas_tb, "Timing Breakdown")

        # latency comparison
        self._fig_lc = Figure(figsize=(6, 4))
        self._ax_lc = self._fig_lc.add_subplot(111)
        self._canvas_lc = FigureCanvasQTAgg(self._fig_lc)
        apply_dark(self._fig_lc, [self._ax_lc])
        self._quick_plots_tab.addTab(self._canvas_lc, "Latency Comparison")

        # ── Latency Subplots (per-policy T_total with n/s coloring) + zoom ──
        wv_container = QWidget()
        wv_layout = QVBoxLayout(wv_container)
        wv_layout.setContentsMargins(0, 0, 0, 0)

        self._fig_wv = Figure(figsize=(6, 8))
        self._axes_wv = []  # dynamically created in _plot_waveform_subplots
        self._canvas_wv = FigureCanvasQTAgg(self._fig_wv)
        apply_dark(self._fig_wv, [])
        wv_layout.addWidget(self._canvas_wv, stretch=1)

        # Connect mouse events for scroll-wheel zoom and drag-to-pan
        self._canvas_wv.mpl_connect("scroll_event", self._on_scroll_zoom)
        self._canvas_wv.mpl_connect("button_press_event", self._on_pan_press)
        self._canvas_wv.mpl_connect("button_release_event", self._on_pan_release)
        self._canvas_wv.mpl_connect("motion_notify_event", self._on_pan_move)
        self._pan_active = False
        self._pan_start_x = None

        # Zoom controls — clean bar: [-] [===slider===] [+] [Fit All]
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)

        self._btn_zoom_out = QPushButton("−")
        self._btn_zoom_out.setFixedWidth(32)
        self._btn_zoom_out.setToolTip("Zoom out (show more frames)")
        zoom_row.addWidget(self._btn_zoom_out)

        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(0, 1000)
        self._zoom_slider.setValue(1000)  # 1000 = fully zoomed out (all frames)
        self._zoom_slider.setToolTip(
            "Drag to zoom in/out.\n"
            "You can also scroll the mouse wheel on the plot."
        )
        zoom_row.addWidget(self._zoom_slider, stretch=1)

        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setFixedWidth(32)
        self._btn_zoom_in.setToolTip("Zoom in (show fewer frames)")
        zoom_row.addWidget(self._btn_zoom_in)

        self._zoom_range_label = QLabel("All frames")
        self._zoom_range_label.setMinimumWidth(120)
        self._zoom_range_label.setAlignment(Qt.AlignCenter)
        zoom_row.addWidget(self._zoom_range_label)

        self._btn_zoom_reset = QPushButton("Fit All")
        self._btn_zoom_reset.setToolTip("Reset to show all frames")
        zoom_row.addWidget(self._btn_zoom_reset)

        wv_layout.addLayout(zoom_row)

        # Pan slider — visible only when zoomed in
        pan_row = QHBoxLayout()
        pan_row.setSpacing(6)
        pan_row.addWidget(QLabel("Pan:"))
        self._pan_slider = QSlider(Qt.Horizontal)
        self._pan_slider.setRange(0, 1000)
        self._pan_slider.setValue(0)
        self._pan_slider.setToolTip(
            "Drag to scroll through frames.\n"
            "You can also click-drag on the plot to pan."
        )
        pan_row.addWidget(self._pan_slider, stretch=1)
        self._pan_row_widget = QWidget()
        self._pan_row_widget.setLayout(pan_row)
        self._pan_row_widget.setVisible(False)
        wv_layout.addWidget(self._pan_row_widget)

        # Internal zoom state
        self._zoom_level = 1.0   # 1.0 = all frames visible
        self._zoom_center = 0.5  # 0-1, center of view as fraction of total

        # Connect signals
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        self._pan_slider.valueChanged.connect(self._on_pan_slider)
        self._btn_zoom_in.clicked.connect(lambda: self._zoom_step(0.7))
        self._btn_zoom_out.clicked.connect(lambda: self._zoom_step(1.4))
        self._btn_zoom_reset.clicked.connect(self._on_zoom_reset)

        self._quick_plots_tab.addTab(wv_container, "Latency Subplots")

        # C-score traces
        self._fig_ct = Figure(figsize=(6, 4))
        self._ax_ct = self._fig_ct.add_subplot(111)
        self._canvas_ct = FigureCanvasQTAgg(self._fig_ct)
        apply_dark(self._fig_ct, [self._ax_ct])
        self._quick_plots_tab.addTab(self._canvas_ct, "Switching Signals")

    # ------------------------------------------------------------------
    # Zoom / Pan handlers
    # ------------------------------------------------------------------
    def _get_max_frame(self) -> int:
        """Return the max frame index across all loaded results."""
        mx = 0
        for r in self._results:
            if r.frame_indices:
                mx = max(mx, max(r.frame_indices))
        return max(mx, 1)

    def _apply_zoom(self):
        """Apply current _zoom_level and _zoom_center to all waveform axes."""
        if not self._axes_wv:
            return
        max_f = self._get_max_frame()
        if self._zoom_level >= 0.99:
            # Fully zoomed out — show everything
            for ax in self._axes_wv:
                ax.set_xlim(auto=True)
            self._zoom_range_label.setText("All frames")
            self._pan_row_widget.setVisible(False)
        else:
            window = max(10, int(max_f * self._zoom_level))
            half = window / 2.0
            center = self._zoom_center * max_f
            start = max(0, center - half)
            end = start + window
            if end > max_f:
                end = max_f
                start = max(0, end - window)
            for ax in self._axes_wv:
                ax.set_xlim(start, end)
            self._zoom_range_label.setText(
                f"{int(start)}–{int(end)}  ({window} frames)"
            )
            self._pan_row_widget.setVisible(True)
        self._canvas_wv.draw_idle()

    def _on_zoom_slider(self, val):
        """Zoom slider: 1000 = all frames, 0 = max zoom in."""
        self._zoom_level = max(0.01, val / 1000.0)
        self._apply_zoom()
        # Sync pan slider position
        self._pan_slider.blockSignals(True)
        self._pan_slider.setValue(int(self._zoom_center * 1000))
        self._pan_slider.blockSignals(False)

    def _on_pan_slider(self, val):
        """Pan slider: 0 = start, 1000 = end."""
        self._zoom_center = val / 1000.0
        self._apply_zoom()

    def _zoom_step(self, factor: float):
        """Zoom in (factor < 1) or out (factor > 1)."""
        self._zoom_level = min(1.0, max(0.01, self._zoom_level * factor))
        # Update the zoom slider to match
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(int(self._zoom_level * 1000))
        self._zoom_slider.blockSignals(False)
        self._apply_zoom()

    def _on_zoom_reset(self):
        self._zoom_level = 1.0
        self._zoom_center = 0.5
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(1000)
        self._zoom_slider.blockSignals(False)
        self._pan_slider.blockSignals(True)
        self._pan_slider.setValue(500)
        self._pan_slider.blockSignals(False)
        self._apply_zoom()

    # --- Mouse scroll-wheel zoom on canvas ---
    def _on_scroll_zoom(self, event):
        """Scroll wheel on the canvas: zoom in/out centered on cursor."""
        if not self._axes_wv or event.inaxes is None:
            return
        max_f = self._get_max_frame()
        if max_f <= 1:
            return
        # Zoom factor per scroll tick
        factor = 0.8 if event.button == "up" else 1.25
        self._zoom_level = min(1.0, max(0.01, self._zoom_level * factor))
        # Re-center on cursor position
        if event.xdata is not None:
            self._zoom_center = max(0.0, min(1.0, event.xdata / max_f))
        # Update slider to match
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(int(self._zoom_level * 1000))
        self._zoom_slider.blockSignals(False)
        self._pan_slider.blockSignals(True)
        self._pan_slider.setValue(int(self._zoom_center * 1000))
        self._pan_slider.blockSignals(False)
        self._apply_zoom()

    # --- Click-drag to pan on canvas ---
    def _on_pan_press(self, event):
        if event.button == 1 and event.inaxes in self._axes_wv:
            self._pan_active = True
            self._pan_start_x = event.xdata

    def _on_pan_release(self, event):
        self._pan_active = False
        self._pan_start_x = None

    def _on_pan_move(self, event):
        if not self._pan_active or event.xdata is None or self._pan_start_x is None:
            return
        if not self._axes_wv or self._zoom_level >= 0.99:
            return
        max_f = self._get_max_frame()
        if max_f <= 1:
            return
        dx = (self._pan_start_x - event.xdata) / max_f
        self._zoom_center = max(0.0, min(1.0, self._zoom_center + dx))
        self._pan_slider.blockSignals(True)
        self._pan_slider.setValue(int(self._zoom_center * 1000))
        self._pan_slider.blockSignals(False)
        self._apply_zoom()
        self._pan_start_x = event.xdata

    # ------------------------------------------------------------------
    # Data Explorer
    # ------------------------------------------------------------------
    def _build_data_explorer(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        # -- Left control panel --
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(300)
        ctrl_scroll.setMinimumWidth(220)

        ctrl = QWidget()
        cvbox = QVBoxLayout(ctrl)
        cvbox.setSpacing(8)

        # Plot mode
        cvbox.addWidget(QLabel("<b>Plot Mode</b>"))
        self._explorer_mode = QComboBox()
        self._explorer_mode.addItems([
            "Trace (per-frame)",
            "Aggregate Comparison",
            "Distribution",
            "Correlation Heatmap",
        ])
        self._explorer_mode.currentIndexChanged.connect(
            self._on_explorer_mode_changed
        )
        cvbox.addWidget(self._explorer_mode)

        # X-axis
        self._x_label = QLabel("<b>X-Axis</b>")
        cvbox.addWidget(self._x_label)
        self._x_combo = QComboBox()
        self._x_combo.addItems(list(TRACE_METRICS.keys()))
        cvbox.addWidget(self._x_combo)

        # Y-axis
        self._y_label = QLabel("<b>Y-Axis</b>")
        cvbox.addWidget(self._y_label)
        self._y_combo = QComboBox()
        self._y_combo.addItems(list(TRACE_METRICS.keys()))
        self._y_combo.setCurrentIndex(1)  # T_total by default
        cvbox.addWidget(self._y_combo)

        # Aggregate metric (for comparison mode)
        self._agg_label = QLabel("<b>Metric</b>")
        cvbox.addWidget(self._agg_label)
        self._agg_combo = QComboBox()
        self._agg_combo.addItems(list(AGGREGATE_METRICS.keys()))
        cvbox.addWidget(self._agg_combo)

        # Chart type
        cvbox.addWidget(QLabel("<b>Chart Type</b>"))
        self._chart_type_combo = QComboBox()
        cvbox.addWidget(self._chart_type_combo)

        # Run selection
        cvbox.addWidget(QLabel("<b>Select Runs</b>"))
        self._run_list = QListWidget()
        self._run_list.setSelectionMode(QListWidget.MultiSelection)
        self._run_list.setMaximumHeight(180)
        cvbox.addWidget(self._run_list)

        # Buttons
        self._btn_generate = QPushButton("Generate Plot")
        self._btn_generate.setStyleSheet(
            "background-color: #00bcd4; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        self._btn_generate.clicked.connect(self._generate_explorer_plot)
        cvbox.addWidget(self._btn_generate)

        self._btn_export_explorer = QPushButton("Export This Plot")
        self._btn_export_explorer.clicked.connect(self._export_explorer_plot)
        cvbox.addWidget(self._btn_export_explorer)

        cvbox.addStretch()
        ctrl_scroll.setWidget(ctrl)
        layout.addWidget(ctrl_scroll)

        # -- Right: canvas --
        self._explorer_fig = Figure(figsize=(8, 5))
        self._explorer_ax = self._explorer_fig.add_subplot(111)
        apply_dark(self._explorer_fig, [self._explorer_ax])
        self._explorer_canvas = FigureCanvasQTAgg(self._explorer_fig)
        self._explorer_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self._explorer_canvas)

        layout.setStretchFactor(ctrl_scroll, 0)
        layout.setStretchFactor(self._explorer_canvas, 1)

        # Initial mode setup
        self._on_explorer_mode_changed(0)

        return widget

    def _on_explorer_mode_changed(self, idx: int):
        mode = self._explorer_mode.currentText()
        is_trace = "Trace" in mode
        is_agg = "Aggregate" in mode
        is_dist = "Distribution" in mode

        self._x_label.setVisible(is_trace)
        self._x_combo.setVisible(is_trace)
        self._y_label.setVisible(is_trace or is_dist)
        self._y_combo.setVisible(is_trace or is_dist)
        self._agg_label.setVisible(is_agg)
        self._agg_combo.setVisible(is_agg)

        self._chart_type_combo.clear()
        if is_trace:
            self._chart_type_combo.addItems(["Line", "Scatter", "Area"])
        elif is_agg:
            self._chart_type_combo.addItems(
                ["Bar", "Grouped Bar (Mean/P95/P99)"]
            )
        elif is_dist:
            self._chart_type_combo.addItems(
                ["Histogram", "Box Plot", "Violin Plot"]
            )
        else:
            self._chart_type_combo.addItems(["Heatmap"])

    def _get_selected_runs(self) -> List[RunSummary]:
        selected = []
        for i in range(self._run_list.count()):
            item = self._run_list.item(i)
            if item.isSelected():
                idx = item.data(Qt.UserRole)
                if idx is not None and 0 <= idx < len(self._results):
                    selected.append(self._results[idx])
        if not selected:
            return list(self._results)
        return selected

    def _generate_explorer_plot(self):
        if not self._results:
            return
        ax = self._explorer_ax
        ax.clear()
        apply_dark(self._explorer_fig, [ax])

        runs = self._get_selected_runs()
        mode = self._explorer_mode.currentText()
        chart = self._chart_type_combo.currentText()

        if "Trace" in mode:
            self._plot_trace(ax, runs, chart)
        elif "Aggregate" in mode:
            self._plot_aggregate(ax, runs, chart)
        elif "Distribution" in mode:
            self._plot_distribution(ax, runs, chart)
        elif "Correlation" in mode:
            self._plot_correlation(ax, runs)

        self._explorer_fig.tight_layout()
        self._explorer_canvas.draw_idle()

    # ---------- Explorer plot implementations ----------

    def _plot_trace(self, ax, runs: List[RunSummary], chart: str):
        x_name = list(TRACE_METRICS.keys())[self._x_combo.currentIndex()]
        y_name = list(TRACE_METRICS.keys())[self._y_combo.currentIndex()]
        x_attr = TRACE_METRICS[x_name]
        y_attr = TRACE_METRICS[y_name]

        for i, r in enumerate(runs):
            x_data = _get_trace_data(r, x_attr)
            y_data = _get_trace_data(r, y_attr)
            if x_data is None or y_data is None:
                continue
            n = min(len(x_data), len(y_data))
            if n == 0:
                continue
            x_data, y_data = x_data[:n], y_data[:n]
            c = color_for(r.policy, i)
            hist = getattr(r, "is_history", False)
            label = (
                f"{r.policy} ({r.run_timestamp[:19]})"
                if r.run_timestamp
                else r.policy
            )
            alpha = 0.35 if hist else 0.85
            ls = "--" if hist else "-"

            if chart == "Line":
                ax.plot(x_data, y_data, color=c, linewidth=1.3,
                        label=label, alpha=alpha, linestyle=ls)
            elif chart == "Scatter":
                ax.scatter(x_data, y_data, color=c, s=8,
                           label=label, alpha=alpha * 0.7)
            elif chart == "Area":
                ax.fill_between(x_data, y_data, color=c,
                                alpha=0.2 if hist else 0.3, label=label)
                ax.plot(x_data, y_data, color=c, linewidth=1.0,
                        alpha=alpha * 0.8, linestyle=ls)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{y_name} vs {x_name}")
        if len(runs) <= 10:
            ax.legend(loc="best", fontsize=7, facecolor="#16213e",
                      edgecolor="#2a3a5c", labelcolor="#c0c0c0")

    def _plot_aggregate(self, ax, runs: List[RunSummary], chart: str):
        if "Grouped" in chart:
            labels = [r.policy for r in runs]
            means = [r.T_total_ms_mean for r in runs]
            p95s = [r.T_total_ms_p95 for r in runs]
            p99s = [r.T_total_ms_p99 for r in runs]

            x = np.arange(len(labels))
            w = 0.25
            ax.bar(x - w, means, width=w, color="#00bcd4", label="Mean")
            ax.bar(x, p95s, width=w, color="#ff9800", label="P95")
            ax.bar(x + w, p99s, width=w, color="#e94560", label="P99")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("T_total (ms)")
            ax.set_title("Latency Comparison (Mean / P95 / P99)")
            ax.legend(loc="best", fontsize=8, facecolor="#16213e",
                      edgecolor="#2a3a5c", labelcolor="#c0c0c0")
        else:
            metric_name = self._agg_combo.currentText()
            attr = AGGREGATE_METRICS.get(metric_name, "T_total_ms_mean")
            labels = [r.policy for r in runs]
            values = [getattr(r, attr, 0) for r in runs]
            colors = [color_for(r.policy, i) for i, r in enumerate(runs)]

            x = np.arange(len(labels))
            ax.bar(x, values, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} by Policy")
            for xi, v in zip(x, values):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                        color="#c0c0c0", fontsize=8)

    def _plot_distribution(self, ax, runs: List[RunSummary], chart: str):
        y_name = list(TRACE_METRICS.keys())[self._y_combo.currentIndex()]
        y_attr = TRACE_METRICS[y_name]

        datasets, labels, colors = [], [], []
        for i, r in enumerate(runs):
            data = _get_trace_data(r, y_attr)
            if data and len(data) > 0:
                datasets.append(data)
                labels.append(r.policy)
                colors.append(color_for(r.policy, i))

        if not datasets:
            ax.set_title("No data to display")
            return

        if chart == "Histogram":
            for data, label, c in zip(datasets, labels, colors):
                ax.hist(data, bins=50, color=c, alpha=0.5, label=label)
            ax.set_xlabel(y_name)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {y_name}")
            ax.legend(loc="best", fontsize=8, facecolor="#16213e",
                      edgecolor="#2a3a5c", labelcolor="#c0c0c0")
        elif chart == "Box Plot":
            bp = ax.boxplot(datasets, patch_artist=True, labels=labels)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            for el in ["whiskers", "caps", "medians"]:
                for item in bp[el]:
                    item.set_color("#c0c0c0")
            ax.set_ylabel(y_name)
            ax.set_title(f"Box Plot: {y_name}")
            ax.tick_params(axis="x", rotation=20)
        elif chart == "Violin Plot":
            parts = ax.violinplot(datasets, showmeans=True, showmedians=True)
            for j, pc in enumerate(parts.get("bodies", [])):
                pc.set_facecolor(colors[j % len(colors)])
                pc.set_alpha(0.5)
            for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
                if key in parts:
                    parts[key].set_color("#c0c0c0")
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(y_name)
            ax.set_title(f"Violin Plot: {y_name}")

    def _plot_correlation(self, ax, runs: List[RunSummary]):
        if not runs:
            ax.set_title("No data")
            return
        r = runs[0]
        metric_names, metric_data = [], []
        for name, attr in TRACE_METRICS.items():
            if attr == "choice_trace":
                data = [0 if c == "n" else 1 for c in (r.choice_trace or [])]
            else:
                data = getattr(r, attr, None)
            if data and len(data) > 1:
                metric_names.append(name)
                metric_data.append(np.array(data, dtype=float))

        if len(metric_names) < 2:
            ax.set_title("Not enough data for correlation")
            return
        min_len = min(len(d) for d in metric_data)
        matrix = np.column_stack([d[:min_len] for d in metric_data])
        corr = np.corrcoef(matrix.T)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metric_names)))
        ax.set_yticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(metric_names, fontsize=7)
        ax.set_title(f"Correlation Heatmap: {r.policy}")
        self._explorer_fig.colorbar(im, ax=ax, shrink=0.8)

    def _export_explorer_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Explorer Plot", "explorer_plot.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if path:
            self._explorer_fig.savefig(
                path, dpi=220,
                facecolor=self._explorer_fig.get_facecolor(),
                bbox_inches="tight",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_run_result(self, summary: RunSummary):
        summary.is_history = False
        self._results.append(summary)
        self._add_table_row(summary)
        self._refresh_run_list()
        self._refresh_quick_plots()
        self._update_summary_text()

    def load_history(self, summaries: List[RunSummary]):
        for s in summaries:
            s.is_history = True
            self._results.append(s)
            self._add_table_row(s)
        self._refresh_run_list()
        self._refresh_quick_plots()
        self._update_summary_text()

    def clear_results(self):
        self._results.clear()
        self._table.setRowCount(0)
        self._fig_wv.clf()
        self._axes_wv = []
        self._zoom_level = 1.0
        self._zoom_center = 0.5
        self._zoom_slider.setValue(1000)
        self._pan_slider.setValue(500)
        self._pan_row_widget.setVisible(False)
        self._zoom_range_label.setText("All frames")
        apply_dark(self._fig_wv, [])
        for ax in (self._ax_tb, self._ax_lc, self._ax_ct):
            ax.clear()
            apply_dark(ax.get_figure(), [ax])
        for c in (self._canvas_tb, self._canvas_lc, self._canvas_wv,
                  self._canvas_ct):
            c.draw_idle()
        self._summary_text.clear()
        self._run_list.clear()
        self._explorer_ax.clear()
        apply_dark(self._explorer_fig, [self._explorer_ax])
        self._explorer_canvas.draw_idle()

    def _on_delete_selected(self):
        selected_rows = sorted(
            set(idx.row() for idx in self._table.selectedIndexes()),
            reverse=True,
        )
        if not selected_rows:
            QMessageBox.information(
                self, "No Selection",
                "Select rows in the table to delete.",
            )
            return

        reply = QMessageBox.question(
            self, "Delete Selected",
            f"Delete {len(selected_rows)} selected result(s)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for row in selected_rows:
            if 0 <= row < len(self._results):
                self._results.pop(row)
            self._table.removeRow(row)

        self._refresh_run_list()
        self._refresh_quick_plots()
        self._update_summary_text()
        self._explorer_ax.clear()
        apply_dark(self._explorer_fig, [self._explorer_ax])
        self._explorer_canvas.draw_idle()

    def _on_clear_clicked(self):
        if not self._results:
            return
        reply = QMessageBox.question(
            self, "Clear Results",
            "Clear all run results from this session?\n"
            "(Historical runs are still saved on disk.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.clear_results()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    def _add_table_row(self, s: RunSummary):
        r = self._table.rowCount()
        self._table.insertRow(r)

        video_info = ""
        if s.output_video_path and os.path.isfile(s.output_video_path):
            video_info = f"{s.output_video_size_mb:.1f} MB"
        elif s.output_video_path:
            video_info = "pending"

        source = "History" if getattr(s, "is_history", False) else "Session"

        vals = [
            s.policy,
            s.mode,
            f"{s.T_total_ms_mean:.1f}",
            f"{s.T_total_ms_p95:.1f}",
            f"{s.slow_pct:.1f}",
            f"{s.sw_per_100:.1f}",
            str(s.total_frames),
            source,
            video_info,
        ]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(r, c, item)

    def _refresh_run_list(self):
        self._run_list.clear()
        for idx, r in enumerate(self._results):
            ts = r.run_timestamp[:19] if r.run_timestamp else f"run-{idx}"
            src = "H" if getattr(r, "is_history", False) else "S"
            text = f"[{src}] {r.policy} | {ts} | {r.total_frames}f"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            self._run_list.addItem(item)
            item.setSelected(True)

    # ------------------------------------------------------------------
    # Quick Plots
    # ------------------------------------------------------------------
    def _refresh_quick_plots(self):
        if not self._results:
            return
        self._plot_timing_breakdown()
        self._plot_latency_comparison()
        self._plot_waveform_subplots()
        self._plot_c_traces()

    def _plot_timing_breakdown(self):
        ax = self._ax_tb
        ax.clear()
        apply_dark(self._fig_tb, [ax])

        labels = [r.policy for r in self._results]
        scene = np.array([r.T_scene_ms_mean for r in self._results])
        ctrl = np.array([r.T_ctrl_ms_mean for r in self._results])
        inf_n = np.array([r.T_infer_n_ms_mean for r in self._results])
        inf_s = np.array([r.T_infer_s_ms_mean for r in self._results])

        x = np.arange(len(labels))
        b = np.zeros_like(scene)
        ax.bar(x, scene, bottom=b, color=TIMING_COLORS["T_scene"], label="T_scene")
        b += scene
        ax.bar(x, ctrl, bottom=b, color=TIMING_COLORS["T_ctrl"], label="T_ctrl")
        b += ctrl
        ax.bar(x, inf_n, bottom=b, color=TIMING_COLORS["T_infer_n"], label="T_infer_n")
        b += inf_n
        ax.bar(x, inf_s, bottom=b, color=TIMING_COLORS["T_infer_s"], label="T_infer_s")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Mean latency (ms)")
        ax.set_title("Timing Breakdown per Policy")
        ax.legend(loc="upper left", fontsize=8, facecolor="#16213e",
                  edgecolor="#2a3a5c", labelcolor="#c0c0c0")
        self._fig_tb.tight_layout()
        self._canvas_tb.draw_idle()

    def _plot_latency_comparison(self):
        ax = self._ax_lc
        ax.clear()
        apply_dark(self._fig_lc, [ax])

        labels = [r.policy for r in self._results]
        means = [r.T_total_ms_mean for r in self._results]
        p95s = [r.T_total_ms_p95 for r in self._results]
        p99s = [r.T_total_ms_p99 for r in self._results]

        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, means, width=w, color="#00bcd4", label="Mean")
        ax.bar(x, p95s, width=w, color="#ff9800", label="P95")
        ax.bar(x + w, p99s, width=w, color="#e94560", label="P99")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("T_total (ms)")
        ax.set_title("Latency Comparison")
        ax.legend(loc="upper left", fontsize=8, facecolor="#16213e",
                  edgecolor="#2a3a5c", labelcolor="#c0c0c0")
        self._fig_lc.tight_layout()
        self._canvas_lc.draw_idle()

    def _plot_waveform_subplots(self):
        """Per-policy subplots showing T_total coloured by model choice.

        Each policy gets its own subplot in a 2-column grid.  Green
        (fill_between) marks frames where the n-model (YOLOv8n) was used;
        red marks s-model (YOLOv8s) frames.
        """
        # Clear previous dynamic axes
        self._fig_wv.clf()
        self._axes_wv = []

        n_results = len(self._results)
        if n_results == 0:
            apply_dark(self._fig_wv, [])
            self._canvas_wv.draw_idle()
            return

        # Grid layout: 2 columns, as many rows as needed
        ncols = 2 if n_results > 1 else 1
        nrows = (n_results + ncols - 1) // ncols

        axes = self._fig_wv.subplots(nrows, ncols, squeeze=False)
        all_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

        # The axes we actually use (one per result)
        self._axes_wv = all_axes[:n_results]

        # Hide any leftover axes in the last row
        for ax in all_axes[n_results:]:
            ax.set_visible(False)

        apply_dark(self._fig_wv, self._axes_wv)

        color_n = "#4caf50"   # green — YOLOv8n
        color_s = "#e94560"   # red   — YOLOv8s

        for idx, r in enumerate(self._results):
            ax = self._axes_wv[idx]

            if not r.frame_indices or not r.T_total_trace:
                ax.set_title(r.policy, fontsize=9, color="#c0c0c0")
                continue

            frames = np.array(r.frame_indices)
            t_total = np.array(r.T_total_trace)
            choices = r.choice_trace  # list of "n" or "s" (or None)

            if choices and len(choices) >= len(frames):
                n_mask = np.array([c == "n" for c in choices[:len(frames)]])
                ax.fill_between(
                    frames, 0, t_total,
                    where=n_mask, color=color_n, alpha=0.7,
                    label="YOLOv8n", interpolate=True,
                )
                ax.fill_between(
                    frames, 0, t_total,
                    where=~n_mask, color=color_s, alpha=0.7,
                    label="YOLOv8s", interpolate=True,
                )
                # Thin outline for readability
                ax.plot(frames, t_total, color="#c0c0c0",
                        linewidth=0.3, alpha=0.4)
            else:
                # No choice trace (e.g. n_only / s_only) — single colour
                c = color_for(r.policy, idx)
                ax.fill_between(
                    frames, 0, t_total,
                    color=c, alpha=0.5, label="T_total",
                )
                ax.plot(frames, t_total, color=c, linewidth=0.6, alpha=0.8)

            ts_label = ""
            if r.run_timestamp:
                ts_label = f"  ({r.run_timestamp[:19]})"
            hist = getattr(r, "is_history", False)
            src_tag = " [H]" if hist else ""
            ax.set_title(
                f"{r.policy}{src_tag}{ts_label}",
                fontsize=9, color="#c0c0c0",
            )
            ax.set_ylabel("T_total (ms)", fontsize=7, color="#c0c0c0")
            ax.set_xlabel("Frame", fontsize=7, color="#c0c0c0")
            ax.legend(
                loc="upper right", fontsize=7,
                facecolor="#16213e", edgecolor="#2a3a5c",
                labelcolor="#c0c0c0",
            )

        self._fig_wv.tight_layout()
        self._canvas_wv.draw_idle()

    # Policies that use the C-score (scene-analysis based)
    _C_SCORE_POLICIES = {
        "entropy_only", "combined", "combined_hyst",
        "local_contrast_hyst", "multi_proxy",
    }

    def _plot_c_traces(self):
        """Plot switching signals: C-score for scene policies, conf-drop
        for conf_ema, NIQE score for niqe_switch.  n_only / s_only are
        skipped since they never switch.
        """
        from gui.plot_utils import theme_colors
        tc = theme_colors()

        ax = self._ax_ct
        ax.clear()
        apply_dark(self._fig_ct, [ax])

        has_data = False
        signal_types = set()  # track which signal categories are plotted

        for i, r in enumerate(self._results):
            # Skip baselines — they don't switch
            if r.policy in ("n_only", "s_only"):
                continue

            c = color_for(r.policy, i)
            hist = getattr(r, "is_history", False)
            alpha = 0.3 if hist else 0.85
            ls = "--" if hist else "-"
            frames = r.frame_indices

            if r.policy in self._C_SCORE_POLICIES:
                # Scene-analysis policies → plot C-score
                if not r.C_trace or not any(v != 0 for v in r.C_trace):
                    continue
                ax.plot(frames, r.C_trace, color=c, linewidth=1.3,
                        label=f"{r.policy} — C-score", alpha=alpha,
                        linestyle=ls)
                has_data = True
                signal_types.add("C-score")
                # Plot this run's OWN thresholds
                if r.c_low_trace:
                    ax.plot(frames, r.c_low_trace, color=c,
                            linewidth=0.8, linestyle=":", alpha=0.4,
                            label=f"{r.policy} c_low/c_high")
                if r.c_high_trace:
                    ax.plot(frames, r.c_high_trace, color=c,
                            linewidth=0.8, linestyle=":", alpha=0.4)

            elif r.policy == "conf_ema":
                # Confidence-EMA → plot conf_drop (EMA rise signal)
                trace = getattr(r, "conf_drop_trace", None)
                if not trace or not any(v != 0 for v in trace):
                    continue
                ax.plot(frames[:len(trace)], trace, color=c,
                        linewidth=1.3,
                        label=f"conf_ema — conf drop",
                        alpha=alpha, linestyle=ls)
                has_data = True
                signal_types.add("Conf-EMA")
                # Plot conf_ema thresholds from c_low/c_high traces
                if r.c_low_trace:
                    ax.plot(frames[:len(r.c_low_trace)],
                            r.c_low_trace, color=c,
                            linewidth=0.8, linestyle=":", alpha=0.4,
                            label="conf_ema c_low/c_high")
                if r.c_high_trace:
                    ax.plot(frames[:len(r.c_high_trace)],
                            r.c_high_trace, color=c,
                            linewidth=0.8, linestyle=":", alpha=0.4)

            elif r.policy == "niqe_switch":
                # NIQE-Switch → plot NIQE score trace
                trace = getattr(r, "niqe_trace", None)
                if not trace or not any(v != 0 for v in trace):
                    continue
                ax.plot(frames[:len(trace)], trace, color=c,
                        linewidth=1.3,
                        label=f"niqe_switch — NIQE score",
                        alpha=alpha, linestyle=ls)
                has_data = True
                signal_types.add("NIQE")

        ax.set_xlabel("Frame", color=tc["fg"])
        if has_data:
            # Build a descriptive y-label from the signal types present
            ylabel = " / ".join(sorted(signal_types))
            ax.set_ylabel(ylabel, color=tc["fg"])
            title_parts = sorted(signal_types)
            ax.set_title(
                f"Switching Signals: {', '.join(title_parts)}",
                color=tc["fg"],
            )
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    loc="upper right", fontsize=7,
                    facecolor=tc["legend_bg"],
                    edgecolor=tc["legend_edge"],
                    labelcolor=tc["fg"],
                )
        else:
            ax.set_ylabel("Signal value", color=tc["fg"])
            ax.set_title(
                "No switching signals (run a switching policy first)",
                color=tc["fg"],
            )

        self._fig_ct.tight_layout()
        self._canvas_ct.draw_idle()

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------
    def _update_summary_text(self):
        lines = ["=" * 60, "  DMS-Raptor Run Summary", "=" * 60, ""]
        for r in self._results:
            ts = r.run_timestamp[:19] if r.run_timestamp else "N/A"
            src = "History" if getattr(r, "is_history", False) else "Session"
            lines.append(
                f"[{src}] Policy: {r.policy}  |  Mode: {r.mode}  "
                f"|  Frames: {r.total_frames}  |  Time: {ts}"
            )
            lines.append(f"  Mean T_total : {r.T_total_ms_mean:8.2f} ms")
            lines.append(f"  P95  T_total : {r.T_total_ms_p95:8.2f} ms")
            lines.append(f"  P99  T_total : {r.T_total_ms_p99:8.2f} ms")
            lines.append(f"  Slow %       : {r.slow_pct:8.2f} %")
            lines.append(
                f"  Switches     : {r.switches}  "
                f"({r.sw_per_100:.1f} per 100 frames)"
            )
            lines.append(f"  T_scene mean : {r.T_scene_ms_mean:8.3f} ms")
            lines.append(f"  T_ctrl  mean : {r.T_ctrl_ms_mean:8.3f} ms")
            lines.append(f"  T_inf_n mean : {r.T_infer_n_ms_mean:8.2f} ms")
            lines.append(f"  T_inf_s mean : {r.T_infer_s_ms_mean:8.2f} ms")
            if r.output_video_path:
                lines.append(
                    f"  Output video : {r.output_video_path}  "
                    f"({r.output_video_size_mb:.1f} MB)"
                )
            lines.append("-" * 60)

        switching = [
            r for r in self._results
            if r.policy not in ("n_only", "s_only")
        ]
        if switching:
            best = min(switching, key=lambda r: r.T_total_ms_mean)
            lines.append("")
            lines.append(
                f">> Best switching policy: {best.policy} "
                f"(mean {best.T_total_ms_mean:.2f} ms)"
            )

        self._summary_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "dms_raptor_results.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Policy", "Mode", "Frames", "Timestamp", "Source",
                "Mean_T_total_ms", "P95_T_total_ms", "P99_T_total_ms",
                "Slow_pct", "Sw_per_100", "Switches",
                "T_scene_mean", "T_ctrl_mean", "T_inf_n_mean", "T_inf_s_mean",
                "Output_Video", "Video_Size_MB",
            ])
            for r in self._results:
                src = (
                    "History"
                    if getattr(r, "is_history", False)
                    else "Session"
                )
                w.writerow([
                    r.policy, r.mode, r.total_frames, r.run_timestamp, src,
                    f"{r.T_total_ms_mean:.3f}", f"{r.T_total_ms_p95:.3f}",
                    f"{r.T_total_ms_p99:.3f}",
                    f"{r.slow_pct:.2f}", f"{r.sw_per_100:.2f}", r.switches,
                    f"{r.T_scene_ms_mean:.4f}", f"{r.T_ctrl_ms_mean:.4f}",
                    f"{r.T_infer_n_ms_mean:.3f}",
                    f"{r.T_infer_s_ms_mean:.3f}",
                    r.output_video_path, f"{r.output_video_size_mb:.2f}",
                ])

    def _export_plots(self):
        if not self._results:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder for plot PNGs"
        )
        if not folder:
            return
        plots = [
            (self._fig_tb, "timing_breakdown.png"),
            (self._fig_lc, "latency_comparison.png"),
            (self._fig_wv, "latency_subplots.png"),
            (self._fig_ct, "switching_signals.png"),
        ]
        for fig, name in plots:
            fig.savefig(
                os.path.join(folder, name), dpi=220,
                facecolor=fig.get_facecolor(), bbox_inches="tight",
            )

    def _open_video_folder(self):
        for r in self._results:
            if r.output_video_path and os.path.isfile(r.output_video_path):
                folder = os.path.dirname(r.output_video_path)
                import subprocess
                import sys
                if sys.platform == "win32":
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
                return
        QMessageBox.information(
            self, "No Videos",
            "No annotated output videos found.\n"
            "Enable 'Save annotated output video' in Setup before running.",
        )
