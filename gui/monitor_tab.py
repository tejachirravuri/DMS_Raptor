"""Live monitoring tab for DMS-Raptor — video canvas, stats panel, waveform charts."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.config import FrameResult

_ROLLING_WINDOW = 500
_REDRAW_EVERY = 3

# Dark-theme palette
_BG = "#0a0e1a"
_GRID = "#1a2040"
_SPINE = "#2a3a5c"
_TICK = "#8899aa"
_TITLE = "#c0c0c0"

# Chart colours
_CYAN = "#00bcd4"
_ORANGE = "#ff9800"
_RED = "#e94560"
_GREEN = "#4caf50"


class MonitorTab(QWidget):
    """Real-time monitoring panel: video feed, live stats, rolling waveforms."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Trace buffers (rolling window)
        self._frames: deque[int] = deque(maxlen=_ROLLING_WINDOW)
        self._t_total: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._c_score: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._c_low: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._c_high: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._choices: deque[str] = deque(maxlen=_ROLLING_WINDOW)

        self._redraw_counter: int = 0
        self._policy_name: str = ""
        self._total_frames: int = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter_v = QSplitter(Qt.Vertical)
        root.addWidget(splitter_v)

        # --- Top section: video + stats ---
        top_splitter = QSplitter(Qt.Horizontal)
        splitter_v.addWidget(top_splitter)

        # Video canvas
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setStyleSheet("background-color: black;")
        self._video_label.setMinimumSize(320, 240)
        top_splitter.addWidget(self._video_label)

        # Stats panel
        stats_container = QWidget()
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.setSpacing(6)

        self._stat_labels: dict[str, QLabel] = {}
        stat_keys = [
            "Frame", "FPS", "Model", "Detections",
            "T_total", "T_scene", "Dwell", "C-score",
            "Mean Conf", "Conf Drop", "NIQE", "Policy",
        ]
        for key in stat_keys:
            card = QWidget()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(6, 4, 6, 4)
            card_layout.setSpacing(1)

            lbl_key = QLabel(key)
            lbl_key.setObjectName("statsKey")
            lbl_key.setStyleSheet(
                "font-size: 11px; color: #8899aa; font-weight: bold;"
            )

            lbl_val = QLabel("—")
            lbl_val.setObjectName("statsValue")
            lbl_val.setStyleSheet("font-size: 15px; color: #e0e0e0;")

            card_layout.addWidget(lbl_key)
            card_layout.addWidget(lbl_val)
            stats_layout.addWidget(card)
            self._stat_labels[key] = lbl_val

        stats_layout.addStretch()
        top_splitter.addWidget(stats_container)

        # Splitter proportions: 75% video / 25% stats
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)

        # --- Bottom section: matplotlib charts ---
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self._setup_charts()
        chart_layout.addWidget(self._canvas)
        splitter_v.addWidget(chart_widget)

        # Vertical splitter proportions: 60% top / 40% bottom
        splitter_v.setStretchFactor(0, 3)
        splitter_v.setStretchFactor(1, 2)

    def _setup_charts(self) -> None:
        """Create the matplotlib Figure with 3 vertically-stacked subplots."""
        self._fig = Figure(figsize=(8, 4), dpi=80)
        self._fig.patch.set_facecolor(_BG)

        self._axes = self._fig.subplots(3, 1, sharex=True)
        self._fig.subplots_adjust(hspace=0.35, left=0.08, right=0.96, top=0.95, bottom=0.10)

        # Initial cosmetic pass
        titles = ["T_total (ms)", "C-score", "Model Choice"]
        ylabels = ["ms", "C", ""]
        for ax, title, ylabel in zip(self._axes, titles, ylabels):
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            self._apply_dark_theme(ax)

        self._axes[2].set_xlabel("Frame", fontsize=9)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @staticmethod
    def _apply_dark_theme(ax) -> None:
        """Apply the dark colour scheme to a single Axes."""
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_TICK)
        ax.xaxis.label.set_color(_TICK)
        ax.yaxis.label.set_color(_TICK)
        ax.title.set_color(_TITLE)
        for spine in ax.spines.values():
            spine.set_color(_SPINE)
        ax.grid(True, alpha=0.2, color=_GRID)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_frame(self, fr: FrameResult) -> None:
        """Main slot — call once per frame with the latest FrameResult."""
        # 1. Video canvas (fill entire area, center-crop to remove black bars)
        if fr.annotated_frame is not None:
            pm = self._frame_to_pixmap(fr.annotated_frame)
            label_size = self._video_label.size()
            scaled = pm.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._video_label.setPixmap(scaled)

        # 2. Stat labels
        self._stat_labels["Frame"].setText(str(fr.frame_idx))

        if fr.T_total_ms > 0:
            fps = 1000.0 / fr.T_total_ms
            self._stat_labels["FPS"].setText(f"{fps:.1f}")
        else:
            self._stat_labels["FPS"].setText("—")

        if fr.choice == "n":
            self._stat_labels["Model"].setText(
                '<span style="color:#4caf50; font-weight:bold;">YOLOv8n</span>'
            )
            self._stat_labels["Model"].setTextFormat(Qt.RichText)
        else:
            self._stat_labels["Model"].setText(
                '<span style="color:#e94560; font-weight:bold;">YOLOv8s</span>'
            )
            self._stat_labels["Model"].setTextFormat(Qt.RichText)

        self._stat_labels["Detections"].setText(str(fr.num_detections))
        self._stat_labels["T_total"].setText(f"{fr.T_total_ms:.1f} ms")
        self._stat_labels["T_scene"].setText(f"{fr.T_scene_ms:.1f} ms")
        self._stat_labels["Dwell"].setText(str(fr.dwell))
        self._stat_labels["C-score"].setText(f"{fr.C:.3f}")
        self._stat_labels["Mean Conf"].setText(f"{fr.mean_conf:.3f}")
        self._stat_labels["Conf Drop"].setText(f"{fr.conf_drop:.4f}")
        self._stat_labels["NIQE"].setText(f"{fr.niqe_score:.2f}")
        self._stat_labels["Policy"].setText(self._policy_name or "—")

        # 3. Append trace data
        self._frames.append(fr.frame_idx)
        self._t_total.append(fr.T_total_ms)
        self._c_score.append(fr.C)
        self._c_low.append(fr.c_low)
        self._c_high.append(fr.c_high)
        self._choices.append(fr.choice)

        # 4. Throttled chart redraw
        self._redraw_counter += 1
        if self._redraw_counter % _REDRAW_EVERY == 0:
            self._redraw_charts()

    def set_policy_info(self, policy: str, total_frames: int) -> None:
        """Update the displayed policy name and total frame count."""
        self._policy_name = policy
        self._total_frames = total_frames
        if "Policy" in self._stat_labels:
            self._stat_labels["Policy"].setText(policy or "—")

    def reset(self) -> None:
        """Clear all buffers, blank the video, and reset charts."""
        self._frames.clear()
        self._t_total.clear()
        self._c_score.clear()
        self._c_low.clear()
        self._c_high.clear()
        self._choices.clear()
        self._redraw_counter = 0

        self._video_label.clear()
        self._video_label.setStyleSheet("background-color: black;")

        for lbl in self._stat_labels.values():
            lbl.setText("—")

        for ax in self._axes:
            ax.clear()
        titles = ["T_total (ms)", "C-score", "Model Choice"]
        ylabels = ["ms", "C", ""]
        for ax, title, ylabel in zip(self._axes, titles, ylabels):
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            self._apply_dark_theme(ax)
        self._axes[2].set_xlabel("Frame", fontsize=9)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _redraw_charts(self) -> None:
        """Clear and redraw all three subplots from current deque data."""
        if not self._frames:
            return

        xs = list(self._frames)

        # --- Chart 1: T_total ---
        ax0 = self._axes[0]
        ax0.clear()
        ax0.plot(xs, list(self._t_total), color=_CYAN, linewidth=1.5)
        ax0.set_title("T_total (ms)", fontsize=10)
        ax0.set_ylabel("ms", fontsize=9)
        self._apply_dark_theme(ax0)

        # --- Chart 2: C-score ---
        ax1 = self._axes[1]
        ax1.clear()
        ax1.plot(xs, list(self._c_score), color=_ORANGE, linewidth=1.5)

        # Threshold lines
        c_low_vals = list(self._c_low)
        c_high_vals = list(self._c_high)
        if c_low_vals:
            ax1.plot(xs, c_low_vals, color=_RED, linewidth=1.0, linestyle="--")
        if c_high_vals:
            ax1.plot(xs, c_high_vals, color=_RED, linewidth=1.0, linestyle="--")

        ax1.set_title("C-score", fontsize=10)
        ax1.set_ylabel("C", fontsize=9)
        self._apply_dark_theme(ax1)

        # --- Chart 3: Model Choice ---
        ax2 = self._axes[2]
        ax2.clear()

        choices = list(self._choices)
        numeric = [0 if c == "n" else 1 for c in choices]

        if len(xs) > 0:
            # Build masks for fill_between
            xs_arr = np.array(xs, dtype=float)
            num_arr = np.array(numeric, dtype=float)

            # Green fill where model is "n" (value 0 → fill from 0 to 0 won't show,
            # so we fill the complement: green when n, red when s)
            ax2.fill_between(
                xs_arr, 0, 1,
                where=(num_arr == 0),
                color=_GREEN, alpha=0.7, step="post",
            )
            ax2.fill_between(
                xs_arr, 0, 1,
                where=(num_arr == 1),
                color=_RED, alpha=0.7, step="post",
            )

            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["YOLOv8n", "YOLOv8s"])
            ax2.set_ylim(-0.1, 1.1)

        ax2.set_title("Model Choice", fontsize=10)
        ax2.set_xlabel("Frame", fontsize=9)
        self._apply_dark_theme(ax2)

        self._canvas.draw_idle()

    @staticmethod
    def _frame_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
        """Convert a BGR numpy array to a QPixmap."""
        h, w = frame_bgr.shape[:2]
        ch = frame_bgr.shape[2] if frame_bgr.ndim == 3 else 1

        if ch == 3:
            # BGR → RGB
            rgb = frame_bgr[:, :, ::-1].copy()
            fmt = QImage.Format_RGB888
            bytes_per_line = 3 * w
        elif ch == 4:
            rgb = frame_bgr[:, :, [2, 1, 0, 3]].copy()
            fmt = QImage.Format_RGBA8888
            bytes_per_line = 4 * w
        else:
            # Grayscale
            rgb = frame_bgr.copy()
            fmt = QImage.Format_Grayscale8
            bytes_per_line = w

        qimg = QImage(rgb.data, w, h, bytes_per_line, fmt)
        # QImage doesn't own the buffer — force a deep copy via QPixmap
        return QPixmap.fromImage(qimg.copy())
