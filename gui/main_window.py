"""DMS-Raptor main window — wires Setup, Monitor, and Results tabs."""
from __future__ import annotations

import logging
import os

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QMessageBox, QApplication, QAction,
)
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from PyQt5.QtCore import Qt

from core.config import APP_NAME, APP_VERSION, APP_SUBTITLE
from core.history import save_run, load_all_runs, clear_history
from gui.setup_tab import SetupTab
from gui.monitor_tab import MonitorTab
from gui.results_tab import ResultsTab
from gui.batch_tab import BatchTab
from gui.extraction_tab import ExtractionTab
from gui.validation_tab import ValidationTab
from gui.thesis_tab import ThesisExperimentsTab
from gui.analysis_tab import AnalysisTab
from gui.worker import InferenceWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} \u2014 {APP_SUBTITLE}")
        self.resize(1440, 920)
        self._load_icon()

        # App root (for history storage)
        self._app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # --- tabs ---
        self.tabs = QTabWidget()
        self.setup_tab = SetupTab()
        self.monitor_tab = MonitorTab()
        self.results_tab = ResultsTab()
        self.batch_tab = BatchTab()
        self.extraction_tab = ExtractionTab()
        self.validation_tab = ValidationTab()
        self.thesis_tab = ThesisExperimentsTab()
        self.analysis_tab = AnalysisTab()

        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.monitor_tab, "Monitor")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.batch_tab, "Batch Run")
        self.tabs.addTab(self.extraction_tab, "Frame Extraction")
        self.tabs.addTab(self.validation_tab, "Validation")
        self.tabs.addTab(self.thesis_tab, "Thesis Experiments")
        self.tabs.addTab(self.analysis_tab, "Thesis Analysis")

        self.setCentralWidget(self.tabs)

        # --- status / menu ---
        self.statusBar().showMessage("Ready")
        self._setup_menu()

        # --- worker handle ---
        self._worker: InferenceWorker | None = None

        # --- signals ---
        self.setup_tab.run_requested.connect(self._on_run)
        self.setup_tab.stop_requested.connect(self._on_stop)

        # Bridge validation tab results → thesis experiments tab
        self.validation_tab.validation_complete.connect(
            self.thesis_tab._on_validation_done
        )

        # --- load persistent history ---
        self._load_history()

    # ------------------------------------------------------------------
    # Icon
    # ------------------------------------------------------------------
    def _load_icon(self):
        logo = os.path.join(os.path.dirname(__file__), "..", "resources", "logo.svg")
        if not os.path.exists(logo):
            return
        try:
            from PyQt5.QtSvg import QSvgRenderer
            renderer = QSvgRenderer(logo)
            pm = QPixmap(64, 64)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            renderer.render(p)
            p.end()
            self.setWindowIcon(QIcon(pm))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _setup_menu(self):
        bar = self.menuBar()
        file_m = bar.addMenu("File")
        exit_a = QAction("Exit", self)
        exit_a.triggered.connect(self.close)
        file_m.addAction(exit_a)

        view_m = bar.addMenu("View")
        theme_m = view_m.addMenu("Theme")
        for theme_name in ["Dark", "Light", "System"]:
            action = QAction(theme_name, self)
            action.triggered.connect(
                lambda checked, t=theme_name.lower(): self._set_theme(t)
            )
            theme_m.addAction(action)

        history_m = bar.addMenu("History")
        clear_a = QAction("Clear All History Files", self)
        clear_a.triggered.connect(self._on_clear_history)
        history_m.addAction(clear_a)

        help_m = bar.addMenu("Help")
        about_a = QAction("About DMS-Raptor", self)
        about_a.triggered.connect(self._about)
        help_m.addAction(about_a)

    def _about(self):
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h2>{APP_NAME} v{APP_VERSION}</h2>"
            f"<p>{APP_SUBTITLE}</p>"
            f"<p>Complexity-aware dynamic model switching for "
            f"real-time object detection on UAV platforms.</p>"
            f"<p>&copy; 2025-2026 Teja Chirravuri. All rights reserved.</p>",
        )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def _set_theme(self, theme: str):
        """Switch the application between dark, light, or system theme."""
        from gui.plot_utils import set_theme

        if theme == "system":
            palette = QApplication.palette()
            is_dark = palette.color(palette.Window).lightness() < 128
            theme = "dark" if is_dark else "light"

        set_theme(theme)

        # Load the appropriate QSS file
        qss_name = "style.qss" if theme == "dark" else "style_light.qss"
        qss_path = os.path.join(
            os.path.dirname(__file__), "..", "resources", qss_name
        )
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                QApplication.instance().setStyleSheet(f.read())

        # Refresh matplotlib plots so they match the new theme
        self._refresh_all_plots()

    def _refresh_all_plots(self):
        """Redraw all matplotlib canvases after a theme change."""
        try:
            self.results_tab._refresh_quick_plots()
        except Exception:
            logger.debug("Could not refresh results plots", exc_info=True)
        try:
            self.thesis_tab.refresh_plots()
        except Exception:
            logger.debug("Could not refresh thesis plots", exc_info=True)

    # ------------------------------------------------------------------
    # Persistent History
    # ------------------------------------------------------------------
    def _load_history(self):
        """Load all previously saved run results from disk."""
        try:
            runs = load_all_runs(self._app_root)
            if runs:
                self.results_tab.load_history(runs)
                self.statusBar().showMessage(
                    f"Loaded {len(runs)} historical run(s)."
                )
                logger.info("Loaded %d historical runs", len(runs))
        except Exception:
            logger.exception("Failed to load history")

    def _on_clear_history(self):
        reply = QMessageBox.question(
            self, "Clear History",
            "Delete ALL saved run history files?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            n = clear_history(self._app_root)
            self.results_tab.clear_results()
            self.statusBar().showMessage(f"Cleared {n} history file(s).")

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------
    def _on_run(self, params: dict):
        if self._worker is not None and self._worker.isRunning():
            return

        self.monitor_tab.reset()
        self.tabs.setCurrentWidget(self.monitor_tab)
        self.statusBar().showMessage("Running pipeline...")
        self.setup_tab.set_running(True)

        self._worker = InferenceWorker(
            source_path=params["source_path"],
            source_type=params["source_type"],
            policies=params["policies"],
            base_config=params["config"],
            inf_params=params["inf_params"],
            model_n_path=params["model_n_path"],
            model_s_path=params["model_s_path"],
            trt_export=params.get("trt_export", False),
            trt_precision=params.get("trt_precision", "fp16"),
            save_video=params.get("save_video", False),
            output_video_dir=params.get("output_video_dir", ""),
        )
        self._worker.frame_ready.connect(self.monitor_tab.update_frame)
        self._worker.policy_finished.connect(self._on_policy_done)
        self._worker.all_finished.connect(self._on_all_done)
        self._worker.stopped_early.connect(self._on_stopped_early)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.start()

    def _on_policy_done(self, summary):
        # Save to persistent history
        try:
            save_run(self._app_root, summary)
        except Exception:
            logger.exception("Failed to save run to history")

        self.results_tab.add_run_result(summary)
        self.thesis_tab.add_run_result(summary)
        self.monitor_tab.reset()
        self.statusBar().showMessage(
            f"Completed: {summary.policy} \u2014 Mean T_total: "
            f"{summary.T_total_ms_mean:.1f} ms"
        )

    def _on_all_done(self):
        self.setup_tab.set_running(False)
        self.tabs.setCurrentWidget(self.results_tab)
        self.statusBar().showMessage("Pipeline complete.")
        self._cleanup_worker()

    def _on_stopped_early(self):
        """Handle graceful stop — app stays open, user returns to Setup."""
        self.setup_tab.set_running(False)
        self.statusBar().showMessage("Pipeline stopped by user.")
        self._cleanup_worker()
        # Stay on current tab so user can see partial results or go back to setup

    def _on_error(self, msg: str):
        self.setup_tab.set_running(False)
        self.statusBar().showMessage(f"Error: {msg}")
        QMessageBox.critical(self, "Pipeline Error", msg)
        self._cleanup_worker()

    def _on_progress(self, current: int, total: int):
        if total > 0:
            pct = int(100 * current / total)
            self.statusBar().showMessage(f"Frame {current}/{total} ({pct}%)")
        else:
            self.statusBar().showMessage(f"Frame {current} (streaming)")

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self.statusBar().showMessage("Stopping... please wait.")

    def _cleanup_worker(self):
        """Safely clean up the worker thread reference."""
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.wait(3000)
            # Disconnect all signals to prevent stale connections
            try:
                self._worker.frame_ready.disconnect()
                self._worker.policy_finished.disconnect()
                self._worker.all_finished.disconnect()
                self._worker.stopped_early.disconnect()
                self._worker.error_occurred.disconnect()
                self._worker.progress_updated.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._worker = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(5000)
        event.accept()
