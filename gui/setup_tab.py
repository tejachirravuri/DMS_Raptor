"""Setup / configuration tab for DMS-Raptor GUI."""
from __future__ import annotations

import logging
import os
from dataclasses import fields as dc_fields
from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import (
    POLICIES,
    INPUT_MODES,
    RunConfig,
    InferenceParams,
)

logger = logging.getLogger(__name__)

# Default dataclass instances used for "Auto" fallback values.
_DEFAULT_CFG = RunConfig()
_DEFAULT_INF = InferenceParams()


class SetupTab(QWidget):
    """Left-hand panel that collects all run parameters and emits them."""

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    run_requested = pyqtSignal(dict)
    stop_requested = pyqtSignal()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Bookkeeping for param widgets:  field_name -> (widget, auto_cb)
        self._param_widgets: Dict[str, Tuple[QWidget, QCheckBox]] = {}

        self._build_ui()
        self._connect_signals()
        # Trigger initial visibility state
        self._on_mode_changed(0)

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        scroll.setWidget(container)

        self._build_input_source_group()
        self._build_models_group()
        self._build_policies_group()
        self._build_output_group()
        self._build_controller_params_group()
        self._build_actions_row()

        self._layout.addStretch()

    # ------------------------------------------------------------------
    # 1. Input Source
    # ------------------------------------------------------------------
    def _build_input_source_group(self) -> None:
        grp = QGroupBox("Input Source")
        grid = QGridLayout(grp)

        # Mode selector
        grid.addWidget(QLabel("Mode:"), 0, 0)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(INPUT_MODES)
        grid.addWidget(self._mode_combo, 0, 1, 1, 2)

        # Source path (for Video File / Image Folder)
        grid.addWidget(QLabel("Path:"), 1, 0)
        self._source_path_edit = QLineEdit()
        self._source_path_edit.setPlaceholderText("Select a file or folder...")
        grid.addWidget(self._source_path_edit, 1, 1)
        self._browse_source_btn = QPushButton("Browse")
        grid.addWidget(self._browse_source_btn, 1, 2)

        # Live stream: camera index
        self._cam_label = QLabel("Camera index:")
        grid.addWidget(self._cam_label, 2, 0)
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 20)
        grid.addWidget(self._cam_spin, 2, 1, 1, 2)

        # Live stream: RTSP URL
        self._rtsp_label = QLabel("RTSP URL:")
        grid.addWidget(self._rtsp_label, 3, 0)
        self._rtsp_edit = QLineEdit()
        self._rtsp_edit.setPlaceholderText("rtsp://...")
        grid.addWidget(self._rtsp_edit, 3, 1, 1, 2)

        # Video info label
        self._video_info_label = QLabel("")
        self._video_info_label.setWordWrap(True)
        grid.addWidget(self._video_info_label, 4, 0, 1, 3)

        self._layout.addWidget(grp)

    # ------------------------------------------------------------------
    # 2. Detection Models
    # ------------------------------------------------------------------
    def _build_models_group(self) -> None:
        grp = QGroupBox("Detection Models")
        grid = QGridLayout(grp)

        # Fast model (n)
        grid.addWidget(QLabel("Fast model (n):"), 0, 0)
        self._model_n_edit = QLineEdit()
        self._model_n_edit.setPlaceholderText("path/to/yolo-n.pt")
        grid.addWidget(self._model_n_edit, 0, 1)
        btn_n = QPushButton("Browse")
        btn_n.clicked.connect(lambda: self._browse_file(self._model_n_edit, "YOLO weights (*.pt)"))
        grid.addWidget(btn_n, 0, 2)

        # Accurate model (s)
        grid.addWidget(QLabel("Accurate model (s):"), 1, 0)
        self._model_s_edit = QLineEdit()
        self._model_s_edit.setPlaceholderText("path/to/yolo-s.pt")
        grid.addWidget(self._model_s_edit, 1, 1)
        btn_s = QPushButton("Browse")
        btn_s.clicked.connect(lambda: self._browse_file(self._model_s_edit, "YOLO weights (*.pt)"))
        grid.addWidget(btn_s, 1, 2)

        # Device
        grid.addWidget(QLabel("Device:"), 2, 0)
        self._device_combo = QComboBox()
        devices = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    devices.append(f"cuda:{i}")
                    # detect Jetson (Tegra GPU)
                    if "tegra" in name.lower() or "orin" in name.lower() or "xavier" in name.lower():
                        devices.append(f"jetson:{i}")
        except Exception:
            pass
        # always offer Jetson option for cross-device config
        if not any(d.startswith("jetson") for d in devices):
            devices.append("jetson (remote)")
        self._device_combo.addItems(devices)
        grid.addWidget(self._device_combo, 2, 1, 1, 2)

        # Image size
        grid.addWidget(QLabel("Image size (imgsz):"), 3, 0)
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(320, 1280)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        grid.addWidget(self._imgsz_spin, 3, 1, 1, 2)

        # TensorRT optimization
        grid.addWidget(QLabel("TensorRT:"), 4, 0)
        self._trt_check = QCheckBox("Export to TensorRT (.engine) before run")
        self._trt_check.setToolTip(
            "When enabled and a Jetson/CUDA device is selected, models are\n"
            "exported to TensorRT FP16 .engine format for maximum throughput.\n"
            "The export runs once; subsequent runs reuse the .engine file."
        )
        grid.addWidget(self._trt_check, 4, 1, 1, 2)

        self._trt_precision = QComboBox()
        self._trt_precision.addItems(["fp16", "fp32", "int8"])
        self._trt_precision.setToolTip("TensorRT precision mode (fp16 recommended for Jetson)")
        grid.addWidget(QLabel("TRT Precision:"), 5, 0)
        grid.addWidget(self._trt_precision, 5, 1, 1, 2)

        self._layout.addWidget(grp)

    # ------------------------------------------------------------------
    # 3. Policies
    # ------------------------------------------------------------------
    def _build_policies_group(self) -> None:
        grp = QGroupBox("Policies")
        vbox = QVBoxLayout(grp)

        self._policy_checks: Dict[str, QCheckBox] = {}
        _POLICY_LABELS = {
            "n_only": "n_only  (always fast model)",
            "s_only": "s_only  (always accurate model)",
            "entropy_only": "entropy_only  (switch on entropy median)",
            "combined": "combined  (percentile-normalized C)",
            "combined_hyst": "combined_hyst  (percentile + hysteresis)",
            "conf_ema": "conf_ema  (confidence-EMA, model-intrinsic)",
            "niqe_switch": "niqe_switch  (NR-IQA, input-intrinsic)",
            "multi_proxy": "multi_proxy  (7-proxy weighted composite)",
        }
        for pol in POLICIES:
            label = _POLICY_LABELS.get(pol, pol)
            cb = QCheckBox(label)
            cb.setChecked(True)
            self._policy_checks[pol] = cb
            vbox.addWidget(cb)

        self._layout.addWidget(grp)

    # ------------------------------------------------------------------
    # 3.5  Output / Download Options
    # ------------------------------------------------------------------
    def _build_output_group(self) -> None:
        grp = QGroupBox("Output Options")
        grid = QGridLayout(grp)

        # Save annotated video checkbox
        self._save_video_check = QCheckBox("Save annotated output video")
        self._save_video_check.setToolTip(
            "Write the annotated (overlay) video to disk.\n"
            "One MP4 file per policy is created in the selected output folder."
        )
        grid.addWidget(self._save_video_check, 0, 0, 1, 3)

        # Output directory
        grid.addWidget(QLabel("Output folder:"), 1, 0)
        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setPlaceholderText("Select output folder for videos...")
        self._output_dir_edit.setEnabled(False)
        grid.addWidget(self._output_dir_edit, 1, 1)

        self._browse_output_btn = QPushButton("Browse")
        self._browse_output_btn.setEnabled(False)
        self._browse_output_btn.clicked.connect(
            lambda: self._browse_folder(self._output_dir_edit)
        )
        grid.addWidget(self._browse_output_btn, 1, 2)

        # Info label about estimated size
        self._output_info_label = QLabel(
            "Videos will be named annotated_<policy>.mp4 in the chosen folder."
        )
        self._output_info_label.setWordWrap(True)
        self._output_info_label.setStyleSheet("color: #8899aa; font-size: 11px;")
        grid.addWidget(self._output_info_label, 2, 0, 1, 3)

        # Toggle output fields
        self._save_video_check.toggled.connect(self._output_dir_edit.setEnabled)
        self._save_video_check.toggled.connect(self._browse_output_btn.setEnabled)

        self._layout.addWidget(grp)

    # ------------------------------------------------------------------
    # 4. Controller Parameters
    # ------------------------------------------------------------------
    def _build_controller_params_group(self) -> None:
        grp = QGroupBox("Controller Parameters")
        grid = QGridLayout(grp)
        row = 0

        def _heading(text: str) -> None:
            nonlocal row
            lbl = QLabel(f"<b>{text}</b>")
            grid.addWidget(lbl, row, 0, 1, 4)
            row += 1

        def _int_row(name: str, label: str, lo: int, hi: int, default: int, step: int = 1) -> None:
            nonlocal row
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(default)
            self._add_param_row(grid, row, name, label, w, default)
            row += 1

        def _float_row(name: str, label: str, lo: float, hi: float, default: float,
                        step: float = 0.01, decimals: int = 3) -> None:
            nonlocal row
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(decimals)
            w.setValue(default)
            self._add_param_row(grid, row, name, label, w, default)
            row += 1

        # -- Scene Analysis -----------------------------------------------
        _heading("Scene Analysis")
        _float_row("alpha", "alpha", 0.0, 1.0, _DEFAULT_CFG.alpha, step=0.05)
        _int_row("proxy_size", "proxy_size", 32, 512, _DEFAULT_CFG.proxy_size)
        _int_row("hist_bins", "hist_bins", 16, 256, _DEFAULT_CFG.hist_bins)
        _int_row("probe_every_k", "probe_every_k", 1, 20, _DEFAULT_CFG.probe_every_k)

        # -- Thresholds ---------------------------------------------------
        _heading("Thresholds (percentile policies)")
        _float_row("c_low", "c_low", 0.0, 1.0, _DEFAULT_CFG.c_low, step=0.01)
        _float_row("c_high", "c_high", 0.0, 1.0, _DEFAULT_CFG.c_high, step=0.01)
        _float_row("combined_mid", "combined_mid", 0.0, 1.0, _DEFAULT_CFG.combined_mid, step=0.01)

        # -- Confidence-EMA (conf_ema policy) --------------------------------
        _heading("Confidence-EMA (conf_ema policy)")
        _float_row("conf_ema_fast_beta", "conf_ema_fast_beta", 0.01, 0.99, _DEFAULT_CFG.conf_ema_fast_beta, step=0.05)
        _float_row("conf_ema_slow_beta", "conf_ema_slow_beta", 0.001, 0.50, _DEFAULT_CFG.conf_ema_slow_beta, step=0.005)
        _float_row("conf_ema_c_high", "conf_ema_c_high (switch to s)", 0.01, 0.50, _DEFAULT_CFG.conf_ema_c_high, step=0.01)
        _float_row("conf_ema_c_low", "conf_ema_c_low (switch to n)", 0.0, 0.50, _DEFAULT_CFG.conf_ema_c_low, step=0.01)

        # -- NIQE-Switch (niqe_switch policy) ---------------------------------
        _heading("NIQE-Switch (niqe_switch policy)")
        _float_row("niqe_fast_beta", "niqe_fast_beta", 0.01, 0.99, _DEFAULT_CFG.niqe_fast_beta, step=0.05)
        _float_row("niqe_slow_beta", "niqe_slow_beta", 0.001, 0.50, _DEFAULT_CFG.niqe_slow_beta, step=0.005)
        _float_row("niqe_c_high", "niqe_c_high (switch to s)", 0.001, 0.50, _DEFAULT_CFG.niqe_c_high, step=0.001)
        _float_row("niqe_c_low", "niqe_c_low (switch to n)", 0.0, 0.50, _DEFAULT_CFG.niqe_c_low, step=0.001)

        # -- Multi-Proxy weighted composite (multi_proxy policy) -------------
        _heading("Multi-Proxy (multi_proxy policy)")
        _float_row("mp_w_laplacian", "w: Laplacian (L)", 0.0, 5.0, _DEFAULT_CFG.mp_w_laplacian, step=0.05)
        _float_row("mp_w_entropy", "w: Entropy (H)", 0.0, 5.0, _DEFAULT_CFG.mp_w_entropy, step=0.05)
        _float_row("mp_w_local_contrast", "w: Local Contrast", 0.0, 5.0, _DEFAULT_CFG.mp_w_local_contrast, step=0.05)
        _float_row("mp_w_color_entropy", "w: Color Entropy", 0.0, 5.0, _DEFAULT_CFG.mp_w_color_entropy, step=0.05)
        _float_row("mp_w_tenengrad", "w: Tenengrad", 0.0, 5.0, _DEFAULT_CFG.mp_w_tenengrad, step=0.05)
        _float_row("mp_w_edge_density", "w: Edge Density", 0.0, 5.0, _DEFAULT_CFG.mp_w_edge_density, step=0.05)
        _float_row("mp_w_brenner", "w: Brenner", 0.0, 5.0, _DEFAULT_CFG.mp_w_brenner, step=0.05)
        _float_row("mp_fast_beta", "mp_fast_beta (EMA fast)", 0.01, 0.99, _DEFAULT_CFG.mp_fast_beta, step=0.05)
        _float_row("mp_slow_beta", "mp_slow_beta (EMA slow)", 0.001, 0.50, _DEFAULT_CFG.mp_slow_beta, step=0.005)
        _float_row("mp_c_high", "mp_c_high (switch to s)", 0.001, 0.50, _DEFAULT_CFG.mp_c_high, step=0.01)
        _float_row("mp_c_low", "mp_c_low (switch to n)", 0.0, 0.50, _DEFAULT_CFG.mp_c_low, step=0.01)

        # -- Stability ----------------------------------------------------
        _heading("Stability")
        _int_row("min_dwell_frames", "min_dwell_frames", 1, 100, _DEFAULT_CFG.min_dwell_frames)
        _int_row("max_switches_per_100", "max_switches_per_100", 1, 50, _DEFAULT_CFG.max_switches_per_100)

        # -- Processing ---------------------------------------------------
        _heading("Processing")
        _float_row("conf_min", "conf_min", 0.0, 1.0, _DEFAULT_INF.conf_min, step=0.001)
        _float_row("iou_nms", "iou_nms", 0.0, 1.0, _DEFAULT_INF.iou_nms, step=0.05)
        _float_row("conf_show", "conf_show", 0.0, 1.0, _DEFAULT_INF.conf_show, step=0.01)
        _int_row("max_frames", "max_frames", 0, 999999, _DEFAULT_INF.max_frames)
        _int_row("stride", "stride", 1, 30, _DEFAULT_INF.stride)
        _int_row("latency_smooth_window", "latency_smooth_window", 1, 100,
                 _DEFAULT_CFG.latency_smooth_window)

        self._layout.addWidget(grp)

    def _add_param_row(
        self,
        grid: QGridLayout,
        row: int,
        name: str,
        label: str,
        widget: QWidget,
        default_value: Any,
    ) -> None:
        """Add a labelled parameter row with an *Auto* checkbox."""
        lbl = QLabel(label)
        auto_cb = QCheckBox("Auto")
        auto_cb.setChecked(True)
        widget.setEnabled(False)

        grid.addWidget(lbl, row, 0)
        grid.addWidget(widget, row, 1, 1, 2)
        grid.addWidget(auto_cb, row, 3)

        # Toggle input enabled state
        auto_cb.toggled.connect(lambda checked, w=widget, d=default_value: self._on_auto_toggled(checked, w, d))

        self._param_widgets[name] = (widget, auto_cb)

    @staticmethod
    def _on_auto_toggled(checked: bool, widget: QWidget, default_value: Any) -> None:
        widget.setEnabled(not checked)
        if checked:
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(default_value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(default_value))

    # ------------------------------------------------------------------
    # 5. Actions
    # ------------------------------------------------------------------
    def _build_actions_row(self) -> None:
        h = QHBoxLayout()

        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setObjectName("runButton")
        h.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setEnabled(False)
        h.addWidget(self._stop_btn)

        self._layout.addLayout(h)

    # ==================================================================
    # Signal wiring
    # ==================================================================
    def _connect_signals(self) -> None:
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._browse_source_btn.clicked.connect(self._on_browse_source)
        self._source_path_edit.textChanged.connect(self._update_video_info)
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

    # ==================================================================
    # Slot implementations
    # ==================================================================
    def _on_mode_changed(self, idx: int) -> None:
        mode = self._mode_combo.currentText()
        is_video = mode == "Video File"
        is_live = mode == "Live Stream"
        is_folder = mode == "Image Folder"

        # Path row visible for Video File / Image Folder
        self._source_path_edit.setVisible(is_video or is_folder)
        self._browse_source_btn.setVisible(is_video or is_folder)

        # Camera / RTSP visible for Live Stream
        self._cam_label.setVisible(is_live)
        self._cam_spin.setVisible(is_live)
        self._rtsp_label.setVisible(is_live)
        self._rtsp_edit.setVisible(is_live)

        # Video info only for video files
        self._video_info_label.setVisible(is_video)

        # Update placeholder
        if is_video:
            self._source_path_edit.setPlaceholderText("Select a video file...")
        elif is_folder:
            self._source_path_edit.setPlaceholderText("Select an image folder...")

    def _on_browse_source(self) -> None:
        mode = self._mode_combo.currentText()
        if mode == "Video File":
            self._browse_file(
                self._source_path_edit,
                "Video files (*.mp4 *.avi *.mov *.mkv *.webm)",
            )
        elif mode == "Image Folder":
            self._browse_folder(self._source_path_edit)

    def _update_video_info(self) -> None:
        mode = self._mode_combo.currentText()
        if mode != "Video File":
            self._video_info_label.setText("")
            return
        path = self._source_path_edit.text().strip()
        if not path or not os.path.isfile(path):
            self._video_info_label.setText("")
            return
        try:
            from core.engine import count_video_frames
            n_frames, fps, w, h = count_video_frames(path)
            duration = n_frames / fps if fps > 0 else 0
            self._video_info_label.setText(
                f"{w}\u00d7{h}  |  {fps:.1f} FPS  |  {n_frames} frames  |  {duration:.1f}s"
            )
        except Exception as exc:
            self._video_info_label.setText(f"Could not read video: {exc}")

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _browse_file(line_edit: QLineEdit, filter_str: str) -> None:
        path, _ = QFileDialog.getOpenFileName(None, "Select File", "", filter_str)
        if path:
            line_edit.setText(path)

    @staticmethod
    def _browse_folder(line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if path:
            line_edit.setText(path)

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------
    def _on_run_clicked(self) -> None:
        # -- Validate inputs -----------------------------------------------
        mode = self._mode_combo.currentText()

        # Source path / type
        if mode == "Video File":
            source_path = self._source_path_edit.text().strip()
            if not source_path or not os.path.isfile(source_path):
                QMessageBox.warning(self, "Validation", "Please select a valid video file.")
                return
            source_type = "video"
        elif mode == "Image Folder":
            source_path = self._source_path_edit.text().strip()
            if not source_path or not os.path.isdir(source_path):
                QMessageBox.warning(self, "Validation", "Please select a valid image folder.")
                return
            source_type = "folder"
        else:  # Live Stream
            rtsp = self._rtsp_edit.text().strip()
            if rtsp:
                source_path = rtsp
            else:
                source_path = str(self._cam_spin.value())
            source_type = "stream"

        # Models
        model_n_path = self._model_n_edit.text().strip()
        model_s_path = self._model_s_edit.text().strip()
        if not model_n_path or not os.path.isfile(model_n_path):
            QMessageBox.warning(self, "Validation", "Fast model (n) path is invalid.")
            return
        if not model_s_path or not os.path.isfile(model_s_path):
            QMessageBox.warning(self, "Validation", "Accurate model (s) path is invalid.")
            return

        # Policies
        policies = [p for p, cb in self._policy_checks.items() if cb.isChecked()]
        if not policies:
            QMessageBox.warning(self, "Validation", "Select at least one policy.")
            return

        # Output options
        save_video = self._save_video_check.isChecked()
        output_video_dir = ""
        if save_video:
            output_video_dir = self._output_dir_edit.text().strip()
            if not output_video_dir:
                QMessageBox.warning(self, "Validation", "Please select an output folder for videos.")
                return
            os.makedirs(output_video_dir, exist_ok=True)

        # -- Build RunConfig -----------------------------------------------
        cfg_kwargs: Dict[str, Any] = {}
        _cfg_field_names = {f.name for f in dc_fields(RunConfig)}
        for name, (widget, auto_cb) in self._param_widgets.items():
            if name in _cfg_field_names:
                if auto_cb.isChecked():
                    cfg_kwargs[name] = getattr(_DEFAULT_CFG, name)
                else:
                    cfg_kwargs[name] = widget.value()

        config = RunConfig(**cfg_kwargs)

        # -- Build InferenceParams -----------------------------------------
        inf_kwargs: Dict[str, Any] = {}
        _inf_field_names = {f.name for f in dc_fields(InferenceParams)}
        for name, (widget, auto_cb) in self._param_widgets.items():
            if name in _inf_field_names:
                if auto_cb.isChecked():
                    inf_kwargs[name] = getattr(_DEFAULT_INF, name)
                else:
                    inf_kwargs[name] = widget.value()

        inf_kwargs["imgsz"] = self._imgsz_spin.value()
        inf_kwargs["device"] = self._device_combo.currentText()
        inf_params = InferenceParams(**inf_kwargs)

        # -- TensorRT settings ---------------------------------------------
        trt_export = self._trt_check.isChecked()
        trt_precision = self._trt_precision.currentText()

        # -- Emit -----------------------------------------------------------
        self.run_requested.emit({
            "source_path": source_path,
            "source_type": source_type,
            "policies": policies,
            "config": config,
            "inf_params": inf_params,
            "model_n_path": model_n_path,
            "model_s_path": model_s_path,
            "trt_export": trt_export,
            "trt_precision": trt_precision,
            "save_video": save_video,
            "output_video_dir": output_video_dir,
        })

    def _on_stop_clicked(self) -> None:
        self.stop_requested.emit()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_running(self, is_running: bool) -> None:
        """Toggle run/stop button states."""
        self._run_btn.setEnabled(not is_running)
        self._stop_btn.setEnabled(is_running)
