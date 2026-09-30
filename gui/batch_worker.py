"""Background worker for batch processing multiple datasets."""
from __future__ import annotations
import dataclasses
import logging
import os
import csv
from typing import Dict, List, Optional
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from core.config import RunConfig, InferenceParams, RunSummary, POLICIES

logger = logging.getLogger(__name__)

class BatchWorker(QThread):
    # Signals
    status_update = pyqtSignal(str)          # status message
    frame_progress = pyqtSignal(int, int)    # current_frame, total_frames
    run_finished = pyqtSignal(str, object)   # subfolder, RunSummary
    dataset_complete = pyqtSignal(str)       # subfolder name
    all_complete = pyqtSignal()
    stopped = pyqtSignal()                   # emitted when user-requested stop completes
    error_occurred = pyqtSignal(str)

    def __init__(self, datasets, policies, base_config, inf_params,
                 output_root, save_videos=True, generate_plots=True,
                 export_csv=True, run_validation=True):
        super().__init__()
        # datasets: list of dicts with keys: subfolder, videos (list of paths), model_n, model_s
        # policies: list of policy strings
        self.datasets = datasets
        self.policies = policies
        self.base_config = base_config
        self.inf_params = inf_params
        self.output_root = output_root
        self.save_videos = save_videos
        self.generate_plots = generate_plots
        self.export_csv_flag = export_csv
        self.run_validation = run_validation
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            if not self._stop_requested:
                logger.exception("BatchWorker failed")
                self.error_occurred.emit(str(exc))
            else:
                self.stopped.emit()
            return
        if self._stop_requested:
            self.stopped.emit()
        else:
            self.all_complete.emit()

    def _run_impl(self):
        from ultralytics import YOLO
        from core.engine import StreamingEngine

        for ds in self.datasets:
            if self._stop_requested:
                break
            subfolder = ds["subfolder"]
            videos = ds["videos"]
            model_n_path = ds["model_n"]
            model_s_path = ds["model_s"]

            self.status_update.emit(f"Loading models for {subfolder}...")
            model_n = YOLO(model_n_path)
            model_s = YOLO(model_s_path)

            # Warmup
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            dev = self.inf_params.device
            for mdl in (model_n, model_s):
                try:
                    mdl.predict(dummy, imgsz=self.inf_params.imgsz, device=dev, verbose=False)
                except Exception:
                    pass

            out_dir = os.path.join(self.output_root, subfolder)
            os.makedirs(out_dir, exist_ok=True)

            all_summaries = []

            for video_path in videos:
                if self._stop_requested:
                    break
                video_name = os.path.splitext(os.path.basename(video_path))[0]

                for policy in self.policies:
                    if self._stop_requested:
                        break

                    self.status_update.emit(f"{subfolder} / {video_name} — {policy}")

                    cfg = dataclasses.replace(
                        self.base_config,
                        policy=policy,
                        name=f"{subfolder}_{video_name}_{policy}",
                        mode="fixed",
                    )
                    if self.save_videos:
                        vid_out = os.path.join(out_dir, f"annotated_{video_name}_{policy}.mp4")
                        cfg = dataclasses.replace(cfg, save_annotated_video=True, output_video_path=vid_out)

                    engine = StreamingEngine(video_path, cfg, model_n, model_s, self.inf_params, "video")
                    total = engine.total_frames
                    gen = engine.run()
                    try:
                        for fr in gen:
                            if self._stop_requested:
                                break
                            self.frame_progress.emit(fr.frame_idx, total)
                    finally:
                        gen.close()

                    try:
                        summary = engine.get_summary()
                        if summary is not None:
                            all_summaries.append(summary)
                            self.run_finished.emit(subfolder, summary)
                    except Exception as exc:
                        logger.warning("Summary failed for %s/%s: %s", subfolder, policy, exc)

            # After all policies for this dataset
            if all_summaries and self.export_csv_flag:
                self._write_csv(out_dir, subfolder, all_summaries)

            if all_summaries and self.generate_plots:
                self._write_plots(out_dir, subfolder, all_summaries)

            # Proxy validation
            if self.run_validation and not self._stop_requested:
                for video_path in videos:
                    if self._stop_requested:
                        break
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    self.status_update.emit(f"{subfolder} / {video_name} — Proxy Validation")
                    try:
                        from validation.validator import ProxyValidator
                        from validation.report import ValidationReport
                        pv = ProxyValidator(video_path, model_n, model_s, self.inf_params)
                        summary_val = None
                        vgen = pv.run()
                        try:
                            for vfr in vgen:
                                if self._stop_requested:
                                    break
                                summary_val = pv.summary
                        finally:
                            vgen.close()
                        if summary_val and not self._stop_requested:
                            report = ValidationReport(summary_val)
                            report_path = os.path.join(out_dir, f"validation_{video_name}.txt")
                            with open(report_path, "w", encoding="utf-8") as f:
                                f.write(report.generate_text_report())
                            # Correlation matrix plot
                            try:
                                fig = report.plot_correlation_matrix()
                                if fig:
                                    fig.savefig(os.path.join(out_dir, f"correlation_{video_name}.png"),
                                                dpi=150, bbox_inches="tight")
                            except Exception:
                                pass
                    except Exception as exc:
                        logger.warning("Validation failed for %s/%s: %s", subfolder, video_name, exc)

            self.dataset_complete.emit(subfolder)

    def _write_csv(self, out_dir, subfolder, summaries):
        path = os.path.join(out_dir, "summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Dataset", "Policy", "Mode", "Frames",
                         "Mean_T_total_ms", "P95_T_total_ms", "P99_T_total_ms",
                         "Slow_pct", "Sw_per_100", "Switches",
                         "T_scene_mean", "T_ctrl_mean", "T_inf_n_mean", "T_inf_s_mean"])
            for r in summaries:
                w.writerow([subfolder, r.policy, r.mode, r.total_frames,
                            f"{r.T_total_ms_mean:.3f}", f"{r.T_total_ms_p95:.3f}",
                            f"{r.T_total_ms_p99:.3f}", f"{r.slow_pct:.2f}",
                            f"{r.sw_per_100:.2f}", r.switches,
                            f"{r.T_scene_ms_mean:.4f}", f"{r.T_ctrl_ms_mean:.4f}",
                            f"{r.T_infer_n_ms_mean:.3f}", f"{r.T_infer_s_ms_mean:.3f}"])

    def _write_plots(self, out_dir, subfolder, summaries):
        from matplotlib.figure import Figure
        from gui.plot_utils import apply_dark, color_for

        # Comparison bar chart
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        apply_dark(fig, [ax])
        labels = [r.policy for r in summaries]
        means = [r.T_total_ms_mean for r in summaries]
        colors = [color_for(r.policy, i) for i, r in enumerate(summaries)]
        x = np.arange(len(labels))
        ax.bar(x, means, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Mean T_total (ms)")
        ax.set_title(f"{subfolder} — Policy Comparison")
        for xi, v in zip(x, means):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", color="#c0c0c0", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "comparison_plot.png"), dpi=150,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
