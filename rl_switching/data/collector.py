"""
Dual-model data collector.

Runs BOTH YOLOv8n and YOLOv8s on every frame of every video, saving:
  - Per-frame feature vectors (12-dim)
  - Both models' detections (boxes + confidences)
  - Both models' inference times
  - Per-frame IoU agreement score

Output: one HDF5 file per video, plus a combined manifest JSON.
This creates the offline dataset for bandit/RL training.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from ..config import RLConfig, FEATURE_DIM, FEATURE_NAMES
from ..features import compute_all_features
from ..rewards import compute_set_iou


def _run_model(model, img_bgr: np.ndarray, imgsz: int,
               device: str, conf: float, iou_nms: float
               ) -> Tuple[List[np.ndarray], float]:
    """Run a YOLO model and return (detections, inference_time_ms)."""
    t0 = time.perf_counter()
    res = model.predict(source=img_bgr, imgsz=imgsz, device=device,
                        conf=conf, iou=iou_nms, verbose=False)[0]
    t_ms = (time.perf_counter() - t0) * 1000.0

    dets: List[np.ndarray] = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, cf):
            dets.append(np.array([b[0], b[1], b[2], b[3], c], dtype=np.float32))
    return dets, t_ms


class DualModelCollector:
    """Collect dual-model data from a set of videos.

    Usage::

        collector = DualModelCollector(cfg, model_n, model_s)
        collector.collect_video("path/to/video.mp4", "video_001")
        collector.collect_all(video_paths_dict)
        manifest = collector.save_manifest()
    """

    def __init__(self, cfg: RLConfig, model_n, model_s):
        self.cfg = cfg
        self.model_n = model_n
        self.model_s = model_s
        self.manifest: Dict[str, dict] = {}

        os.makedirs(cfg.data_cache_dir, exist_ok=True)

    def collect_video(
        self,
        video_path: str,
        video_id: str,
        progress_callback=None,
    ) -> str:
        """Process one video, saving results to HDF5.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        video_id : str
            Unique identifier for this video (used as filename).
        progress_callback : callable, optional
            Called with (frame_idx, total_frames) for progress reporting.

        Returns
        -------
        str
            Path to the output HDF5 file.
        """
        cfg = self.cfg
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if cfg.max_frames > 0:
            total_frames = min(total_frames, cfg.max_frames * cfg.collect_stride)

        out_path = os.path.join(cfg.data_cache_dir, f"{video_id}.h5")

        # Pre-allocate lists (faster than extending HDF5 incrementally)
        all_features = []
        all_iou = []
        all_t_n = []
        all_t_s = []
        all_conf_n = []     # mean confidence of n-model
        all_conf_s = []     # mean confidence of s-model
        all_n_dets_n = []   # detection count from n-model
        all_n_dets_s = []   # detection count from s-model
        # Store raw detections as variable-length arrays
        all_dets_n = []
        all_dets_s = []

        prev_conf = 0.0
        prev_n_dets = 0
        prev_action = 0
        dwell = 0

        frame_idx = 0
        kept = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if cfg.collect_stride > 1 and (frame_idx % cfg.collect_stride) != 0:
                frame_idx += 1
                continue

            if cfg.max_frames > 0 and kept >= cfg.max_frames:
                break

            # ── Extract features ─────────────────────────────────────
            features = compute_all_features(
                frame,
                proxy_size=cfg.proxy_size,
                hist_bins=cfg.hist_bins,
                prev_conf=prev_conf,
                prev_n_dets=prev_n_dets,
                prev_action=prev_action,
                dwell_time=dwell,
            )

            # ── Run both models ──────────────────────────────────────
            dets_n, t_n = _run_model(
                self.model_n, frame, cfg.imgsz, cfg.device,
                cfg.conf_min, cfg.iou_nms
            )
            dets_s, t_s = _run_model(
                self.model_s, frame, cfg.imgsz, cfg.device,
                cfg.conf_min, cfg.iou_nms
            )

            # ── Compute IoU agreement ────────────────────────────────
            iou = compute_set_iou(dets_n, dets_s, conf_threshold=0.25)

            # ── Mean confidences ─────────────────────────────────────
            high_conf_n = [float(d[4]) for d in dets_n if float(d[4]) >= 0.25]
            high_conf_s = [float(d[4]) for d in dets_s if float(d[4]) >= 0.25]
            mean_conf_n = float(np.mean(high_conf_n)) if high_conf_n else 0.0
            mean_conf_s = float(np.mean(high_conf_s)) if high_conf_s else 0.0

            # ── Store ────────────────────────────────────────────────
            all_features.append(features)
            all_iou.append(iou)
            all_t_n.append(t_n)
            all_t_s.append(t_s)
            all_conf_n.append(mean_conf_n)
            all_conf_s.append(mean_conf_s)
            all_n_dets_n.append(len(high_conf_n))
            all_n_dets_s.append(len(high_conf_s))

            # Flatten detections for HDF5 storage
            if dets_n:
                all_dets_n.append(np.stack(dets_n))
            else:
                all_dets_n.append(np.zeros((0, 5), dtype=np.float32))
            if dets_s:
                all_dets_s.append(np.stack(dets_s))
            else:
                all_dets_s.append(np.zeros((0, 5), dtype=np.float32))

            # Update temporal state for next frame's features
            prev_conf = mean_conf_n  # use n-model conf as default
            prev_n_dets = len(high_conf_n)
            # prev_action stays 0 during collection (no switching)
            dwell += 1

            kept += 1
            frame_idx += 1

            if progress_callback:
                progress_callback(kept, total_frames)

        cap.release()

        # ── Save to HDF5 ────────────────────────────────────────────
        features_arr = np.stack(all_features)  # (N, FEATURE_DIM)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("features", data=features_arr, dtype="float32")
            f.create_dataset("iou", data=np.array(all_iou, dtype=np.float32))
            f.create_dataset("t_infer_n", data=np.array(all_t_n, dtype=np.float32))
            f.create_dataset("t_infer_s", data=np.array(all_t_s, dtype=np.float32))
            f.create_dataset("conf_n", data=np.array(all_conf_n, dtype=np.float32))
            f.create_dataset("conf_s", data=np.array(all_conf_s, dtype=np.float32))
            f.create_dataset("n_dets_n", data=np.array(all_n_dets_n, dtype=np.int32))
            f.create_dataset("n_dets_s", data=np.array(all_n_dets_s, dtype=np.int32))
            f.attrs["video_path"] = video_path
            f.attrs["video_id"] = video_id
            f.attrs["n_frames"] = kept
            f.attrs["feature_names"] = json.dumps(FEATURE_NAMES)

            # Variable-length detections: store as groups
            dets_grp = f.create_group("detections")
            for i in range(kept):
                frame_grp = dets_grp.create_group(str(i))
                frame_grp.create_dataset("n", data=all_dets_n[i])
                frame_grp.create_dataset("s", data=all_dets_s[i])

        # Update manifest
        self.manifest[video_id] = {
            "video_path": video_path,
            "h5_path": out_path,
            "n_frames": kept,
            "mean_iou": float(np.mean(all_iou)),
            "mean_t_n": float(np.mean(all_t_n)),
            "mean_t_s": float(np.mean(all_t_s)),
        }

        return out_path

    def collect_all(
        self,
        video_paths: Dict[str, str],
        progress_callback=None,
    ) -> Dict[str, str]:
        """Collect data from all videos.

        Parameters
        ----------
        video_paths : dict
            Mapping of video_id -> video_file_path.
        progress_callback : callable, optional
            Called with (video_id, video_idx, total_videos).

        Returns
        -------
        dict
            Mapping of video_id -> h5_path.
        """
        results = {}
        for i, (vid_id, vid_path) in enumerate(video_paths.items()):
            if progress_callback:
                progress_callback(vid_id, i, len(video_paths))
            print(f"[{i+1}/{len(video_paths)}] Collecting: {vid_id} ({vid_path})")
            h5_path = self.collect_video(vid_path, vid_id)
            results[vid_id] = h5_path
            print(f"  → {self.manifest[vid_id]['n_frames']} frames, "
                  f"mean IoU={self.manifest[vid_id]['mean_iou']:.3f}")
        return results

    def save_manifest(self) -> str:
        """Save the collection manifest to JSON."""
        path = os.path.join(self.cfg.data_cache_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return path
