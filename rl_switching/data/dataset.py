"""
Offline dataset for bandit and RL training.

Loads pre-collected HDF5 files (from DualModelCollector) and provides:
  - SwitchingDataset: frame-level access for bandit training
  - SequentialDataset: video-level episodes for DQN training
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from ..config import RLConfig, FEATURE_DIM, ACTION_N, ACTION_S
from ..features import FeatureNormalizer
from ..rewards import compute_set_iou


class SwitchingDataset:
    """Frame-level dataset for contextual bandit training.

    Loads all HDF5 files from a manifest and provides:
      - Random access to individual frames
      - Train/test splitting by video
      - Feature normalization (fit on training set)

    Each sample is a dict with keys:
      features, iou, t_infer_n, t_infer_s, conf_n, conf_s,
      n_dets_n, n_dets_s, video_id, frame_idx
    """

    def __init__(
        self,
        data_dir: str,
        video_ids: Optional[List[str]] = None,
        normalize: bool = True,
        clip: float = 5.0,
    ):
        """
        Parameters
        ----------
        data_dir : str
            Directory containing HDF5 files and manifest.json.
        video_ids : list of str, optional
            Subset of videos to load.  If None, loads all from manifest.
        normalize : bool
            Whether to z-score normalize features.
        clip : float
            Clip normalized features to [-clip, clip].
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.clip = clip

        # Load manifest
        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        if video_ids is None:
            video_ids = list(manifest.keys())
        self.video_ids = video_ids

        # Load all data into memory
        self.features_list: List[np.ndarray] = []
        self.iou_list: List[np.ndarray] = []
        self.t_n_list: List[np.ndarray] = []
        self.t_s_list: List[np.ndarray] = []
        self.conf_n_list: List[np.ndarray] = []
        self.conf_s_list: List[np.ndarray] = []
        self.n_dets_n_list: List[np.ndarray] = []
        self.n_dets_s_list: List[np.ndarray] = []
        self.video_id_per_frame: List[str] = []
        self.frame_idx_per_frame: List[int] = []

        # Also store per-video data for sequential access
        self.video_data: Dict[str, dict] = {}

        for vid_id in video_ids:
            if vid_id not in manifest:
                print(f"Warning: {vid_id} not in manifest, skipping")
                continue

            h5_path = manifest[vid_id]["h5_path"]
            if not os.path.exists(h5_path):
                # Try relative path from data_dir
                h5_path = os.path.join(data_dir, f"{vid_id}.h5")
            if not os.path.exists(h5_path):
                print(f"Warning: HDF5 not found for {vid_id}, skipping")
                continue

            with h5py.File(h5_path, "r") as f:
                features = f["features"][:]
                iou = f["iou"][:]
                t_n = f["t_infer_n"][:]
                t_s = f["t_infer_s"][:]
                conf_n = f["conf_n"][:]
                conf_s = f["conf_s"][:]
                n_dets_n = f["n_dets_n"][:]
                n_dets_s = f["n_dets_s"][:]

            n_frames = len(features)
            start_idx = len(self.features_list)

            self.features_list.append(features)
            self.iou_list.append(iou)
            self.t_n_list.append(t_n)
            self.t_s_list.append(t_s)
            self.conf_n_list.append(conf_n)
            self.conf_s_list.append(conf_s)
            self.n_dets_n_list.append(n_dets_n)
            self.n_dets_s_list.append(n_dets_s)
            self.video_id_per_frame.extend([vid_id] * n_frames)
            self.frame_idx_per_frame.extend(range(n_frames))

            self.video_data[vid_id] = {
                "features": features,
                "iou": iou,
                "t_infer_n": t_n,
                "t_infer_s": t_s,
                "conf_n": conf_n,
                "conf_s": conf_s,
                "n_dets_n": n_dets_n,
                "n_dets_s": n_dets_s,
                "n_frames": n_frames,
            }

        # Concatenate
        if self.features_list:
            self.features = np.concatenate(self.features_list, axis=0)
            self.iou = np.concatenate(self.iou_list)
            self.t_n = np.concatenate(self.t_n_list)
            self.t_s = np.concatenate(self.t_s_list)
            self.conf_n = np.concatenate(self.conf_n_list)
            self.conf_s = np.concatenate(self.conf_s_list)
            self.n_dets_n = np.concatenate(self.n_dets_n_list)
            self.n_dets_s = np.concatenate(self.n_dets_s_list)
        else:
            self.features = np.zeros((0, FEATURE_DIM), dtype=np.float32)
            self.iou = np.zeros(0, dtype=np.float32)
            self.t_n = np.zeros(0, dtype=np.float32)
            self.t_s = np.zeros(0, dtype=np.float32)
            self.conf_n = np.zeros(0, dtype=np.float32)
            self.conf_s = np.zeros(0, dtype=np.float32)
            self.n_dets_n = np.zeros(0, dtype=np.int32)
            self.n_dets_s = np.zeros(0, dtype=np.int32)

        # Fit normalizer
        self.normalizer = FeatureNormalizer(FEATURE_DIM)
        if normalize and len(self.features) > 0:
            self.normalizer.fit(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        feat = self.features[idx]
        if self.normalize:
            feat = self.normalizer.transform(feat, clip=self.clip)
        return {
            "features": feat,
            "iou": float(self.iou[idx]),
            "t_infer_n": float(self.t_n[idx]),
            "t_infer_s": float(self.t_s[idx]),
            "conf_n": float(self.conf_n[idx]),
            "conf_s": float(self.conf_s[idx]),
            "n_dets_n": int(self.n_dets_n[idx]),
            "n_dets_s": int(self.n_dets_s[idx]),
            "video_id": self.video_id_per_frame[idx],
            "frame_idx": self.frame_idx_per_frame[idx],
        }

    def get_video_episode(self, video_id: str) -> dict:
        """Get all frames for a video as arrays (for sequential RL)."""
        vd = self.video_data[video_id]
        if self.normalize:
            features = self.normalizer.transform(vd["features"], clip=self.clip)
        else:
            features = vd["features"]
        return {
            "features": features,
            "iou": vd["iou"],
            "t_infer_n": vd["t_infer_n"],
            "t_infer_s": vd["t_infer_s"],
            "conf_n": vd["conf_n"],
            "conf_s": vd["conf_s"],
            "n_dets_n": vd["n_dets_n"],
            "n_dets_s": vd["n_dets_s"],
            "n_frames": vd["n_frames"],
        }

    def shuffled_indices(self, seed: Optional[int] = None) -> np.ndarray:
        """Return shuffled frame indices for epoch iteration."""
        rng = np.random.RandomState(seed)
        idx = np.arange(len(self))
        rng.shuffle(idx)
        return idx

    def label_optimal_action(self, iou_threshold: float = 0.5) -> np.ndarray:
        """Compute 'ground truth' optimal action for each frame.

        If IoU >= threshold, n is sufficient → optimal action = 0 (n).
        If IoU < threshold, s is needed → optimal action = 1 (s).
        """
        return (self.iou < iou_threshold).astype(np.int32)
