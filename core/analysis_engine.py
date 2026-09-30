"""Thesis-analysis engine — offline dual-model evaluation with correct semantics.

This module runs INDEPENDENTLY of the live demonstrator engine (core/engine.py).

Per-frame topology is POLICY-CLASS SPECIFIC:

Static baselines (n_only / s_only):
    - n_only:  run fast only.         t_deployed = t_fast
    - s_only:  run accurate only.     t_deployed = t_accurate
    No ctrl cost, no proxy cost.

Scene-feature policies (entropy_only, combined, combined_hyst,
                        multi_proxy, local_contrast_hyst):
    1. Compute scene proxies
    2. Run controller (policy.decide)
    3. Run selected model only
    t_deployed = t_proxy + t_ctrl + t_selected_model

Confidence EMA (conf_ema):
    1. Run fast detector (watchdog — always paid)
    2. Feed fast-model confidence into EMA
    3. Run controller (policy.decide)
    4. If escalated, run accurate detector
    t_deployed(n) = t_fast + t_ctrl
    t_deployed(s) = t_fast + t_ctrl + t_accurate

Offline evaluation:
    Both models always run so that accurate-reference metrics and
    benefit_positive can be computed.  Evaluation-only inference cost
    is NEVER added to t_total_deployed_ms.

The selected output is ALWAYS one model's detections per frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np

from .inference_backend import Detections, InferenceBackend, time_inference
from .matching import greedy_match
from .metrics import (
    benefit_positive_frame,
    count_agree_frame,
    det_coverage_frame,
    det_recall_frame,
    iou_match_frame,
)
from .policies import FrameFeatures, Policy, PolicyConfig, make_policy
from .proxies import ProxyConfig, compute_proxies


_SCENE_POLICIES = frozenset({
    "entropy_only", "combined", "combined_hyst",
    "local_contrast_hyst", "multi_proxy",
})
_STATIC_POLICIES = frozenset({"n_only", "s_only"})
_CONF_EMA_POLICIES = frozenset({"conf_ema"})


@dataclass
class FrameResult:
    """Per-frame output from a thesis-analysis run."""

    frame_idx: int
    policy: str
    choice: str
    selected_model: str

    fast_dets: Detections
    accurate_dets: Detections
    selected_dets: Detections

    fast_mean_conf: float

    accurate_selected: bool
    accurate_reference_ran: bool
    fast_reference_ran: bool

    diag: Dict[str, float]
    proxy_features: Dict[str, float]

    t_fast_ms: float
    t_accurate_ms: float
    t_ctrl_ms: float
    t_proxy_ms: float
    t_total_deployed_ms: float

    iou_match: bool
    det_coverage: bool
    det_recall: float
    count_agree: bool
    benefit_positive: bool


def _needs_proxies(policy_name: str) -> bool:
    return policy_name in _SCENE_POLICIES


def _fast_mean_conf(dets: Detections) -> float:
    if len(dets) == 0:
        return 0.0
    return float(dets.scores.mean())


def run_analysis(
    video_path: str,
    fast_backend: InferenceBackend,
    accurate_backend: InferenceBackend,
    policy_name: str,
    policy_config: Optional[PolicyConfig] = None,
    proxy_config: Optional[ProxyConfig] = None,
    iou_threshold: float = 0.5,
    max_frames: int = 0,
    start_frame: int = 0,
) -> Iterator[FrameResult]:
    """Run thesis-analysis on a video, yielding one FrameResult per frame.

    Args:
        video_path:       Path to the input video file.
        fast_backend:     Inference backend for the fast (n) model.
        accurate_backend: Inference backend for the accurate (s) model.
        policy_name:      One of the 8 registered policy names.
        policy_config:    Optional override; uses defaults if None.
        proxy_config:     Scene-proxy settings; only used if policy needs proxies.
        iou_threshold:    IoU threshold for matching metrics.
        max_frames:       Stop after this many frames (0 = all frames).
        start_frame:      First video frame to process (0 = beginning).
                          Frame indices in results use original video numbering.
    """
    cfg = policy_config or PolicyConfig(name=policy_name)
    cfg.name = policy_name
    policy = make_policy(policy_name, cfg)

    pcfg = proxy_config or ProxyConfig()
    is_scene = policy_name in _SCENE_POLICIES
    is_static = policy_name in _STATIC_POLICIES
    is_conf_ema = policy_name in _CONF_EMA_POLICIES

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    last_fast_mean_conf = 0.0
    frame_idx = int(start_frame)
    frames_processed = 0

    try:
        while True:
            if 0 < max_frames <= frames_processed:
                break
            ret, frame = cap.read()
            if not ret:
                break

            # Deployed timing accumulators (only what the policy pays)
            t_fast_dep = 0.0
            t_acc_dep = 0.0
            t_ctrl_dep = 0.0
            t_proxy_dep = 0.0

            fast_dets: Optional[Detections] = None
            accurate_dets: Optional[Detections] = None
            fast_mc = 0.0
            proxy_feats: Dict[str, float] = {}

            # ============================================================
            # A. Pre-decision deployed work
            # ============================================================

            # conf_ema: fast detector is the watchdog signal (deployed)
            if is_conf_ema:
                fast_dets, t_fast_dep = time_inference(fast_backend, frame)
                fast_mc = _fast_mean_conf(fast_dets)

            # Scene policies: compute proxies (deployed)
            if is_scene:
                t0_proxy = time.perf_counter()
                proxy_feats = compute_proxies(frame, pcfg)
                t_proxy_dep = (time.perf_counter() - t0_proxy) * 1000.0

            # ============================================================
            # B. Decision
            # ============================================================

            t0_ctrl = time.perf_counter()
            features = FrameFeatures(
                frame_idx=frame_idx,
                L=proxy_feats.get("L"),
                H=proxy_feats.get("H"),
                color_entropy=proxy_feats.get("color_entropy"),
                tenengrad=proxy_feats.get("tenengrad"),
                edge_density=proxy_feats.get("edge_density"),
                local_contrast=proxy_feats.get("local_contrast"),
                bright_fraction=proxy_feats.get("bright_fraction"),
                hue_std=proxy_feats.get("hue_std"),
                last_mean_conf=last_fast_mean_conf,
            )
            choice = policy.decide(features)
            diag = dict(policy.last_diag)
            t_ctrl_raw = (time.perf_counter() - t0_ctrl) * 1000.0

            # Static baselines have no deployed ctrl cost
            if not is_static:
                t_ctrl_dep = t_ctrl_raw

            # ============================================================
            # C. Deployed inference for selected model
            # ============================================================

            if choice == "n":
                if fast_dets is None:
                    fast_dets, t_fast_dep = time_inference(fast_backend, frame)
                    fast_mc = _fast_mean_conf(fast_dets)
                selected_dets = fast_dets
                selected_model = "n"
            else:
                accurate_dets, t_acc_dep = time_inference(accurate_backend, frame)
                selected_dets = accurate_dets
                selected_model = "s"

            accurate_selected = (choice == "s")

            # ============================================================
            # D. Evaluation-only inference (not added to deployed cost)
            # ============================================================

            if fast_dets is None:
                fast_dets, _ = time_inference(fast_backend, frame)
                fast_mc = _fast_mean_conf(fast_dets)

            if accurate_dets is None:
                accurate_dets, _ = time_inference(accurate_backend, frame)

            # Both models always available in offline analysis
            fast_reference_ran = True
            accurate_reference_ran = True

            # ============================================================
            # E. Deployed cost = sum of deployed-only components
            # ============================================================

            t_deployed = t_fast_dep + t_acc_dep + t_ctrl_dep + t_proxy_dep

            # ============================================================
            # F. Metrics (policy output vs accurate reference)
            # ============================================================

            iou_m = iou_match_frame(
                selected_dets.boxes, accurate_dets.boxes,
                iou_threshold,
                policy_classes=selected_dets.classes,
                ref_classes=accurate_dets.classes,
            )
            det_cov = det_coverage_frame(
                selected_dets.boxes, accurate_dets.boxes,
                iou_threshold,
                policy_classes=selected_dets.classes,
                ref_classes=accurate_dets.classes,
            )
            det_rec = det_recall_frame(
                selected_dets.boxes, accurate_dets.boxes,
                iou_threshold,
                policy_classes=selected_dets.classes,
                ref_classes=accurate_dets.classes,
            )
            cnt_ag = count_agree_frame(selected_dets.boxes, accurate_dets.boxes)
            ben_pos = benefit_positive_frame(
                fast_dets.boxes, accurate_dets.boxes,
                iou_threshold,
                fast_classes=fast_dets.classes,
                accurate_classes=accurate_dets.classes,
            )

            # ============================================================
            # G. Update EMA input for next frame
            # ============================================================

            last_fast_mean_conf = fast_mc

            yield FrameResult(
                frame_idx=frame_idx,
                policy=policy_name,
                choice=choice,
                selected_model=selected_model,
                fast_dets=fast_dets,
                accurate_dets=accurate_dets,
                selected_dets=selected_dets,
                fast_mean_conf=fast_mc,
                accurate_selected=accurate_selected,
                accurate_reference_ran=accurate_reference_ran,
                fast_reference_ran=fast_reference_ran,
                diag=diag,
                proxy_features=proxy_feats,
                t_fast_ms=t_fast_dep,
                t_accurate_ms=t_acc_dep,
                t_ctrl_ms=t_ctrl_dep,
                t_proxy_ms=t_proxy_dep,
                t_total_deployed_ms=t_deployed,
                iou_match=iou_m,
                det_coverage=det_cov,
                det_recall=det_rec,
                count_agree=cnt_ag,
                benefit_positive=ben_pos,
            )
            frame_idx += 1
            frames_processed += 1
    finally:
        cap.release()


def summarise_analysis(results: List[FrameResult]) -> Dict:
    """Aggregate per-frame results into a single run summary."""
    n = len(results)
    if n == 0:
        return {"n_frames": 0}

    def _rate(attr: str) -> float:
        return sum(1 for r in results if getattr(r, attr)) / n

    def _mean(attr: str) -> float:
        return sum(getattr(r, attr) for r in results) / n

    return {
        "n_frames": n,
        "policy": results[0].policy,
        "accurate_usage_rate": sum(1 for r in results if r.choice == "s") / n,
        "mean_iou_match": _rate("iou_match"),
        "mean_det_coverage": _rate("det_coverage"),
        "mean_det_recall": _mean("det_recall"),
        "mean_count_agree": _rate("count_agree"),
        "benefit_rate": _rate("benefit_positive"),
        "mean_t_total_deployed_ms": _mean("t_total_deployed_ms"),
        "mean_t_fast_ms": _mean("t_fast_ms"),
        "mean_t_accurate_ms": _mean("t_accurate_ms"),
        "mean_t_ctrl_ms": _mean("t_ctrl_ms"),
        "mean_t_proxy_ms": _mean("t_proxy_ms"),
    }


__all__ = [
    "FrameResult",
    "run_analysis",
    "summarise_analysis",
]
