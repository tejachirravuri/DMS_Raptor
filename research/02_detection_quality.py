"""
Phase 1B: Detection quality evaluator using s_only as pseudo-oracle.

For each video, runs BOTH models on every frame and computes:
  - Per-policy "missed detection rate" (detections s_only found but policy's
    chosen model missed)
  - Per-policy "false positive rate" (detections n_only found but s_only didn't)
  - Detection coverage (fraction of s_only detections preserved)
  - Confidence-weighted coverage (preserves high-confidence detections)
  - Pareto data: (mean_latency, detection_coverage) per policy

This uses the existing dual-model validation engine but adds per-policy
quality accounting on top.

Usage:
    python research/02_detection_quality.py --video <path> --material glass
    python research/02_detection_quality.py --all    # run all videos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RunConfig, InferenceParams, POLICIES
from core.research_paths import MODELS, VIDEOS
from core.engine import (
    run_model_single, nms_xyxy, box_iou_xyxy,
    complexity_proxies_fast, FastRollingPercentile,
    _compute_extended_proxies, _compute_color_entropy,
)
from core.niqe import compute_nriqa_score


# ── Configuration ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "research_results" / "02_detection_quality"

# Inference params matching DMS-Raptor defaults
INF = InferenceParams(imgsz=640, device="cpu", conf_min=0.001, iou_nms=0.45,
                      conf_show=0.25, max_frames=0, stride=1)

CONF_THRESHOLD = 0.25  # confidence threshold for "real" detections
IOU_MATCH = 0.5        # IoU threshold for box matching


# ── Box matching ───────────────────────────────────────────────────────
def match_detections(dets_a: List[np.ndarray], dets_b: List[np.ndarray],
                     iou_th: float = 0.5
                     ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy bipartite matching by IoU. Returns (matched, unmatched_a, unmatched_b)."""
    if not dets_a or not dets_b:
        return [], list(range(len(dets_a))), list(range(len(dets_b)))
    n_a, n_b = len(dets_a), len(dets_b)
    iou_mat = np.zeros((n_a, n_b), dtype=np.float32)
    for i in range(n_a):
        for j in range(n_b):
            iou_mat[i, j] = box_iou_xyxy(dets_a[i][:4], dets_b[j][:4])
    matched, used_a, used_b = [], set(), set()
    while True:
        mx = iou_mat.max()
        if mx < iou_th:
            break
        idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        i, j = int(idx[0]), int(idx[1])
        matched.append((i, j, float(mx)))
        used_a.add(i); used_b.add(j)
        iou_mat[i, :] = 0; iou_mat[:, j] = 0
    return matched, [i for i in range(n_a) if i not in used_a], \
           [j for j in range(n_b) if j not in used_b]


# ── Policy simulator ──────────────────────────────────────────────────
class PolicySimulator:
    """Simulate a DMS policy decision given per-frame signals.

    Instead of re-running inference, this replays the policy logic on
    precomputed signals to determine which model *would have been* chosen.
    """

    def __init__(self, policy: str, cfg: RunConfig):
        self.policy = policy
        self.cfg = cfg
        self.last_choice: Optional[str] = None
        self.dwell = 0
        self.switch_events = []
        self.frame_count = 0

        # Rolling percentiles for C-score normalization
        self.rolling_L = FastRollingPercentile(cfg.history_window_size)
        self.rolling_H = FastRollingPercentile(cfg.history_window_size)
        self.rolling_C = FastRollingPercentile(cfg.history_window_size)
        self.c_ema_state: Optional[float] = None

        # EMA states for conf_ema
        self.conf_ema_fast: Optional[float] = None
        self.conf_ema_slow: Optional[float] = None
        self.last_mean_conf: float = 0.0

        # EMA states for niqe_switch
        self.niqe_ema_fast: Optional[float] = None
        self.niqe_ema_slow: Optional[float] = None

        # EMA states for multi_proxy
        self.mp_ema: Dict[str, Tuple[float, float]] = {}

    def decide(self, L: float, H: float, mean_conf_n: float,
               niqe_val: float, extended_proxies: Optional[dict] = None,
               color_entropy: float = 0.0,
               mean_conf_s: float = 0.0) -> str:
        """Return 'n' or 's' for this frame.

        For conf_ema: last_mean_conf is updated from the chosen model's
        detections AFTER the final routing decision (including actuator
        protections). This matches the engine's behavior where the EMA
        signal comes from whichever model actually ran.
        """
        cfg = self.cfg
        policy = self.policy
        C: Optional[float] = None

        # ── Static policies ──
        if policy == "n_only":
            choice = "n"
        elif policy == "s_only":
            choice = "s"
        elif policy in ("entropy_only", "combined", "combined_hyst"):
            # Scene-proxy policies
            self.rolling_L.add(L)
            self.rolling_H.add(H)
            L_lo = self.rolling_L.percentile(0)
            L_hi = self.rolling_L.percentile(100)
            H_lo = self.rolling_H.percentile(0)
            H_hi = self.rolling_H.percentile(100)
            Ln = float(np.clip((L - L_lo) / (L_hi - L_lo + 1e-9), 0, 1))
            Hn = float(np.clip((H - H_lo) / (H_hi - H_lo + 1e-9), 0, 1))
            C_raw = cfg.alpha * Ln + (1 - cfg.alpha) * Hn

            if cfg.c_ema_beta <= 0:
                C = C_raw
            else:
                if self.c_ema_state is None:
                    self.c_ema_state = C_raw
                else:
                    self.c_ema_state = (1 - cfg.c_ema_beta) * self.c_ema_state + cfg.c_ema_beta * C_raw
                C = self.c_ema_state

            self.rolling_C.add(C)
            Hmed = self.rolling_H.percentile(50)

            if policy == "entropy_only":
                choice = "s" if H >= Hmed else "n"
            elif policy == "combined":
                choice = "s" if C >= cfg.combined_mid else "n"
            elif policy == "combined_hyst":
                if self.last_choice is None:
                    self.last_choice = "n"
                choice = ("s" if C >= cfg.c_high else "n") if self.last_choice == "n" \
                    else ("n" if C <= cfg.c_low else "s")

        elif policy == "conf_ema":
            bf = cfg.conf_ema_fast_beta
            bs = cfg.conf_ema_slow_beta
            if self.conf_ema_fast is None:
                self.conf_ema_fast = self.last_mean_conf
                self.conf_ema_slow = self.last_mean_conf
            else:
                self.conf_ema_fast = bf * self.last_mean_conf + (1 - bf) * self.conf_ema_fast
                self.conf_ema_slow = bs * self.last_mean_conf + (1 - bs) * self.conf_ema_slow
            conf_drop = max(0.0, (self.conf_ema_slow - self.conf_ema_fast) / (self.conf_ema_slow + 1e-9))
            C = conf_drop
            if self.last_choice is None:
                self.last_choice = "n"
            if self.frame_count < 5:
                choice = "n"
            else:
                choice = ("s" if C >= cfg.conf_ema_c_high else "n") if self.last_choice == "n" \
                    else ("n" if C <= cfg.conf_ema_c_low else "s")

        elif policy == "niqe_switch":
            bf = cfg.niqe_fast_beta
            bs = cfg.niqe_slow_beta
            if self.niqe_ema_fast is None:
                self.niqe_ema_fast = niqe_val
                self.niqe_ema_slow = niqe_val
            else:
                self.niqe_ema_fast = bf * niqe_val + (1 - bf) * self.niqe_ema_fast
                self.niqe_ema_slow = bs * niqe_val + (1 - bs) * self.niqe_ema_slow
            niqe_rise = max(0.0, (self.niqe_ema_fast - self.niqe_ema_slow) / (self.niqe_ema_slow + 1e-9))
            C = niqe_rise
            if self.last_choice is None:
                self.last_choice = "n"
            choice = ("s" if C >= cfg.niqe_c_high else "n") if self.last_choice == "n" \
                else ("n" if C <= cfg.niqe_c_low else "s")

        elif policy == "multi_proxy":
            if extended_proxies is None:
                extended_proxies = {}
            mp_raw = {"L": L, "H": H, **extended_proxies, "color_entropy": color_entropy}
            proxy_weights = {
                "L": cfg.mp_w_laplacian, "H": cfg.mp_w_entropy,
                "tenengrad": cfg.mp_w_tenengrad, "edge_density": cfg.mp_w_edge_density,
                "local_contrast": cfg.mp_w_local_contrast, "brenner": cfg.mp_w_brenner,
                "color_entropy": cfg.mp_w_color_entropy,
            }
            bf_mp, bs_mp = cfg.mp_fast_beta, cfg.mp_slow_beta
            weighted_sum, w_sum = 0.0, 0.0
            for pname, pval in mp_raw.items():
                w = proxy_weights.get(pname, 0.0)
                if w < 1e-12:
                    continue
                pval_f = float(pval)
                if pname not in self.mp_ema:
                    self.mp_ema[pname] = (pval_f, pval_f)
                else:
                    fp, sp = self.mp_ema[pname]
                    self.mp_ema[pname] = (bf_mp * pval_f + (1 - bf_mp) * fp,
                                          bs_mp * pval_f + (1 - bs_mp) * sp)
                fv, sv = self.mp_ema[pname]
                drop = abs(fv - sv) / (sv + 1e-9)
                weighted_sum += w * drop
                w_sum += w
            C = weighted_sum / w_sum if w_sum > 1e-12 else 0.0
            if self.last_choice is None:
                self.last_choice = "n"
            choice = ("s" if C >= cfg.mp_c_high else "n") if self.last_choice == "n" \
                else ("n" if C <= cfg.mp_c_low else "s")
        else:
            choice = "n"

        # ── Actuator protections ──
        if policy not in ("n_only", "s_only"):
            if self.last_choice is None:
                self.last_choice = choice
            # Clean old switch events
            self.switch_events = [e for e in self.switch_events
                                  if e > self.frame_count - 100]
            requested_switch = (choice != self.last_choice)
            if requested_switch and self.dwell < cfg.min_dwell_frames:
                choice = self.last_choice
                requested_switch = False
            if requested_switch and len(self.switch_events) >= cfg.max_switches_per_100:
                choice = self.last_choice
                requested_switch = False
            if choice == self.last_choice:
                self.dwell += 1
            else:
                self.switch_events.append(self.frame_count)
                self.dwell = 1
                self.last_choice = choice
        else:
            self.dwell += 1

        # conf_ema: update last_mean_conf from the CHOSEN model's detections.
        # This matches engine.py where cur_mean_conf comes from whichever
        # model actually ran on that frame.
        if policy == "conf_ema":
            self.last_mean_conf = mean_conf_n if choice == "n" else mean_conf_s

        self.frame_count += 1
        return choice


# ── Per-video analysis ─────────────────────────────────────────────────
@dataclass
class PolicyQuality:
    """Detection quality metrics for a single policy on a single video."""
    policy: str = ""
    total_frames: int = 0
    n_frames: int = 0           # frames assigned to n-model
    s_frames: int = 0           # frames assigned to s-model
    s_usage_pct: float = 0.0    # % of frames using s-model

    # Detection quality (s_only as oracle)
    total_s_oracle_dets: int = 0       # total detections by s_only (oracle)
    total_policy_dets: int = 0         # total detections by policy's chosen model
    matched_with_oracle: int = 0       # policy dets that match oracle dets
    missed_by_policy: int = 0          # oracle dets that policy missed
    extra_by_policy: int = 0           # policy dets not in oracle (potential FPs or correct extras by n)
    detection_coverage: float = 0.0    # matched / oracle_total
    missed_rate: float = 0.0           # missed / oracle_total

    # Confidence-weighted coverage
    conf_weighted_coverage: float = 0.0  # sum(conf of matched) / sum(conf of oracle)

    # Timing
    mean_T_total_ms: float = 0.0
    p95_T_total_ms: float = 0.0
    mean_T_scene_ms: float = 0.0
    mean_T_infer_ms: float = 0.0

    # Switching
    switches: int = 0
    sw_per_100: float = 0.0


def analyze_video(video_path: str, material: str, max_frames: int = 0,
                  stride: int = 1) -> List[PolicyQuality]:
    """Run dual-model inference and compute per-policy detection quality."""
    from ultralytics import YOLO

    video_name = Path(video_path).stem
    print(f"\n{'='*70}")
    print(f"  Analyzing: {video_name} ({material})")
    print(f"{'='*70}")

    model_n = YOLO(MODELS[material]["y8n"])
    model_s = YOLO(MODELS[material]["y8s"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    effective_frames = total_video_frames // max(1, stride)
    if max_frames > 0:
        effective_frames = min(effective_frames, max_frames)

    # Initialize policy simulators for all 8 policies
    cfg = RunConfig()  # default parameters
    simulators = {p: PolicySimulator(p, RunConfig(policy=p)) for p in POLICIES}

    # Per-frame storage
    frame_data = []  # list of dicts with per-frame info
    policy_choices = {p: [] for p in POLICIES}

    idx = 0
    kept = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if stride > 1 and idx % stride != 0:
            idx += 1
            continue
        if max_frames > 0 and kept >= max_frames:
            break

        # Progress
        if kept % 100 == 0:
            elapsed = time.time() - t_start
            fps = kept / elapsed if elapsed > 0 else 0
            eta = (effective_frames - kept) / fps if fps > 0 else 0
            print(f"  Frame {kept}/{effective_frames} | "
                  f"{fps:.1f} fps | ETA {eta:.0f}s", end="\r")

        # ── 1) Scene proxies ──
        t0 = time.perf_counter()
        L, H = complexity_proxies_fast(frame, downsample=8, proxy_size=160,
                                        hist_bins=64)
        niqe_val = compute_nriqa_score(frame, proxy_size=160)

        # Extended proxies for multi_proxy
        small = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ext = _compute_extended_proxies(gray)
        col_ent = _compute_color_entropy(small)
        T_scene = (time.perf_counter() - t0) * 1000.0

        # ── 2) Run BOTH models ──
        t0 = time.perf_counter()
        dets_n_raw = run_model_single(model_n, frame, INF.imgsz, INF.device,
                                       INF.conf_min, INF.iou_nms)
        T_infer_n = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        dets_s_raw = run_model_single(model_s, frame, INF.imgsz, INF.device,
                                       INF.conf_min, INF.iou_nms)
        T_infer_s = (time.perf_counter() - t0) * 1000.0

        # Filter by confidence threshold
        dets_n = [d for d in dets_n_raw if float(d[4]) >= CONF_THRESHOLD]
        dets_s = [d for d in dets_s_raw if float(d[4]) >= CONF_THRESHOLD]

        # Mean confidence for conf_ema policy (both models needed)
        mean_conf_n = float(np.mean([float(d[4]) for d in dets_n])) if dets_n else 0.0
        mean_conf_s = float(np.mean([float(d[4]) for d in dets_s])) if dets_s else 0.0

        # ── 3) Policy decisions ──
        for p in POLICIES:
            choice = simulators[p].decide(
                L=L, H=H, mean_conf_n=mean_conf_n, niqe_val=niqe_val,
                extended_proxies=ext, color_entropy=col_ent,
                mean_conf_s=mean_conf_s,
            )
            policy_choices[p].append(choice)

        # ── 4) Store frame data ──
        frame_data.append({
            "idx": kept,
            "dets_n": dets_n,
            "dets_s": dets_s,
            "T_scene": T_scene,
            "T_infer_n": T_infer_n,
            "T_infer_s": T_infer_s,
            "L": L, "H": H,
            "mean_conf_n": mean_conf_n,
            "mean_conf_s": mean_conf_s,
            "niqe": niqe_val,
            "n_det_count": len(dets_n),
            "s_det_count": len(dets_s),
        })

        kept += 1
        idx += 1

    cap.release()
    print(f"  Processed {kept} frames in {time.time() - t_start:.1f}s")

    # ── 5) Compute per-policy quality metrics ──
    results = []
    for p in POLICIES:
        pq = PolicyQuality(policy=p, total_frames=kept)
        choices = policy_choices[p]

        total_oracle_dets = 0
        total_policy_dets = 0
        total_matched = 0
        total_missed = 0
        total_extra = 0
        oracle_conf_sum = 0.0
        matched_conf_sum = 0.0
        timing_totals = []
        switches = 0

        for i, fd in enumerate(frame_data):
            choice = choices[i]

            # Count model usage
            if choice == "n":
                pq.n_frames += 1
                chosen_dets = fd["dets_n"]
                T_infer = fd["T_infer_n"]
            else:
                pq.s_frames += 1
                chosen_dets = fd["dets_s"]
                T_infer = fd["T_infer_s"]

            # Oracle = s_only detections
            oracle_dets = fd["dets_s"]

            # Timing: scene + infer (ctrl is negligible)
            T_scene = fd["T_scene"] if p not in ("n_only", "s_only", "conf_ema") else 0.0
            if p == "conf_ema":
                T_scene = 0.0  # conf_ema has no scene processing
            T_total = T_scene + T_infer
            timing_totals.append(T_total)

            # Match chosen model's detections against oracle
            matched, unmatched_chosen, unmatched_oracle = match_detections(
                chosen_dets, oracle_dets, IOU_MATCH
            )

            total_oracle_dets += len(oracle_dets)
            total_policy_dets += len(chosen_dets)
            total_matched += len(matched)
            total_missed += len(unmatched_oracle)
            total_extra += len(unmatched_chosen)

            # Confidence-weighted coverage
            for d in oracle_dets:
                oracle_conf_sum += float(d[4])
            for m_i, m_j, m_iou in matched:
                matched_conf_sum += float(oracle_dets[m_j][4])

            # Switches
            if i > 0 and choices[i] != choices[i - 1]:
                switches += 1

        # Aggregate
        pq.s_usage_pct = 100.0 * pq.s_frames / max(1, kept)
        pq.total_s_oracle_dets = total_oracle_dets
        pq.total_policy_dets = total_policy_dets
        pq.matched_with_oracle = total_matched
        pq.missed_by_policy = total_missed
        pq.extra_by_policy = total_extra
        pq.detection_coverage = total_matched / max(1, total_oracle_dets)
        pq.missed_rate = total_missed / max(1, total_oracle_dets)
        pq.conf_weighted_coverage = matched_conf_sum / max(1e-9, oracle_conf_sum)

        t_arr = np.array(timing_totals)
        pq.mean_T_total_ms = float(np.mean(t_arr)) if len(t_arr) > 0 else 0.0
        pq.p95_T_total_ms = float(np.percentile(t_arr, 95)) if len(t_arr) > 0 else 0.0
        pq.mean_T_scene_ms = float(np.mean([fd["T_scene"] for fd in frame_data]))
        pq.mean_T_infer_ms = pq.mean_T_total_ms - pq.mean_T_scene_ms
        pq.switches = switches
        pq.sw_per_100 = 100.0 * switches / max(1, kept)

        results.append(pq)

    return results, frame_data, policy_choices


def print_quality_table(results: List[PolicyQuality], video_name: str):
    """Print a formatted quality comparison table."""
    print(f"\n{'='*90}")
    print(f"  Detection Quality: {video_name}")
    print(f"  Oracle: s_only | Conf threshold: {CONF_THRESHOLD} | IoU match: {IOU_MATCH}")
    print(f"{'='*90}")
    print(f"{'Policy':<16} {'s%':>5} {'Coverage':>9} {'Missed%':>8} "
          f"{'ConfWCov':>8} {'MeanT':>7} {'P95T':>7} {'Sw/100':>7}")
    print("-" * 90)

    for pq in results:
        print(f"{pq.policy:<16} {pq.s_usage_pct:>5.1f} "
              f"{pq.detection_coverage:>9.4f} {pq.missed_rate*100:>7.2f}% "
              f"{pq.conf_weighted_coverage:>8.4f} "
              f"{pq.mean_T_total_ms:>7.1f} {pq.p95_T_total_ms:>7.1f} "
              f"{pq.sw_per_100:>7.2f}")

    # Pareto analysis
    print(f"\n  Pareto Analysis (latency vs coverage):")
    sorted_r = sorted(results, key=lambda r: r.mean_T_total_ms)
    best_cov = -1.0
    for pq in sorted_r:
        is_pareto = pq.detection_coverage > best_cov
        marker = " ** PARETO" if is_pareto else ""
        if is_pareto:
            best_cov = pq.detection_coverage
        print(f"    {pq.policy:<16} T={pq.mean_T_total_ms:>7.1f}ms  "
              f"Cov={pq.detection_coverage:.4f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Detection quality evaluator")
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--material", type=str, default="glass",
                        choices=["glass", "porcelain"])
    parser.add_argument("--all", action="store_true", help="Run all videos")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit frames per video (0=all)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every N-th frame")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        all_results = {}
        for material, video_list in VIDEOS.items():
            for vpath in video_list:
                if not os.path.isfile(vpath):
                    print(f"WARNING: video not found: {vpath}, skipping")
                    continue
                vname = Path(vpath).stem
                results, frame_data, choices = analyze_video(
                    vpath, material, args.max_frames, args.stride
                )
                print_quality_table(results, vname)

                # Save per-video results
                out = {pq.policy: asdict(pq) for pq in results}
                out_file = OUTPUT_DIR / f"{material}_{vname}_quality.json"
                with open(out_file, "w") as f:
                    json.dump(out, f, indent=2)
                all_results[f"{material}/{vname}"] = out

        # Save combined
        with open(OUTPUT_DIR / "all_detection_quality.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Print aggregate summary
        print(f"\n\n{'='*90}")
        print("AGGREGATE SUMMARY ACROSS ALL VIDEOS")
        print("=" * 90)
        for p in POLICIES:
            coverages, missed_rates, latencies = [], [], []
            for vkey, vdata in all_results.items():
                if p in vdata:
                    coverages.append(vdata[p]["detection_coverage"])
                    missed_rates.append(vdata[p]["missed_rate"])
                    latencies.append(vdata[p]["mean_T_total_ms"])
            if coverages:
                print(f"  {p:<16} Coverage: {np.mean(coverages):.4f} +/- {np.std(coverages):.4f} | "
                      f"Missed: {np.mean(missed_rates)*100:.2f}% | "
                      f"Latency: {np.mean(latencies):.1f}ms")

    elif args.video:
        if not os.path.isfile(args.video):
            print(f"ERROR: video not found: {args.video}")
            sys.exit(1)
        results, frame_data, choices = analyze_video(
            args.video, args.material, args.max_frames, args.stride
        )
        vname = Path(args.video).stem
        print_quality_table(results, vname)

        out = {pq.policy: asdict(pq) for pq in results}
        out_file = OUTPUT_DIR / f"{args.material}_{vname}_quality.json"
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {out_file}")

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python research/02_detection_quality.py --all --stride 3")
        print("  python research/02_detection_quality.py --video path/to/video.mp4 --material glass")


if __name__ == "__main__":
    main()
