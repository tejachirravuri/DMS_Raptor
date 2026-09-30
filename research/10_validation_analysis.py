"""
Phase 10: Comprehensive Validation Analysis

Reads overnight pipeline results (from 09_overnight_full_pipeline.py) AND
detection quality results (from 02_detection_quality.py) to produce a
complete validation report proving the proxy-based routing mechanism works.

Analyses produced:
  1. Detection Quality: Precision, Recall, F1 per policy (s_only as oracle)
  2. Proxy-Routing Correlation: Do image proxies predict when s-model is needed?
  3. Routing Accuracy: When policy switches to s, does s actually find MORE?
  4. Proxy-Disagreement Correlation: Pearson r for each proxy vs model disagreement
  5. Cross-Video Consistency: Coefficient of variation across videos
  6. Grand Summary Table: LaTeX-ready table for the paper

Data sources:
  - research_results/overnight_full/  (from 09_overnight_full_pipeline.py)
  - research_results/02_detection_quality/  (from 02_detection_quality.py)

Usage:
    python research/10_validation_analysis.py
    python research/10_validation_analysis.py --overnight-dir <path>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import POLICIES

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
RESEARCH_ROOT = Path(__file__).resolve().parent.parent / "research_results"
OVERNIGHT_DIR = RESEARCH_ROOT / "overnight_full"
DETECTION_DIR = RESEARCH_ROOT / "02_detection_quality"
OUTPUT_DIR = RESEARCH_ROOT / "10_validation"

POLICY_SHORT = {
    "n_only": "n-only", "s_only": "s-only", "entropy_only": "entropy",
    "combined": "combined", "combined_hyst": "comb-hyst",
    "conf_ema": "conf-ema", "niqe_switch": "niqe-sw", "multi_proxy": "multi-px",
}

POLICY_COLORS = {
    "n_only": "#4CAF50", "s_only": "#E94560", "entropy_only": "#FFA726",
    "combined": "#42A5F5", "combined_hyst": "#AB47BC",
    "conf_ema": "#26C6DA", "niqe_switch": "#EC407A", "multi_proxy": "#66BB6A",
}


def pearsonr(x, y):
    """Pearson r with approximate p-value (no scipy dependency)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    try:
        from scipy.stats import pearsonr as _pr
        r, p = _pr(x, y)
        return float(r), float(p)
    except ImportError:
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) >= 1.0:
            return r, 0.0
        t = r * np.sqrt((n - 2) / (1 - r * r + 1e-12))
        p = float(2 * np.exp(-0.717 * t * t - 0.416 * abs(t)))
        return r, max(0, min(1, p))


def strength_label(r: float) -> str:
    a = abs(r)
    if a >= 0.7: return "Strong"
    if a >= 0.5: return "Moderate"
    if a >= 0.3: return "Weak"
    return "Negligible"


# ═══════════════════════════════════════════════════════════════════════
# 1. Load data from overnight pipeline and detection quality
# ═══════════════════════════════════════════════════════════════════════
def load_overnight_summaries(overnight_dir: Path) -> dict:
    """Load all summary.json files from overnight pipeline output.
    Returns {video_name: {policy: summary_dict}}
    """
    results = {}
    if not overnight_dir.exists():
        print(f"  WARNING: Overnight dir not found: {overnight_dir}")
        return results
    for vid_dir in sorted(overnight_dir.iterdir()):
        if not vid_dir.is_dir() or vid_dir.name.startswith("grand"):
            continue
        vname = vid_dir.name
        results[vname] = {}
        for pol_dir in sorted(vid_dir.iterdir()):
            if not pol_dir.is_dir():
                continue
            summary_file = pol_dir / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        results[vname][pol_dir.name] = json.load(f)
                except Exception:
                    pass
    return results


def load_overnight_traces(overnight_dir: Path) -> dict:
    """Load per_frame_trace.csv files.
    Returns {video_name: {policy: list_of_row_dicts}}
    """
    results = {}
    if not overnight_dir.exists():
        return results
    for vid_dir in sorted(overnight_dir.iterdir()):
        if not vid_dir.is_dir() or vid_dir.name.startswith("grand"):
            continue
        vname = vid_dir.name
        results[vname] = {}
        for pol_dir in sorted(vid_dir.iterdir()):
            if not pol_dir.is_dir():
                continue
            csv_file = pol_dir / "per_frame_trace.csv"
            if csv_file.exists():
                try:
                    with open(csv_file) as f:
                        reader = csv.DictReader(f)
                        results[vname][pol_dir.name] = list(reader)
                except Exception:
                    pass
    return results


def load_detection_quality(det_dir: Path) -> dict:
    """Load detection quality results from 02_detection_quality."""
    agg_file = det_dir / "all_detection_quality.json"
    if agg_file.exists():
        with open(agg_file) as f:
            return json.load(f)
    # Try individual files
    results = {}
    if det_dir.exists():
        for f in det_dir.glob("*_quality.json"):
            with open(f) as fh:
                data = json.load(fh)
                # Merge into results
                for key, val in data.items():
                    results[key] = val
    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. Analysis: Detection Quality F1 Table
# ═══════════════════════════════════════════════════════════════════════
def analyze_detection_quality(det_data: dict) -> dict:
    """Compute precision, recall, F1 per policy across all videos.

    Uses s_only as oracle (ground truth proxy):
    - Recall = matched_with_oracle / total_s_oracle_dets
    - Precision = matched_with_oracle / total_policy_dets
    - F1 = 2 * P * R / (P + R)
    """
    # Aggregate per policy
    policy_agg = defaultdict(lambda: {
        "matched": 0, "oracle_total": 0, "policy_total": 0,
        "missed": 0, "extra": 0, "conf_w_num": 0.0, "conf_w_den": 0.0,
        "videos": 0,
    })

    for video_key, video_data in det_data.items():
        if not isinstance(video_data, dict):
            continue
        policies_data = video_data.get("policies", video_data)
        if not isinstance(policies_data, dict):
            continue
        for policy, pdata in policies_data.items():
            if not isinstance(pdata, dict):
                continue
            agg = policy_agg[policy]
            agg["matched"] += pdata.get("matched_with_oracle", 0)
            agg["oracle_total"] += pdata.get("total_s_oracle_dets", 0)
            agg["policy_total"] += pdata.get("total_policy_dets", 0)
            agg["missed"] += pdata.get("missed_by_policy", 0)
            agg["extra"] += pdata.get("extra_by_policy", 0)
            agg["conf_w_num"] += pdata.get("conf_weighted_coverage", 0) * pdata.get("total_s_oracle_dets", 1)
            agg["conf_w_den"] += pdata.get("total_s_oracle_dets", 1)
            agg["videos"] += 1

    results = {}
    for policy in POLICIES:
        agg = policy_agg.get(policy)
        if agg is None or agg["oracle_total"] == 0:
            continue
        recall = agg["matched"] / max(1, agg["oracle_total"])
        precision = agg["matched"] / max(1, agg["policy_total"])
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        conf_cov = agg["conf_w_num"] / max(1e-9, agg["conf_w_den"])
        results[policy] = {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "detection_coverage_pct": round(recall * 100, 1),
            "conf_weighted_coverage_pct": round(conf_cov * 100, 1),
            "missed_rate_pct": round((1 - recall) * 100, 1),
            "matched": agg["matched"],
            "oracle_total": agg["oracle_total"],
            "policy_total": agg["policy_total"],
            "n_videos": agg["videos"],
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# 3. Analysis: Proxy-Routing Correlation
# ═══════════════════════════════════════════════════════════════════════
def analyze_proxy_routing(traces: dict) -> dict:
    """For switching policies, correlate C-score with model choice.

    If the proxy is working correctly, C-score should be HIGH when s is chosen
    and LOW when n is chosen. We measure:
    - Mean C when n chosen vs mean C when s chosen
    - Point-biserial correlation between C and choice (0=n, 1=s)
    - Routing accuracy: % of s-frames where C > median(C)
    """
    results = {}

    for vname, vpolicies in traces.items():
        for policy, rows in vpolicies.items():
            if policy in ("n_only", "s_only"):
                continue
            if not rows:
                continue

            c_vals = []
            choices = []
            for r in rows:
                try:
                    c = float(r.get("C_score", 0))
                    ch = 1 if r.get("choice", "n") == "s" else 0
                    c_vals.append(c)
                    choices.append(ch)
                except (ValueError, TypeError):
                    continue

            if len(c_vals) < 10:
                continue

            c_arr = np.array(c_vals)
            ch_arr = np.array(choices)

            c_when_n = c_arr[ch_arr == 0]
            c_when_s = c_arr[ch_arr == 1]

            mean_c_n = float(np.mean(c_when_n)) if len(c_when_n) > 0 else 0
            mean_c_s = float(np.mean(c_when_s)) if len(c_when_s) > 0 else 0

            # Point-biserial correlation (same as Pearson for binary variable)
            r_pb, p_val = pearsonr(c_arr, ch_arr.astype(float))

            # Routing accuracy: when s chosen, was C above median?
            c_median = float(np.median(c_arr))
            if len(c_when_s) > 0:
                routing_acc = float(np.mean(c_when_s > c_median)) * 100
            else:
                routing_acc = 0.0

            s_usage = float(np.mean(ch_arr)) * 100

            key = f"{vname}__{policy}"
            results[key] = {
                "video": vname,
                "policy": policy,
                "mean_C_when_n": round(mean_c_n, 4),
                "mean_C_when_s": round(mean_c_s, 4),
                "c_separation": round(mean_c_s - mean_c_n, 4),
                "point_biserial_r": round(r_pb, 3),
                "p_value": round(p_val, 6),
                "strength": strength_label(r_pb),
                "routing_accuracy_pct": round(routing_acc, 1),
                "s_usage_pct": round(s_usage, 1),
                "n_frames": len(c_when_n),
                "s_frames": len(c_when_s),
            }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. Analysis: Per-Proxy Correlation with Disagreement Signal
# ═══════════════════════════════════════════════════════════════════════

# Policies that actually compute each proxy signal in the engine:
_PROXY_SOURCE_POLICIES = {
    "L": ["combined_hyst", "combined", "entropy_only", "multi_proxy"],
    "H": ["combined_hyst", "combined", "entropy_only", "multi_proxy"],
    "niqe_score": ["niqe_switch"],
    # mean_conf is computed unconditionally for ALL policies
    "mean_conf": None,  # None = use n_only (always valid)
}


def _find_proxy_source(vpolicies: dict, proxy_field: str) -> list:
    """Find the best available trace for a proxy field.

    Returns the trace rows from the first available policy that actually
    computes this signal, or empty list if none found.
    """
    source_list = _PROXY_SOURCE_POLICIES.get(proxy_field)
    if source_list is None:
        # mean_conf: n_only is valid (computed unconditionally)
        return vpolicies.get("n_only", [])
    for pol in source_list:
        rows = vpolicies.get(pol, [])
        if rows:
            return rows
    return []


def analyze_proxy_disagreement(traces: dict) -> dict:
    """Correlate individual proxies (L, H, mean_conf, NR-IQA) with
    the n-vs-s disagreement signal from the trace data.

    Disagreement = |num_detections_when_n - num_detections_when_s| per frame.
    For n_only and s_only runs, compute per-frame detection difference
    as the disagreement signal.

    IMPORTANT: L and H are ONLY populated in traces of scene-proxy policies
    (entropy_only, combined, combined_hyst, multi_proxy). NR-IQA scores
    are only populated for niqe_switch. This function sources proxy values
    from the correct policy traces rather than n_only (where they are zero).
    """
    results = {}

    for vname, vpolicies in traces.items():
        # We need both n_only and s_only traces for disagreement signal
        n_rows = vpolicies.get("n_only", [])
        s_rows = vpolicies.get("s_only", [])

        if not n_rows or not s_rows:
            continue

        n_frames = min(len(n_rows), len(s_rows))
        if n_frames < 50:
            continue

        # Source proxy values from the correct policies
        L_source = _find_proxy_source(vpolicies, "L")
        H_source = _find_proxy_source(vpolicies, "H")
        niqe_source = _find_proxy_source(vpolicies, "niqe_score")

        # Build arrays — align to the common frame count
        L_vals, H_vals, conf_vals, niqe_vals = [], [], [], []
        n_dets, s_dets = [], []
        skipped = 0

        for i in range(n_frames):
            try:
                # Detection counts: always from n_only / s_only
                n_dets.append(int(n_rows[i].get("num_detections", 0)))
                s_dets.append(int(s_rows[i].get("num_detections", 0)))

                # mean_conf: valid from n_only (computed unconditionally)
                conf_vals.append(float(n_rows[i].get("mean_conf", 0)))

                # L/H: from scene-proxy policy trace (if available)
                if i < len(L_source):
                    L_vals.append(float(L_source[i].get("L", 0)))
                else:
                    L_vals.append(0.0)
                if i < len(H_source):
                    H_vals.append(float(H_source[i].get("H", 0)))
                else:
                    H_vals.append(0.0)

                # NR-IQA: from niqe_switch trace (if available)
                if i < len(niqe_source):
                    niqe_vals.append(float(niqe_source[i].get("niqe_score", 0)))
                else:
                    niqe_vals.append(0.0)

            except (ValueError, TypeError):
                skipped += 1
                continue

        if len(n_dets) < 50:
            continue

        L_arr = np.array(L_vals)
        H_arr = np.array(H_vals)
        conf_arr = np.array(conf_vals)
        niqe_arr = np.array(niqe_vals)
        n_det_arr = np.array(n_dets, dtype=float)
        s_det_arr = np.array(s_dets, dtype=float)

        # Disagreement: frames where s finds more than n (policy should switch)
        disagree = s_det_arr - n_det_arr  # positive = s is better
        abs_disagree = np.abs(disagree)

        # Compute correlations — only include proxies with actual variance
        proxies = {}
        source_info = {}

        if np.std(L_arr) > 1e-6:
            proxies["Laplacian (L)"] = L_arr
            source_info["Laplacian (L)"] = L_source[0].get("policy", "unknown") if L_source else "n/a"
        if np.std(H_arr) > 1e-6:
            proxies["Entropy (H)"] = H_arr
            source_info["Entropy (H)"] = H_source[0].get("policy", "unknown") if H_source else "n/a"

        # mean_conf always has variance for n_only
        if np.std(conf_arr) > 1e-6:
            proxies["Mean Conf (n)"] = conf_arr
            source_info["Mean Conf (n)"] = "n_only"

        # NR-IQA only if niqe_switch trace exists and has variance
        if np.std(niqe_arr) > 1e-6:
            proxies["NR-IQA"] = niqe_arr
            source_info["NR-IQA"] = "niqe_switch"

        video_results = {}
        for proxy_name, proxy_arr in proxies.items():
            # Correlation with disagreement magnitude
            r_mag, p_mag = pearsonr(proxy_arr, abs_disagree)
            # Correlation with signed disagreement (s - n)
            r_signed, p_signed = pearsonr(proxy_arr, disagree)

            video_results[proxy_name] = {
                "r_with_abs_disagreement": round(r_mag, 3),
                "p_abs": round(p_mag, 6),
                "r_with_signed_disagreement": round(r_signed, 3),
                "p_signed": round(p_signed, 6),
                "strength": strength_label(r_mag),
                "proxy_mean": round(float(np.mean(proxy_arr)), 4),
                "proxy_std": round(float(np.std(proxy_arr)), 4),
                "proxy_cov": round(float(np.std(proxy_arr) / (np.mean(proxy_arr) + 1e-9)), 4),
                "source_policy": source_info.get(proxy_name, "unknown"),
            }

        results[vname] = {
            "n_frames": len(n_dets),
            "mean_disagreement": round(float(np.mean(abs_disagree)), 2),
            "frames_s_better": int(np.sum(disagree > 0)),
            "frames_n_better": int(np.sum(disagree < 0)),
            "frames_equal": int(np.sum(disagree == 0)),
            "proxy_data_notes": (
                "L/H sourced from scene-proxy policy traces; "
                "NR-IQA from niqe_switch; mean_conf from n_only. "
                "Proxies with zero variance excluded."
            ),
            "proxies": video_results,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. Analysis: Cross-Video Consistency
# ═══════════════════════════════════════════════════════════════════════
def analyze_consistency(summaries: dict) -> dict:
    """Compute coefficient of variation for key metrics across videos."""
    # Collect per-policy metrics across videos
    policy_data = defaultdict(lambda: {
        "T_total": [], "slow_pct": [], "sw_per_100": [],
    })

    for vname, vpolicies in summaries.items():
        for policy, s in vpolicies.items():
            policy_data[policy]["T_total"].append(s.get("T_total_ms_mean", 0))
            policy_data[policy]["slow_pct"].append(s.get("slow_pct", 0))
            policy_data[policy]["sw_per_100"].append(s.get("sw_per_100", 0))

    results = {}
    for policy in POLICIES:
        pd = policy_data.get(policy)
        if not pd or len(pd["T_total"]) < 2:
            continue

        r = {}
        for metric in ["T_total", "slow_pct", "sw_per_100"]:
            vals = np.array(pd[metric])
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            cv = abs(std / (mean + 1e-9)) * 100
            r[metric] = {
                "mean": round(mean, 2),
                "std": round(std, 2),
                "cv_pct": round(cv, 1),
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2),
                "n_videos": len(vals),
            }
        results[policy] = r

    return results


# ═══════════════════════════════════════════════════════════════════════
# 6. Plotting
# ═══════════════════════════════════════════════════════════════════════
def generate_plots(f1_data: dict, routing_data: dict, proxy_disagree: dict,
                   consistency: dict, summaries: dict, out_dir: Path):
    """Generate all validation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: F1 / Precision / Recall bar chart ──
    if f1_data:
        policies = [p for p in POLICIES if p in f1_data]
        if policies:
            fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
            fig.patch.set_facecolor("#0A0E1A")
            ax.set_facecolor("#0A0E1A")

            x = np.arange(len(policies))
            w = 0.25
            prec = [f1_data[p]["precision"] for p in policies]
            rec = [f1_data[p]["recall"] for p in policies]
            f1 = [f1_data[p]["f1"] for p in policies]

            ax.bar(x - w, prec, w, label="Precision", color="#42A5F5")
            ax.bar(x, rec, w, label="Recall", color="#66BB6A")
            ax.bar(x + w, f1, w, label="F1", color="#FFA726")

            for i in range(len(policies)):
                ax.text(i + w, f1[i] + 0.5, f"{f1[i]:.1f}", ha="center",
                        color="white", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([POLICY_SHORT.get(p, p) for p in policies],
                               color="white", fontsize=9, rotation=30, ha="right")
            ax.set_ylabel("Score (%)", color="white")
            ax.set_title("Detection Quality: Precision / Recall / F1 (s_only as oracle)",
                         color="white", fontweight="bold")
            ax.set_ylim(0, 105)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#333")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
            fig.tight_layout()
            fig.savefig(str(out_dir / "detection_f1_scores.png"), dpi=150,
                        facecolor="#0A0E1A", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: detection_f1_scores.png")

    # ── Plot 2: Proxy-Routing Correlation heatmap ──
    if routing_data:
        # Aggregate per policy
        pol_routing = defaultdict(list)
        for key, rd in routing_data.items():
            pol_routing[rd["policy"]].append(rd["point_biserial_r"])

        policies = [p for p in POLICIES if p in pol_routing and p not in ("n_only", "s_only")]
        if policies:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
            fig.patch.set_facecolor("#0A0E1A")

            # (a) Point-biserial r per policy (mean +/- std across videos)
            ax = axes[0]
            ax.set_facecolor("#0A0E1A")
            x = np.arange(len(policies))
            means = [np.mean(pol_routing[p]) for p in policies]
            stds = [np.std(pol_routing[p]) for p in policies]
            colors = [POLICY_COLORS.get(p, "#888") for p in policies]
            ax.bar(x, means, 0.6, yerr=stds, color=colors, capsize=3,
                   error_kw={"ecolor": "white", "alpha": 0.5})
            for i, m in enumerate(means):
                ax.text(i, m + stds[i] + 0.02, f"{m:.2f}", ha="center",
                        color="white", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([POLICY_SHORT.get(p, p) for p in policies],
                               color="white", fontsize=9, rotation=30, ha="right")
            ax.set_ylabel("Point-Biserial r", color="white")
            ax.set_title("(a) C-score vs Model Choice Correlation",
                         color="white", fontweight="bold")
            ax.axhline(0, color="white", alpha=0.2)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#333")

            # (b) C separation (mean_C_s - mean_C_n)
            ax = axes[1]
            ax.set_facecolor("#0A0E1A")
            pol_sep = defaultdict(list)
            for key, rd in routing_data.items():
                pol_sep[rd["policy"]].append(rd["c_separation"])
            sep_means = [np.mean(pol_sep[p]) for p in policies]
            sep_stds = [np.std(pol_sep[p]) for p in policies]
            ax.bar(x, sep_means, 0.6, yerr=sep_stds, color=colors, capsize=3,
                   error_kw={"ecolor": "white", "alpha": 0.5})
            for i, m in enumerate(sep_means):
                ax.text(i, m + sep_stds[i] + 0.002, f"{m:.3f}", ha="center",
                        color="white", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([POLICY_SHORT.get(p, p) for p in policies],
                               color="white", fontsize=9, rotation=30, ha="right")
            ax.set_ylabel("Mean C(s) - Mean C(n)", color="white")
            ax.set_title("(b) C-score Separation by Model Choice",
                         color="white", fontweight="bold")
            ax.axhline(0, color="white", alpha=0.2)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#333")

            fig.suptitle("Proxy-Routing Validation", color="white",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(str(out_dir / "proxy_routing_correlation.png"), dpi=150,
                        facecolor="#0A0E1A", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: proxy_routing_correlation.png")

    # ── Plot 3: Proxy-Disagreement Correlation ──
    if proxy_disagree:
        proxy_names = set()
        for vname, vdata in proxy_disagree.items():
            proxy_names.update(vdata.get("proxies", {}).keys())
        proxy_names = sorted(proxy_names)

        if proxy_names:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            fig.patch.set_facecolor("#0A0E1A")
            ax.set_facecolor("#0A0E1A")

            x = np.arange(len(proxy_names))
            proxy_colors = ["#42A5F5", "#66BB6A", "#FFA726", "#EC407A", "#AB47BC"]

            all_rs = {pn: [] for pn in proxy_names}
            for vname, vdata in proxy_disagree.items():
                for pn in proxy_names:
                    pinfo = vdata.get("proxies", {}).get(pn)
                    if pinfo:
                        all_rs[pn].append(abs(pinfo["r_with_abs_disagreement"]))

            means = [np.mean(all_rs[pn]) if all_rs[pn] else 0 for pn in proxy_names]
            stds = [np.std(all_rs[pn]) if len(all_rs[pn]) > 1 else 0 for pn in proxy_names]

            bars = ax.bar(x, means, 0.6, yerr=stds, capsize=4,
                          color=proxy_colors[:len(proxy_names)],
                          error_kw={"ecolor": "white", "alpha": 0.5})
            for i, m in enumerate(means):
                ax.text(i, m + stds[i] + 0.01, f"|r|={m:.2f}", ha="center",
                        color="white", fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(proxy_names, color="white", fontsize=9,
                               rotation=25, ha="right")
            ax.set_ylabel("|Pearson r| with Disagreement", color="white")
            ax.set_title("Proxy Correlation with Model Disagreement (n vs s detections)",
                         color="white", fontweight="bold")
            ax.set_ylim(0, max(means) * 1.3 + 0.05 if means else 1)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#333")

            # Significance threshold line
            ax.axhline(0.3, color="#FF6B6B", linestyle="--", alpha=0.4,
                       label="|r|=0.3 (weak threshold)")
            ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

            fig.tight_layout()
            fig.savefig(str(out_dir / "proxy_disagreement_correlation.png"), dpi=150,
                        facecolor="#0A0E1A", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: proxy_disagreement_correlation.png")

    # ── Plot 4: Cross-Video Consistency (CoV) ──
    if consistency:
        policies = [p for p in POLICIES if p in consistency]
        if policies:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
            fig.patch.set_facecolor("#0A0E1A")

            metrics = [("T_total", "T_total CoV (%)", "Latency Consistency"),
                       ("slow_pct", "s-usage CoV (%)", "Model Usage Consistency"),
                       ("sw_per_100", "Switching CoV (%)", "Switching Consistency")]

            for ax, (metric, ylabel, title) in zip(axes, metrics):
                ax.set_facecolor("#0A0E1A")
                x = np.arange(len(policies))
                vals = []
                for p in policies:
                    cv = consistency[p].get(metric, {}).get("cv_pct", 0)
                    vals.append(cv)
                colors = [POLICY_COLORS.get(p, "#888") for p in policies]
                ax.bar(x, vals, 0.6, color=colors)
                for i, v in enumerate(vals):
                    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center",
                            color="white", fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels([POLICY_SHORT.get(p, p) for p in policies],
                                   color="white", fontsize=8, rotation=35, ha="right")
                ax.set_ylabel(ylabel, color="white")
                ax.set_title(title, color="white", fontsize=10, fontweight="bold")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_color("#333")

            fig.suptitle("Cross-Video Consistency (lower CoV = more consistent)",
                         color="white", fontsize=12, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(str(out_dir / "cross_video_consistency.png"), dpi=150,
                        facecolor="#0A0E1A", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: cross_video_consistency.png")

    # ── Plot 5: Grand Validation Summary (2x2) ──
    if f1_data and consistency and summaries:
        policies = [p for p in POLICIES if p in f1_data and p in consistency]
        if len(policies) >= 3:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
            fig.patch.set_facecolor("#0A0E1A")

            for ax in axes.flat:
                ax.set_facecolor("#0A0E1A")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_color("#333")

            x = np.arange(len(policies))
            labels = [POLICY_SHORT.get(p, p) for p in policies]
            colors = [POLICY_COLORS.get(p, "#888") for p in policies]

            # (a) F1 scores
            ax = axes[0, 0]
            f1_vals = [f1_data[p]["f1"] for p in policies]
            ax.bar(x, f1_vals, 0.6, color=colors)
            for i, v in enumerate(f1_vals):
                ax.text(i, v + 0.5, f"{v:.1f}", ha="center", color="white", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, color="white", fontsize=8, rotation=30, ha="right")
            ax.set_ylabel("F1 (%)", color="white")
            ax.set_title("(a) Detection F1 Score", color="white", fontweight="bold")
            ax.set_ylim(0, 105)

            # (b) Latency
            ax = axes[0, 1]
            t_vals = []
            for p in policies:
                t_list = [s.get("T_total_ms_mean", 0) for v in summaries.values()
                          for pp, s in v.items() if pp == p]
                t_vals.append(np.mean(t_list) if t_list else 0)
            ax.bar(x, t_vals, 0.6, color=colors)
            for i, v in enumerate(t_vals):
                ax.text(i, v + 3, f"{v:.0f}", ha="center", color="white", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, color="white", fontsize=8, rotation=30, ha="right")
            ax.set_ylabel("Mean T_total (ms)", color="white")
            ax.set_title("(b) Mean Latency", color="white", fontweight="bold")

            # (c) Latency CoV (consistency)
            ax = axes[1, 0]
            cv_vals = [consistency[p]["T_total"]["cv_pct"] for p in policies]
            ax.bar(x, cv_vals, 0.6, color=colors)
            for i, v in enumerate(cv_vals):
                ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", color="white", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, color="white", fontsize=8, rotation=30, ha="right")
            ax.set_ylabel("T_total CoV (%)", color="white")
            ax.set_title("(c) Latency Consistency (lower = better)",
                         color="white", fontweight="bold")

            # (d) Pareto: F1 vs Latency
            ax = axes[1, 1]
            for i, p in enumerate(policies):
                ax.scatter(t_vals[i], f1_vals[i], c=colors[i], s=100, zorder=5,
                           edgecolors="white", linewidth=0.5)
                ax.annotate(labels[i], (t_vals[i], f1_vals[i]),
                            textcoords="offset points", xytext=(6, 4),
                            color="white", fontsize=8)
            ax.set_xlabel("Mean T_total (ms)", color="white")
            ax.set_ylabel("F1 (%)", color="white")
            ax.set_title("(d) F1 vs Latency Trade-off", color="white", fontweight="bold")

            fig.suptitle("Grand Validation Summary", color="white",
                         fontsize=14, fontweight="bold")
            fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3, bottom=0.1)
            fig.savefig(str(out_dir / "grand_validation_summary.png"), dpi=150,
                        facecolor="#0A0E1A", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: grand_validation_summary.png")


# ═══════════════════════════════════════════════════════════════════════
# 7. LaTeX table generation
# ═══════════════════════════════════════════════════════════════════════
def generate_latex_table(f1_data: dict, consistency: dict,
                         summaries: dict, out_path: Path):
    """Generate a LaTeX-ready validation summary table."""
    policies = [p for p in POLICIES if p in f1_data]
    if not policies:
        return

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comprehensive Validation Summary (oracle-relative metrics, s\_only as reference)}")
    lines.append(r"\label{tab:validation}")
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Policy & Prec.$^\dagger$ & Rec.$^\dagger$ & F1$^\dagger$ & $\bar{T}$ (ms) & CoV$_T$ (\%) & sw/100 \\")
    lines.append(r"\midrule")

    for p in policies:
        f = f1_data[p]
        c = consistency.get(p, {})
        t_cv = c.get("T_total", {}).get("cv_pct", 0)
        # Get mean latency and sw/100
        t_vals, sw_vals = [], []
        for v in summaries.values():
            if p in v:
                t_vals.append(v[p].get("T_total_ms_mean", 0))
                sw_vals.append(v[p].get("sw_per_100", 0))
        t_mean = np.mean(t_vals) if t_vals else 0
        sw_mean = np.mean(sw_vals) if sw_vals else 0
        short = POLICY_SHORT.get(p, p).replace("_", r"\_")

        lines.append(
            f"  {short} & {f['precision']:.1f} & {f['recall']:.1f} & "
            f"{f['f1']:.1f} & {t_mean:.0f} & {t_cv:.1f} & {sw_mean:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\footnotesize{$^\dagger$ Oracle-relative: s\_only used as pseudo ground truth.}")
    lines.append(r"\end{table}")

    with open(out_path, "w") as fout:
        fout.write("\n".join(lines))
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Comprehensive DMS-Raptor Validation Analysis")
    parser.add_argument("--overnight-dir", type=str, default=None)
    parser.add_argument("--detection-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    overnight_dir = Path(args.overnight_dir) if args.overnight_dir else OVERNIGHT_DIR
    det_dir = Path(args.detection_dir) if args.detection_dir else DETECTION_DIR
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DMS-Raptor: Comprehensive Validation Analysis")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/7] Loading overnight pipeline summaries...")
    summaries = load_overnight_summaries(overnight_dir)
    n_vids = len(summaries)
    n_pols = len(set(p for v in summaries.values() for p in v))
    print(f"  Found: {n_vids} videos, {n_pols} policies")

    print("\n[2/7] Loading overnight pipeline traces...")
    traces = load_overnight_traces(overnight_dir)
    n_traces = sum(len(v) for v in traces.values())
    print(f"  Found: {n_traces} trace files")

    print("\n[3/7] Loading detection quality results...")
    det_data = load_detection_quality(det_dir)
    print(f"  Found: {len(det_data)} video entries")

    # ── Run analyses ──
    print("\n[4/7] Computing Oracle-Relative Detection Quality (s_only as reference)...")
    f1_data = analyze_detection_quality(det_data)
    if f1_data:
        for p in POLICIES:
            if p in f1_data:
                d = f1_data[p]
                print(f"  {POLICY_SHORT.get(p, p):>10s}: P={d['precision']:.1f}% "
                      f"R={d['recall']:.1f}% F1={d['f1']:.1f}%")
    else:
        print("  No detection quality data available. Run 02_detection_quality.py first.")

    print("\n[5/7] Computing Proxy-Routing Correlation...")
    routing_data = analyze_proxy_routing(traces)
    if routing_data:
        # Aggregate per policy
        pol_agg = defaultdict(list)
        for key, rd in routing_data.items():
            pol_agg[rd["policy"]].append(rd["point_biserial_r"])
        for p in POLICIES:
            if p in pol_agg:
                rs = pol_agg[p]
                print(f"  {POLICY_SHORT.get(p, p):>10s}: r={np.mean(rs):.3f} +/- {np.std(rs):.3f} "
                      f"({strength_label(np.mean(rs))})")
    else:
        print("  No trace data available for routing analysis.")

    print("\n[6/7] Computing Proxy-Disagreement Correlation...")
    proxy_disagree = analyze_proxy_disagreement(traces)
    if proxy_disagree:
        # Aggregate across videos
        all_proxies = defaultdict(list)
        for vname, vdata in proxy_disagree.items():
            for pn, pinfo in vdata.get("proxies", {}).items():
                all_proxies[pn].append(abs(pinfo["r_with_abs_disagreement"]))
        for pn, rs in sorted(all_proxies.items(), key=lambda x: -np.mean(x[1])):
            print(f"  {pn:>18s}: |r|={np.mean(rs):.3f} +/- {np.std(rs):.3f} "
                  f"({strength_label(np.mean(rs))})")
    else:
        print("  No n_only/s_only trace pairs available for disagreement analysis.")

    print("\n[7/7] Computing Cross-Video Consistency...")
    consistency = analyze_consistency(summaries)
    if consistency:
        for p in POLICIES:
            if p in consistency:
                tc = consistency[p]["T_total"]
                print(f"  {POLICY_SHORT.get(p, p):>10s}: T_mean={tc['mean']:.0f}ms "
                      f"CoV={tc['cv_pct']:.1f}%")
    else:
        print("  No summary data available for consistency analysis.")

    # ── Save results ──
    print(f"\n{'=' * 70}")
    print("  Generating outputs...")
    print(f"{'=' * 70}")

    # JSON results
    all_results = {
        "detection_quality_f1": f1_data,
        "proxy_routing_correlation": routing_data,
        "proxy_disagreement_correlation": proxy_disagree,
        "cross_video_consistency": consistency,
    }
    with open(out_dir / "validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: validation_results.json")

    # Plots
    generate_plots(f1_data, routing_data, proxy_disagree, consistency,
                   summaries, out_dir)

    # LaTeX table
    if f1_data and consistency:
        generate_latex_table(f1_data, consistency, summaries,
                             out_dir / "validation_table.tex")

    # Summary CSV
    if f1_data:
        csv_path = out_dir / "validation_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["policy", "oracle_rel_precision", "oracle_rel_recall",
                             "oracle_rel_f1",
                             "detection_coverage", "conf_weighted_coverage",
                             "T_total_mean_ms", "T_total_cv_pct", "sw_per_100"])
            for p in POLICIES:
                if p not in f1_data:
                    continue
                fd = f1_data[p]
                tc = consistency.get(p, {}).get("T_total", {})
                t_vals = [s.get("T_total_ms_mean", 0) for v in summaries.values()
                          for pp, s in v.items() if pp == p]
                sw_vals = [s.get("sw_per_100", 0) for v in summaries.values()
                           for pp, s in v.items() if pp == p]
                writer.writerow([
                    p, fd["precision"], fd["recall"], fd["f1"],
                    fd["detection_coverage_pct"], fd["conf_weighted_coverage_pct"],
                    f"{np.mean(t_vals):.1f}" if t_vals else "",
                    f"{tc.get('cv_pct', 0):.1f}",
                    f"{np.mean(sw_vals):.1f}" if sw_vals else "",
                ])
        print(f"  Saved: validation_summary.csv")

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION COMPLETE")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
