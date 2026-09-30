"""
Validation report generator for DMS-Raptor proxy validation.

Computes correlations, generates plots, and exports CSV data
from dual-model validation results.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .validator import ValidationSummary


# ======================================================================
# Correlation result container
# ======================================================================
class CorrelationResult:
    """Holds correlation results between two variables."""

    def __init__(self, name_x: str, name_y: str,
                 pearson_r: float, pearson_p: float,
                 spearman_rho: float, spearman_p: float,
                 n: int):
        self.name_x = name_x
        self.name_y = name_y
        self.pearson_r = pearson_r
        self.pearson_p = pearson_p
        self.spearman_rho = spearman_rho
        self.spearman_p = spearman_p
        self.n = n

    @property
    def strength(self) -> str:
        r = abs(self.pearson_r)
        if r >= 0.7:
            return "Strong"
        elif r >= 0.5:
            return "Moderate"
        elif r >= 0.3:
            return "Weak"
        else:
            return "Negligible"

    def __repr__(self):
        return (
            f"{self.name_x} vs {self.name_y}: "
            f"Pearson r={self.pearson_r:.4f} (p={self.pearson_p:.2e}), "
            f"Spearman ρ={self.spearman_rho:.4f} (p={self.spearman_p:.2e}) "
            f"[{self.strength}]"
        )


# ======================================================================
# Sensitivity result
# ======================================================================
class SensitivityResult:
    """Compares switching decisions across proxy settings."""

    def __init__(self):
        self.setting_pairs: List[Tuple[str, str]] = []
        self.agreement_pcts: List[float] = []
        self.details: Dict[str, Dict] = {}

    def add(self, setting_a: str, setting_b: str, agreement_pct: float,
            n_frames: int):
        self.setting_pairs.append((setting_a, setting_b))
        self.agreement_pcts.append(agreement_pct)
        self.details[f"{setting_a}_vs_{setting_b}"] = {
            "agreement_pct": agreement_pct,
            "n_frames": n_frames,
        }


# ======================================================================
# Validation Report
# ======================================================================
class ValidationReport:
    """
    Generates comprehensive analysis from a ValidationSummary.

    Usage::

        report = ValidationReport(summary)
        correlations = report.compute_correlations()
        sensitivity = report.compute_sensitivity(threshold=0.5)
        report.export_csv(path)
    """

    def __init__(self, summary: ValidationSummary):
        self.summary = summary

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------
    def compute_correlations(self) -> List[CorrelationResult]:
        """
        Compute Pearson and Spearman correlations between
        scene proxies (L, H, C) and disagreement metrics.
        """
        s = self.summary
        if s.total_frames < 10:
            return []

        proxy_vars = {
            "C-score": np.array(s.C_trace, dtype=float),
            "Laplacian (L)": np.array(s.L_trace, dtype=float),
            "Entropy (H)": np.array(s.H_trace, dtype=float),
        }
        # Add extended proxies if data is available
        if hasattr(s, 'n_mean_confs') and s.n_mean_confs:
            proxy_vars["Mean Conf (n)"] = np.array(s.n_mean_confs, dtype=float)
        if hasattr(s, 'niqe_scores') and s.niqe_scores:
            proxy_vars["NIQE Score"] = np.array(s.niqe_scores, dtype=float)
        # New extended proxies (from ablation study)
        _ext = [
            ("Tenengrad", "tenengrad_trace"),
            ("Edge Density", "edge_density_trace"),
            ("Local Contrast", "local_contrast_trace"),
            ("Brenner", "brenner_trace"),
            ("Color Entropy", "color_entropy_trace"),
        ]
        for name, attr in _ext:
            data = getattr(s, attr, [])
            if data and len(data) >= 10:
                proxy_vars[name] = np.array(data, dtype=float)

        disagree_vars = {
            "Det Count Gap": np.array(s.det_count_gaps, dtype=float),
            "Mean Conf Gap": np.array(s.mean_conf_gaps, dtype=float),
            "Max Conf Gap": np.array(s.max_conf_gaps, dtype=float),
            "IoU Agreement": np.array(s.iou_agreements, dtype=float),
            "Extra Dets (s)": np.array(s.extra_dets_s_trace, dtype=float),
            "Disagreement Score": np.array(s.disagreement_scores, dtype=float),
        }

        results: List[CorrelationResult] = []
        for px_name, px_data in proxy_vars.items():
            for dg_name, dg_data in disagree_vars.items():
                n = min(len(px_data), len(dg_data))
                if n < 10:
                    continue
                x, y = px_data[:n], dg_data[:n]

                # Skip if either is constant
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    continue

                pr, pp = stats.pearsonr(x, y)
                sr, sp = stats.spearmanr(x, y)
                results.append(CorrelationResult(
                    name_x=px_name, name_y=dg_name,
                    pearson_r=float(pr), pearson_p=float(pp),
                    spearman_rho=float(sr), spearman_p=float(sp),
                    n=n,
                ))

        return results

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------
    def compute_sensitivity(self, threshold: float = 0.5) -> SensitivityResult:
        """
        Compare switching decisions across proxy settings.

        For each frame and each setting, determine if C > threshold → "s" else → "n".
        Then compare agreement rates between all setting pairs.
        """
        s = self.summary
        result = SensitivityResult()

        if not s.proxy_settings or s.total_frames < 1:
            return result

        # Compute decisions per setting
        decisions: Dict[str, List[str]] = {}
        for key in s.proxy_settings:
            c_trace = s.C_traces.get(key, [])
            # Normalize C-trace to [0,1] using min-max
            arr = np.array(c_trace, dtype=float)
            lo, hi = arr.min(), arr.max()
            if hi - lo > 1e-9:
                arr_norm = (arr - lo) / (hi - lo)
            else:
                arr_norm = np.zeros_like(arr)
            decisions[key] = ["s" if v >= threshold else "n" for v in arr_norm]

        # Compare all pairs
        keys = list(s.proxy_settings)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k_a, k_b = keys[i], keys[j]
                dec_a = decisions[k_a]
                dec_b = decisions[k_b]
                n = min(len(dec_a), len(dec_b))
                if n == 0:
                    continue
                agree = sum(
                    1 for a, b in zip(dec_a[:n], dec_b[:n]) if a == b
                )
                pct = 100.0 * agree / n
                result.add(k_a, k_b, pct, n)

        return result

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def export_csv(self, path: str) -> None:
        """Export full per-frame validation data to CSV."""
        s = self.summary
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            # Header
            header = [
                "frame_idx", "L", "H", "C",
                "tenengrad", "edge_density", "local_contrast", "brenner", "color_entropy",
                "niqe",
                "n_det_count", "n_mean_conf", "n_max_conf", "n_latency_ms",
                "s_det_count", "s_mean_conf", "s_max_conf", "s_latency_ms",
                "det_count_gap", "mean_conf_gap", "max_conf_gap",
                "iou_agreement", "extra_dets_s", "extra_dets_n",
                "disagreement_score",
            ]
            # Add multi-setting columns
            for key in s.proxy_settings:
                header.extend([f"L_{key}", f"H_{key}", f"C_{key}"])
            w.writerow(header)

            # Data rows
            for i in range(s.total_frames):
                def _g(trace, idx, fmt=".4f"):
                    return f"{trace[idx]:{fmt}}" if idx < len(trace) else ""

                row = [
                    s.frame_indices[i] if i < len(s.frame_indices) else i,
                    _g(s.L_trace, i), _g(s.H_trace, i), _g(s.C_trace, i, ".6f"),
                    _g(getattr(s, 'tenengrad_trace', []), i),
                    _g(getattr(s, 'edge_density_trace', []), i),
                    _g(getattr(s, 'local_contrast_trace', []), i),
                    _g(getattr(s, 'brenner_trace', []), i),
                    _g(getattr(s, 'color_entropy_trace', []), i),
                    _g(getattr(s, 'niqe_scores', []), i),
                    s.n_det_counts[i] if i < len(s.n_det_counts) else "",
                    f"{s.n_mean_confs[i]:.4f}" if i < len(s.n_mean_confs) else "",
                    f"{s.n_max_confs[i]:.4f}" if i < len(s.n_max_confs) else "",
                    f"{s.n_latencies[i]:.2f}" if i < len(s.n_latencies) else "",
                    s.s_det_counts[i] if i < len(s.s_det_counts) else "",
                    f"{s.s_mean_confs[i]:.4f}" if i < len(s.s_mean_confs) else "",
                    f"{s.s_max_confs[i]:.4f}" if i < len(s.s_max_confs) else "",
                    f"{s.s_latencies[i]:.2f}" if i < len(s.s_latencies) else "",
                    s.det_count_gaps[i] if i < len(s.det_count_gaps) else "",
                    f"{s.mean_conf_gaps[i]:.4f}" if i < len(s.mean_conf_gaps) else "",
                    f"{s.max_conf_gaps[i]:.4f}" if i < len(s.max_conf_gaps) else "",
                    f"{s.iou_agreements[i]:.4f}" if i < len(s.iou_agreements) else "",
                    s.extra_dets_s_trace[i] if i < len(s.extra_dets_s_trace) else "",
                    s.extra_dets_n_trace[i] if i < len(s.extra_dets_n_trace) else "",
                    f"{s.disagreement_scores[i]:.4f}" if i < len(s.disagreement_scores) else "",
                ]
                # Multi-setting data
                for key in s.proxy_settings:
                    lt = s.L_traces.get(key, [])
                    ht = s.H_traces.get(key, [])
                    ct = s.C_traces.get(key, [])
                    row.append(f"{lt[i]:.4f}" if i < len(lt) else "")
                    row.append(f"{ht[i]:.4f}" if i < len(ht) else "")
                    row.append(f"{ct[i]:.6f}" if i < len(ct) else "")
                w.writerow(row)

    # ------------------------------------------------------------------
    # Correlation export
    # ------------------------------------------------------------------
    def export_correlations_csv(self, path: str) -> None:
        """Export correlation table to CSV."""
        correlations = self.compute_correlations()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Proxy", "Disagreement Metric",
                "Pearson r", "Pearson p-value",
                "Spearman rho", "Spearman p-value",
                "Strength", "N",
            ])
            for c in correlations:
                w.writerow([
                    c.name_x, c.name_y,
                    f"{c.pearson_r:.6f}", f"{c.pearson_p:.2e}",
                    f"{c.spearman_rho:.6f}", f"{c.spearman_p:.2e}",
                    c.strength, c.n,
                ])

    # ------------------------------------------------------------------
    # Plot generation (Matplotlib, for GUI embedding and export)
    # ------------------------------------------------------------------
    def plot_correlation_scatter(self, ax, proxy: str = "C-score",
                                 metric: str = "Disagreement Score"):
        """Plot scatter + regression line for one proxy vs one metric."""
        s = self.summary
        proxy_map = {
            "C-score": s.C_trace,
            "Laplacian (L)": s.L_trace,
            "Entropy (H)": s.H_trace,
            "Mean Conf (n)": getattr(s, 'n_mean_confs', []),
            "NIQE Score": getattr(s, 'niqe_scores', []),
            "Tenengrad": getattr(s, 'tenengrad_trace', []),
            "Edge Density": getattr(s, 'edge_density_trace', []),
            "Local Contrast": getattr(s, 'local_contrast_trace', []),
            "Brenner": getattr(s, 'brenner_trace', []),
            "Color Entropy": getattr(s, 'color_entropy_trace', []),
        }
        metric_map = {
            "Det Count Gap": s.det_count_gaps,
            "Mean Conf Gap": s.mean_conf_gaps,
            "Max Conf Gap": s.max_conf_gaps,
            "IoU Agreement": s.iou_agreements,
            "Extra Dets (s)": s.extra_dets_s_trace,
            "Disagreement Score": s.disagreement_scores,
        }

        x = np.array(proxy_map.get(proxy, s.C_trace), dtype=float)
        y = np.array(metric_map.get(metric, s.disagreement_scores), dtype=float)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        ax.scatter(x, y, s=6, alpha=0.4, color="#00bcd4", edgecolors="none")

        # Regression line
        if n > 2 and np.std(x) > 1e-12:
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color="#e94560", linewidth=2,
                    label=f"r={r_val:.3f}, p={p_val:.2e}")
            ax.legend(loc="best", fontsize=8, facecolor="#16213e",
                      edgecolor="#2a3a5c", labelcolor="#c0c0c0")

        ax.set_xlabel(proxy)
        ax.set_ylabel(metric)
        ax.set_title(f"{proxy} vs {metric}")

    def plot_timeseries_overlay(self, ax):
        """Plot C-score and disagreement score as overlaid time series."""
        s = self.summary
        if not s.frame_indices:
            return

        frames = np.array(s.frame_indices)
        c_vals = np.array(s.C_trace, dtype=float)
        d_vals = np.array(s.disagreement_scores, dtype=float)

        # Normalize C to [0,1] for visual comparison
        c_min, c_max = c_vals.min(), c_vals.max()
        if c_max - c_min > 1e-9:
            c_norm = (c_vals - c_min) / (c_max - c_min)
        else:
            c_norm = np.zeros_like(c_vals)

        ax.plot(frames, c_norm, color="#00bcd4", linewidth=1.2,
                alpha=0.8, label="C-score (normalized)")
        ax.plot(frames, d_vals, color="#e94560", linewidth=1.2,
                alpha=0.8, label="Disagreement Score")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Score (0-1)")
        ax.set_title("C-Score vs Model Disagreement Over Time")
        ax.legend(loc="best", fontsize=8, facecolor="#16213e",
                  edgecolor="#2a3a5c", labelcolor="#c0c0c0")

    def plot_sensitivity_heatmap(self, ax):
        """Plot sensitivity analysis as agreement % heatmap."""
        sensitivity = self.compute_sensitivity()
        if not sensitivity.setting_pairs:
            ax.set_title("No sensitivity data")
            return

        keys = list(self.summary.proxy_settings)
        n = len(keys)
        matrix = np.full((n, n), 100.0)  # diagonal = 100%

        for (ka, kb), pct in zip(sensitivity.setting_pairs, sensitivity.agreement_pcts):
            i = keys.index(ka)
            j = keys.index(kb)
            matrix[i, j] = pct
            matrix[j, i] = pct

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        labels = [k.replace("_", "\n") for k in keys]
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=7)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.0f}%",
                        ha="center", va="center",
                        color="black", fontsize=8, fontweight="bold")

        ax.set_title("Sensitivity: Decision Agreement %")
        # Return im for optional colorbar
        return im

    def plot_detection_comparison(self, ax):
        """Bar chart comparing n vs s detection counts."""
        s = self.summary
        if not s.n_det_counts:
            return

        mean_n = np.mean(s.n_det_counts)
        mean_s = np.mean(s.s_det_counts)
        std_n = np.std(s.n_det_counts)
        std_s = np.std(s.s_det_counts)

        x = [0, 1]
        heights = [mean_n, mean_s]
        errs = [std_n, std_s]
        colors = ["#4caf50", "#e94560"]
        labels = ["YOLOv8n", "YOLOv8s"]

        bars = ax.bar(x, heights, yerr=errs, color=colors,
                       width=0.5, capsize=10, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean Detections / Frame")
        ax.set_title("Detection Count: n-model vs s-model")

        for xi, h in zip(x, heights):
            ax.text(xi, h + 0.5, f"{h:.1f}", ha="center", va="bottom",
                    color="#c0c0c0", fontsize=10)

    def plot_correlation_matrix(self, ax, fig):
        """Full correlation heatmap of all proxy and disagreement metrics."""
        s = self.summary
        if s.total_frames < 10:
            ax.set_title("Insufficient data")
            return

        all_vars = {
            "L": s.L_trace,
            "H": s.H_trace,
            "C": s.C_trace,
        }
        # Include all available proxies
        if hasattr(s, 'n_mean_confs') and s.n_mean_confs and len(s.n_mean_confs) > 0:
            all_vars["MeanConf(n)"] = s.n_mean_confs
        if hasattr(s, 'niqe_scores') and s.niqe_scores and len(s.niqe_scores) > 0:
            all_vars["NIQE"] = s.niqe_scores
        _ext_matrix = [
            ("Tenengrad", "tenengrad_trace"),
            ("EdgeDens", "edge_density_trace"),
            ("LocalContr", "local_contrast_trace"),
            ("Brenner", "brenner_trace"),
            ("ColorH", "color_entropy_trace"),
        ]
        for name, attr in _ext_matrix:
            data = getattr(s, attr, [])
            if data and len(data) >= 10:
                all_vars[name] = data
        all_vars.update({
            "Det Gap": s.det_count_gaps,
            "Conf Gap": s.mean_conf_gaps,
            "IoU Agree": s.iou_agreements,
            "Extra(s)": s.extra_dets_s_trace,
            "Disagree": s.disagreement_scores,
        })

        names = list(all_vars.keys())
        arrays = [np.array(v, dtype=float) for v in all_vars.values()]
        n = min(len(a) for a in arrays)
        matrix = np.column_stack([a[:n] for a in arrays])
        corr = np.corrcoef(matrix.T)

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)

        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{corr[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if abs(corr[i, j]) > 0.5 else "black",
                        fontsize=7)

        ax.set_title("Proxy ↔ Disagreement Correlation Matrix")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------
    def generate_text_report(self) -> str:
        """Generate a human-readable validation report."""
        s = self.summary
        lines = [
            "=" * 70,
            "  DMS-Raptor Proxy Validation Report",
            "=" * 70,
            "",
            f"Video: {s.video}",
            f"Frames analyzed: {s.total_frames}",
            f"Alpha (L/H weighting): {s.alpha}",
            f"Proxy settings tested: {len(s.proxy_settings)}",
            "",
        ]

        # Model comparison summary
        if s.n_det_counts and s.s_det_counts:
            lines.append("--- Model Comparison ---")
            lines.append(f"  n-model mean detections: {np.mean(s.n_det_counts):.1f}")
            lines.append(f"  s-model mean detections: {np.mean(s.s_det_counts):.1f}")
            lines.append(f"  n-model mean confidence: {np.mean(s.n_mean_confs):.3f}")
            lines.append(f"  s-model mean confidence: {np.mean(s.s_mean_confs):.3f}")
            lines.append(f"  n-model mean latency:    {np.mean(s.n_latencies):.1f} ms")
            lines.append(f"  s-model mean latency:    {np.mean(s.s_latencies):.1f} ms")
            lines.append(f"  Mean IoU agreement:      {np.mean(s.iou_agreements):.3f}")
            lines.append(f"  Mean disagreement score: {np.mean(s.disagreement_scores):.3f}")
            lines.append("")

        # Correlation results
        correlations = self.compute_correlations()
        if correlations:
            lines.append("--- Correlation Analysis ---")
            lines.append(f"{'Proxy':<16} {'Metric':<22} {'Pearson r':>10} "
                         f"{'p-value':>12} {'Spearman ρ':>12} {'Strength':>10}")
            lines.append("-" * 86)
            for c in correlations:
                lines.append(
                    f"{c.name_x:<16} {c.name_y:<22} {c.pearson_r:>10.4f} "
                    f"{c.pearson_p:>12.2e} {c.spearman_rho:>12.4f} {c.strength:>10}"
                )
            lines.append("")

            # Highlight key finding
            key_corr = [c for c in correlations
                        if c.name_x == "C-score" and c.name_y == "Disagreement Score"]
            if key_corr:
                kc = key_corr[0]
                lines.append(f">> KEY RESULT: C-score ↔ Disagreement: "
                             f"r = {kc.pearson_r:.4f} ({kc.strength})")
                if abs(kc.pearson_r) >= 0.3:
                    lines.append("   ✓ C-score has meaningful predictive power for model disagreement")
                else:
                    lines.append("   ⚠ C-score shows weak correlation — proxy may need refinement")
                lines.append("")

        # Sensitivity
        sensitivity = self.compute_sensitivity()
        if sensitivity.setting_pairs:
            lines.append("--- Sensitivity Analysis ---")
            for (ka, kb), pct in zip(sensitivity.setting_pairs, sensitivity.agreement_pcts):
                status = "✓" if pct >= 90 else "⚠" if pct >= 80 else "✗"
                lines.append(f"  {status} {ka} vs {kb}: {pct:.1f}% agreement")
            lines.append("")

            min_agree = min(sensitivity.agreement_pcts) if sensitivity.agreement_pcts else 0
            if min_agree >= 90:
                lines.append("   ✓ Proxy settings are stable — 160/64 is sufficient")
            elif min_agree >= 80:
                lines.append("   ⚠ Some sensitivity to proxy settings — consider 320/128")
            else:
                lines.append("   ✗ Significant sensitivity — higher resolution proxies recommended")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)
