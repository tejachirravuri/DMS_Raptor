"""
Generate a self-contained DMS-Raptor Thesis Experiment Report (.docx).

This script produces a 12-chapter research-level handout that explains:
- System architecture and pipeline
- Mathematical foundations (human-readable formulas)
- Policy descriptions with algorithm steps
- Experiment results with per-policy analysis
- Validation findings, hyperparameter sensitivity, feature engineering
- Discussion, conclusions, and future work

Designed to be readable by a thesis committee member with no code access.
"""
from __future__ import annotations

import os
import datetime
from typing import List, Optional, Dict, Any

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (consistent with generate_thesis_defense_guide.py)
# ═══════════════════════════════════════════════════════════════════════════

def _set_cell_shading(cell, color_hex: str):
    """Set a table cell's background colour."""
    tc_pr = cell._element.get_or_add_tcPr()
    shd = tc_pr.makeelement(qn("w:shd"), {
        qn("w:fill"): color_hex,
        qn("w:val"): "clear",
    })
    tc_pr.append(shd)


def _add_heading(doc, text: str, level: int = 1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)
    return h


def _add_para(doc, text: str, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return p


def _add_equation(doc, equation_text: str, description: str = ""):
    """Add a centred, styled equation block."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(equation_text)
    run.font.size = Pt(12)
    run.font.name = "Cambria Math"
    run.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    if description:
        d = doc.add_paragraph()
        d.alignment = WD_ALIGN_PARAGRAPH.CENTER
        dr = d.add_run(description)
        dr.font.size = Pt(10)
        dr.italic = True
        dr.font.color.rgb = RGBColor(100, 100, 100)


def _add_numbered_steps(doc, steps: List[str]):
    """Add numbered algorithm steps."""
    for i, step in enumerate(steps, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.4)
        num_run = p.add_run(f"Step {i}:  ")
        num_run.bold = True
        num_run.font.size = Pt(11)
        num_run.font.color.rgb = RGBColor(0, 102, 153)
        text_run = p.add_run(step)
        text_run.font.size = Pt(11)


def _add_bullet(doc, text: str, indent: float = 0.3):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(indent)
    for run in p.runs:
        run.font.size = Pt(11)
    if not p.runs:
        p.clear()
        run = p.add_run(text)
        run.font.size = Pt(11)
    else:
        p.runs[0].text = text
    return p


def _add_table(doc, headers: List[str], rows: List[List[str]],
               col_widths: Optional[List[float]] = None):
    """Add a formatted table with coloured header and alternating rows."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = h
        _set_cell_shading(cell, "003366")
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.bold = True
                run.font.size = Pt(10)

    # Data rows
    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(val)
            if i % 2 == 0:
                _set_cell_shading(cell, "E8F4FD")
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)

    if col_widths:
        for j, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[j].width = Inches(w)

    doc.add_paragraph()
    return table


def _fmt(val, decimals=2) -> str:
    """Format a number for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ═══════════════════════════════════════════════════════════════════════════
# Main report generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_thesis_report(
    summaries: List,
    output_path: str,
    validation_summary=None,
    plot_dir: str = "",
    video_name: str = "",
):
    """Generate the complete 12-chapter thesis experiment report.

    Parameters
    ----------
    summaries : list[RunSummary]
        List of completed run summaries with per-frame traces.
    output_path : str
        Path to write the .docx file.
    validation_summary : ValidationSummary or None
        Optional validation results (dual-model comparison).
    plot_dir : str
        Directory containing exported plot PNGs (for figure references).
    video_name : str
        Name of the input video (for titles and references).
    """
    doc = Document()

    # Page setup
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    # Default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    now = datetime.datetime.now().strftime("%B %d, %Y")
    vid_label = video_name or "Experiment Video"

    # ══════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ══════════════════════════════════════════════════════════════════════
    for _ in range(4):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run(
        "DMS-Raptor: Dynamic Model Switching\n"
        "for Real-Time UAV Inspection"
    )
    r.font.size = Pt(26)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 51, 102)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = subtitle.add_run(
        "\nThesis Experiment Report\n"
        f"Input: {vid_label}"
    )
    r2.font.size = Pt(16)
    r2.font.color.rgb = RGBColor(0, 102, 153)

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rm = meta.add_run(f"Generated: {now}  |  Policies evaluated: {len(summaries)}")
    rm.font.size = Pt(12)
    rm.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "Table of Contents", level=1)
    toc_entries = [
        "1.  Executive Summary",
        "2.  System Architecture",
        "3.  Per-Frame Pipeline Walkthrough",
        "4.  Mathematical Foundations",
        "5.  Policy Descriptions",
        "6.  Latency Analysis",
        "7.  Switching Behaviour",
        "8.  Proxy Validation Results",
        "9.  Hyperparameter Sensitivity",
        "10. Feature Engineering Analysis",
        "11. Experiment Discussion",
        "12. Conclusions & Future Work",
    ]
    for entry in toc_entries:
        _add_para(doc, entry, size=11)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 1 — EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "1. Executive Summary", level=1)

    _add_para(
        doc,
        "DMS-Raptor (Dynamic Model Switching for Real-time Adaptive Processing "
        "and Tracking on dRones) is a complexity-aware inference framework that "
        "dynamically selects between a fast, lightweight object detection model "
        "(YOLOv8n) and a more accurate but slower model (YOLOv8s) on a per-frame "
        "basis. The system targets real-time UAV inspection scenarios where "
        "computational resources are constrained and scenes vary in complexity "
        "throughout a flight.",
    )

    _add_para(
        doc,
        "The core hypothesis is that not every frame requires the same model: "
        "simple, well-lit scenes with clear objects can be handled by the fast "
        "model, while complex scenes (blur, clutter, poor lighting) benefit from "
        "the heavier model. By switching intelligently, DMS-Raptor achieves "
        "accuracy close to the slow model while maintaining latency close to "
        "the fast model.",
    )

    # Results summary table
    if summaries:
        _add_heading(doc, "Key Results", level=2)
        headers = [
            "Policy", "Frames", "T_total (ms)", "P95 (ms)",
            "Slow %", "Switches", "Sw/100",
        ]
        rows = []
        for s in summaries:
            rows.append([
                s.policy,
                str(s.total_frames),
                _fmt(s.T_total_ms_mean),
                _fmt(s.T_total_ms_p95),
                _fmt(s.slow_pct, 1),
                str(s.switches),
                _fmt(s.sw_per_100, 1),
            ])
        _add_table(doc, headers, rows, col_widths=[1.3, 0.7, 0.9, 0.9, 0.7, 0.8, 0.7])

        # Best policy recommendation
        switching = [s for s in summaries if s.policy not in ("n_only", "s_only")]
        if switching:
            best = min(switching, key=lambda s: s.T_total_ms_mean)
            _add_para(
                doc,
                f"Recommendation: Among switching policies, '{best.policy}' "
                f"achieved the lowest mean latency ({_fmt(best.T_total_ms_mean)} ms) "
                f"with {_fmt(best.slow_pct, 1)}% slow-model usage and "
                f"{_fmt(best.sw_per_100, 1)} switches per 100 frames.",
                bold=True,
            )

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 2 — SYSTEM ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "2. System Architecture", level=1)

    _add_para(
        doc,
        "DMS-Raptor follows a modular architecture with clear separation of concerns. "
        "The system is organised into five main modules:",
    )

    arch_table = [
        ["core/engine.py", "StreamingEngine",
         "Orchestrates the per-frame pipeline: capture, proxy computation, "
         "controller decision, model inference, overlay, trace accumulation."],
        ["core/controller.py", "SwitchController",
         "Implements all 7 switching policies. Receives proxy signals and "
         "returns a model choice ('n' or 's') with hysteresis and dwell guards."],
        ["core/complexity.py", "complexity_proxies_fast()",
         "Computes scene complexity proxies: Laplacian variance (L), histogram "
         "entropy (H), combined C-score, and NIQE quality index."],
        ["core/config.py", "RunConfig / InferenceParams",
         "Dataclasses holding all configurable parameters: thresholds, EMA "
         "betas, proxy sizes, actuator guards."],
        ["validation/validator.py", "ProxyValidator",
         "Dual-model validation: runs both YOLOv8n and YOLOv8s on every frame "
         "to measure detection disagreement and correlate with proxy signals."],
    ]
    _add_table(
        doc,
        ["Module", "Class / Function", "Responsibility"],
        arch_table,
        col_widths=[1.5, 1.5, 3.5],
    )

    _add_heading(doc, "Two-Model Rationale", level=2)
    _add_para(
        doc,
        "YOLOv8n (nano, 3.2M parameters) offers fast inference (~4-8 ms on GPU) "
        "but lower accuracy on challenging scenes. YOLOv8s (small, 11.2M parameters) "
        "provides higher accuracy (~8-18 ms on GPU) but consumes more compute. "
        "The switching framework exploits this complementarity: use the fast model "
        "when the scene is simple, and the accurate model only when complexity or "
        "quality degradation demands it.",
    )

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 3 — PER-FRAME PIPELINE WALKTHROUGH
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "3. Per-Frame Pipeline Walkthrough", level=1)

    _add_para(
        doc,
        "For every frame in the input video, the StreamingEngine executes the "
        "following six steps. Each step is timed individually, and the total "
        "per-frame latency (T_total) is the sum of all steps.",
    )

    # Step 1
    _add_heading(doc, "Step 1: Frame Capture", level=2)
    _add_para(
        doc,
        "The frame is read from the video source using OpenCV's VideoCapture. "
        "For live streams (RTSP/USB), the frame is grabbed in real time. For "
        "recorded video files, frames are read sequentially. A stride parameter "
        "allows skipping frames (e.g., stride=2 processes every other frame). "
        "This step is not timed in T_total as it is I/O-bound and independent "
        "of the detection pipeline.",
    )

    # Step 2
    _add_heading(doc, "Step 2: Scene Proxy Computation (T_scene)", level=2)
    _add_para(
        doc,
        "The captured frame is analysed to estimate scene complexity. The full-resolution "
        "frame is resized to a smaller proxy image (default: 160x160 pixels) to reduce "
        "computation time. Three proxies are computed on this downsampled image:",
    )
    _add_numbered_steps(doc, [
        "Convert the proxy image to greyscale.",
        "Compute the Laplacian variance (L): Apply a 3x3 Laplacian kernel to the "
        "greyscale image, then take the variance of the result. High variance indicates "
        "sharp edges and texture; low variance indicates blur or uniform areas.",
        "Compute the histogram entropy (H): Calculate the intensity histogram of the "
        "greyscale image (default: 64 bins), normalise it to a probability distribution, "
        "then compute Shannon entropy. High entropy indicates diverse intensity values "
        "(complex scene); low entropy indicates uniform lighting.",
        "Compute the combined C-score: Normalise L and H to [0, 1] using rolling "
        "percentile normalisation, then combine as C = alpha * L_norm + (1 - alpha) * H_norm. "
        "The default alpha = 0.6 gives 60% weight to texture sharpness.",
        "(Optional) Compute the NIQE quality index on the proxy image. This is a "
        "no-reference image quality metric based on natural scene statistics.",
    ])
    _add_para(
        doc,
        "Typical timing: T_scene ranges from 0.2 to 1.5 ms depending on proxy size "
        "and hardware. Using proxy_size=160 on GPU typically costs under 0.5 ms.",
        italic=True,
    )

    # Step 3
    _add_heading(doc, "Step 3: Controller Decision (T_ctrl)", level=2)
    _add_para(
        doc,
        "The SwitchController receives the proxy signals (C, L, H, mean_conf, "
        "conf_drop, niqe) and decides which model to use for the current frame. "
        "The decision logic depends on the active policy (see Chapter 5 for details). "
        "All switching policies share two stabiliser mechanisms:",
    )
    _add_numbered_steps(doc, [
        "Dwell guard: After switching models, the controller holds the new model "
        "for at least min_dwell_frames (default: 10 frames) before allowing another "
        "switch. This prevents rapid oscillation.",
        "Rate limiter: At most max_switches_per_100 (default: 12) switches are allowed "
        "in any rolling window of 100 frames. If this limit is exceeded, the controller "
        "suppresses further switches until the window clears.",
    ])
    _add_para(
        doc,
        "Typical timing: T_ctrl is negligible (0.01-0.05 ms) as it is pure "
        "arithmetic on scalar values.",
        italic=True,
    )

    # Step 4
    _add_heading(doc, "Step 4: Model Inference (T_infer)", level=2)
    _add_para(
        doc,
        "Based on the controller's decision, the selected YOLO model (n or s) runs "
        "forward inference on the full-resolution frame. The inference pipeline includes:",
    )
    _add_numbered_steps(doc, [
        "Resize the frame to the model's input size (default: 640x640).",
        "Run the neural network forward pass on the GPU (or CPU).",
        "Apply Non-Maximum Suppression (NMS) with IoU threshold (default: 0.45) "
        "and confidence threshold (default: 0.001 internal, 0.25 display).",
        "Extract detection bounding boxes, confidence scores, and class labels.",
    ])
    _add_para(
        doc,
        "The mean detection confidence from this step feeds back into the conf_ema "
        "policy's EMA tracker for the next frame's decision.",
        italic=True,
    )

    # Step 5
    _add_heading(doc, "Step 5: Overlay Annotation", level=2)
    _add_para(
        doc,
        "Bounding boxes, confidence scores, and a status header panel are drawn onto "
        "the frame. The header shows: active policy, selected model, frame index, "
        "T_total, C-score, detection count. If save_annotated_video is enabled, "
        "this annotated frame is written to the output video file.",
    )

    # Step 6
    _add_heading(doc, "Step 6: Trace Accumulation", level=2)
    _add_para(
        doc,
        "Every per-frame metric is appended to trace arrays for post-hoc analysis: "
        "T_total, T_scene, T_ctrl, T_infer (for the chosen model), C-score, L, H, "
        "model choice, dwell counter, detection count, mean confidence, conf_drop, "
        "NIQE score. These traces form the basis of all plots and analysis in this "
        "report.",
    )

    # Timing summary
    _add_heading(doc, "Timing Summary", level=2)
    _add_para(doc, "Total per-frame latency:")
    _add_equation(
        doc,
        "T_total = T_scene + T_ctrl + T_infer",
        "where T_infer is T_infer_n or T_infer_s depending on the controller's choice.",
    )

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 4 — MATHEMATICAL FOUNDATIONS
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "4. Mathematical Foundations", level=1)

    _add_para(
        doc,
        "This chapter presents all mathematical formulas used in DMS-Raptor in "
        "human-readable notation. Each formula is followed by an explanation of "
        "what it computes and why it is used.",
    )

    # 4.1 Laplacian Variance
    _add_heading(doc, "4.1 Laplacian Variance (L)", level=2)
    _add_equation(
        doc,
        "L = Var( Laplacian(I_grey) )",
        "I_grey is the greyscale proxy image; Laplacian applies the 3x3 discrete "
        "Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]].",
    )
    _add_para(
        doc,
        "Purpose: L measures the amount of texture and edge content in the scene. "
        "Sharp, textured scenes (e.g., detailed infrastructure) yield high L values; "
        "blurred or uniform scenes (e.g., sky, smooth surfaces) yield low L values. "
        "A drop in L suggests the model should switch to the more accurate detector "
        "because fewer edges make detection harder.",
    )

    # 4.2 Histogram Entropy
    _add_heading(doc, "4.2 Histogram Entropy (H)", level=2)
    _add_equation(
        doc,
        "H = - sum( p_i * log2(p_i) )   for i = 1, 2, ..., hist_bins",
        "p_i is the normalised frequency of intensity bin i; bins with p_i = 0 "
        "are excluded from the sum.",
    )
    _add_para(
        doc,
        "Purpose: H captures lighting diversity. A scene with uniform illumination "
        "(low entropy) may indicate fog, overexposure, or shadow, all of which "
        "degrade detection performance. High entropy suggests a well-lit scene "
        "with varied object appearances.",
    )

    # 4.3 Combined C-score
    _add_heading(doc, "4.3 Combined C-Score", level=2)
    _add_equation(
        doc,
        "C = alpha * L_norm + (1 - alpha) * H_norm",
        "alpha = 0.6 by default (60% weight on texture sharpness).",
    )
    _add_para(doc, "Normalisation (rolling percentile):")
    _add_equation(
        doc,
        "L_norm = (L - P_lo(L)) / (P_hi(L) - P_lo(L))",
        "P_lo and P_hi are the 10th and 90th percentiles over a rolling window "
        "of the last 200 frames. Same normalisation is applied to H.",
    )
    _add_para(
        doc,
        "Purpose: The C-score combines texture and illumination into a single "
        "[0, 1] complexity index. C close to 1 means high complexity "
        "(switch to the accurate model); C close to 0 means low complexity "
        "(stay with the fast model).",
    )

    # 4.4 Confidence EMA
    _add_heading(doc, "4.4 Confidence-EMA Tracking", level=2)
    _add_para(
        doc,
        "Instead of analysing the input image, the conf_ema policy monitors the "
        "detection output: if confidence drops, the scene is getting harder.",
    )
    _add_equation(
        doc,
        "EMA_fast = beta_fast * x + (1 - beta_fast) * EMA_fast_prev",
        "beta_fast = 0.30 (responds in ~3 frames).",
    )
    _add_equation(
        doc,
        "EMA_slow = beta_slow * x + (1 - beta_slow) * EMA_slow_prev",
        "beta_slow = 0.02 (responds in ~50 frames, establishes a baseline).",
    )
    _add_equation(
        doc,
        "rise = max(0, (EMA_fast - EMA_slow) / EMA_slow)",
        "Positive rise means recent confidence is above the baseline (scene is easy).",
    )
    _add_equation(
        doc,
        "drop = max(0, (EMA_slow - EMA_fast) / EMA_slow)",
        "Positive drop means recent confidence is below the baseline (scene is hard).",
    )
    _add_para(
        doc,
        "Switching rule: If drop > conf_ema_c_high (0.12), switch to the slow model. "
        "If drop < conf_ema_c_low (0.04), switch back to the fast model. Otherwise, "
        "hold the current model. This is a hysteresis mechanism to prevent chattering.",
    )

    # 4.5 NIQE
    _add_heading(doc, "4.5 NIQE (Natural Image Quality Evaluator)", level=2)
    _add_para(
        doc,
        "NIQE is a no-reference image quality assessment metric based on natural "
        "scene statistics (NSS). It works by:",
    )
    _add_numbered_steps(doc, [
        "Compute the Mean Subtracted Contrast Normalised (MSCN) coefficients: "
        "MSCN(i,j) = (I(i,j) - mu(i,j)) / (sigma(i,j) + 1), where mu and sigma "
        "are local mean and standard deviation computed over a Gaussian window.",
        "Fit a Generalised Gaussian Distribution (GGD) to the MSCN coefficients "
        "to extract shape parameter beta and variance.",
        "Compute pairwise products of adjacent MSCN coefficients in four directions "
        "(horizontal, vertical, and two diagonals) and fit an Asymmetric GGD (AGGD) "
        "to each.",
        "Collect all fitted parameters into a feature vector and compute the "
        "Mahalanobis distance from a pristine natural image model.",
    ])
    _add_para(
        doc,
        "Important observation: In typical UAV inspection footage, NIQE scores have "
        "very low variation (coefficient of variation 0.2-0.5%). Empirical evaluation "
        "showed NIQE-based switching had negligible impact on scene-proxy decisions, "
        "leading to its retirement from the active policy set. The NIQE score remains "
        "available as a diagnostic trace metric.",
        italic=True,
    )

    # 4.6 Hysteresis & Dwell Guard
    _add_heading(doc, "4.6 Hysteresis Switching with Dwell Guard", level=2)
    _add_para(doc, "The hysteresis switching rule (used by combined_hyst, local_contrast_hyst, conf_ema):")
    _add_equation(
        doc,
        "If signal > c_high:  choice = 's'  (switch to accurate model)",
    )
    _add_equation(
        doc,
        "If signal < c_low:   choice = 'n'  (switch to fast model)",
    )
    _add_equation(
        doc,
        "If c_low <= signal <= c_high:  choice = previous_choice  (hold current model)",
        "This dead zone prevents oscillation when the signal is near a boundary.",
    )
    _add_para(doc, "Dwell guard constraint:")
    _add_equation(
        doc,
        "Switch is suppressed if frames_on_current_model < min_dwell_frames",
        "Default: min_dwell_frames = 10 (minimum 10 frames before allowing a switch).",
    )
    _add_para(doc, "Rate limiter constraint:")
    _add_equation(
        doc,
        "Switch is suppressed if switches_in_last_100_frames > max_switches_per_100",
        "Default: max_switches_per_100 = 12.",
    )

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 5 — POLICY DESCRIPTIONS
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "5. Policy Descriptions", level=1)

    _add_para(
        doc,
        "DMS-Raptor supports 7 switching policies. Two are baselines (always use "
        "one model), and five are dynamic switching policies that select the model "
        "on a per-frame basis.",
    )

    # n_only
    _add_heading(doc, "5.1 Baseline: n_only", level=2)
    _add_para(
        doc,
        "Always uses YOLOv8n (fast model) for every frame. Provides the lowest "
        "possible latency but may miss detections in complex scenes. This policy "
        "establishes the lower bound for latency and upper bound for missed detections.",
    )

    # s_only
    _add_heading(doc, "5.2 Baseline: s_only", level=2)
    _add_para(
        doc,
        "Always uses YOLOv8s (accurate model) for every frame. Provides the highest "
        "detection accuracy but at the cost of higher latency. This policy "
        "establishes the upper bound for both latency and accuracy.",
    )

    # entropy_only
    _add_heading(doc, "5.3 entropy_only", level=2)
    _add_para(doc, "Uses histogram entropy (H) as the sole switching signal.")
    _add_numbered_steps(doc, [
        "Compute H (histogram entropy) on the proxy image.",
        "Normalise H to [0, 1] using rolling percentile normalisation.",
        "If H_norm > c_high (default 0.55): switch to YOLOv8s.",
        "If H_norm < c_low (default 0.45): switch to YOLOv8n.",
        "Otherwise: hold the current model.",
        "Apply dwell guard and rate limiter.",
    ])
    _add_para(
        doc,
        "Trade-off: Simple and fast proxy computation. However, entropy alone may "
        "miss blur-related complexity that Laplacian variance captures.",
        italic=True,
    )

    # combined
    _add_heading(doc, "5.4 combined", level=2)
    _add_para(doc, "Uses the combined C-score (L + H) with a single threshold.")
    _add_numbered_steps(doc, [
        "Compute L (Laplacian variance) and H (histogram entropy).",
        "Normalise both to [0, 1] using rolling percentiles.",
        "Compute C = alpha * L_norm + (1 - alpha) * H_norm.",
        "If C > combined_mid (default 0.50): switch to YOLOv8s.",
        "If C <= combined_mid: switch to YOLOv8n.",
        "Apply dwell guard and rate limiter.",
    ])
    _add_para(
        doc,
        "Trade-off: Combines texture and illumination signals. However, a single "
        "threshold with no hysteresis band makes this policy prone to chattering "
        "when C oscillates around the threshold.",
        italic=True,
    )

    # combined_hyst
    _add_heading(doc, "5.5 combined_hyst (Recommended Scene-Based Policy)", level=2)
    _add_para(
        doc,
        "Uses the combined C-score with hysteresis (dual thresholds) for "
        "smooth switching.",
    )
    _add_numbered_steps(doc, [
        "Compute L, H, and C exactly as in the combined policy.",
        "Apply EMA smoothing to C: C_smooth = beta * C + (1 - beta) * C_smooth_prev.",
        "If C_smooth > c_high (default 0.55): switch to YOLOv8s.",
        "If C_smooth < c_low (default 0.45): switch to YOLOv8n.",
        "If c_low <= C_smooth <= c_high: hold the current model (dead zone).",
        "Apply dwell guard (min 10 frames) and rate limiter (max 12 per 100).",
    ])
    _add_para(
        doc,
        "Trade-off: The hysteresis dead zone and EMA smoothing significantly "
        "reduce chattering compared to the plain combined policy. This is the "
        "recommended scene-based policy.",
        italic=True,
    )

    # conf_ema
    _add_heading(doc, "5.6 conf_ema (Recommended Output-Based Policy)", level=2)
    _add_para(
        doc,
        "Instead of analysing the input image, this policy monitors the model's "
        "own detection confidence. If confidence drops below a baseline, the scene "
        "has become harder and the accurate model should be used.",
    )
    _add_numbered_steps(doc, [
        "After each frame's inference, compute the mean detection confidence.",
        "Update the fast EMA (beta = 0.30) and slow EMA (beta = 0.02).",
        "Compute the relative drop: drop = max(0, (EMA_slow - EMA_fast) / EMA_slow).",
        "If drop > conf_ema_c_high (default 0.12): switch to YOLOv8s.",
        "If drop < conf_ema_c_low (default 0.04): switch to YOLOv8n.",
        "If between c_low and c_high: hold the current model.",
        "Apply dwell guard and rate limiter.",
    ])
    _add_para(
        doc,
        "Key advantage: This policy does not require any proxy image computation "
        "(no T_scene cost for proxy calculation). It reacts directly to detection "
        "difficulty as perceived by the model itself. In practice, conf_ema often "
        "outperforms C-score-based policies because it captures detection difficulty "
        "that image-level statistics may miss.",
        italic=True,
    )

    # local_contrast_hyst
    _add_heading(doc, "5.7 local_contrast_hyst", level=2)
    _add_para(
        doc,
        "Uses local contrast (RMS of block standard deviations) as the sole scene "
        "proxy, combined with hysteresis switching. Designed for structurally "
        "homogeneous domains (e.g., glass insulators) where Laplacian and entropy "
        "signals carry redundant information.",
    )
    _add_numbered_steps(doc, [
        "Compute local contrast on the proxy image (block-wise RMS of std deviations).",
        "Normalize the proxy value using EMA-relative normalization.",
        "Apply hysteresis switching: if C > c_high switch to YOLOv8s, if C < c_low switch to YOLOv8n.",
        "If c_low <= C <= c_high: hold the current model (dead zone prevents oscillation).",
        "Apply dwell guard and rate limiter.",
    ])
    _add_para(
        doc,
        "Rationale: In the thesis evaluation, combined feature fusion (Laplacian + "
        "entropy) amplified noise for glass-insulator domains. A single structural "
        "proxy avoids this issue and was found to be the best scene-proxy policy for "
        "glass when paired with hysteresis switching.",
        italic=True,
    )

    # Parameters table
    _add_heading(doc, "5.8 Complete Parameter Reference", level=2)
    param_rows = [
        ["proxy_size", "160", "Resize target for proxy image (pixels)", "Scene proxies"],
        ["hist_bins", "64", "Number of bins for intensity histogram", "Entropy (H)"],
        ["alpha", "0.6", "Weight for Laplacian vs Entropy in C-score", "C-score"],
        ["c_low", "0.45", "Lower hysteresis threshold for C-score", "Hysteresis"],
        ["c_high", "0.55", "Upper hysteresis threshold for C-score", "Hysteresis"],
        ["c_ema_beta", "0.25", "EMA smoothing for C-score signal", "Smoothing"],
        ["conf_ema_fast_beta", "0.30", "Fast EMA smoothing (~3 frame response)", "conf_ema"],
        ["conf_ema_slow_beta", "0.02", "Slow EMA baseline (~50 frame response)", "conf_ema"],
        ["conf_ema_c_high", "0.12", "Switch to s when confidence drops 12%", "conf_ema"],
        ["conf_ema_c_low", "0.04", "Switch back to n when within 4%", "conf_ema"],
        ["mp_c_high", "0.10", "Switch to s when composite proxy drop >= 10%", "multi_proxy"],
        ["mp_c_low", "0.03", "Switch back to n when drop <= 3%", "multi_proxy"],
        ["min_dwell_frames", "10", "Minimum frames before allowing a switch", "Stability"],
        ["max_switches_per_100", "12", "Maximum switches in any 100-frame window", "Stability"],
        ["imgsz", "640", "Model input image size (pixels)", "Inference"],
        ["iou_nms", "0.45", "IoU threshold for Non-Maximum Suppression", "Inference"],
    ]
    _add_table(
        doc,
        ["Parameter", "Default", "Description", "Used By"],
        param_rows,
        col_widths=[1.5, 0.6, 2.8, 1.0],
    )

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 6 — LATENCY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "6. Latency Analysis", level=1)

    if summaries:
        _add_para(
            doc,
            f"This chapter analyses the per-frame latency (T_total) across "
            f"{len(summaries)} evaluated policies on the input video '{vid_label}'.",
        )

        # Detailed latency table
        _add_heading(doc, "6.1 Aggregate Latency Metrics", level=2)
        lat_headers = [
            "Policy", "Mean (ms)", "Median (ms)", "Std (ms)",
            "P95 (ms)", "P99 (ms)", "Min (ms)", "Max (ms)",
        ]
        lat_rows = []
        for s in summaries:
            t = np.array(s.T_total_trace) if s.T_total_trace else np.array([0])
            lat_rows.append([
                s.policy,
                _fmt(np.mean(t)),
                _fmt(np.median(t)),
                _fmt(np.std(t)),
                _fmt(np.percentile(t, 95)),
                _fmt(np.percentile(t, 99)),
                _fmt(np.min(t)),
                _fmt(np.max(t)),
            ])
        _add_table(doc, lat_headers, lat_rows)

        # Timing breakdown
        _add_heading(doc, "6.2 Timing Breakdown", level=2)
        _add_para(
            doc,
            "The following table shows how T_total is composed for each policy. "
            "T_scene covers proxy computation, T_ctrl covers the switching decision, "
            "and T_infer covers neural network forward pass plus NMS.",
        )
        bd_headers = ["Policy", "T_scene (ms)", "T_ctrl (ms)", "T_infer_n (ms)", "T_infer_s (ms)"]
        bd_rows = []
        for s in summaries:
            bd_rows.append([
                s.policy,
                _fmt(s.T_scene_ms_mean),
                _fmt(s.T_ctrl_ms_mean),
                _fmt(s.T_infer_n_ms_mean),
                _fmt(s.T_infer_s_ms_mean),
            ])
        _add_table(doc, bd_headers, bd_rows)

        # Interpretation
        _add_heading(doc, "6.3 Interpretation", level=2)

        n_run = next((s for s in summaries if s.policy == "n_only"), None)
        s_run = next((s for s in summaries if s.policy == "s_only"), None)

        if n_run and s_run:
            speedup_range = (
                f"n_only achieves {_fmt(n_run.T_total_ms_mean)} ms mean latency, "
                f"while s_only requires {_fmt(s_run.T_total_ms_mean)} ms. "
                f"The latency gap ({_fmt(s_run.T_total_ms_mean - n_run.T_total_ms_mean)} ms) "
                f"represents the potential savings from intelligent switching."
            )
            _add_para(doc, speedup_range)

        switching = [s for s in summaries if s.policy not in ("n_only", "s_only")]
        if switching:
            best_sw = min(switching, key=lambda s: s.T_total_ms_mean)
            _add_para(
                doc,
                f"Among switching policies, '{best_sw.policy}' achieves the lowest "
                f"mean latency ({_fmt(best_sw.T_total_ms_mean)} ms) with "
                f"{_fmt(best_sw.slow_pct, 1)}% slow-model usage. P95 latency is "
                f"{_fmt(best_sw.T_total_ms_p95)} ms, meaning 95% of frames are "
                f"processed within this time budget.",
            )

        _add_para(
            doc,
            "Refer to the exported Latency Dashboard plots for visual CDF, "
            "time series, box plots, and timing breakdown charts.",
            italic=True,
        )
    else:
        _add_para(doc, "No run data available for latency analysis.", italic=True)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 7 — SWITCHING BEHAVIOUR
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "7. Switching Behaviour", level=1)

    if summaries:
        _add_heading(doc, "7.1 Switching Statistics", level=2)
        sw_headers = ["Policy", "Total Switches", "Sw/100 Frames", "Slow Model %", "Frames on n", "Frames on s"]
        sw_rows = []
        for s in summaries:
            ct = s.choice_trace if s.choice_trace else []
            n_count = sum(1 for c in ct if c == "n")
            s_count = sum(1 for c in ct if c == "s")
            sw_rows.append([
                s.policy,
                str(s.switches),
                _fmt(s.sw_per_100, 1),
                _fmt(s.slow_pct, 1),
                str(n_count),
                str(s_count),
            ])
        _add_table(doc, sw_headers, sw_rows)

        # Dwell analysis
        _add_heading(doc, "7.2 Dwell Analysis", level=2)
        _add_para(
            doc,
            "Dwell time measures how many consecutive frames the system stays on "
            "each model before switching. Longer dwells indicate stable switching; "
            "short dwells indicate chattering.",
        )

        for s in summaries:
            if s.policy in ("n_only", "s_only"):
                continue
            if not s.dwell_trace:
                continue
            dw = np.array(s.dwell_trace)
            _add_para(
                doc,
                f"{s.policy}: mean dwell = {_fmt(np.mean(dw), 1)} frames, "
                f"min = {int(np.min(dw))}, max = {int(np.max(dw))}, "
                f"median = {_fmt(np.median(dw), 1)}.",
            )

        _add_heading(doc, "7.3 Choice Distribution Interpretation", level=2)
        _add_para(
            doc,
            "Baseline policies show 100% usage of their respective models. "
            "Switching policies should show a majority of fast-model (n) frames "
            "with selective use of the slow model (s) during complex scenes. "
            "A good switching policy achieves low slow_pct while maintaining "
            "detection quality comparable to s_only.",
        )
        _add_para(
            doc,
            "Refer to the Switching Analysis plots for binary heatmap timelines, "
            "rolling switching rate curves, and dwell histograms.",
            italic=True,
        )
    else:
        _add_para(doc, "No run data available for switching analysis.", italic=True)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 8 — PROXY VALIDATION RESULTS
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "8. Proxy Validation Results", level=1)

    if validation_summary is not None:
        _add_para(
            doc,
            "The ProxyValidator runs both YOLOv8n and YOLOv8s on every frame and "
            "measures detection disagreement. This provides ground truth for whether "
            "model switching was beneficial. Correlation analysis reveals which proxy "
            "signals best predict when the slow model is needed.",
        )

        _add_heading(doc, "8.1 Correlation Analysis", level=2)
        _add_para(
            doc,
            "Pearson and Spearman correlations between proxy signals and detection "
            "disagreement metrics (difference in detection count, confidence gap, "
            "IoU disagreement) indicate which proxies are most predictive.",
        )

        # Compute correlation table from raw traces
        from scipy.stats import pearsonr as _pearsonr
        proxy_defs = [
            ("L (Laplacian)", "L_trace"),
            ("H (Entropy)", "H_trace"),
            ("C (Combined)", "C_trace"),
            ("NIQE", "niqe_scores"),
            ("Tenengrad", "tenengrad_trace"),
            ("Edge density", "edge_density_trace"),
            ("Local contrast", "local_contrast_trace"),
            ("Brenner", "brenner_trace"),
            ("Color entropy", "color_entropy_trace"),
        ]
        metric_defs = [
            ("Det gap (s-n)", "det_count_gaps"),
            ("Conf gap (s-n)", "mean_conf_gaps"),
            ("Disagree score", "disagreement_scores"),
        ]
        corr_headers = ["Proxy Signal", "vs. Metric", "Pearson r", "p-value", "Strength"]
        corr_rows = []
        for pname, pattr in proxy_defs:
            pdata = getattr(validation_summary, pattr, [])
            if len(pdata) < 10:
                continue
            pdata = np.array(pdata, dtype=float)
            for mname, mattr in metric_defs:
                mdata = getattr(validation_summary, mattr, [])
                n = min(len(pdata), len(mdata))
                if n < 10:
                    continue
                r_val, p_val = _pearsonr(pdata[:n], np.array(mdata[:n], dtype=float))
                strength = "Strong" if abs(r_val) > 0.5 else "Moderate" if abs(r_val) > 0.3 else "Weak"
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                corr_rows.append([pname, mname, f"{r_val:+.3f}{stars}", f"{p_val:.2e}", strength])
        if corr_rows:
            _add_table(doc, corr_headers, corr_rows)
        elif hasattr(validation_summary, 'correlations') and validation_summary.correlations:
            old_headers = ["Proxy Signal", "vs. Metric", "Pearson r", "Strength"]
            old_rows = []
            for (proxy, metric), (r_val, p_val) in validation_summary.correlations.items():
                strength = "Strong" if abs(r_val) > 0.5 else "Moderate" if abs(r_val) > 0.3 else "Weak"
                old_rows.append([proxy, metric, _fmt(r_val, 3), strength])
            if old_rows:
                _add_table(doc, old_headers, old_rows)

        _add_heading(doc, "8.2 Proxy Effectiveness Ranking", level=2)
        _add_para(
            doc,
            "Based on correlation strength, proxies are ranked by their ability "
            "to predict when the slow model produces different (usually better) "
            "results than the fast model. The following proxies were evaluated: "
            "Laplacian variance (L), Shannon entropy (H), combined C-score, NIQE, "
            "Tenengrad (Sobel gradient energy), edge density (Canny), local contrast "
            "(RMS), Brenner gradient, and color entropy.",
        )
    else:
        _add_para(
            doc,
            "No validation data available. Run the ProxyValidator from the "
            "Thesis Experiments tab to generate dual-model comparison results.",
            italic=True,
        )

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 9 — HYPERPARAMETER SENSITIVITY
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "9. Hyperparameter Sensitivity", level=1)

    _add_para(
        doc,
        "Hyperparameter sensitivity analysis examines how changes in switching "
        "thresholds affect the system's behaviour without re-running inference. "
        "Using the recorded per-frame traces, the switching decision can be "
        "simulated with different threshold values to produce trade-off curves.",
    )

    _add_heading(doc, "9.1 Threshold Sweep Methodology", level=2)
    _add_numbered_steps(doc, [
        "Take the recorded proxy signal trace (C, conf_drop, or NIQE) from a completed run.",
        "For each pair of thresholds (c_low, c_high), replay the trace and simulate "
        "the switching decision frame by frame, applying the hysteresis rule.",
        "Record the resulting switching rate (sw_per_100) and slow-model usage (slow_pct) "
        "for each threshold pair.",
        "Visualise the results as heatmaps and a Pareto front.",
    ])
    _add_para(
        doc,
        "This approach is computationally free (pure NumPy on scalar arrays) and "
        "allows exploring thousands of threshold combinations in seconds.",
    )

    _add_heading(doc, "9.2 Key Trade-offs", level=2)
    _add_para(
        doc,
        "Wider hysteresis band (larger gap between c_low and c_high): Fewer switches "
        "(more stable) but slower response to scene changes. "
        "Narrower band: More responsive switching but higher switching rate.",
    )
    _add_para(
        doc,
        "Lower c_high: More aggressive use of the slow model (higher accuracy but "
        "higher latency). Higher c_high: More conservative, favouring the fast model.",
    )
    _add_para(
        doc,
        "min_dwell_frames: Higher values enforce longer stays on each model, reducing "
        "oscillation but potentially missing rapid scene transitions.",
    )
    _add_para(
        doc,
        "Refer to the Hyperparameter Sensitivity sub-tab plots for threshold sweep "
        "heatmaps and Pareto front visualisations.",
        italic=True,
    )

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 10 — FEATURE ENGINEERING ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "10. Feature Engineering Analysis", level=1)

    _add_para(
        doc,
        "This chapter analyses which signal features are most predictive of the "
        "switching decision and how they interact.",
    )

    _add_heading(doc, "10.1 Feature Importance (Correlation with Choice)", level=2)
    _add_para(
        doc,
        "For each proxy signal, the absolute Pearson correlation |r| with the binary "
        "model choice (n=0, s=1) is computed. Higher |r| means the signal is a better "
        "predictor of when the system switches to the slow model.",
    )

    if summaries:
        # Find a switching policy with traces
        switch_run = None
        for pref in ["conf_ema", "combined_hyst", "local_contrast_hyst", "combined", "entropy_only"]:
            switch_run = next((s for s in summaries if s.policy == pref and s.choice_trace), None)
            if switch_run:
                break

        if switch_run:
            # Compute feature importances
            choice_binary = np.array(
                [0 if c == "n" else 1 for c in switch_run.choice_trace],
                dtype=float,
            )
            signals = {
                "C-score": switch_run.C_trace,
                "Laplacian (L)": switch_run.L_trace,
                "Entropy (H)": switch_run.H_trace,
                "Mean Confidence": switch_run.mean_conf_trace,
                "Conf Drop": switch_run.conf_drop_trace,
                "NIQE": switch_run.niqe_trace,
                "Detections": switch_run.num_detections_trace,
            }
            fi_rows = []
            for name, trace in signals.items():
                if trace and len(trace) == len(choice_binary):
                    arr = np.array(trace, dtype=float)
                    if np.std(arr) > 0 and np.std(choice_binary) > 0:
                        r = float(np.corrcoef(arr, choice_binary)[0, 1])
                        fi_rows.append([name, _fmt(abs(r), 3), _fmt(r, 3),
                                        "Strong" if abs(r) > 0.5 else
                                        "Moderate" if abs(r) > 0.3 else "Weak"])

            if fi_rows:
                fi_rows.sort(key=lambda x: float(x[1]), reverse=True)
                _add_table(
                    doc,
                    ["Signal", "|r|", "r", "Strength"],
                    fi_rows,
                    col_widths=[1.5, 0.7, 0.7, 1.0],
                )
                _add_para(
                    doc,
                    f"Analysis based on '{switch_run.policy}' policy with "
                    f"{switch_run.total_frames} frames.",
                    italic=True,
                )

    _add_heading(doc, "10.2 Decision Boundary Visualisation", level=2)
    _add_para(
        doc,
        "The decision boundary plot shows L (Laplacian variance) vs H (histogram "
        "entropy) for each frame, coloured by the model choice (green = fast model, "
        "red = slow model). If the signals are well-separated by colour, a linear "
        "decision boundary exists, and the C-score approach is effective. Overlapping "
        "colours suggest the combined C-score is insufficient and a different signal "
        "(like conf_ema) may be more appropriate.",
    )

    _add_heading(doc, "10.3 Signal Lag Cross-Correlation", level=2)
    _add_para(
        doc,
        "Cross-correlation at different lag offsets reveals whether proxy signals "
        "lead or lag the switching decision. A peak at positive lag means the signal "
        "changes before the model switch occurs (predictive). A peak at negative lag "
        "means the signal responds after the switch (reactive, not predictive). "
        "The ideal proxy signal peaks at lag 0 or small positive lags.",
    )

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 11 — EXPERIMENT DISCUSSION
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "11. Experiment Discussion", level=1)

    _add_heading(doc, "11.1 What Worked", level=2)
    _add_para(
        doc,
        "The dynamic model switching concept is validated: intelligent switching "
        "achieves latency close to the fast model (n_only) while maintaining "
        "detection quality. The hysteresis mechanism with dwell guard effectively "
        "prevents chattering and provides stable switching behaviour.",
    )

    if summaries:
        n_run = next((s for s in summaries if s.policy == "n_only"), None)
        s_run = next((s for s in summaries if s.policy == "s_only"), None)
        ce_run = next((s for s in summaries if s.policy == "conf_ema"), None)

        if ce_run and n_run:
            overhead = ce_run.T_total_ms_mean - n_run.T_total_ms_mean
            _add_para(
                doc,
                f"The conf_ema policy adds only {_fmt(overhead)} ms average overhead "
                f"compared to n_only, while using the slow model for "
                f"{_fmt(ce_run.slow_pct, 1)}% of frames where it matters most.",
            )

    _add_heading(doc, "11.2 Why conf_ema Often Outperforms C-Score", level=2)
    _add_para(
        doc,
        "The confidence-EMA policy monitors the detection output directly rather "
        "than the input image. This has two advantages: (1) it captures detection "
        "difficulty that image-level statistics may miss (e.g., small or occluded "
        "objects that don't significantly change L or H), and (2) it avoids the "
        "proxy computation cost entirely. The EMA mechanism provides natural "
        "temporal smoothing.",
    )

    _add_heading(doc, "11.3 Why NIQE Has Low Variation", level=2)
    _add_para(
        doc,
        "NIQE measures deviation from a natural image model. In typical UAV "
        "inspection footage, the camera and scene characteristics remain relatively "
        "consistent (same altitude, similar structures, daylight). This results in "
        "NIQE scores with very low coefficient of variation (0.2-0.5%), requiring "
        "extremely tight switching thresholds (0.008/0.002 instead of 0.12/0.04). "
        "Using a smaller proxy_size (e.g., 160 instead of 640) increases NIQE "
        "variation slightly because less spatial averaging preserves local quality "
        "differences.",
    )

    _add_heading(doc, "11.4 Limitations", level=2)
    _add_para(
        doc,
        "Several limitations should be noted for future work:",
    )
    limitations = [
        "The proxy signals are computed on a downsampled image (160x160), which may "
        "miss fine-grained local complexity that only appears at full resolution.",
        "The switching thresholds are fixed (not learned). While this provides "
        "data-agnostic defaults, per-video or per-domain tuning could improve results.",
        "The current system supports only two models (n and s). Extending to a "
        "spectrum of model sizes (n/s/m/l/x) would require a multi-tier controller.",
        "NIQE's low variation in UAV footage limits its discriminative power. "
        "Alternative NR-IQA metrics (BRISQUE, NRQM) may provide better signal.",
        "The dwell guard and rate limiter are heuristic. Adaptive guard tuning "
        "based on scene change rate could be more effective.",
    ]
    for lim in limitations:
        _add_bullet(doc, lim)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 12 — CONCLUSIONS & FUTURE WORK
    # ══════════════════════════════════════════════════════════════════════
    _add_heading(doc, "12. Conclusions & Future Work", level=1)

    _add_heading(doc, "12.1 Key Conclusions", level=2)

    conclusions = [
        "Dynamic model switching is viable for real-time UAV inspection: the system "
        "achieves significant latency savings with minimal accuracy trade-off.",
        "Output-based switching (conf_ema) is generally more effective than "
        "input-based switching (C-score) because it directly measures detection "
        "difficulty rather than relying on image-level proxies.",
        "Hysteresis and dwell guards are essential for stable operation. Without "
        "them, switching policies exhibit excessive chattering.",
        "The combined_hyst policy provides a good balance for scene-based switching, "
        "while conf_ema is recommended when proxy computation overhead is a concern.",
    ]
    for c in conclusions:
        _add_bullet(doc, c)

    if summaries:
        switching = [s for s in summaries if s.policy not in ("n_only", "s_only")]
        if switching:
            best = min(switching, key=lambda s: s.T_total_ms_mean)
            _add_para(
                doc,
                f"Best performing switching policy in this experiment: '{best.policy}' "
                f"with mean T_total = {_fmt(best.T_total_ms_mean)} ms, "
                f"slow model usage = {_fmt(best.slow_pct, 1)}%, and "
                f"{_fmt(best.sw_per_100, 1)} switches per 100 frames.",
                bold=True,
            )

    _add_heading(doc, "12.2 Recommendations", level=2)
    recommendations = [
        "Use conf_ema as the default switching policy for general UAV inspection.",
        "Use combined_hyst when scene complexity information is needed for logging "
        "or when the application requires input-based switching decisions.",
        "Set proxy_size = 160 for a good balance between signal quality and computation speed.",
        "Set min_dwell_frames = 10 and max_switches_per_100 = 12 as conservative "
        "stability defaults.",
        "Run the ProxyValidator on representative videos to verify that the chosen "
        "policy's switching decisions correlate with actual detection difficulty.",
    ]
    for rec in recommendations:
        _add_bullet(doc, rec)

    _add_heading(doc, "12.3 Future Work", level=2)
    future = [
        "Multi-tier model cascade: Extend beyond two models to n/s/m/l, with a "
        "cost-aware controller that selects the optimal model size per frame.",
        "Learned switching: Train a lightweight classifier to predict the optimal "
        "model from frame features, replacing heuristic thresholds.",
        "Edge deployment optimisation: Profile and optimise for NVIDIA Jetson "
        "platforms with TensorRT acceleration.",
        "Temporal context: Incorporate short-term scene prediction (e.g., "
        "optical flow trends) to anticipate complexity changes.",
        "Multi-video evaluation: Benchmark across diverse inspection scenarios "
        "(power lines, bridges, wind turbines) to assess generalisability.",
    ]
    for f in future:
        _add_bullet(doc, f)

    # ──────────────────────────────────────────────────────────────────────
    # Final note
    # ──────────────────────────────────────────────────────────────────────
    doc.add_page_break()
    _add_heading(doc, "Appendix: Plot Reference", level=1)
    _add_para(
        doc,
        "The following plots are generated by the DMS-Raptor Thesis Experiments "
        "tab and should be reviewed alongside this report. If an export directory "
        "was specified, plots are saved with the input video name in their filenames.",
    )

    plot_refs = [
        ("Latency Dashboard", "latency_dashboard", "T_total time series, CDF, box plot, timing breakdown"),
        ("Switching Analysis", "switching_analysis", "Choice heatmap, rolling switch rate, dwell histogram"),
        ("Proxy Signals", "proxy_scatter", "Scatter plots with regression lines for any proxy pair"),
        ("Model Comparison", "model_comparison", "Detection/confidence scatter, latency comparison"),
        ("Hyperparameter Sensitivity", "threshold_heatmap", "Threshold sweep heatmaps, Pareto front"),
        ("EDA Statistics", "eda_statistics", "Distribution KDE, Q-Q plot, outlier detection, correlation matrix"),
        ("Feature Engineering", "feature_engineering", "Feature importance bars, decision boundary, lag analysis"),
    ]
    plot_rows = []
    for name, stem, desc in plot_refs:
        fname = f"{stem}_{video_name}.png" if video_name else f"{stem}.png"
        plot_rows.append([name, fname, desc])
    _add_table(
        doc,
        ["Sub-Tab", "Filename", "Contents"],
        plot_rows,
        col_widths=[1.5, 2.0, 3.0],
    )

    # ──────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────
    doc.save(output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    print("This script is designed to be called from the DMS-Raptor GUI.")
    print("Usage: generate_thesis_report(summaries, output_path, ...)")
    print("No standalone run data available.")
    sys.exit(0)
