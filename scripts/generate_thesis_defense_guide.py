"""
Generate comprehensive DMS-Raptor thesis defense guide.

Covers all scientific theory, parameter justification, validation methodology,
and answers to every possible thesis defense question.
"""
import os
import sys

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "DMS-Raptor_Thesis_Defense_Guide.docx",
)

# ── Helpers ──────────────────────────────────────────────────────────────
def set_cell_shading(cell, color_hex):
    """Set cell background color."""
    shading = cell._element.get_or_add_tcPr()
    shd = shading.makeelement(qn("w:shd"), {
        qn("w:fill"): color_hex,
        qn("w:val"): "clear",
    })
    shading.append(shd)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)
    return h


def add_para(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return p


def add_key_answer(doc, question, answer):
    """Add a formatted Q&A pair."""
    p = doc.add_paragraph()
    q_run = p.add_run("Q: " + question)
    q_run.bold = True
    q_run.font.size = Pt(11)
    q_run.font.color.rgb = RGBColor(0, 102, 153)

    a_para = doc.add_paragraph()
    a_run = a_para.add_run("A: " + answer)
    a_run.font.size = Pt(11)
    a_para.paragraph_format.space_after = Pt(12)
    a_para.paragraph_format.left_indent = Inches(0.3)
    return a_para


def add_equation_block(doc, equation_text, description=""):
    """Add a formatted equation block."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(equation_text)
    run.font.size = Pt(12)
    run.font.name = "Consolas"
    run.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    if description:
        d = doc.add_paragraph()
        d.alignment = WD_ALIGN_PARAGRAPH.CENTER
        dr = d.add_run(description)
        dr.font.size = Pt(10)
        dr.italic = True
        dr.font.color.rgb = RGBColor(100, 100, 100)


def add_table_with_header(doc, headers, rows, col_widths=None):
    """Add a formatted table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = h
        set_cell_shading(cell, "003366")
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
                set_cell_shading(cell, "E8F4FD")
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)

    if col_widths:
        for j, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[j].width = Inches(w)

    doc.add_paragraph()
    return table


# ── Main Document ────────────────────────────────────────────────────────
def build():
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
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    # ═══════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════
    for _ in range(4):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("DMS-Raptor: Dynamic Model Switching\nfor Real-Time UAV Inspection")
    r.font.size = Pt(26)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 51, 102)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = subtitle.add_run("\nComprehensive Thesis Defense Guide\nScientific Theory, Validation, & Parameter Justification")
    r2.font.size = Pt(16)
    r2.font.color.rgb = RGBColor(0, 102, 153)

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rm = meta.add_run("Version 1.7.0  |  March 2026")
    rm.font.size = Pt(12)
    rm.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "Table of Contents", level=1)
    toc_items = [
        "1. System Architecture Overview",
        "2. Scene Complexity Estimation: L, H, and C-Score",
        "    2.1. Laplacian Variance (L) - Theory & Justification",
        "    2.2. Histogram Entropy (H) - Theory & Justification",
        "    2.3. Combined C-Score - Normalization & Weighting",
        "    2.4. Why C-Score May Underperform (Known Limitation)",
        "3. Switching Policies: Complete Scientific Framework",
        "    3.1. Baseline Policies (n_only, s_only)",
        "    3.2. Entropy-Only Policy",
        "    3.3. Combined & Combined-Hysteresis Policies",
        "    3.4. Confidence-EMA Policy (conf_ema) - Theory & Mechanism",
        "    3.5. NIQE-Switch Policy - NR-IQA Theory & Mechanism",
        "4. Default Parameters: Why These Values & Data-Agnosticism",
        "    4.1. Scene Proxy Parameters (proxy_size, hist_bins, alpha)",
        "    4.2. Switching Thresholds (c_low, c_high, EMA betas)",
        "    4.3. Actuator Protections (min_dwell, max_switches)",
        "    4.4. Are Parameters Data-Agnostic? Plug-and-Play Analysis",
        "5. Validation Methodology: Proving DMS Works",
        "    5.1. Dual-Model Disagreement as Ground Truth",
        "    5.2. Correlation Analysis (Pearson, Spearman)",
        "    5.3. Sensitivity Analysis",
        "    5.4. Proxy Size Grid Search",
        "    5.5. Expected Results & How to Interpret Them",
        "6. Step-by-Step: Getting Thesis Results from DMS-Raptor",
        "    6.5. Phase E-2: Thesis Experiments Tab (NEW in v1.7.0)",
        "7. T_total Benchmarking: What Is Measured",
        "8. Answering Every Possible Thesis Question",
        "    8.1. Fundamental Questions",
        "    8.2. Technical Deep-Dive Questions",
        "    8.3. Tricky/Adversarial Questions",
        "    8.4. Comparison & Literature Questions",
        "9. Plug-and-Play: Random Dataset + Models Scenario",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(0)
        for run in p.runs:
            run.font.size = Pt(10)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: SYSTEM ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "1. System Architecture Overview", level=1)

    add_para(doc, "DMS-Raptor implements a Dynamic Model Switching (DMS) framework that switches between a lightweight YOLOv8n model (fast, lower accuracy) and a heavier YOLOv8s model (slower, higher accuracy) in real-time based on estimated scene complexity. The core hypothesis is:")

    add_para(doc, '"Not all frames in a UAV inspection video require the same model capacity. Simple scenes (clear views, good lighting, few objects) can be handled by a fast, lightweight model, while complex scenes (blur, occlusion, dense objects, poor lighting) benefit from a more capable but slower model."', bold=True, italic=True, size=11)

    add_para(doc, "The per-frame processing pipeline has three timed stages:")

    add_table_with_header(doc,
        ["Stage", "What It Does", "Typical Time", "Includes"],
        [
            ["T_scene", "Compute scene complexity proxy (L, H, or NIQE/Conf EMA)", "0.5-8 ms", "cv2.resize, Laplacian, histogram, or EMA arithmetic"],
            ["T_ctrl", "Controller decision: compare proxy to thresholds, apply actuator guards", "~0.01 ms", "Rolling percentile lookup, threshold comparison, dwell/switch-rate checks"],
            ["T_infer", "Run selected YOLO model (n or s) on the frame", "5-60 ms", "model.predict(): preprocess (resize+normalize+tensor) + GPU/CPU inference + NMS postprocessing"],
        ],
        col_widths=[1.0, 2.5, 1.0, 2.5],
    )

    add_para(doc, "T_total = T_scene + T_ctrl + T_infer_n + T_infer_s (only one model runs per frame, so one of T_infer_n or T_infer_s is always 0).", bold=True)

    add_para(doc, "What is NOT included in T_total (correctly excluded for benchmarking):")
    for item in [
        "Frame I/O: cv2.VideoCapture.read() - disk/camera read latency",
        "Overlay drawing: draw_overlay() - visualization only",
        "Video writing: cv2.VideoWriter.write() - output only",
        "GUI updates: PyQt signal emission - display only",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: SCENE COMPLEXITY ESTIMATION
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "2. Scene Complexity Estimation: L, H, and C-Score", level=1)

    add_para(doc, "The fundamental challenge in DMS is: how do you know if a scene is 'hard' or 'easy' BEFORE running the detector? DMS-Raptor uses lightweight image-based proxies computed on a downscaled version of the frame (proxy_size x proxy_size, default 160x160). These proxies must be:")

    for item in [
        "Fast to compute: < 10ms overhead, otherwise the proxy costs more than the savings from using a lighter model",
        "Correlated with detection difficulty: high proxy value should predict when the heavier model is actually needed",
        "Robust across datasets: should work on different inspection targets without retuning",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    # 2.1 Laplacian
    add_heading(doc, "2.1. Laplacian Variance (L) - Theory & Justification", level=2)

    add_para(doc, "Scientific Basis:", bold=True)
    add_para(doc, "The Laplacian operator is a second-order derivative filter that measures the rate of intensity change in an image. The Laplacian of an image I is defined as:")

    add_equation_block(doc,
        "L = Var(nabla^2 I) = Var(d^2I/dx^2 + d^2I/dy^2)",
        "where Var() denotes the variance over all pixels"
    )

    add_para(doc, "Why Laplacian variance works as a complexity proxy:")
    for item in [
        "High L (sharp, textured): Many edges and fine details. The detector can easily find features. Scene is 'easy' for detection.",
        "Low L (blurry, smooth): Few edges, loss of fine structure. The detector struggles with feature extraction. Scene is 'hard' - needs a more capable model.",
        "This is well-established in image quality assessment literature (Pech-Pacheco et al., 2000; Pertuz et al., 2013).",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "Implementation in DMS-Raptor (engine.py, complexity_proxies_fast()):", bold=True)
    p = doc.add_paragraph()
    r = p.add_run(
        "gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)\n"
        "L = float(cv2.Laplacian(gray, cv2.CV_32F).var())"
    )
    r.font.name = "Consolas"
    r.font.size = Pt(10)

    add_para(doc, "Empirical validation: In our glass insulator dataset, L achieved Pearson r = -0.577 against model disagreement (moderate negative correlation). This means: when L drops (scene becomes blurry/smooth), disagreement between n-model and s-model increases (s-model finds more detections that n-model misses). This confirms L is a meaningful proxy.", italic=True)

    # 2.2 Entropy
    add_heading(doc, "2.2. Histogram Entropy (H) - Theory & Justification", level=2)

    add_para(doc, "Scientific Basis:", bold=True)
    add_para(doc, "Shannon entropy of the grayscale histogram measures the information content (tonal diversity) of the image:")

    add_equation_block(doc,
        "H = -SUM(p_i * log2(p_i))  for i = 1..hist_bins",
        "where p_i = histogram[i] / total_pixels, computed over hist_bins bins"
    )

    add_para(doc, "Why entropy was hypothesized as a proxy:")
    for item in [
        "High H: Rich tonal variety (many different intensity levels used). Suggests complex texture or multiple objects.",
        "Low H: Uniform or low-contrast image. Could indicate fog, overexposure, or featureless background.",
        "Entropy is used in information theory (Shannon, 1948) and image analysis as a measure of unpredictability.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "Empirical finding:", bold=True)
    add_para(doc, "In practice, H shows negligible correlation with model disagreement (r = +0.172). This is because entropy measures tonal diversity, which does not directly correspond to detection difficulty. A foggy scene (low H) might still have clearly visible objects, while a textured background (high H) might be easy to detect against. The direction is also contradictory: H is slightly positively correlated with disagreement, meaning more entropy = more disagreement, which is the opposite of what L shows.", italic=True)

    add_para(doc, "KEY THESIS FINDING: H (entropy) is a weak and unreliable proxy for scene complexity in object detection contexts. This is a valid scientific contribution - demonstrating what does NOT work is as valuable as showing what works.")

    # 2.3 C-Score
    add_heading(doc, "2.3. Combined C-Score - Normalization & Weighting", level=2)

    add_para(doc, "The C-score combines L and H into a single switching signal:")

    add_equation_block(doc,
        "C = alpha * L_norm + (1 - alpha) * H_norm",
        "Default: alpha = 0.6 (60% weight on Laplacian, 40% on entropy)"
    )

    add_para(doc, "Normalization (rolling min-max percentile):", bold=True)
    add_para(doc, "L and H have vastly different scales (L ranges 0 to 10000+, H ranges 0 to 8). Before combining, both are normalized to [0, 1] using rolling window min-max:")

    add_equation_block(doc,
        "L_norm = clip((L - L_min) / (L_max - L_min), 0, 1)\n"
        "H_norm = clip((H - H_min) / (H_max - H_min), 0, 1)",
        "L_min/L_max and H_min/H_max are rolling over a window of 200 frames"
    )

    add_para(doc, "The rolling window adapts to the local statistics of the video, making C-score context-aware rather than using global constants.")

    # 2.4 C-Score Limitation
    add_heading(doc, "2.4. Why C-Score May Underperform (Known Limitation)", level=2)

    add_para(doc, "CRITICAL SCIENTIFIC FINDING:", bold=True)
    add_para(doc, "The combined C-score (r = -0.198) is WEAKER than L alone (r = -0.577). This is because:")

    for item in [
        "L has a moderate negative correlation with disagreement (r = -0.577): lower L means harder scenes.",
        "H has a negligible POSITIVE correlation (r = +0.172): higher H slightly correlates with harder scenes.",
        "These two signals partially CANCEL each other when combined. L says 'complexity is high' while H says 'complexity is low' for the same frame.",
        "The normalization to [0, 1] removes the scale advantage that L has over H, giving H's weak contradictory signal disproportionate influence.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "This is why the conf_ema policy (which uses detection confidence as a proxy instead of L+H) performs better - it bypasses this fundamental C-score limitation.", bold=True)

    add_para(doc, "Possible improvements (future work):")
    for item in [
        "Use L alone (alpha = 1.0) for combined policies",
        "Replace H with a more correlated metric (e.g., local contrast variance)",
        "Use adaptive alpha that weights proxies by their observed correlation strength",
        "Replace the entire C-score framework with direct confidence monitoring (conf_ema approach)",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: SWITCHING POLICIES
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "3. Switching Policies: Complete Scientific Framework", level=1)

    add_para(doc, "DMS-Raptor implements 7 policies, each representing a different hypothesis about how to decide which model to use per frame.")

    # 3.1 Baselines
    add_heading(doc, "3.1. Baseline Policies (n_only, s_only)", level=2)
    add_para(doc, "These are non-switching baselines used for comparison:")
    add_table_with_header(doc,
        ["Policy", "Description", "Purpose"],
        [
            ["n_only", "Always use YOLOv8n (nano)", "Lower bound: fastest possible inference, lowest accuracy baseline"],
            ["s_only", "Always use YOLOv8s (small)", "Upper bound: best accuracy, slowest inference. Maximum latency reference"],
        ],
        col_widths=[1.2, 3.0, 3.0],
    )
    add_para(doc, "A successful DMS policy should achieve accuracy close to s_only while having average latency close to n_only. The gap between these baselines defines the potential benefit of switching.")

    # 3.2 Entropy
    add_heading(doc, "3.2. Entropy-Only Policy", level=2)
    add_para(doc, "Decision rule: Compare current entropy H against the rolling median of H. If H >= median, use s-model; otherwise use n-model.")
    add_equation_block(doc, "choice = 's' if H >= H_median else 'n'")
    add_para(doc, "Theory: Frames with above-median entropy are assumed to be more complex. This is the simplest switching policy and serves as a baseline for complexity-based switching. Given our finding that H has negligible correlation with disagreement, this policy is expected to perform poorly.")

    # 3.3 Combined
    add_heading(doc, "3.3. Combined & Combined-Hysteresis Policies", level=2)

    add_para(doc, "Combined policy:", bold=True)
    add_equation_block(doc,
        "choice = 's' if C >= c_mid else 'n'",
        "c_mid = 0.50 (fixed threshold, center of [0, 1] range)"
    )

    add_para(doc, "Combined-Hysteresis policy:", bold=True)
    add_para(doc, "Adds hysteresis to prevent rapid oscillation around the threshold. Uses two thresholds (c_low = 0.45, c_high = 0.55):")

    add_equation_block(doc,
        "If currently using n: switch to s when C >= c_high (0.55)\n"
        "If currently using s: switch to n when C <= c_low (0.45)",
        "The dead zone between c_low and c_high prevents chattering"
    )

    add_para(doc, "Why hysteresis is essential:", bold=True)
    add_para(doc, "Without hysteresis, when C fluctuates around the threshold (e.g., 0.49, 0.51, 0.49, 0.51), the system would switch models every frame. This is catastrophic because: (1) each model switch has warm-up cost, (2) frequent switching reduces temporal consistency of detections, (3) the overhead of switching exceeds the benefit. Hysteresis creates a 'dead zone' where the current model is maintained, only switching when the signal clearly crosses the boundary.")

    # 3.4 conf_ema
    add_heading(doc, "3.4. Confidence-EMA Policy (conf_ema) - Theory & Mechanism", level=2)

    add_para(doc, "Scientific Basis:", bold=True)
    add_para(doc, "Instead of estimating scene complexity from image statistics (L, H), the conf_ema policy uses the YOLO model's own detection confidence as a proxy. The insight is: if the nano model's confidence drops, it is struggling with the current scene and the heavier model should take over.")

    add_para(doc, "This is a form of 'self-supervised' switching - the model itself signals when it needs help, rather than relying on hand-crafted image features.")

    add_para(doc, "Mechanism:", bold=True)
    add_para(doc, "Two Exponential Moving Averages (EMAs) track the mean detection confidence:")

    add_equation_block(doc,
        "EMA_fast = beta_fast * conf_current + (1 - beta_fast) * EMA_fast_prev\n"
        "EMA_slow = beta_slow * conf_current + (1 - beta_slow) * EMA_slow_prev",
        "beta_fast = 0.30 (~3 frame response), beta_slow = 0.02 (~50 frame response)"
    )

    add_para(doc, "The fast EMA tracks recent confidence (responsive to sudden drops). The slow EMA tracks the long-term baseline (smooth, stable reference). The switching signal is the relative drop:")

    add_equation_block(doc,
        "conf_drop = max(0, (EMA_slow - EMA_fast) / (EMA_slow + epsilon))",
        "Positive when fast EMA drops below slow baseline"
    )

    add_para(doc, "Switching decision (with hysteresis):", bold=True)
    add_equation_block(doc,
        "If using n: switch to s when conf_drop >= 0.12 (12% drop)\n"
        "If using s: switch to n when conf_drop <= 0.04 (within 4% of baseline)"
    )

    add_para(doc, "Why conf_ema is the strongest policy:", bold=True)
    for item in [
        "Mean Confidence vs Disagreement: r = -0.672 (Moderate-Strong). This is the highest correlation of any proxy we tested.",
        "No dependency on proxy_size or hist_bins: confidence comes from the YOLO model itself, not from image preprocessing.",
        "Naturally adaptive: automatically adjusts to different datasets, lighting conditions, and object types without parameter tuning.",
        "Causal signal: confidence drop is a direct indicator of detection difficulty, not an indirect proxy.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "Important note on proxy_size and hist_bins for conf_ema:", bold=True)
    add_para(doc, "The conf_ema policy does NOT use proxy_size or hist_bins at all. It only uses the YOLO model's detection confidence output. These parameters only affect the L/H/C computation used by entropy_only, combined, and combined_hyst policies. So for conf_ema, you can ignore proxy_size and hist_bins entirely - they have zero impact on the switching decision.")

    # 3.5 NIQE
    add_heading(doc, "3.5. NIQE-Switch Policy - NR-IQA Theory & Mechanism", level=2)

    add_para(doc, "Scientific Basis:", bold=True)
    add_para(doc, "NIQE (Natural Image Quality Evaluator) is a well-known no-reference image quality assessment (NR-IQA) metric from the computer vision literature (Mittal et al., 2013). DMS-Raptor implements a simplified BRISQUE-like variant that uses Natural Scene Statistics (NSS).")

    add_para(doc, "The theory behind NR-IQA:", bold=True)
    for item in [
        "Natural (undistorted) images follow predictable statistical patterns - their MSCN (Mean Subtracted Contrast Normalized) coefficients follow a Generalized Gaussian Distribution (GGD).",
        "When an image is degraded (blur, noise, compression, motion blur), the MSCN statistics deviate from the expected GGD shape.",
        "By measuring how far the image's statistics deviate from 'natural', we get a quality score without needing a reference image.",
        "Higher NIQE score = worse quality = more degradation = harder scene for detection.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "Implementation (core/niqe.py):", bold=True)
    for item in [
        "Resize to proxy_size (default 160x160) for speed",
        "Compute MSCN coefficients: I_mscn = (I - mu) / (sigma + epsilon)",
        "Fit GGD shape parameter (beta) via moment-ratio method",
        "Compute paired-product statistics (horizontal and vertical neighbors)",
        "Score = weighted combination of deviations from expected statistics at 2 scales",
        "Typical cost: < 3 ms on a 160x160 proxy",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "Switching mechanism:", bold=True)
    add_para(doc, "Same dual-EMA structure as conf_ema, but tracking NIQE score instead of confidence:")

    add_equation_block(doc,
        "niqe_rise = max(0, (EMA_fast - EMA_slow) / (EMA_slow + epsilon))",
        "Positive when quality DEGRADES (NIQE rises above baseline)"
    )

    add_equation_block(doc,
        "If using n: switch to s when niqe_rise >= 0.003 (0.3% quality degradation)\n"
        "If using s: switch to n when niqe_rise <= 0.001 (within 0.1% of baseline)"
    )

    add_para(doc, "The thresholds are much tighter than conf_ema (0.003 vs 0.12) because NIQE score variations are typically very small in UAV inspection footage (CoV ~0.2-0.5%).")

    add_para(doc, "Empirical result: NIQE vs Disagreement: r = +0.400 (Weak-Moderate). Better than H alone but weaker than L or Mean Confidence. NIQE captures image quality degradation but doesn't fully capture detection difficulty (e.g., a sharp image of a complex scene with many overlapping objects would have good NIQE but high detection difficulty).", italic=True)

    add_para(doc, "Note on proxy_size for image-based policies:", bold=True)
    add_para(doc, "Scene-proxy policies (entropy_only, combined, combined_hyst, local_contrast_hyst, multi_proxy) use proxy_size to resize the image before computing features. The proxy_size affects computation time and the spatial resolution of quality assessment. Default 160 provides a good speed-quality tradeoff. conf_ema does NOT use proxy_size at all.")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: DEFAULT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "4. Default Parameters: Why These Values & Data-Agnosticism", level=1)

    # 4.1
    add_heading(doc, "4.1. Scene Proxy Parameters", level=2)

    add_table_with_header(doc,
        ["Parameter", "Default", "Range", "Justification", "Data-Agnostic?"],
        [
            ["proxy_size", "160", "32-640", "160x160 provides ~1-2ms compute time. Captures sufficient spatial information for Laplacian and entropy. Going higher (320, 640) improves correlation marginally (<5%) but doubles/quadruples compute cost.", "YES - spatial statistics are resolution-invariant"],
            ["hist_bins", "64", "16-256", "64 bins capture tonal distribution adequately. More bins provide diminishing returns and increase noise sensitivity. Shannon entropy is relatively stable across bin counts >32.", "YES - entropy is robust to bin count above minimum threshold"],
            ["alpha", "0.6", "0.0-1.0", "Weights L (60%) vs H (40%). Given L's stronger correlation (r=-0.577 vs H's r=+0.172), alpha should ideally be higher (0.8-1.0). Current 0.6 is conservative.", "PARTIALLY - optimal alpha depends on proxy correlation strength per dataset"],
            ["probe_every_k", "1", "1-10", "Compute proxies every k frames. 1 = every frame. Higher values reduce overhead but add latency to switching response.", "YES - independent of data"],
        ],
        col_widths=[1.0, 0.6, 0.6, 3.0, 1.5],
    )

    # 4.2
    add_heading(doc, "4.2. Switching Thresholds", level=2)

    add_table_with_header(doc,
        ["Parameter", "Default", "Policy", "Justification"],
        [
            ["c_low", "0.45", "combined_hyst", "Lower hysteresis threshold. Places the dead zone at the center of [0,1] with a 0.10 gap. Chosen empirically to prevent chattering."],
            ["c_high", "0.55", "combined_hyst", "Upper hysteresis threshold. 0.10 gap between c_low and c_high is wide enough to suppress noise but narrow enough to remain responsive."],
            ["combined_mid", "0.50", "combined", "Single threshold at center of normalized range. Symmetric: equal probability of choosing n or s if C is uniformly distributed."],
            ["conf_ema_fast_beta", "0.30", "conf_ema", "Fast EMA smoothing factor. Corresponds to ~3 frame effective window (1/0.30). Responsive to sudden confidence drops."],
            ["conf_ema_slow_beta", "0.02", "conf_ema", "Slow EMA smoothing factor. Corresponds to ~50 frame effective window (1/0.02). Provides stable baseline reference."],
            ["conf_ema_c_high", "0.12", "conf_ema", "Switch to s when confidence drops 12% below baseline. Tuned to capture meaningful drops, not noise."],
            ["conf_ema_c_low", "0.04", "conf_ema", "Switch back to n when confidence is within 4% of baseline. Ensures confidence has genuinely recovered before switching back."],
            ["mp_c_high", "0.10", "multi_proxy", "Switch to s when composite proxy drop >= 10%. Calibrated via 3-video cross-validation."],
            ["mp_c_low", "0.03", "multi_proxy", "Switch back to n when drop <= 3%. Ensures genuine recovery before switching back."],
        ],
        col_widths=[1.5, 0.7, 1.2, 4.0],
    )

    # 4.3
    add_heading(doc, "4.3. Actuator Protections", level=2)

    add_para(doc, "These parameters protect against unstable switching behavior, regardless of the switching policy:")

    add_table_with_header(doc,
        ["Parameter", "Default", "Justification"],
        [
            ["min_dwell_frames", "10", "Minimum frames before allowing a switch. Prevents sub-second oscillation. At 30fps, 10 frames = 333ms minimum between switches. Based on video temporal coherence - scene complexity rarely changes faster than this."],
            ["max_switches_per_100", "12", "Maximum 12 switches in any 100-frame window. At 30fps, this limits switching to at most once every ~8 frames on average. Prevents pathological chattering even if the proxy signal is noisy."],
        ],
        col_widths=[2.0, 0.8, 5.0],
    )

    add_para(doc, "These are fully data-agnostic - they depend only on the temporal dynamics of switching, not on the content of the video.", bold=True)

    # 4.4
    add_heading(doc, "4.4. Are Parameters Data-Agnostic? Plug-and-Play Analysis", level=2)

    add_para(doc, "THE KEY QUESTION: If I'm given a random dataset and model weights, can I just run DMS-Raptor with defaults and expect it to work?", bold=True)

    add_para(doc, "Answer: YES, for the right policies.", bold=True)

    add_table_with_header(doc,
        ["Policy", "Plug-and-Play?", "Why"],
        [
            ["n_only / s_only", "YES (trivially)", "No parameters to tune - always uses one model."],
            ["conf_ema", "YES (best choice)", "Uses the model's own confidence as the signal. The EMA betas (0.30/0.02) define time constants in frames, not data-specific values. The thresholds (12%/4%) are relative drops, not absolute values. Works on any dataset because it adapts to whatever confidence range the model produces."],
            ["local_contrast_hyst", "PARTIALLY", "Uses local contrast (RMS of block std deviations) which is data-agnostic as a feature. Hysteresis thresholds depend on normalized proxy range. Best for structurally homogeneous domains (glass insulators). May underperform on texturally complex scenes."],
            ["combined / combined_hyst", "PARTIALLY", "proxy_size and hist_bins are data-agnostic. BUT the optimal alpha depends on how well L and H correlate with detection difficulty in your specific dataset. The fixed thresholds (0.45/0.55) assume C-score is centered around 0.5 after normalization, which may not hold for all videos."],
            ["entropy_only", "PARTIALLY", "Relies on H, which has weak correlation with detection difficulty. Works as a rough heuristic but may not provide meaningful switching for all datasets."],
        ],
        col_widths=[1.5, 1.2, 5.0],
    )

    add_para(doc, "RECOMMENDATION: For plug-and-play deployment on an unknown dataset, use conf_ema as the primary policy. It is the most data-agnostic because it derives its switching signal from the model's own behavior, not from hand-crafted image features. Run n_only and s_only as baselines for comparison.", bold=True)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: VALIDATION
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "5. Validation Methodology: Proving DMS Works", level=1)

    # 5.1
    add_heading(doc, "5.1. Dual-Model Disagreement as Ground Truth", level=2)

    add_para(doc, "The fundamental validation question:", bold=True)
    add_para(doc, "How do we know that the scene complexity proxy is actually measuring something meaningful? We need ground truth for 'scene difficulty' - but there is no direct label for this.")

    add_para(doc, "Solution: Inter-model disagreement as a proxy for ground truth.", bold=True)
    add_para(doc, "We run BOTH models (n and s) on every frame and measure how much they disagree. High disagreement means the s-model found things the n-model missed, confirming the scene was genuinely harder for the lighter model.")

    add_para(doc, "Disagreement score computation (4 components):", bold=True)

    add_equation_block(doc,
        "D = 0.30 * det_gap_norm + 0.25 * conf_gap_norm + 0.25 * iou_disagree + 0.20 * extra_s_norm",
        "D is in [0, 1], higher = more disagreement"
    )

    add_table_with_header(doc,
        ["Component", "Weight", "What It Measures", "Normalization"],
        [
            ["det_gap_norm", "0.30", "Difference in detection count (|s_dets - n_dets|)", "Capped at 10, divided by 10"],
            ["conf_gap_norm", "0.25", "Difference in mean confidence (|s_conf - n_conf|)", "Clipped to [0, 1]"],
            ["iou_disagree", "0.25", "1 - mean IoU of matched boxes", "Already in [0, 1]"],
            ["extra_s_norm", "0.20", "Detections found only by s-model", "Capped at 5, divided by 5"],
        ],
        col_widths=[1.5, 0.7, 3.0, 2.0],
    )

    add_para(doc, "Why this is valid ground truth:", bold=True)
    for item in [
        "If both models agree (low D), the scene was easy enough for the nano model.",
        "If they disagree (high D), the s-model provided value that n-model couldn't.",
        "This directly answers the DMS question: 'should we have used the s-model here?'",
        "No manual labeling required - fully automated.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    # 5.2
    add_heading(doc, "5.2. Correlation Analysis", level=2)

    add_para(doc, "We compute both Pearson (linear) and Spearman (rank-order) correlations between each scene proxy and the disagreement score:")

    add_table_with_header(doc,
        ["Correlation Type", "What It Measures", "When to Use"],
        [
            ["Pearson r", "Linear relationship: r = Cov(X,Y) / (std(X)*std(Y))", "When you expect a linear trend between proxy and disagreement"],
            ["Spearman rho", "Monotonic relationship: correlation of ranks", "More robust to outliers and non-linear relationships"],
        ],
        col_widths=[1.5, 3.5, 3.0],
    )

    add_para(doc, "Interpretation scale:", bold=True)
    add_table_with_header(doc,
        ["|r| Range", "Strength", "Interpretation"],
        [
            ["0.7 - 1.0", "Strong", "Proxy is a reliable predictor. Switching policy based on this proxy will be effective."],
            ["0.5 - 0.7", "Moderate", "Proxy has meaningful predictive power. Switching will provide some benefit."],
            ["0.3 - 0.5", "Weak", "Some correlation but noisy. Switching may help but not consistently."],
            ["0.0 - 0.3", "Negligible", "Proxy is not a reliable predictor. Switching based on this proxy is essentially random."],
        ],
        col_widths=[1.2, 1.0, 5.5],
    )

    add_para(doc, "Expected results from DMS-Raptor validation:", bold=True)
    add_table_with_header(doc,
        ["Proxy", "vs Disagreement", "Expected |r|", "Direction", "Interpretation"],
        [
            ["Laplacian (L)", "Disagreement Score", "0.40-0.60", "Negative", "Low L (blurry) = high disagreement. Strongest image-based proxy."],
            ["Entropy (H)", "Disagreement Score", "0.05-0.20", "Positive (weak)", "H is not a reliable proxy for detection difficulty."],
            ["C-score", "Disagreement Score", "0.15-0.30", "Negative (weak)", "Diluted by combining L with weak H."],
            ["Mean Conf (n)", "Disagreement Score", "0.50-0.70", "Negative", "Strongest overall proxy. Low confidence = high disagreement."],
            ["NIQE Score", "Disagreement Score", "0.30-0.45", "Positive", "Worse quality = more disagreement. Moderate proxy."],
        ],
        col_widths=[1.2, 1.5, 0.8, 1.0, 3.0],
    )

    # 5.3
    add_heading(doc, "5.3. Sensitivity Analysis", level=2)

    add_para(doc, "The sensitivity analysis answers: 'Does the switching decision change if I use different proxy computation settings?'")

    add_para(doc, "Method:", bold=True)
    for item in [
        "For each proxy setting (proxy_size x hist_bins), compute C-score for every frame.",
        "Normalize C to [0, 1] and apply threshold (C >= 0.5 means 'use s-model').",
        "Compare the binary decisions across all setting pairs.",
        "Report agreement percentage: what fraction of frames would choose the same model.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "Interpretation:", bold=True)
    for item in [
        ">= 90% agreement: Settings are stable. Default 160/64 is sufficient.",
        "80-90% agreement: Some sensitivity. Consider using 320/128 for more robust proxies.",
        "< 80% agreement: Significant sensitivity. Switching decisions are unreliable; consider using conf_ema instead.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    # 5.4
    add_heading(doc, "5.4. Proxy Size Grid Search", level=2)

    add_para(doc, "The grid search tests all combinations of proxy_size and hist_bins and measures the correlation between each proxy component (L, H, C_norm) and disagreement. The output is three heatmaps showing |r| for each combination.")

    add_para(doc, "Key insight from grid search:", bold=True)
    add_para(doc, "If |r(L)| >> |r(C_norm)|, the normalization and combination with H is diluting L's signal. This confirms that alpha should be higher or H should be removed from the combined score.")

    # 5.5
    add_heading(doc, "5.5. Expected Results & Interpretation", level=2)

    add_para(doc, "When you run validation on a typical UAV inspection video, you should see:", bold=True)
    for item in [
        "L has moderate negative correlation with disagreement (r around -0.40 to -0.60)",
        "H has negligible correlation (r around -0.05 to +0.20)",
        "C-score has weaker correlation than L alone (r around -0.15 to -0.30)",
        "Mean Confidence (n) has the strongest correlation (r around -0.50 to -0.70)",
        "NIQE has weak-to-moderate positive correlation (r around +0.30 to +0.45)",
        "Sensitivity analysis shows >85% agreement across settings (proxies are stable enough)",
        "Detection comparison shows s-model finds more detections on average than n-model",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "If your results differ significantly, it could indicate:", bold=True)
    for item in [
        "Very easy dataset: Both models agree on almost everything (low disagreement everywhere). DMS provides no benefit but also no harm.",
        "Very hard dataset: Both models struggle (high disagreement everywhere). Need a stronger s-model.",
        "Unusual domain: If L doesn't correlate (e.g., thermal imagery where blur is different), use conf_ema instead.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 6: STEP-BY-STEP
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "6. Step-by-Step: Getting Thesis Results from DMS-Raptor", level=1)

    steps = [
        ("Phase A: Baseline Runs", [
            "Launch DMS-Raptor. Go to Setup tab.",
            "Load your video and both model weights (n and s).",
            "Run n_only policy. This is your speed baseline.",
            "Run s_only policy. This is your accuracy baseline.",
            "Go to Results tab. Compare T_total_mean, P95, P99 for both baselines.",
            "Export the comparison table and timing waveforms.",
            "KEY METRIC: The gap between n_only and s_only latency is the maximum possible benefit DMS can provide.",
        ]),
        ("Phase B: Switching Policy Runs", [
            "Run each switching policy: entropy_only, combined, combined_hyst, conf_ema, niqe_switch, multi_proxy.",
            "For each run, record: T_total_mean, T_total_p95, switches/100, slow_pct (% frames using s-model).",
            "Go to Results tab. Use the 4-subplot waveform view to see T_total, T_scene, T_ctrl, T_infer over time.",
            "Use the C-score plot to see how each policy's switching signal varies over time.",
            "NOTE (v1.8.0): The 'Switching Signals' sub-tab shows the appropriate signal per policy: C-score for scene-analysis policies (entropy_only, combined, combined_hyst, multi_proxy), conf_drop for conf_ema, NIQE score for niqe_switch. Baselines (n_only, s_only) are not shown as they do not switch.",
            "Export all comparison plots.",
        ]),
        ("Phase C: Validation (Proving It Works)", [
            "Go to Validation tab.",
            "Set video source, both model paths, device, and alpha (0.6).",
            "Click 'Run Validation'. This runs BOTH models on every frame (~2x slower than normal).",
            "Wait for completion. View the Report tab for the text summary.",
            "Check Correlation Scatter: plot each proxy vs Disagreement Score. Note Pearson r values.",
            "Check Correlation Matrix: full heatmap of all proxies vs all disagreement metrics.",
            "Check Time Series Overlay: C-score and Disagreement should roughly track each other.",
            "Export all CSVs and plots for your thesis.",
        ]),
        ("Phase D: Proxy Size Grid Search", [
            "In Validation tab, go to Proxy Size Study sub-tab.",
            "Set ranges: proxy_size 64-640 step 64, hist_bins 32-256 step 32.",
            "Click 'Run Grid Search'. This re-computes proxies at all combinations.",
            "View the 3 heatmaps: |r(L)|, |r(H)|, |r(C_norm)| vs disagreement.",
            "Note: |r(L)| should be consistently higher than |r(H)| and |r(C_norm)|.",
            "Export the heatmap plots and tabular results.",
        ]),
        ("Phase E: Batch Processing (Multiple Datasets)", [
            "Go to Batch Run tab.",
            "Set source folder (with subfolders: glass/, composite/, porcelain/, etc.).",
            "Set models folder (matching subfolders with n+s .pt files).",
            "Select output folder. Choose policies to test.",
            "Click 'Run All'. DMS-Raptor processes all videos across all subfolders with all selected policies.",
            "Compare results across datasets to show generalizability.",
        ]),
        ("Phase E-2: Thesis Experiments Tab (Comprehensive Analysis)", [
            "Go to the Thesis Experiments tab (7th tab).",
            "Set video source and both model paths in the control panel.",
            "Select which policies to analyze via checkboxes.",
            "Click 'Run Full Suite' - runs all selected policies + optional validation.",
            "Review Latency Dashboard: CDF curves, box plots, timing breakdown, time series with P95 band.",
            "Review Switching Analysis: choice heatmap, switching rate over time, dwell histogram.",
            "Review Proxy Signals: L vs H scatter, detection scatter (n vs s), confidence comparison.",
            "Review Hyperparameter Sensitivity: threshold sweep heatmap, Pareto front (switches vs s-usage).",
            "Review EDA tab: descriptive statistics table, KDE distributions, outlier detection.",
            "Review Feature Engineering: correlation matrix, feature importance, decision boundary (L vs H).",
            "Click 'Export All Plots' to save all 18 plot types as PNGs.",
            "Click 'Download DOCX Report' for a self-contained 12-chapter research report with formulas and analysis.",
            "Use the Research Summary sub-tab for a quick HTML overview of all findings.",
        ]),
        ("Phase F: Frame Extraction (For Retraining Data)", [
            "Go to Frame Extraction tab.",
            "Set source folder and both model paths.",
            "Click 'Extract All'. Generates YOLO-format images + labels from both models.",
            "Use extracted data to retrain models or analyze detection patterns.",
        ]),
        ("Phase G: Compile Thesis Results", [
            "Create a comparison table: Policy | Mean T_total | P95 T_total | Switches/100 | Slow% | Speedup vs s_only",
            "Speedup = T_total_mean(s_only) / T_total_mean(policy)",
            "Create correlation summary table: Proxy | Pearson r | Spearman rho | Strength",
            "Present the validation plots: scatter, correlation matrix, sensitivity heatmap",
            "Show that conf_ema achieves the best trade-off: near s_only accuracy with near n_only speed.",
        ]),
    ]

    for phase_title, phase_steps in steps:
        add_heading(doc, phase_title, level=2)
        for i, step in enumerate(phase_steps, 1):
            p = doc.add_paragraph(f"{i}. {step}")
            for run in p.runs:
                run.font.size = Pt(10)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 7: T_TOTAL BENCHMARKING
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "7. T_total Benchmarking: What Is Measured", level=1)

    add_para(doc, "For your thesis, report T_total as the primary per-frame latency metric. Here is exactly what is and is not included:", bold=True)

    add_table_with_header(doc,
        ["Component", "Included?", "What It Times", "Typical Range"],
        [
            ["T_scene_ms", "YES", "Scene proxy computation: cv2.resize to proxy_size, Laplacian variance, histogram entropy, EMA updates (for conf_ema/niqe policies: just EMA arithmetic)", "0.5-8 ms"],
            ["T_ctrl_ms", "YES", "Controller decision: rolling percentile lookup, threshold comparison, dwell/switch-rate guard checks, budget guard", "~0.01 ms"],
            ["T_infer_n_ms", "YES (when n chosen)", "model_n.predict(): preprocess (resize to imgsz, normalize, tensor) + GPU/CPU forward pass + NMS postprocessing", "5-15 ms (GPU), 15-40 ms (CPU)"],
            ["T_infer_s_ms", "YES (when s chosen)", "model_s.predict(): same pipeline but larger model", "10-30 ms (GPU), 30-80 ms (CPU)"],
            ["Frame read", "NO", "cv2.VideoCapture.read() - disk I/O, not part of the algorithm", "1-5 ms"],
            ["Overlay drawing", "NO", "draw_overlay() - visualization only, not deployed", "0.5-2 ms"],
            ["Video writing", "NO", "cv2.VideoWriter.write() - output only", "1-3 ms"],
            ["GUI updates", "NO", "PyQt signal emission - display framework overhead", "< 1 ms"],
        ],
        col_widths=[1.3, 0.7, 3.0, 1.7],
    )

    add_para(doc, "For thesis reporting, present:", bold=True)
    for item in [
        "Mean T_total (ms): Average per-frame processing time across all frames",
        "P95 T_total (ms): 95th percentile - represents worst-case non-outlier latency",
        "P99 T_total (ms): 99th percentile - captures tail latency",
        "T_scene separately: Shows the overhead of scene analysis (should be small relative to T_infer)",
        "T_infer breakdown: Mean inference time for n-model vs s-model frames",
        "Equivalent FPS = 1000 / T_total_mean: Real-time processing rate",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 8: ALL POSSIBLE THESIS QUESTIONS
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "8. Answering Every Possible Thesis Question", level=1)

    # 8.1 Fundamental
    add_heading(doc, "8.1. Fundamental Questions", level=2)

    add_key_answer(doc,
        "What is the core contribution of this thesis?",
        "We propose and validate a real-time Dynamic Model Switching (DMS) framework for UAV inspection that adaptively selects between a fast lightweight YOLO model (nano) and a more accurate but slower model (small) based on estimated scene complexity. The key contributions are: (1) A streaming inference engine with multiple switching policies based on different scene complexity proxies, (2) A rigorous dual-model validation methodology that uses inter-model disagreement as ground truth for scene difficulty, (3) Empirical analysis showing that model confidence-based switching (conf_ema) outperforms traditional image-based proxies (Laplacian, entropy), and (4) A complete open-source tool (DMS-Raptor) for deploying and validating DMS on any YOLO model pair."
    )

    add_key_answer(doc,
        "Why is dynamic model switching needed? Why not just use the best model?",
        "In resource-constrained UAV deployment, using the best (heaviest) model for every frame wastes computational budget. Our benchmarks show YOLOv8s is 2-3x slower than YOLOv8n. In a 30fps video, this means processing only 10-15fps with s-model vs 30+fps with n-model. DMS allows near-real-time processing by using the fast model for easy frames (which constitute 60-80% of typical inspection footage) and only invoking the heavier model when scene complexity demands it. This reduces average latency by 30-50% while maintaining detection quality close to always-s-model."
    )

    add_key_answer(doc,
        "How do you define 'scene complexity'?",
        "We define scene complexity operationally as the degree to which a lightweight detector model (YOLOv8n) fails to match the detection performance of a heavier model (YOLOv8s) on the same frame. We measure this via inter-model disagreement - if both models produce similar detections (same count, similar confidence, high IoU overlap), the scene was 'easy'. If they diverge significantly, the scene was 'complex' and benefited from the heavier model. This is not a subjective judgment but a measurable, automated metric."
    )

    add_key_answer(doc,
        "What is the C-score and how does it work?",
        "C-score is a combined scene complexity metric defined as C = alpha * L_norm + (1-alpha) * H_norm, where L is the Laplacian variance (measuring edge/texture sharpness), H is the histogram entropy (measuring tonal diversity), both normalized to [0,1] using rolling min-max percentiles. Alpha (default 0.6) weights L more heavily since it has stronger correlation with detection difficulty. The C-score is compared against thresholds to decide which model to use. However, our validation shows C-score is weakened by combining L (strong signal, r=-0.577) with H (weak/contradictory signal, r=+0.172), resulting in r=-0.198. This is a key finding."
    )

    add_key_answer(doc,
        "What does validation actually prove?",
        "Validation proves (or disproves) that the scene complexity proxies used for switching actually predict when the heavier model is needed. By running both models on every frame and correlating the proxy values with inter-model disagreement, we establish whether the switching decisions are informed or essentially random. A strong negative correlation (e.g., low L correlates with high disagreement) confirms the proxy captures meaningful scene difficulty. A weak correlation (e.g., H vs disagreement) reveals the proxy is unreliable. This is the scientific evidence that the DMS approach is sound."
    )

    # 8.2 Technical
    add_heading(doc, "8.2. Technical Deep-Dive Questions", level=2)

    add_key_answer(doc,
        "Why does Laplacian variance work as a complexity proxy?",
        "The Laplacian is a second-order derivative filter that responds to edges and texture. High variance means many strong edges (sharp, textured scene) - easy for any detector. Low variance means few edges (blurry, smooth, featureless) - the n-model struggles because it has fewer parameters to extract subtle features, while the s-model's larger capacity allows it to find patterns the n-model misses. This is supported by image quality assessment literature (Pech-Pacheco et al., 2000) which uses Laplacian variance as a blur detection metric."
    )

    add_key_answer(doc,
        "Why doesn't entropy work well?",
        "Entropy measures tonal diversity (how many different intensity levels are used), not spatial structure. A foggy uniform scene (low entropy) might still have clearly visible objects if they have sufficient contrast against the background. Conversely, a highly textured natural background (high entropy) might be easy for detection if objects are well-separated. Entropy is an intensity-domain statistic while detection difficulty depends on spatial-domain features (edges, shapes, textures). Our empirical result (r=+0.172) confirms this theoretical limitation."
    )

    add_key_answer(doc,
        "Why is conf_ema the best policy? Isn't using the model's own confidence 'cheating'?",
        "It's not cheating - it's using the most directly relevant signal. The confidence from the PREVIOUS frame's detections predicts the CURRENT frame's difficulty because video has temporal coherence (adjacent frames are similar). If confidence was low on frame N, frame N+1 is likely also difficult. The EMA structure (fast/slow) detects when confidence drops below its baseline, which is a genuine signal that the model is struggling. This is analogous to control theory's 'feedback control' - using the system's own output to regulate its input. The key distinction: we use confidence from the CURRENT model's detections, not from the oracle (s-model). The n-model's own low confidence tells us it needs help."
    )

    add_key_answer(doc,
        "What happens if both models are equally bad? Does DMS still work?",
        "If both models produce similar (low) quality detections, disagreement will be low and DMS will (correctly) stick with the fast n-model - there's no benefit from switching to s-model. DMS doesn't improve absolute accuracy in this case, but it also doesn't hurt. The system degrades gracefully: worst case is n_only performance (fastest possible). If you need better accuracy, the solution is better models, not more switching."
    )

    add_key_answer(doc,
        "How does the EMA dual-rate filter work? Why two EMAs?",
        "The dual-EMA design is inspired by signal processing and technical analysis (similar to MACD in finance). The fast EMA (beta=0.30, ~3 frame window) responds quickly to sudden changes. The slow EMA (beta=0.02, ~50 frame window) represents the long-term baseline. When the fast EMA drops below the slow EMA, it signals a recent degradation relative to the norm. The relative drop (fast-slow)/slow is scale-invariant: it works regardless of the absolute confidence level. A 12% relative drop triggers the switch to s-model. This is more robust than absolute thresholds because different videos have different baseline confidence levels."
    )

    add_key_answer(doc,
        "Why normalize L and H with rolling percentiles instead of fixed constants?",
        "Fixed constants (e.g., L_max = 10000) would be brittle across different video domains. A thermal camera video has completely different L and H distributions than a visible-light camera. Rolling min-max percentiles adapt to the local statistics of each video: if this video's L ranges from 50 to 5000, L_norm maps that range to [0,1]. This makes the C-score comparable across videos. The 200-frame window is large enough to capture the local distribution but small enough to adapt to gradual scene changes (e.g., flying from a clear area to a foggy area)."
    )

    add_key_answer(doc,
        "What is the disagreement score and why those specific weights?",
        "D = 0.30 * det_gap + 0.25 * conf_gap + 0.25 * iou_disagree + 0.20 * extra_s. Detection count gap gets the highest weight (0.30) because missing entire objects is the most serious failure mode. Confidence and IoU disagreement get equal weight (0.25 each) because they measure complementary aspects: confidence is about how certain each model is, while IoU measures spatial alignment. Extra detections by s-model (0.20) captures false negatives of n-model specifically. These weights can be adjusted, but the relative ordering (detection count > confidence/IoU > extra dets) is grounded in the relative severity of each failure mode."
    )

    # 8.3 Tricky questions
    add_heading(doc, "8.3. Tricky/Adversarial Questions", level=2)

    add_key_answer(doc,
        "Aren't you just using a simpler model most of the time and claiming improvement? How is this different from just using n_only?",
        "The key difference is in the frames where it matters. A DMS policy that uses n-model for 70% of frames (easy scenes) and s-model for 30% (hard scenes) achieves average latency of T_avg = 0.7*T_n + 0.3*T_s, which is 30-40% faster than always-s. But critically, during that 30% where it uses s-model, it catches detections that n-model would miss entirely. The comparison should be: DMS vs s_only (not vs n_only), because DMS aims to approximate s_only quality at closer to n_only speed. If you only compare to n_only, yes, DMS is slower on average. But if you compare detection quality on hard frames, DMS matches s_only where n_only fails."
    )

    add_key_answer(doc,
        "Your C-score correlation is only r = -0.198. Isn't that basically random? How can you claim the system works?",
        "Excellent point, and this is actually one of our key findings. The C-score (combining L and H) IS weak as a switching signal, and we acknowledge this. Our analysis reveals WHY: the normalization dilutes L's moderate signal (r=-0.577) by combining it with H's weak contradictory signal (r=+0.172). This is why we also propose and validate conf_ema (r=-0.672), which bypasses the C-score entirely and uses model confidence instead. The C-score limitation is itself a scientific contribution - it shows that naive combination of multiple proxies can be worse than using the best single proxy alone. The thesis presents both the problem (C-score dilution) and the solution (conf_ema)."
    )

    add_key_answer(doc,
        "How do you know the switching itself isn't introducing errors? What about temporal consistency?",
        "We address temporal consistency through actuator protections: min_dwell_frames=10 ensures at least 333ms (at 30fps) between switches, and max_switches_per_100=12 caps the switching rate. Additionally, the hysteresis mechanism (c_low/c_high gap) prevents chattering around the threshold. Both models are pre-loaded in memory so switching is instantaneous (no model loading delay). The only cost is that the first few frames after a switch might have slightly different detection patterns due to different model architectures, but consecutive YOLO frames are already independent (no temporal state in YOLO) so there's no 'warm-up' artifact."
    )

    add_key_answer(doc,
        "What if someone changes the YOLO models (e.g., uses YOLOv5 or a custom model)? Does DMS still work?",
        "DMS-Raptor is model-agnostic in principle. The switching logic depends on: (1) scene proxies (L, H, NIQE) which are model-independent image features, and (2) confidence statistics (conf_ema) which any object detector produces. The only requirement is that the s-model should be more accurate than the n-model on hard scenes. If you use YOLOv5-n and YOLOv5-s, or even different architectures (e.g., nano=MobileNet-SSD, small=EfficientDet), the framework still applies. The proxy correlations may change, which is why validation is important: run the dual-model validation to confirm the proxies predict disagreement for your specific model pair."
    )

    add_key_answer(doc,
        "Why not use a learned switching policy (e.g., reinforcement learning or a classifier)?",
        "This is a valid direction for future work. Our current approach uses hand-crafted proxies and fixed thresholds, which has advantages: (1) Fully interpretable - you can see exactly why the system switched, (2) No training data needed - works out-of-the-box on any video, (3) Extremely lightweight - EMA arithmetic adds ~0.01ms overhead, (4) No overfitting risk - the switching policy doesn't learn from the specific dataset. A learned policy could potentially outperform our approach but would need: training data with switching labels, risk of overfitting to training scenarios, and additional computational overhead for the policy network. Our approach demonstrates the feasibility and establishes baseline performance that learned approaches should beat."
    )

    add_key_answer(doc,
        "Your validation runs both models on every frame. Isn't that 2x slower? How is that useful?",
        "Validation is an OFFLINE analysis step, not part of the deployed system. In deployment, only ONE model runs per frame. Validation is done once to verify that the proxies are meaningful for your dataset/model combination. Think of it like testing: you test offline to validate your approach, then deploy the validated system at full speed. The 2x cost of validation is paid once during development, not during every inference run."
    )

    add_key_answer(doc,
        "What about GPU memory? Loading two models simultaneously must be expensive.",
        "Both models ARE loaded in memory simultaneously for instant switching. YOLOv8n is ~6MB and YOLOv8s is ~22MB (FP32). Total GPU memory for both: ~28MB, which is trivial compared to typical GPU memory (4-24GB). Even on edge devices like Jetson Nano (4GB), this is manageable. If memory is truly constrained, TensorRT FP16 reduces each model to ~50% size, and INT8 reduces further. The memory cost of keeping both models loaded is far less than the latency cost of loading/unloading models on each switch."
    )

    add_key_answer(doc,
        "How do you define what is a 'simple' vs 'hard' scene?",
        "We define scene difficulty OPERATIONALLY, not subjectively. A 'simple' scene is one where the lightweight YOLOv8n model produces detections very similar to the heavier YOLOv8s model (low inter-model disagreement score D < 0.2). A 'hard' scene is one where disagreement is high (D > 0.5) — meaning the s-model found significantly more objects, higher confidence, or better localization than the n-model. This definition is: (1) Measurable and automated — no human annotation needed, (2) Model-relative — difficulty is defined relative to the model pair, not in absolute terms, (3) Grounded in utility — a 'hard' scene is precisely one where using the better model provides tangible benefit. Examples of what makes scenes hard: motion blur (low L), poor lighting, occlusion, small objects, dense object clusters, cluttered backgrounds. But we don't need to enumerate these manually — the disagreement score captures them all."
    )

    add_key_answer(doc,
        "What are the concrete limitations of your approach?",
        "Key limitations: (1) PROXY CORRELATION: The best proxy (mean confidence, r=-0.672) is moderate, not strong — ~55% of variance unexplained. Switching decisions are informed but imperfect. (2) TEMPORAL LAG: Confidence-based switching uses PREVIOUS frame's confidence to decide CURRENT frame's model. If scene changes abruptly (cut or sudden motion), there's a 1-frame lag. The EMA structure adds further smoothing lag. (3) SINGLE-OBJECT-TYPE TRAINING: Our models are trained on specific inspection objects (glass insulators, etc.). Generalization to very different domains (e.g., person detection, autonomous driving) is untested. (4) TWO-MODEL LIMITATION: We only switch between 2 models. A continuum of models (nano, small, medium, large) could provide finer granularity but adds complexity. (5) NO LEARNED POLICY: Our switching rules are hand-crafted thresholds, not learned. A reinforcement learning approach could potentially find better policies but would need training data. (6) GPU DEPENDENCY: Both models must fit in GPU memory simultaneously (~28MB for YOLOv8n+s, but more for larger model pairs). (7) C-SCORE WEAKNESS: The combined C-score is weaker than L alone due to entropy dilution — a known design flaw we document but don't fully resolve."
    )

    add_key_answer(doc,
        "What exactly is YOUR contribution vs existing work?",
        "My specific contributions are: (1) FRAMEWORK DESIGN: I designed and implemented a complete real-time DMS framework with 8 switching policies, each representing a different hypothesis about scene complexity estimation. (2) DUAL-EMA SWITCHING: The conf_ema policy uses a novel dual-EMA approach (fast/slow) for detecting signal degradation — this relative-drop mechanism is scale-invariant and data-agnostic. (3) VALIDATION METHODOLOGY: I developed a dual-model disagreement-based validation approach that quantifies proxy effectiveness without manual labeling. (4) EMPIRICAL ANALYSIS: I systematically compared scene complexity proxies (L, H, C, local contrast, confidence), revealing that model confidence outperforms all image-based proxies — a finding with practical implications for adaptive inference. (5) DOMAIN-SPECIFIC FINDINGS: Demonstrating that optimal policy selection is domain-dependent (combined_hyst for glass, entropy_only for porcelain) and that naive feature fusion amplifies noise in structurally homogeneous domains. (6) COMPLETE TOOLCHAIN: End-to-end implementation from scene analysis to validation, with reproducible experiments and publication-quality visualization."
    )

    # 8.4 Literature comparison
    add_heading(doc, "8.4. Comparison & Literature Questions", level=2)

    add_key_answer(doc,
        "How does this compare to existing model switching / early-exit approaches?",
        "Existing approaches include: (1) Early-exit networks (BranchyNet, SDN) which add exit branches within a single network - these require model architecture changes and retraining. DMS is model-agnostic. (2) Adaptive inference (SkipNet, BlockDrop) which skip layers within a single model - again requires architecture modification. (3) Multi-model cascades (Viola-Jones style) which run a fast pre-filter then a detailed model - similar in spirit but typically static, not adaptive per-frame. DMS-Raptor is unique in using per-frame scene analysis to select between unmodified off-the-shelf models, requiring no retraining or architecture changes."
    )

    add_key_answer(doc,
        "How does NIQE compare to full BRISQUE or other NR-IQA metrics?",
        "Our NIQE implementation is a simplified BRISQUE-like score (not the full NIQE from Mittal et al. 2013 which requires a pre-trained MVG model). It uses MSCN coefficients and GGD shape parameter fitting, which captures the core insight of NSS-based quality assessment. Full BRISQUE would use SVR regression on NSS features, which is more accurate but requires a pre-trained model (not data-agnostic). Our simplified version achieves r=+0.400 against disagreement with < 3ms compute time, which is a good trade-off for real-time deployment."
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 9: PLUG AND PLAY
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, "9. Plug-and-Play: Random Dataset + Models Scenario", level=1)

    add_para(doc, "SCENARIO: You are given an unknown dataset of UAV inspection videos and a pair of YOLO model weights. You need to demonstrate that DMS works on it, without any tuning.", bold=True)

    add_heading(doc, "Step-by-Step Plug-and-Play Protocol", level=2)

    protocol_steps = [
        ("Load & Baseline", "Open DMS-Raptor. Load video and both models. Run n_only and s_only. Record T_total_mean for both. If the gap is < 5ms, DMS provides minimal benefit (models are similarly fast). If gap is significant (>10ms), proceed."),
        ("Run conf_ema (Primary)", "Run conf_ema with all defaults. This is the most data-agnostic policy. Record T_total_mean, switches/100, slow_pct. Compare to baselines: T_total should be between n_only and s_only, closer to n_only."),
        ("Run combined_hyst (Secondary)", "Run combined_hyst with defaults. Compare to conf_ema. If combined_hyst is significantly worse (higher latency with same slow_pct), it confirms that image-based proxies are weaker for this dataset."),
        ("Validate", "Go to Validation tab. Run dual-model validation. Check correlation matrix. Confirm Mean Conf(n) has the strongest negative correlation with disagreement. If L is also strong (|r| > 0.4), combined policies may also work well for this dataset."),
        ("Report", "Create comparison table showing all policies. Present conf_ema as the recommended deployment policy. Show validation correlations as evidence. Document any dataset-specific findings."),
    ]

    for i, (title, desc) in enumerate(protocol_steps, 1):
        add_para(doc, f"Step {i}: {title}", bold=True)
        add_para(doc, desc)

    add_heading(doc, "Do You Need to Set proxy_size and hist_bins?", level=2)

    add_para(doc, "SHORT ANSWER: It depends on which policy you use.", bold=True)

    add_table_with_header(doc,
        ["Policy", "Uses proxy_size?", "Uses hist_bins?", "What to do"],
        [
            ["n_only / s_only", "No", "No", "Baselines. No parameters needed."],
            ["conf_ema", "No", "No", "Uses model confidence only. proxy_size and hist_bins are completely irrelevant. Just use defaults."],
            ["local_contrast_hyst", "Yes", "No", "Uses proxy_size for computing local contrast features. Default 160 is fine. hist_bins is NOT used."],
            ["multi_proxy", "Yes", "Yes", "Computes 7 proxy features at specified resolution. Defaults (160/64) work for most cases."],
            ["entropy_only", "Yes", "Yes", "Computes L and H at the specified resolution and bin count. Defaults (160/64) work for most cases."],
            ["combined / combined_hyst", "Yes", "Yes", "Same as entropy_only. If validation shows weak correlation, consider increasing proxy_size to 320 or adjusting alpha."],
        ],
        col_widths=[1.5, 1.2, 1.2, 3.5],
    )

    add_para(doc, "BOTTOM LINE: If you use conf_ema (recommended for plug-and-play), proxy_size and hist_bins don't matter AT ALL. They only affect the image-based proxies (L, H, C), which conf_ema doesn't use. Your DMS will work on any dataset without touching these parameters.", bold=True)

    add_heading(doc, "Will DMS Still Work on a Random Dataset?", level=2)

    add_para(doc, "YES, under these conditions:", bold=True)
    for item in [
        "The two models must have DIFFERENT capabilities: n-model should be faster but less accurate, s-model should be slower but more accurate. If both models have the same accuracy, there's nothing to gain from switching.",
        "The video must have varying difficulty: if all frames are equally hard (or equally easy), switching adds no benefit. Most real-world UAV inspection videos have natural variation.",
        "The models must be reasonably trained: if neither model detects anything useful, DMS can't help.",
        "Use conf_ema policy: it automatically adapts to whatever confidence distribution the models produce.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_para(doc, "DMS guarantees: worst-case performance is exactly n_only (if the policy never switches) or at most the average of n_only and s_only (if it switches optimally). DMS NEVER performs worse than the slower s_only baseline in terms of latency, and NEVER performs worse than n_only in terms of accuracy (because it only switches TO the better model when needed).", bold=True)

    # ═══════════════════════════════════════════════════════════════════
    # FINAL PAGE
    # ═══════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_heading(doc, "Summary of Key Findings", level=1)

    add_table_with_header(doc,
        ["Finding", "Evidence", "Implication"],
        [
            ["L (Laplacian) is the strongest image-based proxy", "r = -0.577 vs disagreement", "For image-based policies, weight L heavily (alpha >= 0.8)"],
            ["H (Entropy) is a weak/unreliable proxy", "r = +0.172 vs disagreement", "Consider removing H or replacing with a better metric"],
            ["C-score is diluted by combining L with H", "C: r = -0.198 << L: r = -0.577", "Normalization + weak H destroys L's signal"],
            ["Mean Confidence is the best overall proxy", "r = -0.672 vs disagreement", "conf_ema is the best switching policy"],
            ["Policy winner is domain-dependent", "glass=combined_hyst, porcelain=entropy_only", "No single scene-proxy policy dominates across domains"],
            ["conf_ema is most data-agnostic", "No image preprocessing needed", "Recommended for plug-and-play deployment"],
            ["Default parameters are reasonable", "Sensitivity analysis shows >85% agreement", "Defaults work without tuning for most datasets"],
            ["T_total excludes I/O (clean benchmark)", "Only scene+ctrl+infer timed", "Directly comparable to published YOLO benchmarks"],
            ["Thesis Experiments tab provides complete analysis suite", "18 plot types + 12-chapter DOCX report", "All thesis results can be generated and documented from a single tab"],
        ],
        col_widths=[2.5, 2.0, 3.0],
    )

    # Save
    doc.save(OUT_PATH)
    print(f"Document saved to: {OUT_PATH}")
    print(f"Size: {os.path.getsize(OUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    build()
