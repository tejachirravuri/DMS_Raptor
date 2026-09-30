"""
Generate comprehensive thesis supervisor handout document.

Presents the dynamic model switching research methodology and findings
as a research framework — no mention of any GUI application.
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
    "Thesis_Supervisor_Handout.docx",
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


def add_bullet(doc, text, bold_prefix="", indent_level=0):
    """Add a bullet point paragraph."""
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        b_run = p.add_run(bold_prefix)
        b_run.bold = True
        b_run.font.size = Pt(11)
    run = p.add_run(text)
    run.font.size = Pt(11)
    if indent_level > 0:
        p.paragraph_format.left_indent = Inches(0.3 * indent_level)
    return p


def add_page_break(doc):
    doc.add_page_break()


# ── Main Document ────────────────────────────────────────────────────────

def build_document():
    doc = Document()

    # ── Default font ──
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)
    font.color.rgb = RGBColor(0, 0, 0)

    # ── Margins ──
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ================================================================
    # TITLE PAGE
    # ================================================================
    for _ in range(6):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Dynamic Model Switching\nfor Real-Time UAV Inspection")
    run.font.size = Pt(28)
    run.font.color.rgb = RGBColor(0, 51, 102)
    run.bold = True

    doc.add_paragraph()

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Thesis Research Summary & Methodology Handout")
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(80, 80, 80)
    run.italic = True

    doc.add_paragraph()
    doc.add_paragraph()

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = author.add_run("Teja Chirravuri")
    run.font.size = Pt(14)
    run.bold = True

    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_p.add_run("March 2026")
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(100, 100, 100)

    add_page_break(doc)

    # ================================================================
    # TABLE OF CONTENTS
    # ================================================================
    add_heading(doc, "Table of Contents", level=1)
    doc.add_paragraph()

    toc_items = [
        ("Chapter 1", "Research Overview"),
        ("Chapter 2", "Defining Scene Complexity — Simple vs Hard Scenes"),
        ("Chapter 3", "Scene Complexity Proxies — How We Estimate Difficulty"),
        ("Chapter 4", "Switching Policies — The Decision Logic"),
        ("Chapter 5", "The Per-Frame Processing Pipeline"),
        ("Chapter 6", "Validation Methodology"),
        ("Chapter 7", "Step-by-Step — How to Reproduce Results"),
        ("Chapter 8", "Key Findings and Discussion"),
        ("Chapter 9", "Limitations and Future Work"),
        ("Chapter 10", "Answering Supervisor Questions"),
        ("Chapter 11", "Mathematical Foundations Reference"),
        ("Chapter 12", "Summary & Conclusions"),
    ]

    for num, title_text in toc_items:
        p = doc.add_paragraph()
        r1 = p.add_run(num + ":  ")
        r1.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = RGBColor(0, 51, 102)
        r2 = p.add_run(title_text)
        r2.font.size = Pt(11)

    add_page_break(doc)

    # ================================================================
    # CHAPTER 1: RESEARCH OVERVIEW
    # ================================================================
    add_heading(doc, "Chapter 1: Research Overview", level=1)

    add_heading(doc, "1.1  Problem Statement", level=2)
    add_para(doc,
        "Unmanned Aerial Vehicle (UAV) inspection of infrastructure — bridges, wind turbines, "
        "solar panels, pipelines — increasingly relies on automated object detection to identify "
        "defects such as cracks, corrosion, insulation damage, and structural deformation. "
        "State-of-the-art detectors based on the YOLO family deliver excellent accuracy, but "
        "heavier model variants (e.g., YOLOv8s, YOLOv8m) incur substantial inference latency "
        "that can make frame-by-frame processing infeasible at video rate on embedded hardware. "
        "Conversely, lightweight variants (e.g., YOLOv8n) are fast enough for real-time use but "
        "sacrifice detection recall on challenging frames — exactly the frames where detection "
        "matters most."
    )
    add_para(doc,
        "The fundamental tension is clear: running the heavier model on every frame wastes "
        "compute on easy scenes, while running only the lighter model misses detections on "
        "hard scenes. The challenge is to get the best of both worlds."
    )

    add_heading(doc, "1.2  Core Hypothesis", level=2)
    add_para(doc,
        "Not all video frames require the same model capacity. Scene complexity varies "
        "continuously throughout a UAV inspection flight — long stretches of clear, well-lit "
        "surfaces are interspersed with brief episodes of motion blur, poor lighting, partial "
        "occlusion, or cluttered backgrounds. If we can estimate scene complexity cheaply "
        "(in under 5 ms), we can route easy frames to the fast model and hard frames to the "
        "accurate model, achieving near-best-model accuracy at near-fastest-model speed."
    )

    add_heading(doc, "1.3  Research Question", level=2)
    add_para(doc,
        "Can we build a real-time framework that adaptively switches between a fast lightweight "
        "detector (YOLOv8n) and a more accurate but slower detector (YOLOv8s), based on "
        "estimated scene complexity, such that the combined system achieves:",
        bold=True,
    )
    add_bullet(doc, "Detection quality approaching the heavier model (s_only baseline)")
    add_bullet(doc, "Mean inference latency approaching the lighter model (n_only baseline)")
    add_bullet(doc, "Switching behavior that is temporally stable (no oscillation)")
    add_bullet(doc, "Parameters that are reasonably data-agnostic across inspection scenarios")

    add_heading(doc, "1.4  Scope", level=2)
    add_para(doc,
        "The proposed framework is evaluated using YOLOv8n (nano, ~3.2M parameters) vs "
        "YOLOv8s (small, ~11.2M parameters) on UAV inspection footage. The model pair was "
        "chosen because: (a) they share the same architecture family, differing only in depth "
        "and width multipliers, which isolates the capacity variable; (b) YOLOv8 is the "
        "current state-of-the-art single-stage detector widely used in industrial inspection; "
        "(c) the nano-to-small gap is the most practically relevant — it represents the "
        "smallest step up that yields a meaningful accuracy improvement while remaining "
        "deployable on edge hardware."
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 2: DEFINING SCENE COMPLEXITY
    # ================================================================
    add_heading(doc, "Chapter 2: Defining Scene Complexity — Simple vs Hard Scenes", level=1)

    add_para(doc,
        "This chapter provides the formal, operational definition of scene difficulty that "
        "underpins the entire framework. The definition is fully automated, reproducible, "
        "and grounded in measurable inter-model behavior rather than subjective human judgment.",
        italic=True,
    )

    add_heading(doc, "2.1  Operational Definition", level=2)
    add_para(doc,
        "We define scene difficulty operationally using inter-model disagreement. The key "
        "insight is: if a lightweight model and a heavier model produce nearly identical "
        "detections on a given frame, then the heavier model was not needed — the scene was "
        "\"simple\" for this model pair. Conversely, if the heavier model finds substantially "
        "more objects, produces higher confidence, or localizes objects differently, then the "
        "scene was \"hard\" — the extra model capacity was necessary."
    )

    add_heading(doc, "2.2  Simple Scenes (D < 0.2)", level=2)
    add_para(doc,
        "A simple scene is one where the lightweight model (YOLOv8n) and the heavier model "
        "(YOLOv8s) produce very similar detection outputs. Specifically, the disagreement "
        "score D is below 0.2, indicating:"
    )
    add_bullet(doc, "Both models detect approximately the same number of objects")
    add_bullet(doc, "Confidence scores are similar across matched detections")
    add_bullet(doc, "Bounding box locations overlap substantially (high IoU)")
    add_bullet(doc, "The heavier model finds few or no additional objects that the lighter model missed")

    add_para(doc, "Typical visual characteristics of simple scenes:", bold=True)
    add_bullet(doc, "Clear, unobstructed view of the inspection surface")
    add_bullet(doc, "Good, even lighting conditions")
    add_bullet(doc, "Well-separated objects with sufficient spacing")
    add_bullet(doc, "Sharp focus, minimal motion blur")
    add_bullet(doc, "High contrast between defects and background")
    add_bullet(doc, "Textured surfaces that provide strong edge features")

    add_heading(doc, "2.3  Hard Scenes (D > 0.5)", level=2)
    add_para(doc,
        "A hard scene exhibits significant inter-model disagreement (D > 0.5), meaning "
        "the heavier model produces substantially different — and presumably better — "
        "detection output compared to the lightweight model. This indicates that the extra "
        "model capacity was genuinely needed."
    )
    add_para(doc, "Typical visual characteristics of hard scenes:", bold=True)
    add_bullet(doc, "Motion blur from UAV movement or wind-induced vibration")
    add_bullet(doc, "Poor or uneven lighting (shadows, glare, underexposure)")
    add_bullet(doc, "Partial occlusion of objects by structural elements")
    add_bullet(doc, "Small or distant objects near the detection size limit")
    add_bullet(doc, "Cluttered backgrounds with many texture-similar distractors")
    add_bullet(doc, "Low contrast between defects and surrounding surface")
    add_bullet(doc, "Atmospheric interference (haze, rain, condensation)")

    add_heading(doc, "2.4  The Disagreement Score D", level=2)
    add_para(doc,
        "The disagreement score D is a composite metric computed by comparing the detection "
        "outputs of both models on the same frame. It ranges from 0.0 (perfect agreement) "
        "to 1.0 (maximum disagreement)."
    )

    add_equation_block(doc,
        "D = 0.30 * det_gap_norm + 0.25 * conf_gap_norm + 0.25 * iou_disagree + 0.20 * extra_s_norm",
        "Weighted composite disagreement score"
    )

    add_para(doc, "Component breakdown:", bold=True)
    doc.add_paragraph()

    add_para(doc, "det_gap_norm  (weight 0.30) — Detection Count Gap", bold=True)
    add_para(doc,
        "Measures how many more objects the heavier model detected compared to the lighter "
        "model. Computed as: |N_s - N_n| / 10, capped at 1.0. A value of 0 means both models "
        "found the same number of objects. A value of 1.0 means the heavier model found 10 or "
        "more additional detections. This is the most heavily weighted component because a "
        "detection count difference is the clearest indicator that the heavier model was needed."
    )

    add_para(doc, "conf_gap_norm  (weight 0.25) — Confidence Gap", bold=True)
    add_para(doc,
        "Measures the difference in mean detection confidence between the two models. "
        "Computed as: max(0, mean_conf_s - mean_conf_n). Ranges from 0.0 to ~0.5 in practice. "
        "When the heavier model is substantially more confident in its detections, it suggests "
        "the scene contains ambiguous features that the lighter model cannot resolve with "
        "certainty."
    )

    add_para(doc, "iou_disagree  (weight 0.25) — Localization Disagreement", bold=True)
    add_para(doc,
        "Measures how much the bounding box predictions differ between matched detections. "
        "Computed as: 1.0 - mean_IoU across all matched detection pairs. An IoU disagreement "
        "of 0 means the bounding boxes are identical; a value near 1.0 means they barely "
        "overlap. Hungarian matching is used to pair detections between the two models."
    )

    add_para(doc, "extra_s_norm  (weight 0.20) — Extra Detections by Heavier Model", bold=True)
    add_para(doc,
        "Counts the number of detections made only by the heavier model (unmatched after "
        "Hungarian matching). Computed as: N_extra_s / 5, capped at 1.0. This directly "
        "measures the heavier model's ability to find objects that the lighter model missed "
        "entirely — the most operationally relevant indicator for inspection applications."
    )

    add_heading(doc, "2.5  Why This Definition Is Scientifically Valid", level=2)
    add_para(doc, "The operational definition of scene difficulty via inter-model disagreement "
        "satisfies several important scientific criteria:"
    )
    add_bullet(doc, "No manual labeling needed: ",
        bold_prefix="Fully automated — ")
    add_bullet(doc,
        "The metric directly answers the core question: 'Was the heavier model actually "
        "needed for this frame?' This is precisely the information the switching policy needs.",
        bold_prefix="Operationally relevant — ")
    add_bullet(doc,
        "Difficulty is defined relative to the specific model pair being used. A frame that "
        "is 'hard' for nano-vs-small might be 'easy' for small-vs-medium.",
        bold_prefix="Model-relative — ")
    add_bullet(doc,
        "Given the same video and the same models, the disagreement scores are deterministic "
        "and identical across runs.",
        bold_prefix="Reproducible — ")
    add_bullet(doc,
        "Difficulty is not a binary label but a continuous score [0, 1], allowing analysis "
        "at any threshold.",
        bold_prefix="Continuous — ")

    add_heading(doc, "2.6  Intermediate Zone (0.2 < D < 0.5)", level=2)
    add_para(doc,
        "Frames with disagreement scores between 0.2 and 0.5 fall in an ambiguous zone "
        "where the heavier model provides some benefit but the lighter model is not "
        "dramatically worse. The switching policy must decide whether the marginal accuracy "
        "gain justifies the latency cost. This is where threshold tuning and hysteresis "
        "mechanisms become important (see Chapter 4)."
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 3: SCENE COMPLEXITY PROXIES
    # ================================================================
    add_heading(doc, "Chapter 3: Scene Complexity Proxies — How We Estimate Difficulty", level=1)

    add_para(doc,
        "The challenge: We need to estimate scene complexity BEFORE running the detector. "
        "If we had to run both models first to compute disagreement, there would be no point "
        "in switching — we would already have both outputs. Therefore, we need cheap proxy "
        "signals that correlate with the disagreement score D and can be computed in under "
        "5 ms.",
        italic=True,
    )

    # ── 3.1 Laplacian ──
    add_heading(doc, "3.1  Laplacian Variance (L)", level=2)

    add_para(doc, "What it is:", bold=True)
    add_para(doc,
        "The Laplacian is a second-order spatial derivative operator that measures the rate "
        "of intensity change across the image. The variance of the Laplacian response over "
        "all pixels quantifies the overall sharpness and texture richness of the scene."
    )

    add_equation_block(doc,
        "L = Var( nabla^2 I )",
        "Variance of the Laplacian over all pixels of the grayscale image I"
    )

    add_para(doc, "Why it works:", bold=True)
    add_para(doc,
        "Sharp, textured scenes produce high Laplacian variance because edges and fine "
        "details create large second-derivative responses. These scenes are typically 'easy' "
        "for object detectors because features are well-defined. Conversely, blurry or smooth "
        "scenes produce low Laplacian variance, and these tend to be harder for detectors "
        "because discriminative features are suppressed."
    )

    add_para(doc, "Implementation details:", bold=True)
    add_bullet(doc, "Computed on a small resized proxy image (160 x 160 pixels) for speed")
    add_bullet(doc, "Conversion to grayscale before computing the Laplacian")
    add_bullet(doc, "3x3 Laplacian kernel applied via OpenCV's cv2.Laplacian()")
    add_bullet(doc, "Computation time: approximately 1-2 ms per frame")

    add_para(doc, "Empirical result:", bold=True)
    add_para(doc,
        "Pearson correlation r = -0.577 against the disagreement score D. This is a moderate "
        "negative correlation: as Laplacian variance increases (sharper image), disagreement "
        "decreases (easier scene). This makes L the strongest image-based proxy."
    )

    add_para(doc, "Literature basis:", bold=True)
    add_para(doc,
        "The Laplacian variance as a focus/blur metric is well-established in computer vision. "
        "Pech-Pacheco et al. (2000) introduced it as a no-reference focus measure. Pertuz et al. "
        "(2013) conducted a comprehensive comparison of focus measures and confirmed the "
        "Laplacian variance as one of the most robust and computationally efficient options."
    )

    # ── 3.2 Entropy ──
    add_heading(doc, "3.2  Histogram Entropy (H)", level=2)

    add_para(doc, "What it is:", bold=True)
    add_para(doc,
        "Shannon entropy of the grayscale intensity histogram. It measures the information "
        "content or tonal diversity of the image."
    )

    add_equation_block(doc,
        "H = - SUM( p_i * log2(p_i) )  over all histogram bins",
        "Shannon entropy where p_i is the normalized frequency of bin i"
    )

    add_para(doc, "Why it was hypothesized:", bold=True)
    add_para(doc,
        "The hypothesis was that scenes with more tonal variety (higher entropy) would be "
        "more visually complex and therefore harder for detection. A uniform, low-entropy "
        "image (e.g., a blank wall) should be trivially easy, while a high-entropy image "
        "(many different intensity values) might contain more complex structures."
    )

    add_para(doc, "Empirical result:", bold=True)
    add_para(doc,
        "Pearson correlation r = +0.172 against the disagreement score D. This is a WEAK, "
        "near-zero correlation. Entropy does NOT reliably predict detection difficulty."
    )

    add_para(doc, "Why entropy fails:", bold=True)
    add_para(doc,
        "Entropy measures global tonal distribution but is blind to spatial structure. A "
        "well-lit, sharp image of a textured surface can have high entropy (many different "
        "intensity values) but be very easy for a detector. Conversely, a blurred image can "
        "have moderate entropy but be very hard. Entropy conflates 'visually interesting' "
        "with 'hard for detection,' and these are not the same thing."
    )

    add_para(doc,
        "KEY FINDING: Demonstrating that entropy is an unreliable proxy for detection "
        "difficulty is itself a scientific contribution. It warns future researchers against "
        "naively using information-theoretic image statistics as difficulty predictors.",
        bold=True,
    )

    # ── 3.3 C-Score ──
    add_heading(doc, "3.3  Combined C-Score", level=2)

    add_para(doc, "What it is:", bold=True)
    add_para(doc,
        "A weighted combination of the Laplacian variance and histogram entropy, both "
        "min-max normalized to [0, 1] using a rolling window."
    )

    add_equation_block(doc,
        "C = alpha * L_norm + (1 - alpha) * H_norm     (default alpha = 0.6)",
        "Weighted combination where L_norm and H_norm are rolling-normalized"
    )

    add_para(doc, "Normalization:", bold=True)
    add_para(doc,
        "Both L and H are normalized to [0, 1] using rolling min-max percentile normalization "
        "over a 200-frame window. This makes the C-score adaptive to the video's local "
        "statistics rather than requiring per-dataset calibration."
    )

    add_para(doc, "Empirical result:", bold=True)
    add_para(doc,
        "Pearson correlation r = -0.198 against D. This is WEAKER than Laplacian alone "
        "(r = -0.577). The combination actually hurt performance."
    )

    add_para(doc, "Why combining L and H makes things worse:", bold=True)
    add_para(doc,
        "Since entropy (H) has near-zero correlation with D, adding it to the strong "
        "Laplacian signal (r = -0.577) introduces noise. The entropy component pushes the "
        "combined score in directions unrelated to actual detection difficulty, diluting the "
        "Laplacian's predictive power. Additionally, the rolling normalization compresses "
        "the Laplacian's range, further reducing its discriminative ability."
    )

    add_para(doc,
        "LESSON: Naive combination of image-based proxies can hurt performance when one "
        "component is unreliable. Feature selection matters more than feature engineering.",
        bold=True,
    )

    # ── 3.4 Confidence EMA ──
    add_heading(doc, "3.4  Detection Confidence (Conf-EMA)", level=2)

    add_para(doc, "What it is:", bold=True)
    add_para(doc,
        "The detector's own mean confidence score, tracked over time via dual Exponential "
        "Moving Averages (EMAs). This is a fundamentally different kind of proxy: instead "
        "of analyzing the raw image, we analyze the model's output behavior."
    )

    add_para(doc, "Dual-EMA structure:", bold=True)
    add_bullet(doc,
        "Responds to sudden changes in confidence. With beta = 0.30, the effective window "
        "is approximately 3 frames. A sudden drop in detection confidence is captured "
        "within 1-2 frames.",
        bold_prefix="Fast EMA (beta = 0.30): ")
    add_bullet(doc,
        "Tracks the long-term baseline confidence level. With beta = 0.02, the effective "
        "window is approximately 50 frames. This provides a stable reference point that "
        "represents 'normal' confidence for the current scene type.",
        bold_prefix="Slow EMA (beta = 0.02): ")

    add_para(doc, "Signal computation:", bold=True)
    add_equation_block(doc,
        "conf_drop = max(0, (EMA_slow - EMA_fast) / (EMA_slow + epsilon))",
        "Relative confidence drop: how much has recent confidence fallen below baseline?"
    )

    add_para(doc, "Empirical result:", bold=True)
    add_para(doc,
        "Pearson correlation r = -0.672 against D. This is the STRONGEST correlation of "
        "all proxies tested, indicating that the model's own confidence is the most "
        "informative signal for predicting inter-model disagreement."
    )

    add_para(doc, "Why it works:", bold=True)
    add_para(doc,
        "The detection model itself is the best judge of whether it is struggling. When the "
        "lightweight model encounters a genuinely hard scene, its confidence drops because "
        "features become ambiguous. This is a self-supervised signal — no image preprocessing "
        "is needed, no separate analysis pipeline, just monitoring the model's own output. "
        "The dual-EMA structure makes the signal robust to single-frame noise while remaining "
        "responsive to genuine difficulty transitions."
    )

    # ── 3.5 NIQE ──
    add_heading(doc, "3.5  NIQE Score (No-Reference Image Quality)", level=2)

    add_para(doc, "What it is:", bold=True)
    add_para(doc,
        "The Natural Image Quality Evaluator (NIQE) is a no-reference image quality metric "
        "based on Natural Scene Statistics (NSS). It measures how much an image deviates "
        "from a statistical model of 'natural' images, without requiring a reference image."
    )

    add_para(doc, "Technical basis:", bold=True)
    add_bullet(doc,
        "Compute Mean-Subtracted Contrast-Normalized (MSCN) coefficients for each pixel")
    add_bullet(doc,
        "Fit a Generalized Gaussian Distribution (GGD) to the MSCN coefficients")
    add_bullet(doc,
        "Compare the shape and variance parameters to those of a pre-trained 'pristine' model")
    add_bullet(doc,
        "Higher NIQE scores indicate worse image quality (more deviation from natural statistics)")

    add_para(doc, "Empirical result:", bold=True)
    add_para(doc,
        "Pearson correlation r = +0.400 against D. This is a weak-to-moderate positive "
        "correlation: worse image quality (higher NIQE) corresponds to higher disagreement "
        "(harder scene). The direction is correct but the magnitude is limited."
    )

    add_para(doc, "Literature:", bold=True)
    add_para(doc,
        "Mittal, Soundararajan, and Bovik (2013): 'Making a Completely Blind Image Quality "
        "Analyzer.' IEEE Signal Processing Letters. The NIQE metric is widely used in image "
        "quality assessment research."
    )

    # ── Proxy Comparison Table ──
    add_heading(doc, "3.6  Proxy Comparison Summary", level=2)

    add_table_with_header(doc,
        ["Proxy", "Pearson r vs D", "Strength", "Data-Agnostic?", "Compute Cost"],
        [
            ["Laplacian Variance (L)", "-0.577", "Moderate", "Yes", "~1-2 ms"],
            ["Histogram Entropy (H)", "+0.172", "Negligible", "Yes", "~0.5 ms"],
            ["Combined C-Score", "-0.198", "Weak", "Partially", "~2 ms"],
            ["Mean Confidence (Conf-EMA)", "-0.672", "Moderate-Strong", "Yes", "~0.01 ms"],
            ["NIQE Score", "+0.400", "Weak-Moderate", "Yes", "~5-8 ms"],
        ],
        col_widths=[2.0, 1.2, 1.2, 1.2, 1.0],
    )

    add_para(doc,
        "Key takeaway: The model's own confidence signal (Conf-EMA) is the strongest proxy "
        "for scene difficulty, with the additional advantage of being essentially free to "
        "compute (just monitoring existing output). Image-based proxies are weaker and more "
        "computationally expensive.",
        bold=True,
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 4: SWITCHING POLICIES
    # ================================================================
    add_heading(doc, "Chapter 4: Switching Policies — The Decision Logic", level=1)

    add_para(doc,
        "Each switching policy defines the decision logic that determines which model "
        "processes a given frame. For each policy, we present: the hypothesis, the algorithm "
        "in plain steps, the switching rule, and the expected behavior.",
        italic=True,
    )

    # ── 4.1 Baselines ──
    add_heading(doc, "4.1  Baselines (n_only, s_only)", level=2)

    add_para(doc, "n_only — Always Lightweight:", bold=True)
    add_para(doc,
        "Every frame is processed by YOLOv8n (nano). This establishes the lower bound for "
        "latency and the upper bound for processing speed. Detection quality will be reduced "
        "on hard scenes where the nano model misses objects, but latency is minimal and "
        "consistent."
    )

    add_para(doc, "s_only — Always Heavier:", bold=True)
    add_para(doc,
        "Every frame is processed by YOLOv8s (small). This establishes the upper bound for "
        "detection quality and the lower bound for speed. No frames are missed, but latency "
        "is consistently higher."
    )

    add_para(doc, "Purpose:", bold=True)
    add_para(doc,
        "These baselines define the performance envelope within which all switching policies "
        "operate. An effective switching policy should achieve latency close to n_only while "
        "maintaining detection quality close to s_only. Any policy that falls outside this "
        "envelope (slower than s_only or worse quality than n_only) is counterproductive."
    )

    # ── 4.2 Entropy-Only ──
    add_heading(doc, "4.2  Entropy-Only Policy", level=2)

    add_para(doc, "Hypothesis:", bold=True)
    add_para(doc,
        "Frames with above-median entropy represent more complex scenes that would benefit "
        "from the heavier model."
    )

    add_para(doc, "Switching rule:", bold=True)
    add_para(doc,
        "Compute the histogram entropy H for each frame. If H is greater than or equal to "
        "the running median of H, switch to the heavier model (YOLOv8s). Otherwise, use "
        "the lightweight model (YOLOv8n)."
    )

    add_para(doc, "Expected behavior:", bold=True)
    add_para(doc,
        "Poor switching quality. Since entropy has negligible correlation with disagreement "
        "(r = +0.172), the policy will switch to the heavier model on frames that are not "
        "actually harder, while failing to switch on genuinely hard frames. The s-model "
        "usage rate will be approximately 50% (by definition of median splitting) but the "
        "switches will be poorly targeted."
    )

    # ── 4.3 Combined ──
    add_heading(doc, "4.3  Combined (C-Score Based) Policy", level=2)

    add_para(doc, "Hypothesis:", bold=True)
    add_para(doc,
        "A weighted combination of blur (Laplacian) and tonal complexity (entropy) provides "
        "a better difficulty estimate than either alone."
    )

    add_para(doc, "Switching rule:", bold=True)
    add_para(doc,
        "Compute the combined C-score for each frame. If C >= 0.50 (center of the normalized "
        "range), switch to YOLOv8s. Otherwise, use YOLOv8n."
    )

    add_para(doc, "Expected behavior:", bold=True)
    add_para(doc,
        "Better than entropy-only due to the inclusion of Laplacian (which has genuine "
        "predictive power), but weaker than using Laplacian alone because the entropy "
        "component dilutes the signal. The C-score correlation (r = -0.198) is much weaker "
        "than Laplacian alone (r = -0.577)."
    )

    # ── 4.4 Combined with Hysteresis ──
    add_heading(doc, "4.4  Combined with Hysteresis", level=2)

    add_para(doc, "Motivation:", bold=True)
    add_para(doc,
        "Without hysteresis, the C-score fluctuating around the 0.50 threshold causes the "
        "system to switch models nearly every frame, creating oscillation that is both "
        "computationally wasteful and operationally disruptive."
    )

    add_para(doc, "Hysteresis mechanisms:", bold=True)
    add_bullet(doc,
        "Two thresholds instead of one: c_low = 0.45 (switch to nano), "
        "c_high = 0.55 (switch to small). Scores between 0.45 and 0.55 maintain "
        "the current model — no switching occurs in this dead zone.",
        bold_prefix="Dead zone: ")
    add_bullet(doc,
        "The system must remain on the current model for at least 10 frames before "
        "switching again, preventing rapid back-and-forth.",
        bold_prefix="Minimum dwell (10 frames): ")
    add_bullet(doc,
        "No more than 12 model switches per 100 frames. If the cap is reached, "
        "switching is suppressed until the rate falls back below the limit.",
        bold_prefix="Switch rate cap (12 per 100 frames): ")

    add_para(doc, "Expected behavior:", bold=True)
    add_para(doc,
        "More stable switching with fewer oscillations. The latency profile becomes smoother "
        "but the underlying C-score weakness remains — the system just oscillates less "
        "frequently around a still-suboptimal signal."
    )

    # ── 4.5 Confidence EMA ──
    add_heading(doc, "4.5  Confidence-EMA Policy (RECOMMENDED)", level=2)

    add_para(doc, "Hypothesis:", bold=True)
    add_para(doc,
        "If the detection model's own confidence drops relative to its recent baseline, "
        "it is struggling with the current scene, and the heavier model should take over."
    )

    add_para(doc, "Algorithm (step by step):", bold=True)
    add_para(doc, "Step 1: After processing each frame, record the mean detection confidence "
        "across all detected objects.")
    add_para(doc, "Step 2: Update the fast EMA (beta = 0.30, responds in approximately 3 frames) "
        "with the new confidence value.")
    add_para(doc, "Step 3: Update the slow EMA (beta = 0.02, responds in approximately 50 frames) "
        "with the same confidence value.")
    add_para(doc, "Step 4: Compute the relative confidence drop: "
        "conf_drop = max(0, (EMA_slow - EMA_fast) / (EMA_slow + epsilon)).")
    add_para(doc, "Step 5: Decision logic:")
    add_bullet(doc,
        "If conf_drop >= 0.12 (12%): Switch to the heavier model. "
        "The detector is struggling — recent confidence has fallen significantly "
        "below the long-term baseline.")
    add_bullet(doc,
        "If conf_drop <= 0.04 (4%): Switch back to the lightweight model. "
        "Confidence has recovered to near-baseline levels, indicating the scene "
        "has become easier.")
    add_bullet(doc,
        "If 0.04 < conf_drop < 0.12: Maintain the current model (hysteresis zone).")

    add_para(doc, "Why this is the strongest policy:", bold=True)
    add_bullet(doc, "Uses the most direct signal available — the model's own assessment of scene difficulty")
    add_bullet(doc, "No image preprocessing needed — just monitoring existing model output")
    add_bullet(doc, "Naturally adaptive to any dataset — the EMA baselines adjust to local conditions")
    add_bullet(doc, "Built-in temporal smoothing via the dual-EMA structure")
    add_bullet(doc, "Strongest correlation with actual disagreement (r = -0.672)")

    # ── 4.6 NIQE Switch ──
    add_heading(doc, "4.6  NIQE-Switch Policy", level=2)

    add_para(doc, "Hypothesis:", bold=True)
    add_para(doc,
        "Degraded image quality (as measured by NIQE) predicts detection difficulty, and "
        "a sudden quality degradation should trigger switching to the heavier model."
    )

    add_para(doc, "Implementation:", bold=True)
    add_para(doc,
        "Uses the same dual-EMA structure as the confidence-based policy, but tracks the "
        "NIQE score instead of detection confidence. Because NIQE variations in UAV footage "
        "are much smaller than confidence variations (coefficient of variation ~0.2-0.5%), "
        "tighter thresholds are used: 4% for escalation and 1% for de-escalation."
    )

    add_para(doc, "Expected behavior:", bold=True)
    add_para(doc,
        "Moderate effectiveness (correlation r = +0.400 between NIQE and D). Better than "
        "entropy-only but weaker than confidence-based switching. The tight thresholds make "
        "the policy sensitive to small NIQE fluctuations, which can cause unnecessary switches."
    )

    # ── Policy Comparison Table ──
    add_heading(doc, "4.7  Policy Comparison Summary", level=2)

    add_table_with_header(doc,
        ["Policy", "Proxy Used", "Proxy r vs D", "Hysteresis", "Recommended?"],
        [
            ["n_only", "None (always nano)", "N/A", "N/A", "Baseline only"],
            ["s_only", "None (always small)", "N/A", "N/A", "Baseline only"],
            ["entropy_only", "Histogram Entropy", "+0.172", "No", "No — weak proxy"],
            ["combined", "C-Score (L + H)", "-0.198", "No", "No — diluted signal"],
            ["combined_hysteresis", "C-Score (L + H)", "-0.198", "Yes", "Marginal improvement"],
            ["conf_ema", "Detection Confidence", "-0.672", "Built-in (dual EMA)", "YES — recommended"],
            ["niqe_switch", "NIQE Score", "+0.400", "Yes (tight)", "Secondary option"],
            ["multi_proxy", "7-proxy composite", "(weighted)", "Yes (EMA-relative)", "Experimental"],
            ["local_contrast_hyst*", "Local Contrast", "(domain-dep.)", "Yes", "Best scene-proxy for glass"],
        ],
        col_widths=[1.5, 1.5, 1.0, 1.2, 1.4],
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 5: PER-FRAME PROCESSING PIPELINE
    # ================================================================
    add_heading(doc, "Chapter 5: The Per-Frame Processing Pipeline", level=1)

    add_para(doc,
        "This chapter provides a detailed, step-by-step walkthrough of what happens for "
        "every single frame in the proposed system.",
        italic=True,
    )

    add_heading(doc, "5.1  Step 1 — Frame Acquisition", level=2)
    add_para(doc,
        "The raw frame is read from the video source (file or live camera feed). The frame "
        "arrives in BGR color format at the source resolution (e.g., 1920x1080 or 3840x2160)."
    )

    add_heading(doc, "5.2  Step 2 — Proxy Computation (T_scene)", level=2)
    add_para(doc,
        "Before running the detector, we compute the scene complexity proxy. The specific "
        "computation depends on the active switching policy:"
    )

    add_para(doc, "For scene-based policies (entropy, combined, combined_hysteresis):", bold=True)
    add_bullet(doc, "Resize the frame to a small proxy image (160 x 160 pixels)")
    add_bullet(doc, "Convert to grayscale")
    add_bullet(doc, "Compute Laplacian variance: apply 3x3 Laplacian kernel, compute variance of response")
    add_bullet(doc, "Compute histogram entropy: build 256-bin intensity histogram, compute Shannon entropy")
    add_bullet(doc, "Apply rolling min-max normalization over a 200-frame window")
    add_bullet(doc, "Compute C-score: alpha * L_norm + (1 - alpha) * H_norm")

    add_para(doc, "For conf_ema policy:", bold=True)
    add_bullet(doc,
        "No image analysis needed. Simply update the dual-EMA from the previous frame's "
        "mean detection confidence. This makes it the cheapest proxy (~0.01 ms).")

    add_para(doc, "For niqe_switch policy:", bold=True)
    add_bullet(doc, "Resize to proxy image (160 x 160)")
    add_bullet(doc, "Compute MSCN coefficients for each pixel")
    add_bullet(doc, "Fit GGD shape and variance parameters")
    add_bullet(doc, "Compute NIQE score as Mahalanobis distance from pristine model")

    add_heading(doc, "5.3  Step 3 — Controller Decision (T_ctrl)", level=2)
    add_para(doc,
        "The controller compares the proxy signal to the policy's thresholds and applies "
        "actuator guards to prevent unstable switching:"
    )
    add_bullet(doc, "Compare proxy value to escalation threshold (switch to heavier model)")
    add_bullet(doc, "Compare proxy value to de-escalation threshold (switch back to lighter model)")
    add_bullet(doc,
        "Check minimum dwell: has the system stayed on the current model for at least the "
        "minimum number of frames?")
    add_bullet(doc,
        "Check switch rate cap: have there been too many switches in the recent window?")
    add_bullet(doc, "If all guards pass and the signal crosses a threshold, issue a model switch command")
    add_para(doc, "This step takes approximately 0.01 ms — essentially instantaneous.")

    add_heading(doc, "5.4  Step 4 — Model Inference (T_infer)", level=2)
    add_para(doc,
        "The selected model runs inference on the FULL-RESOLUTION frame. Only ONE model "
        "runs per frame — this is where the speed savings come from."
    )
    add_bullet(doc, "Preprocessing: Resize to inference size (640 x 640), normalize pixel values, "
        "create input tensor")
    add_bullet(doc, "Forward pass: Run the model's neural network on GPU or CPU")
    add_bullet(doc, "Postprocessing: Apply Non-Maximum Suppression (NMS) to remove duplicate detections, "
        "filter by confidence threshold")

    add_heading(doc, "5.5  Step 5 — Output", level=2)
    add_para(doc,
        "The final output for each frame consists of the detection results: bounding boxes "
        "(x, y, width, height), class labels, and confidence scores for each detected object."
    )

    add_heading(doc, "5.6  Timing Breakdown", level=2)
    add_equation_block(doc,
        "T_total = T_scene + T_ctrl + T_infer",
        "Only one of T_infer_n or T_infer_s is nonzero per frame"
    )

    add_table_with_header(doc,
        ["Component", "GPU Time", "CPU Time", "Notes"],
        [
            ["T_scene (proxy)", "0.5–8 ms", "0.5–8 ms", "Same on GPU/CPU; conf_ema ~0.01 ms"],
            ["T_ctrl (decision)", "~0.01 ms", "~0.01 ms", "Threshold comparison + guards"],
            ["T_infer_n (YOLOv8n)", "5–15 ms", "15–40 ms", "Lightweight model"],
            ["T_infer_s (YOLOv8s)", "10–30 ms", "30–80 ms", "Heavier model"],
        ],
        col_widths=[1.5, 1.0, 1.0, 3.0],
    )

    add_para(doc,
        "Net result: When the switching policy routes most frames to the lightweight model "
        "and only escalates to the heavier model on genuinely hard frames, the mean T_total "
        "approaches the n_only baseline while detection quality on hard frames is maintained.",
        bold=True,
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 6: VALIDATION METHODOLOGY
    # ================================================================
    add_heading(doc, "Chapter 6: Validation Methodology", level=1)

    add_para(doc,
        "This chapter describes how we rigorously prove that the proposed approach works. "
        "Validation is performed offline to allow comprehensive analysis without real-time "
        "constraints.",
        italic=True,
    )

    add_heading(doc, "6.1  Dual-Model Validation", level=2)
    add_para(doc,
        "The cornerstone of our validation methodology is dual-model validation: running "
        "BOTH models on EVERY frame of the test videos. This is computationally expensive "
        "(approximately 2x the cost of single-model processing) and cannot be done in "
        "real-time, but it provides the ground truth needed to evaluate proxy quality."
    )
    add_para(doc, "Process:", bold=True)
    add_bullet(doc, "For each frame in the test video, run YOLOv8n and record all detections")
    add_bullet(doc, "For the same frame, run YOLOv8s and record all detections")
    add_bullet(doc, "Compute the disagreement score D by comparing the two detection sets")
    add_bullet(doc, "Simultaneously compute all proxy values (L, H, C, confidence, NIQE)")
    add_bullet(doc, "Result: a per-frame dataset of proxy values paired with the ground-truth difficulty D")

    add_heading(doc, "6.2  Correlation Analysis", level=2)
    add_para(doc, "For each proxy, we compute two correlation metrics against D:")

    add_bullet(doc,
        "Measures the strength and direction of the linear relationship between the proxy "
        "and D. Values range from -1 (perfect negative) to +1 (perfect positive). "
        "|r| > 0.5 indicates a meaningful relationship; |r| > 0.7 is strong.",
        bold_prefix="Pearson r: ")
    add_bullet(doc,
        "Measures the monotonic (rank-order) relationship, which is more robust to outliers "
        "and non-linear relationships. Interpretation is similar to Pearson r.",
        bold_prefix="Spearman rho: ")

    add_para(doc,
        "Both metrics are reported to provide a complete picture. If Pearson and Spearman "
        "diverge substantially, it suggests a non-linear relationship that might benefit "
        "from a more sophisticated policy.",
    )

    add_heading(doc, "6.3  Sensitivity Analysis", level=2)
    add_para(doc,
        "Sensitivity analysis tests whether the switching decisions are robust to changes "
        "in proxy computation parameters. This is critical for establishing that the "
        "framework is not overfitted to specific parameter choices."
    )
    add_para(doc, "Process:", bold=True)
    add_bullet(doc, "Systematically vary proxy computation settings (e.g., proxy image size, "
        "histogram bin count)")
    add_bullet(doc, "For each parameter combination, compute the proxy values and the resulting "
        "switching decisions")
    add_bullet(doc, "Measure agreement between the baseline configuration and each alternative")
    add_bullet(doc, "If agreement exceeds 85% across the tested range, the parameter is considered robust")

    add_heading(doc, "6.4  Proxy Size Grid Search", level=2)
    add_para(doc,
        "A systematic sweep over proxy_size x hist_bins combinations to find the optimal "
        "configuration for each proxy component:"
    )
    add_bullet(doc, "Proxy sizes tested: 80, 120, 160, 200, 240, 320 pixels")
    add_bullet(doc, "Histogram bins tested: 64, 128, 256, 512")
    add_bullet(doc, "For each combination: compute proxy, correlate with D, record computation time")
    add_bullet(doc, "Select the configuration that maximizes |r| while keeping computation under 5 ms")

    add_page_break(doc)

    # ================================================================
    # CHAPTER 7: HOW TO REPRODUCE RESULTS
    # ================================================================
    add_heading(doc, "Chapter 7: Step-by-Step — How to Reproduce Results", level=1)

    add_para(doc,
        "This chapter provides concrete instructions for reproducing all experimental "
        "results from this research.",
        italic=True,
    )

    add_heading(doc, "7.1  Prerequisites", level=2)
    add_bullet(doc, "UAV inspection video dataset (one or more inspection videos)")
    add_bullet(doc, "Trained YOLOv8n weights (.pt file)")
    add_bullet(doc, "Trained YOLOv8s weights (.pt file)")
    add_bullet(doc, "Python 3.9+ with ultralytics, OpenCV, NumPy, SciPy, pandas")

    add_heading(doc, "7.2  Step 1 — Run Baselines", level=2)
    add_para(doc,
        "Process every frame of each test video with the n_only policy (YOLOv8n on every "
        "frame), then repeat with the s_only policy (YOLOv8s on every frame). For each "
        "policy, record:"
    )
    add_bullet(doc, "Mean total processing time per frame (T_total_mean)")
    add_bullet(doc, "95th percentile latency (P95)")
    add_bullet(doc, "99th percentile latency (P99)")
    add_bullet(doc, "Total frame count and frames per second achieved")

    add_heading(doc, "7.3  Step 2 — Run Each Switching Policy", level=2)
    add_para(doc,
        "For each policy (entropy_only, combined, combined_hyst, conf_ema, niqe_switch, multi_proxy), "
        "process the same test videos with default parameters. Record:"
    )
    add_bullet(doc, "Mean T_total and P95/P99 latencies")
    add_bullet(doc, "Number of model switches and switches per 100 frames")
    add_bullet(doc, "Percentage of frames processed by the heavier model (s-model usage %)")
    add_bullet(doc, "Speedup relative to s_only baseline")

    add_heading(doc, "7.4  Step 3 — Run Dual-Model Validation", level=2)
    add_para(doc,
        "Process every frame with BOTH models. Compute disagreement score D and all proxy "
        "values for each frame. Compute Pearson r and Spearman rho for each proxy vs D."
    )

    add_heading(doc, "7.5  Step 4 — Generate Comparison Table", level=2)
    add_table_with_header(doc,
        ["Policy", "Mean T_total", "P95", "Switches/100", "S-Usage %", "Speedup vs s_only"],
        [
            ["n_only", "(measured)", "(measured)", "0", "0%", "(reference)"],
            ["s_only", "(measured)", "(measured)", "0", "100%", "1.00x"],
            ["entropy_only", "(measured)", "(measured)", "(measured)", "(measured)", "(computed)"],
            ["combined", "(measured)", "(measured)", "(measured)", "(measured)", "(computed)"],
            ["combined_hyst", "(measured)", "(measured)", "(measured)", "(measured)", "(computed)"],
            ["conf_ema", "(measured)", "(measured)", "(measured)", "(measured)", "(computed)"],
            ["niqe_switch", "(measured)", "(measured)", "(measured)", "(measured)", "(computed)"],
            ["multi_proxy", "(measured)", "(measured)", "(measured)", "(measured)", "(computed)"],
        ],
    )

    add_heading(doc, "7.6  Step 5 — Generate Validation Plots", level=2)
    add_bullet(doc, "Scatter plots: Each proxy vs D, colored by frame density")
    add_bullet(doc, "Correlation matrix: All proxies pairwise + D")
    add_bullet(doc, "Sensitivity heatmap: Agreement percentage across parameter grid")
    add_bullet(doc, "Pareto front: Switching rate vs s-model usage across policies")
    add_bullet(doc, "Time-series: Proxy values, switching decisions, and D over frame sequence")

    add_heading(doc, "7.7  Step 6 — Export Results", level=2)
    add_para(doc,
        "Export all numerical results as CSV files and all plots as publication-quality "
        "figures (PDF/PNG at 300 DPI). Organize outputs by video name and policy."
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 8: KEY FINDINGS AND DISCUSSION
    # ================================================================
    add_heading(doc, "Chapter 8: Key Findings and Discussion", level=1)

    add_heading(doc, "8.1  Confidence-EMA Achieves the Best Tradeoff", level=2)
    add_para(doc,
        "The confidence-based switching policy (conf_ema) consistently achieves the best "
        "latency-accuracy tradeoff across all tested videos. By monitoring the model's own "
        "confidence rather than analyzing raw image properties, it avoids the limitations "
        "of image-based proxies and responds directly to the signal that matters: whether "
        "the model is struggling."
    )

    add_heading(doc, "8.2  Laplacian Is the Strongest Image-Based Proxy", level=2)
    add_para(doc,
        "Among image-based proxies, Laplacian variance shows the strongest correlation with "
        "disagreement (r = -0.577). This confirms the well-known relationship between image "
        "blur and detection difficulty. However, it remains weaker than the confidence-based "
        "signal because it cannot capture all sources of difficulty (e.g., occlusion, "
        "class ambiguity)."
    )

    add_heading(doc, "8.3  Entropy Is Unreliable", level=2)
    add_para(doc,
        "Histogram entropy shows negligible correlation with disagreement (r = +0.172). "
        "This is a negative result but a valuable one: it demonstrates that tonal diversity "
        "is not a reliable predictor of detection difficulty, contrary to the intuitive "
        "assumption that 'complex-looking' images are harder to analyze."
    )

    add_heading(doc, "8.4  Combining L and H Hurts Performance (C-Score Dilution)", level=2)
    add_para(doc,
        "The combined C-score (r = -0.198) is substantially weaker than Laplacian alone "
        "(r = -0.577). This demonstrates a cautionary principle: naively combining features "
        "can degrade performance when one feature introduces noise. The rolling normalization "
        "further compounds the problem by compressing the Laplacian's dynamic range."
    )

    add_heading(doc, "8.5  Default Parameters Are Reasonably Data-Agnostic", level=2)
    add_para(doc,
        "Sensitivity analysis shows that switching decisions remain stable (>85% agreement) "
        "across the tested parameter range. The dual-EMA structure with its adaptive baseline "
        "is key: because the slow EMA tracks the dataset's local statistics, the conf_drop "
        "metric is inherently normalized to the current video's characteristics."
    )

    add_heading(doc, "8.6  Framework Generalizes Across Inspection Materials", level=2)
    add_para(doc,
        "The methodology is not specific to any particular type of inspection target. "
        "Whether inspecting concrete bridges, metal pipelines, or composite wind turbine "
        "blades, the underlying principle remains: scene difficulty varies, and the model's "
        "confidence is the best available signal for detecting that variation."
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 9: LIMITATIONS AND FUTURE WORK
    # ================================================================
    add_heading(doc, "Chapter 9: Limitations and Future Work", level=1)

    add_para(doc,
        "Honest acknowledgment of limitations is essential for scientific credibility. "
        "This chapter documents all known limitations and outlines promising directions "
        "for future research.",
        italic=True,
    )

    add_heading(doc, "9.1  Moderate Proxy Correlation (~55% Variance Unexplained)", level=2)
    add_para(doc,
        "Even the strongest proxy (confidence-EMA, r = -0.672) explains only about 45% of "
        "the variance in disagreement scores (R-squared = 0.45). The remaining 55% is "
        "influenced by factors that no single proxy captures: object-level class ambiguity, "
        "contextual scene understanding, dataset-specific biases, and stochastic model "
        "behavior. Future work could explore multi-proxy fusion with learned weights, "
        "attention-based feature extraction, or lightweight auxiliary networks specifically "
        "trained to predict inter-model disagreement."
    )

    add_heading(doc, "9.2  Temporal Lag in Confidence-Based Switching", level=2)
    add_para(doc,
        "The confidence-based policy has an inherent 1-frame delay: the confidence signal "
        "from frame t is used to make the decision for frame t+1. Additionally, the EMA "
        "smoothing introduces further lag — the fast EMA needs approximately 2-3 frames to "
        "fully respond to a sudden confidence drop. This means the system may process 2-3 "
        "hard frames with the lightweight model before switching. For video at 30 fps, this "
        "represents approximately 66-100 ms of delay."
    )

    add_heading(doc, "9.3  Two-Model Limitation", level=2)
    add_para(doc,
        "The current framework switches between exactly two models (nano and small). In "
        "principle, the approach could be extended to a continuum of models (nano, small, "
        "medium, large, extra-large) with more complex switching policies. This would "
        "require multi-threshold decision logic and more sophisticated hysteresis mechanisms."
    )

    add_heading(doc, "9.4  Hand-Crafted Thresholds", level=2)
    add_para(doc,
        "The switching thresholds (12% escalation, 4% de-escalation for conf_ema) were "
        "determined through empirical analysis. While sensitivity testing shows they are "
        "reasonably robust, they could potentially be replaced with learned policies using "
        "reinforcement learning (RL). An RL agent could learn optimal switching behavior "
        "by maximizing a reward function that balances latency and detection quality."
    )

    add_heading(doc, "9.5  Single Domain Tested", level=2)
    add_para(doc,
        "All experiments were conducted on UAV inspection footage. While the methodology "
        "is domain-agnostic in principle, the specific correlation strengths, optimal "
        "thresholds, and proxy effectiveness may differ in other domains (autonomous "
        "driving, surveillance, medical imaging). Cross-domain validation is needed."
    )

    add_heading(doc, "9.6  C-Score Design Flaw", level=2)
    add_para(doc,
        "The C-score design flaw (entropy diluting Laplacian) was identified after the "
        "initial design. A better approach would be to use Laplacian alone or replace "
        "entropy with a more informative second proxy (e.g., edge density, local contrast "
        "variance, or frequency domain features). This is left as future work."
    )

    add_heading(doc, "9.7  NIQE Sensitivity in UAV Footage", level=2)
    add_para(doc,
        "NIQE scores in UAV inspection footage show very low variation (coefficient of "
        "variation 0.2-0.5%). This makes threshold-based switching extremely sensitive to "
        "small fluctuations, reducing the practical reliability of NIQE-based switching. "
        "The low variation may be because UAV footage tends to have consistent imaging "
        "conditions (similar altitude, similar camera, similar lighting)."
    )

    add_heading(doc, "9.8  No Direct mAP Evaluation", level=2)
    add_para(doc,
        "We measure switching quality (latency, switch rate, s-model usage) and proxy "
        "quality (correlation with disagreement) but not end-to-end detection accuracy "
        "(mAP). Computing mAP requires ground-truth bounding box annotations, which were "
        "not available for the test videos. Future work should include mAP evaluation on "
        "annotated datasets to directly quantify the detection quality impact of switching."
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 10: ANSWERING SUPERVISOR QUESTIONS
    # ================================================================
    add_heading(doc, "Chapter 10: Answering Supervisor Questions", level=1)

    add_para(doc,
        "This chapter provides thorough answers to every question a thesis supervisor might "
        "ask. Questions are organized by category: fundamental, technical, critical/adversarial, "
        "and comparison/literature.",
        italic=True,
    )

    # ── Fundamental ──
    add_heading(doc, "10.1  Fundamental Questions", level=2)

    add_key_answer(doc,
        "What exactly is your work? What did you do?",
        "We designed and implemented a real-time framework for adaptive model switching in "
        "UAV-based object detection. The system dynamically selects between a fast lightweight "
        "detector (YOLOv8n) and a more accurate but slower detector (YOLOv8s) on a per-frame "
        "basis, based on estimated scene complexity. We developed and evaluated multiple scene "
        "complexity proxies (Laplacian variance, histogram entropy, NIQE, detection confidence), "
        "designed several switching policies with built-in hysteresis mechanisms, and conducted "
        "rigorous validation through dual-model analysis and correlation studies. The key "
        "finding is that the model's own confidence signal, tracked via dual exponential moving "
        "averages, is the most effective proxy for switching decisions."
    )

    add_key_answer(doc,
        "What is the research question?",
        "Can we build a real-time system that adaptively switches between a lightweight and a "
        "heavier object detector based on estimated scene complexity, achieving detection "
        "quality close to the heavier model at processing speed close to the lighter model? "
        "Secondary questions include: which scene complexity proxies are most predictive, and "
        "how robust are the switching decisions to parameter variation?"
    )

    add_key_answer(doc,
        "What is your hypothesis?",
        "Not all video frames require the same model capacity. Scene complexity varies "
        "throughout a UAV inspection flight, and this variation can be estimated cheaply "
        "(under 5 ms) using proxy signals. By routing easy frames to the fast model and hard "
        "frames to the accurate model, we can approximate the best-model accuracy at the "
        "fastest-model speed."
    )

    add_key_answer(doc,
        "What does 'dynamic model switching' mean?",
        "Dynamic model switching means selecting which neural network model processes each "
        "video frame at runtime, rather than using the same model for every frame. The "
        "selection is based on a real-time estimate of how 'hard' the current scene is for "
        "the detector. When the scene is easy, we use the fast model; when it is hard, we "
        "switch to the accurate model. The switching happens automatically, frame by frame, "
        "without human intervention."
    )

    add_key_answer(doc,
        "How do you define a simple vs hard scene?",
        "We define scene difficulty operationally using inter-model disagreement. A 'simple' "
        "scene (disagreement D < 0.2) is one where the lightweight model and the heavier model "
        "produce nearly identical detections — both find the same objects with similar confidence "
        "and bounding box locations. This means the heavier model was not needed. A 'hard' scene "
        "(D > 0.5) is one where the heavier model produces substantially different (and better) "
        "detections — more objects found, higher confidence, or different localization. This "
        "definition is fully automated, reproducible, and directly tied to the switching "
        "decision: difficulty means 'the heavier model was needed here.'"
    )

    add_key_answer(doc,
        "Why YOLO specifically? Would this work with other detectors?",
        "We chose YOLO for several practical reasons: it is the state-of-the-art single-stage "
        "detector widely used in industrial inspection, the YOLOv8 family provides a natural "
        "capacity hierarchy (nano, small, medium, large) sharing the same architecture, and "
        "it offers excellent speed-accuracy tradeoffs on edge hardware. However, the framework "
        "is fundamentally model-agnostic — it only requires two models of different capacity "
        "from the same family. The same approach would work with EfficientDet (D0 vs D2), "
        "SSD (MobileNet vs ResNet backbone), or any other detector pair where the heavier "
        "variant provides better accuracy at higher latency."
    )

    # ── Technical ──
    add_heading(doc, "10.2  Technical Questions", level=2)

    add_key_answer(doc,
        "How does the switching decision happen in real-time?",
        "The switching decision takes approximately 0.01 ms and involves three checks: "
        "(1) Compare the proxy signal to the escalation/de-escalation thresholds. For conf_ema, "
        "this means checking if the relative confidence drop exceeds 12% (escalate) or has "
        "fallen below 4% (de-escalate). (2) Check the minimum dwell constraint — has the "
        "system stayed on the current model for enough frames? (3) Check the switch rate cap — "
        "have there been too many switches recently? If all conditions are met, the model "
        "index is updated. The next frame is then processed by the newly selected model."
    )

    add_key_answer(doc,
        "What is Laplacian variance and why does it measure complexity?",
        "The Laplacian is a second-order spatial derivative operator that responds to "
        "intensity changes. Applied to an image, it produces high responses at edges, "
        "textures, and sharp features, and low responses in smooth or blurry regions. The "
        "variance of this response across all pixels quantifies how much edge/texture content "
        "the image contains. High Laplacian variance indicates a sharp, well-focused image "
        "with rich texture — typically easy for detectors because features are well-defined. "
        "Low variance indicates blur or smooth surfaces — typically harder because "
        "discriminative features are suppressed. This is a well-established focus metric "
        "in computer vision literature."
    )

    add_key_answer(doc,
        "What is the EMA approach and why dual-rate?",
        "An Exponential Moving Average (EMA) is a weighted running average that gives more "
        "weight to recent observations. We use two EMAs with different smoothing rates: a "
        "fast EMA (beta = 0.30, effective window ~3 frames) that responds quickly to sudden "
        "changes, and a slow EMA (beta = 0.02, effective window ~50 frames) that tracks the "
        "long-term baseline. The key insight is the ratio between them: when fast EMA drops "
        "significantly below slow EMA, it means recent confidence has fallen below the "
        "historical baseline, signaling a difficulty transition. The dual-rate structure "
        "provides both responsiveness (fast EMA detects sudden drops) and stability (slow "
        "EMA provides a robust reference point). This is analogous to the MACD indicator "
        "in financial analysis."
    )

    add_key_answer(doc,
        "How do you validate that switching helps?",
        "Validation has three components: (1) Dual-model validation — run both models on "
        "every frame offline, compute disagreement D as ground truth for difficulty, then "
        "correlate each proxy with D to verify it predicts difficulty. (2) Policy comparison — "
        "run each switching policy on the same videos, measuring latency metrics (mean, P95, P99) "
        "and switching statistics (switches per 100 frames, s-model usage %). Compare against "
        "n_only and s_only baselines to quantify the speedup and quality tradeoff. "
        "(3) Sensitivity analysis — verify that switching decisions are robust (>85% agreement) "
        "across parameter variations."
    )

    add_key_answer(doc,
        "What is the disagreement score?",
        "The disagreement score D is a composite metric that quantifies how differently two "
        "models interpret the same frame. It ranges from 0 (identical detections) to 1 (maximum "
        "disagreement). It combines four components: detection count gap (30% weight), confidence "
        "gap (25%), localization disagreement via IoU (25%), and extra detections by the heavier "
        "model (20%). The formula is: D = 0.30 * det_gap_norm + 0.25 * conf_gap_norm + "
        "0.25 * iou_disagree + 0.20 * extra_s_norm. This score serves as the ground truth "
        "for scene difficulty in our validation framework."
    )

    add_key_answer(doc,
        "Why does entropy NOT work?",
        "Histogram entropy measures the tonal diversity of pixel intensities — how evenly "
        "distributed the brightness values are. The hypothesis was that more tonal complexity "
        "implies harder scenes. But empirically, the correlation with disagreement is only "
        "r = +0.172, which is negligible. The reason: entropy is blind to spatial structure. "
        "A sharp image of a textured surface has high entropy (many different intensity values) "
        "but is easy for a detector. A blurred image can have moderate entropy but be very hard. "
        "Entropy conflates 'visually diverse' with 'hard for detection,' and these are "
        "fundamentally different properties."
    )

    add_key_answer(doc,
        "Why does combining L and H make things worse?",
        "The combined C-score (r = -0.198) is weaker than Laplacian alone (r = -0.577) because "
        "entropy (r = +0.172) introduces noise. When you combine a strong signal (L) with a "
        "near-zero signal (H) via weighted averaging, the noise from H pushes the combined "
        "score in random directions, diluting the Laplacian's predictive power. Additionally, "
        "the rolling min-max normalization compresses the Laplacian's dynamic range, further "
        "reducing its discrimination ability. This is a classic example of the 'garbage in, "
        "garbage out' principle in feature combination."
    )

    # ── Critical / Adversarial ──
    add_heading(doc, "10.3  Critical and Adversarial Questions", level=2)

    add_key_answer(doc,
        "Isn't this just using the fast model most of the time?",
        "Yes, and that is precisely the point. In typical UAV inspection footage, the majority "
        "of frames (often 60-80%) are 'easy' — clear views, good lighting, well-defined objects. "
        "These frames genuinely do not need the heavier model. The value of switching is not "
        "that it uses the heavier model frequently, but that it uses the heavier model "
        "PRECISELY when needed — on the 20-40% of frames where the lightweight model would "
        "miss detections or produce unreliable results. The speedup comes from avoiding "
        "unnecessary computation on easy frames."
    )

    add_key_answer(doc,
        "Your C-score correlation is only r = -0.198. How can you claim it works?",
        "We do NOT claim the C-score works well. In fact, demonstrating that the C-score is "
        "weak (r = -0.198) compared to Laplacian alone (r = -0.577) is one of our findings. "
        "It shows that naively combining image features can degrade performance. The C-score "
        "policy exists to demonstrate this negative result and to provide contrast with the "
        "stronger confidence-based approach. Our recommended policy is conf_ema (r = -0.672), "
        "which does not use the C-score at all."
    )

    add_key_answer(doc,
        "How do you know switching isn't introducing errors?",
        "The switching logic only decides WHICH model processes each frame — it does not modify "
        "the model's detections. Each frame is processed entirely by one model (nano or small), "
        "and that model's detections are used as-is. The potential error mode is using the "
        "lightweight model on a frame that was actually 'hard,' resulting in missed detections. "
        "This is mitigated by (a) the proxy signal triggering escalation, and (b) the fact "
        "that even without switching (n_only baseline), the lightweight model processes ALL "
        "frames. Switching can only improve upon n_only, never make it worse — in the worst "
        "case, it matches n_only; in the best case, it catches hard frames."
    )

    add_key_answer(doc,
        "What if both models are equally bad?",
        "If both models perform equally poorly on a frame (e.g., complete whiteout, total "
        "occlusion), the disagreement score D would be low (both producing similar poor "
        "results), and the proxy would classify the frame as 'easy.' This is correct behavior — "
        "if the heavier model cannot improve upon the lighter model's output, there is no "
        "benefit to switching. The framework optimizes for cases where model capacity "
        "differentially impacts performance, not for cases where no model can help."
    )

    add_key_answer(doc,
        "What about computational overhead of the switching logic itself?",
        "The switching logic overhead is negligible. The controller decision (threshold "
        "comparison + guard checks) takes approximately 0.01 ms. For the recommended conf_ema "
        "policy, the proxy computation is also essentially free (~0.01 ms) because it only "
        "requires updating two EMA values from the previous frame's mean confidence. Even for "
        "image-based proxies (Laplacian, entropy), the computation on a 160x160 proxy image "
        "takes 1-2 ms, which is less than 10% of the inference time saved by using the "
        "lightweight model. The overhead is always smaller than the savings."
    )

    add_key_answer(doc,
        "Are your parameters tuned specifically for your dataset or generalizable?",
        "The parameters were initially set based on first principles and then validated "
        "empirically. The dual-EMA structure with beta = 0.30 (fast) and beta = 0.02 (slow) "
        "corresponds to standard signal processing practice for anomaly detection (short and "
        "long moving windows). The 12%/4% thresholds were selected as natural break points "
        "in the confidence drop distribution. Sensitivity analysis shows >85% decision "
        "agreement across significant parameter variation, indicating robustness. The most "
        "important factor for generalizability is the dual-EMA normalization itself — because "
        "the slow EMA adapts to local statistics, the framework automatically adjusts to "
        "different datasets."
    )

    add_key_answer(doc,
        "What about real-time deployment on edge devices (Jetson)?",
        "The framework is designed with edge deployment in mind. The switching logic has "
        "negligible overhead (~0.01 ms). For conf_ema, no additional image processing is "
        "needed. The models (YOLOv8n and YOLOv8s) are both deployable on Jetson-class "
        "hardware, and the framework's value increases on constrained hardware where the "
        "latency difference between models is larger. On a Jetson Nano, YOLOv8n might run "
        "at 30 fps while YOLOv8s runs at 10 fps — making intelligent switching even more "
        "valuable. Both models can be loaded into GPU memory simultaneously (combined ~57 MB)."
    )

    add_key_answer(doc,
        "Why not use reinforcement learning for the switching policy?",
        "RL is a promising direction for future work but introduces significant complexity. "
        "Our handcrafted policies are interpretable (you can explain exactly why each switch "
        "happens), require no training data or reward engineering, and achieve strong results. "
        "RL would require: defining a reward function that balances latency and accuracy, "
        "collecting training episodes, handling the credit assignment problem (the benefit of "
        "switching is only apparent after processing the frame), and dealing with distribution "
        "shift across datasets. For a first investigation of dynamic model switching in this "
        "domain, establishing transparent baselines is more valuable than jumping to black-box "
        "optimization."
    )

    # ── Comparison & Literature ──
    add_heading(doc, "10.4  Comparison and Literature Questions", level=2)

    add_key_answer(doc,
        "How does this compare to early-exit networks?",
        "Early-exit networks (e.g., BranchyNet, SDN) insert classification heads at "
        "intermediate layers, allowing easy inputs to exit early. Our approach differs "
        "fundamentally: we switch between two COMPLETE models rather than exiting a single "
        "model early. The advantages of our approach are: (a) we use standard, off-the-shelf "
        "models without architectural modifications, (b) both models are independently trained "
        "and validated, and (c) the switching logic is entirely decoupled from the model "
        "architecture, making it applicable to any model pair. The disadvantage is that we "
        "cannot achieve the fine-grained compute scaling that early-exit networks offer."
    )

    add_key_answer(doc,
        "How does this compare to model cascades?",
        "Model cascades (e.g., Viola-Jones for face detection) process every input with "
        "the cheap model first, then forward uncertain cases to the expensive model. Our "
        "approach is different: we process each frame with exactly ONE model. A cascade "
        "would run both models on hard frames (nano first, then small), doubling the latency "
        "on those frames. Our approach avoids this by making the switching decision BEFORE "
        "inference, based on proxy signals, ensuring only one model runs per frame."
    )

    add_key_answer(doc,
        "What related work exists on adaptive inference?",
        "Adaptive inference is an active research area. Key related work includes: "
        "(1) Anytime prediction networks (Hu et al., 2019) that allocate variable compute; "
        "(2) Dynamic neural networks (Han et al., 2021) that adapt architecture at runtime; "
        "(3) Content-adaptive encoding in video streaming; (4) Conditional computation "
        "(Bengio et al., 2016) that skips network components. Our work is closest to "
        "runtime model selection, which has been explored for classification but is "
        "relatively understudied for detection, particularly in the UAV inspection domain."
    )

    add_key_answer(doc,
        "How does NIQE compare to full BRISQUE?",
        "NIQE and BRISQUE are both no-reference image quality metrics based on Natural "
        "Scene Statistics (NSS). BRISQUE requires training on a dataset of images with "
        "known quality scores (opinion-aware), while NIQE is completely blind — it only "
        "uses a model of pristine image statistics (opinion-unaware). We chose NIQE because "
        "it requires no domain-specific training, making it more generalizable. BRISQUE might "
        "perform better if fine-tuned on UAV-specific quality judgments, but this would "
        "reduce the framework's plug-and-play applicability. In our experiments, NIQE "
        "achieves r = +0.400 with disagreement, which is moderate but not strong enough "
        "to be the primary switching signal."
    )

    add_page_break(doc)

    # ================================================================
    # CHAPTER 11: MATHEMATICAL FOUNDATIONS REFERENCE
    # ================================================================
    add_heading(doc, "Chapter 11: Mathematical Foundations Reference", level=1)

    add_para(doc,
        "This chapter collects all mathematical formulas used in the framework in one "
        "convenient reference, with human-readable explanations and typical value ranges.",
        italic=True,
    )

    # ── Laplacian Variance ──
    add_heading(doc, "11.1  Laplacian Variance", level=2)
    add_equation_block(doc,
        "L = Var( nabla^2 I ) = (1/N) * SUM_i ( nabla^2 I(x_i) - mu )^2",
        ""
    )
    add_para(doc, "Symbol meanings:", bold=True)
    add_bullet(doc, "I: Grayscale image (2D array of pixel intensities)")
    add_bullet(doc, "nabla^2: Laplacian operator (second-order spatial derivative)")
    add_bullet(doc, "x_i: The i-th pixel location")
    add_bullet(doc, "N: Total number of pixels in the image")
    add_bullet(doc, "mu: Mean of the Laplacian response across all pixels")
    add_para(doc, "Plain English:", bold=True)
    add_para(doc,
        "Apply the edge-detection filter to the image, then compute how much the filter "
        "response varies across the image. High variance means lots of sharp edges and "
        "textures. Low variance means the image is blurry or smooth."
    )
    add_para(doc, "Typical range: 0 (completely blurred) to ~5000+ (extremely sharp/textured)")

    # ── Shannon Entropy ──
    add_heading(doc, "11.2  Shannon Entropy", level=2)
    add_equation_block(doc,
        "H = - SUM_{i=0}^{B-1} p_i * log2(p_i)",
        "Where B = number of histogram bins (default 256)"
    )
    add_para(doc, "Symbol meanings:", bold=True)
    add_bullet(doc, "p_i: Probability (normalized frequency) of intensity bin i")
    add_bullet(doc, "B: Number of histogram bins (typically 256 for 8-bit images)")
    add_bullet(doc, "log2: Base-2 logarithm (result in bits)")
    add_para(doc, "Plain English:", bold=True)
    add_para(doc,
        "How 'spread out' are the pixel brightness values? If all pixels have the same "
        "brightness (e.g., blank white image), entropy is 0. If brightness values are "
        "perfectly uniformly distributed across all 256 levels, entropy is 8 bits (maximum). "
        "Most real images fall between 5 and 8 bits."
    )
    add_para(doc, "Typical range: 4.0 to 8.0 bits")

    # ── C-Score ──
    add_heading(doc, "11.3  Combined C-Score", level=2)
    add_equation_block(doc,
        "C = alpha * L_norm + (1 - alpha) * H_norm",
        "Default alpha = 0.6 (Laplacian weighted more heavily)"
    )
    add_para(doc, "Normalization (rolling min-max):", bold=True)
    add_equation_block(doc,
        "L_norm = (L - L_min_window) / (L_max_window - L_min_window + epsilon)",
        "Where L_min_window, L_max_window are 5th/95th percentiles over 200 frames"
    )
    add_para(doc, "Plain English:", bold=True)
    add_para(doc,
        "Scale the Laplacian and entropy values to [0, 1] using a recent history window, "
        "then combine them with 60/40 weighting. The rolling normalization adapts to the "
        "video's local statistics rather than requiring global calibration."
    )
    add_para(doc, "Typical range: 0.0 to 1.0 (by construction)")

    # ── Dual EMA ──
    add_heading(doc, "11.4  Dual Exponential Moving Average (EMA)", level=2)
    add_equation_block(doc,
        "EMA_fast(t) = beta_fast * x(t) + (1 - beta_fast) * EMA_fast(t-1)",
        "Fast EMA: beta_fast = 0.30, effective window ~3 frames"
    )
    add_equation_block(doc,
        "EMA_slow(t) = beta_slow * x(t) + (1 - beta_slow) * EMA_slow(t-1)",
        "Slow EMA: beta_slow = 0.02, effective window ~50 frames"
    )
    add_para(doc, "Symbol meanings:", bold=True)
    add_bullet(doc, "x(t): Input signal at time t (e.g., mean detection confidence)")
    add_bullet(doc, "beta: Smoothing factor (higher = more responsive, less smooth)")
    add_bullet(doc, "Effective window: ~2/beta frames (how far back the average 'looks')")
    add_para(doc, "Plain English:", bold=True)
    add_para(doc,
        "Two running averages of the same signal, one fast (responds in 3 frames) and one "
        "slow (responds in 50 frames). The fast EMA captures recent behavior while the "
        "slow EMA provides a stable baseline. The gap between them signals regime changes."
    )
    add_para(doc, "Time constant derivation:", bold=True)
    add_para(doc,
        "The effective time constant tau = 1/beta. For beta = 0.30, tau = 3.3 frames. "
        "For beta = 0.02, tau = 50 frames. After 2*tau frames, the EMA has incorporated "
        "~86% of a step change. After 3*tau frames, ~95%."
    )

    # ── Conf Drop ──
    add_heading(doc, "11.5  Confidence Drop Metric", level=2)
    add_equation_block(doc,
        "conf_drop = max(0, (EMA_slow - EMA_fast) / (EMA_slow + epsilon))",
        "Relative drop in recent confidence below the long-term baseline"
    )
    add_para(doc, "Symbol meanings:", bold=True)
    add_bullet(doc, "EMA_slow: Long-term baseline confidence (slow-moving average)")
    add_bullet(doc, "EMA_fast: Recent confidence (fast-moving average)")
    add_bullet(doc, "epsilon: Small constant (1e-6) to prevent division by zero")
    add_bullet(doc, "max(0, ...): Drop is clamped to non-negative (we only care about drops, not rises)")
    add_para(doc, "Plain English:", bold=True)
    add_para(doc,
        "How much has the model's recent confidence fallen below its normal level? "
        "If the answer is 'not at all' (fast EMA >= slow EMA), the drop is 0. "
        "A value of 0.12 means recent confidence is 12% below baseline — a significant drop."
    )
    add_para(doc, "Typical range: 0.00 to ~0.50 (0.12 is the default escalation threshold)")

    # ── NIQE/MSCN ──
    add_heading(doc, "11.6  NIQE and MSCN Coefficients", level=2)
    add_equation_block(doc,
        "MSCN(x,y) = ( I(x,y) - mu_local(x,y) ) / ( sigma_local(x,y) + C )",
        "Mean-Subtracted Contrast-Normalized coefficients"
    )
    add_para(doc, "Symbol meanings:", bold=True)
    add_bullet(doc, "I(x,y): Pixel intensity at location (x,y)")
    add_bullet(doc, "mu_local: Local mean computed over a Gaussian-weighted window (7x7)")
    add_bullet(doc, "sigma_local: Local standard deviation over the same window")
    add_bullet(doc, "C: Stabilization constant (typically 1.0)")
    add_para(doc, "NIQE computation:", bold=True)
    add_para(doc,
        "The MSCN coefficients of a natural, undistorted image follow a Generalized Gaussian "
        "Distribution (GGD). NIQE fits a GGD to the MSCN coefficients, extracts the shape "
        "and variance parameters, and computes the Mahalanobis distance between these "
        "parameters and those of a pre-trained pristine image model. Higher distance = worse "
        "quality."
    )
    add_para(doc, "Typical range: 2.0 to 10.0 (lower is better quality)")

    # ── Disagreement Score ──
    add_heading(doc, "11.7  Disagreement Score", level=2)
    add_equation_block(doc,
        "D = 0.30 * det_gap_norm + 0.25 * conf_gap_norm + 0.25 * iou_disagree + 0.20 * extra_s_norm",
        "Composite inter-model disagreement"
    )
    add_para(doc, "Component formulas:", bold=True)
    add_equation_block(doc,
        "det_gap_norm = min(1.0, |N_s - N_n| / 10)",
        "Detection count gap, capped at 10 detections"
    )
    add_equation_block(doc,
        "conf_gap_norm = max(0, mean_conf_s - mean_conf_n)",
        "Confidence gap (positive when heavier model is more confident)"
    )
    add_equation_block(doc,
        "iou_disagree = 1.0 - mean_IoU(matched_pairs)",
        "Localization disagreement from Hungarian matching"
    )
    add_equation_block(doc,
        "extra_s_norm = min(1.0, N_unmatched_s / 5)",
        "Extra detections by heavier model, capped at 5"
    )
    add_para(doc, "Typical range: 0.0 (perfect agreement) to 1.0 (maximum disagreement)")

    add_page_break(doc)

    # ================================================================
    # CHAPTER 12: SUMMARY & CONCLUSIONS
    # ================================================================
    add_heading(doc, "Chapter 12: Summary & Conclusions", level=1)

    add_heading(doc, "12.1  Contributions", level=2)
    add_para(doc,
        "This research makes the following contributions to the field of adaptive inference "
        "for UAV-based object detection:"
    )

    add_bullet(doc,
        "A complete framework for real-time adaptive model switching between lightweight "
        "and heavier detectors, including scene complexity estimation, switching policy design, "
        "and hysteresis mechanisms for temporal stability.",
        bold_prefix="1. Dynamic model switching framework: ")

    add_bullet(doc,
        "Scene difficulty is defined operationally via inter-model disagreement — a fully "
        "automated, reproducible, and model-relative metric that avoids subjective labeling.",
        bold_prefix="2. Operational definition of scene difficulty: ")

    add_bullet(doc,
        "Five proxy signals (Laplacian, entropy, C-score, detection confidence, NIQE) were "
        "evaluated against the disagreement ground truth, revealing that the model's own "
        "confidence is the strongest predictor of scene difficulty.",
        bold_prefix="3. Comprehensive proxy evaluation: ")

    add_bullet(doc,
        "Histogram entropy (r = +0.172) is not a reliable proxy, and combining it with "
        "Laplacian via the C-score (r = -0.198) actually degrades the Laplacian's standalone "
        "performance (r = -0.577). These negative results warn against naive feature combination.",
        bold_prefix="4. Negative findings on entropy and C-score: ")

    add_bullet(doc,
        "Sensitivity analysis, proxy grid search, and dual-model validation provide rigorous "
        "evidence for the framework's robustness and the proxy rankings.",
        bold_prefix="5. Rigorous validation methodology: ")

    add_heading(doc, "12.2  Key Takeaway", level=2)
    add_para(doc,
        "The confidence-based switching policy (conf_ema) is the most effective and data-agnostic "
        "approach for dynamic model switching. It achieves the best latency-accuracy tradeoff "
        "by using the most direct signal available — the model's own assessment of scene "
        "difficulty — without requiring any image preprocessing or domain-specific calibration.",
        bold=True,
    )

    add_heading(doc, "12.3  Model Agnosticism", level=2)
    add_para(doc,
        "The framework is fundamentally model-agnostic. While evaluated with YOLOv8n and "
        "YOLOv8s, the core principles — scene complexity estimation, dual-EMA tracking, "
        "hysteresis-based switching — apply to any model pair where: (a) the heavier model "
        "provides better accuracy at higher latency, (b) not all frames require the heavier "
        "model, and (c) the model's confidence (or an equivalent output signal) can be "
        "monitored. This includes detector families beyond YOLO, as well as non-detection "
        "tasks (segmentation, classification) where adaptive compute allocation is beneficial."
    )

    add_heading(doc, "12.4  Value of Negative Findings", level=2)
    add_para(doc,
        "The negative findings (entropy unreliability, C-score dilution) are genuine "
        "scientific contributions. They establish what does NOT work and why, preventing "
        "future researchers from pursuing dead ends. The demonstration that a conceptually "
        "appealing feature (entropy as scene complexity) fails empirically is as valuable "
        "as demonstrating that another feature (confidence monitoring) succeeds."
    )

    add_heading(doc, "12.5  Final Statement", level=2)
    add_para(doc,
        "Dynamic model switching offers a practical, deployable approach to the "
        "latency-accuracy tradeoff in real-time object detection. By treating model "
        "selection as a per-frame decision informed by cheap proxy signals, we move beyond "
        "the static choice between 'fast but inaccurate' and 'accurate but slow' toward a "
        "system that adapts to the actual demands of each moment. The confidence-based "
        "approach, in particular, represents an elegant self-referential solution: the "
        "detector itself tells us when it needs help."
    )

    # ── Save ──
    doc.save(OUT_PATH)
    print(f"[OK] Supervisor handout saved to: {OUT_PATH}")
    print(f"     Total sections: 12 chapters")


if __name__ == "__main__":
    build_document()
