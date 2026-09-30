# DMS-Raptor Proxy Validation Methodology

## The Problem: Validating Scene Complexity Proxies

DMS-Raptor uses two lightweight image statistics — **Laplacian variance (L)** and
**histogram entropy (H)** — combined into a composite **C-score** to decide
whether a scene is "simple" (use fast YOLOv8n) or "complex" (use accurate
YOLOv8s).

The scientific question is: **Do L and H actually predict when the heavier
model is needed?**

---

## Key Insight: Ground Truth Without Annotations

We define scene "difficulty" **operationally** as the disagreement between
the two models.  No external bounding-box annotations are required.

| Condition | Interpretation |
|-----------|---------------|
| Both models produce similar detections | Scene is **easy** — n was sufficient |
| s finds more / higher-confidence detections than n | Scene is **hard** — s was genuinely needed |

This is **not circular** because:
- C-score is computed from **pixel statistics** (L, H)
- Disagreement is computed from **model outputs** (detection counts, confidences, IoU)
- These are **independent measurements** on the same frame

---

## Validation Pipeline

### Step 1 — Dual-Model Inference (per frame)

For every frame in the test video, run **both** YOLOv8n and YOLOv8s and
record their raw outputs.

### Step 2 — Compute Disagreement Metrics (per frame)

| Metric | Formula | What It Captures |
|--------|---------|-----------------|
| **Detection Count Gap** | `det_s − det_n` | s finds objects n missed |
| **Mean Confidence Gap** | `mean_conf_s − mean_conf_n` | s is more certain |
| **Max Confidence Gap** | `max_conf_s − max_conf_n` | s's best detection vs n's best |
| **IoU Agreement** | mean IoU of matched boxes (Hungarian) | Spatial alignment of detections |
| **Extra Detections (s)** | count of s-detections with no n-match | Objects only s found |
| **Extra Detections (n)** | count of n-detections with no s-match | False positives by n |

### Step 3 — Compute Scene Proxies (per frame)

At multiple proxy settings to test sensitivity:
- Proxy size: 160, 320, 640 (or full resolution)
- Histogram bins: 64, 128, 256

Record L, H, C for each setting.

### Step 4 — Correlation Analysis

Compute Pearson and Spearman rank correlations between:
- C-score ↔ Detection Count Gap
- C-score ↔ Confidence Gap
- C-score ↔ IoU Agreement (expect negative — higher C, lower agreement)
- L ↔ each disagreement metric
- H ↔ each disagreement metric

### Step 5 — Sensitivity Analysis

Compare switching decisions across proxy parameter settings.  If decisions are
stable (>90 % agreement), the 160/64 setting retains sufficient discriminative
power despite the compression.

### Step 6 — Reporting

Generate:
- Correlation coefficient table (Pearson r, Spearman ρ, p-values)
- Scatter plots: C vs each disagreement metric
- Time-series overlay: C-score + disagreement over frames
- Sensitivity matrix: agreement % across proxy settings
- Full per-frame CSV for external analysis (Excel, R, Python notebooks)

---

## How to Interpret Results

| Result | What It Means |
|--------|--------------|
| Strong positive correlation (r > 0.5) between C and disagreement | L,H are **valid** proxies — high C correctly identifies hard scenes |
| Weak/no correlation (|r| < 0.3) | L,H are **poor** proxies — switching may be near-random |
| Negative correlation between C and IoU agreement | Expected and **good** — complex scenes = models disagree spatially |
| Proxy settings show >90% agreement | 160/64 is sufficient — no information loss that matters |
| Proxy settings diverge significantly | Higher resolution needed — 160/64 loses discriminative detail |

---

## Thesis Argument Structure

1. **Define** scene difficulty as inter-model disagreement (operationally valid,
   no annotations needed)
2. **Validate** that C-score correlates with this difficulty measure
3. **Justify** proxy parameters (160/64) via sensitivity analysis
4. **Conclude**: even with approximate proxies, the DMS system achieves the
   target latency-accuracy tradeoff — and we have evidence the proxies
   capture meaningful signal about detection difficulty

---

## Files in This Module

| File | Purpose |
|------|---------|
| `validator.py` | Dual-model inference engine, per-frame metrics |
| `report.py` | Correlation analysis, plots, CSV export |
| `METHODOLOGY.md` | This document |
