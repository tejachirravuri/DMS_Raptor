# DMS-Raptor Thesis Reference Document

**Last updated:** 2026-04-02
**Author:** Ganapathi Teja Chirravuri
**University:** Technische Universitat Chemnitz, Faculty of Computer Science
**Paper Title:** "Dynamic Model Switching for Real-Time UAV Inspection: From Proxy Failures to Confidence-Based Adaptation"

---

## 1. PROJECT OVERVIEW

**DMS-Raptor** is a Dynamic Model Switching framework that adaptively selects between YOLOv8n (nano, fast) and YOLOv8s (small, accurate) on a per-frame basis for real-time UAV power-line insulator inspection.

### Core Idea
Instead of running one fixed model on all frames, DMS analyzes each frame's scene complexity and routes it to the appropriate model:
- **Simple frames** (clear sky, far insulators) -> YOLOv8n (~170ms, good enough)
- **Complex frames** (clutter, reflections, close-ups) -> YOLOv8s (~256ms, more accurate)

### Dataset
- **12 UAV inspection videos** (9 glass + 3 porcelain insulators)
- **78,225 total frames**
- Models trained per material: glass_y8n/y8s, porcelain_y8n/y8s, composite_y8n/y8s
- No ground-truth bounding box annotations; uses **pseudo-oracle validation** (s_only detections as ground truth)

---

## 2. THE 8 SWITCHING POLICIES

### Thesis Analysis Policies (offline pipeline — `core/analysis_engine.py`)

| # | Policy | Type | Mechanism | Key Parameters |
|---|--------|------|-----------|----------------|
| 1 | `n_only` | Baseline | Always use YOLOv8n | None |
| 2 | `s_only` | Baseline | Always use YOLOv8s (oracle) | None |
| 3 | `conf_ema` | Reactive | Dual-EMA on n-model's detection confidence | fast_beta=0.30, slow_beta=0.02, c_high=0.12, c_low=0.04 |
| 4 | `local_contrast_hyst` | Proactive | Structural-texture proxy (local contrast RMS) with hysteresis | c_low=0.45, c_high=0.55 |
| 5 | `combined_hyst` | Proactive | C-score with 3-threshold hysteresis | c_low=0.45, c_high=0.55 |
| 6 | `entropy_only` | Proactive | Switch on Shannon entropy vs median | H_median threshold |
| 7 | `combined` | Proactive | C-score single threshold | c_mid=0.50, alpha=0.6 |
| 8 | `multi_proxy` | Proactive | 4-proxy weighted composite (L+H+Tenengrad+ColorEntropy) with EMA-relative normalization | mp_c_high=0.10, mp_c_low=0.03, proxy weights cross-validated |

### Live Demonstrator Policies (real-time GUI — `core/engine.py`)

Same as thesis analysis except: **replaces `local_contrast_hyst` with `niqe_switch`** (NIQE no-reference image quality, niqe_c_high=0.003, niqe_c_low=0.001). `local_contrast_hyst` is thesis-analysis-only; `niqe_switch` is live-engine-only.

### Policy Categories
- **Proactive (image-only):** entropy_only, combined, combined_hyst, local_contrast_hyst, multi_proxy -- analyze the image BEFORE inference to decide which model to run
- **Reactive (inference-output):** conf_ema -- uses the PREVIOUS frame's model confidence to decide for the CURRENT frame (zero image processing cost, T_scene=0)

### Actuator Stabilizers (applied to all switching policies)
- `min_dwell_frames=10` -- minimum frames before allowing a switch
- `max_switches_per_100=12` -- rate limiter
- `c_ema_beta=0.25` -- EMA smoothing on C-score

---

## 3. TIMING ARCHITECTURE

### Per-Frame Breakdown
```
T_total = T_scene + T_ctrl + T_infer   (only ONE model runs per frame)
```

| Component | What it measures | Typical range |
|-----------|-----------------|---------------|
| **T_scene** | Image proxy computation (resize + Laplacian/Entropy/contrast/etc.) | 0ms (conf_ema) to ~5ms (scene proxies) |
| **T_ctrl** | Controller logic (normalization, EMA, thresholds, hysteresis, dwell guard) | ~0.05-0.2ms |
| **T_infer_n** | YOLOv8n forward pass + internal NMS (if n chosen) | ~170ms (CPU) |
| **T_infer_s** | YOLOv8s forward pass + internal NMS (if s chosen) | ~256ms (CPU) |

### Critical Bug Fixed (2026-04-02)
**Redundant Python NMS removed from `run_model_single()`**: YOLO's `.predict()` already applies C++/CUDA NMS internally. A second O(n^2) Python NMS was running on the output, which made n_only SLOWER than s_only because n produces more low-confidence boxes (236 vs 100 at conf_min=0.001). After fix: n_only=170ms, s_only=256ms (correct ratio).

### Timing Sources
- **CORRECT timing**: `03_repeated_trials.py` and `09_overnight_full_pipeline.py` -- use `StreamingEngine` which runs ONE model per frame
- **INFLATED timing**: `02_detection_quality.py` -- runs BOTH models on every frame and computes ALL proxies (needed for detection coverage comparison, but timing is not representative)

---

## 4. DETECTION QUALITY METRICS

Using s_only as pseudo-oracle (ground truth):

| Policy | Precision | Recall (Coverage) | F1 | Conf-Weighted Coverage |
|--------|-----------|-------------------|-----|----------------------|
| n_only | 76.6% | 79.5% | 78.0% | -- |
| s_only | 100.0% | 100.0% | 100.0% | 100.0% |
| entropy_only | 88.6% | 89.6% | 89.1% | -- |
| combined | 87.5% | 88.9% | 88.2% | -- |
| **combined_hyst** | **88.2%** | **89.3%** | **88.8%** | -- |
| conf_ema | 81.1% | 85.8% | 83.3% | -- |
| ~~niqe_switch~~ | 77.6% | 80.4% | 79.0% | -- | *(retired — negligible impact)* |
| multi_proxy | 79.7% | 83.5% | 81.6% | -- |

**Source:** `research_results/02_detection_quality/all_detection_quality.json` (12 videos, 78,225 frames)
**Note:** niqe_switch retired in v1.8.0; replaced by local_contrast_hyst in thesis-canonical policy set.

---

## 5. CORRECT AGGREGATE TIMING (from StreamingEngine)

From `03_repeated_trials.py` (5 videos x 3 trials each):

| Policy | Mean T_total (ms) | Relative to s_only |
|--------|-------------------|-------------------|
| n_only | 128 | 0.50x (baseline fast) |
| s_only | 257 | 1.00x (baseline slow) |
| entropy_only | 198 | 0.77x |
| combined | 206 | 0.80x |
| combined_hyst | 205 | 0.80x |
| conf_ema | 187 | 0.73x (27% faster than s_only) |
| ~~niqe_switch~~ | 158 | 0.61x | *(retired)* |
| multi_proxy | 172 | 0.67x |

**Note:** After the redundant NMS fix (Apr 2), fresh timing from `09_overnight_full_pipeline.py` on 633-01 (200 frames) gives: n_only=169ms, s_only=256ms, conf_ema=178ms, combined_hyst=237ms. These are single-video numbers; the repeated_trials numbers are 5-video averages with proper warmup.

---

## 6. KEY RESEARCH FINDINGS

### Finding 1: C-Score Normalization Failure (Central Contribution)
Raw Laplacian (L) correlates moderately with model disagreement: |r|=0.39 (glass), |r|=0.73 (porcelain). But rolling percentile normalization destroys this signal: C-score correlation drops to |r|=0.098 (glass), |r|=0.054 (porcelain). This is because normalization compresses values to [0,1] within a sliding window, losing the absolute magnitude that matters.

### Finding 2: combined_hyst Works Despite Weak Proxy
Despite the weak C-score signal, combined_hyst achieves 89% detection coverage by using s-model for 50% of frames. The hysteresis mechanism provides stability (only 0.83 switches/100 frames). It works not because the proxy is accurate, but because it's conservative enough to use s when uncertain.

### Finding 3: conf_ema is the Best Speed-Accuracy Trade-off
- 86% coverage at 187ms (27% faster than s_only)
- Best cross-video generalization: T_cv=1.0% (most consistent latency across videos)
- Zero image processing cost (T_scene=0ms) -- uses previous frame's confidence
- Only 27% s-model usage

### Finding 4: multi_proxy Under-Switches
Despite using 4 image proxies with cross-validated weights, multi_proxy only achieves 84% coverage with 21% s-usage. The EMA-relative normalization works better than percentile normalization, but the composite signal still isn't strong enough for aggressive switching.

### Finding 5: Cross-Material Generalization
- conf_ema generalizes best across glass/porcelain (T_ratio=0.96, T_cv=1.0%)
- combined_hyst shows T_cv=16.6% (more variable across videos)
- Porcelain is an easier task (y8n F1=0.962 > y8s F1=0.958) so DMS benefit is limited there

---

## 7. PROJECT FILE STRUCTURE

```
G:\Teja_Master_Thesis\
|
+-- DMS_Raptor\                          # Main codebase (v1.7.0)
|   +-- core\
|   |   +-- config.py                    # RunConfig, POLICIES, InferenceParams, FrameResult, RunSummary
|   |   +-- engine.py                    # StreamingEngine, draw_overlay, run_model_single, complexity_proxies
|   |   +-- niqe.py                      # NIQE no-reference image quality score
|   |   +-- history.py                   # Session history management
|   |
|   +-- gui\                             # 7-tab PyQt5 GUI (14 files)
|   |   +-- main_window.py              # Main application window
|   |   +-- setup_tab.py                # Model/video configuration
|   |   +-- thesis_tab.py               # Thesis experiment runner
|   |   +-- results_tab.py              # Results explorer
|   |   +-- thesis_plots.py             # 20+ thesis plot types
|   |   +-- monitor_tab.py              # Real-time monitoring
|   |   +-- (+ 8 more files)
|   |
|   +-- research\                        # 10 research scripts (~5000 LOC)
|   |   +-- 01_model_baselines.py        # Phase 1A: YOLO validation on glass/porcelain/composite
|   |   +-- 02_detection_quality.py      # Phase 1B: Detection coverage (s_only as oracle), all 12 videos
|   |   +-- 03_repeated_trials.py        # Phase 2: 3-trial statistical analysis (CORRECT timing source)
|   |   +-- 04_ablation_study.py         # Phase 4: L-only, H-only, combined variants
|   |   +-- 05_cross_video_generalization.py  # Phase 5: LOO + cross-material
|   |   +-- 06_cscore_analysis.py        # Phase 6: Proxy correlation, normalization failure analysis
|   |   +-- 07_generate_figures.py       # Phase 7: 50 publication-quality figures
|   |   +-- 08_generate_annotated_videos.py   # Phase 8: Annotated video generator
|   |   +-- 09_overnight_full_pipeline.py     # Phase 9: Full overnight pipeline (all 8 policies x 12 videos)
|   |   +-- 10_validation_analysis.py    # Phase 10: F1, precision/recall, proxy-routing correlation
|   |   +-- run_all.py                   # Master runner (--quick, --phase)
|   |
|   +-- research_results\                # All experiment outputs
|   |   +-- 01_model_baselines\          # Baseline F1/mAP/speed JSONs
|   |   +-- 02_detection_quality\        # Detection coverage across 12 videos
|   |   +-- 03_repeated_trials\          # 3-trial timing statistics
|   |   +-- 04_ablation\                 # Ablation component results
|   |   +-- 05_generalization\           # Cross-video/material results
|   |   +-- 06_cscore_analysis\          # Proxy correlation data
|   |   +-- figures\                     # 50 PNGs (7 categories)
|   |   +-- overnight_full\              # Full pipeline outputs (per-video x per-policy)
|   |   +-- 10_validation\               # Validation analysis outputs
|   |   +-- annotated_videos\            # MP4s with overlay
|   |
|   +-- validation\                      # Validation framework
|   +-- resources\                       # Logo, stylesheets
|   +-- main.py                          # PyQt5 entry point
|   +-- THESIS_REFERENCE.md              # THIS FILE
|
+-- IEEE_Paper_DMS\                      # IEEE paper
|   +-- DMS_UAV_Inspection.tex           # Main LaTeX source
|   +-- DMS_UAV_Inspection.pdf           # Compiled paper
|   +-- figs\                            # 26 figures used in paper
|
+-- MT_Chirravuri\                       # Models and data
|   +-- models\                          # 6 YOLO .pt weights (glass/porcelain/composite x n/s)
|   +-- input_videos\
|   |   +-- glass_insulator_videos\      # 9 videos
|   |   +-- porcelain_insulator_videos\  # 3 videos
|   +-- dms_overnight_pipeline.py        # Legacy overnight script (5 policies only)
|
+-- dms_outputs_overnight\               # Legacy overnight results (5 policies, old engine)
    +-- glass\                           # 7 videos x 5 policies
    +-- porcelain\                       # 3 videos x 5 policies
```

---

## 8. MODEL BASELINES

| Dataset | Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Infer (ms) |
|---------|-------|-----------|--------|-----|-------|----------|------------|
| Glass | y8n | 0.881 | 0.860 | 0.870 | 0.902 | 0.579 | 97.1 |
| Glass | y8s | 0.917 | 0.869 | 0.892 | 0.927 | 0.598 | 233.6 |
| Porcelain | y8n | 0.957 | 0.967 | 0.962 | 0.977 | 0.877 | 84.9 |
| Porcelain | y8s | 0.952 | 0.964 | 0.958 | 0.979 | 0.882 | 213.1 |
| Composite | y8n | 0.994 | 0.923 | 0.957 | 0.946 | 0.735 | 124.6 |
| Composite | y8s | 0.951 | 0.895 | 0.922 | 0.955 | 0.726 | 285.4 |

**Key insight:** Glass has meaningful n-vs-s accuracy gap (F1: 0.870 vs 0.892). Porcelain gap is negligible (0.962 vs 0.958), limiting DMS benefit.

---

## 9. WHAT HAS BEEN ACHIEVED

### Code & Infrastructure
- [x] StreamingEngine with all 8 policies (core/engine.py)
- [x] 7-tab PyQt5 GUI application (v1.7.0)
- [x] 10-phase research pipeline (research/*.py)
- [x] Pseudo-oracle validation framework (no ground-truth needed)
- [x] Per-material model pairs (glass_y8n/y8s, porcelain_y8n/y8s)
- [x] PyInstaller build system
- [x] Zero-detection gate optimization (force n when no objects visible)
- [x] Redundant NMS bug fix (Apr 2 -- n_only was slower than s_only due to O(n^2) Python NMS)

### Research Results
- [x] Phase 1A: Model baselines on glass/porcelain/composite (01_model_baselines.py)
- [x] Phase 1B: Detection quality across all 12 videos, 78,225 frames (02_detection_quality.py)
- [x] Phase 2: Repeated trials 5 videos x 3 trials for timing variance (03_repeated_trials.py)
- [x] Phase 4: Ablation study -- L-only, H-only, combined variants (04_ablation_study.py)
- [x] Phase 5: Cross-video generalization + cross-material transfer (05_cross_video_generalization.py)
- [x] Phase 6: C-score failure analysis + proxy correlation (06_cscore_analysis.py)
- [x] Phase 7: 50 publication-quality figures generated (07_generate_figures.py)
- [x] Phase 10: Validation analysis -- F1/Precision/Recall computed (10_validation_analysis.py)

### IEEE Paper
- [x] Full paper written (LaTeX, ~38KB)
- [x] Abstract with correct numbers
- [x] 26 figures integrated
- [x] Tables with correct StreamingEngine timing (not inflated detection_quality timing)
- [x] Discussion of C-score failure as central methodological contribution
- [x] PDF compiled

### Annotated Videos
- [x] draw_overlay() with thick green (n) / red (s) borders
- [x] Annotated video generator (08_generate_annotated_videos.py)
- [x] Full overnight pipeline with selective MP4 generation (09_overnight_full_pipeline.py)

---

## 10. WHAT IS PENDING

### Critical (Must Complete)
- [ ] **Full overnight pipeline run**: All 8 policies x 12 videos with correct engine (redundant NMS fix applied). Previous results used old code with inflated timing. Command: `python research/09_overnight_full_pipeline.py --annotate-policies conf_ema multi_proxy`
- [ ] **Re-run 03_repeated_trials.py** with NMS fix to get corrected aggregate timing numbers
- [ ] **Re-run 02_detection_quality.py** -- detection coverage numbers should be unaffected by NMS fix, but need to verify
- [ ] **Update IEEE paper numbers** if timing changes significantly after NMS fix
- [ ] **Full validation analysis** (10_validation_analysis.py) after overnight pipeline completes -- produces proxy-routing correlation, proxy-disagreement correlation, cross-video consistency, LaTeX table

### Important (Should Complete)
- [ ] **Zero-detection gate ablation**: Run overnight pipeline WITH --zero-det-gate and compare s-model usage reduction vs baseline
- [ ] **Annotated videos for switching policies** (conf_ema, local_contrast_hyst, multi_proxy) across representative videos
- [ ] **Re-generate 07_generate_figures.py** output after NMS fix to ensure all 50 figures use correct timing
- [ ] **Grand comparison table**: side-by-side old overnight results (5 policies) vs new StreamingEngine results (8 policies)

### Nice-to-Have
- [ ] Jetson Nano deployment and real-device timing
- [ ] Additional videos from different inspection scenarios
- [ ] conf_ema threshold sensitivity analysis
- [ ] Per-frame detection count export for zero-detection gate impact analysis

---

## 11. CRITICAL BUGS FOUND AND FIXED

### Bug 1: Inflated Timing in 02_detection_quality.py (Found ~Mar 29)
**Problem:** `02_detection_quality.py` runs BOTH models on every frame and computes ALL proxies, inflating T_scene from ~11ms to ~59ms and adding cross-model cache pollution.
**Impact:** combined_hyst was reported as 311ms (inflated) vs 205ms (correct).
**Fix:** `07_generate_figures.py` modified to load timing from `03_repeated_trials/repeated_trials.json` (StreamingEngine) instead of detection_quality. Paper numbers updated.

### Bug 2: Redundant O(n^2) Python NMS (Found Apr 2)
**Problem:** `run_model_single()` in `core/engine.py` applied a second Python NMS after YOLO's internal C++ NMS. With `conf_min=0.001`, n-model produces ~236 boxes vs s-model's ~100. The O(n^2) NMS cost (55,696 vs 10,000 IoU ops) made n_only SLOWER than s_only (340ms vs 333ms).
**Fix:** Removed `nms_xyxy()` call from `run_model_single()`. Result: n_only=169ms, s_only=256ms (correct ratio).
**Impact:** ALL previous timing numbers are inflated. Repeated trials and overnight pipeline need re-running.

### Bug 3: Border Visibility in Annotated Videos (Found ~Apr 1)
**Problem:** `draw_overlay()` border was 4px (barely visible at 1080p).
**Fix:** Dynamic border = max(8, min(h,w)//50) with alpha blending. Green for n-model, red for s-model.

---

## 12. KEY COMMANDS

```bash
# Activate environment
conda activate uav-thesis

# Run full overnight pipeline (all 8 policies, all 12 videos)
cd G:\Teja_Master_Thesis\DMS_Raptor
python research/09_overnight_full_pipeline.py --annotate-policies conf_ema multi_proxy

# Run with zero-detection gate
python research/09_overnight_full_pipeline.py --annotate-policies conf_ema multi_proxy --zero-det-gate

# Run validation analysis (after overnight completes)
python research/10_validation_analysis.py

# Run specific phases
python research/run_all.py --phase 1a        # baselines only
python research/run_all.py --phase 1b        # detection quality (slow, 8-12h)
python research/run_all.py --phase 2         # repeated trials
python research/run_all.py --quick           # all phases with stride=5

# Generate figures
python research/07_generate_figures.py

# Test on single video
python research/09_overnight_full_pipeline.py --videos 633-01 --max-frames 500

# Launch GUI
python main.py
```

---

## 13. PAPER STRUCTURE

1. **Introduction** -- UAV inspection challenge, DMS motivation
2. **Related Work** -- UAV inspection ML, adaptive inference, model switching
3. **Methodology**
   - System overview (n/s model pairs, per-material)
   - Scene complexity proxies (L, H, NIQE, Tenengrad, etc.)
   - C-score formulation and normalization
   - conf_ema reactive policy
   - multi_proxy proactive policy
   - Hysteresis and actuator stabilizers
   - Experimental setup (12 videos, pseudo-oracle validation)
4. **Results**
   - Model baselines (Table I)
   - Detection quality comparison (Table III -- main results table)
   - Timing analysis
   - Ablation study
   - Cross-video generalization
   - C-score failure analysis
5. **Discussion** -- Why conf_ema is best trade-off, limitations
6. **Conclusion**

---

## 14. ENVIRONMENT

- **Python environment:** conda `uav-thesis`
- **Python location:** `C:\Users\tejac\anaconda3\envs\uav-thesis\python.exe`
- **Key packages:** ultralytics (YOLO), PyQt5, opencv-python, matplotlib, numpy
- **Hardware:** CPU inference (no GPU timing available)
- **OS:** Windows (paths use G:\ drive)
