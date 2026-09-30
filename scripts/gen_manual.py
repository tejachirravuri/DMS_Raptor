"""Generate DMS-Raptor User Manual v1.1.0 and Logo Significance document."""
import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO = os.path.join(ROOT, "resources", "logo.png")
OUT_MANUAL = os.path.join(ROOT, "DMS_Raptor_User_Manual.docx")
OUT_LOGO = os.path.join(ROOT, "DMS_Raptor_Logo_Significance.docx")

TEAL = RGBColor(0x00, 0xBC, 0xD4)
DARK_TEAL = RGBColor(0x00, 0x83, 0x8F)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0x99, 0x99, 0x99)


def set_cell_shading(cell, color_hex):
    shading_elm = cell._tc.get_or_add_tcPr()
    shading = shading_elm.makeelement(qn('w:shd'), {
        qn('w:fill'): color_hex,
        qn('w:val'): 'clear',
    })
    shading_elm.append(shading)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.name = 'Arial'
        set_cell_shading(cell, '00838f')
    # Data rows
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(10)
            run.font.name = 'Arial'
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table


def add_section_header_row(table, row_idx, text, num_cols):
    """Merge cells and add a section header row to a table."""
    row = table.rows[row_idx]
    # Merge all cells
    for i in range(1, num_cols):
        row.cells[0].merge(row.cells[i])
    cell = row.cells[0]
    cell.text = ''
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = DARK_TEAL
    run.font.name = 'Arial'
    set_cell_shading(cell, 'E0F7FA')


# ===================================================================
# USER MANUAL
# ===================================================================
def build_manual():
    doc = Document()

    # --- Set default font ---
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)

    # Set heading styles
    for i in range(1, 4):
        hs = doc.styles[f'Heading {i}']
        hs.font.name = 'Arial'
        if i == 1:
            hs.font.size = Pt(16)
            hs.font.color.rgb = DARK_TEAL
        elif i == 2:
            hs.font.size = Pt(14)
        elif i == 3:
            hs.font.size = Pt(12)

    # --- Title Page ---
    doc.add_paragraph()  # spacing
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(LOGO, width=Inches(2.0))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('DMS-Raptor')
    run.font.size = Pt(28)
    run.bold = True
    run.font.color.rgb = TEAL
    run.font.name = 'Arial'

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Dynamic Model Switching for Real-Time UAV Inspection')
    run.font.size = Pt(14)
    run.font.color.rgb = GRAY
    run.font.name = 'Arial'

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.space_before = Pt(20)
    run = p.add_run('User Manual')
    run.font.size = Pt(18)
    run.bold = True
    run.font.name = 'Arial'

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Version 1.1.0')
    run.font.size = Pt(12)
    run.font.color.rgb = LIGHT_GRAY
    run.font.name = 'Arial'

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.space_before = Pt(30)
    run = p.add_run('\u00A9 2025\u20132026 Teja Chirravuri. All rights reserved.')
    run.font.size = Pt(10)
    run.font.color.rgb = LIGHT_GRAY
    run.font.name = 'Arial'

    doc.add_page_break()

    # ====================================================================
    # 1. INTRODUCTION
    # ====================================================================
    doc.add_heading('1. Introduction', level=1)
    doc.add_paragraph(
        'DMS-Raptor (Dynamic Model Switching \u2014 Raptor) is a desktop application '
        'designed for complexity-aware, real-time object detection on UAV (drone) platforms. '
        'It dynamically switches between a fast, lightweight YOLOv8n model and a more accurate '
        'YOLOv8s model based on scene complexity, balancing speed and accuracy in real time.'
    )

    doc.add_heading('1.1 Key Features', level=2)
    features = [
        'Dynamic model switching between YOLOv8n (fast) and YOLOv8s (accurate)',
        'Five configurable switching policies with hysteresis support',
        'Adaptive mode with rolling percentile-based thresholds and latency budgets',
        'Real-time monitoring with live video feed, stats panel, and rolling waveform charts',
        'Data Explorer for interactive visualization with 16 trace metrics and 11 aggregate metrics',
        'Persistent run history saved across sessions',
        'Annotated video output with per-policy MP4 files',
        'TensorRT export for NVIDIA Jetson deployment (FP16/FP32/INT8)',
        'Export capabilities: CSV, PNG plots, and annotated videos',
        'Dark-themed professional GUI built with PyQt5 and Matplotlib',
    ]
    for f in features:
        doc.add_paragraph(f, style='List Bullet')

    doc.add_heading('1.2 System Requirements', level=2)
    reqs = [
        'Python 3.10 or later',
        'PyQt5 5.15+',
        'Ultralytics YOLOv8 library',
        'OpenCV (cv2) 4.5+',
        'Matplotlib 3.5+',
        'NumPy 1.21+',
        'PyTorch 2.0+ (CPU or CUDA)',
        'Optional: NVIDIA GPU with CUDA support for accelerated inference',
        'Optional: TensorRT for Jetson deployment',
    ]
    for r in reqs:
        doc.add_paragraph(r, style='List Bullet')

    # ====================================================================
    # 2. INSTALLATION
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('2. Installation', level=1)

    doc.add_heading('2.1 Setting Up the Environment', level=2)
    doc.add_paragraph('It is recommended to use a Conda environment:')
    p = doc.add_paragraph()
    run = p.add_run('conda create -n uav-thesis python=3.10\nconda activate uav-thesis')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)

    doc.add_heading('2.2 Installing Dependencies', level=2)
    p = doc.add_paragraph()
    run = p.add_run('pip install ultralytics PyQt5 matplotlib numpy opencv-python')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)

    doc.add_paragraph('For GPU support, install the appropriate PyTorch CUDA version:')
    p = doc.add_paragraph()
    run = p.add_run('pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)

    doc.add_heading('2.3 Running the Application', level=2)
    doc.add_paragraph('Navigate to the DMS_Raptor directory and run:')
    p = doc.add_paragraph()
    run = p.add_run('python main.py')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    doc.add_paragraph('The main window will appear with three tabs: Setup, Monitor, and Results.')

    doc.add_heading('2.4 Running the Executable (Windows)', level=2)
    doc.add_paragraph(
        'A standalone .exe is available in the dist/DMS-Raptor/ folder. '
        'Double-click DMS-Raptor.exe to launch without needing Python installed. '
        'Important: Always run the .exe from the dist/DMS-Raptor/ folder (NOT from build/). '
        'The build/ folder is PyInstaller\'s intermediate workspace and does not contain all required DLLs.'
    )

    # ====================================================================
    # 3. APPLICATION OVERVIEW
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('3. Application Overview', level=1)
    doc.add_paragraph('DMS-Raptor has a tabbed interface with three main tabs:')
    doc.add_paragraph('Setup Tab \u2014 Configure input source, models, policies, parameters, and output options', style='List Bullet')
    doc.add_paragraph('Monitor Tab \u2014 Real-time video feed, live statistics, and rolling waveform charts', style='List Bullet')
    doc.add_paragraph('Results Tab \u2014 Run summary table, Quick Plots, Data Explorer, and export options', style='List Bullet')
    doc.add_paragraph('The application also includes a menu bar with File (Exit), History (Clear All History Files), and Help (About) options.')

    # ====================================================================
    # 4. SETUP TAB
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('4. Setup Tab', level=1)
    doc.add_paragraph('The Setup tab is where you configure everything before running the inference pipeline.')

    doc.add_heading('4.1 Input Source', level=2)
    doc.add_paragraph('Select one of three input modes:')
    add_table(doc,
        ['Mode', 'Description'],
        [
            ['Video File', 'Select a local video file (.mp4, .avi, .mov, .mkv, .webm). Resolution, FPS, frame count, and duration are shown.'],
            ['Live Stream', 'Use a camera index (0\u201320) or an RTSP URL (rtsp://...) for real-time streaming from webcam, IP camera, or drone feed.'],
            ['Image Folder', 'Select a folder containing image files. Each image is processed as a separate frame.'],
        ],
        col_widths=[1.5, 5.0]
    )

    doc.add_heading('4.2 Detection Models', level=2)
    doc.add_paragraph('Specify paths to two YOLO model weight files:')
    doc.add_paragraph('Fast model (n): Path to YOLOv8n .pt weights (lightweight, faster inference)', style='List Bullet')
    doc.add_paragraph('Accurate model (s): Path to YOLOv8s .pt weights (higher accuracy, slower inference)', style='List Bullet')
    doc.add_paragraph('Additional model settings:')
    doc.add_paragraph('Device: Select CPU, CUDA GPU (cuda:0, cuda:1), or Jetson for edge deployment', style='List Bullet')
    doc.add_paragraph('Image size (imgsz): Input image resolution in pixels (320\u20131280, default 640)', style='List Bullet')
    doc.add_paragraph('TensorRT Export: Enable to convert models to .engine format before running (CUDA/Jetson only)', style='List Bullet')
    doc.add_paragraph('TRT Precision: FP16 (recommended), FP32, or INT8 quantization', style='List Bullet')

    doc.add_heading('4.3 Policies', level=2)
    doc.add_paragraph('Select which switching policies to evaluate. Multiple policies can run in a single session:')
    add_table(doc,
        ['Policy', 'Description'],
        [
            ['n_only', 'Always uses YOLOv8n (fastest baseline). No switching.'],
            ['s_only', 'Always uses YOLOv8s (most accurate baseline). No switching.'],
            ['entropy_only', 'Switches based on histogram entropy (H) alone. Simple threshold comparison.'],
            ['combined', 'Uses composite complexity score C = alpha * L_norm + (1 - alpha) * H_norm. Single-threshold switching.'],
            ['combined_hyst', 'Uses composite C-score with dual hysteresis thresholds (c_low, c_high). Most stable switching policy with dead-zone.'],
        ],
        col_widths=[1.5, 5.0]
    )
    doc.add_paragraph()
    doc.add_paragraph('The Adaptive Budget option enables rolling percentile-based threshold adaptation for the combined_hyst policy when a latency budget is specified.')

    doc.add_heading('4.4 Output Options', level=2)
    doc.add_paragraph('Configure video output before running:')
    doc.add_paragraph('Save annotated output video: Enable to write overlay-annotated MP4 files to disk', style='List Bullet')
    doc.add_paragraph('Output folder: Select directory where annotated_<policy>.mp4 files will be saved', style='List Bullet')
    doc.add_paragraph('One video is generated per policy, containing the original frames with detection bounding boxes, model choice labels, and timing overlays.')

    doc.add_heading('4.5 Controller Parameters', level=2)
    doc.add_paragraph('Each parameter has an Auto checkbox. When checked, the default value is used. Uncheck to customize.')

    # Build the parameters table with section headers
    params = [
        ('__SECTION__', 'Scene Analysis', ''),
        ('alpha', '0.6', 'Weight for Laplacian (L) in composite score: C = alpha * L + (1-alpha) * H'),
        ('proxy_size', '160', 'Downscale resolution for scene complexity proxy computation'),
        ('hist_bins', '64', 'Number of histogram bins for entropy calculation'),
        ('probe_every_k', '1', 'Compute proxy every k-th frame (1 = every frame)'),
        ('__SECTION__', 'Thresholds', ''),
        ('c_low', '0.45', 'Lower hysteresis threshold for combined_hyst policy'),
        ('c_high', '0.55', 'Upper hysteresis threshold for combined_hyst policy'),
        ('combined_mid', '0.50', 'Single threshold for combined and entropy_only policies'),
        ('__SECTION__', 'Stability', ''),
        ('c_ema_beta', '0.25', 'EMA smoothing factor for composite score (lower = smoother)'),
        ('min_dwell_frames', '10', 'Minimum frames on current model before allowing a switch'),
        ('max_switches_per_100', '12', 'Maximum switches allowed per 100 frames (rate limiter)'),
        ('history_window_size', '200', 'Rolling window size for adaptive percentile computation'),
        ('__SECTION__', 'Adaptive', ''),
        ('norm_lo / norm_hi', '10 / 90', 'Percentile bounds for normalizing L and H values'),
        ('thr_lo / thr_hi / thr_mid', '35/65/50', 'Percentile bounds for adaptive threshold calculation'),
        ('latency_penalty_factor', '0.01', 'Penalty weight applied when latency exceeds budget'),
        ('budget_guard_margin', '0.10', 'Margin (10%) above budget that forces switch to fast model'),
        ('budget_guard_frames', '8', 'Frames to hold fast model after budget guard activation'),
        ('__SECTION__', 'Processing', ''),
        ('conf_min', '0.001', 'Minimum confidence threshold for YOLO prediction'),
        ('iou_nms', '0.45', 'IoU threshold for Non-Maximum Suppression'),
        ('conf_show', '0.25', 'Minimum confidence for displaying detections on overlay'),
        ('max_frames', '0', 'Maximum frames to process (0 = all frames)'),
        ('stride', '1', 'Process every N-th frame (1 = every frame)'),
        ('latency_smooth_window', '15', 'Window size for smoothed latency averaging'),
    ]

    # Regular rows (non-section)
    regular_rows = [(p, d, desc) for p, d, desc in params if p != '__SECTION__']
    table = add_table(doc,
        ['Parameter', 'Default', 'Description'],
        regular_rows,
        col_widths=[2.0, 0.8, 3.7]
    )

    doc.add_heading('4.6 Running the Pipeline', level=2)
    steps = [
        'Configure all settings in the Setup tab.',
        'Click the Run Pipeline button.',
        'The app automatically switches to the Monitor tab to show live progress.',
        'Each selected policy runs sequentially. Results appear in the Results tab.',
        'Press Stop at any time to gracefully halt the pipeline. The app stays open and partial results are preserved.',
    ]
    for i, s in enumerate(steps, 1):
        doc.add_paragraph(f'{i}. {s}')

    # ====================================================================
    # 5. MONITOR TAB
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('5. Monitor Tab', level=1)
    doc.add_paragraph('The Monitor tab provides real-time feedback during pipeline execution.')

    doc.add_heading('5.1 Live Video Feed', level=2)
    doc.add_paragraph(
        'The left area displays the annotated video frames as they are processed. '
        'Detections are shown with bounding boxes, confidence scores, and class labels. '
        'A text overlay shows the current model, frame number, and timing information.'
    )

    doc.add_heading('5.2 Stats Panel', level=2)
    doc.add_paragraph('The right panel displays live statistics updated every frame:')
    stats = [
        'Frame: Current frame index',
        'FPS: Effective frames per second (computed from T_total)',
        'Model: Currently active model (YOLOv8n in green, YOLOv8s in red)',
        'Detections: Number of objects detected in the current frame',
        'T_total: Total processing time for the current frame (ms)',
        'T_scene: Scene complexity computation time (ms)',
        'Dwell: How many consecutive frames the current model has been active',
        'C-score: Current composite complexity score (0.0\u20131.0)',
        'Policy: Name of the currently running policy',
    ]
    for s in stats:
        doc.add_paragraph(s, style='List Bullet')

    doc.add_heading('5.3 Rolling Waveform Charts', level=2)
    doc.add_paragraph('Three rolling waveform charts (500-frame window) are displayed at the bottom:')
    doc.add_paragraph('T_total (ms): Per-frame total latency over time (cyan line)', style='List Bullet')
    doc.add_paragraph('C-score: Composite complexity score with dashed threshold lines (orange line)', style='List Bullet')
    doc.add_paragraph('Model Choice: Green fill for YOLOv8n, red fill for YOLOv8s \u2014 visually shows switching patterns', style='List Bullet')
    doc.add_paragraph('Charts update every 3 frames for smooth performance.')

    # ====================================================================
    # 6. RESULTS TAB
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('6. Results Tab', level=1)
    doc.add_paragraph('After the pipeline completes, the Results tab presents all collected data.')

    doc.add_heading('6.1 Summary Table', level=2)
    doc.add_paragraph('Each policy run appears as a row with 8 columns:')
    cols = [
        'Policy: Policy name (e.g., combined_hyst)',
        'Mode: fixed or adaptive',
        'Mean T (ms): Average total processing time per frame',
        'P95 T (ms): 95th percentile latency',
        'Slow %: Percentage of frames exceeding the latency budget',
        'Sw/100: Model switches per 100 frames',
        'Frames: Total frames processed',
        'Video: Output video file size (if saved)',
    ]
    for c in cols:
        doc.add_paragraph(c, style='List Bullet')

    doc.add_heading('6.2 Quick Plots', level=2)
    doc.add_paragraph('Four pre-built analysis plots are available in the Quick Plots sub-tab:')
    plots = [
        'Timing Breakdown: Stacked bar chart showing mean T_scene, T_ctrl, T_infer_n, T_infer_s per policy',
        'Latency Comparison: Grouped bar chart comparing Mean, P95, and P99 T_total per policy',
        'T_total Waveforms: Per-frame latency traces for all policies overlaid on a single chart',
        'C-Score Traces: Composite complexity score waveforms for switching policies, with threshold lines',
    ]
    for p in plots:
        doc.add_paragraph(p, style='List Bullet')

    doc.add_heading('6.3 Data Explorer', level=2)
    doc.add_paragraph(
        'The Data Explorer is a flexible, interactive visualization tool for in-depth data analysis \u2014 '
        'similar to a simplified Tableau experience.'
    )

    doc.add_heading('6.3.1 Plot Modes', level=3)
    add_table(doc,
        ['Mode', 'Description'],
        [
            ['Trace (per-frame)', 'Plot any trace metric against any other. X and Y axes are independently selectable. Chart types: Line, Scatter, Area.'],
            ['Aggregate Comparison', 'Compare aggregate statistics across policies. Choose from 11 aggregate metrics. Chart types: Bar, Grouped Bar (Mean/P95/P99).'],
            ['Distribution', 'Analyze the statistical distribution of any trace metric. Chart types: Histogram, Box Plot, Violin Plot.'],
            ['Correlation Heatmap', 'Pearson correlation matrix across all 16 trace metrics for the selected run. Visualized as a color-coded heatmap (-1 to +1).'],
        ],
        col_widths=[2.0, 4.5]
    )

    doc.add_heading('6.3.2 Available Trace Metrics (16)', level=3)
    add_table(doc,
        ['Metric', 'Description'],
        [
            ['Frame Index', 'Sequential frame number'],
            ['T_total (ms)', 'Total per-frame processing time'],
            ['T_scene (ms)', 'Scene complexity computation time'],
            ['T_ctrl (ms)', 'Controller decision time'],
            ['T_infer_n (ms)', 'YOLOv8n inference time'],
            ['T_infer_s (ms)', 'YOLOv8s inference time'],
            ['C-score', 'Composite complexity score'],
            ['Laplacian (L)', 'Laplacian variance (edge density proxy)'],
            ['Entropy (H)', 'Histogram entropy (texture complexity proxy)'],
            ['c_low threshold', 'Lower hysteresis threshold (adaptive)'],
            ['c_high threshold', 'Upper hysteresis threshold (adaptive)'],
            ['Dwell', 'Consecutive frames on current model'],
            ['Detections', 'Number of detected objects'],
            ['Penalty', 'Latency penalty value'],
            ['Avg T_total (ms)', 'Smoothed average of T_total'],
            ['Choice (0=n, 1=s)', 'Model selection (numeric)'],
        ],
        col_widths=[2.5, 4.0]
    )

    doc.add_heading('6.3.3 Available Aggregate Metrics (11)', level=3)
    add_table(doc,
        ['Metric', 'Description'],
        [
            ['Mean T_total (ms)', 'Average processing time across all frames'],
            ['P95 T_total (ms)', '95th percentile latency'],
            ['P99 T_total (ms)', '99th percentile latency'],
            ['Slow %', 'Percentage of frames exceeding latency budget'],
            ['Switches / 100', 'Model switches per 100 frames'],
            ['Total Switches', 'Absolute number of model switches'],
            ['Mean T_scene (ms)', 'Average scene complexity computation time'],
            ['Mean T_ctrl (ms)', 'Average controller decision time'],
            ['Mean T_infer_n (ms)', 'Average YOLOv8n inference time'],
            ['Mean T_infer_s (ms)', 'Average YOLOv8s inference time'],
            ['Total Frames', 'Total number of processed frames'],
        ],
        col_widths=[2.5, 4.0]
    )

    doc.add_heading('6.3.4 Using the Data Explorer', level=3)
    explorer_steps = [
        'Select a Plot Mode from the dropdown.',
        'Choose X-Axis and Y-Axis metrics (for Trace mode) or an Aggregate Metric (for Aggregate mode).',
        'Select a Chart Type (e.g., Line, Scatter, Box Plot).',
        'Select one or more runs from the run list (all runs are selected by default).',
        'Click Generate Plot to render the visualization.',
        'Click Export This Plot to save the current chart as PNG, SVG, or PDF.',
    ]
    for i, s in enumerate(explorer_steps, 1):
        doc.add_paragraph(f'{i}. {s}')

    doc.add_heading('6.4 Summary Panel', level=2)
    doc.add_paragraph(
        'The Summary sub-tab displays a plain-text summary of all runs, including detailed timing '
        'breakdowns and output video information. The best switching policy (lowest mean T_total) '
        'is highlighted at the bottom.'
    )

    doc.add_heading('6.5 Export Options', level=2)
    exports = [
        'Export CSV: Save all run results as a CSV file with 16 columns covering all aggregate metrics',
        'Export Plots: Save all 4 Quick Plots as high-resolution PNG files (220 DPI) to a folder',
        'Export This Plot: Save the current Data Explorer chart as PNG/SVG/PDF',
        'Open Video Folder: Open the file explorer to the folder containing saved annotated videos',
        'Clear All: Clear in-session results (historical data on disk is preserved)',
    ]
    for e in exports:
        doc.add_paragraph(e, style='List Bullet')

    # ====================================================================
    # 7. PERSISTENT HISTORY
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('7. Persistent Run History', level=1)
    doc.add_paragraph(
        'DMS-Raptor automatically saves every completed policy run to disk as a JSON file. '
        'This means your results are preserved even when you close and reopen the application.'
    )

    doc.add_heading('7.1 How It Works', level=2)
    how = [
        'Each policy run is saved as a JSON file in the history/ directory inside the DMS_Raptor folder.',
        'File naming convention: {timestamp}_{policy}.json',
        'On app startup, all historical runs are automatically loaded into the Results tab.',
        'Full per-frame trace data is preserved, so the Data Explorer works with historical runs.',
    ]
    for h in how:
        doc.add_paragraph(h, style='List Bullet')

    doc.add_heading('7.2 Managing History', level=2)
    mgmt = [
        'View history: All past runs appear in the Results tab summary table on startup.',
        'Clear session: Use the Clear All button in Results to remove runs from the current view (files remain on disk).',
        'Delete all history: Use the menu History > Clear All History Files to permanently delete all saved JSON files.',
    ]
    for m in mgmt:
        doc.add_paragraph(m, style='List Bullet')

    # ====================================================================
    # 8. TENSORRT
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('8. TensorRT Optimization', level=1)
    doc.add_paragraph(
        'For maximum throughput on NVIDIA GPUs and Jetson edge devices, DMS-Raptor supports TensorRT model export.'
    )

    doc.add_heading('8.1 How to Enable', level=2)
    trt_steps = [
        'In the Setup tab, select a CUDA or Jetson device.',
        'Check the "Export to TensorRT (.engine) before run" checkbox.',
        'Select precision: FP16 (recommended for Jetson), FP32, or INT8.',
        'Click Run Pipeline. The first run exports models to .engine files (may take several minutes).',
    ]
    for i, s in enumerate(trt_steps, 1):
        doc.add_paragraph(f'{i}. {s}')
    doc.add_paragraph('Subsequent runs automatically reuse existing .engine files for instant startup.')

    doc.add_heading('8.2 Jetson Deployment', level=2)
    doc.add_paragraph(
        'When a Jetson device is detected (Tegra, Orin, Xavier), it appears as "jetson:0" in the '
        'device dropdown. The device label is internally mapped to "cuda:0", which is the standard '
        'CUDA device on Jetson boards. FP16 precision is recommended for optimal speed-accuracy tradeoff.'
    )

    # ====================================================================
    # 9. SCENE COMPLEXITY
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('9. Scene Complexity Analysis', level=1)
    doc.add_paragraph(
        'DMS-Raptor evaluates scene complexity using two lightweight proxies computed on a downscaled frame:'
    )

    doc.add_heading('9.1 Laplacian Variance (L)', level=2)
    doc.add_paragraph(
        'Measures edge density in the frame. High L indicates many edges/textures (complex scene). '
        'Computed as the variance of the Laplacian of the grayscale image.'
    )

    doc.add_heading('9.2 Histogram Entropy (H)', level=2)
    doc.add_paragraph(
        'Measures texture complexity through pixel intensity distribution. High entropy indicates '
        'a diverse distribution (complex scene). Computed on the grayscale histogram.'
    )

    doc.add_heading('9.3 Composite Score (C)', level=2)
    doc.add_paragraph('Both proxies are normalized to [0, 1] using rolling percentile bounds and combined:')
    p = doc.add_paragraph()
    run = p.add_run('C = alpha * L_norm + (1 - alpha) * H_norm')
    run.font.name = 'Consolas'
    run.font.size = Pt(11)
    run.bold = True
    doc.add_paragraph(
        'The alpha parameter (default 0.6) controls the weighting. Higher alpha gives more weight to edge density.'
    )

    # ====================================================================
    # 10. SWITCHING POLICIES DETAIL
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('10. Switching Policy Details', level=1)

    doc.add_heading('10.1 Actuator Stabilizers', level=2)
    doc.add_paragraph('All switching policies use stabilizers to prevent erratic oscillation:')
    stabilizers = [
        'EMA Smoothing: The C-score is smoothed using exponential moving average (c_ema_beta)',
        'Minimum Dwell: After switching, the current model must be used for at least min_dwell_frames before another switch',
        'Rate Limiter: No more than max_switches_per_100 switches in any 100-frame window',
        'Budget Guard: If latency exceeds budget + margin for budget_guard_frames, force switch to the fast model',
    ]
    for s in stabilizers:
        doc.add_paragraph(s, style='List Bullet')

    doc.add_heading('10.2 Adaptive Mode', level=2)
    doc.add_paragraph(
        'When Enable Adaptive Budget is checked and the combined_hyst policy is selected, the system '
        'uses rolling percentile-based thresholds that automatically adapt to the video content. '
        'A latency penalty is applied when the rolling average T_total exceeds the latency budget, '
        'biasing the system toward the faster model.'
    )

    # ====================================================================
    # 11. TROUBLESHOOTING
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('11. Troubleshooting', level=1)
    add_table(doc,
        ['Issue', 'Solution'],
        [
            ['App closes when pressing Stop', 'Upgrade to v1.1.0. This bug was fixed with a dedicated stopped_early signal and safe worker cleanup.'],
            ['"No module named ultralytics"', 'Run: pip install ultralytics'],
            ['CUDA out of memory', 'Reduce imgsz (e.g., to 416 or 320), or use CPU device.'],
            ['TensorRT export fails', 'Ensure TensorRT is installed and compatible with your CUDA version. The app falls back to .pt automatically.'],
            ['Video file not recognized', 'Ensure OpenCV can read the format. Try converting to .mp4 (H.264).'],
            ['Charts appear empty', 'Ensure at least one policy has completed. Check that switching policies are selected for C-score traces.'],
            ['History not loading', 'Check that the history/ directory exists and contains valid .json files.'],
            ['Annotated video not saving', 'Enable "Save annotated output video" in Setup and select a valid output folder.'],
            ['"Failed to load Python DLL"', 'You are running the .exe from the wrong folder. Always use dist/DMS-Raptor/DMS-Raptor.exe, NOT the one in build/.'],
        ],
        col_widths=[2.5, 4.0]
    )

    # ====================================================================
    # 12. ARCHITECTURE
    # ====================================================================
    doc.add_page_break()
    doc.add_heading('12. Architecture Overview', level=1)

    doc.add_heading('12.1 Module Structure', level=2)
    structure = [
        'DMS_Raptor/',
        '  main.py                   # Entry point',
        '  core/',
        '    config.py               # Dataclasses, constants, metrics',
        '    engine.py               # StreamingEngine (generator-based)',
        '    history.py              # Persistent JSON storage',
        '    __init__.py             # Public API exports',
        '  gui/',
        '    main_window.py          # MainWindow (tab wiring, signals)',
        '    setup_tab.py            # SetupTab (parameter UI)',
        '    monitor_tab.py          # MonitorTab (live video + charts)',
        '    results_tab.py          # ResultsTab (Data Explorer + plots)',
        '    worker.py               # InferenceWorker (QThread)',
        '  resources/',
        '    logo.svg, logo.png      # Application logo',
        '    style.qss               # Dark theme stylesheet',
        '  history/                  # Auto-created, stores run JSON files',
    ]
    for line in structure:
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.name = 'Consolas'
        run.font.size = Pt(9)

    doc.add_heading('12.2 Data Flow', level=2)
    flow = [
        'User configures parameters in SetupTab and clicks Run Pipeline.',
        'SetupTab emits run_requested signal with all parameters as a dict.',
        'MainWindow creates an InferenceWorker (QThread) with the parameters.',
        'Worker loads YOLO models, optionally exports to TensorRT, and warms up.',
        'For each policy, worker creates a StreamingEngine and iterates its generator.',
        'Each frame yields a FrameResult, emitted via frame_ready signal to MonitorTab.',
        'On policy completion, a RunSummary is emitted via policy_finished signal.',
        'MainWindow saves the RunSummary to history/ and adds it to ResultsTab.',
        'When all policies complete, all_finished signal triggers switch to Results tab.',
    ]
    for i, s in enumerate(flow, 1):
        doc.add_paragraph(f'{i}. {s}')

    doc.add_heading('12.3 Key Design Patterns', level=2)
    patterns = [
        'Generator-based streaming: StreamingEngine.run() yields FrameResult per frame, enabling clean iteration',
        'Signal-slot architecture: PyQt5 signals decouple the worker thread from the GUI thread',
        'Graceful stop: Dedicated stopped_early signal with safe thread cleanup prevents crashes',
        'Persistent history: JSON serialization of RunSummary with to_dict() / from_dict()',
    ]
    for p in patterns:
        doc.add_paragraph(p, style='List Bullet')

    # END
    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.space_before = Pt(100)
    run = p.add_run('End of User Manual')
    run.font.size = Pt(14)
    run.font.color.rgb = LIGHT_GRAY
    run.italic = True
    run.font.name = 'Arial'

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('DMS-Raptor v1.1.0 \u2014 \u00A9 2025\u20132026 Teja Chirravuri')
    run.font.size = Pt(11)
    run.font.color.rgb = LIGHT_GRAY
    run.font.name = 'Arial'

    doc.save(OUT_MANUAL)
    print(f'User Manual saved: {OUT_MANUAL}')


# ===================================================================
# LOGO SIGNIFICANCE DOCUMENT
# ===================================================================
def build_logo_doc():
    doc = Document()

    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)

    for i in range(1, 3):
        hs = doc.styles[f'Heading {i}']
        hs.font.name = 'Arial'
        if i == 1:
            hs.font.size = Pt(16)
            hs.font.color.rgb = DARK_TEAL
        elif i == 2:
            hs.font.size = Pt(14)

    # Logo image
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(LOGO, width=Inches(2.8))

    # Title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('DMS-Raptor Logo')
    run.font.size = Pt(24)
    run.bold = True
    run.font.color.rgb = TEAL
    run.font.name = 'Arial'

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Design Significance & Symbolism')
    run.font.size = Pt(14)
    run.font.color.rgb = GRAY
    run.font.name = 'Arial'

    doc.add_paragraph()

    # Overview
    doc.add_heading('Overview', level=1)
    doc.add_paragraph(
        'The DMS-Raptor logo is a hexagonal raptor-eye targeting reticle that encapsulates the '
        'core philosophy of the application: precision, intelligence, and speed in UAV-based object detection.'
    )

    # Design Elements
    doc.add_heading('Design Elements', level=1)

    doc.add_heading('1. Hexagonal Frame', level=2)
    doc.add_paragraph(
        'The outer hexagonal shape represents the structured, systematic nature of the dynamic model '
        'switching framework. Hexagons are a recurring motif in aerospace and defense HUDs (Heads-Up '
        'Displays), connecting the logo to its UAV context. The teal-colored (#00bcd4) hexagonal border '
        'also evokes drone rotors and technological precision.'
    )

    doc.add_heading('2. Raptor Eye (Central Circle)', level=2)
    doc.add_paragraph(
        'The circular element at the center represents the eye of a raptor (bird of prey). Raptors like '
        'eagles and hawks are known for their extraordinary visual acuity \u2014 they can spot prey from '
        'great heights with incredible precision. This mirrors how DMS-Raptor analyzes every frame to '
        'detect objects with optimal accuracy. The dark inner circle (#1a1a2e) surrounded by teal '
        'symbolizes focused vision peering through the darkness.'
    )

    doc.add_heading('3. Targeting Reticle / Crosshair', level=2)
    doc.add_paragraph(
        'The four crosshair lines extending from the center represent a targeting reticle, as seen in '
        'military and surveillance optics. This symbolizes the precision-targeting nature of object '
        'detection \u2014 finding and localizing objects in each frame. The crosshair also implies the '
        'system is always actively scanning and making decisions.'
    )

    doc.add_heading('4. Corner Brackets', level=2)
    doc.add_paragraph(
        'The L-shaped corner brackets at each corner of the hexagon represent a bounding box \u2014 the '
        'fundamental output of object detection models. Every detection drawn on screen is a bounding box, '
        'and these corner elements are a direct visual metaphor for that. They also suggest a viewfinder or '
        'camera frame, reinforcing the real-time video processing context.'
    )

    doc.add_heading('5. Red Accent Dot', level=2)
    doc.add_paragraph(
        'The small red circle (#e94560) at the center is a deliberate contrast element. It represents the '
        'critical decision point where the system must choose between the fast model (YOLOv8n) and the '
        'accurate model (YOLOv8s). The red color conveys urgency and importance \u2014 this is the moment '
        'of dynamic switching. It also serves as a focal point that draws the eye inward, just as the '
        'algorithm focuses on the most critical frames.'
    )

    # Color Palette
    doc.add_heading('Color Palette', level=1)
    add_table(doc,
        ['Color', 'Hex Code', 'Symbolism'],
        [
            ['Teal', '#00bcd4', 'Technology, clarity, intelligence, UAV operations'],
            ['Red Accent', '#e94560', 'Critical decision points, urgency, active detection'],
            ['Dark Navy', '#1a1a2e', 'Night operations, stealth, the application\'s dark theme'],
        ],
        col_widths=[1.3, 1.3, 3.9]
    )

    # Name Significance
    doc.add_heading('Name Significance', level=1)
    p = doc.add_paragraph()
    run = p.add_run('DMS')
    run.bold = True
    run.font.name = 'Arial'
    run = p.add_run(
        ' stands for Dynamic Model Switching \u2014 the core algorithm that switches between detection '
        'models in real time. '
    )
    run.font.name = 'Arial'
    run = p.add_run('Raptor')
    run.bold = True
    run.font.name = 'Arial'
    run = p.add_run(
        ' refers to birds of prey known for their exceptional vision and hunting precision. Together, '
        'DMS-Raptor means a system that sees like a raptor and switches like a reflex.'
    )
    run.font.name = 'Arial'

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('\u00A9 2025\u20132026 Teja Chirravuri. All rights reserved.')
    run.font.size = Pt(10)
    run.font.color.rgb = LIGHT_GRAY
    run.font.name = 'Arial'

    doc.save(OUT_LOGO)
    print(f'Logo document saved: {OUT_LOGO}')


if __name__ == '__main__':
    build_manual()
    build_logo_doc()
    print('DONE')
