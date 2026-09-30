/**
 * Generate DMS-Raptor User Manual v1.1.0 using docx-js.
 * Run with: node scripts/gen_manual.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, PageOrientation,
  LevelFormat, ExternalHyperlink, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
  TableOfContents,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const LOGO_PATH = path.join(ROOT, "resources", "logo.png");
const OUT_MANUAL = path.join(ROOT, "DMS_Raptor_User_Manual.docx");
const OUT_LOGO = path.join(ROOT, "DMS_Raptor_Logo_Significance.docx");

const logoData = fs.readFileSync(LOGO_PATH);

// --- Shared helpers ---
const TEAL = "00bcd4";
const DARK = "1a1a2e";
const BORDER = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const CELL_MARGINS = { top: 60, bottom: 60, left: 100, right: 100 };

function heading1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 200 }, children: [new TextRun({ text, bold: true, size: 32, font: "Arial" })] });
}
function heading2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 160 }, children: [new TextRun({ text, bold: true, size: 28, font: "Arial" })] });
}
function heading3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 120 }, children: [new TextRun({ text, bold: true, size: 24, font: "Arial" })] });
}
function para(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text, size: 22, font: "Arial", ...opts })] });
}
function paraMulti(runs) {
  return new Paragraph({ spacing: { after: 120 }, children: runs.map(r => typeof r === "string" ? new TextRun({ text: r, size: 22, font: "Arial" }) : new TextRun({ size: 22, font: "Arial", ...r })) });
}
function bullet(text, ref = "bullets") {
  return new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text, size: 22, font: "Arial" })] });
}
function bullet2(text, ref = "bullets") {
  return new Paragraph({ numbering: { reference: ref, level: 1 }, spacing: { after: 40 }, children: [new TextRun({ text, size: 22, font: "Arial" })] });
}
function numberedItem(text, ref = "numbers") {
  return new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text, size: 22, font: "Arial" })] });
}
function code(text) {
  return new Paragraph({ spacing: { after: 80 }, indent: { left: 360 }, children: [new TextRun({ text, size: 20, font: "Consolas" })] });
}
function headerCell(text, w) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders: BORDERS, margins: CELL_MARGINS,
    shading: { fill: "00838f", type: ShadingType.CLEAR },
    children: [new Paragraph({ children: [new TextRun({ text, size: 20, font: "Arial", bold: true, color: "FFFFFF" })] })],
  });
}
function cell(text, w) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders: BORDERS, margins: CELL_MARGINS,
    children: [new Paragraph({ children: [new TextRun({ text, size: 20, font: "Arial" })] })],
  });
}

// ===================================================================
// USER MANUAL
// ===================================================================
function buildManual() {
  const children = [];

  // --- Title Page ---
  children.push(new Paragraph({ spacing: { before: 2000 }, alignment: AlignmentType.CENTER, children: [
    new ImageRun({ type: "png", data: logoData, transformation: { width: 180, height: 180 },
      altText: { title: "DMS-Raptor Logo", description: "DMS-Raptor hexagonal raptor-eye logo", name: "logo" } }),
  ] }));
  children.push(new Paragraph({ spacing: { before: 400 }, alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "DMS-Raptor", size: 56, bold: true, font: "Arial", color: TEAL }),
  ] }));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "Dynamic Model Switching for Real-Time UAV Inspection", size: 28, font: "Arial", color: "666666" }),
  ] }));
  children.push(new Paragraph({ spacing: { before: 300 }, alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "User Manual", size: 36, bold: true, font: "Arial" }),
  ] }));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "Version 1.1.0", size: 24, font: "Arial", color: "888888" }),
  ] }));
  children.push(new Paragraph({ spacing: { before: 300 }, alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "\u00A9 2025\u20132026 Teja Chirravuri. All rights reserved.", size: 20, font: "Arial", color: "999999" }),
  ] }));
  children.push(new Paragraph({ children: [new PageBreak()] }));

  // --- Table of Contents ---
  children.push(heading1("Table of Contents"));
  children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }));
  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ====================================================================
  // 1. INTRODUCTION
  // ====================================================================
  children.push(heading1("1. Introduction"));
  children.push(para("DMS-Raptor (Dynamic Model Switching \u2014 Raptor) is a desktop application designed for complexity-aware, real-time object detection on UAV (drone) platforms. It dynamically switches between a fast, lightweight YOLOv8n model and a more accurate YOLOv8s model based on scene complexity, balancing speed and accuracy in real time."));
  children.push(heading2("1.1 Key Features"));
  children.push(bullet("Dynamic model switching between YOLOv8n (fast) and YOLOv8s (accurate)"));
  children.push(bullet("Five configurable switching policies with hysteresis support"));
  children.push(bullet("Adaptive mode with rolling percentile-based thresholds and latency budgets"));
  children.push(bullet("Real-time monitoring with live video feed, stats panel, and rolling waveform charts"));
  children.push(bullet("Data Explorer for interactive visualization with 16 trace metrics and 11 aggregate metrics"));
  children.push(bullet("Persistent run history saved across sessions"));
  children.push(bullet("Annotated video output with per-policy MP4 files"));
  children.push(bullet("TensorRT export for NVIDIA Jetson deployment (FP16/FP32/INT8)"));
  children.push(bullet("Export capabilities: CSV, PNG plots, and annotated videos"));
  children.push(bullet("Dark-themed professional GUI built with PyQt5 and Matplotlib"));

  children.push(heading2("1.2 System Requirements"));
  children.push(bullet("Python 3.10 or later"));
  children.push(bullet("PyQt5 5.15+"));
  children.push(bullet("Ultralytics YOLOv8 library"));
  children.push(bullet("OpenCV (cv2) 4.5+"));
  children.push(bullet("Matplotlib 3.5+"));
  children.push(bullet("NumPy 1.21+"));
  children.push(bullet("PyTorch 2.0+ (CPU or CUDA)"));
  children.push(bullet("Optional: NVIDIA GPU with CUDA support for accelerated inference"));
  children.push(bullet("Optional: TensorRT for Jetson deployment"));

  // ====================================================================
  // 2. INSTALLATION
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("2. Installation"));
  children.push(heading2("2.1 Setting Up the Environment"));
  children.push(para("It is recommended to use a Conda environment:"));
  children.push(code("conda create -n uav-thesis python=3.10"));
  children.push(code("conda activate uav-thesis"));
  children.push(heading2("2.2 Installing Dependencies"));
  children.push(code("pip install ultralytics PyQt5 matplotlib numpy opencv-python"));
  children.push(para("For GPU support, install the appropriate PyTorch CUDA version:"));
  children.push(code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"));
  children.push(heading2("2.3 Running the Application"));
  children.push(para("Navigate to the DMS_Raptor directory and run:"));
  children.push(code("python main.py"));
  children.push(para("The main window will appear with three tabs: Setup, Monitor, and Results."));
  children.push(heading2("2.4 Running the Executable (Windows)"));
  children.push(para("A standalone .exe is available in the dist/ folder. Double-click DMS-Raptor.exe to launch without needing Python installed. Note: the executable is platform-specific (built on the same OS)."));

  // ====================================================================
  // 3. APPLICATION OVERVIEW
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("3. Application Overview"));
  children.push(para("DMS-Raptor has a tabbed interface with three main tabs:"));
  children.push(bullet("Setup Tab \u2014 Configure input source, models, policies, parameters, and output options"));
  children.push(bullet("Monitor Tab \u2014 Real-time video feed, live statistics, and rolling waveform charts"));
  children.push(bullet("Results Tab \u2014 Run summary table, Quick Plots, Data Explorer, and export options"));
  children.push(para("The application also includes a menu bar with File (Exit), History (Clear All History Files), and Help (About) options."));

  // ====================================================================
  // 4. SETUP TAB
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("4. Setup Tab"));
  children.push(para("The Setup tab is where you configure everything before running the inference pipeline."));

  children.push(heading2("4.1 Input Source"));
  children.push(para("Select one of three input modes:"));

  // Input modes table
  children.push(new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2200, 7160],
    rows: [
      new TableRow({ children: [headerCell("Mode", 2200), headerCell("Description", 7160)] }),
      new TableRow({ children: [cell("Video File", 2200), cell("Select a local video file (.mp4, .avi, .mov, .mkv, .webm). Resolution, FPS, frame count, and duration are shown below the path.", 7160)] }),
      new TableRow({ children: [cell("Live Stream", 2200), cell("Use a camera index (0\u201320) or an RTSP URL (rtsp://...) for real-time streaming from webcam, IP camera, or drone feed.", 7160)] }),
      new TableRow({ children: [cell("Image Folder", 2200), cell("Select a folder containing image files. Each image is processed as a separate frame, useful for datasets or frame sequences.", 7160)] }),
    ],
  }));

  children.push(heading2("4.2 Detection Models"));
  children.push(para("Specify paths to two YOLO model weight files:"));
  children.push(bullet("Fast model (n): Path to YOLOv8n .pt weights (lightweight, faster inference)"));
  children.push(bullet("Accurate model (s): Path to YOLOv8s .pt weights (higher accuracy, slower inference)"));
  children.push(para("Additional model settings:"));
  children.push(bullet("Device: Select CPU, CUDA GPU (cuda:0, cuda:1), or Jetson for edge deployment"));
  children.push(bullet("Image size (imgsz): Input image resolution in pixels (320\u20131280, default 640)"));
  children.push(bullet("TensorRT Export: Enable to convert models to .engine format before running (CUDA/Jetson only)"));
  children.push(bullet("TRT Precision: FP16 (recommended), FP32, or INT8 quantization"));

  children.push(heading2("4.3 Policies"));
  children.push(para("Select which switching policies to evaluate. Multiple policies can be run in a single session \u2014 each produces an independent result:"));

  // Policies table
  children.push(new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2200, 7160],
    rows: [
      new TableRow({ children: [headerCell("Policy", 2200), headerCell("Description", 7160)] }),
      new TableRow({ children: [cell("n_only", 2200), cell("Always uses YOLOv8n (fastest baseline). No switching.", 7160)] }),
      new TableRow({ children: [cell("s_only", 2200), cell("Always uses YOLOv8s (most accurate baseline). No switching.", 7160)] }),
      new TableRow({ children: [cell("entropy_only", 2200), cell("Switches based on histogram entropy (H) alone. Simple threshold comparison.", 7160)] }),
      new TableRow({ children: [cell("combined", 2200), cell("Uses composite complexity score C = alpha * L_norm + (1 - alpha) * H_norm. Single-threshold switching.", 7160)] }),
      new TableRow({ children: [cell("combined_hyst", 2200), cell("Uses composite C-score with dual hysteresis thresholds (c_low, c_high). Most stable policy with dead-zone to prevent oscillation.", 7160)] }),
    ],
  }));

  children.push(para("The Adaptive Budget option (checkbox below policies) enables rolling percentile-based threshold adaptation for the combined_hyst policy when a latency budget is specified."));

  children.push(heading2("4.4 Output Options"));
  children.push(para("Configure video output before running:"));
  children.push(bullet("Save annotated output video: Enable to write overlay-annotated MP4 files to disk"));
  children.push(bullet("Output folder: Select directory where annotated_<policy>.mp4 files will be saved"));
  children.push(para("One video is generated per policy, containing the original frames with detection bounding boxes, model choice labels, and timing overlays."));

  children.push(heading2("4.5 Controller Parameters"));
  children.push(para("Each parameter has an Auto checkbox. When checked, the default value is used. Uncheck to customize."));

  // Parameters table
  const paramRows = [
    ["Scene Analysis", "", ""],
    ["alpha", "0.6", "Weight for Laplacian (L) in the composite score: C = alpha * L + (1-alpha) * H"],
    ["proxy_size", "160", "Downscale resolution for scene complexity proxy computation"],
    ["hist_bins", "64", "Number of histogram bins for entropy calculation"],
    ["probe_every_k", "1", "Compute proxy every k-th frame (1 = every frame)"],
    ["Thresholds", "", ""],
    ["c_low", "0.45", "Lower hysteresis threshold for combined_hyst policy"],
    ["c_high", "0.55", "Upper hysteresis threshold for combined_hyst policy"],
    ["combined_mid", "0.50", "Single threshold for combined and entropy_only policies"],
    ["Stability", "", ""],
    ["c_ema_beta", "0.25", "EMA smoothing factor for composite score (lower = smoother)"],
    ["min_dwell_frames", "10", "Minimum frames on current model before allowing a switch"],
    ["max_switches_per_100", "12", "Maximum switches allowed per 100 frames (rate limiter)"],
    ["history_window_size", "200", "Rolling window size for adaptive percentile computation"],
    ["Adaptive", "", ""],
    ["norm_lo / norm_hi", "10 / 90", "Percentile bounds for normalizing L and H values"],
    ["thr_lo / thr_hi / thr_mid", "35 / 65 / 50", "Percentile bounds for adaptive threshold calculation"],
    ["latency_penalty_factor", "0.01", "Penalty weight applied when latency exceeds budget"],
    ["budget_guard_margin", "0.10", "Margin (10%) above budget that forces switch to fast model"],
    ["budget_guard_frames", "8", "Frames to hold fast model after budget guard activation"],
    ["Processing", "", ""],
    ["conf_min", "0.001", "Minimum confidence threshold for YOLO prediction"],
    ["iou_nms", "0.45", "IoU threshold for Non-Maximum Suppression"],
    ["conf_show", "0.25", "Minimum confidence for displaying detections on overlay"],
    ["max_frames", "0", "Maximum frames to process (0 = all frames)"],
    ["stride", "1", "Process every N-th frame (1 = every frame)"],
    ["latency_smooth_window", "15", "Window size for smoothed latency averaging"],
  ];

  const tRows = [new TableRow({ children: [headerCell("Parameter", 2200), headerCell("Default", 1400), headerCell("Description", 5760)] })];
  for (const [p, d, desc] of paramRows) {
    if (desc === "" && d === "") {
      // Section header row
      tRows.push(new TableRow({ children: [
        new TableCell({ width: { size: 9360, type: WidthType.DXA }, borders: BORDERS, margins: CELL_MARGINS,
          shading: { fill: "E0F7FA", type: ShadingType.CLEAR }, columnSpan: 3,
          children: [new Paragraph({ children: [new TextRun({ text: p, size: 20, font: "Arial", bold: true, color: "00838f" })] })],
        }),
      ] }));
    } else {
      tRows.push(new TableRow({ children: [cell(p, 2200), cell(d, 1400), cell(desc, 5760)] }));
    }
  }
  children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [2200, 1400, 5760], rows: tRows }));

  children.push(heading2("4.6 Running the Pipeline"));
  children.push(numberedItem("Configure all settings in the Setup tab."));
  children.push(numberedItem("Click the Run Pipeline button."));
  children.push(numberedItem("The app automatically switches to the Monitor tab to show live progress."));
  children.push(numberedItem("Each selected policy runs sequentially. Results appear in the Results tab."));
  children.push(numberedItem("Press Stop at any time to gracefully halt the pipeline. The app stays open and partial results are preserved."));

  // ====================================================================
  // 5. MONITOR TAB
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("5. Monitor Tab"));
  children.push(para("The Monitor tab provides real-time feedback during pipeline execution."));

  children.push(heading2("5.1 Live Video Feed"));
  children.push(para("The left area displays the annotated video frames as they are processed. Detections are shown with bounding boxes, confidence scores, and class labels. A text overlay shows the current model, frame number, and timing information."));

  children.push(heading2("5.2 Stats Panel"));
  children.push(para("The right panel displays live statistics updated every frame:"));
  children.push(bullet("Frame: Current frame index"));
  children.push(bullet("FPS: Effective frames per second (computed from T_total)"));
  children.push(bullet("Model: Currently active model (YOLOv8n in green, YOLOv8s in red)"));
  children.push(bullet("Detections: Number of objects detected in the current frame"));
  children.push(bullet("T_total: Total processing time for the current frame (ms)"));
  children.push(bullet("T_scene: Scene complexity computation time (ms)"));
  children.push(bullet("Dwell: How many consecutive frames the current model has been active"));
  children.push(bullet("C-score: Current composite complexity score (0.0\u20131.0)"));
  children.push(bullet("Policy: Name of the currently running policy"));

  children.push(heading2("5.3 Rolling Waveform Charts"));
  children.push(para("Three rolling waveform charts (500-frame window) are displayed at the bottom:"));
  children.push(bullet("T_total (ms): Per-frame total latency over time (cyan line)"));
  children.push(bullet("C-score: Composite complexity score with dashed threshold lines (orange line)"));
  children.push(bullet("Model Choice: Green fill for YOLOv8n, red fill for YOLOv8s \u2014 visually shows switching patterns"));
  children.push(para("Charts update every 3 frames for smooth performance."));

  // ====================================================================
  // 6. RESULTS TAB
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("6. Results Tab"));
  children.push(para("After the pipeline completes, the Results tab presents all collected data."));

  children.push(heading2("6.1 Summary Table"));
  children.push(para("Each policy run appears as a row with 8 columns:"));
  children.push(bullet("Policy: Policy name (e.g., combined_hyst)"));
  children.push(bullet("Mode: fixed or adaptive"));
  children.push(bullet("Mean T (ms): Average total processing time per frame"));
  children.push(bullet("P95 T (ms): 95th percentile latency"));
  children.push(bullet("Slow %: Percentage of frames exceeding the latency budget"));
  children.push(bullet("Sw/100: Model switches per 100 frames"));
  children.push(bullet("Frames: Total frames processed"));
  children.push(bullet("Video: Output video file size (if saved)"));

  children.push(heading2("6.2 Quick Plots"));
  children.push(para("Four pre-built analysis plots are available in the Quick Plots sub-tab:"));
  children.push(bullet("Timing Breakdown: Stacked bar chart showing mean T_scene, T_ctrl, T_infer_n, T_infer_s per policy"));
  children.push(bullet("Latency Comparison: Grouped bar chart comparing Mean, P95, and P99 T_total per policy"));
  children.push(bullet("T_total Waveforms: Per-frame latency traces for all policies overlaid on a single chart"));
  children.push(bullet("C-Score Traces: Composite complexity score waveforms for switching policies, with threshold lines"));

  children.push(heading2("6.3 Data Explorer"));
  children.push(para("The Data Explorer is a flexible, interactive visualization tool for in-depth data analysis \u2014 similar to a simplified Tableau experience."));

  children.push(heading3("6.3.1 Plot Modes"));
  children.push(new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2600, 6760],
    rows: [
      new TableRow({ children: [headerCell("Mode", 2600), headerCell("Description", 6760)] }),
      new TableRow({ children: [cell("Trace (per-frame)", 2600), cell("Plot any trace metric against any other. X and Y axes are independently selectable. Chart types: Line, Scatter, Area.", 6760)] }),
      new TableRow({ children: [cell("Aggregate Comparison", 2600), cell("Compare aggregate statistics across policies. Choose from 11 aggregate metrics. Chart types: Bar, Grouped Bar (Mean/P95/P99).", 6760)] }),
      new TableRow({ children: [cell("Distribution", 2600), cell("Analyze the statistical distribution of any trace metric. Chart types: Histogram, Box Plot, Violin Plot.", 6760)] }),
      new TableRow({ children: [cell("Correlation Heatmap", 2600), cell("Pearson correlation matrix across all 16 trace metrics for the selected run. Visualized as a color-coded heatmap (-1 to +1).", 6760)] }),
    ],
  }));

  children.push(heading3("6.3.2 Available Trace Metrics (16)"));
  const traceMetrics = [
    ["Frame Index", "Sequential frame number"],
    ["T_total (ms)", "Total per-frame processing time"],
    ["T_scene (ms)", "Scene complexity computation time"],
    ["T_ctrl (ms)", "Controller decision time"],
    ["T_infer_n (ms)", "YOLOv8n inference time"],
    ["T_infer_s (ms)", "YOLOv8s inference time"],
    ["C-score", "Composite complexity score"],
    ["Laplacian (L)", "Laplacian variance (edge density proxy)"],
    ["Entropy (H)", "Histogram entropy (texture complexity proxy)"],
    ["c_low threshold", "Lower hysteresis threshold (adaptive)"],
    ["c_high threshold", "Upper hysteresis threshold (adaptive)"],
    ["Dwell", "Consecutive frames on current model"],
    ["Detections", "Number of detected objects"],
    ["Penalty", "Latency penalty value"],
    ["Avg T_total (ms)", "Smoothed average of T_total"],
    ["Choice (0=n, 1=s)", "Model selection (numeric)"],
  ];
  const tmRows = [new TableRow({ children: [headerCell("Metric", 3500), headerCell("Description", 5860)] })];
  for (const [m, d] of traceMetrics) {
    tmRows.push(new TableRow({ children: [cell(m, 3500), cell(d, 5860)] }));
  }
  children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3500, 5860], rows: tmRows }));

  children.push(heading3("6.3.3 Available Aggregate Metrics (11)"));
  const aggMetrics = [
    ["Mean T_total (ms)", "Average processing time across all frames"],
    ["P95 T_total (ms)", "95th percentile latency"],
    ["P99 T_total (ms)", "99th percentile latency"],
    ["Slow %", "Percentage of frames exceeding latency budget"],
    ["Switches / 100", "Model switches per 100 frames"],
    ["Total Switches", "Absolute number of model switches"],
    ["Mean T_scene (ms)", "Average scene complexity computation time"],
    ["Mean T_ctrl (ms)", "Average controller decision time"],
    ["Mean T_infer_n (ms)", "Average YOLOv8n inference time"],
    ["Mean T_infer_s (ms)", "Average YOLOv8s inference time"],
    ["Total Frames", "Total number of processed frames"],
  ];
  const amRows = [new TableRow({ children: [headerCell("Metric", 3500), headerCell("Description", 5860)] })];
  for (const [m, d] of aggMetrics) {
    amRows.push(new TableRow({ children: [cell(m, 3500), cell(d, 5860)] }));
  }
  children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3500, 5860], rows: amRows }));

  children.push(heading3("6.3.4 Using the Data Explorer"));
  children.push(numberedItem("Select a Plot Mode from the dropdown."));
  children.push(numberedItem("Choose X-Axis and Y-Axis metrics (for Trace mode) or an Aggregate Metric (for Aggregate mode)."));
  children.push(numberedItem("Select a Chart Type (e.g., Line, Scatter, Box Plot)."));
  children.push(numberedItem("Select one or more runs from the run list (all runs are selected by default)."));
  children.push(numberedItem("Click Generate Plot to render the visualization."));
  children.push(numberedItem("Click Export This Plot to save the current chart as PNG, SVG, or PDF."));

  children.push(heading2("6.4 Summary Panel"));
  children.push(para("The Summary sub-tab displays a plain-text summary of all runs, including detailed timing breakdowns and output video information. The best switching policy (lowest mean T_total) is highlighted at the bottom."));

  children.push(heading2("6.5 Export Options"));
  children.push(bullet("Export CSV: Save all run results as a CSV file with 16 columns covering all aggregate metrics"));
  children.push(bullet("Export Plots: Save all 4 Quick Plots as high-resolution PNG files (220 DPI) to a folder"));
  children.push(bullet("Export This Plot: Save the current Data Explorer chart as PNG/SVG/PDF"));
  children.push(bullet("Open Video Folder: Open the file explorer to the folder containing saved annotated videos"));
  children.push(bullet("Clear All: Clear in-session results (historical data on disk is preserved)"));

  // ====================================================================
  // 7. PERSISTENT HISTORY
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("7. Persistent Run History"));
  children.push(para("DMS-Raptor automatically saves every completed policy run to disk as a JSON file. This means your results are preserved even when you close and reopen the application."));

  children.push(heading2("7.1 How It Works"));
  children.push(bullet("Each policy run is saved as a JSON file in the history/ directory inside the DMS_Raptor folder."));
  children.push(bullet("File naming convention: {timestamp}_{policy}.json"));
  children.push(bullet("On app startup, all historical runs are automatically loaded into the Results tab."));
  children.push(bullet("Full per-frame trace data is preserved, so the Data Explorer works with historical runs."));

  children.push(heading2("7.2 Managing History"));
  children.push(bullet("View history: All past runs appear in the Results tab summary table on startup."));
  children.push(bullet("Clear session: Use the Clear All button in Results to remove runs from the current view (files remain on disk)."));
  children.push(bullet("Delete all history: Use the menu History > Clear All History Files to permanently delete all saved JSON files."));

  // ====================================================================
  // 8. TENSORRT OPTIMIZATION
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("8. TensorRT Optimization"));
  children.push(para("For maximum throughput on NVIDIA GPUs and Jetson edge devices, DMS-Raptor supports TensorRT model export."));

  children.push(heading2("8.1 How to Enable"));
  children.push(numberedItem("In the Setup tab, select a CUDA or Jetson device."));
  children.push(numberedItem("Check the \"Export to TensorRT (.engine) before run\" checkbox."));
  children.push(numberedItem("Select precision: FP16 (recommended for Jetson), FP32, or INT8."));
  children.push(numberedItem("Click Run Pipeline. The first run will export models to .engine files (this may take several minutes)."));
  children.push(para("Subsequent runs automatically reuse existing .engine files for instant startup."));

  children.push(heading2("8.2 Jetson Deployment"));
  children.push(para("When a Jetson device is detected (Tegra, Orin, Xavier), it appears as \"jetson:0\" in the device dropdown. The device label is internally mapped to \"cuda:0\", which is the standard CUDA device on Jetson boards. FP16 precision is recommended for optimal speed-accuracy tradeoff on Jetson."));

  // ====================================================================
  // 9. SCENE COMPLEXITY
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("9. Scene Complexity Analysis"));
  children.push(para("DMS-Raptor evaluates scene complexity using two lightweight proxies computed on a downscaled frame:"));

  children.push(heading2("9.1 Laplacian Variance (L)"));
  children.push(para("Measures edge density in the frame. High L indicates many edges/textures (complex scene). Computed as the variance of the Laplacian of the grayscale image."));

  children.push(heading2("9.2 Histogram Entropy (H)"));
  children.push(para("Measures texture complexity through pixel intensity distribution. High entropy indicates a diverse distribution (complex scene). Computed on the grayscale histogram."));

  children.push(heading2("9.3 Composite Score (C)"));
  children.push(para("Both proxies are normalized to [0, 1] using rolling percentile bounds and combined:"));
  children.push(code("C = alpha * L_norm + (1 - alpha) * H_norm"));
  children.push(para("The alpha parameter (default 0.6) controls the weighting. Higher alpha gives more weight to edge density."));

  // ====================================================================
  // 10. SWITCHING POLICIES
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("10. Switching Policy Details"));

  children.push(heading2("10.1 Actuator Stabilizers"));
  children.push(para("All switching policies use stabilizers to prevent erratic oscillation:"));
  children.push(bullet("EMA Smoothing: The C-score is smoothed using exponential moving average (c_ema_beta)"));
  children.push(bullet("Minimum Dwell: After switching, the current model must be used for at least min_dwell_frames before another switch is allowed"));
  children.push(bullet("Rate Limiter: No more than max_switches_per_100 switches in any 100-frame window"));
  children.push(bullet("Budget Guard: If latency exceeds budget + margin for budget_guard_frames, force switch to the fast model"));

  children.push(heading2("10.2 Adaptive Mode"));
  children.push(para("When Enable Adaptive Budget is checked and the combined_hyst policy is selected, the system uses rolling percentile-based thresholds that automatically adapt to the video content. A latency penalty is applied when the rolling average T_total exceeds the latency budget, biasing the system toward the faster model."));

  // ====================================================================
  // 11. TROUBLESHOOTING
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("11. Troubleshooting"));

  const troubleRows = [
    new TableRow({ children: [headerCell("Issue", 3800), headerCell("Solution", 5560)] }),
    new TableRow({ children: [cell("App closes when pressing Stop", 3800), cell("Upgrade to v1.1.0. This bug was fixed with a dedicated stopped_early signal and safe worker cleanup.", 5560)] }),
    new TableRow({ children: [cell("\"No module named ultralytics\"", 3800), cell("Run: pip install ultralytics", 5560)] }),
    new TableRow({ children: [cell("CUDA out of memory", 3800), cell("Reduce imgsz (e.g., to 416 or 320), or use CPU device.", 5560)] }),
    new TableRow({ children: [cell("TensorRT export fails", 3800), cell("Ensure TensorRT is installed and compatible with your CUDA version. The app falls back to .pt automatically.", 5560)] }),
    new TableRow({ children: [cell("Video file not recognized", 3800), cell("Ensure OpenCV can read the format. Try converting to .mp4 (H.264).", 5560)] }),
    new TableRow({ children: [cell("Charts appear empty", 3800), cell("Ensure at least one policy has completed. Check that switching policies are selected for C-score traces.", 5560)] }),
    new TableRow({ children: [cell("History not loading", 3800), cell("Check that the history/ directory exists in the DMS_Raptor folder and contains valid .json files.", 5560)] }),
    new TableRow({ children: [cell("Annotated video not saving", 3800), cell("Enable \"Save annotated output video\" in Setup and select a valid output folder before running.", 5560)] }),
  ];
  children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3800, 5560], rows: troubleRows }));

  // ====================================================================
  // 12. ARCHITECTURE
  // ====================================================================
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1("12. Architecture Overview"));

  children.push(heading2("12.1 Module Structure"));
  children.push(code("DMS_Raptor/"));
  children.push(code("  main.py                   # Entry point"));
  children.push(code("  core/"));
  children.push(code("    config.py               # Dataclasses, constants, metrics"));
  children.push(code("    engine.py               # StreamingEngine (generator-based)"));
  children.push(code("    history.py              # Persistent JSON storage"));
  children.push(code("    __init__.py             # Public API exports"));
  children.push(code("  gui/"));
  children.push(code("    main_window.py          # MainWindow (tab wiring, signals)"));
  children.push(code("    setup_tab.py            # SetupTab (parameter UI)"));
  children.push(code("    monitor_tab.py          # MonitorTab (live video + charts)"));
  children.push(code("    results_tab.py          # ResultsTab (Data Explorer + plots)"));
  children.push(code("    worker.py               # InferenceWorker (QThread)"));
  children.push(code("  resources/"));
  children.push(code("    logo.svg, logo.png      # Application logo"));
  children.push(code("    style.qss               # Dark theme stylesheet"));
  children.push(code("  history/                  # Auto-created, stores run JSON files"));

  children.push(heading2("12.2 Data Flow"));
  children.push(numberedItem("User configures parameters in SetupTab and clicks Run Pipeline."));
  children.push(numberedItem("SetupTab emits run_requested signal with all parameters as a dict."));
  children.push(numberedItem("MainWindow creates an InferenceWorker (QThread) with the parameters."));
  children.push(numberedItem("Worker loads YOLO models, optionally exports to TensorRT, and warms up."));
  children.push(numberedItem("For each policy, worker creates a StreamingEngine and iterates its generator."));
  children.push(numberedItem("Each frame yields a FrameResult, emitted via frame_ready signal to MonitorTab."));
  children.push(numberedItem("On policy completion, a RunSummary is emitted via policy_finished signal."));
  children.push(numberedItem("MainWindow saves the RunSummary to history/ and adds it to ResultsTab."));
  children.push(numberedItem("When all policies complete, all_finished signal triggers switch to Results tab."));

  children.push(heading2("12.3 Key Design Patterns"));
  children.push(bullet("Generator-based streaming: StreamingEngine.run() yields FrameResult per frame, enabling clean iteration"));
  children.push(bullet("Signal-slot architecture: PyQt5 signals decouple the worker thread from the GUI thread"));
  children.push(bullet("Graceful stop: Dedicated stopped_early signal with safe thread cleanup prevents crashes"));
  children.push(bullet("Persistent history: JSON serialization of RunSummary with to_dict() / from_dict()"));

  // FINAL
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(new Paragraph({ spacing: { before: 2000 }, alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "End of User Manual", size: 28, font: "Arial", color: "888888", italics: true }),
  ] }));
  children.push(new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "DMS-Raptor v1.1.0 \u2014 \u00A9 2025\u20132026 Teja Chirravuri", size: 22, font: "Arial", color: "AAAAAA" }),
  ] }));

  return children;
}

// ===================================================================
// LOGO SIGNIFICANCE DOCUMENT
// ===================================================================
function buildLogoDoc() {
  const children = [];

  children.push(new Paragraph({ spacing: { before: 800 }, alignment: AlignmentType.CENTER, children: [
    new ImageRun({ type: "png", data: logoData, transformation: { width: 250, height: 250 },
      altText: { title: "DMS-Raptor Logo", description: "DMS-Raptor hexagonal raptor-eye targeting reticle", name: "logo" } }),
  ] }));

  children.push(new Paragraph({ spacing: { before: 400 }, alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "DMS-Raptor Logo", size: 48, bold: true, font: "Arial", color: TEAL }),
  ] }));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "Design Significance & Symbolism", size: 28, font: "Arial", color: "666666" }),
  ] }));

  children.push(new Paragraph({ spacing: { before: 600 }, children: [] }));

  children.push(heading1("Overview"));
  children.push(para("The DMS-Raptor logo is a hexagonal raptor-eye targeting reticle that encapsulates the core philosophy of the application: precision, intelligence, and speed in UAV-based object detection."));

  children.push(heading1("Design Elements"));

  children.push(heading2("1. Hexagonal Frame"));
  children.push(para("The outer hexagonal shape represents the structured, systematic nature of the dynamic model switching framework. Hexagons are a recurring motif in aerospace and defense HUDs (Heads-Up Displays), connecting the logo to its UAV context. The teal-colored (#00bcd4) hexagonal border also evokes drone rotors and technological precision."));

  children.push(heading2("2. Raptor Eye (Central Circle)"));
  children.push(para("The circular element at the center represents the eye of a raptor (bird of prey). Raptors like eagles and hawks are known for their extraordinary visual acuity \u2014 they can spot prey from great heights with incredible precision. This mirrors how DMS-Raptor analyzes every frame to detect objects with optimal accuracy. The dark inner circle (#1a1a2e) surrounded by teal symbolizes focused vision peering through the darkness."));

  children.push(heading2("3. Targeting Reticle / Crosshair"));
  children.push(para("The four crosshair lines extending from the center represent a targeting reticle, as seen in military and surveillance optics. This symbolizes the precision-targeting nature of object detection \u2014 finding and localizing objects in each frame. The crosshair also implies the system is always actively scanning and making decisions."));

  children.push(heading2("4. Corner Brackets"));
  children.push(para("The L-shaped corner brackets at each corner of the hexagon represent a bounding box \u2014 the fundamental output of object detection models. Every detection drawn on screen is a bounding box, and these corner elements are a direct visual metaphor for that. They also suggest a viewfinder or camera frame, reinforcing the real-time video processing context."));

  children.push(heading2("5. Red Accent Dot"));
  children.push(para("The small red circle (#e94560) at the center is a deliberate contrast element. It represents the critical decision point where the system must choose between the fast model (YOLOv8n) and the accurate model (YOLOv8s). The red color conveys urgency and importance \u2014 this is the moment of dynamic switching. It also serves as a focal point that draws the eye inward, just as the algorithm focuses on the most critical frames."));

  children.push(heading1("Color Palette"));

  children.push(new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2000, 2000, 5360],
    rows: [
      new TableRow({ children: [headerCell("Color", 2000), headerCell("Hex Code", 2000), headerCell("Symbolism", 5360)] }),
      new TableRow({ children: [cell("Teal", 2000), cell("#00bcd4", 2000), cell("Technology, clarity, intelligence, UAV operations", 5360)] }),
      new TableRow({ children: [cell("Red Accent", 2000), cell("#e94560", 2000), cell("Critical decision points, urgency, active detection", 5360)] }),
      new TableRow({ children: [cell("Dark Navy", 2000), cell("#1a1a2e", 2000), cell("Night operations, stealth, the application\u2019s dark theme", 5360)] }),
    ],
  }));

  children.push(heading1("Name Significance"));
  children.push(paraMulti([
    { text: "DMS", bold: true },
    " stands for Dynamic Model Switching \u2014 the core algorithm that switches between detection models in real time. ",
    { text: "Raptor", bold: true },
    " refers to birds of prey known for their exceptional vision and hunting precision. Together, DMS-Raptor means a system that sees like a raptor and switches like a reflex.",
  ]));

  children.push(new Paragraph({ spacing: { before: 600 }, children: [] }));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "\u00A9 2025\u20132026 Teja Chirravuri. All rights reserved.", size: 20, font: "Arial", color: "999999" }),
  ] }));

  return children;
}

// ===================================================================
// GENERATE BOTH DOCUMENTS
// ===================================================================
async function main() {
  const numbering = {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        ],
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      },
    ],
  };

  const styles = {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "00838f" },
        paragraph: { spacing: { before: 300, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  };

  const pageProps = {
    page: {
      size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
    },
  };

  // --- User Manual ---
  console.log("Generating User Manual...");
  const manualDoc = new Document({
    numbering,
    styles,
    sections: [{
      properties: {
        ...pageProps,
      },
      headers: {
        default: new Header({ children: [
          new Paragraph({ alignment: AlignmentType.RIGHT, children: [
            new TextRun({ text: "DMS-Raptor User Manual v1.1.0", size: 18, font: "Arial", color: "999999" }),
          ] }),
        ] }),
      },
      footers: {
        default: new Footer({ children: [
          new Paragraph({ alignment: AlignmentType.CENTER, children: [
            new TextRun({ text: "Page ", size: 18, font: "Arial", color: "999999" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: "999999" }),
          ] }),
        ] }),
      },
      children: buildManual(),
    }],
  });
  const manualBuf = await Packer.toBuffer(manualDoc);
  fs.writeFileSync(OUT_MANUAL, manualBuf);
  console.log("User Manual saved: " + OUT_MANUAL);

  // --- Logo Document ---
  console.log("Generating Logo Significance document...");
  const logoDoc = new Document({
    numbering,
    styles,
    sections: [{
      properties: { ...pageProps },
      footers: {
        default: new Footer({ children: [
          new Paragraph({ alignment: AlignmentType.CENTER, children: [
            new TextRun({ text: "Page ", size: 18, font: "Arial", color: "999999" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: "999999" }),
          ] }),
        ] }),
      },
      children: buildLogoDoc(),
    }],
  });
  const logoBuf = await Packer.toBuffer(logoDoc);
  fs.writeFileSync(OUT_LOGO, logoBuf);
  console.log("Logo document saved: " + OUT_LOGO);

  console.log("DONE");
}

main().catch(err => { console.error(err); process.exit(1); });
