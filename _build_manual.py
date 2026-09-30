"""Build DMS-Raptor User Manual as .docx — with cross-platform distribution guide."""
import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DMS_Raptor_User_Manual.docx")

doc = Document()

# ── Styles ──────────────────────────────────────────────────────────────
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)

for level, size, color in [
    ("Heading 1", 22, RGBColor(0, 0x6D, 0x77)),
    ("Heading 2", 16, RGBColor(0, 0x6D, 0x77)),
    ("Heading 3", 13, RGBColor(0x33, 0x33, 0x33)),
]:
    s = doc.styles[level]
    s.font.name = "Calibri"
    s.font.size = Pt(size)
    s.font.color.rgb = color
    s.font.bold = True

# ── Helpers ─────────────────────────────────────────────────────────────
def h1(text): return doc.add_heading(text, level=1)
def h2(text): return doc.add_heading(text, level=2)
def h3(text): return doc.add_heading(text, level=3)

def para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    return p

def bullet(text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    if level > 0:
        p.paragraph_format.left_indent = Cm(1.27 * (level + 1))
    return p

def code_block(lines):
    """Add a code block (monospace, small font)."""
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.left_indent = Cm(1.0)
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0x20, 0x20, 0x20)

def code_line(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(10)
    return p

def table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(10)
    for row_data in rows:
        row = t.add_row()
        for i, val in enumerate(row_data):
            row.cells[i].text = str(val)
            for paragraph in row.cells[i].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                if i < len(row.cells):
                    row.cells[i].width = Inches(w)
    return t

def note(text):
    p = doc.add_paragraph()
    run = p.add_run("Note: ")
    run.bold = True
    run.font.color.rgb = RGBColor(0, 0x6D, 0x77)
    p.add_run(text)
    return p

def page_break():
    doc.add_page_break()


# ══════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ══════════════════════════════════════════════════════════════════════════
for _ in range(6):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("DMS-Raptor")
run.font.size = Pt(42)
run.font.color.rgb = RGBColor(0, 0xBC, 0xD4)
run.bold = True

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run("Dynamic Model Switching for Real-Time UAV Inspection")
run.font.size = Pt(18)
run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

doc.add_paragraph()

ver = doc.add_paragraph()
ver.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = ver.add_run("User Manual  \u2014  Version 1.0.0")
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

doc.add_paragraph()

author = doc.add_paragraph()
author.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = author.add_run("\u00a9 2025\u20132026 Teja Chirravuri. All rights reserved.")
run.font.size = Pt(11)
run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

page_break()

# ══════════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ══════════════════════════════════════════════════════════════════════════
h1("Table of Contents")
toc_items = [
    "1. Introduction",
    "2. System Requirements & Installation",
    "   2.1 Windows",
    "   2.2 Linux (Ubuntu/Debian)",
    "   2.3 macOS",
    "   2.4 NVIDIA Jetson",
    "3. Building Executables (Distribution)",
    "   3.1 Windows (.exe)",
    "   3.2 Linux (AppImage / Binary)",
    "   3.3 macOS (.app Bundle)",
    "   3.4 Android / iOS (Not Supported)",
    "4. Quick Start Guide",
    "5. The Setup Tab",
    "   5.1 Input Source",
    "   5.2 Detection Models",
    "   5.3 Policies",
    "   5.4 Controller Parameters",
    "6. The Monitor Tab",
    "7. The Results Tab",
    "8. Complete Parameter Reference",
    "9. Switching Policies Explained",
    "10. Adaptive Budget Mode",
    "11. TensorRT Optimization Guide",
    "12. Output & Export",
    "13. Troubleshooting",
    "   13.1 Windows Issues",
    "   13.2 Linux Issues",
    "   13.3 macOS Issues",
    "   13.4 Jetson Issues",
    "   13.5 General Issues",
    "14. Architecture Overview",
    "15. Glossary",
]
for item in toc_items:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(2)

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════
h1("1. Introduction")

h2("1.1 What is DMS-Raptor?")
para(
    "DMS-Raptor (Dynamic Model Switching \u2014 Real-time Adaptive Policy for "
    "Throughput-Optimized Recognition) is a desktop application designed for "
    "benchmarking and deploying dynamic model-switching strategies on UAV "
    "(Unmanned Aerial Vehicle) video feeds. It pairs a lightweight, fast object "
    "detection model (YOLOv8n) with a more accurate but slower model (YOLOv8s) "
    "and intelligently switches between them based on real-time scene complexity."
)

h2("1.2 Why Dynamic Model Switching?")
para(
    "UAV inspection platforms face a fundamental trade-off: lightweight models "
    "deliver high frame rates but miss detections in complex scenes, while "
    "accurate models provide better detection quality but at the cost of latency. "
    "Running the heavy model on every frame wastes compute on simple scenes. "
    "Dynamic Model Switching solves this by:"
)
bullet("Using the fast model (YOLOv8n) on simple, low-complexity frames")
bullet("Switching to the accurate model (YOLOv8s) only when scene complexity exceeds a threshold")
bullet("Maintaining real-time frame rates while preserving detection accuracy where it matters")
bullet("Adapting thresholds dynamically based on rolling scene statistics and latency budgets")

h2("1.3 Supported Platforms")
table(
    ["Platform", "Support Level", "Distribution Format"],
    [
        ["Windows 10/11", "Full (primary)", ".exe via PyInstaller"],
        ["Ubuntu/Debian Linux", "Full", "Binary via PyInstaller"],
        ["macOS (Intel & Apple Silicon)", "Full", ".app bundle via PyInstaller"],
        ["NVIDIA Jetson (JetPack)", "Full (GPU optimized)", "Run from source + TensorRT"],
        ["Android / iOS", "Not supported", "PyQt5 is a desktop framework"],
    ],
    col_widths=[2.2, 1.5, 2.6],
)

h2("1.4 Key Features")
bullet("Three input modes: Video File, Live Stream (RTSP/webcam), Image Folder")
bullet("Five switching policies: n_only, s_only, entropy_only, combined, combined_hyst")
bullet("Adaptive budget mode with rolling percentile-based thresholds")
bullet("Live video feed with detection overlays and bounding boxes")
bullet("Real-time waveform charts: T_total latency, C-score, and model choice timeline")
bullet("Post-run comparison dashboard with timing breakdowns and latency percentiles")
bullet("Device support: CPU, CUDA GPUs, NVIDIA Jetson platforms")
bullet("TensorRT optimization with FP16, FP32, and INT8 precision modes")
bullet("CSV and PNG export for thesis-quality figures and data")
bullet("Dark teal professional UI theme")
bullet("All parameters have Auto/Manual toggle with sensible defaults")

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 2. SYSTEM REQUIREMENTS & INSTALLATION
# ══════════════════════════════════════════════════════════════════════════
h1("2. System Requirements & Installation")

h2("2.1 Windows")
h3("Requirements")
table(
    ["Component", "Minimum", "Recommended"],
    [
        ["OS", "Windows 10 (64-bit)", "Windows 11"],
        ["Python", "3.9", "3.10 (Anaconda)"],
        ["RAM", "8 GB", "16 GB"],
        ["GPU (optional)", "NVIDIA GTX 1060", "NVIDIA RTX 3060+"],
        ["CUDA (optional)", "11.7", "12.x"],
        ["Disk Space", "2 GB", "5 GB (with TensorRT)"],
    ],
    col_widths=[1.8, 2.0, 2.5],
)

h3("Installation")
para("Option A: Run from source (recommended for development)", bold=True)
code_block([
    "# 1. Install Anaconda or Miniconda",
    "# 2. Create and activate environment",
    "conda create -n uav-thesis python=3.10 -y",
    "conda activate uav-thesis",
    "",
    "# 3. Install PyTorch with CUDA (if using GPU)",
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
    "",
    "# 4. Navigate to DMS-Raptor directory",
    "cd G:\\Teja_Master_Thesis\\DMS_Raptor",
    "",
    "# 5. Install dependencies",
    "pip install -r requirements.txt",
    "",
    "# 6. Launch",
    "python main.py",
])

para("Option B: Run pre-built .exe", bold=True)
code_block([
    "# 1. Download DMS-Raptor-Windows.zip from distribution",
    "# 2. Extract to any folder",
    "# 3. Double-click DMS-Raptor.exe",
    "# No Python installation required!",
])
note("The .exe bundles Python and all dependencies. YOLO model weights (.pt files) are NOT included \u2014 you must provide your own.")

h2("2.2 Linux (Ubuntu/Debian)")
h3("Requirements")
table(
    ["Component", "Minimum", "Recommended"],
    [
        ["OS", "Ubuntu 20.04 / Debian 11", "Ubuntu 22.04"],
        ["Python", "3.9", "3.10"],
        ["RAM", "8 GB", "16 GB"],
        ["GPU (optional)", "NVIDIA GTX 1060 + Driver 525+", "RTX 3060+"],
        ["System packages", "libxcb, libgl1", "See below"],
    ],
    col_widths=[1.8, 2.2, 2.3],
)

h3("Installation")
code_block([
    "# 1. Install system dependencies for Qt5 and OpenCV",
    "sudo apt update",
    "sudo apt install -y python3-pip python3-venv \\",
    "    libxcb-xinerama0 libxcb-cursor0 libgl1-mesa-glx \\",
    "    libglib2.0-0 libsm6 libxrender1 libxext6 \\",
    "    libxkbcommon-x11-0 libegl1",
    "",
    "# 2. Create virtual environment",
    "python3 -m venv ~/dms-raptor-env",
    "source ~/dms-raptor-env/bin/activate",
    "",
    "# 3. Install PyTorch (CPU or CUDA)",
    "pip install torch torchvision  # CPU",
    "# OR for CUDA:",
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
    "",
    "# 4. Install DMS-Raptor dependencies",
    "cd /path/to/DMS_Raptor",
    "pip install -r requirements.txt",
    "",
    "# 5. Launch",
    "python main.py",
])

note("If you get 'xcb plugin not loaded' errors, install the missing libxcb packages listed above.")

h2("2.3 macOS")
h3("Requirements")
table(
    ["Component", "Minimum", "Recommended"],
    [
        ["OS", "macOS 11 (Big Sur)", "macOS 13+ (Ventura)"],
        ["Architecture", "Intel x86_64", "Apple Silicon (M1/M2/M3)"],
        ["Python", "3.9", "3.10 (via Homebrew)"],
        ["RAM", "8 GB", "16 GB"],
    ],
    col_widths=[1.8, 2.2, 2.3],
)

h3("Installation")
code_block([
    "# 1. Install Homebrew (if not installed)",
    '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
    "",
    "# 2. Install Python",
    "brew install python@3.10",
    "",
    "# 3. Create virtual environment",
    "python3.10 -m venv ~/dms-raptor-env",
    "source ~/dms-raptor-env/bin/activate",
    "",
    "# 4. Install PyTorch",
    "pip install torch torchvision  # MPS acceleration on Apple Silicon",
    "",
    "# 5. Install DMS-Raptor dependencies",
    "cd /path/to/DMS_Raptor",
    "pip install -r requirements.txt",
    "",
    "# 6. Launch",
    "python main.py",
])

note("macOS does not have NVIDIA CUDA support. Use 'cpu' device or 'mps' (Apple Silicon GPU) if supported by your PyTorch version. TensorRT is not available on macOS.")

h2("2.4 NVIDIA Jetson")
h3("Requirements")
table(
    ["Component", "Minimum"],
    [
        ["Board", "Jetson Nano / Xavier NX / Orin Nano"],
        ["JetPack", "5.0+"],
        ["Python", "3.8+ (system Python)"],
        ["RAM", "4 GB (8 GB recommended)"],
    ],
    col_widths=[2.0, 4.3],
)

h3("Installation")
code_block([
    "# 1. Jetson comes with PyTorch and TensorRT pre-installed in JetPack",
    "# 2. Install additional dependencies",
    "pip3 install PyQt5 opencv-python matplotlib ultralytics pandas",
    "",
    "# 3. If PyQt5 fails, install from apt",
    "sudo apt install python3-pyqt5",
    "",
    "# 4. Launch",
    "cd /path/to/DMS_Raptor",
    "python3 main.py",
    "",
    "# 5. Enable max performance",
    "sudo nvpmodel -m 0",
    "sudo jetson_clocks",
])

note("On Jetson, select 'jetson:0' as device and enable TensorRT FP16 for maximum performance. The first run exports .engine files (takes 5\u201310 minutes), but subsequent runs reuse them.")

h2("2.5 YOLO Model Weights")
para("You need two YOLO model weight files (.pt format):")
bullet("Fast model (n): e.g., yolov8n.pt \u2014 lightweight, high FPS")
bullet("Accurate model (s): e.g., yolov8s.pt \u2014 better detection, slower")
para(
    "Download pre-trained weights from the Ultralytics repository or use your "
    "own custom-trained models. Place them anywhere accessible and browse to them "
    "in the Setup tab."
)

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 3. BUILDING EXECUTABLES (DISTRIBUTION)
# ══════════════════════════════════════════════════════════════════════════
h1("3. Building Executables (Distribution)")
para(
    "DMS-Raptor can be packaged as standalone executables using PyInstaller. "
    "Each platform requires building ON that platform \u2014 cross-compilation is "
    "not supported by PyInstaller."
)

h2("3.1 Windows (.exe)")
h3("Prerequisites")
code_block([
    "conda activate uav-thesis",
    "pip install pyinstaller",
])

h3("Build Steps")
para("Option A: Using the build script (recommended)", bold=True)
code_block([
    "cd G:\\Teja_Master_Thesis\\DMS_Raptor",
    "",
    "# Directory mode (recommended \u2014 faster startup)",
    "python build_exe.py",
    "",
    "# OR single-file mode (one large .exe)",
    "python build_exe.py --onefile",
    "",
    "# Clean previous builds first",
    "python build_exe.py --clean",
])

para("Option B: Using PyInstaller directly", bold=True)
code_block([
    "cd G:\\Teja_Master_Thesis\\DMS_Raptor",
    "pyinstaller --noconfirm --clean dms_raptor.spec",
])

h3("Output")
bullet("Directory mode: dist/DMS-Raptor/ folder containing DMS-Raptor.exe and all dependencies")
bullet("Single-file mode: dist/DMS-Raptor.exe (one file, ~500 MB+, slower startup)")

h3("Distribution")
bullet("Zip the dist/DMS-Raptor/ folder and share")
bullet("Users extract and double-click DMS-Raptor.exe")
bullet("No Python installation required on the target machine")
bullet("YOLO .pt model weights are NOT bundled \u2014 users provide their own")

h2("3.2 Linux (Binary)")
h3("Prerequisites")
code_block([
    "source ~/dms-raptor-env/bin/activate",
    "pip install pyinstaller",
])

h3("Build Steps")
code_block([
    "cd /path/to/DMS_Raptor",
    "",
    "# Directory mode (recommended)",
    "python build_exe.py",
    "",
    "# OR single-file binary",
    "python build_exe.py --onefile",
])

h3("Output and Distribution")
bullet("Directory mode: dist/DMS-Raptor/ folder containing the DMS-Raptor binary")
bullet("Single-file: dist/DMS-Raptor (executable binary)")
bullet("Make executable: chmod +x dist/DMS-Raptor/DMS-Raptor")
bullet("Package as .tar.gz: tar -czf DMS-Raptor-Linux.tar.gz -C dist DMS-Raptor/")
bullet("Users extract and run: ./DMS-Raptor/DMS-Raptor")

note("Linux binaries are generally portable across distributions with the same glibc version. Build on the oldest supported distro (e.g., Ubuntu 20.04) for maximum compatibility.")

h2("3.3 macOS (.app Bundle)")
h3("Prerequisites")
code_block([
    "source ~/dms-raptor-env/bin/activate",
    "pip install pyinstaller",
])

h3("Build Steps")
code_block([
    "cd /path/to/DMS_Raptor",
    "",
    "# Build .app bundle",
    "python build_exe.py",
    "",
    "# OR single-file binary",
    "python build_exe.py --onefile",
])

h3("Output and Distribution")
bullet("Directory mode: dist/DMS-Raptor/ folder (can be converted to .app bundle)")
bullet("Single-file: dist/DMS-Raptor binary")
bullet("Package as .dmg for distribution using hdiutil or create-dmg tool")
bullet("Users may need to allow the app in System Preferences > Security & Privacy")

note("For Apple Silicon Macs, build natively on an M1/M2/M3 machine. Intel builds will run via Rosetta 2 but with reduced performance. macOS Gatekeeper may block unsigned apps \u2014 users can right-click > Open to bypass.")

h2("3.4 Android / iOS (Not Supported)")
para(
    "DMS-Raptor is built with PyQt5, a desktop GUI framework that does not "
    "run on mobile platforms. Porting to Android or iOS would require a "
    "complete rewrite using a mobile framework such as:"
)
bullet("Kivy + Buildozer (Python, cross-platform mobile)")
bullet("Flutter or React Native (Dart/JavaScript)")
bullet("Native Android (Kotlin) / iOS (Swift)")
para(
    "For mobile UAV inspection, consider running DMS-Raptor on a companion "
    "laptop or Jetson device connected to the UAV, rather than on a phone."
)

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 4. QUICK START GUIDE
# ══════════════════════════════════════════════════════════════════════════
h1("4. Quick Start Guide")
para("Follow these steps to run your first benchmark:", bold=True)
doc.add_paragraph()

steps = [
    ("Launch the application", "Run 'python main.py' from the DMS_Raptor directory (or double-click the .exe)."),
    ("Select input source", "In the Setup tab, choose 'Video File' and browse to your UAV video (.mp4, .avi, .mov, .mkv, .webm)."),
    ("Load models", "Browse to your YOLOv8n.pt (fast) and YOLOv8s.pt (accurate) weight files."),
    ("Choose device", "Select CPU, cuda:0, or jetson:0 from the device dropdown."),
    ("Select policies", "Check the policies you want to benchmark (all five are selected by default)."),
    ("Leave parameters on Auto", "All controller parameters default to 'Auto' with sensible values. Customize later."),
    ("Click 'Run Pipeline'", "The app switches to the Monitor tab and begins processing."),
    ("Watch live results", "Observe the live video feed, detection overlays, and waveform charts in real time."),
    ("Review results", "After completion, the Results tab shows comparison tables, charts, and waveform overlays."),
    ("Export data", "Use 'Export CSV' for raw data or 'Export Plots' for publication-quality PNG figures."),
]
for i, (title_text, desc) in enumerate(steps, 1):
    p = doc.add_paragraph()
    run = p.add_run(f"Step {i}: {title_text}")
    run.bold = True
    para(desc)

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 5. THE SETUP TAB
# ══════════════════════════════════════════════════════════════════════════
h1("5. The Setup Tab")
para(
    "The Setup tab is the primary configuration interface. It is organized "
    "into five sections: Input Source, Detection Models, Policies, Controller "
    "Parameters, and Action Buttons."
)

h2("5.1 Input Source")
para("DMS-Raptor supports three input modes, selectable from the Mode dropdown:")

h3("Video File")
bullet("Select a pre-recorded video file (.mp4, .avi, .mov, .mkv, .webm)")
bullet("Click 'Browse' to navigate to the file")
bullet("Video metadata (resolution, FPS, frame count, duration) is displayed automatically")
bullet("Best for reproducible benchmarking and thesis experiments")

h3("Live Stream")
bullet("Connect to a live camera feed (webcam or RTSP stream)")
bullet("Camera index: integer (0 = default webcam, 1 = second camera, etc.)")
bullet("RTSP URL: for IP cameras and UAV video feeds (e.g., rtsp://192.168.1.10:554/stream)")
bullet("If an RTSP URL is provided, it takes priority over the camera index")
bullet("Total frame count is unknown for live streams (shown as 'streaming')")

h3("Image Folder")
bullet("Process a folder of images (.jpg, .jpeg, .png, .bmp, .tiff, .webp)")
bullet("Images are sorted alphabetically and processed sequentially")
bullet("Useful for evaluation datasets or frame-by-frame analysis")

h2("5.2 Detection Models")
para("Fast model (n):", bold=True)
para("The lightweight model optimized for speed. Typically YOLOv8n or a custom nano-variant.")

para("Accurate model (s):", bold=True)
para("The more capable model for complex scenes. Typically YOLOv8s or a custom small-variant.")

para("Device:", bold=True)
table(
    ["Device", "Description"],
    [
        ["cpu", "CPU inference (slowest, always available)"],
        ["cuda:0", "First NVIDIA GPU via CUDA"],
        ["cuda:N", "Nth NVIDIA GPU (multi-GPU systems)"],
        ["jetson:0", "NVIDIA Jetson platform (mapped to cuda:0 internally)"],
        ["jetson (remote)", "Placeholder for cross-device configuration"],
    ],
    col_widths=[1.8, 4.5],
)

para("Image size (imgsz):", bold=True)
para("Input image resolution for YOLO inference. Default 640. Range: 320\u20131280, step: 32.")

para("TensorRT Export:", bold=True)
para("When enabled, models are exported to TensorRT .engine format. Cached for reuse. See Section 11.")

h2("5.3 Policies")
table(
    ["Policy", "Description"],
    [
        ["n_only", "Always use the fast model (YOLOv8n). Baseline for speed."],
        ["s_only", "Always use the accurate model (YOLOv8s). Baseline for accuracy."],
        ["entropy_only", "Switch based on histogram entropy of the scene."],
        ["combined", "Switch based on composite score C (Laplacian + entropy)."],
        ["combined_hyst", "Switch with hysteresis band to prevent oscillation."],
    ],
    col_widths=[1.8, 4.5],
)

para("Enable Adaptive Budget:", bold=True)
para("When checked, the combined_hyst policy uses adaptive rolling percentile-based thresholds with a latency budget (default: 150 ms).")

h2("5.4 Controller Parameters")
para("All parameters have an 'Auto' checkbox. When Auto is checked, the default value is used. Uncheck Auto to manually adjust. See Section 8 for the complete reference.")

h2("5.5 Run and Stop")
bullet("Run Pipeline: Validates inputs and starts processing. Switches to Monitor tab.")
bullet("Stop: Gracefully stops the current run. Partial results are still available.")

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 6. THE MONITOR TAB
# ══════════════════════════════════════════════════════════════════════════
h1("6. The Monitor Tab")
para("The Monitor tab provides real-time visualization during pipeline execution.")

h2("6.1 Live Video Feed")
bullet("Green bounding boxes: detections from the fast model (n)")
bullet("Red bounding boxes: detections from the accurate model (s)")
bullet("Confidence scores displayed above each box")
bullet("Header overlay: policy, mode, active model, dwell, FPS, T_total, thresholds, penalty")

h2("6.2 Statistics Panel")
bullet("Frame index and active model choice")
bullet("Number of detections")
bullet("Laplacian variance (L), histogram entropy (H), composite score (C)")
bullet("Thresholds (c_low, c_high)")
bullet("Latency components: T_scene, T_ctrl, T_infer, T_total")
bullet("Average T_total and instantaneous FPS")

h2("6.3 Waveform Charts")
para("Three live-updating line charts (rolling window of 500 frames):")

para("T_total Waveform (cyan):", bold=True)
para("Per-frame total latency in milliseconds.")

para("C-score with Thresholds (orange line, red dashed thresholds):", bold=True)
para("Composite complexity score over time with threshold bands.")

para("Choice Timeline (green/red fill):", bold=True)
para("Green = fast model (n), Red = accurate model (s).")

note("Charts redraw every 3 frames for performance.")

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 7. THE RESULTS TAB
# ══════════════════════════════════════════════════════════════════════════
h1("7. The Results Tab")

h2("7.1 Comparison Table")
table(
    ["Column", "Description"],
    [
        ["Policy", "Name of the switching policy"],
        ["Mode", "'fixed' or 'adaptive'"],
        ["Mean T (ms)", "Average total latency per frame"],
        ["P95 T (ms)", "95th percentile total latency"],
        ["Slow %", "Percentage of frames using the s-model"],
        ["Sw/100", "Model switches per 100 frames"],
        ["Frames", "Total frames processed"],
    ],
    col_widths=[1.5, 4.8],
)

h2("7.2 Plot Tabs")
bullet("Timing Breakdown: stacked bar of T_scene, T_ctrl, T_infer_n, T_infer_s")
bullet("Latency Comparison: grouped bar of Mean, P95, P99")
bullet("T_total Waveforms: overlay line chart for all policies")
bullet("C-Score Traces: C-score lines with threshold bands")
bullet("Summary: text summary with best policy recommendation")

h2("7.3 Action Buttons")
bullet("Clear All: Remove all results")
bullet("Export CSV: Save summaries to CSV (user picks location)")
bullet("Export Plots: Save 4 PNG charts at 220 DPI (user picks folder)")

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 8. COMPLETE PARAMETER REFERENCE
# ══════════════════════════════════════════════════════════════════════════
h1("8. Complete Parameter Reference")

h2("8.1 Inference Parameters")
table(
    ["Parameter", "Default", "Range", "Description"],
    [
        ["imgsz", "640", "320\u20131280", "Input image size for YOLO. Larger = better accuracy, slower."],
        ["device", "cpu", "\u2014", "Compute device: cpu, cuda:N, jetson:N."],
        ["conf_min", "0.001", "0.0\u20131.0", "Minimum confidence for NMS filtering."],
        ["iou_nms", "0.45", "0.0\u20131.0", "IoU threshold for non-maximum suppression."],
        ["conf_show", "0.25", "0.0\u20131.0", "Minimum confidence to display a detection."],
        ["max_frames", "0", "0\u2013999999", "Max frames to process (0 = all)."],
        ["stride", "1", "1\u201330", "Process every Nth frame."],
    ],
    col_widths=[1.5, 0.8, 1.2, 3.0],
)

h2("8.2 Scene Analysis Parameters")
table(
    ["Parameter", "Default", "Range", "Description"],
    [
        ["alpha", "0.60", "0.0\u20131.0", "Weight of Laplacian in C. C = alpha*Ln + (1-alpha)*Hn."],
        ["proxy_size", "160", "32\u2013512", "Resize target for complexity proxy computation."],
        ["hist_bins", "64", "16\u2013256", "Histogram bins for entropy calculation."],
        ["probe_every_k", "1", "1\u201320", "Recompute scene proxies every k frames."],
    ],
    col_widths=[1.5, 0.8, 1.0, 3.2],
)

h2("8.3 Threshold Parameters")
table(
    ["Parameter", "Default", "Range", "Description"],
    [
        ["c_low", "0.45", "0.0\u20131.0", "Lower hysteresis threshold. Below \u2192 fast model."],
        ["c_high", "0.55", "0.0\u20131.0", "Upper hysteresis threshold. Above \u2192 accurate model."],
        ["combined_mid", "0.50", "0.0\u20131.0", "Single threshold for 'combined' policy."],
    ],
    col_widths=[1.5, 0.8, 1.0, 3.2],
)

h2("8.4 Stability Parameters")
table(
    ["Parameter", "Default", "Range", "Description"],
    [
        ["c_ema_beta", "0.25", "0.0\u20131.0", "EMA smoothing for C-score. Lower = smoother."],
        ["min_dwell_frames", "10", "1\u2013100", "Min frames between model switches."],
        ["max_switches_per_100", "12", "1\u201350", "Max switches per 100 frames."],
        ["history_window_size", "200", "10\u20132000", "Rolling window for percentile computations."],
    ],
    col_widths=[2.0, 0.7, 0.9, 2.9],
)

h2("8.5 Adaptive Parameters")
table(
    ["Parameter", "Default", "Range", "Description"],
    [
        ["norm_lo", "10.0", "0\u201350", "Lower percentile for normalizing L and H."],
        ["norm_hi", "90.0", "50\u2013100", "Upper percentile for normalizing L and H."],
        ["thr_lo", "35.0", "0\u2013100", "Percentile of C for c_low."],
        ["thr_hi", "65.0", "0\u2013100", "Percentile of C for c_high."],
        ["thr_mid", "50.0", "0\u2013100", "Percentile of C for c_mid."],
        ["latency_budget_ms", "150.0", "0\u20131000", "Target max average latency (ms)."],
        ["latency_penalty_factor", "0.010", "0.0\u20131.0", "Penalty aggressiveness when over budget."],
        ["budget_guard_margin", "0.10", "0.0\u20131.0", "Fraction above budget to trigger guard."],
        ["budget_guard_frames", "8", "0\u201350", "Frames to force n-model after guard triggers."],
    ],
    col_widths=[2.0, 0.7, 0.9, 2.9],
)

h2("8.6 Processing Parameters")
table(
    ["Parameter", "Default", "Range", "Description"],
    [
        ["latency_smooth_window", "15", "1\u2013100", "Window for smoothing avg T_total."],
    ],
    col_widths=[2.2, 0.7, 0.9, 2.7],
)

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 9. SWITCHING POLICIES EXPLAINED
# ══════════════════════════════════════════════════════════════════════════
h1("9. Switching Policies Explained")

h2("9.1 n_only (Fast Model Baseline)")
para("Always uses YOLOv8n. No scene analysis. Speed baseline.")

h2("9.2 s_only (Accurate Model Baseline)")
para("Always uses YOLOv8s. Accuracy baseline. Highest latency.")

h2("9.3 entropy_only")
para("Switches based on histogram entropy (H). If H > rolling median, use s-model.")

h2("9.4 combined")
para("Uses composite score C = alpha * Ln + (1-alpha) * Hn. If C >= combined_mid, use s-model.")

h2("9.5 combined_hyst (Recommended)")
para(
    "Uses C with hysteresis: switch n\u2192s when C > c_high, switch s\u2192n when C < c_low. "
    "Dead-band prevents oscillation. With Adaptive Budget: rolling percentile thresholds, "
    "latency penalty, budget guard, EMA smoothing, dwell and rate limiting."
)

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 10. ADAPTIVE BUDGET MODE
# ══════════════════════════════════════════════════════════════════════════
h1("10. Adaptive Budget Mode")

h2("10.1 How It Works")
bullet("1. Normalize L and H using rolling percentiles (norm_lo to norm_hi)")
bullet("2. Compute thresholds from rolling percentiles of C history")
bullet("3. Compute penalty = max(0, avg_latency - budget) * penalty_factor")
bullet("4. Add penalty to thresholds, shifting toward fast model")
bullet("5. Budget guard: force n-model for budget_guard_frames if latency >> budget")

h2("10.2 When to Use")
bullet("Strict real-time requirements (e.g., 30 FPS)")
bullet("Variable-complexity video where fixed thresholds under-perform")
bullet("Jetson deployment with thermal throttling")

h2("10.3 Tuning Tips")
bullet("Start with budget = 150 ms, adjust for your hardware")
bullet("Lower latency_penalty_factor for gentler shifts")
bullet("Increase history_window_size for stability")
bullet("Increase min_dwell_frames to reduce switching")

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 11. TENSORRT OPTIMIZATION GUIDE
# ══════════════════════════════════════════════════════════════════════════
h1("11. TensorRT Optimization Guide")

h2("11.1 How to Enable")
bullet("Check 'Export to TensorRT (.engine) before run' in Setup tab")
bullet("Select precision: fp16 (recommended), fp32, or int8")
bullet("Ensure CUDA or Jetson device is selected")
bullet("First run exports .engine files (several minutes)")
bullet("Subsequent runs reuse cached .engine files")

h2("11.2 Precision Modes")
table(
    ["Mode", "Speed", "Accuracy", "Best For"],
    [
        ["fp32", "Moderate", "Highest", "Debugging, accuracy-critical"],
        ["fp16", "Fast", "Near-identical", "General deployment (recommended)"],
        ["int8", "Fastest", "May degrade", "Maximum throughput on Jetson"],
    ],
    col_widths=[0.8, 1.2, 1.5, 3.0],
)

h2("11.3 Platform Availability")
table(
    ["Platform", "TensorRT Support"],
    [
        ["Windows + NVIDIA GPU", "Yes (install tensorrt via pip)"],
        ["Linux + NVIDIA GPU", "Yes (pip install tensorrt or JetPack)"],
        ["NVIDIA Jetson", "Yes (pre-installed in JetPack)"],
        ["macOS", "No (no NVIDIA GPU support)"],
        ["CPU-only systems", "No (TRT requires NVIDIA GPU)"],
    ],
    col_widths=[2.5, 3.8],
)

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 12. OUTPUT & EXPORT
# ══════════════════════════════════════════════════════════════════════════
h1("12. Output & Export")

h2("12.1 Where Results Are Saved")
para("DMS-Raptor does not auto-save. All outputs are on-demand via the Results tab.")

h2("12.2 CSV Export")
para("Click 'Export CSV'. Columns:")
table(
    ["Column", "Description"],
    [
        ["Policy", "Policy name"],
        ["Mode", "fixed or adaptive"],
        ["Frames", "Total frames processed"],
        ["Mean_T_total_ms", "Average total latency (ms)"],
        ["P95_T_total_ms", "95th percentile latency"],
        ["P99_T_total_ms", "99th percentile latency"],
        ["Slow_pct", "% frames using s-model"],
        ["Sw_per_100", "Switches per 100 frames"],
        ["Switches", "Total switches"],
        ["T_scene_mean", "Mean scene analysis time (ms)"],
        ["T_ctrl_mean", "Mean controller time (ms)"],
        ["T_inf_n_mean", "Mean n-model inference time (ms)"],
        ["T_inf_s_mean", "Mean s-model inference time (ms)"],
    ],
    col_widths=[2.0, 4.3],
)

h2("12.3 Plot Export")
para("Click 'Export Plots'. Four PNG files at 220 DPI:")
bullet("timing_breakdown.png")
bullet("latency_comparison.png")
bullet("T_total_waveforms.png")
bullet("C_score_traces.png")

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 13. TROUBLESHOOTING
# ══════════════════════════════════════════════════════════════════════════
h1("13. Troubleshooting")

# --- Windows ---
h2("13.1 Windows Issues")

h3("DLL load error / Application won't start")
para("Symptom: ImportError: DLL load failed while importing QtWidgets", italic=True)
bullet("Ensure correct Conda env: conda activate uav-thesis")
bullet("Reinstall PyQt5: pip install --force-reinstall PyQt5")
bullet("Install Visual C++ Redistributable 2015\u20132022")
bullet("If using PyQt6, switch to PyQt5 (known DLL issues on some Windows builds)")

h3("Unicode/encoding errors in console")
para("Symptom: UnicodeEncodeError when printing to Windows console", italic=True)
bullet("Set environment variable: set PYTHONIOENCODING=utf-8")
bullet("Use Windows Terminal (supports UTF-8) instead of cmd.exe")

h3("Antivirus blocks the .exe")
para("Symptom: Windows Defender or antivirus quarantines DMS-Raptor.exe", italic=True)
bullet("PyInstaller executables are sometimes flagged as false positives")
bullet("Add an exception for the DMS-Raptor folder in your antivirus settings")
bullet("Or run from source instead: python main.py")

# --- Linux ---
h2("13.2 Linux Issues")

h3("Qt platform plugin 'xcb' not found")
para("Symptom: 'Could not load the Qt platform plugin xcb'", italic=True)
code_block([
    "sudo apt install libxcb-xinerama0 libxcb-cursor0 \\",
    "    libxkbcommon-x11-0 libegl1 libgl1-mesa-glx",
])

h3("OpenCV headless conflict")
para("Symptom: cv2.imshow fails or import errors", italic=True)
bullet("Ensure opencv-python (not opencv-python-headless) is installed")
bullet("pip install opencv-python --force-reinstall")

h3("Permission denied on binary")
para("Symptom: ./DMS-Raptor: Permission denied", italic=True)
code_block(["chmod +x dist/DMS-Raptor/DMS-Raptor"])

# --- macOS ---
h2("13.3 macOS Issues")

h3("App is damaged / can't be opened")
para("Symptom: macOS Gatekeeper blocks unsigned app", italic=True)
bullet("Right-click the app > Open > Open (bypasses Gatekeeper)")
bullet("Or: xattr -cr /path/to/DMS-Raptor.app")
bullet("Or run from source: python main.py")

h3("No GPU acceleration")
para("Symptom: Only 'cpu' appears in device dropdown on macOS", italic=True)
bullet("macOS has no NVIDIA CUDA support")
bullet("Apple Silicon users: check if your PyTorch version supports 'mps' device")
bullet("CPU mode works for benchmarking but is slower")

# --- Jetson ---
h2("13.4 Jetson Issues")

h3("TensorRT export out of memory")
para("Symptom: OOM during .engine export on Jetson Nano (4GB)", italic=True)
bullet("Close all other applications to free memory")
bullet("Use a swap file: sudo fallocate -l 4G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")
bullet("Export on a more powerful machine and copy .engine files to Jetson")
bullet("Reduce imgsz (e.g., 320)")

h3("Very slow without TensorRT")
para("Symptom: Low FPS even on Jetson GPU", italic=True)
bullet("Enable TensorRT FP16 \u2014 this is critical for Jetson performance")
bullet("Set max performance: sudo nvpmodel -m 0 && sudo jetson_clocks")
bullet("Reduce imgsz to 320 or 480")

h3("PyQt5 installation fails")
para("Symptom: pip install PyQt5 fails on Jetson", italic=True)
bullet("Use system package: sudo apt install python3-pyqt5")
bullet("Or install from conda-forge: conda install -c conda-forge pyqt")

# --- General ---
h2("13.5 General Issues")

h3("CUDA device not detected")
bullet("Verify: python -c \"import torch; print(torch.cuda.is_available())\"")
bullet("Update NVIDIA drivers")
bullet("Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")

h3("Model loading fails")
bullet("Ensure .pt file exists and path is correct")
bullet("Use Browse button or absolute paths")
bullet("Verify ultralytics is installed: pip install ultralytics")

h3("Video won't open")
bullet("Verify video is not corrupted (test in VLC)")
bullet("Supported: .mp4, .avi, .mov, .mkv, .webm")
bullet("Ensure opencv-python has codec support")

h3("Live stream disconnects")
bullet("Check network for RTSP streams")
bullet("Verify RTSP URL and camera accessibility")
bullet("Consider recording to file first for reliability")

h3("Very slow / low FPS")
bullet("Use GPU (cuda:0), not CPU")
bullet("Enable TensorRT for NVIDIA GPUs")
bullet("Reduce imgsz (320 or 480)")
bullet("Increase stride to skip frames")
bullet("Close other GPU-intensive apps")

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 14. ARCHITECTURE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
h1("14. Architecture Overview")

h2("14.1 File Structure")
table(
    ["File", "Purpose"],
    [
        ["main.py", "Entry point; loads theme, creates MainWindow"],
        ["build_exe.py", "Build script for PyInstaller executables"],
        ["dms_raptor.spec", "PyInstaller spec file (directory mode)"],
        ["requirements.txt", "Python dependencies"],
        ["core/config.py", "Dataclasses: RunConfig, InferenceParams, FrameResult, RunSummary"],
        ["core/engine.py", "StreamingEngine: generator-based inference pipeline"],
        ["gui/main_window.py", "MainWindow: 3 tabs, signal wiring, worker management"],
        ["gui/setup_tab.py", "Configuration UI with Auto/Manual toggles"],
        ["gui/monitor_tab.py", "Live video + waveform charts (matplotlib)"],
        ["gui/results_tab.py", "Post-run dashboard with comparison plots and export"],
        ["gui/worker.py", "QThread worker: model loading, TRT export, policy iteration"],
        ["resources/style.qss", "Dark teal theme stylesheet"],
        ["resources/logo.svg", "Hexagonal raptor-eye application icon"],
    ],
    col_widths=[2.2, 4.1],
)

h2("14.2 Data Flow")
bullet("1. Setup Tab collects parameters and emits run_requested(dict)")
bullet("2. MainWindow creates an InferenceWorker (QThread)")
bullet("3. Worker loads models, optionally exports to TensorRT")
bullet("4. For each policy, Worker creates a StreamingEngine")
bullet("5. StreamingEngine.run() yields FrameResult per frame (generator)")
bullet("6. Worker emits frame_ready \u2192 Monitor Tab updates live display")
bullet("7. Worker emits policy_finished \u2192 Results Tab adds run summary")
bullet("8. Worker emits all_finished \u2192 UI switches to Results Tab")

h2("14.3 Scene Complexity Computation")
bullet("Laplacian variance (L): texture/edge density")
bullet("Histogram entropy (H): tonal diversity")
bullet("Composite: C = alpha * Ln + (1-alpha) * Hn (normalized to [0,1])")

h2("14.4 Actuator Stabilizers")
bullet("EMA smoothing on C-score (c_ema_beta)")
bullet("Minimum dwell: min_dwell_frames before switching")
bullet("Switch rate limiting: max_switches_per_100 per 100-frame window")
bullet("Budget guard: forces n-model when latency >> budget")
bullet("Hysteresis band: dead-band between c_low and c_high")

page_break()

# ══════════════════════════════════════════════════════════════════════════
# 15. GLOSSARY
# ══════════════════════════════════════════════════════════════════════════
h1("15. Glossary")
table(
    ["Term", "Definition"],
    [
        ("C-score", "Composite scene complexity score (Laplacian + entropy)."),
        ("DMS", "Dynamic Model Switching."),
        ("Dwell", "Consecutive frames the current model has been active."),
        ("EMA", "Exponential Moving Average \u2014 smoothing filter."),
        ("FPS", "Frames Per Second."),
        ("Hysteresis", "Separate up/down thresholds to prevent oscillation."),
        ("L (Laplacian)", "Variance of Laplacian. Measures texture complexity."),
        ("H (Entropy)", "Shannon entropy of intensity histogram."),
        ("NMS", "Non-Maximum Suppression."),
        ("P95 / P99", "95th / 99th percentile latency."),
        ("PyInstaller", "Tool to package Python apps as standalone executables."),
        ("RTSP", "Real-Time Streaming Protocol for IP cameras."),
        ("Slow %", "% frames using the accurate (slower) model."),
        ("Sw/100", "Model switches per 100 frames."),
        ("T_ctrl", "Controller decision time."),
        ("T_infer_n", "Fast model inference time."),
        ("T_infer_s", "Accurate model inference time."),
        ("T_scene", "Scene complexity computation time."),
        ("T_total", "Total per-frame time: T_scene + T_ctrl + T_infer."),
        ("TensorRT", "NVIDIA's inference optimizer for GPUs."),
        ("UAV", "Unmanned Aerial Vehicle."),
        ("YOLO", "You Only Look Once \u2014 object detection architecture."),
    ],
    col_widths=[1.8, 4.5],
)

# ── Footer ──────────────────────────────────────────────────────────────
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("End of DMS-Raptor User Manual v1.0.0")
run.font.size = Pt(10)
run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
run.italic = True

# ── Save ────────────────────────────────────────────────────────────────
doc.save(OUT)
print(f"User manual saved to: {OUT}")
