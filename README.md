# DMS-Raptor

**Dynamic Model Switching for Real-Time UAV Inspection**
Ganapathi Teja Chirravuri — TU Chemnitz, Faculty of Computer Science

DMS-Raptor is a desktop application for running and analysing **dynamic
model switching** on UAV inspection footage. A lightweight policy decides,
per frame, whether to run a fast detector or switch to a slower, more
accurate one — keeping latency low on easy frames while recovering
detections on hard ones. The GUI wraps the whole workflow: live
monitoring, batch runs, frame extraction, validation, and the offline
thesis-analysis pipeline.

Built with PyQt5; detection runs on Ultralytics YOLO (PyTorch).

## Getting the application

There are two ways to run DMS-Raptor. Running from source works on
**Windows, Linux, and macOS** and is the smallest download. A packaged
Windows build is also available for a double-click experience.

### Option A — Run from source (Windows / Linux / macOS)

```bash
# 1. Clone
git clone https://github.com/tejachirravuri/DMS_Raptor.git
cd DMS_Raptor

# 2. Create an environment (Python 3.10 recommended)
conda create -n dms-raptor python=3.10 -y
conda activate dms-raptor
#    — or without conda —
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
python main.py
```

The first launch takes 15–20 seconds while PyTorch and Ultralytics load —
this is normal, not a freeze.

### Option B — Packaged Windows build (double-click)

Download the latest **`DMS-Raptor-Windows.zip`** from the
[Releases page](https://github.com/tejachirravuri/DMS_Raptor/releases),
extract it anywhere, and run **`DMS-Raptor.exe`** inside the extracted
folder. No Python installation is required.

> The packaged build is large (~1.8 GB) because it bundles PyTorch and
> the CUDA/vision libraries the detector needs. If no release is posted
> yet, use Option A, or build it yourself (below).

## Providing model weights

Detector weights are **not** bundled — you point the app at your own
trained YOLO models at runtime. In the **Setup** tab, select a *fast (n)*
model and an *accurate (s)* model (`.pt`, `.onnx`, or `.engine`), choose
an input video or image folder, pick the policies to run, and start.

## Features

The interface is organised into tabs:

| Tab                | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| Setup              | choose models, input, policies, and runtime parameters       |
| Monitor            | live annotated frames and per-frame timing while a run plays |
| Results            | per-policy metrics, plots, and saved run history             |
| Batch Run          | queue several videos / policy sets in one session            |
| Frame Extraction   | pull frames from video for labelling or inspection           |
| Validation         | frame-level agreement checks against the accurate reference  |
| Thesis Experiments | reproduce the thesis sweeps and figures                      |
| Thesis Analysis    | offline dual-model evaluation and summary tables             |

## Building the Windows executable

To produce the packaged build yourself (on Windows, inside the
environment above with PyInstaller installed):

```bash
pip install pyinstaller
python build_exe.py
```

The bundle is written to `dist/DMS-Raptor/`; run
`dist/DMS-Raptor/DMS-Raptor.exe`. To publish it, zip that folder as
`DMS-Raptor-Windows.zip` and upload it as a GitHub Release asset.

## Repository layout

```
main.py            application entry point
build_exe.py       PyInstaller build script
dms_raptor.spec    PyInstaller spec
requirements.txt   runtime dependencies
gui/               PyQt5 windows, tabs, and worker threads
core/              switching engine, policies, proxies, metrics, config
resources/         icons and stylesheets
research/          thesis experiment scripts
scripts/           report / manual generation helpers
validation/        offline validation module
rl_switching/      reinforcement-learning switching experiments
tests/             unit tests
```

## Documentation

A full user manual (`DMS-Raptor_User_Manual_v1.7.0.docx`) is included in
the repository root.
