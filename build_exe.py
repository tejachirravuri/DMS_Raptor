"""
Build script for DMS-Raptor executable.

Usage:
    python build_exe.py          # Build for current platform
    python build_exe.py --onefile # Single-file .exe (larger, slower to start)
    python build_exe.py --clean   # Clean build artifacts first
"""

import os
import sys
import shutil
import subprocess
import platform

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(APP_DIR, "dist")
BUILD_DIR = os.path.join(APP_DIR, "build")
SPEC_FILE = os.path.join(APP_DIR, "dms_raptor.spec")


def clean():
    """Remove previous build artifacts."""
    for d in [DIST_DIR, BUILD_DIR]:
        if os.path.isdir(d):
            print(f"Removing {d}")
            shutil.rmtree(d)


def build(onefile=False):
    """Run PyInstaller to build the executable."""
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python:   {sys.version}")
    print()

    if onefile:
        # One-file mode: everything packed into a single exe
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--windowed",
            "--name", "DMS-Raptor",
            "--add-data", f"resources/style.qss{os.pathsep}resources",
            "--add-data", f"resources/style_light.qss{os.pathsep}resources",
            "--add-data", f"resources/logo.svg{os.pathsep}resources",
            "--add-data", f"resources/logo.png{os.pathsep}resources",
            "--add-data", f"resources/logo.ico{os.pathsep}resources",
            "--icon", "resources/logo.ico",
            "--hidden-import", "PyQt5.QtSvg",
            "--hidden-import", "matplotlib.backends.backend_qtagg",
            "--hidden-import", "docx",
            "--hidden-import", "docx.shared",
            "--hidden-import", "docx.enum.text",
            "--hidden-import", "docx.enum.table",
            "--hidden-import", "gui.thesis_tab",
            "--hidden-import", "gui.thesis_plots",
            "--hidden-import", "gui.thesis_worker",
            "--hidden-import", "gui.analysis_tab",
            "--hidden-import", "gui.analysis_worker",
            "--hidden-import", "core.analysis_engine",
            "--hidden-import", "core.analysis_runner",
            "--hidden-import", "core.csv_logger",
            "--hidden-import", "core.informed_gain",
            "--hidden-import", "core.controllers",
            "--hidden-import", "core.policies",
            "--hidden-import", "core.matching",
            "--hidden-import", "core.metrics",
            "--hidden-import", "core.proxies",
            "--hidden-import", "core.inference_backend",
            "--hidden-import", "scripts.generate_thesis_report",
            "--exclude-module", "tkinter",
            "--exclude-module", "pytest",
            "--exclude-module", "IPython",
            "main.py",
        ]
    else:
        # Directory mode using spec file (recommended)
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm",
            "--clean",
            SPEC_FILE,
        ]

    print("Running:", " ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, cwd=APP_DIR)

    if result.returncode == 0:
        if onefile:
            exe_path = os.path.join(DIST_DIR, "DMS-Raptor.exe" if platform.system() == "Windows" else "DMS-Raptor")
        else:
            exe_path = os.path.join(DIST_DIR, "DMS-Raptor")
        print()
        print("=" * 60)
        print(f"BUILD SUCCESSFUL!")
        print(f"Output: {exe_path}")
        print()
        print("To run:")
        if platform.system() == "Windows":
            if onefile:
                print(f'  "{exe_path}"')
            else:
                print(f'  "{os.path.join(exe_path, "DMS-Raptor.exe")}"')
        else:
            if onefile:
                print(f"  ./{os.path.basename(exe_path)}")
            else:
                print(f"  ./{os.path.join(os.path.basename(exe_path), 'DMS-Raptor')}")
        print()
        print("IMPORTANT: YOLO model weights (.pt files) are NOT bundled.")
        print("Users must provide their own model files at runtime.")
    else:
        print()
        print("BUILD FAILED. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--clean" in args:
        clean()
        if len(args) == 1:
            print("Clean complete.")
            sys.exit(0)

    onefile = "--onefile" in args
    build(onefile=onefile)
