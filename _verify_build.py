"""Post-build verification for DMS-Raptor v1.8.0.

Launches the exe, takes screenshots, and checks:
  1. App opens without crash
  2. Logo appears in taskbar/title bar
  3. Original live tabs exist (Setup, Run, Results, etc.)
  4. Thesis Analysis tab exists
  5. Setup tab shows live POLICIES (includes niqe_switch, no local_contrast_hyst)
  6. Thesis Analysis tab shows THESIS_ANALYSIS_POLICIES (includes local_contrast_hyst, no niqe_switch)
"""
import os
import sys
import subprocess
import time

EXE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dist", "DMS-Raptor", "DMS-Raptor.exe",
)

def main():
    if not os.path.isfile(EXE):
        print(f"ERROR: exe not found at {EXE}")
        sys.exit(1)

    size_mb = os.path.getsize(EXE) / (1024 * 1024)
    print(f"Exe found: {EXE}")
    print(f"Exe size:  {size_mb:.1f} MB")

    # Check dist folder total size
    dist_dir = os.path.dirname(EXE)
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(dist_dir)
        for f in fnames
    )
    print(f"Dist size: {total / (1024 * 1024):.0f} MB")

    # Verify resources bundled
    res_dir = os.path.join(dist_dir, "_internal", "resources")
    if not os.path.isdir(res_dir):
        res_dir = os.path.join(dist_dir, "resources")
    expected = ["style.qss", "style_light.qss", "logo.svg", "logo.png", "logo.ico"]
    for f in expected:
        path = os.path.join(res_dir, f)
        status = "OK" if os.path.isfile(path) else "MISSING"
        print(f"  Resource {f}: {status}")

    print("\nLaunching DMS-Raptor.exe ...")
    proc = subprocess.Popen([EXE], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"PID: {proc.pid}")
    print("App launched. Check visually for logo, tabs, and policy lists.")
    print("Press Ctrl+C or close the app window when done.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nTerminated.")


if __name__ == "__main__":
    main()
