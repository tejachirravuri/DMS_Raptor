"""
Master runner: Execute all research experiments in sequence.

This is the overnight script. Run it and go to sleep.

Usage:
    python research/run_all.py                    # full run (12+ hours on CPU)
    python research/run_all.py --quick             # quick validation (stride=5, 3 videos)
    python research/run_all.py --phase 1           # run only phase 1
    python research/run_all.py --phase 2           # run only phase 2
    python research/run_all.py --phase 3           # run only phases 3-5

Estimated times on CPU (i5-10210U):
    Phase 1A (model baselines): ~10 min
    Phase 1B+C (detection quality, all 12 videos): ~8-12 hours
    Phase 2 (3x trials, 5 videos): ~6-8 hours
    Phase 4 (ablation, 4 videos): ~4-6 hours
    Phase 5 (generalization): ~3-4 hours

Quick mode (stride=5): ~2-3 hours total
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = sys.executable
RESEARCH_DIR = Path(__file__).resolve().parent
LOG_DIR = RESEARCH_DIR.parent / "research_results" / "logs"


def run_script(name: str, args: list, timeout_hours: float = 12):
    """Run a research script and log output."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    cmd = [PYTHON, str(RESEARCH_DIR / name)] + args
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Args:    {' '.join(args)}")
    print(f"  Log:     {log_file}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"\n  >> {name}: {status} ({elapsed/60:.1f} min)")
    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: stride=5, fewer videos")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only this phase (1, 2, or 3)")
    args = parser.parse_args()

    stride = ["--stride", "5"] if args.quick else []
    max_frames = ["--max-frames", "500"] if args.quick else []

    print(f"DMS-Raptor Research Pipeline")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Phase: {'ALL' if args.phase == 0 else args.phase}")
    print(f"Started: {datetime.now().isoformat()}")

    # ── Phase 1A: Model baselines ──
    if args.phase in (0, 1):
        run_script("01_model_baselines.py", [])

    # ── Phase 1B+C: Detection quality ──
    if args.phase in (0, 1):
        run_script("02_detection_quality.py",
                   ["--all"] + stride + max_frames)

    # ── Phase 2: Repeated trials ──
    if args.phase in (0, 2):
        run_script("03_repeated_trials.py",
                   ["--all", "--repeats", "3"] + stride + max_frames)

    # ── Phase 4: Ablation study ──
    if args.phase in (0, 3):
        run_script("04_ablation_study.py",
                   ["--all"] + stride + max_frames)

    # ── Phase 5: Cross-video generalization ──
    if args.phase in (0, 3):
        run_script("05_cross_video_generalization.py", [])

    print(f"\n{'='*60}")
    print(f"  ALL PHASES COMPLETE")
    print(f"  Finished: {datetime.now().isoformat()}")
    print(f"  Results:  {RESEARCH_DIR.parent / 'research_results'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
