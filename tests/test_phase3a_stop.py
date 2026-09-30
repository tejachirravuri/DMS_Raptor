"""Phase 3A tests — cancellation support.

Tests:
  1. Runner stops before second policy when should_stop fires
  2. Completed policy CSV is still written after stop
  3. Summary CSV written for completed policies only
  4. AnalysisWorker has request_stop method

Run: python tests/test_phase3a_stop.py
"""
from __future__ import annotations

import csv
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PASS = 0
FAIL = 0


def check(label, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {label}")
        PASS += 1
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        FAIL += 1


from core.inference_backend import Detections


class StubBackend:
    def __init__(self, n_boxes=1, score=0.9):
        self._n = n_boxes
        self._score = score

    def predict(self, frame_bgr):
        if self._n == 0:
            return Detections.empty()
        boxes = np.array(
            [[i * 20, 0, i * 20 + 15, 15] for i in range(self._n)], dtype=float,
        )
        return Detections(
            boxes=boxes,
            scores=np.full(self._n, self._score, dtype=float),
            classes=np.zeros(self._n, dtype=int),
        )


def _make_test_video(n_frames=10, w=64, h=64):
    import cv2
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp.name, fourcc, 10.0, (w, h))
    for _ in range(n_frames):
        writer.write(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    writer.release()
    return tmp.name


# ===================================================================
# 1. Runner stops before second policy
# ===================================================================
def test_stop_before_second():
    print("\n=== 1. Runner stops before second policy ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    policies_run = []
    stop_after_first = {"count": 0}

    def _progress(pname, status):
        if status == "done":
            policies_run.append(pname)

    def _should_stop():
        return len(policies_run) >= 1

    def _check():
        summaries = run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only", "conf_ema"],
            out_dir=out_dir,
            max_frames=N,
            progress_callback=_progress,
            should_stop=_should_stop,
        )
        assert len(summaries) == 1, f"expected 1, got {len(summaries)}"
        assert "n_only" in summaries, f"expected n_only, got {list(summaries)}"

    check("runner stops after first policy", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 2. Completed policy CSV written after stop
# ===================================================================
def test_csv_written_on_stop():
    print("\n=== 2. Completed policy CSV on stop ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    done = []

    def _progress(p, s):
        if s == "done":
            done.append(p)

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only"],
            out_dir=out_dir,
            max_frames=N,
            progress_callback=_progress,
            should_stop=lambda: len(done) >= 1,
        )
        csv_path = out_dir / "n_only_per_frame.csv"
        assert csv_path.exists(), "per-frame CSV should be written"
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == N

        assert not (out_dir / "s_only_per_frame.csv").exists(), \
            "s_only CSV should NOT exist (stopped before it ran)"

    check("completed policy CSV written, skipped policy absent", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 3. Summary CSV written for completed policies only
# ===================================================================
def test_summary_csv_on_stop():
    print("\n=== 3. Summary CSV on stop ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    done = []

    def _progress(p, s):
        if s == "done":
            done.append(p)

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only", "conf_ema"],
            out_dir=out_dir,
            max_frames=N,
            progress_callback=_progress,
            should_stop=lambda: len(done) >= 1,
        )
        summary_csv = out_dir / "analysis_policy_summary.csv"
        assert summary_csv.exists(), "summary CSV should be written"
        with open(summary_csv) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
        assert rows[0]["policy"] == "n_only"

    check("summary CSV has one row for completed policy", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 4. AnalysisWorker has request_stop
# ===================================================================
def test_worker_has_request_stop():
    print("\n=== 4. AnalysisWorker.request_stop ===")
    from gui.analysis_worker import AnalysisWorker

    def _check():
        assert hasattr(AnalysisWorker, "request_stop"), "missing request_stop"
        w = AnalysisWorker(
            video_path="dummy.mp4",
            fast_weights="dummy.pt",
            accurate_weights="dummy.pt",
            policies=["n_only"],
            out_dir="/tmp",
        )
        assert not w._stop_requested
        w.request_stop()
        assert w._stop_requested

    check("AnalysisWorker.request_stop sets flag", _check)


# ===================================================================
# 5. No should_stop = full run (backwards compat)
# ===================================================================
def test_no_should_stop():
    print("\n=== 5. No should_stop = full run ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        summaries = run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only"],
            out_dir=out_dir,
            max_frames=N,
        )
        assert len(summaries) == 2

    check("no should_stop runs all policies", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_stop_before_second()
    test_csv_written_on_stop()
    test_summary_csv_on_stop()
    test_worker_has_request_stop()
    test_no_should_stop()

    print(f"\n{'='*50}")
    print(f"TOTAL: {PASS + FAIL} checks | PASS: {PASS} | FAIL: {FAIL}")
    if FAIL:
        print("*** SOME CHECKS FAILED ***")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
