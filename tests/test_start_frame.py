"""Start-frame support tests.

Tests:
  1. Default start_frame=0 produces frame indices starting at 0
  2. start_frame=3 produces frame indices starting at 3
  3. max_frames respected after start_frame offset
  4. Runner passes start_frame through to analysis engine

Run: python tests/test_start_frame.py
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


def _make_test_video(n_frames=20, w=64, h=64):
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
# 1. Default start_frame=0 gives indices from 0
# ===================================================================
def test_default_start_frame():
    print("\n=== 1. Default start_frame=0 ===")
    from core.analysis_engine import run_analysis

    video = _make_test_video(10)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(1),
            policy_name="n_only",
            max_frames=5,
        ))
        assert len(results) == 5
        assert results[0].frame_idx == 0, f"first idx={results[0].frame_idx}"
        assert results[-1].frame_idx == 4, f"last idx={results[-1].frame_idx}"

    check("default start_frame=0 yields indices 0..4", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 2. start_frame=3 offsets frame indices
# ===================================================================
def test_start_frame_offset():
    print("\n=== 2. start_frame=3 ===")
    from core.analysis_engine import run_analysis

    video = _make_test_video(20)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(1),
            policy_name="n_only",
            max_frames=5,
            start_frame=3,
        ))
        assert len(results) == 5, f"got {len(results)} results"
        assert results[0].frame_idx == 3, f"first idx={results[0].frame_idx}"
        assert results[-1].frame_idx == 7, f"last idx={results[-1].frame_idx}"

    check("start_frame=3 yields indices 3..7", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 3. max_frames limits count after start_frame
# ===================================================================
def test_max_frames_after_start():
    print("\n=== 3. max_frames after start_frame ===")
    from core.analysis_engine import run_analysis

    video = _make_test_video(20)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(1),
            policy_name="s_only",
            max_frames=3,
            start_frame=10,
        ))
        assert len(results) == 3, f"got {len(results)} results"
        assert results[0].frame_idx == 10
        assert results[-1].frame_idx == 12

    check("max_frames=3 from start_frame=10 yields 3 results (10,11,12)", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 4. Runner passes start_frame through
# ===================================================================
def test_runner_start_frame():
    print("\n=== 4. Runner passes start_frame ===")
    from core.analysis_runner import run_analysis_policies

    video = _make_test_video(20)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(1),
            policies=["n_only"],
            out_dir=out_dir,
            max_frames=4,
            start_frame=5,
        )
        csv_path = out_dir / "n_only_per_frame.csv"
        assert csv_path.exists()
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4, f"got {len(rows)} rows"
        assert int(rows[0]["frame_idx"]) == 5, f"first frame_idx={rows[0]['frame_idx']}"
        assert int(rows[-1]["frame_idx"]) == 8, f"last frame_idx={rows[-1]['frame_idx']}"

    check("runner CSV shows frame indices 5..8", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_default_start_frame()
    test_start_frame_offset()
    test_max_frames_after_start()
    test_runner_start_frame()

    print(f"\n{'='*50}")
    print(f"TOTAL: {PASS + FAIL} checks | PASS: {PASS} | FAIL: {FAIL}")
    if FAIL:
        print("*** SOME CHECKS FAILED ***")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
