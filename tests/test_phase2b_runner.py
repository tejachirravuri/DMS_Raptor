"""Phase 2B unit tests — multi-policy analysis runner.

Tests:
  1. Runner writes per-policy CSVs
  2. Runner writes analysis_policy_summary.csv
  3. Informed gain computed when both baselines present
  4. Informed gain is NaN when baselines missing
  5. Summary CSV has correct columns
  6. Per-policy summary JSONs exist
  7. mean_t_proxy_ms present in summary

Run: python tests/test_phase2b_runner.py
"""
from __future__ import annotations

import csv
import json
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


# ===================================================================
# Stub backend
# ===================================================================
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
# 1. Runner writes per-policy CSVs
# ===================================================================
def test_per_policy_csvs():
    print("\n=== 1. Per-policy CSVs ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())
    policies = ["n_only", "s_only", "conf_ema"]

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=policies,
            out_dir=out_dir,
            max_frames=N,
        )
        for p in policies:
            csv_path = out_dir / f"{p}_per_frame.csv"
            assert csv_path.exists(), f"missing: {csv_path}"
            with open(csv_path) as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == N, f"{p}: expected {N} rows, got {len(rows)}"

    check("per-policy CSVs written with correct row count", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 2. Runner writes summary CSV
# ===================================================================
def test_summary_csv_exists():
    print("\n=== 2. Summary CSV ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only"],
            out_dir=out_dir,
            max_frames=N,
        )
        summary_path = out_dir / "analysis_policy_summary.csv"
        assert summary_path.exists()
        with open(summary_path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2

    check("analysis_policy_summary.csv exists with correct row count", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 3. Informed gain computed when baselines present
# ===================================================================
def test_informed_gain_with_baselines():
    print("\n=== 3. Informed gain with baselines ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        summaries = run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only", "conf_ema"],
            out_dir=out_dir,
            max_frames=N,
        )
        conf_ema_s = summaries["conf_ema"]
        assert "informed_gain_iou_match" in conf_ema_s, "missing informed_gain"
        assert "expected_random_iou_match" in conf_ema_s, "missing expected_random"
        gain = conf_ema_s["informed_gain_iou_match"]
        assert np.isfinite(gain), f"gain should be finite, got {gain}"

    check("informed gain computed for conf_ema", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 4. Informed gain is NaN when baselines missing
# ===================================================================
def test_informed_gain_without_baselines():
    print("\n=== 4. Informed gain without baselines ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        summaries = run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["conf_ema"],
            out_dir=out_dir,
            max_frames=N,
        )
        gain = summaries["conf_ema"]["informed_gain_iou_match"]
        assert np.isnan(gain), f"gain should be NaN without baselines, got {gain}"

    check("informed gain is NaN without baselines", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 5. Summary CSV has correct columns
# ===================================================================
def test_summary_columns():
    print("\n=== 5. Summary CSV columns ===")
    from core.analysis_runner import run_analysis_policies, SUMMARY_ALL_COLUMNS

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only", "conf_ema"],
            out_dir=out_dir,
            max_frames=N,
        )
        with open(out_dir / "analysis_policy_summary.csv") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
        for col in SUMMARY_ALL_COLUMNS:
            assert col in header, f"missing column: {col}"

    check("summary CSV has all required columns", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 6. Per-policy summary JSONs exist
# ===================================================================
def test_summary_jsons():
    print("\n=== 6. Per-policy summary JSONs ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())
    policies = ["n_only", "s_only", "entropy_only"]

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=policies,
            out_dir=out_dir,
            max_frames=N,
        )
        for p in policies:
            json_path = out_dir / f"{p}_summary.json"
            assert json_path.exists(), f"missing: {json_path}"
            with open(json_path) as fh:
                data = json.load(fh)
            assert data["n_frames"] == N

    check("per-policy summary JSONs exist and contain n_frames", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 7. mean_t_proxy_ms in summary
# ===================================================================
def test_proxy_ms_in_summary():
    print("\n=== 7. mean_t_proxy_ms in summary ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        summaries = run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["entropy_only"],
            out_dir=out_dir,
            max_frames=N,
        )
        s = summaries["entropy_only"]
        assert "mean_t_proxy_ms" in s, "missing mean_t_proxy_ms"
        assert s["mean_t_proxy_ms"] > 0.0, "scene policy should have proxy cost"

    check("mean_t_proxy_ms present and >0 for scene policy", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 8. Baselines have NaN gain in summary CSV
# ===================================================================
def test_baselines_blank_gain():
    print("\n=== 8. Baselines have blank gain in CSV ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only", "conf_ema"],
            out_dir=out_dir,
            max_frames=N,
        )
        with open(out_dir / "analysis_policy_summary.csv") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            if row["policy"] in ("n_only", "s_only"):
                assert row["informed_gain_iou_match"] == "", (
                    f"{row['policy']}: gain should be blank, got "
                    f"{row['informed_gain_iou_match']!r}"
                )

    check("n_only and s_only have blank gain in summary CSV", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 9. Informed gain in summary CSV is finite for dynamic policy
# ===================================================================
def test_gain_in_summary_csv():
    print("\n=== 9. Informed gain in summary CSV ===")
    from core.analysis_runner import run_analysis_policies

    N = 5
    video = _make_test_video(N)
    out_dir = Path(tempfile.mkdtemp())

    def _check():
        run_analysis_policies(
            video_path=video,
            fast_backend=StubBackend(1),
            accurate_backend=StubBackend(2),
            policies=["n_only", "s_only", "conf_ema"],
            out_dir=out_dir,
            max_frames=N,
        )
        with open(out_dir / "analysis_policy_summary.csv") as fh:
            rows = {r["policy"]: r for r in csv.DictReader(fh)}
        gain = rows["conf_ema"]["informed_gain_iou_match"]
        assert gain != "", "conf_ema gain should not be blank"
        assert np.isfinite(float(gain)), f"gain should be finite, got {gain}"

    check("conf_ema has finite gain in summary CSV", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_per_policy_csvs()
    test_summary_csv_exists()
    test_informed_gain_with_baselines()
    test_informed_gain_without_baselines()
    test_summary_columns()
    test_summary_jsons()
    test_proxy_ms_in_summary()
    test_baselines_blank_gain()
    test_gain_in_summary_csv()

    print(f"\n{'='*50}")
    print(f"TOTAL: {PASS + FAIL} checks | PASS: {PASS} | FAIL: {FAIL}")
    if FAIL:
        print("*** SOME CHECKS FAILED ***")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
