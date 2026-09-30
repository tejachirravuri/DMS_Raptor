"""Phase 2A unit tests — thesis-analysis engine correctness.

Tests per-policy-class runtime topology:
  Static baselines:   n_only  -> t_deployed = t_fast
                      s_only  -> t_deployed = t_accurate
  Scene policies:     choice n -> t_deployed = t_proxy + t_ctrl + t_fast
                      choice s -> t_deployed = t_proxy + t_ctrl + t_accurate
  conf_ema:           choice n -> t_deployed = t_fast + t_ctrl
                      choice s -> t_deployed = t_fast + t_ctrl + t_accurate

Plus: confidence source, output selection, benefit_positive, CSV columns,
      evaluation-only cost isolation.

Run: python tests/test_phase2a_analysis.py
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


# ===================================================================
# Stub backend
# ===================================================================
from core.inference_backend import Detections


class StubBackend:
    """Returns pre-programmed detections per call index."""

    def __init__(self, detections_sequence):
        self._seq = list(detections_sequence)
        self._idx = 0

    def predict(self, frame_bgr):
        if self._idx < len(self._seq):
            d = self._seq[self._idx]
        else:
            d = Detections.empty()
        self._idx += 1
        return d

    @property
    def call_count(self):
        return self._idx


def _make_dets(n_boxes, score=0.9, cls=0):
    if n_boxes == 0:
        return Detections.empty()
    boxes = np.array(
        [[i * 20, 0, i * 20 + 15, 15] for i in range(n_boxes)], dtype=float,
    )
    scores = np.full(n_boxes, score, dtype=float)
    classes = np.full(n_boxes, cls, dtype=int)
    return Detections(boxes=boxes, scores=scores, classes=classes)


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
# 1. n_only: t_deployed = t_fast (no ctrl, no accurate, no proxy)
# ===================================================================
def test_n_only_deployed():
    print("\n=== 1. n_only deployed cost ===")
    from core.analysis_engine import run_analysis

    N = 5
    video = _make_test_video(N)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(2)] * (N * 2)),
            policy_name="n_only",
            max_frames=N,
        ))
        for r in results:
            assert r.choice == "n"
            assert r.selected_model == "n"
            assert r.accurate_selected is False
            assert len(r.selected_dets) == 1
            assert r.t_fast_ms > 0.0
            assert r.t_accurate_ms == 0.0, (
                f"n_only: t_accurate_ms should be 0, got {r.t_accurate_ms}"
            )
            assert r.t_ctrl_ms == 0.0, (
                f"n_only: t_ctrl_ms should be 0, got {r.t_ctrl_ms}"
            )
            assert r.t_proxy_ms == 0.0
            assert abs(r.t_total_deployed_ms - r.t_fast_ms) < 0.01, (
                f"n_only: t_deployed={r.t_total_deployed_ms:.3f} "
                f"!= t_fast={r.t_fast_ms:.3f}"
            )

    check("n_only: t_deployed = t_fast only", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 2. s_only: t_deployed = t_accurate (no fast cost, no ctrl, no proxy)
# ===================================================================
def test_s_only_deployed():
    print("\n=== 2. s_only deployed cost ===")
    from core.analysis_engine import run_analysis

    N = 5
    video = _make_test_video(N)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(3)] * (N * 2)),
            policy_name="s_only",
            max_frames=N,
        ))
        for r in results:
            assert r.choice == "s"
            assert r.selected_model == "s"
            assert r.accurate_selected is True
            assert len(r.selected_dets) == 3
            assert r.t_accurate_ms > 0.0
            assert r.t_fast_ms == 0.0, (
                f"s_only: t_fast_ms should be 0, got {r.t_fast_ms}"
            )
            assert r.t_ctrl_ms == 0.0, (
                f"s_only: t_ctrl_ms should be 0, got {r.t_ctrl_ms}"
            )
            assert r.t_proxy_ms == 0.0
            assert abs(r.t_total_deployed_ms - r.t_accurate_ms) < 0.01, (
                f"s_only: t_deployed={r.t_total_deployed_ms:.3f} "
                f"!= t_accurate={r.t_accurate_ms:.3f}"
            )

    check("s_only: t_deployed = t_accurate only", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 3. conf_ema choosing n: t_deployed = t_fast + t_ctrl
# ===================================================================
def test_conf_ema_n_deployed():
    print("\n=== 3. conf_ema (n) deployed cost ===")
    from core.analysis_engine import run_analysis
    from core.policies import PolicyConfig

    N = 5
    video = _make_test_video(N)
    cfg = PolicyConfig(
        name="conf_ema",
        conf_ema_warmup=999,
        min_dwell_frames=1,
    )

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1, score=0.9)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(2)] * (N * 2)),
            policy_name="conf_ema",
            policy_config=cfg,
            max_frames=N,
        ))
        for r in results:
            assert r.choice == "n", f"warmup should force n, got {r.choice}"
            assert r.t_fast_ms > 0.0, "conf_ema always runs fast"
            assert r.t_ctrl_ms > 0.0, "conf_ema has ctrl cost"
            assert r.t_accurate_ms == 0.0, (
                f"conf_ema(n): t_accurate_ms should be 0, got {r.t_accurate_ms}"
            )
            expected = r.t_fast_ms + r.t_ctrl_ms
            assert abs(r.t_total_deployed_ms - expected) < 0.01, (
                f"conf_ema(n): t_deployed={r.t_total_deployed_ms:.3f} "
                f"!= fast+ctrl={expected:.3f}"
            )

    check("conf_ema(n): t_deployed = t_fast + t_ctrl", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 4. conf_ema choosing s: t_deployed = t_fast + t_ctrl + t_accurate
# ===================================================================
def test_conf_ema_s_deployed():
    print("\n=== 4. conf_ema (s) deployed cost ===")
    from core.analysis_engine import run_analysis
    from core.policies import PolicyConfig

    N = 8
    video = _make_test_video(N)
    cfg = PolicyConfig(
        name="conf_ema",
        conf_ema_warmup=2,
        conf_ema_fast_beta=0.8,
        conf_ema_slow_beta=0.2,
        conf_ema_c_low=0.001,
        conf_ema_c_high=0.002,
        min_dwell_frames=1,
        max_switches_per_100=50,
    )
    fast_scores = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    fast_seq = [_make_dets(2, score=s) for s in fast_scores]
    acc_seq = [_make_dets(3, score=0.95) for _ in range(N * 2)]

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend(fast_seq),
            accurate_backend=StubBackend(acc_seq),
            policy_name="conf_ema",
            policy_config=cfg,
            max_frames=N,
        ))
        s_found = False
        for r in results:
            assert r.t_fast_ms > 0.0, "conf_ema always pays fast"
            assert r.t_ctrl_ms > 0.0
            if r.choice == "s":
                s_found = True
                assert r.t_accurate_ms > 0.0
                expected = r.t_fast_ms + r.t_ctrl_ms + r.t_accurate_ms
                assert abs(r.t_total_deployed_ms - expected) < 0.01, (
                    f"conf_ema(s): t_deployed={r.t_total_deployed_ms:.3f} "
                    f"!= fast+ctrl+acc={expected:.3f}"
                )
        assert s_found, "expected at least one escalation frame"

    check("conf_ema(s): t_deployed = t_fast + t_ctrl + t_accurate", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 5. Scene policy choosing n: t_deployed = t_proxy + t_ctrl + t_fast
# ===================================================================
def test_scene_n_deployed():
    print("\n=== 5. scene policy (n) deployed cost ===")
    from core.analysis_engine import run_analysis
    from core.policies import PolicyConfig

    N = 5
    video = _make_test_video(N)
    cfg = PolicyConfig(
        name="entropy_only",
        history_window_size=50,
        min_dwell_frames=1,
        max_switches_per_100=50,
    )

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(2)] * (N * 2)),
            policy_name="entropy_only",
            policy_config=cfg,
            max_frames=N,
        ))
        n_frames = [r for r in results if r.choice == "n"]
        assert len(n_frames) > 0, "expected at least one n choice"
        for r in n_frames:
            assert r.t_proxy_ms > 0.0, "scene policy must pay proxy cost"
            assert r.t_ctrl_ms > 0.0, "scene policy has ctrl cost"
            assert r.t_fast_ms > 0.0, "selected n pays fast cost"
            assert r.t_accurate_ms == 0.0, (
                f"scene(n): t_accurate_ms should be 0, got {r.t_accurate_ms}"
            )
            expected = r.t_proxy_ms + r.t_ctrl_ms + r.t_fast_ms
            assert abs(r.t_total_deployed_ms - expected) < 0.01, (
                f"scene(n): t_deployed={r.t_total_deployed_ms:.3f} "
                f"!= proxy+ctrl+fast={expected:.3f}"
            )

    check("scene(n): t_deployed = t_proxy + t_ctrl + t_fast", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 6. Scene policy choosing s: t_deployed = t_proxy + t_ctrl + t_accurate
# ===================================================================
def test_scene_s_deployed():
    print("\n=== 6. scene policy (s) deployed cost ===")
    from core.analysis_engine import run_analysis
    from core.policies import PolicyConfig

    N = 5
    video = _make_test_video(N)
    cfg = PolicyConfig(
        name="entropy_only",
        history_window_size=50,
        min_dwell_frames=1,
        max_switches_per_100=50,
    )

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(2)] * (N * 2)),
            policy_name="entropy_only",
            policy_config=cfg,
            max_frames=N,
        ))
        s_frames = [r for r in results if r.choice == "s"]
        assert len(s_frames) > 0, "expected at least one s choice"
        for r in s_frames:
            assert r.t_proxy_ms > 0.0
            assert r.t_ctrl_ms > 0.0
            assert r.t_accurate_ms > 0.0
            assert r.t_fast_ms == 0.0, (
                f"scene(s): t_fast_ms should be 0, got {r.t_fast_ms}"
            )
            expected = r.t_proxy_ms + r.t_ctrl_ms + r.t_accurate_ms
            assert abs(r.t_total_deployed_ms - expected) < 0.01, (
                f"scene(s): t_deployed={r.t_total_deployed_ms:.3f} "
                f"!= proxy+ctrl+acc={expected:.3f}"
            )

    check("scene(s): t_deployed = t_proxy + t_ctrl + t_accurate", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 7. Evaluation-only runs don't affect deployed cost
# ===================================================================
def test_eval_only_cost_isolation():
    print("\n=== 7. evaluation-only cost isolation ===")
    from core.analysis_engine import run_analysis

    N = 3
    video = _make_test_video(N)

    def _n_only():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(2)] * (N * 2)),
            policy_name="n_only",
            max_frames=N,
        ))
        for r in results:
            assert r.accurate_reference_ran is True, "eval should run accurate"
            assert r.t_accurate_ms == 0.0, "eval-only accurate must not be in deployed"
            assert r.t_total_deployed_ms == r.t_fast_ms

    def _s_only():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(2)] * (N * 2)),
            policy_name="s_only",
            max_frames=N,
        ))
        for r in results:
            assert r.fast_reference_ran is True, "eval should run fast"
            assert r.t_fast_ms == 0.0, "eval-only fast must not be in deployed"
            assert r.t_total_deployed_ms == r.t_accurate_ms

    check("n_only: eval-only accurate not in deployed", _n_only)
    check("s_only: eval-only fast not in deployed", _s_only)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 8. conf_ema receives fast confidence, not selected
# ===================================================================
def test_conf_ema_fast_confidence():
    print("\n=== 8. conf_ema fast confidence only ===")
    from core.analysis_engine import run_analysis
    from core.policies import PolicyConfig

    N = 6
    fast_score = 0.35
    cfg = PolicyConfig(name="conf_ema", conf_ema_warmup=2)

    video = _make_test_video(N)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(2, score=fast_score)] * N),
            accurate_backend=StubBackend([_make_dets(2, score=0.95)] * (N * 2)),
            policy_name="conf_ema",
            policy_config=cfg,
            max_frames=N,
        ))
        for r in results:
            assert abs(r.fast_mean_conf - fast_score) < 0.01, (
                f"frame {r.frame_idx}: fast_mean_conf={r.fast_mean_conf}"
            )

    check("conf_ema always reports fast confidence", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 9. Selected output correctness
# ===================================================================
def test_selected_output():
    print("\n=== 9. selected output correctness ===")
    from core.analysis_engine import run_analysis

    N = 5
    video = _make_test_video(N)

    def _n():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1, score=0.9)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(3, score=0.8)] * (N * 2)),
            policy_name="n_only", max_frames=N,
        ))
        for r in results:
            assert len(r.selected_dets) == 1
            assert r.selected_model == "n"

    def _s():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1, score=0.9)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(3, score=0.8)] * (N * 2)),
            policy_name="s_only", max_frames=N,
        ))
        for r in results:
            assert len(r.selected_dets) == 3
            assert r.selected_model == "s"

    check("n_only: selected = fast (1 det)", _n)
    check("s_only: selected = accurate (3 dets)", _s)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 10. benefit_positive uses fast + accurate (not selected)
# ===================================================================
def test_benefit_positive():
    print("\n=== 10. benefit_positive uses fast+accurate ===")
    from core.analysis_engine import run_analysis

    N = 3
    video = _make_test_video(N)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1, score=0.9)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(2, score=0.8)] * (N * 2)),
            policy_name="n_only", max_frames=N,
        ))
        for r in results:
            assert r.benefit_positive is True

    check("benefit_positive True when accurate has extra det", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 11. CSV header contains required columns
# ===================================================================
def test_csv_header():
    print("\n=== 11. CSV header columns ===")
    from core.csv_logger import COLUMNS

    required = [
        "frame_idx", "policy", "choice", "selected_model",
        "accurate_selected", "accurate_reference_ran", "fast_reference_ran",
        "fast_mean_conf",
        "policy_score", "policy_thresh_low", "policy_thresh_high",
        "t_fast_ms", "t_accurate_ms", "t_ctrl_ms",
        "t_proxy_ms", "t_total_deployed_ms",
        "iou_match", "det_coverage", "det_recall",
        "count_agree", "benefit_positive",
        "L", "H", "color_entropy", "tenengrad", "edge_density",
        "local_contrast", "bright_fraction", "hue_std",
    ]

    def _check():
        for col in required:
            assert col in COLUMNS, f"missing required column: {col}"

    check("all required columns present", _check)


# ===================================================================
# 12. summarise_analysis keys
# ===================================================================
def test_summary_keys():
    print("\n=== 12. summarise_analysis keys ===")
    from core.analysis_engine import run_analysis, summarise_analysis

    N = 5
    video = _make_test_video(N)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(1)] * (N * 2)),
            policy_name="n_only", max_frames=N,
        ))
        summary = summarise_analysis(results)
        for k in [
            "n_frames", "accurate_usage_rate",
            "mean_iou_match", "mean_det_coverage", "mean_det_recall",
            "mean_count_agree", "benefit_rate",
            "mean_t_total_deployed_ms", "mean_t_fast_ms",
            "mean_t_accurate_ms", "mean_t_ctrl_ms",
        ]:
            assert k in summary, f"missing key: {k}"
        assert summary["n_frames"] == N

    check("summary contains all required keys", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 13. CSV round-trip
# ===================================================================
def test_csv_roundtrip():
    print("\n=== 13. CSV round-trip ===")
    from core.analysis_engine import run_analysis
    from core.csv_logger import COLUMNS, write_results_csv

    N = 5
    video = _make_test_video(N)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(1)] * (N * 2)),
            accurate_backend=StubBackend([_make_dets(1)] * (N * 2)),
            policy_name="n_only", max_frames=N,
        ))
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()
        count = write_results_csv(results, tmp.name)
        assert count == N
        with open(tmp.name, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for col in COLUMNS:
                assert col in reader.fieldnames, f"missing: {col}"
            rows = list(reader)
            assert len(rows) == N
        Path(tmp.name).unlink(missing_ok=True)

    check("CSV write + read round-trip", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# 14. conf_ema: fast_mean_conf tracks fast scores across frames
# ===================================================================
def test_conf_ema_no_feedback_loop():
    print("\n=== 14. conf_ema no feedback loop ===")
    from core.analysis_engine import run_analysis
    from core.policies import PolicyConfig

    N = 8
    fast_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.5, 0.8]
    cfg = PolicyConfig(
        name="conf_ema", conf_ema_warmup=2,
        conf_ema_c_low=0.10, conf_ema_c_high=0.20,
        min_dwell_frames=1, max_switches_per_100=50,
    )
    video = _make_test_video(N)

    def _check():
        results = list(run_analysis(
            video_path=video,
            fast_backend=StubBackend([_make_dets(2, score=s) for s in fast_scores]),
            accurate_backend=StubBackend([_make_dets(3, score=0.95)] * (N * 2)),
            policy_name="conf_ema", policy_config=cfg, max_frames=N,
        ))
        for i, r in enumerate(results):
            assert abs(r.fast_mean_conf - fast_scores[i]) < 0.01, (
                f"frame {i}: fast_mean_conf={r.fast_mean_conf:.4f}, "
                f"expected={fast_scores[i]:.4f}"
            )
            if r.choice == "s":
                assert r.selected_model == "s"
                assert len(r.selected_dets) == 3
            else:
                assert r.selected_model == "n"
                assert len(r.selected_dets) == 2

    check("conf_ema: fast_mean_conf tracks fast scores", _check)
    Path(video).unlink(missing_ok=True)


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_n_only_deployed()
    test_s_only_deployed()
    test_conf_ema_n_deployed()
    test_conf_ema_s_deployed()
    test_scene_n_deployed()
    test_scene_s_deployed()
    test_eval_only_cost_isolation()
    test_conf_ema_fast_confidence()
    test_selected_output()
    test_benefit_positive()
    test_csv_header()
    test_summary_keys()
    test_csv_roundtrip()
    test_conf_ema_no_feedback_loop()

    print(f"\n{'='*50}")
    print(f"TOTAL: {PASS + FAIL} checks | PASS: {PASS} | FAIL: {FAIL}")
    if FAIL:
        print("*** SOME CHECKS FAILED ***")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
