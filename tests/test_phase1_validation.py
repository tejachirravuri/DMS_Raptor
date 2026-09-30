"""Phase 1 validation — confirms all thesis-compatible modules import
and behave correctly without touching engine.py or the GUI.

Run: python -m pytest tests/test_phase1_validation.py -v
  or: python tests/test_phase1_validation.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

PASS = 0
FAIL = 0


def check(label: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {label}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        FAIL += 1


# ===================================================================
# 1. Module imports
# ===================================================================
def test_imports():
    print("\n=== 1. Module imports ===")

    def _controllers():
        from core.controllers import RollingPercentile, Hysteresis, SwitchStabiliser
        assert RollingPercentile and Hysteresis and SwitchStabiliser

    def _policies():
        from core.policies import (
            make_policy, POLICY_REGISTRY, CANONICAL_POLICIES,
            EXPERIMENTAL_POLICIES, FrameFeatures, PolicyConfig,
        )
        assert len(POLICY_REGISTRY) == 8
        assert len(CANONICAL_POLICIES) == 7
        assert EXPERIMENTAL_POLICIES == ["local_contrast_hyst"]

    def _matching():
        from core.matching import iou_matrix, box_iou, greedy_match, filter_by_score
        assert iou_matrix and box_iou and greedy_match and filter_by_score

    def _metrics():
        from core.metrics import (
            iou_match_frame, det_recall_frame, det_coverage_frame,
            count_agree_frame, benefit_positive_frame,
            latency_summary, summarise_run,
        )
        assert iou_match_frame and benefit_positive_frame

    def _proxies():
        from core.proxies import ProxyConfig, compute_proxies
        assert ProxyConfig and compute_proxies

    def _inference():
        from core.inference_backend import Detections, InferenceBackend, time_inference
        assert Detections and time_inference

    def _informed_gain():
        from core.informed_gain import (
            expected_random_metric, informed_gain, compute_informed_gains, GAIN_METRICS,
        )
        assert len(GAIN_METRICS) == 4

    check("controllers", _controllers)
    check("policies", _policies)
    check("matching", _matching)
    check("metrics", _metrics)
    check("proxies", _proxies)
    check("inference_backend", _inference)
    check("informed_gain", _informed_gain)


# ===================================================================
# 2. make_policy for all 8 policies
# ===================================================================
def test_make_policy():
    print("\n=== 2. make_policy (all 8 policies) ===")
    from core.policies import make_policy, POLICY_REGISTRY

    for name in sorted(POLICY_REGISTRY):
        def _make(n=name):
            p = make_policy(n)
            assert p is not None
            assert hasattr(p, "decide")
            assert hasattr(p, "reset")
        check(f"make_policy('{name}')", _make)


# ===================================================================
# 3. local_contrast_hyst decides on sample features
# ===================================================================
def test_local_contrast_hyst():
    print("\n=== 3. local_contrast_hyst decide ===")
    from core.policies import make_policy, FrameFeatures, PolicyConfig

    def _decide_sequence():
        cfg = PolicyConfig(
            local_contrast_low=0.40,
            local_contrast_high=0.60,
            history_window_size=50,
        )
        p = make_policy("local_contrast_hyst", cfg)
        choices = []
        for i in range(60):
            lc = 30.0 + i * 2.0
            ff = FrameFeatures(frame_idx=i, local_contrast=lc)
            choices.append(p.decide(ff))
        assert "n" in choices, "should produce at least one 'n'"
        assert "s" in choices, "should produce at least one 's'"
        assert all(c in ("n", "s") for c in choices)

    check("local_contrast_hyst sequence", _decide_sequence)


# ===================================================================
# 4. informed_gain hand-calculated example
# ===================================================================
def test_informed_gain():
    print("\n=== 4. informed_gain (hand-calculated) ===")
    from core.informed_gain import expected_random_metric, informed_gain

    def _basic():
        fast = np.array([1.0, 0.0, 1.0, 0.0])
        accurate = np.array([1.0, 1.0, 1.0, 1.0])
        u = 0.5
        er = expected_random_metric(fast, accurate, u)
        # (1-0.5)*[1,0,1,0] + 0.5*[1,1,1,1] = [1, 0.5, 1, 0.5]
        # mean = 0.75
        assert abs(er - 0.75) < 1e-9, f"expected 0.75, got {er}"

    def _gain_positive():
        er = 0.75
        policy_val = 0.90
        g = informed_gain(policy_val, er)
        assert abs(g - 0.15) < 1e-9, f"expected 0.15, got {g}"

    def _gain_zero():
        g = informed_gain(0.75, 0.75)
        assert abs(g) < 1e-9, f"expected 0.0, got {g}"

    def _u_zero():
        fast = np.array([0.8, 0.6])
        accurate = np.array([1.0, 1.0])
        er = expected_random_metric(fast, accurate, 0.0)
        assert abs(er - 0.7) < 1e-9, f"u=0 should give fast mean 0.7, got {er}"

    def _u_one():
        fast = np.array([0.8, 0.6])
        accurate = np.array([1.0, 1.0])
        er = expected_random_metric(fast, accurate, 1.0)
        assert abs(er - 1.0) < 1e-9, f"u=1 should give accurate mean 1.0, got {er}"

    check("expected_random_metric basic", _basic)
    check("informed_gain positive", _gain_positive)
    check("informed_gain zero", _gain_zero)
    check("expected_random u=0 (pure fast)", _u_zero)
    check("expected_random u=1 (pure accurate)", _u_one)


# ===================================================================
# 5. matching on synthetic boxes
# ===================================================================
def test_matching():
    print("\n=== 5. matching (synthetic boxes) ===")
    from core.matching import iou_matrix, box_iou, greedy_match, MatchResult

    def _iou_identical():
        box = np.array([[10, 10, 50, 50]])
        m = iou_matrix(box, box)
        assert abs(m[0, 0] - 1.0) < 1e-9

    def _iou_disjoint():
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[20, 20, 30, 30]])
        assert abs(box_iou(a, b)) < 1e-9

    def _iou_partial():
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 0, 15, 10])
        iou = box_iou(a, b)
        # intersection = 5*10=50, union = 100+100-50=150
        assert abs(iou - 50.0 / 150.0) < 1e-6, f"expected 0.333, got {iou}"

    def _greedy_perfect():
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        res = greedy_match(boxes, boxes, 0.5)
        assert len(res.pairs) == 2
        assert len(res.unmatched_a) == 0
        assert len(res.unmatched_b) == 0

    def _greedy_mismatch():
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
        res = greedy_match(a, b, 0.5)
        assert len(res.pairs) == 1
        assert len(res.unmatched_b) == 1

    def _greedy_empty():
        empty = np.zeros((0, 4))
        full = np.array([[0, 0, 10, 10]])
        res = greedy_match(empty, full, 0.5)
        assert len(res.pairs) == 0
        assert len(res.unmatched_b) == 1

    check("IoU identical boxes", _iou_identical)
    check("IoU disjoint boxes", _iou_disjoint)
    check("IoU partial overlap", _iou_partial)
    check("greedy_match perfect", _greedy_perfect)
    check("greedy_match count mismatch", _greedy_mismatch)
    check("greedy_match empty vs non-empty", _greedy_empty)


# ===================================================================
# 6. metrics on synthetic boxes
# ===================================================================
def test_metrics():
    print("\n=== 6. metrics (synthetic boxes) ===")
    from core.metrics import (
        iou_match_frame, det_recall_frame, det_coverage_frame,
        count_agree_frame, benefit_positive_frame,
        latency_summary, summarise_run,
    )

    boxes_a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    boxes_b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    boxes_c = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]])

    def _iou_match_same():
        assert iou_match_frame(boxes_a, boxes_b) is True

    def _iou_match_diff():
        assert iou_match_frame(boxes_a, boxes_c) is False

    def _recall_perfect():
        assert abs(det_recall_frame(boxes_a, boxes_b) - 1.0) < 1e-9

    def _recall_partial():
        r = det_recall_frame(boxes_a, boxes_c)
        assert abs(r - 2.0 / 3.0) < 1e-6

    def _coverage_yes():
        assert det_coverage_frame(boxes_a, boxes_b) is True

    def _coverage_no():
        assert det_coverage_frame(boxes_a, boxes_c) is False

    def _count_agree():
        assert count_agree_frame(boxes_a, boxes_b) is True
        assert count_agree_frame(boxes_a, boxes_c) is False

    def _benefit_positive():
        fast = np.array([[0, 0, 10, 10]])
        accurate = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
        assert benefit_positive_frame(fast, accurate) is True

    def _benefit_negative():
        same = np.array([[0, 0, 10, 10]])
        assert benefit_positive_frame(same, same) is False

    def _benefit_empty_accurate():
        fast = np.array([[0, 0, 10, 10]])
        empty = np.zeros((0, 4))
        assert benefit_positive_frame(fast, empty) is False

    def _latency():
        lat = latency_summary([10.0, 20.0, 30.0])
        assert lat["n"] == 3
        assert abs(lat["mean"] - 20.0) < 1e-9

    check("iou_match same", _iou_match_same)
    check("iou_match different count", _iou_match_diff)
    check("det_recall perfect", _recall_perfect)
    check("det_recall partial", _recall_partial)
    check("det_coverage yes", _coverage_yes)
    check("det_coverage no", _coverage_no)
    check("count_agree", _count_agree)
    check("benefit_positive true", _benefit_positive)
    check("benefit_positive false (same)", _benefit_negative)
    check("benefit_positive false (empty accurate)", _benefit_empty_accurate)
    check("latency_summary", _latency)


# ===================================================================
# 7. Detections dataclass
# ===================================================================
def test_detections():
    print("\n=== 7. Detections dataclass ===")
    from core.inference_backend import Detections

    def _create():
        d = Detections(
            boxes=np.array([[0, 0, 10, 10]]),
            scores=np.array([0.9]),
            classes=np.array([0]),
        )
        assert len(d) == 1

    def _empty():
        d = Detections.empty()
        assert len(d) == 0

    def _filter():
        d = Detections(
            boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            scores=np.array([0.3, 0.8]),
            classes=np.array([0, 1]),
        )
        filtered = d.filter_score(0.5)
        assert len(filtered) == 1
        assert abs(filtered.scores[0] - 0.8) < 1e-9

    check("Detections create", _create)
    check("Detections.empty()", _empty)
    check("Detections.filter_score()", _filter)


# ===================================================================
# 8. proxies (on synthetic image)
# ===================================================================
def test_proxies():
    print("\n=== 8. proxies (synthetic image) ===")
    from core.proxies import ProxyConfig, compute_proxies

    def _compute():
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cfg = ProxyConfig(size=160, features=("L", "H", "local_contrast"))
        out = compute_proxies(img, cfg)
        assert "L" in out
        assert "H" in out
        assert "local_contrast" in out
        assert "time_ms" in out
        assert out["proxy_size"] == 160

    def _all_features():
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cfg = ProxyConfig(size=160)
        out = compute_proxies(img, cfg)
        assert len([k for k in out if not k.startswith("time") and k != "proxy_size"]) == 9

    check("compute_proxies (3 features)", _compute)
    check("compute_proxies (all 9 features)", _all_features)


# ===================================================================
# 9. controllers unit checks
# ===================================================================
def test_controllers():
    print("\n=== 9. controllers ===")
    from core.controllers import RollingPercentile, Hysteresis, SwitchStabiliser

    def _rolling_pct():
        rp = RollingPercentile(window=5)
        for v in [10, 20, 30, 40, 50]:
            rp.add(v)
        med = rp.percentile(50)
        assert abs(med - 30.0) < 1e-9, f"expected 30, got {med}"

    def _hysteresis():
        h = Hysteresis(c_low=0.3, c_high=0.7, initial="n")
        assert h.step(0.5) == "n"
        assert h.step(0.8) == "s"
        assert h.step(0.5) == "s"
        assert h.step(0.2) == "n"

    def _stabiliser_dwell():
        st = SwitchStabiliser(min_dwell_frames=5, max_switches_per_100=50)
        st.step(0, "n")
        assert st.step(1, "s") == "n", "should suppress switch within dwell"
        for i in range(2, 6):
            st.step(i, "n")
        assert st.step(6, "s") == "s", "should allow switch after dwell"

    check("RollingPercentile median", _rolling_pct)
    check("Hysteresis state machine", _hysteresis)
    check("SwitchStabiliser dwell", _stabiliser_dwell)


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    test_imports()
    test_make_policy()
    test_local_contrast_hyst()
    test_informed_gain()
    test_matching()
    test_metrics()
    test_detections()
    test_proxies()
    test_controllers()

    print(f"\n{'='*50}")
    print(f"TOTAL: {PASS + FAIL} checks | PASS: {PASS} | FAIL: {FAIL}")
    if FAIL:
        print("*** SOME CHECKS FAILED ***")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
