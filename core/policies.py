"""Thesis-compatible DMS policies — decision functions mapping per-frame features to a model choice.

Each policy is a stateful object with a clean interface::

    policy = make_policy("conf_ema", PolicyConfig(...))
    for frame_idx, features in stream:
        choice = policy.decide(features)   # returns 'n' or 's'

Canonical policies (the 7 evaluated in the thesis sweep):
    1. n_only         — always fast (latency baseline)
    2. s_only         — always accurate (always-accurate reference)
    3. entropy_only   — switch on H >= rolling-median(H)
    4. combined       — linear threshold on weighted (L, H)
    5. combined_hyst  — combined with hysteresis
    6. conf_ema       — switch on fast-model confidence drop (EMA)
    7. multi_proxy    — weighted EMA-deviation composite

Exploratory policy:
    local_contrast_hyst — structural-texture policy for glass-like scenes.
                          Exploratory evidence, not part of the locked
                          7-policy baseline. Included in final thesis
                          figures as agreement-oriented candidate.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .controllers import Hysteresis, RollingPercentile, SwitchStabiliser


_NAN = float("nan")
_DIAG_KEYS = (
    "policy_score",
    "policy_thresh_low",
    "policy_thresh_high",
    "policy_thresh_mid",
    "fast_ema",
    "slow_ema",
    "conf_drop",
)


def _empty_diag() -> Dict[str, float]:
    return {k: _NAN for k in _DIAG_KEYS}


# ===========================================================================
# Frame features — input to a policy on frame k
# ===========================================================================
@dataclass
class FrameFeatures:
    """Observable signals available to a policy on frame k."""

    frame_idx: int

    L: Optional[float] = None
    H: Optional[float] = None
    color_entropy: Optional[float] = None
    tenengrad: Optional[float] = None
    edge_density: Optional[float] = None
    local_contrast: Optional[float] = None
    bright_fraction: Optional[float] = None
    hue_std: Optional[float] = None

    # CONTRACT: must be the FAST-MODEL mean confidence on frame k-1,
    # not the chosen-model confidence. Mixing fast and accurate score
    # distributions across the EMA history corrupts conf_ema's trigger.
    last_mean_conf: float = 0.0


# ===========================================================================
# Policy configuration
# ===========================================================================
@dataclass
class PolicyConfig:
    """Shared configuration for all policies.

    Defaults match the thesis sweep for reproducibility.
    """

    name: str = ""

    alpha: float = 0.6
    L_direction: int = 1
    H_direction: int = 1

    history_window_size: int = 200
    norm_lo_pct: float = 10.0
    norm_hi_pct: float = 90.0

    c_mid: float = 0.50
    c_low: float = 0.45
    c_high: float = 0.55

    # LocalContrastHyst (exploratory)
    local_contrast_low: float = 0.45
    local_contrast_high: float = 0.65

    # ConfEMA
    conf_ema_fast_beta: float = 0.30
    conf_ema_slow_beta: float = 0.02
    conf_ema_c_low: float = 0.04
    conf_ema_c_high: float = 0.12
    conf_ema_warmup: int = 5

    # MultiProxy
    mp_w: Dict[str, float] = field(default_factory=lambda: {
        "L": 0.37,
        "H": 0.29,
        "tenengrad": 0.38,
        "color_entropy": 0.28,
        "edge_density": 0.0,
        "local_contrast": 0.0,
    })
    mp_fast_beta: float = 0.30
    mp_slow_beta: float = 0.02
    mp_c_low: float = 0.03
    mp_c_high: float = 0.10
    mp_directional: bool = False

    # Stabilisers
    min_dwell_frames: int = 10
    max_switches_per_100: int = 12


# ===========================================================================
# Policy base class
# ===========================================================================
class Policy(ABC):
    """Abstract base class for switching policies."""

    name: str = "base"

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.last_diag: Dict[str, float] = _empty_diag()

    @abstractmethod
    def decide(self, features: FrameFeatures) -> str:
        """Return 'n' or 's' for this frame."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal state. Override in stateful policies."""


# ===========================================================================
# Trivial policies
# ===========================================================================
class NOnly(Policy):
    """Always fast. Latency baseline."""
    name = "n_only"

    def decide(self, features: FrameFeatures) -> str:
        return "n"


class SOnly(Policy):
    """Always accurate. Accurate-reference baseline."""
    name = "s_only"

    def decide(self, features: FrameFeatures) -> str:
        return "s"


# ===========================================================================
# Scene-feature policies
# ===========================================================================
class EntropyOnly(Policy):
    """Switch when current H >= rolling-median(H).

    Relative-difficulty policy by construction: ~50% of frames go to 's'.
    """
    name = "entropy_only"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.rolling_H = RollingPercentile(config.history_window_size)
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def decide(self, features: FrameFeatures) -> str:
        if features.H is None:
            raise ValueError("EntropyOnly requires features.H")
        self.rolling_H.add(features.H)
        H_med = self.rolling_H.percentile(50)
        requested = "s" if features.H >= H_med else "n"
        self.last_diag = {
            "policy_score": float(features.H),
            "policy_thresh_low": float(H_med),
            "policy_thresh_high": float(H_med),
            "policy_thresh_mid": float(H_med),
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.rolling_H = RollingPercentile(self.config.history_window_size)
        self.stabiliser.reset()


class Combined(Policy):
    """Linear threshold on weighted (L, H) with explicit direction signs.

    C = alpha * signed(L_normalised) + (1 - alpha) * signed(H_normalised)
    """
    name = "combined"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.rolling_L = RollingPercentile(config.history_window_size)
        self.rolling_H = RollingPercentile(config.history_window_size)
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def _compute_C(self, features: FrameFeatures) -> float:
        if features.L is None or features.H is None:
            raise ValueError("Combined requires features.L and features.H")
        self.rolling_L.add(features.L)
        self.rolling_H.add(features.H)
        L_lo = self.rolling_L.percentile(self.config.norm_lo_pct)
        L_hi = self.rolling_L.percentile(self.config.norm_hi_pct)
        H_lo = self.rolling_H.percentile(self.config.norm_lo_pct)
        H_hi = self.rolling_H.percentile(self.config.norm_hi_pct)
        L_n = max(0.0, min(1.0,
                           (features.L - L_lo) / max(1e-9, L_hi - L_lo)))
        H_n = max(0.0, min(1.0,
                           (features.H - H_lo) / max(1e-9, H_hi - H_lo)))
        L_signed = L_n if self.config.L_direction >= 0 else 1.0 - L_n
        H_signed = H_n if self.config.H_direction >= 0 else 1.0 - H_n
        return (
            self.config.alpha * L_signed
            + (1.0 - self.config.alpha) * H_signed
        )

    def decide(self, features: FrameFeatures) -> str:
        C = self._compute_C(features)
        requested = "s" if C >= self.config.c_mid else "n"
        self.last_diag = {
            "policy_score": float(C),
            "policy_thresh_low": float(self.config.c_mid),
            "policy_thresh_high": float(self.config.c_mid),
            "policy_thresh_mid": float(self.config.c_mid),
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.rolling_L = RollingPercentile(self.config.history_window_size)
        self.rolling_H = RollingPercentile(self.config.history_window_size)
        self.stabiliser.reset()


class CombinedHyst(Combined):
    """Combined with hysteresis (dual thresholds c_low / c_high)."""
    name = "combined_hyst"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.hyst = Hysteresis(config.c_low, config.c_high, initial="n")

    def decide(self, features: FrameFeatures) -> str:
        C = self._compute_C(features)
        requested = self.hyst.step(C)
        self.last_diag = {
            "policy_score": float(C),
            "policy_thresh_low": float(self.config.c_low),
            "policy_thresh_high": float(self.config.c_high),
            "policy_thresh_mid": _NAN,
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        super().reset()
        self.hyst.reset()


class LocalContrastHyst(Policy):
    """Exploratory local-contrast hysteresis policy.

    Uses rolling-percentile-normalised local_contrast with hysteresis.
    Higher local contrast favours the accurate model. Included in final
    thesis figures as agreement-oriented candidate for glass-like scenes.

    EXPLORATORY: not part of the locked 7-policy baseline. Treat as
    evidence for structural-texture-based switching, not as a universally
    optimal solution.
    """
    name = "local_contrast_hyst"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.rolling_C = RollingPercentile(config.history_window_size)
        self.hyst = Hysteresis(
            config.local_contrast_low,
            config.local_contrast_high,
            initial="n",
        )
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def _score(self, features: FrameFeatures) -> float:
        if features.local_contrast is None:
            raise ValueError("LocalContrastHyst requires features.local_contrast")
        value = float(features.local_contrast)
        self.rolling_C.add(value)
        lo = self.rolling_C.percentile(self.config.norm_lo_pct)
        hi = self.rolling_C.percentile(self.config.norm_hi_pct)
        return max(0.0, min(1.0, (value - lo) / max(1e-9, hi - lo)))

    def decide(self, features: FrameFeatures) -> str:
        score = self._score(features)
        requested = self.hyst.step(score)
        self.last_diag = {
            "policy_score": float(score),
            "policy_thresh_low": float(self.config.local_contrast_low),
            "policy_thresh_high": float(self.config.local_contrast_high),
            "policy_thresh_mid": _NAN,
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.rolling_C = RollingPercentile(self.config.history_window_size)
        self.hyst.reset()
        self.stabiliser.reset()


# ===========================================================================
# Confidence-driven policy
# ===========================================================================
class ConfEMA(Policy):
    """Switch when fast-model confidence drops, measured by fast vs slow EMA.

    Input contract (LOCKED): ``features.last_mean_conf`` MUST be the
    fast-model mean confidence from the previous frame, regardless of
    which model the policy selected. Mixing fast and accurate score
    distributions corrupts the deviation signal.
    """
    name = "conf_ema"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.hyst = Hysteresis(
            config.conf_ema_c_low, config.conf_ema_c_high, initial="n",
        )
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def _conf_drop(self, last_mean_conf: float) -> float:
        bf = self.config.conf_ema_fast_beta
        bs = self.config.conf_ema_slow_beta
        if self.fast_ema is None:
            self.fast_ema = float(last_mean_conf)
            self.slow_ema = float(last_mean_conf)
        else:
            self.fast_ema = (
                bf * float(last_mean_conf) + (1.0 - bf) * self.fast_ema
            )
            self.slow_ema = (
                bs * float(last_mean_conf) + (1.0 - bs) * self.slow_ema
            )
        denom = self.slow_ema + 1e-9
        return max(0.0, (self.slow_ema - self.fast_ema) / denom)

    def decide(self, features: FrameFeatures) -> str:
        c = self._conf_drop(features.last_mean_conf)
        if features.frame_idx < self.config.conf_ema_warmup:
            requested = "n"
        else:
            requested = self.hyst.step(c)
        self.last_diag = {
            "policy_score": float(c),
            "policy_thresh_low": float(self.config.conf_ema_c_low),
            "policy_thresh_high": float(self.config.conf_ema_c_high),
            "policy_thresh_mid": _NAN,
            "fast_ema": float(self.fast_ema)
            if self.fast_ema is not None else _NAN,
            "slow_ema": float(self.slow_ema)
            if self.slow_ema is not None else _NAN,
            "conf_drop": float(c),
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.fast_ema = None
        self.slow_ema = None
        self.hyst.reset()
        self.stabiliser.reset()


# ===========================================================================
# Composite policy
# ===========================================================================
class MultiProxy(Policy):
    """Weighted EMA-deviation composite over multiple scene proxies.

    Documented in thesis as falsified-hypothesis baseline: naive feature
    fusion amplifies noise rather than improving routing.
    """
    name = "multi_proxy"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.ema: Dict[str, tuple] = {}
        self.hyst = Hysteresis(config.mp_c_low, config.mp_c_high, initial="n")
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def _composite(self, features: FrameFeatures) -> float:
        bf = self.config.mp_fast_beta
        bs = self.config.mp_slow_beta
        proxy_values = {
            "L": features.L,
            "H": features.H,
            "color_entropy": features.color_entropy,
            "tenengrad": features.tenengrad,
            "edge_density": features.edge_density,
            "local_contrast": features.local_contrast,
        }
        weighted = 0.0
        wsum = 0.0
        for pname, val in proxy_values.items():
            w = self.config.mp_w.get(pname, 0.0)
            if abs(w) < 1e-12 or val is None:
                continue
            v = float(val)
            if pname not in self.ema:
                self.ema[pname] = (v, v)
            else:
                f_prev, s_prev = self.ema[pname]
                self.ema[pname] = (
                    bf * v + (1.0 - bf) * f_prev,
                    bs * v + (1.0 - bs) * s_prev,
                )
            f_v, s_v = self.ema[pname]
            denom = s_v + 1e-9
            if self.config.mp_directional:
                drop = (s_v - f_v) / denom
            else:
                drop = abs(f_v - s_v) / denom
            weighted += w * drop
            wsum += abs(w)
        return weighted / wsum if wsum > 1e-12 else 0.0

    def decide(self, features: FrameFeatures) -> str:
        C = self._composite(features)
        requested = self.hyst.step(C)
        self.last_diag = {
            "policy_score": float(C),
            "policy_thresh_low": float(self.config.mp_c_low),
            "policy_thresh_high": float(self.config.mp_c_high),
            "policy_thresh_mid": _NAN,
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.ema.clear()
        self.hyst.reset()
        self.stabiliser.reset()


# ===========================================================================
# Registry + factory
# ===========================================================================
POLICY_REGISTRY: Dict[str, type] = {
    "n_only": NOnly,
    "s_only": SOnly,
    "entropy_only": EntropyOnly,
    "combined": Combined,
    "combined_hyst": CombinedHyst,
    "local_contrast_hyst": LocalContrastHyst,
    "conf_ema": ConfEMA,
    "multi_proxy": MultiProxy,
}

CANONICAL_POLICIES: List[str] = [
    "n_only",
    "s_only",
    "entropy_only",
    "combined",
    "combined_hyst",
    "conf_ema",
    "multi_proxy",
]
EXPERIMENTAL_POLICIES: List[str] = ["local_contrast_hyst"]


def make_policy(name: str, config: Optional[PolicyConfig] = None) -> Policy:
    """Instantiate a policy by name."""
    if name not in POLICY_REGISTRY:
        raise KeyError(
            f"unknown policy: {name!r}. valid: {sorted(POLICY_REGISTRY)}"
        )
    cfg = config or PolicyConfig(name=name)
    return POLICY_REGISTRY[name](cfg)


__all__ = [
    "FrameFeatures",
    "PolicyConfig",
    "Policy",
    "NOnly",
    "SOnly",
    "EntropyOnly",
    "Combined",
    "CombinedHyst",
    "LocalContrastHyst",
    "ConfEMA",
    "MultiProxy",
    "POLICY_REGISTRY",
    "CANONICAL_POLICIES",
    "EXPERIMENTAL_POLICIES",
    "make_policy",
]
