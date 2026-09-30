"""
Per-frame feature extraction for RL switching agents.

Wraps the existing DMS-Raptor proxy functions (Laplacian, entropy,
Tenengrad, etc.) into a fixed-dimension feature vector that agents
consume as their context / state representation.
"""
from __future__ import annotations

import numpy as np
import cv2

from .config import FEATURE_NAMES, FEATURE_DIM


def compute_all_features(
    img_bgr: np.ndarray,
    proxy_size: int = 160,
    hist_bins: int = 64,
    prev_conf: float = 0.0,
    prev_n_dets: int = 0,
    prev_action: int = 0,
    dwell_time: int = 0,
) -> np.ndarray:
    """Extract the full feature vector from a single frame.

    Parameters
    ----------
    img_bgr : np.ndarray
        Raw BGR frame from the video.
    proxy_size : int
        Resize target for proxy computation (square).
    hist_bins : int
        Number of bins for entropy histogram.
    prev_conf : float
        Mean detection confidence of the previous frame.
    prev_n_dets : int
        Number of detections in the previous frame.
    prev_action : int
        Action taken on the previous frame (0=n, 1=s).
    dwell_time : int
        Frames since the last model switch.

    Returns
    -------
    np.ndarray
        Feature vector of shape (FEATURE_DIM,), dtype float32.
        Order matches FEATURE_NAMES exactly.
    """
    # ── Resize once, share across all proxies ────────────────────────
    small = cv2.resize(img_bgr, (proxy_size, proxy_size),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # ── Laplacian variance (sharpness) ───────────────────────────────
    laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # ── Shannon entropy ──────────────────────────────────────────────
    bins = int(max(16, hist_bins))
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = hist.astype(np.float32).ravel()
    total = hist.sum()
    if total > 0:
        p = hist[hist > 1e-12] / total
        entropy = float(-(p * np.log2(p)).sum())
    else:
        entropy = 0.0

    # ── Tenengrad (Sobel gradient energy) ────────────────────────────
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float((gx ** 2 + gy ** 2).mean())

    # ── Color entropy (HSV joint histogram) ──────────────────────────
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    chist = cv2.calcHist([hsv], [0, 1], None, [32, 32],
                         [0, 180, 0, 256]).astype(np.float32).ravel()
    cs = chist.sum()
    if cs > 0:
        cp = chist[chist > 1e-12] / cs
        color_entropy = float(-(cp * np.log2(cp)).sum())
    else:
        color_entropy = 0.0

    # ── NR-IQA score ─────────────────────────────────────────────────
    # Lightweight: MSCN-based sharpness (avoids importing niqe.py for
    # speed; the full NR-IQA is expensive).  We use a fast surrogate:
    # std of locally-normalised coefficients (MSCN).
    gray_f = gray.astype(np.float32)
    mu = cv2.GaussianBlur(gray_f, (7, 7), 1.166)
    sq = cv2.GaussianBlur(gray_f * gray_f, (7, 7), 1.166)
    sigma = np.sqrt(np.maximum(sq - mu * mu, 0.0)) + 1e-7
    mscn = (gray_f - mu) / sigma
    nriqa = float(mscn.std())

    # ── Brightness (mean pixel intensity) ────────────────────────────
    brightness = float(gray.mean()) / 255.0

    # ── Edge density (Canny) ─────────────────────────────────────────
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.astype(np.float32).mean() / 255.0)

    # ── Local contrast (pixel std dev) ───────────────────────────────
    local_contrast = float(gray.astype(np.float32).std())

    # ── Assemble feature vector (order must match FEATURE_NAMES) ─────
    features = np.array([
        laplacian,
        entropy,
        tenengrad,
        color_entropy,
        nriqa,
        brightness,
        edge_density,
        local_contrast,
        prev_conf,
        float(prev_n_dets),
        float(prev_action),
        float(dwell_time),
    ], dtype=np.float32)

    assert features.shape == (FEATURE_DIM,), (
        f"Feature dim mismatch: got {features.shape}, expected ({FEATURE_DIM},)"
    )
    return features


class FeatureNormalizer:
    """Online z-score normalizer with running mean/variance.

    Can be fit on a dataset (batch mode) or updated incrementally
    (streaming mode).  Serializable to/from numpy files.
    """

    def __init__(self, dim: int = FEATURE_DIM):
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 0
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        """Fit on a batch of feature vectors, shape (N, dim)."""
        assert X.ndim == 2 and X.shape[1] == self.dim
        self.mean = X.mean(axis=0).astype(np.float64)
        self.var = X.var(axis=0).astype(np.float64) + 1e-8
        self.count = X.shape[0]
        self._fitted = True
        return self

    def transform(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        """Normalize a single feature vector or batch."""
        if not self._fitted:
            return x.astype(np.float32)
        z = (x.astype(np.float64) - self.mean) / np.sqrt(self.var)
        if clip > 0:
            z = np.clip(z, -clip, clip)
        return z.astype(np.float32)

    def update_online(self, x: np.ndarray) -> None:
        """Welford's online update for a single observation."""
        self.count += 1
        delta = x.astype(np.float64) - self.mean
        self.mean += delta / self.count
        delta2 = x.astype(np.float64) - self.mean
        self.var = ((self.count - 1) * self.var + delta * delta2) / self.count
        self._fitted = True

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load(self, path: str) -> "FeatureNormalizer":
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.count = int(data["count"])
        self._fitted = True
        return self
