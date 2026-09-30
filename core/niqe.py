"""
Lightweight No-Reference Image Quality Assessment (NR-IQA) for DMS-Raptor.

Implements a simplified BRISQUE-like score using Natural Scene Statistics (NSS):
 - Compute MSCN (Mean Subtracted Contrast Normalised) coefficients
 - Fit a Generalised Gaussian Distribution (GGD) via moment matching
 - Derive a single quality score (higher = worse quality)

This is NOT a standard NIQE implementation. It is a lightweight
NSS-based NR-IQA proxy inspired by BRISQUE. For thesis and publication,
refer to it as:
    "lightweight NR-IQA proxy" or "BRISQUE-like quality score"
Do NOT call it "NIQE" without qualification.

Dependencies: numpy, cv2.  scipy.ndimage is optional (Gaussian filter);
falls back to cv2.GaussianBlur if scipy is absent.

Typical cost: < 3 ms on a 160x160 proxy image.

References:
    Mittal et al., "No-Reference Image Quality Assessment in the Spatial
    Domain", IEEE TIP 2012 (BRISQUE).
"""
from __future__ import annotations

import cv2
import numpy as np

# Try scipy for Gaussian filter; fall back to OpenCV
try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur using scipy or cv2."""
    if _HAS_SCIPY:
        return gaussian_filter(img, sigma=sigma)
    ksize = int(2 * round(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _estimate_ggd_shape(x: np.ndarray) -> float:
    """Estimate GGD shape parameter via moment-ratio method.

    Returns the shape parameter beta (higher = more Gaussian;
    lower = heavier tails / more distorted).
    """
    x = x.ravel().astype(np.float64)
    if len(x) < 10:
        return 2.0  # default: Gaussian
    sigma = float(np.std(x))
    if sigma < 1e-7:
        return 10.0  # nearly constant → perfect quality
    mean_abs = float(np.mean(np.abs(x)))
    if mean_abs < 1e-7:
        return 10.0
    # ratio r = sigma^2 / E[|x|]^2  → used to look up beta
    r = (sigma * sigma) / (mean_abs * mean_abs)
    # Approximate inverse lookup: beta ≈ f(r)
    # For GGD, r = Gamma(3/beta) / (Gamma(1/beta))
    # Empirical fit (valid for beta in [0.2, 10]):
    beta = max(0.2, min(10.0, 1.0 / (r - 0.3189 + 1e-6) * 0.7894))
    return float(np.clip(beta, 0.2, 10.0))


def _compute_mscn(gray: np.ndarray, sigma: float = 7.0 / 6.0) -> np.ndarray:
    """Compute MSCN (Mean Subtracted Contrast Normalised) coefficients."""
    mu = _gaussian_blur(gray, sigma)
    mu_sq = _gaussian_blur(gray * gray, sigma)
    sigma_map = np.sqrt(np.maximum(mu_sq - mu * mu, 0.0)) + 1.0 / 255.0
    mscn = (gray - mu) / sigma_map
    return mscn


def compute_nriqa_score(img_bgr: np.ndarray, proxy_size: int = 160) -> float:
    """Compute a lightweight NR-IQA quality score (BRISQUE-like).

    Higher score = worse quality (more distortion, blur, noise).
    Range: roughly 0 (perfect) to 100+ (severe degradation).

    The score is derived from MSCN coefficient statistics:
      - GGD shape parameter of MSCN distribution
      - Variance of MSCN coefficients
      - Paired-product neighbour statistics (horizontal, vertical)

    NOTE: This is NOT standard NIQE. It is a simplified BRISQUE-like
    NR-IQA proxy. See module docstring for details.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image.
    proxy_size : int
        Resize to this square size for speed. 160 → ~1-2 ms.

    Returns
    -------
    float
        Quality score (higher = worse).
    """
    # Resize to proxy
    if proxy_size > 0:
        img_small = cv2.resize(img_bgr, (proxy_size, proxy_size),
                               interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Compute MSCN at two scales
    scores = []
    for scale in (1, 2):
        if scale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (max(16, w // 2), max(16, h // 2)),
                              interpolation=cv2.INTER_AREA)
        mscn = _compute_mscn(gray)

        # GGD shape parameter (lower = more distortion)
        beta = _estimate_ggd_shape(mscn)

        # Variance of MSCN
        mscn_var = float(np.var(mscn))

        # Paired products (horizontal neighbours)
        h_prod = mscn[:, :-1] * mscn[:, 1:]
        beta_h = _estimate_ggd_shape(h_prod)

        # Paired products (vertical neighbours)
        v_prod = mscn[:-1, :] * mscn[1:, :]
        beta_v = _estimate_ggd_shape(v_prod)

        # Lower beta values indicate departure from natural statistics
        # Convert to a "distortion" score: higher = worse
        scale_score = (
            (10.0 - beta) * 3.0 +        # MSCN shape deviation
            (10.0 - beta_h) * 2.0 +       # horizontal structure loss
            (10.0 - beta_v) * 2.0 +       # vertical structure loss
            max(0.0, 1.0 - mscn_var) * 5  # low variance = flat/blurry
        )
        scores.append(max(0.0, scale_score))

    # Average across scales
    return float(np.mean(scores))


# Backward-compatible alias — legacy callers may use the old name.
# New code should use compute_nriqa_score().
compute_niqe_score = compute_nriqa_score
