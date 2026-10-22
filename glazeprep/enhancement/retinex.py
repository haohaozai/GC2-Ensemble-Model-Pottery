"""Illumination correction - Multi-Scale Retinex with Chromaticity Preservation (MSRCP).

Petro, Sbert & Morel, "Multiscale Retinex", IPOL 2014.

Why MSRCP and not classic MSRCR?  MSRCR runs Retinex on each RGB channel
independently, which shifts hue (a cobalt blue can drift toward grey/cyan).
MSRCP runs Retinex only on the intensity channel and rescales R,G,B by the
same factor, so the glaze *chromaticity* (the colour we want to retrieve) is
preserved while uneven lighting / shading is flattened.

Mask-aware: the colour-balance stretch percentiles are computed on the glaze
pixels only, so a bright white backdrop does not dictate the exposure.
"""
from __future__ import annotations

import cv2
import numpy as np


def _simplest_color_balance(x: np.ndarray, ref: np.ndarray, low: float, high: float) -> np.ndarray:
    lo, hi = np.percentile(ref, [low, 100 - high])
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1) * 255.0


def msrcp(rgb: np.ndarray, mask: np.ndarray | None = None,
          sigmas_frac=(0.015, 0.08, 0.25), clip_low: float = 1.0, clip_high: float = 1.0) -> np.ndarray:
    """rgb uint8 HxWx3 -> uint8 HxWx3.

    sigmas_frac: Gaussian surround scales as a fraction of the longest image side
                 (0.015/0.08/0.25 of 1000 px = the classic 15/80/250).
    """
    img = rgb.astype(np.float64)
    intensity = img.mean(axis=2) + 1.0
    side = max(rgb.shape[:2])
    log_i = np.log(intensity)
    msr = np.zeros_like(intensity)
    for f in sigmas_frac:
        s = max(1.0, f * side)
        blur = cv2.GaussianBlur(intensity, (0, 0), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT)
        msr += log_i - np.log(blur)
    msr /= len(sigmas_frac)

    ref = msr[mask] if mask is not None and mask.sum() > 100 else msr
    i1 = _simplest_color_balance(msr, ref, clip_low, clip_high)

    # chromaticity-preserving amplification: same gain for R, G, B (capped to avoid clipping)
    b = np.maximum(img.max(axis=2), 1.0)
    gain = np.minimum(255.0 / b, i1 / intensity)
    out = img * gain[..., None]
    return np.clip(out, 0, 255).round().astype(np.uint8)
