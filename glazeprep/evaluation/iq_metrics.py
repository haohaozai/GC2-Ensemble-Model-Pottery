"""No-reference image-quality metrics computed INSIDE the glaze mask only.

There is no ground-truth "perfectly lit" photo, so each metric measures one
symptom that a correction step is supposed to fix (or must not cause):

| metric                 | symptom                                   | want   | targeted by |
|------------------------|-------------------------------------------|--------|-------------|
| highlight_frac         | bright, colourless specular blobs         | lower  | UnReflect   |
| clipped_frac           | over-exposed pixels (any channel >= 250)  | lower  | UnReflect   |
| illum_nonuniformity    | large-scale brightness gradient / shading | lower  | Retinex, PhaSR |
| dark_frac              | crushed shadows (L* < 15)                 | lower  | Retinex, PhaSR |
| texture_ratio          | local contrast vs 000 (Laplacian variance)| ~1.0   | side effect: <1 smoothed (texture lost), >1 amplified (noise/halo) |
| deltaE00_vs_000        | colour shift vs the segmented-only image  | small  | side effect |
| colourfulness          | Hasler-Suesstrunk colourfulness           | info   |             |
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


def _small(rgb, mask, side=384):
    s = min(1.0, side / max(rgb.shape[:2]))
    if s < 1.0:
        size = (round(rgb.shape[1] * s), round(rgb.shape[0] * s))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return rgb, mask


def _texture(gray, mask):
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # ignore the cut-out edge
    return float(lap[inner].var()) if inner.sum() > 50 else float("nan")


def image_metrics(rgb: np.ndarray, mask: np.ndarray, ref_rgb: np.ndarray | None = None) -> dict:
    rgb, mask = _small(rgb, mask)
    if mask.sum() < 50:
        return {}
    lab = rgb2lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    Lm = L[mask]
    med = np.median(Lm)
    out = {
        "highlight_frac": float(((L > med + 25) & (chroma < 12))[mask].mean()),
        "clipped_frac": float((rgb.max(-1) >= 250)[mask].mean()),
        "dark_frac": float((Lm < 15).mean()),
        "mean_L": float(Lm.mean()),
        "mean_chroma": float(chroma[mask].mean()),
    }
    # illumination field = mask-normalised low-pass of L*  (normalised convolution: background ignored)
    sig = 0.08 * max(L.shape)
    mf = mask.astype(np.float32)
    num = cv2.GaussianBlur(L.astype(np.float32) * mf, (0, 0), sig)
    den = cv2.GaussianBlur(mf, (0, 0), sig) + 1e-6
    out["illum_nonuniformity"] = float((num / den)[mask].std())

    r, g, bb = [rgb[..., i][mask].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - bb
    out["colourfulness"] = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    out["_texture"] = _texture(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mask)

    if ref_rgb is not None:
        ref, _ = _small(ref_rgb, mask)
        ref_lab = rgb2lab(ref)
        out["deltaE00_vs_000"] = float(np.median(deltaE_ciede2000(ref_lab[mask], lab[mask])))
    return out
