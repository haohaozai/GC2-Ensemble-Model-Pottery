"""2-D shape descriptors of one segmented object, from its SAM mask (silhouette) - shape_v1.

Only the silhouette in the photo is described (no 3-D geometry is inferred). Descriptors, all scale-normalised:
  aspect ratio (minimum-area rectangle), extent (area / bounding box), solidity (area / convex hull),
  compactness (4 pi A / P^2), eccentricity (moments), 7 Hu moments (sign * log10|h|), left-right and
  top-bottom mirror symmetry (IoU with the mirrored mask about the centroid), holes (Euler number), convexity
  defects deeper than 5 % of the size, curvature (mean |turning angle| along the contour, share of sharp corners),
  a 32-bin radial profile (centroid distance by angle, max = 1; photos are mostly upright, so it is kept
  rotation-dependent) and 10 elliptic-Fourier harmonics (rotation / scale / start-point normalised).
"""
from __future__ import annotations

import math

import numpy as np

VERSION = "shape_v1"
N_RADIAL = 32
N_EFD = 10
NAMES = (["aspect", "extent", "solidity", "compactness", "eccentricity"] + [f"hu{i}" for i in range(1, 8)] +
         ["sym_lr", "sym_tb", "holes", "defects", "curv_mean", "curv_sharp"] + [f"rad{i}" for i in range(N_RADIAL)] +
         [f"efd{i}_{c}" for i in range(1, N_EFD + 1) for c in "abcd"])


def _efd(contour: np.ndarray, order=N_EFD) -> np.ndarray:
    """Kuhl & Giardina elliptic Fourier descriptors, normalised (first harmonic -> rotation / scale / phase)."""
    d = np.diff(np.vstack([contour, contour[:1]]), axis=0).astype(np.float64)
    dt = np.sqrt((d ** 2).sum(1))
    keep = dt > 0
    d, dt = d[keep], dt[keep]
    if len(dt) < 3:
        return np.zeros(order * 4)
    t = np.concatenate([[0], np.cumsum(dt)])
    T = t[-1]
    phi = 2 * np.pi * t / T
    coeffs = np.zeros((order, 4))
    for n in range(1, order + 1):
        c = T / (2 * n * n * np.pi * np.pi)
        dc, ds = np.cos(n * phi[1:]) - np.cos(n * phi[:-1]), np.sin(n * phi[1:]) - np.sin(n * phi[:-1])
        coeffs[n - 1] = [c * (d[:, 0] / dt * dc).sum(), c * (d[:, 0] / dt * ds).sum(),
                         c * (d[:, 1] / dt * dc).sum(), c * (d[:, 1] / dt * ds).sum()]
    a1, b1, c1, d1 = coeffs[0]
    th = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1 ** 2 - b1 ** 2 + c1 ** 2 - d1 ** 2)
    out = coeffs.copy()
    for n in range(1, order + 1):
        r = np.array([[np.cos(n * th), -np.sin(n * th)], [np.sin(n * th), np.cos(n * th)]])
        out[n - 1] = (np.array([[coeffs[n - 1, 0], coeffs[n - 1, 1]], [coeffs[n - 1, 2], coeffs[n - 1, 3]]]) @ r).ravel()
    a1, c1 = out[0, 0], out[0, 2]
    psi = np.arctan2(c1, a1)
    rot = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
    for n in range(order):
        out[n] = (rot @ out[n].reshape(2, 2)).ravel()
    scale = abs(out[0, 0]) or 1.0
    return (out / scale).ravel()


def describe(mask: np.ndarray) -> np.ndarray:
    """bool mask (H, W) -> feature vector (len(NAMES))."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return np.zeros(len(NAMES), np.float32)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = 256 / max(m.shape)
    if s < 1:
        m = cv2.resize(m, (max(1, round(m.shape[1] * s)), max(1, round(m.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
    m = np.pad(m, 2)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outer = max([c for i, c in enumerate(cs) if hier[0][i][3] < 0], key=cv2.contourArea)
    A = float(m.sum())
    P = cv2.arcLength(outer, True) or 1.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(outer)
    aspect = min(rw, rh) / max(rw, rh, 1e-6)
    x, y, w, h = cv2.boundingRect(outer)
    extent = A / max(1.0, w * h)
    hull = cv2.convexHull(outer)
    solidity = A / max(1.0, cv2.contourArea(hull))
    compact = 4 * math.pi * A / (P * P)
    mo = cv2.moments(m, binaryImage=True)
    mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
    lam = np.sort(np.linalg.eigvalsh(np.array([[mu20, mu11], [mu11, mu02]])))
    ecc = math.sqrt(max(0.0, 1 - lam[0] / max(lam[1], 1e-9)))
    hu = cv2.HuMoments(mo).ravel()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    hu = np.clip(hu, -30, 30) / 10
    cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    H, W = m.shape
    M_lr = cv2.warpAffine(m, np.float32([[-1, 0, 2 * cx], [0, 1, 0]]), (W, H), flags=cv2.INTER_NEAREST)
    M_tb = cv2.warpAffine(m, np.float32([[1, 0, 0], [0, -1, 2 * cy]]), (W, H), flags=cv2.INTER_NEAREST)
    sym = lambda o: float((m & o).sum() / max(1, (m | o).sum()))  # noqa: E731
    holes = sum(1 for i in range(len(cs)) if hier[0][i][3] >= 0 and cv2.contourArea(cs[i]) > 0.002 * A)
    size = max(w, h)
    defects = 0
    try:
        hi = cv2.convexHull(outer, returnPoints=False)
        dd = cv2.convexityDefects(outer, hi)
        if dd is not None:
            defects = int((np.asarray(dd).reshape(-1, 4)[:, 3] / 256.0 > 0.05 * size).sum())   # (N,1,4) or (N,4)
    except cv2.error:
        pass
    pts = outer[:, 0, :].astype(np.float64)
    step = max(1, len(pts) // 128)
    q = pts[::step]
    v1, v2 = np.roll(q, -1, 0) - q, q - np.roll(q, 1, 0)
    ang = np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0], (v1 * v2).sum(1)))
    curv_mean, curv_sharp = float(ang.mean()), float((ang > 0.6).mean())
    rel = pts - [cx, cy]
    th = np.arctan2(rel[:, 1], rel[:, 0])
    r = np.hypot(rel[:, 0], rel[:, 1])
    bins = ((th + np.pi) / (2 * np.pi) * N_RADIAL).astype(int) % N_RADIAL
    rad = np.zeros(N_RADIAL)
    np.maximum.at(rad, bins, r)
    for i in range(N_RADIAL):        # empty bins: neighbour value
        if rad[i] == 0:
            rad[i] = rad[i - 1]
    rad = rad / max(rad.max(), 1e-9)
    efd = _efd(pts)
    v = np.concatenate([[aspect, extent, solidity, compact, ecc], hu, [sym(M_lr), sym(M_tb), min(holes, 5) / 5,
                        min(defects, 10) / 10, curv_mean, curv_sharp], rad, efd])
    return np.nan_to_num(v).astype(np.float32)
