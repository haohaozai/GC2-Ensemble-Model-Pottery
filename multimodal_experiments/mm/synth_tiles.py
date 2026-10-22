"""Synthetic test tiles with KNOWN colour regions, built from real glaze textures of the dataset (attributes_v1).

Textures: square patches cut from the interior of real single-region objects (no background). Each synthetic tile
(240 x 320 px, rounded corners) is a composite of such textures, so the ground-truth region masks and colours are
exact. Cases (the spec's list):
  A  large neutral region + small distinctive band at the bottom (10-25 %)
  B  two-colour tile (top / bottom halves)
  C  three-colour tile (three bands)
  D  same dominant colour, different secondary colour (pairs built for the retrieval test)
  E  same colours, speckles added on one of them (different glaze effect, same colour)
  F  exposed ceramic body: a buff / terracotta clay-coloured band at the foot
  G  shadow (darkening gradient on one side) and specular highlight streak added to A-tiles
Not synthesised: different surface finishes (gloss cannot be faked by recolouring) - covered by the real-data test.
These are controlled composites, not photographs; they test the colour-region logic, not photo realism.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TW, TH = 240, 320


def collect_textures(regions_dir, obj_file, ids, rng, n=400, size=96):
    """square texture patches from objects with ONE meaningful region; returns [{lab, hex, patch, src}]"""
    from skimage.color import rgb2lab
    out = []
    for oid in rng.permutation(ids):
        if len(out) >= n:
            break
        p = regions_dir / f"{oid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("summary", {}).get("n_meaningful") != 1 or d.get("spots", {}).get("density", 1) > 0.01:
            continue
        a = np.asarray(Image.open(obj_file("smaller_object_rgba_raw", f"{oid}.png")).convert("RGBA"))
        s = 480 / max(a.shape[:2])
        if s < 1:
            a = np.asarray(Image.fromarray(a).resize((max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s))), Image.LANCZOS))
        m = a[..., 3] >= 250
        H, W = m.shape
        if H < size + 4 or W < size + 4:
            continue
        for _ in range(40):
            y, x = int(rng.integers(0, H - size)), int(rng.integers(0, W - size))
            if m[y:y + size, x:x + size].all():
                patch = a[y:y + size, x:x + size, :3]
                lab = np.median(rgb2lab(patch.reshape(-1, 1, 3) / 255.0).reshape(-1, 3), 0)
                if lab[0] > 92 or lab[0] < 12:          # avoid blown-out / black patches
                    break
                out.append({"lab": lab, "patch": patch, "src": oid})
                break
    return out


def _fill(patch, h, w):
    reps = (h // patch.shape[0] + 2, w // patch.shape[1] + 2, 1)
    big = np.tile(np.concatenate([patch, patch[:, ::-1]], 1), reps)
    return big[:h, :w]


def compose(bands: list, shadow=False, highlight=False, speckle_band=None, rng=None):
    """bands: [(texture, frac)] top -> bottom. Returns rgba (TH, TW, 4) and the label map (1..k by band, 0 outside)."""
    rng = rng or np.random.default_rng(0)
    img = np.zeros((TH, TW, 3), np.uint8)
    lbl = np.zeros((TH, TW), np.uint8)
    y = 0
    for k, (tex, frac) in enumerate(bands, 1):
        h = TH - y if k == len(bands) else int(round(frac * TH))
        img[y:y + h] = _fill(tex["patch"], h, TW)
        lbl[y:y + h] = k
        y += h
    if speckle_band is not None:
        ys, xs = np.nonzero(lbl == speckle_band)
        pick = rng.choice(len(ys), size=max(20, len(ys) // 180), replace=False)
        for i in pick:
            r = int(rng.integers(1, 3))
            img[max(0, ys[i] - r):ys[i] + r, max(0, xs[i] - r):xs[i] + r] = (35, 30, 28)
    if shadow:
        g = np.ones(TW)
        g[:TW // 4] = np.linspace(0.5, 1.0, TW // 4)
        img = (img * g[None, :, None]).astype(np.uint8)
    if highlight:
        hl = Image.new("L", (TW, TH), 0)
        ImageDraw.Draw(hl).ellipse([TW * 0.55, TH * 0.08, TW * 0.9, TH * 0.22], fill=210)
        hl = np.asarray(hl.filter(ImageFilter.GaussianBlur(8)), np.float32)[..., None] / 255.0
        img = (img * (1 - hl) + 250 * hl).astype(np.uint8)
    alpha = Image.new("L", (TW, TH), 0)
    ImageDraw.Draw(alpha).rounded_rectangle([0, 0, TW - 1, TH - 1], radius=18, fill=255)
    a = np.asarray(alpha)
    lbl[a < 125] = 0
    return np.dstack([img, a]), lbl


def region_match(res, lbl_pred, lbl_true) -> dict:
    """match predicted MEANINGFUL regions to ground-truth regions (Hungarian on IoU, at the analysis size)."""
    from scipy.optimize import linear_sum_assignment
    from PIL import Image as _I
    lt = np.asarray(_I.fromarray(lbl_true).resize(lbl_pred.shape[::-1], _I.NEAREST))
    gts = [k for k in np.unique(lt) if k]
    preds = [r["id"] for r in res["regions"] if r["meaningful"]]
    if not preds:
        return {"gt": len(gts), "pred": 0, "matched": 0, "mean_iou": 0.0, "ious": []}
    iou = np.zeros((len(gts), len(preds)))
    for i, g in enumerate(gts):
        G = lt == g
        for j, p in enumerate(preds):
            P = lbl_pred == p
            iou[i, j] = (G & P).sum() / max(1, (G | P).sum())
    r, c = linear_sum_assignment(-iou)
    ious = [float(iou[i, j]) for i, j in zip(r, c)]
    best = [float(iou[i].max()) for i in range(len(gts))]
    return {"gt": len(gts), "pred": len(preds), "matched": int(sum(1 for v in ious if v >= 0.5)),
            "mean_iou": float(np.mean(best)), "ious": best}
