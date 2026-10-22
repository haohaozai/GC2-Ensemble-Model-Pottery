"""Balanced object-level retrieval (attributes_v1): every segmented object is indexed on its own.

Similarity components (each is a score of the query against every indexed object):
  appearance  cosine of the trained multimodal image head applied to the object's own DINOv2 vector
              (the model of results/final_model.pt; glaze look - colour, texture, gloss - learned with text + formula)
  shape       0.5 x cosine of standardised silhouette geometry (mm/shape.py) + 0.5 x cosine of the DINOv2 silhouette
              embedding (black mask on white: no colour / glaze information)
  surface     Hellinger affinity of predicted class probabilities (surface finish, transparency, glaze effects,
              defects; reviewed labels replace predictions)
  colour      region-aware colour-set similarity (below)
  text / formula (optional, when the query has them): the multimodal text / formula embeddings of the recipe
Fusion: every component is z-scored over the gallery for this query, then combined with the weights of the
search mode (tuned on validation queries - see run_attributes.py stage `tune`); absent components get weight 0.

Region-aware colour similarity (query regions i with weight a_i, object regions j with weight b_j):
  cost(i, j) = min(1, dE2000(i, j) / 40) + LAMBDA_POS * |centroid_i - centroid_j|
  d(q->g) = sum_i a_i min_j cost(i, j);  d(g->q) = sum_j b_j min_i cost(i, j);  s = 1 - (d(q->g) + d(g->q)) / 2
Region weights: human importance from review_tile (important 1 / normal 0.5 / ignore 0) when reviewed, otherwise
the measured auto_importance (distinctiveness + area + contrast; NOT area alone). Variants for the ablation:
  dominant  - only the largest region of each side (= a dominant-colour search)
  palette   - all regions, weights = area, no position (= a multi-colour palette search)
  regions   - all regions, weights = auto importance, with position
  reviewed  - as regions, with human importance where reviewed (query side: the user's importance)
"""
from __future__ import annotations

import json

import numpy as np

K_REG = 8
LAMBDA_POS = 0.5
DE_SCALE = 40.0
COMPONENTS = ("appearance", "shape", "surface", "colour")
IMP_W = {"high": 1.0, "normal": 0.5, "ignore": 0.0}
ATTR_GROUPS = (("surface_finish", 9), ("transparency", 4), ("glaze_effect", 8), ("defect", 5))


def de2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 for broadcastable (..., 3) arrays (Sharma et al. 2005)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs, np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp)) + 0.32 * np.cos(np.radians(3 * hbp + 6))
         - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def region_arrays(reg: dict | None, review: dict | None = None):
    """regions json (+ tile review) -> lab [K,3], pos [K,2], area [K], auto [K], human [K] (nan = not reviewed), valid [K]"""
    lab, pos = np.zeros((K_REG, 3), np.float32), np.zeros((K_REG, 2), np.float32)
    area, auto, human, valid = (np.zeros(K_REG, np.float32), np.zeros(K_REG, np.float32),
                                np.full(K_REG, np.nan, np.float32), np.zeros(K_REG, bool))
    if not reg or not reg.get("regions"):
        return lab, pos, area, auto, human, valid
    rr = (review or {}).get("regions", {})
    regs = sorted(reg["regions"], key=lambda r: -(r["auto_importance"] + r["area_frac"]))[:K_REG]
    for k, r in enumerate(regs):
        lab[k], pos[k], area[k], auto[k], valid[k] = r["lab"], r["centroid"], r["area_frac"], r["auto_importance"], True
        h = rr.get(str(r["id"]))
        if h and h.get("importance"):
            human[k] = IMP_W[h["importance"]]
    return lab, pos, area, auto, human, valid


def _weights(area, auto, human, valid, variant):
    if variant == "palette":
        w = area.copy()
    elif variant == "reviewed":
        w = np.where(np.isnan(human), auto, human)
    else:
        w = auto.copy()
    w = np.where(valid, w, 0)
    if variant == "dominant":
        dom = np.where(valid, area, -1).argmax(-1)
        w = np.zeros_like(w)
        np.put_along_axis(w, dom[..., None], 1.0, axis=-1)
    s = w.sum(-1, keepdims=True)
    return np.where(s > 0, w / np.maximum(s, 1e-9), 0)


def colour_similarity(q, G, variant="reviewed") -> np.ndarray:
    """q: tuple of region arrays of the query; G: dict of stacked gallery arrays -> similarity [N]."""
    qlab, qpos, qarea, qauto, qhuman, qvalid = q[:6]
    usepos = q[6] if len(q) > 6 else np.ones(len(qlab), np.float32)     # 0 for query colours given "anywhere"
    wq = _weights(qarea, qauto, qhuman, qvalid, variant)
    wg = _weights(G["area"], G["auto"], G["human"], G["valid"], variant)
    if not qvalid.any():
        return np.zeros(len(wg), np.float32)
    lam = 0.0 if variant in ("dominant", "palette") else LAMBDA_POS
    d = de2000(qlab[:, None, None, :], G["lab"][None, :, :, :])                    # [Kq, N, K]
    c = np.minimum(1.0, d / DE_SCALE) + lam * np.asarray(usepos)[:, None, None] * np.linalg.norm(qpos[:, None, None, :] - G["pos"][None], axis=-1)
    c = np.where(qvalid[:, None, None] & G["valid"][None], c, 9.0)
    dq = (wq[:, None] * c.min(2)).sum(0)                                            # query -> gallery
    dg = (wg * c.min(0)).sum(-1)                                                    # gallery -> query
    s = 1 - 0.5 * (np.minimum(dq, 2) + np.minimum(dg, 2))
    return np.where(G["valid"].any(-1), s, -1).astype(np.float32)


def colour_query(colours: list[dict]):
    """colours typed / picked by the user: [{lab, centroid, use_position, importance}] -> query region arrays."""
    lab, pos = np.zeros((K_REG, 3), np.float32), np.full((K_REG, 2), 0.5, np.float32)
    area, auto, human, valid = (np.zeros(K_REG, np.float32), np.zeros(K_REG, np.float32),
                                np.full(K_REG, np.nan, np.float32), np.zeros(K_REG, bool))
    usepos = np.zeros(K_REG, np.float32)
    for k, c in enumerate(colours[:K_REG]):
        lab[k], valid[k] = c["lab"], True
        pos[k] = c.get("centroid", [0.5, 0.5])
        usepos[k] = 1.0 if c.get("use_position") else 0.0
        human[k] = IMP_W.get(c.get("importance", "normal"), 0.5)
        auto[k] = human[k]
        area[k] = 0.6 if k == 0 else 0.4 / max(1, len(colours) - 1)
    return lab, pos, area, auto, human, valid, usepos


def hellinger(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(P, 0, None)) @ np.sqrt(np.clip(q, 0, None))


def attr_vector(pred: dict | None, trusted: bool = True) -> np.ndarray:
    """per-group probability vectors (normalised per group) from attribute_predictions.json.

    trusted=True (database side): reviewed labels and Glazy metadata replace the predictions.
    trusted=False (query side): model predictions only - a new photo has no metadata, and using the query's
    metadata would leak the relevance labels into the ranking during evaluation."""
    from .attr_labels import DEFECTS, EFFECTS, SURFACE, TRANSP
    order = {"surface_finish": SURFACE, "transparency": TRANSP, "glaze_effect": EFFECTS, "defect": DEFECTS}
    out = []
    for g, n in ATTR_GROUPS:
        v = np.zeros(n, np.float32)
        p = (pred or {}).get(g) or []
        fin = (pred or {}).get(g + "_final") or {}
        if trusted and fin.get("provenance", "").startswith(("human", "glazy_meta")):
            vals = fin["value"] if isinstance(fin["value"], list) else [fin["value"]]
            for x in vals:
                if x in order[g]:
                    v[order[g].index(x)] = 1.0
        else:
            for lab, pr in p:
                if lab in order[g]:
                    v[order[g].index(lab)] = pr
        s = v.sum()
        out.append(v / s if s > 0 else v)
    return np.concatenate(out)


def zscore(s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask & np.isfinite(s)
    if m.sum() < 2:
        return np.zeros_like(s)
    mu, sd = s[m].mean(), s[m].std() + 1e-9
    return np.where(m, (s - mu) / sd, -3.0)


def fuse(scores: dict, weights: dict, mask: np.ndarray) -> np.ndarray:
    tot = np.zeros(len(mask), np.float32)
    wsum = sum(weights.get(c, 0) for c in scores)
    if wsum <= 0:
        return tot
    for c, s in scores.items():
        w = weights.get(c, 0) / wsum
        if w:
            tot += w * zscore(s, mask)
    return tot


def explain(scores: dict, weights: dict, j: int) -> dict:
    return {c: round(float(s[j]), 3) for c, s in scores.items() if weights.get(c, 0) > 0}


def load_json_lines(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
