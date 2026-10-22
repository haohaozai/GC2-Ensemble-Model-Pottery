"""Object index, weight tuning (validation) and evaluation (test) of Balanced Search and its modes.

Relevance (independent of the features used for ranking, as far as the data allows):
  surface       same surface group (glossy / satin / matte)          Glazy metadata
  transparency  same transparency group (clear / opaque)             Glazy metadata
  effects       shares >= 1 glaze effect named in the recipe text     (only when the query recipe names one)
  cone          firing ranges overlap                                Glazy metadata
  colour        ColorThief palette match: the query's 1st colour (and its 2nd if >= 15 %) within dE2000 <= 20 of a
                colour of the object's ColorThief top 3 (median-cut palette - a different algorithm from the regions)
  form          same object form (zero-shot CLIP, confidence >= 0.5; not used by the shape features)
gain = share of the comparable criteria that match; a result is RELEVANT when gain >= 0.75 with >= 2 comparable
criteria (balanced / glaze), or when the single criterion matches (colour / shape modes).
Mode objectives on validation: balanced - all criteria; glaze/surface - surface + transparency + effects;
colour - colour; shape - form. Weights: simplex grid (step 0.05) over the four image components, each >= 0.05.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from . import balanced as B
from .attr_labels import EFFECTS, SURFACE_GROUP, TRANSP_GROUP

CRITERIA = ("surface", "transparency", "effects", "cone", "colour", "form")
MODE_CRITERIA = {"balanced": CRITERIA, "glaze": ("surface", "transparency", "effects"), "colour": ("colour",), "shape": ("form",)}
KS = (1, 5, 10, 20)


# ---------------------------------------------------------------- index
def build(work, objs, labels_rows, recipes, eng, preds, regions_dir, review, ct_rows) -> dict:
    """arrays for every object (order = objs)."""
    from .query import _std
    from .image_feats import load_cache
    N = len(objs)
    ids = [o["object_id"] for o in objs]
    lab = {r["object_id"]: r for r in labels_rows}
    rec = {r["recipe_id"]: r for r in recipes}
    # appearance: the multimodal image head on each object's own DINOv2 vector
    cache = load_cache("dinov2")
    raw = np.zeros((N, 1024), np.float32)
    m_app = np.zeros(N, bool)
    for k, i in enumerate(ids):
        v = cache.get(i)
        if v is not None:
            raw[k], m_app[k] = v / max(1e-8, np.linalg.norm(v)), True
    app = np.zeros((N, eng.model.dim), np.float32)
    app[m_app] = eng._head("image", _std(raw[m_app], eng.img_mu, eng.img_sd))
    # recipe-level text / formula embeddings of the multimodal model
    rpos = [eng.bank.pos.get(o["recipe_id"], -1) for o in objs]
    def recz(m):
        Z, M = eng.Z[m]
        out, mk = np.zeros((N, Z.shape[1]), np.float32), np.zeros(N, bool)
        for k, p in enumerate(rpos):
            if p >= 0 and M[p]:
                out[k], mk[k] = Z[p], True
        return out, mk
    txt, m_txt = recz("text")
    frm, m_frm = recz("formula")
    # shape
    z = np.load(work / "shape_geom.npz")
    gpos = {i: k for k, i in enumerate(z["ids"].tolist())}
    ZX = z["X"]
    geom = np.stack([ZX[gpos[i]] for i in ids])
    split = np.array([lab[i]["split"] for i in ids])
    tr = split == "train"
    mu, sd = geom[tr].mean(0), geom[tr].std(0) + 1e-6
    geom = np.clip((geom - mu) / sd, -5, 5)
    geom /= np.linalg.norm(geom, axis=1, keepdims=True).clip(1e-6)
    zs = np.load(work / "feat_dinov2_sil.npz")
    spos = {i: k for k, i in enumerate(zs["ids"].tolist())}
    SX = zs["X"]
    sil = np.stack([SX[spos[i]] for i in ids])
    # attributes (predictions; reviewed labels and metadata replace predictions inside attr_vector)
    attr = np.stack([B.attr_vector(preds.get(i)) for i in ids])                       # database side
    attr_pred = np.stack([B.attr_vector(preds.get(i), trusted=False) for i in ids])   # query side (no metadata)
    # colour regions
    R = {k: [] for k in ("lab", "pos", "area", "auto", "human", "valid")}
    for i in ids:
        p = regions_dir / f"{i}.json"
        reg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        arrs = B.region_arrays(reg, review.get(i))
        for k, a in zip(("lab", "pos", "area", "auto", "human", "valid"), arrs):
            R[k].append(a)
    R = {k: np.stack(v) for k, v in R.items()}
    # relevance labels
    sg = {"glossy": 0, "satin": 1, "matte": 2}
    tg = {"clear": 0, "opaque": 1}
    surf = np.array([sg.get(lab[i]["surface_group_meta"], -1) for i in ids])
    transp = np.array([tg.get(lab[i]["transparency_group_meta"], -1) for i in ids])
    eff = np.zeros((N, len(EFFECTS)), bool)
    for k, i in enumerate(ids):
        for e in json.loads(lab[i]["effects_text"] or "[]"):
            eff[k, EFFECTS.index(e)] = True
    eff_has = np.array([lab[i]["has_description"] == "1" for i in ids])
    cone = np.array([[int(rec.get(o["recipe_id"], {}).get("cone_lo_i", -1) or -1), int(rec.get(o["recipe_id"], {}).get("cone_hi_i", -1) or -1)] for o in objs])
    forms = sorted({f for p in preds.values() for f, _ in (p.get("form") or [])})
    form = np.full(N, -1)
    for k, i in enumerate(ids):
        top = (preds.get(i) or {}).get("form") or []
        if top and top[0][1] >= 0.5:
            form[k] = forms.index(top[0][0])
    from .colorlab import hex2lab
    ct = {r["object_id"]: r for r in ct_rows}
    ct_lab, ct_pct = np.zeros((N, 3, 3), np.float32), np.zeros((N, 3), np.float32)
    for k, i in enumerate(ids):
        r = ct.get(i, {})
        for j in range(3):
            h = r.get(f"color{j + 1}_hex", "")
            if h and h.startswith("#") and len(h) == 7:
                ct_lab[k, j] = hex2lab(h)
                ct_pct[k, j] = float(r.get(f"color{j + 1}_pct") or 0) / 100
    return {"ids": np.array(ids), "recipe_id": np.array([o["recipe_id"] for o in objs]), "split": split,
            "app": app, "m_app": m_app, "text": txt, "m_text": m_txt, "formula": frm, "m_formula": m_frm,
            "geom": geom.astype(np.float32), "sil": sil, "attr": attr.astype(np.float32), "attr_pred": attr_pred.astype(np.float32),
            **{"reg_" + k: v for k, v in R.items()},
            "surf": surf, "transp": transp, "eff": eff, "eff_has": eff_has, "cone": cone, "form": form,
            "forms": np.array(forms), "ct_lab": ct_lab, "ct_pct": ct_pct}


# ---------------------------------------------------------------- component scores of one query
def component_scores(I, q: dict, gal: np.ndarray, variant="reviewed") -> dict:
    """q: query features (same keys as one index row); gal: indices into the index. -> {component: [len(gal)]}"""
    S = {}
    if q.get("app") is not None:
        S["appearance"] = I["app"][gal] @ q["app"]
    if q.get("geom") is not None:
        S["shape"] = 0.5 * (I["geom"][gal] @ q["geom"]) + 0.5 * (I["sil"][gal] @ q["sil"])
    if q.get("attr") is not None:
        S["surface"] = B.hellinger(I["attr"][gal], q["attr"]) / 4.0
    if q.get("reg") is not None:
        G = {k: I["reg_" + k][gal] for k in ("lab", "pos", "area", "auto", "human", "valid")}
        S["colour"] = B.colour_similarity(q["reg"], G, variant)
    if q.get("text") is not None:
        S["text"] = np.where(I["m_text"][gal], I["text"][gal] @ q["text"], np.nan)
    if q.get("formula") is not None:
        S["formula"] = np.where(I["m_formula"][gal], I["formula"][gal] @ q["formula"], np.nan)
    return S


def query_from_index(I, k: int) -> dict:
    """an indexed object used as a QUERY: model predictions only for its attributes (no metadata / human labels)."""
    return {"app": I["app"][k] if I["m_app"][k] else None, "geom": I["geom"][k], "sil": I["sil"][k],
            "attr": I["attr_pred"][k] if "attr_pred" in I else I["attr"][k],
            "reg": tuple(I["reg_" + x][k] for x in ("lab", "pos", "area", "auto", "human", "valid"))}


# ---------------------------------------------------------------- relevance
def criteria_matrix(I, qi: np.ndarray, gal: np.ndarray) -> dict:
    """{criterion: (comparable [Q, G] bool, match [Q, G] bool)}"""
    out = {}
    for name, key in (("surface", "surf"), ("transparency", "transp")):
        a, b = I[key][qi][:, None], I[key][gal][None, :]
        out[name] = ((a >= 0) & (b >= 0), a == b)
    qe, ge = I["eff"][qi], I["eff"][gal]
    comp = (qe.any(1)[:, None]) & I["eff_has"][gal][None, :]
    out["effects"] = (comp, (qe.astype(np.int32) @ ge.T.astype(np.int32)) > 0)
    lo, hi = I["cone"][qi, 0][:, None], I["cone"][qi, 1][:, None]
    glo, ghi = I["cone"][gal, 0][None, :], I["cone"][gal, 1][None, :]
    out["cone"] = ((lo >= 0) & (glo >= 0), (lo <= ghi) & (glo <= hi))
    # colour: ColorThief palette
    ql, qp = I["ct_lab"][qi], I["ct_pct"][qi]
    gl, gp = I["ct_lab"][gal], I["ct_pct"][gal]
    d = B.de2000(ql[:, None, :, None, :], gl[None, :, None, :, :])        # [Q, G, 3, 3]
    d = np.where((gp > 0)[None, :, None, :], d, 999)
    m1 = d[:, :, 0, :].min(-1) <= 20
    m2 = np.where((qp[:, 1] >= 0.15)[:, None], d[:, :, 1, :].min(-1) <= 20, True)
    out["colour"] = (np.broadcast_to((qp[:, 0] > 0)[:, None], m1.shape) & (gp[:, 0] > 0)[None, :], m1 & m2)
    a, b = I["form"][qi][:, None], I["form"][gal][None, :]
    out["form"] = ((a >= 0) & (b >= 0), a == b)
    return out


def gains(crit: dict, names) -> tuple[np.ndarray, np.ndarray]:
    comp = sum(crit[c][0].astype(np.int32) for c in names)
    match = sum((crit[c][0] & crit[c][1]).astype(np.int32) for c in names)
    g = np.where(comp > 0, match / np.maximum(comp, 1), 0).astype(np.float32)
    rel = (g >= 0.75) & (comp >= (2 if len(names) > 1 else 1))
    return g, rel


def metrics(order: np.ndarray, gain: np.ndarray, rel: np.ndarray, ks=KS) -> dict:
    """order [Q, >=20] gallery positions; gain / rel [Q, G]."""
    valid = rel.any(1)
    R = np.take_along_axis(rel, order, 1)
    Gn = np.take_along_axis(gain, order, 1)
    out = {"queries": int(valid.sum())}
    for k in ks:
        out[f"Recall@{k}"] = round(float(R[valid, :k].any(1).mean()), 4)
        out[f"P@{k}"] = round(float(R[valid, :k].mean()), 4)
    first = np.where(R[valid].any(1), R[valid].argmax(1) + 1, np.inf)
    out["MRR@20"] = round(float(np.where(np.isfinite(first), 1 / first, 0).mean()), 4)
    disc = 1 / np.log2(np.arange(2, 12))
    ideal = -np.sort(-gain[valid], axis=1)[:, :10]
    out["nDCG@10"] = round(float(((Gn[valid, :10] * disc).sum(1) / np.clip((ideal * disc).sum(1), 1e-9, None)).mean()), 4)
    return out


def weight_grid(comps, step=0.05, wmin=0.05):
    n = len(comps)
    ticks = np.round(np.arange(wmin, 1 + 1e-9, step), 3)
    for w in itertools.product(ticks, repeat=n - 1):
        last = round(1 - sum(w), 3)
        if last >= wmin - 1e-9:
            yield dict(zip(comps, list(w) + [last]))
