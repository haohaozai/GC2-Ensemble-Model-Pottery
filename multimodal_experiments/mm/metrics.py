"""Retrieval metrics.

Same-recipe task (identity): a query modality of recipe r must retrieve recipe r from the gallery built from
another modality set. Reported strict and tie-aware (gallery recipes whose target content is IDENTICAL to r's -
same tag description and / or same rounded oxide analysis - count as ties, because no model can separate them).

Attribute-relevance task ("most appropriate"): relevant = gallery recipe whose dominant glaze colour is within
Delta E 2000 <= 20 of the query photo AND whose known cone range overlaps AND known surface group / transparency
group match the query's (at least one of cone / surface / transparency must be comparable). Graded gain for
nDCG = matched attributes / comparable attributes. Recall@K here = share of queries with >= 1 relevant recipe in
the top K (the proposal deck's definition: "an acceptable recipe is retrieved within the top K").
"""
from __future__ import annotations

import numpy as np

KS = (1, 3, 5, 10)
DE_THRESHOLD = 20.0
PAIRS = [("text", "image"), ("text", "formula"), ("image", "formula")]


def fuse(Z: dict, mods, weights: dict):
    """normalised weighted sum of the available modality vectors -> (V, mask)."""
    n = len(next(iter(Z.values()))[0])
    d = next(iter(Z.values()))[0].shape[1]
    V = np.zeros((n, d), np.float32)
    M = np.zeros(n, bool)
    for m in mods:
        if m in Z:
            z, mk = Z[m]
            V[mk] += weights.get(m, 1.0) * z[mk]
            M |= mk
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.clip(nrm, 1e-8, None)
    return V, M & (nrm[:, 0] > 1e-8)


def ranks(S: np.ndarray, target: np.ndarray, tie_gid: np.ndarray | None):
    st = S[np.arange(len(S)), target]
    strict = 1 + (S > st[:, None]).sum(1)
    if tie_gid is None:
        return strict, strict
    TM = tie_gid[None, :] == tie_gid[target][:, None]
    best = np.where(TM, S, -np.inf).max(1)
    tie = 1 + ((S > best[:, None]) & ~TM).sum(1)
    return strict, tie


def summarise(r: np.ndarray, ks=KS, prefix="") -> dict:
    out = {f"{prefix}R@{k}": round(float((r <= k).mean()), 4) for k in ks}
    out[f"{prefix}MRR"] = round(float((1.0 / r).mean()), 4)
    out[f"{prefix}median_rank"] = float(np.median(r))
    return out


def tie_groups(bank, idx, mods):
    """identical-content groups of the gallery side (text -> tag text, formula -> formula, image -> none)."""
    parts = []
    if "text" in mods:
        parts.append(bank.text_gid[idx])
    if "formula" in mods:
        parts.append(bank.form_gid[idx])
    if not parts:
        return None
    keys = {}
    return np.array([keys.setdefault(t, len(keys)) for t in zip(*parts)])


def same_recipe(Zq: dict, Zg: dict, bank, idx, qmods, gmods, weights=None, ks=KS, keep_ranks=False, qsel=None) -> dict:
    """query = fused qmods, gallery = fused gmods, both over the recipes idx (queries need both sides);
    qsel (bool over idx) restricts which recipes are used as queries (e.g. external ones)."""
    w = weights or {}
    Q, qm = fuse(Zq, qmods, w)
    G, gm = fuse(Zg, gmods, w)
    grow = np.nonzero(gm)[0]
    qrow = np.nonzero(qm & gm & (qsel if qsel is not None else True))[0]
    if len(qrow) == 0 or len(grow) < 2:
        return {"n": 0}
    pos = {g: i for i, g in enumerate(grow)}
    S = Q[qrow] @ G[grow].T
    tgt = np.array([pos[q] for q in qrow])
    tg = tie_groups(bank, np.asarray(idx)[grow], gmods)
    strict, tie = ranks(S, tgt, tg)
    res = {"n": int(len(qrow)), "gallery": int(len(grow)), **summarise(strict, ks), **summarise(tie, ks, "tie_")}
    res["random_R@5"] = round(min(1.0, 5 / len(grow)), 4)
    if keep_ranks:
        res["_ranks"] = (np.asarray(idx)[qrow], strict, tie)
    return res


def pairwise_same_recipe(Z: dict, bank, idx, ks=KS) -> dict:
    out = {}
    for a, b in PAIRS:
        if a in Z and b in Z:
            for q, g in ((a, b), (b, a)):
                r = same_recipe(Z, Z, bank, idx, [q], [g], ks=ks)
                if r.get("n"):
                    out[f"{q[0].upper()}->{g[0].upper()}"] = r
    return out


def cosine_gap(Z: dict, rng_seed=0) -> dict:
    out = {}
    rng = np.random.default_rng(rng_seed)
    for a, b in PAIRS:
        if a in Z and b in Z:
            za, ma = Z[a]
            zb, mb = Z[b]
            rows = np.nonzero(ma & mb)[0]
            if len(rows) < 3:
                continue
            pos = (za[rows] * zb[rows]).sum(1)
            perm = rng.permutation(rows)
            neg = (za[rows] * zb[perm]).sum(1)[rows != perm]
            out[f"{a}-{b}"] = {"pos_mean": round(float(pos.mean()), 4), "neg_mean": round(float(neg.mean()), 4),
                               "gap": round(float(pos.mean() - neg.mean()), 4),
                               "pos_above_neg": round(float((pos[:len(neg)] > neg).mean()), 4)}
    return out


# ---------------------------------------------------------------- attribute relevance
def delta_e2000(lab_q: np.ndarray, lab_g: np.ndarray) -> np.ndarray:
    from skimage.color import deltaE_ciede2000
    out = np.full((len(lab_q), len(lab_g)), np.nan, np.float32)
    for i in range(0, len(lab_q), 64):
        a = np.broadcast_to(lab_q[i:i + 64, None, :], (min(64, len(lab_q) - i), len(lab_g), 3))
        b = np.broadcast_to(lab_g[None, :, :], a.shape)
        out[i:i + 64] = deltaE_ciede2000(a, b)
    return out


def relevance(bank, qi, gi, qattr=None):
    """binary relevance + graded gain [nq, ng]; qattr overrides query attributes (external queries)."""
    A = qattr or {"lo": bank.cone_lo[qi], "hi": bank.cone_hi[qi], "s": bank.surf[qi], "t": bank.transp[qi], "lab": bank.lab[qi]}
    lo_g, hi_g, s_g, t_g = bank.cone_lo[gi], bank.cone_hi[gi], bank.surf[gi], bank.transp[gi]
    cc = (A["lo"][:, None] >= 0) & (lo_g[None] >= 0)
    cm = cc & (A["lo"][:, None] <= hi_g[None]) & (lo_g[None] <= A["hi"][:, None])
    sc = (A["s"][:, None] >= 0) & (s_g[None] >= 0)
    sm = sc & (A["s"][:, None] == s_g[None])
    tc = (A["t"][:, None] >= 0) & (t_g[None] >= 0)
    tm = tc & (A["t"][:, None] == t_g[None])
    dE = delta_e2000(A["lab"], bank.lab[gi])
    colc = ~np.isnan(dE)
    colm = colc & (np.nan_to_num(dE, nan=999) <= DE_THRESHOLD)
    ncomp = cc.astype(int) + sc + tc
    allm = (~cc | cm) & (~sc | sm) & (~tc | tm)
    rel = colm & (ncomp >= 1) & allm
    gain = (cm.astype(float) + sm + tm + colm) / (ncomp + colc).clip(1)
    return rel, gain, dE


def attribute_retrieval(Vq, qm, Vg, gm, bank, qi, gi, ks=KS, qattr=None, rerank=None, keep_top=0) -> dict:
    """Vq/Vg: fused vectors of the query rows qi / gallery rows gi (already selected)."""
    qv = np.nonzero(qm)[0]
    gv = np.nonzero(gm)[0]
    qa = None
    if qattr is not None:
        qa = {k: v[qv] for k, v in qattr.items()}
    rel, gain, dE = relevance(bank, np.asarray(qi)[qv], np.asarray(gi)[gv], qa)
    S = Vq[qv] @ Vg[gv].T
    if rerank is not None:
        S = S + rerank(qv, gv)
    valid = rel.any(1) & ~np.isnan(dE).all(1)
    order = np.argsort(-S, axis=1)[:, :max(ks + (10,))]
    out = {"n_queries": int(len(qv)), "n_scored": int(valid.sum()), "gallery": int(len(gv))}
    R = np.take_along_axis(rel, order, 1)
    Gn = np.take_along_axis(gain, order, 1)
    E = np.take_along_axis(np.nan_to_num(dE, nan=np.nan), order, 1)
    for k in ks:
        out[f"Hit@{k}"] = round(float(R[valid, :k].any(1).mean()), 4) if valid.any() else None
        out[f"P@{k}"] = round(float(R[valid, :k].mean()), 4) if valid.any() else None
    disc = 1 / np.log2(np.arange(2, 12))
    ideal = -np.sort(-gain, axis=1)[:, :10]
    ndcg = (Gn[:, :10] * disc).sum(1) / np.clip((ideal * disc).sum(1), 1e-9, None)
    out["nDCG@10"] = round(float(ndcg[valid].mean()), 4) if valid.any() else None
    prec = np.cumsum(R[:, :10], 1) / np.arange(1, 11)
    ap = (prec * R[:, :10]).sum(1) / np.clip(np.minimum(rel.sum(1), 10), 1, None)
    out["MAP@10"] = round(float(ap[valid].mean()), 4) if valid.any() else None
    out["meanDeltaE@5"] = round(float(np.nanmean(E[:, :5])), 2)
    out["relevant_per_query_median"] = float(np.median(rel.sum(1)[valid])) if valid.any() else 0.0
    if keep_top:
        out["_top"] = {"q": np.asarray(qi)[qv], "order": np.asarray(gi)[gv][order[:, :keep_top]],
                       "score": np.take_along_axis(S, order[:, :keep_top], 1), "rel": R[:, :keep_top],
                       "dE": E[:, :keep_top], "valid": valid}
    return out


def random_attribute(bank, qi, gi, seed=0, ks=KS) -> dict:
    rng = np.random.default_rng(seed)
    n, d = len(qi), 8
    Vq = rng.normal(size=(n, d)).astype(np.float32)
    Vg = rng.normal(size=(len(gi), d)).astype(np.float32)
    return attribute_retrieval(Vq, np.ones(n, bool), Vg, np.ones(len(gi), bool), bank, qi, gi, ks)
