"""Leave-one-out retrieval metrics with weak labels + paired bootstrap vs the 000 baseline.

Every image is used once as a query against all other images of the SAME variant
(query and database are always preprocessed identically - that is how the system
would be deployed).
"""
from __future__ import annotations

import numpy as np


def per_query_scores(emb: np.ndarray, labels: np.ndarray, ks=(5, 10)) -> dict[str, np.ndarray]:
    """emb (N,d) L2-normalised; labels (N,) object array, None = unknown -> excluded."""
    valid = np.array([l is not None and l == l for l in labels])      # drop None / NaN
    E, y = emb[valid], labels[valid]
    sim = E @ E.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, :-1]                          # drop self (last)
    rel = (y[order] == y[:, None])
    n_rel = rel.sum(1)
    keep = n_rel > 0
    out = {f"P@{k}": rel[:, :k].mean(1) for k in ks}
    cum = np.cumsum(rel, 1)
    prec = cum / np.arange(1, rel.shape[1] + 1)
    out["AP"] = np.where(keep, (prec * rel).sum(1) / np.maximum(n_rel, 1), np.nan)
    out["R@5"] = (rel[:, :5].any(1)).astype(float)                   # Recall@K as in the proposal
    out["_index"] = np.where(valid)[0]
    return out


def chem_alignment(emb: np.ndarray, chem: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-query mean cosine similarity of colourant chemistry between a query and its
    top-k visual neighbours (z-scored chemistry). Higher = visual space agrees with chemistry."""
    z = (chem - chem.mean(0)) / (chem.std(0) + 1e-6)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    csim = z @ z.T
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    top = np.argsort(-sim, axis=1)[:, :k]
    return np.take_along_axis(csim, top, 1).mean(1)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    """mean(a-b) and 95 % CI over queries (a, b aligned per query)."""
    d = a - b
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
