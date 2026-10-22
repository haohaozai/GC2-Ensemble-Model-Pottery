"""DBSCAN on the fused recipe embeddings (as in the proposal), compared with retrieval without clustering.

eps from the k-distance curve (k = min_samples = 5, cosine distance, knee = point farthest from the chord),
also 0.8x / 1.2x and the 10th / 25th / 50th percentile of the k-distances (the knee alone can merge everything into
one cluster when the space has no density gaps). Reported: clusters, noise ratio, silhouette (cosine, without
noise), purity against surface group and subtype level 2. Cluster-aware retrieval adds +beta to gallery recipes in
the query's cluster; eps and beta are chosen together on validation, then applied once on test. Noise points (knee
eps) are FLAGGED as possible anomalies, never removed.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import metrics as MT
from .common import log, write_csv, write_json

MODS = ["text", "image", "formula"]


def knee_eps(V: np.ndarray, k: int = 5) -> tuple[float, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(V)
    d = np.sort(nn.kneighbors(V)[0][:, k])
    x = np.linspace(0, 1, len(d))
    y = (d - d[0]) / max(1e-9, d[-1] - d[0])
    knee = int(np.argmax(np.abs(y - x)))
    return float(d[knee]), d


def purity(labels, attr):
    tot, hit = 0, 0
    for c in set(labels) - {-1}:
        vals = [a for l, a in zip(labels, attr) if l == c and a not in ("", -1, None)]
        if vals:
            hit += Counter(vals).most_common(1)[0][1]
            tot += len(vals)
    return round(hit / tot, 4) if tot else None


def run(bank, Z: dict, weights: dict, out_dir) -> dict:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    V, M = MT.fuse(Z, MODS, weights)
    M &= bank.split != "ext"                      # clusters are built from the recipe database only
    rows = np.nonzero(M)[0]
    eps0, kd = knee_eps(V[rows])
    res = {"k_distance_knee_eps": round(eps0, 4), "min_samples": 5, "variants": []}
    cands = [(f"{f}x knee", eps0 * f) for f in (0.8, 1.0, 1.2)] + \
            [(f"k-distance p{p}", float(np.percentile(kd, p))) for p in (10, 25, 50)]
    cands.sort(key=lambda c: c[1])
    all_labels = {}
    for name, eps in cands:
        lab = DBSCAN(eps=eps, min_samples=5, metric="cosine").fit_predict(V[rows])
        n_cl = len(set(lab) - {-1})
        noise = float((lab == -1).mean())
        sil = None
        keep = lab != -1
        if n_cl >= 2 and keep.sum() > 50:
            sub = np.nonzero(keep)[0]
            if len(sub) > 4000:
                sub = np.random.default_rng(0).choice(sub, 4000, replace=False)
            sil = round(float(silhouette_score(V[rows][sub], lab[sub], metric="cosine")), 4)
        sizes = sorted(Counter(lab[lab != -1]).values(), reverse=True)
        v = {"eps": round(eps, 4), "label": name, "clusters": n_cl, "noise_ratio": round(noise, 4),
             "silhouette_cosine": sil, "largest_clusters": sizes[:10],
             "largest_cluster_share": round(sizes[0] / len(lab), 4) if sizes else None,
             "purity_surface": purity(lab, bank.surf[rows].tolist()),
             "purity_subtype_l2": purity(lab, [bank.R[i]["subtype_l2"] for i in rows])}
        res["variants"].append(v)
        full = np.full(bank.n, -2)
        full[rows] = lab
        all_labels[name] = full
    knee = all_labels["1.0x knee"]
    # anomalies = DBSCAN noise at the knee eps (flag only)
    write_csv(out_dir / "dbscan_noise_flagged.csv",
              [{"recipe_id": bank.ids[i], "name": bank.R[i]["name"], "split": bank.split[i],
                "subtype": bank.R[i]["subtype"], "glazy_url": bank.R[i]["glazy_url"]} for i in np.nonzero(knee == -1)[0]])
    np.save(out_dir / "dbscan_labels.npy", knee)
    # cluster-aware re-ranking vs none (attribute task): eps + beta on validation, then test
    tr, va, te = bank.idx("train"), bank.idx("val"), bank.idx("test")

    def evaluate(qi, gi, labels, beta):
        Q, qm = MT.fuse({m: (Z[m][0][qi], Z[m][1][qi]) for m in Z}, MODS, weights)
        G, gm = MT.fuse({m: (Z[m][0][gi], Z[m][1][gi]) for m in Z}, MODS, weights)
        glab = labels[gi]

        def rr(qv, gv):
            S0 = Q[qv] @ G[gv].T
            nn = np.argmax(S0, 1)
            qlab = glab[gv][nn]
            same = (glab[gv][None, :] == qlab[:, None]) & (qlab[:, None] >= 0)
            return beta * same
        return MT.attribute_retrieval(Q, qm, G, gm, bank, qi, gi, rerank=rr if beta else None)
    r0 = evaluate(va, tr, knee, 0.0)
    sel = [{"eps_label": "none", "beta": 0.0, "val_nDCG@10": r0["nDCG@10"], "val_Hit@5": r0["Hit@5"]}]
    for v in res["variants"]:
        if v["clusters"] < 2:
            continue
        for beta in (0.02, 0.05, 0.1):
            r = evaluate(va, tr, all_labels[v["label"]], beta)
            sel.append({"eps_label": v["label"], "beta": beta, "val_nDCG@10": r["nDCG@10"], "val_Hit@5": r["Hit@5"]})
    best = max(sel, key=lambda x: (x["val_nDCG@10"] or 0, -x["beta"]))
    if best["eps_label"] == "none":           # no clustering beats plain retrieval on validation: report the best variant anyway
        best = max(sel[1:], key=lambda x: (x["val_nDCG@10"] or 0, -x["beta"])) if len(sel) > 1 else best
    gi = np.concatenate([tr, va])
    res["rerank_selection_val"] = sel
    res["chosen_beta"] = best["beta"]
    res["chosen_eps_label"] = best["eps_label"]
    res["test_without_clusters"] = evaluate(te, gi, knee, 0.0)
    res["test_with_clusters"] = evaluate(te, gi, all_labels.get(best["eps_label"], knee), best["beta"])
    res["noise_flagged_csv"] = str(out_dir / "dbscan_noise_flagged.csv")
    write_json(out_dir / "E_dbscan.json", res)
    kv = next(v for v in res["variants"] if v["label"] == "1.0x knee")
    log().info(f"[E] DBSCAN knee eps {eps0:.3f}: {kv['clusters']} clusters, noise {kv['noise_ratio']}; "
               f"re-rank {best['eps_label']} beta {best['beta']}; test nDCG@10 "
               f"{res['test_without_clusters']['nDCG@10']} -> {res['test_with_clusters']['nDCG@10']}")
    return res
