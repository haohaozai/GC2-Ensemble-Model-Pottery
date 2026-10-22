"""Why are the external numbers lower than the internal test numbers? Measured, not guessed.

Uses the saved embeddings of the selected final model (results/db_embeddings.npz); nothing is retrained and the
external recipes are not used for any choice. Written to results/E_external_diagnosis.json:
  * input consistency  - external recipes that are also in the Glazy export: parsed oxides / tags / ingredients
                         compared with the export (rules out a parsing or encoding bug)
  * alignment          - mean cosine between the modalities of the SAME recipe, per split (train / test / external)
  * novelty            - distance of each query's oxide wt% to the nearest training recipe, and same-recipe
                         Recall@10 for near-copies vs novel recipes
  * equal galleries    - same-recipe Recall@5 when internal test and external both search a gallery of the same size
  * formula twins      - external recipes whose formula equals an internal one (0.1 wt%, traces < 0.05 ignored)
"""
from __future__ import annotations

import json

import numpy as np

from . import data as D
from . import formula as F
from .common import GLAZY_EXPORT, log, read_csv, read_json, write_json

DIRS = {"I->T+F": (["image"], ["text", "formula"]), "T->I+F": (["text"], ["image", "formula"]),
        "F->T+I": (["formula"], ["text", "image"])}
PAIRS = [("text", "image"), ("text", "formula"), ("image", "formula")]


def _ox(r) -> dict:
    return {k: float(v) for k, v in json.loads(r["oxides_json"] or "{}").items()}


def formula_sig(ox: dict) -> tuple:
    return tuple(sorted((k, round(v, 1)) for k, v in ox.items() if round(v, 1) > 0))


def _rank(s: np.ndarray, qi: int, gal: np.ndarray) -> float:
    v = s[gal]
    t = s[qi]
    return float((v > t).sum() + 1 + max(0, (v == t).sum() - 1) / 2)       # tie-aware


def run(work) -> dict:
    res_dir = work / "results"
    p = res_dir / "db_embeddings.npz"
    E = read_json(res_dir / "E_final.json", {})
    if not p.exists() or not E:
        return {}
    z = np.load(p)
    ids = [str(x) for x in z["ids"]]
    Z = {m: z["Z_" + m] for m in ("text", "image", "formula")}
    M = {m: z["M_" + m].astype(bool) for m in ("text", "image", "formula")}
    w = E["weights"]
    internal = {r["recipe_id"]: r for r in read_csv(work / "data" / "recipes.csv")}
    ext = {r["recipe_id"]: r for r in read_csv(work / "external" / "recipes_ext.csv")} \
        if (work / "external" / "recipes_ext.csv").exists() else {}
    R = {**internal, **ext}
    split = np.array([R[i]["split"] if i in R else "?" for i in ids])
    is_ext = split == "ext"
    db = ~is_ext
    excl = {i for i, r in ext.items() if r.get("near_duplicate_photo") in ("1", 1)}
    out = {"model_seed": E.get("selected_seed"), "weights": w}

    def fuse(mods):
        V = sum(w[m] * Z[m] * M[m][:, None] for m in mods)
        return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)

    # --- input consistency (external recipes that are also in the export)
    exp = {r["ID"]: r for r in read_csv(GLAZY_EXPORT)}
    both = [i for i in ext if i[1:] in exp]
    if both:
        tags = D.clean_tags([{"recipe_id": i, "Name": exp[i[1:]]["Name"], "Type": exp[i[1:]]["Type"],
                              "State": exp[i[1:]]["State"], "Subtype": exp[i[1:]]["Subtype"], "Cone": exp[i[1:]]["Cone"],
                              "Surface": exp[i[1:]]["Surface"], "Transparency": exp[i[1:]]["Transparency"],
                              "Atmospheres": exp[i[1:]]["Atmospheres"], "Country": exp[i[1:]]["Country"],
                              "Other Names": exp[i[1:]]["Other Names"]} for i in both])
        oxcols = D.oxide_columns(list(next(iter(exp.values())).keys()))
        ox_diff, same_tags, same_ing = [], 0, 0
        for i in both:
            e = exp[i[1:]]
            eo = {c: float(e[c]) for c in oxcols if (e.get(c) or "").strip() and float(e[c]) > 0}
            xo = _ox(ext[i])
            ox_diff.append(max(abs(eo.get(k, 0) - xo.get(k, 0)) for k in set(eo) | set(xo)))
            same_tags += tags[i]["desc_tags"] == ext[i]["desc_tags"]
            a = sorted(F.norm_ingredient(x[1]) for x in json.loads(ext[i]["ingredients_json"]))
            b = sorted(F.norm_ingredient(q.replace(" (additional)", "").rsplit(" ", 1)[0])
                       for q in e["Ingredients"].split("; ") if q)
            same_ing += a == b
        out["input_consistency"] = {"n_in_export": len(both), "max_oxide_abs_diff_wt%": round(float(max(ox_diff)), 4),
                                    "tags_identical": same_tags, "ingredient_names_identical": same_ing}

    # --- alignment: same-recipe cosine between modalities, and random-pair cosine
    rng = np.random.default_rng(0)
    al = {}
    for sp in ("train", "val", "test", "ext"):
        rows = np.nonzero(split == sp)[0]
        if not len(rows):
            continue
        d = {}
        for a, b in PAIRS:
            r = rows[M[a][rows] & M[b][rows]]
            pos = np.sum(Z[a][r] * Z[b][r], 1)
            sh = rng.permutation(r)
            neg = np.sum(Z[a][r] * Z[b][sh], 1)
            d[f"{a}-{b}"] = {"same_recipe": round(float(pos.mean()), 4), "random_pair": round(float(neg.mean()), 4)}
        al[sp] = d
    out["alignment"] = al

    # --- novelty: nearest training formula
    cols = sorted({k for r in R.values() for k in _ox(r)})

    def vec(i):
        o = _ox(R[i])
        v = np.array([o.get(k, 0.0) for k in cols])
        return v / v.sum() * 100 if v.sum() > 0 else v
    X = np.array([vec(i) if i in R else np.zeros(len(cols)) for i in ids])
    tr = np.nonzero(split == "train")[0]
    Xt = X[tr]
    groups = {"test": [k for k in np.nonzero(split == "test")[0]],
              "ext": [k for k in np.nonzero(is_ext)[0] if ids[k] not in excl]}
    nn = {g: np.array([np.sqrt(((Xt - X[k]) ** 2).sum(1)).min() for k in rows]) for g, rows in groups.items()}
    out["novelty"] = {g: {"n": len(d), "median_wt%_distance": round(float(np.median(d)), 2),
                          "share_near_copy_lt2": round(float(np.mean(d < 2)), 3),
                          "share_far_gt5": round(float(np.mean(d > 5)), 3)} for g, d in nn.items() if len(d)}

    # --- same-recipe Recall@K by novelty (gallery = database, + the query itself for external)
    byn = {}
    for name, (qm, gm) in DIRS.items():
        Q, G = fuse(qm), fuse(gm)
        qok = np.any([M[m] for m in qm], 0)
        for g, rows in groups.items():
            rk, near = [], []
            for j, k in enumerate(rows):
                if not qok[k]:
                    continue
                gal = db.copy()
                gal[k] = True
                rk.append(_rank(G @ Q[k], k, gal))
                near.append(nn[g][j] < 2)
            rk, near = np.array(rk), np.array(near, bool)
            byn[f"{name} {g}"] = {"n": len(rk), "R@10": round(float(np.mean(rk <= 10)), 4),
                                  "R@10_near_copy": round(float(np.mean(rk[near] <= 10)), 4) if near.any() else None,
                                  "n_near_copy": int(near.sum()),
                                  "R@10_novel": round(float(np.mean(rk[~near] <= 10)), 4) if (~near).any() else None,
                                  "n_novel": int((~near).sum()), "median_rank": float(np.median(rk))}
    out["recall_by_novelty_full_db"] = byn

    # --- equal gallery size: external among external vs internal test among random test subsets of that size
    n_ext = len(groups["ext"])
    eq = {}
    if n_ext:
        for name, (qm, gm) in DIRS.items():
            Q, G = fuse(qm), fuse(gm)

            def r5(rows):
                gal = np.zeros(len(ids), bool)
                gal[rows] = True
                return float(np.mean([_rank(G @ Q[k], k, gal) <= 5 for k in rows if np.any([M[m][k] for m in qm])]))
            ex5 = r5(np.array(groups["ext"]))
            te = np.array(groups["test"])
            draws = [r5(np.random.default_rng(s).choice(te, n_ext, replace=False)) for s in range(5)]
            eq[name] = {"gallery_size": n_ext, "external_R@5": round(ex5, 4),
                        "internal_test_R@5_mean_of_5_draws": round(float(np.mean(draws)), 4),
                        "internal_test_R@5_sd": round(float(np.std(draws)), 4)}
    out["equal_gallery_size"] = eq

    # --- formula twins and text notes
    sig = {}
    for i, r in internal.items():
        sig.setdefault(formula_sig(_ox(r)), []).append(i)
    twins = [{"external": i, "internal": j, "internal_split": internal[j]["split"]}
             for i, r in ext.items() for j in sig.get(formula_sig(_ox(r)), [])]
    out["formula_twins"] = {"n_external_with_twin": len({t["external"] for t in twins}), "pairs": twins}
    out["empty_notes_share"] = {"internal": round(float(np.mean([not r["desc_free"] for r in internal.values()])), 3),
                                "external": round(float(np.mean([not r["desc_free"] for r in ext.values()])), 3) if ext else None}
    write_json(res_dir / "E_external_diagnosis.json", out)
    log().info(f"[report] external diagnosis: alignment text-image train {al.get('train', {}).get('text-image')} | "
               f"test {al.get('test', {}).get('text-image')} | ext {al.get('ext', {}).get('text-image')}; "
               f"novelty {out['novelty']}")
    return out
