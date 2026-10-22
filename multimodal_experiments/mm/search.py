"""Experiment phases A -> E (resource-aware, every run logged and resumable).

A  baselines without training: random, CLIP zero-shot (image <-> text), CCA per modality pair, unimodal kNN
B1 encoder screening: one modality's encoder varied at a time (train -> validation)
B2 GridSearchCV (sklearn) with GroupKFold(3) on the TRAINING split over head / dim / temperature / lr
C  top-5 B2 configurations retrained on train, ranked on validation
D  refinement around the best (dropout x weight decay x batch) with 3 seeds; fusion weights (image >= text >= formula)
E  final: best configuration x 3 seeds trained on train, early-stopped on validation, evaluated ONCE on test;
   modality-pair models + 7 query-modality ablation; cosine gap; colour Delta E
Selection never looks at the test split.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch

from . import metrics as MT
from . import model as MD
from .common import Timer, dir_size_gb, disk_guard, log, read_json, write_json

MODS = ["text", "image", "formula"]
ABL_SETS = [["image"], ["text"], ["formula"], ["image", "text"], ["image", "formula"], ["text", "formula"],
            ["image", "text", "formula"]]
BANK = None


def run_id(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def run_one(bank, cfg, fit_idx, val_idx, out_dir, phase, note=""):
    """Train + validate one configuration; cached on disk by its config hash."""
    rid = run_id({**cfg, "_phase_fit": int(len(fit_idx)), "_val": int(len(val_idx))})
    p = out_dir / phase / f"{rid}.json"
    old = read_json(p)
    if old:
        return old
    disk_guard(f"{phase} {rid}")
    model, data, hist, meta = MD.train(bank, cfg, fit_idx, val_idx)
    vs = MD.val_score(model, data, val_idx, bank)
    res = {"run_id": rid, "phase": phase, "note": note, "cfg": cfg, "history": hist, "meta": meta,
           "val": {"mean_MRR": round(vs["mean_MRR"], 4), "directions": vs["directions"]},
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(p, res)
    log().info(f"  [{phase}] {rid} {note} val mean MRR {res['val']['mean_MRR']:.4f} | {meta['epochs']} ep "
               f"{meta['seconds']} s | peak VRAM {meta['peak_vram_GB']} GB")
    del model, data
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- A: baselines
def phase_a(bank, out_dir) -> dict:
    p = out_dir / "A_baselines.json"
    if p.exists():
        return read_json(p)
    tr, va, te = bank.idx("train"), bank.idx("val"), bank.idx("test")
    res = {"same_recipe": {}, "attribute": {}}
    # CLIP zero-shot: CLIP image features vs CLIP text features of the tag description (same pretrained space)
    Xi, Mi = bank.image_raw("clip")
    texts, Mt = bank.text_raw("clip_text")
    from . import text_feats as TF
    Xt = TF.encode_fixed("clip_text", texts)
    for part, idx in (("val", va), ("test", te)):
        Z = {"image": (Xi[idx], Mi[idx]), "text": (Xt[idx], Mt[idx])}
        res["same_recipe"][f"clip_zero_shot/{part}"] = MT.pairwise_same_recipe(Z, bank, idx)
    # CCA per modality pair (fitted on train)
    from sklearn.cross_decomposition import CCA
    feats = {"text": bank.get("text", "minilm_tags", tr), "image": bank.get("image", "clip", tr),
             "formula": bank.get("formula", "umf", tr)}
    for a, b in MT.PAIRS:
        Xa, Ma, _ = feats[a]
        Xb, Mb, _ = feats[b]
        rows = tr[Ma[tr] & Mb[tr]]
        ca = Xa[rows].std(0) > 1e-3                     # constant columns make CCA's scaling divide by zero
        cb = Xb[rows].std(0) > 1e-3
        Xa, Xb = Xa[:, ca], Xb[:, cb]
        k = int(min(32, Xa.shape[1], Xb.shape[1]))
        try:
            cca = CCA(n_components=k, max_iter=2000).fit(Xa[rows], Xb[rows])
        except Exception as exc:  # noqa: BLE001
            try:
                k = int(min(16, k))
                cca = CCA(n_components=k, max_iter=2000, tol=1e-5).fit(Xa[rows], Xb[rows])
            except Exception as exc2:  # noqa: BLE001
                res["same_recipe"].setdefault("cca_errors", {})[f"{a}-{b}"] = f"{exc}; retry: {exc2}"
                log().info(f"[A] CCA {a}-{b} failed: {exc2}")
                continue
        for part, idx in (("val", va), ("test", te)):
            A, B = cca.transform(Xa[idx], Xb[idx])
            A /= np.linalg.norm(A, axis=1, keepdims=True).clip(1e-8)
            B /= np.linalg.norm(B, axis=1, keepdims=True).clip(1e-8)
            Z = {a: (A, Ma[idx]), b: (B, Mb[idx])}
            res["same_recipe"].setdefault(f"cca/{part}", {}).update(MT.pairwise_same_recipe(Z, bank, idx))
    # attribute relevance: queries = test, gallery = train + val (unimodal kNN, no learning)
    gi = np.concatenate([tr, va])
    res["attribute"]["random"] = MT.random_attribute(bank, te, gi)
    for name, (X, M) in {"image_clip_knn": (Xi, Mi), "text_minilm_knn": (feats["text"][0], feats["text"][1]),
                         "formula_umf_knn": (feats["formula"][0], feats["formula"][1])}.items():
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
        res["attribute"][name] = MT.attribute_retrieval(Xn[te], M[te], Xn[gi], M[gi], bank, te, gi)
    write_json(p, res)
    log().info(f"[A] baselines saved: CLIP zero-shot test {res['same_recipe']['clip_zero_shot/test']}")
    return res


# ---------------------------------------------------------------- B1: encoder screening
def phase_b1(bank, out_dir, encoders: dict, base: dict) -> dict:
    tr, va = bank.idx("train"), bank.idx("val")
    results = []
    for mod, encs in encoders.items():
        for enc in encs:
            cfg = MD.default_cfg(**{**base, mod: enc})
            r = run_one(bank, cfg, tr, va, out_dir, "B1", note=f"{mod}={enc}")
            dirs = r["val"]["directions"]
            letter = mod[0].upper()
            m = np.mean([v["MRR"] for k, v in dirs.items() if letter in k]) if dirs else 0.0
            results.append({"modality": mod, "encoder": enc, "modality_MRR": round(float(m), 4),
                            "mean_MRR": r["val"]["mean_MRR"], "run_id": r["run_id"], "seconds": r["meta"]["seconds"]})
    best = {}
    for mod in encoders:
        rows = [x for x in results if x["modality"] == mod]
        best[mod] = max(rows, key=lambda x: x["modality_MRR"])["encoder"]
    write_json(out_dir / "B1_summary.json", {"results": results, "best": best})
    log().info(f"[B1] best encoders on validation: {best}")
    return best


# ---------------------------------------------------------------- B2: GridSearchCV
from sklearn.base import BaseEstimator  # noqa: E402


class CVEstimator(BaseEstimator):
    """sklearn estimator: X = recipe row indices (n, 1) of the training split; score = mean cross-modal MRR."""

    def __init__(self, head="mlp", dim=128, temperature="learn", lr=1e-3, base=None):
        self.head = head
        self.dim = dim
        self.temperature = temperature
        self.lr = lr
        self.base = base

    def fit(self, X, y=None):
        idx = np.asarray(X)[:, 0]
        g = BANK.group[idx]
        rng = np.random.default_rng(0)
        ug = rng.permutation(np.unique(g))
        es = set(ug[: max(1, int(0.15 * len(ug)))].tolist())
        inner_val = idx[np.isin(g, list(es))]
        inner_tr = idx[~np.isin(g, list(es))]
        cfg = MD.default_cfg(**{**self.base, "head": self.head, "dim": self.dim, "temperature": self.temperature,
                                "lr": self.lr})
        self.model_, self.data_, _, self.meta_ = MD.train(BANK, cfg, inner_tr, inner_val)
        return self

    def score(self, X, y=None):
        s = MD.val_score(self.model_, self.data_, np.asarray(X)[:, 0], BANK)["mean_MRR"]
        del self.model_, self.data_
        torch.cuda.empty_cache()
        return s


def phase_b2(bank, out_dir, base: dict, space: dict) -> list[dict]:
    p = out_dir / "B2_gridsearchcv.json"
    if p.exists():
        return read_json(p)["ranking"]
    global BANK
    BANK = bank
    from sklearn.model_selection import GridSearchCV, GroupKFold
    tr = bank.idx("train")
    X = tr[:, None]
    gs = GridSearchCV(CVEstimator(base=base), {k: list(v) for k, v in space.items()},
                      cv=GroupKFold(n_splits=3), refit=False, n_jobs=1, verbose=2, error_score="raise")
    t = Timer()
    gs.fit(X, groups=bank.group[tr])
    cv = gs.cv_results_
    ranking = sorted([{"params": {k: v for k, v in pr.items()}, "mean_cv_MRR": round(float(m), 4),
                       "std_cv_MRR": round(float(s), 4), "fit_seconds": round(float(ft), 1)}
                      for pr, m, s, ft in zip(cv["params"], cv["mean_test_score"], cv["std_test_score"], cv["mean_fit_time"])],
                     key=lambda x: -x["mean_cv_MRR"])
    write_json(p, {"base": base, "space": space, "ranking": ranking, "seconds": t(), "folds": 3,
                   "cv": "GroupKFold on training split (groups = leakage groups)"})
    log().info(f"[B2] GridSearchCV {len(ranking)} configs x 3 folds in {t() / 60:.1f} min; best {ranking[0]}")
    return ranking


# ---------------------------------------------------------------- C + D
def phase_c(bank, out_dir, base, ranking, top=5) -> dict:
    tr, va = bank.idx("train"), bank.idx("val")
    res = []
    for r in ranking[:top]:
        cfg = MD.default_cfg(**{**base, **r["params"]})
        x = run_one(bank, cfg, tr, va, out_dir, "C", note=str(r["params"]))
        res.append({"params": r["params"], "cv_MRR": r["mean_cv_MRR"], "val_mean_MRR": x["val"]["mean_MRR"]})
    best = max(res, key=lambda x: x["val_mean_MRR"])
    write_json(out_dir / "C_summary.json", {"results": res, "best": best})
    log().info(f"[C] best on validation: {best}")
    return best


def phase_d(bank, out_dir, base, best_c, space, seeds=(0, 1, 2)) -> dict:
    tr, va = bank.idx("train"), bank.idx("val")
    rows = []
    for extra in MD.grid(space):
        vals = []
        for s in seeds:
            cfg = MD.default_cfg(**{**base, **best_c["params"], **extra, "seed": s})
            vals.append(run_one(bank, cfg, tr, va, out_dir, "D", note=f"{extra} seed {s}")["val"]["mean_MRR"])
        rows.append({"extra": extra, "val_MRR_mean": round(float(np.mean(vals)), 4), "val_MRR_sd": round(float(np.std(vals)), 4)})
    best = max(rows, key=lambda x: x["val_MRR_mean"])
    final_cfg = MD.default_cfg(**{**base, **best_c["params"], **best["extra"]})
    write_json(out_dir / "D_summary.json", {"results": rows, "best": best, "final_cfg": final_cfg})
    log().info(f"[D] best refinement {best} -> final cfg {final_cfg}")
    return final_cfg


# strictly image > text > formula (the requested priority); equal weights are evaluated as a reference only
WEIGHTS = [(0.4, 0.35, 0.25), (0.45, 0.35, 0.2), (0.5, 0.3, 0.2), (0.5, 0.35, 0.15), (0.55, 0.3, 0.15),
           (0.6, 0.25, 0.15), (0.6, 0.3, 0.1), (0.7, 0.2, 0.1), (0.45, 0.4, 0.15), (0.42, 0.38, 0.2)]
REFERENCE_WEIGHTS = [(1 / 3, 1 / 3, 1 / 3)]


def tune_fusion(bank, model, data, out_dir) -> dict:
    """weights (image, text, formula) with image >= text >= formula, chosen on validation only."""
    tr, va = bank.idx("train"), bank.idx("val")
    Zv, Zt = MD.embed(model, data, va), MD.embed(model, data, tr)
    rows = []
    for (wi, wt, wf), ref in [(x, False) for x in WEIGHTS] + [(x, True) for x in REFERENCE_WEIGHTS]:
        w = {"image": wi, "text": wt, "formula": wf}
        Qv, qm = MT.fuse(Zv, MODS, w)
        Gt, gm = MT.fuse(Zt, MODS, w)
        attr = MT.attribute_retrieval(Qv, qm, Gt, gm, bank, va, tr)
        ident = MT.same_recipe(Zv, Zv, bank, va, ["image"], ["text", "formula"], w)
        score = 0.5 * (attr["nDCG@10"] or 0) + 0.5 * ident.get("tie_MRR", 0)
        rows.append({"weights": w, "attr_nDCG@10": attr["nDCG@10"], "attr_Hit@5": attr["Hit@5"],
                     "I->T+F_tie_MRR": ident.get("tie_MRR"), "score": round(score, 4), "reference_only": ref})
    best = max((r for r in rows if not r["reference_only"]), key=lambda r: r["score"])
    write_json(out_dir / "D_fusion_weights.json", {"rows": rows, "best": best,
                                                     "rule": "image >= text >= formula (user priority); chosen on validation"})
    log().info(f"[D] fusion weights {best['weights']} (val score {best['score']})")
    return best["weights"]


# ---------------------------------------------------------------- E: final test
def _name(qs):
    return "+".join(x[0].upper() for x in qs)


def evaluate_split(bank, model, data, weights, qpart="test", tops=None) -> dict:
    """internal test: queries = test recipes; identity gallery = test recipes; attribute gallery = train + val."""
    tr, va, te = bank.idx("train"), bank.idx("val"), bank.idx(qpart)
    Zq = MD.embed(model, data, te)
    out = {"pairwise": MT.pairwise_same_recipe(Zq, bank, te), "cosine_gap": MT.cosine_gap(Zq)}
    ident = {}
    for qs in ABL_SETS[:-1]:
        gs = [m for m in MODS if m not in qs]
        ident[_name(qs) + "->" + _name(gs)] = MT.same_recipe(Zq, Zq, bank, te, qs, gs, weights)
    out["same_recipe_sets"] = ident
    # the same with the FULL database as gallery (train + val + test) - the protocol used for external recipes
    db = np.concatenate([tr, va, te])
    Zdb = MD.embed(model, data, db)
    qsel = np.zeros(len(db), bool)
    qsel[len(tr) + len(va):] = True
    out["same_recipe_full_gallery"] = {
        _name(qs) + "->" + _name([m for m in MODS if m not in qs]):
        MT.same_recipe(Zdb, Zdb, bank, db, qs, [m for m in MODS if m not in qs], weights, qsel=qsel)
        for qs in ABL_SETS[:-1]}
    gi = np.concatenate([tr, va])
    Zg = MD.embed(model, data, gi)
    G, gm = MT.fuse(Zg, MODS, weights)
    abl = {}
    for qs in ABL_SETS:
        Q, qm = MT.fuse(Zq, qs, weights)
        r = MT.attribute_retrieval(Q, qm, G, gm, bank, te, gi, keep_top=10 if tops is not None else 0)
        if "_top" in r:
            tops[f"test:{_name(qs)}"] = r.pop("_top")
        abl[_name(qs)] = r
    out["attribute_ablation"] = abl
    return out


def evaluate_external(bank, model, data, weights, tops=None) -> dict:
    """external Glazy recipes (never trained / tuned on). Near-duplicate photos are excluded from the metrics."""
    ext = bank.idx("ext")
    if len(ext) == 0:
        return {"n": 0}
    dup = np.array([bank.R[i].get("near_duplicate_photo") in ("1", 1) for i in ext])
    ext_ok = ext[~dup]
    db = np.nonzero(bank.split != "ext")[0]
    out = {"n_external": int(len(ext)), "n_used": int(len(ext_ok)), "n_near_duplicate_photo_excluded": int(dup.sum()),
           "database": int(len(db))}
    # (1) identity: the external recipe is added to the gallery; its photo / text / formula must find it
    gal = np.concatenate([db, ext_ok])
    Z = MD.embed(model, data, gal)
    qsel = np.zeros(len(gal), bool)
    qsel[len(db):] = True
    ident = {}
    for qs in ABL_SETS[:-1]:
        gs = [m for m in MODS if m not in qs]
        ident[_name(qs) + "->" + _name(gs)] = MT.same_recipe(Z, Z, bank, gal, qs, gs, weights, qsel=qsel)
    out["identity_in_gallery"] = ident
    # (2) most appropriate EXISTING recipes: query = external recipe, gallery = the 7k database (all modalities)
    Zx = MD.embed(model, data, ext_ok)
    Zd = MD.embed(model, data, db)
    G, gm = MT.fuse(Zd, MODS, weights)
    abl = {}
    for qs in ABL_SETS:
        Q, qm = MT.fuse(Zx, qs, weights)
        r = MT.attribute_retrieval(Q, qm, G, gm, bank, ext_ok, db, keep_top=10 if tops is not None else 0)
        if "_top" in r:
            tops[f"ext:{_name(qs)}"] = r.pop("_top")
        abl[_name(qs)] = r
    out["attribute_ablation"] = abl
    # baselines on the same queries
    out["baselines"] = {"random": MT.random_attribute(bank, ext_ok, db)}
    Xi, Mi = bank.image_raw("clip")
    Xn = Xi / np.linalg.norm(Xi, axis=1, keepdims=True).clip(1e-8)
    out["baselines"]["image_clip_knn"] = MT.attribute_retrieval(Xn[ext_ok], Mi[ext_ok], Xn[db], Mi[db], bank, ext_ok, db)
    # formula closeness of the retrieved recipes (standardised UMF space, fitted on train)
    U, Mu, _ = bank.get("formula", "umf", bank.idx("train"))
    Q, qm = MT.fuse(Zx, MODS, weights)
    S = Q @ G.T
    top5 = np.argsort(-S, 1)[:, :5]
    d_top, d_rand = [], []
    rng = np.random.default_rng(0)
    for qi, row in zip(ext_ok, top5):
        if not Mu[qi]:
            continue
        cand = [db[j] for j in row if Mu[db[j]]]
        rnd = rng.choice(db[Mu[db]], 5, replace=False)
        if cand:
            d_top.append(np.linalg.norm(U[cand] - U[qi], axis=1).mean())
            d_rand.append(np.linalg.norm(U[rnd] - U[qi], axis=1).mean())
    out["formula_distance_top5_vs_random"] = {"top5_mean": round(float(np.mean(d_top)), 3),
                                              "random5_mean": round(float(np.mean(d_rand)), 3), "n": len(d_top)}
    return out


def phase_e(bank, out_dir, final_cfg, weights, seeds=(0, 1, 2)) -> dict:
    p = out_dir / "E_final.json"
    if p.exists():
        return read_json(p)
    tr, va = bank.idx("train"), bank.idx("val")
    per_seed, best_val, best = [], -1, None
    for s in seeds:
        cfg = {**final_cfg, "seed": s}
        model, data, hist, meta = MD.train(bank, cfg, tr, va)
        tops = {}
        ev = evaluate_split(bank, model, data, weights, "test", tops)
        ex = evaluate_external(bank, model, data, weights, tops)
        per_seed.append({"seed": s, "meta": meta, "history": hist, "test": ev, "external": ex})
        if meta["best_val_mean_MRR"] > best_val:
            best_val, best = meta["best_val_mean_MRR"], (s, model, data)
            np.savez_compressed(out_dir / "E_tops.npz", **{f"{k}|{f}": v for k, d in tops.items() for f, v in d.items()})
        log().info(f"[E] seed {s}: val MRR {meta['best_val_mean_MRR']} | test I->T+F "
                   f"{ev['same_recipe_sets'].get('I->T+F', {}).get('R@5')} R@5 | attr Hit@5 "
                   f"{ev['attribute_ablation']['I+T+F']['Hit@5']}")
    # pair-trained models for the training ablation (seed 0)
    pair_models = {}
    for mods in (["text", "image"], ["text", "formula"], ["image", "formula"]):
        cfg = {**final_cfg, "modalities": mods, "seed": 0}
        m, d, h, meta = MD.train(bank, cfg, tr, va)
        Zq = MD.embed(m, d, bank.idx("test"))
        pair_models["+".join(mods)] = {"meta": meta, "pairwise": MT.pairwise_same_recipe(Zq, bank, bank.idx("test"))}
        del m, d
    s, model, data = best
    ckpt = out_dir / "final_model.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": {**final_cfg, "seed": s}, "dims": data.dims,
                "weights": weights, "trained": True, "split": "train (early stop on validation)",
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ckpt)
    # database embeddings of every recipe (for retrieval / HTML)
    allidx = np.arange(bank.n)
    Z = MD.embed(model, data, allidx)
    np.savez_compressed(out_dir / "db_embeddings.npz", ids=np.array(bank.ids),
                        **{f"Z_{m}": Z[m][0] for m in Z}, **{f"M_{m}": Z[m][1] for m in Z})
    res = {"final_cfg": final_cfg, "weights": weights, "selected_seed": s, "per_seed": per_seed,
           "pair_trained": pair_models, "checkpoint": str(ckpt), "disk_GB_work": round(dir_size_gb(out_dir), 3)}
    write_json(p, res)
    return res


def summarise_seeds(final: dict) -> dict:
    """mean +- sd over seeds for the headline numbers."""
    rows = {}
    for s in final["per_seed"]:
        t = s["test"]
        for k, v in t["same_recipe_sets"].items():
            for m in ("R@1", "R@3", "R@5", "R@10", "MRR", "tie_R@5", "tie_MRR"):
                rows.setdefault(f"same:{k}:{m}", []).append(v.get(m))
        for k, v in t["attribute_ablation"].items():
            for m in ("Hit@3", "Hit@5", "Hit@10", "P@5", "nDCG@10", "MAP@10", "meanDeltaE@5"):
                rows.setdefault(f"attr:{k}:{m}", []).append(v.get(m))
        for k, v in t["pairwise"].items():
            for m in ("R@1", "R@5", "R@10", "MRR"):
                rows.setdefault(f"pair:{k}:{m}", []).append(v.get(m))
    return {k: {"mean": round(float(np.mean([x for x in v if x is not None])), 4),
                "sd": round(float(np.std([x for x in v if x is not None])), 4)} for k, v in rows.items()
            if any(x is not None for x in v)}
