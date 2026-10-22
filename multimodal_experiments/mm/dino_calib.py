"""Grounding DINO -> six object attributes, tuned and validated against Claude's visual labels (attributes_v2).

The Claude-labelled sample (mm/claude_labels.py) is split by the project's recipe-group split: VALIDATION objects are
used for every choice (prompt variant, score aggregation, model, threshold), TEST objects are scored once at the end.
Rounds, each one recorded with validation and test numbers:

  R1 raw DINO        first prompt set ("mixed"), class score = max phrase score, answer = argmax (no abstention)
  R2 prompt choice   best prompt variant (mixed / names / cues) x aggregation (max / mean) chosen on validation
  R3 DINO metric     learned weighting of the DINO evidence (score, box cover, box count of every phrase) of one prompt
                     set or of all prompt sets: L2 logistic regression, 5-fold out-of-fold predictions on validation
  R4 DINO + image    R3 features + the v1 attribute heads / zero-shot probabilities (stacking)
  R5 (form only)     R4 + form words found in the recipe name / notes ("mug", "bowl", "tile" ...)
Among R3-R5 the best out-of-fold validation score wins; a cheaper option (fewer prompt sets = less GPU time on the
full dataset) is preferred when it is within COST_TOL of the best. The winner is made SELECTIVE: its confidence threshold
is the smallest one whose out-of-fold validation precision is >= TARGET (98 %, at least MIN_ANSWERED answers). Below
it the answer is "uncertain" (surface / transparency then fall back to their group when the group model is validated).
Multi-label (effects, defects): per class a "present" threshold (precision >= TARGET) and an "absent" threshold
(negative predictive value >= TARGET); in between the class is "uncertain".
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from . import claude_labels as CL
from . import dino_attr as DA

TARGET = 0.98
COST_TOL = 0.01
MIN_ANSWERED = 8
SINGLE = ["form", "surface_finish", "transparency", "texture", "surface_group", "transparency_group"]
MULTI = ["glaze_effect", "defect"]
GROUPS = {"surface_finish": ("surface_group", CL.SURFACE_GROUP), "transparency": ("transparency_group", CL.TRANSP_GROUP)}
DINO_ATTR = {"surface_group": "surface_finish", "transparency_group": "transparency"}
CLASSES = {"form": list(CL.FORM.values()), "surface_finish": list(CL.SURFACE.values()), "transparency": list(CL.TRANSP.values()),
           "texture": list(CL.TEXTURE.values()), "surface_group": ["glossy", "satin", "matte"], "transparency_group": ["clear", "opaque"],
           "glaze_effect": list(CL.EFFECT.values()), "defect": list(CL.DEFECT.values())}


def wilson_low(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    return round(float((p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d), 4)


# ---------------------------------------------------------------- inputs
def load_dino(path: Path, ids: list[str]) -> dict:
    """DINO evidence of the given objects; round-2 evidence (<name>_r2.npz: embeddings + tiles) is merged when present."""
    D = {}
    for p in (path, path.with_name(path.stem + "_r2.npz")):
        if not p.exists():
            continue
        z = dict(np.load(p))
        pos = {o: k for k, o in enumerate(z["ids"].tolist())}
        idx = np.array([pos[o] for o in ids])
        for k, v in z.items():
            if k != "ids" and k not in D:
                D[k] = v[idx]
    return D


def class_scores(D: dict, attr: str, variant: str, agg: str) -> np.ndarray:
    """[N, C] class scores of one prompt variant (C in CLASSES order of the DINO attribute)."""
    a = DINO_ATTR.get(attr, attr)
    ph = DA.phrase_list(a, variant)
    S = D[f"{a}|{variant}|score"]
    cls = list(DA.bank(a, variant))
    out = np.zeros((S.shape[0], len(cls)), np.float32)
    for j, c in enumerate(cls):
        cols = [k for k, (cc, _) in enumerate(ph) if cc == c]
        out[:, j] = S[:, cols].max(1) if agg == "max" else S[:, cols].mean(1)
    if attr in ("surface_group", "transparency_group"):
        gmap = CL.SURFACE_GROUP if attr == "surface_group" else CL.TRANSP_GROUP
        G = np.zeros((S.shape[0], len(CLASSES[attr])), np.float32)
        for j, g in enumerate(CLASSES[attr]):
            G[:, j] = out[:, [k for k, c in enumerate(cls) if gmap[c] == g]].max(1)
        return G
    return out


def dino_features(D: dict, attr: str, variants=None, emb: bool = False) -> np.ndarray:
    """DINO evidence of an attribute: per phrase of the given prompt sets (default all full-image sets) score, box cover,
    log box count (+ the mean over the tiles for '<set>_tiles'), and optionally the decoder embedding of the best boxes."""
    a = DINO_ATTR.get(attr, attr)
    F = []
    for v in variants or list(DA.VARIANTS[a]):
        F += [D[f"{a}|{v}|score"], D[f"{a}|{v}|cover"], np.log1p(D[f"{a}|{v}|n"].astype(np.float32))]
        if v.endswith("_tiles"):
            F.append(D[f"{a}|{v}|tmean"])
    if emb:
        F.append(D[f"{a}|mixed|emb"])
    return np.concatenate(F, 1).astype(np.float32)


def has_r2(D: dict, attr: str) -> tuple[bool, bool]:
    a = DINO_ATTR.get(attr, attr)
    return f"{a}|mixed|emb" in D, f"{a}|mixed_tiles|score" in D


def cost(vs, emb) -> int:
    """forward passes per object: 1 per full-image prompt set, 4 per tiled set, +1 for an embedding without 'mixed'."""
    return sum(4 if v.endswith("_tiles") else 1 for v in vs) + (1 if emb and "mixed" not in vs else 0)


def variant_sets(D, attr):
    a = DINO_ATTR.get(attr, attr)
    base = [[v] for v in DA.VARIANTS[a]] + [list(DA.VARIANTS[a])]
    e_ok, t_ok = has_r2(D, attr)
    sets = [(vs, False) for vs in base]
    if t_ok:
        sets += [(["mixed_tiles"], False), (["mixed", "mixed_tiles"], False)]
    if e_ok:
        sets += [(vs, True) for vs, _ in list(sets)]
    return sets


def coverage_at_target(P, y):
    """out-of-fold coverage when answering only at >= TARGET precision (0 if the target is never reached)."""
    thr = select_threshold(P.max(1), P.argmax(1) == y)
    return 0.0 if thr is None else float((P.max(1) >= thr).mean())


FORM_WORDS = {"mug": "cup", "cup": "cup", "bowl": "bowl", "plate": "plate", "dish": "plate", "platter": "plate", "vase": "vase",
              "bottle": "vase", "jar": "jar", "teapot": "other vessel", "pitcher": "other vessel", "jug": "other vessel",
              "tile": "test tile", "sculpture": "sculpture", "figure": "figurine", "figurine": "figurine"}


def form_hints(v1: Path, ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """one-hot of form words in the recipe name / notes ([N, C] in CLASSES['form'] order) + readable hint per object.
    The recipe text describes the glaze, not always the photographed object: a hint, weighed by the model."""
    import re
    from .common import read_csv
    data = Path(v1).parents[0] / "data" / "recipes.csv"
    rec = {r["recipe_id"]: (r.get("name", "") + " " + r.get("desc_free", "")).lower() for r in read_csv(data)}
    rid = {}
    with (v1 / "labels.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rid[r["object_id"]] = r["recipe_id"]
    cls = CLASSES["form"]
    H = np.zeros((len(ids), len(cls)), np.float32)
    txt = []
    for k, o in enumerate(ids):
        t = rec.get(rid.get(o, ""), "")
        found = [w for w in FORM_WORDS if re.search(r"\b" + w + r"s?\b", t)]
        for w in found:
            H[k, cls.index(FORM_WORDS[w])] = 1
        txt.append(", ".join(f"'{w}'" for w in found))
    return H, txt


def v1_probs(v1: Path, ids: list[str]) -> dict:
    """full probability vectors of the v1 heads (+ zero-shot form / texture) for the given objects, in CLASSES order."""
    import joblib
    from .attr_models import TASKS, feature_sets, zero_shot_scores
    fs = feature_sets(v1, ids)
    out = {}
    for p in sorted((v1 / "heads").glob("*.joblib")):
        task = p.stem
        h = joblib.load(p)
        X, M = fs[h["features"]]
        if h["type"] == "zero_shot":
            P = zero_shot_scores(X, h["E"], single=h["kind"] == "single")
            order = [h["classes"].index(c) for c in TASKS[task][2]] if set(TASKS[task][2]) <= set(h["classes"]) else range(len(h["classes"]))
            P = P[:, list(order)]
        elif h["type"] in ("sklearn", "linear"):
            P = h["model"].predict_proba(X)
            if h["kind"] == "single" and P.shape[1] != len(h["classes"]):
                full = np.zeros((len(X), len(h["classes"])))
                full[:, h["model"].classes_] = P
                P = full
        else:
            import torch
            with torch.no_grad():
                o = h["net"](torch.tensor(X))
                P = (torch.softmax(o, 1) if h["kind"] == "single" else torch.sigmoid(o)).numpy()
        names = TASKS[task][2]
        Q = np.zeros((len(ids), len(CLASSES[task])), np.float32)
        for j, c in enumerate(CLASSES[task]):
            if c in names:
                Q[:, j] = P[:, names.index(c)]
        Q[~M] = np.nan
        out[task] = Q
    zt = np.load(v1 / "zeroshot_text.npz")
    Xc, Mc = fs["clip"]
    for field in ("form", "texture"):
        cls = zt[field + "__classes"].tolist()
        S = zero_shot_scores(Xc, zt[field + "__E"], single=True)
        Q = np.zeros((len(ids), len(CLASSES[field])), np.float32)
        for j, c in enumerate(CLASSES[field]):
            if c in cls:
                Q[:, j] = S[:, cls.index(c)]
        Q[~Mc] = np.nan
        out[field] = Q
    return out


def labels(lab: dict, ids: list[str], attr: str):
    cls = CLASSES[attr]
    if attr in MULTI:
        Y = np.zeros((len(ids), len(cls)), np.float32)
        has = np.zeros(len(ids), bool)
        for k, o in enumerate(ids):
            v = lab[o][attr]
            if v is not None:
                has[k] = True
                for c in v:
                    Y[k, cls.index(c)] = 1
        return Y, has
    y = np.array([cls.index(lab[o][attr]) if lab[o][attr] in cls else -1 for o in ids])
    return y, y >= 0


# ---------------------------------------------------------------- models
def _lr(C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=3000)


def oof_single(X, y, C, folds=5, seed=0):
    """out-of-fold class probabilities on the validation objects (classes absent from a fold get 0)."""
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler
    n_cls = int(y.max()) + 1
    P = np.zeros((len(y), n_cls), np.float32)
    counts = np.bincount(y, minlength=n_cls)
    use_strat = counts[counts > 0].min() >= folds
    kf = (StratifiedKFold(folds, shuffle=True, random_state=seed) if use_strat else KFold(folds, shuffle=True, random_state=seed))
    for tr, te in kf.split(X, y if use_strat else None):
        if len(np.unique(y[tr])) < 2:
            P[te, y[tr][0]] = 1
            continue
        sc = StandardScaler().fit(X[tr])
        m = _lr(C).fit(sc.transform(X[tr]), y[tr])
        P[np.ix_(te, m.classes_)] = m.predict_proba(sc.transform(X[te]))
    return P


def fit_single(X, y, C):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    return sc, _lr(C).fit(sc.transform(X), y)


def predict_single(model, X, n_cls):
    sc, m = model
    P = np.zeros((len(X), n_cls), np.float32)
    P[:, m.classes_] = m.predict_proba(sc.transform(X))
    return P


def oof_binary(X, y, C, folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y), np.float32)
    k = int(min(folds, y.sum(), (1 - y).sum()))
    if k < 2:
        return None
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = _lr(C).fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def select_threshold(conf, correct, target=TARGET, min_n=MIN_ANSWERED):
    """smallest confidence threshold whose answered precision >= target (None if never). The precision is Laplace-
    smoothed, (correct + 1) / (answered + 2): with ~230 validation objects the plain precision of a small answered set
    is too optimistic (round-1 check: thresholds reaching 98 % on validation gave 77-96 % on test). With no error,
    the smoothed rule needs >= 48 answers for 98 %."""
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    cum = (np.cumsum(c) + 1) / (np.arange(1, len(c) + 1) + 2)
    best = None
    for i in range(min_n - 1, len(c)):
        if cum[i] >= target:
            best = float(conf[order][i])
    return best


def single_report(y, P, thr):
    pred = P.argmax(1)
    conf = P.max(1)
    ok = pred == y
    ans = conf >= thr if thr is not None else np.zeros(len(y), bool)
    return {"n": int(len(y)), "accuracy_forced": round(float(ok.mean()), 4) if len(y) else None,
            "answered": int(ans.sum()), "coverage": round(float(ans.mean()), 4) if len(y) else None,
            "answered_precision": round(float(ok[ans].mean()), 4) if ans.any() else None,
            "answered_errors": int((~ok[ans]).sum()), "wilson95_low": wilson_low(int(ok[ans].sum()), int(ans.sum()))}


# ---------------------------------------------------------------- calibration
def calibrate(v1: Path, v2: Path) -> dict:
    lab = CL.parse(v2 / "claude_labels_raw.txt")
    split = {}
    with (v1 / "labels.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            split[r["object_id"]] = r["split"]
    ids = [o for o in sorted(lab) if split.get(o) in ("val", "test")]
    sp = np.array([split[o] for o in ids])
    D = load_dino(v2 / "dino_sample.npz", ids)
    V = v1_probs(v1, ids)
    FH, _ = form_hints(v1, ids)
    out = {"objects": {"val": int((sp == "val").sum()), "test": int((sp == "test").sum())}, "target_precision": TARGET,
           "label_source": CL.PROVENANCE, "attributes": {}, "models": {}}
    models = {}
    for attr in SINGLE + MULTI:
        a = DINO_ATTR.get(attr, attr)
        y, has = labels(lab, ids, attr)
        va, te = (sp == "val") & has, (sp == "test") & has
        n_cls = len(CLASSES[attr])
        vkey = attr if attr in V else None
        Fv = np.nan_to_num(V[vkey]) if vkey else None
        rounds = []
        if attr in SINGLE:
            def acc(P, m):
                return round(float((P[m].argmax(1) == y[m]).mean()), 4) if m.any() else None
            R1 = class_scores(D, attr, "mixed", "max")
            rounds.append({"round": "R1 raw DINO (mixed prompts, max)", "val": acc(R1, va), "test": acc(R1, te)})
            best = None
            for v in DA.VARIANTS[a]:
                for agg in ("max", "mean"):
                    S = class_scores(D, attr, v, agg)
                    s = acc(S, va)
                    if best is None or s > best[0]:
                        best = (s, v, agg, S)
            rounds.append({"round": f"R2 prompt choice ({best[1]}, {best[2]})", "val": best[0], "test": acc(best[3], te)})
            cand = {}
            options = []
            for vs, use_emb in variant_sets(D, attr):
                Xd = dino_features(D, attr, vs, emb=use_emb)
                tag = ("all full-image sets" if vs == list(DA.VARIANTS[a]) else " + ".join(vs)) + (" + embeddings" if use_emb else "")
                rnd = "R6" if use_emb or any(v.endswith("_tiles") for v in vs) else "R3"
                options.append((f"{rnd} DINO metric [{tag}]", Xd, vs, False, False, use_emb))
                if Fv is not None:
                    options.append((f"{'R6' if rnd == 'R6' else 'R4'} DINO [{tag}] + image heads", np.concatenate([Xd, Fv], 1), vs, True, False, use_emb))
                    if attr == "form":
                        options.append((f"{'R6' if rnd == 'R6' else 'R5'} DINO [{tag}] + image heads + recipe text", np.concatenate([Xd, Fv, FH], 1), vs, True, True, use_emb))
            if Fv is not None:
                options.append(("R0 image heads only (v1, reference)", Fv, [], True, False, False))
            for name, X, vs, use_v1, use_txt, use_emb in options:
                bestC = None
                for C in (0.03, 0.1, 0.3, 1.0):
                    P = oof_single(X[va], y[va], C)
                    Pf = np.zeros((va.sum(), n_cls), np.float32)
                    Pf[:, :P.shape[1]] = P
                    key_ = (coverage_at_target(Pf, y[va]), float((P.argmax(1) == y[va]).mean()))
                    if bestC is None or key_ > bestC[0]:
                        bestC = (key_, C, Pf)
                mdl = fit_single(X[va], y[va], bestC[1])
                Pte = predict_single(mdl, X[te], n_cls)
                cand[name] = (bestC[0][1], bestC[1], bestC[2], Pte, mdl, X, vs, use_v1, use_txt, use_emb, bestC[0][0])
                rounds.append({"round": name, "val": round(bestC[0][1], 4), "val_coverage_at_target": round(bestC[0][0], 4),
                               "test": round(float((Pte.argmax(1) == y[te]).mean()), 4), "C": bestC[1], "passes_per_object": cost(vs, use_emb)})
            # selection: largest out-of-fold coverage at the target; within COST_TOL the cheaper, then the more accurate
            top = max(cand[k][10] for k in cand if not k.startswith("R0"))
            sel = min((k for k in cand if not k.startswith("R0") and cand[k][10] >= top - COST_TOL),
                      key=lambda k: (cost(cand[k][6], cand[k][9]), -cand[k][0]))
            s_, C, Pva, Pte, mdl, X, vs, use_v1, use_txt, use_emb, cov_ = cand[sel]
            thr = select_threshold(Pva.max(1), Pva.argmax(1) == y[va])
            res = {"rounds": rounds, "selected": sel, "C": C, "threshold": thr,
                   "val_oof": single_report(y[va], Pva, thr), "test": single_report(y[te], Pte, thr),
                   "classes": CLASSES[attr]}
            # per class on test (forced answer)
            res["test_per_class"] = {CLASSES[attr][c]: {"n": int((y[te] == c).sum()),
                                                        "recall": round(float((Pte.argmax(1)[y[te] == c] == c).mean()), 3) if (y[te] == c).any() else None}
                                     for c in range(n_cls)}
            models[attr] = {"kind": "single", "variants": vs, "emb": use_emb, "v1": use_v1, "recipe_text": use_txt, "model": mdl,
                            "threshold": thr, "classes": CLASSES[attr], "selected": sel}
        else:
            per = {}
            Pva_all, Pte_all = np.zeros((va.sum(), n_cls)), np.zeros((te.sum(), n_cls))
            mdl_cls = {}
            reported = [j for j in range(n_cls) if y[va][:, j].sum() >= 4]
            options = []
            for vs, use_emb in variant_sets(D, attr):
                Xd = dino_features(D, attr, vs, emb=use_emb)
                tag = ("all full-image sets" if vs == list(DA.VARIANTS[a]) else " + ".join(vs)) + (" + embeddings" if use_emb else "")
                options.append((f"DINO [{tag}]", Xd, vs, False, use_emb))
                if Fv is not None:
                    options.append((f"DINO [{tag}] + image heads", np.concatenate([Xd, Fv], 1), vs, True, use_emb))
            scored = []
            for name, X, vs, use_v1, use_emb in options:
                aps, keep = [], {}
                for j in reported:
                    yv = y[va][:, j]
                    best = None
                    for C in (0.03, 0.1, 0.3):
                        p_ = oof_binary(X[va], yv, C)
                        if p_ is None:
                            continue
                        ap_ = average_precision_score(yv, p_)
                        if best is None or ap_ > best[0]:
                            best = (ap_, C, p_)
                    if best:
                        aps.append(best[0])
                        keep[j] = best
                scored.append((float(np.mean(aps)) if aps else 0.0, name, X, vs, use_v1, keep, use_emb))
                rounds.append({"round": f"{name}: mean OOF AP over reported classes", "val": round(scored[-1][0], 4),
                               "passes_per_object": cost(vs, use_emb)})
            top = max(sc_[0] for sc_ in scored)
            mAP_, sel, X, vs, use_v1, keep, use_emb = min((sc_ for sc_ in scored if sc_[0] >= top - COST_TOL),
                                                           key=lambda sc_: (cost(sc_[3], sc_[6]), -sc_[0]))
            from sklearn.preprocessing import StandardScaler
            for j, c in enumerate(CLASSES[attr]):
                yv, yt = y[va][:, j], y[te][:, j]
                info = {"val_pos": int(yv.sum()), "test_pos": int(yt.sum())}
                if j not in keep:
                    info["status"] = "not reported (fewer than 4 validation positives)"
                    per[c] = info
                    continue
                ap, C, p = keep[j]
                sc = StandardScaler().fit(X[va])
                m = _lr(C).fit(sc.transform(X[va]), yv)
                pt = m.predict_proba(sc.transform(X[te]))[:, 1]
                hi = select_threshold(p, yv == 1)                      # present
                lo = select_threshold(1 - p, yv == 0)                  # absent (score <= 1 - lo)
                lo = None if lo is None else 1 - lo
                def rep(prob, yy):
                    pres = prob >= hi if hi is not None else np.zeros(len(yy), bool)
                    absn = prob <= lo if lo is not None else np.zeros(len(yy), bool)
                    return {"present_answers": int(pres.sum()), "present_precision": round(float(yy[pres].mean()), 3) if pres.any() else None,
                            "absent_answers": int(absn.sum()), "absent_npv": round(float(1 - yy[absn].mean()), 3) if absn.any() else None,
                            "uncertain": int((~pres & ~absn).sum()),
                            "AP": round(float(average_precision_score(yy, prob)), 3) if 0 < yy.sum() < len(yy) else None}
                info.update({"C": C, "val_AP_oof": round(float(ap), 3), "present_threshold": hi, "absent_threshold": lo,
                             "val_oof": rep(p, yv), "test": rep(pt, yt)})
                per[c] = info
                mdl_cls[c] = {"model": (sc, m), "present": hi, "absent": lo}
                Pva_all[:, j], Pte_all[:, j] = p, pt
            # raw DINO reference (R1): max phrase score per class, thresholds tuned on validation for F1
            R1 = class_scores(D, attr, "mixed", "max")
            res = {"per_class": per, "classes": CLASSES[attr], "rounds": [{"round": "R1 raw DINO (mixed, max) mAP",
                   "val": _map(y[va], R1[va]), "test": _map(y[te], R1[te])},
                   {"round": f"selected: {sel} - mAP", "val": _map(y[va], Pva_all), "test": _map(y[te], Pte_all)}] + rounds,
                   "selected": sel}
            # overall answered precision over all reported classes on test
            tp = fp = tn = fn = 0
            for c, info in per.items():
                if "test" in info:
                    t = info["test"]
                    k = info["test_pos"]
                    if t["present_answers"]:
                        tp += round(t["present_answers"] * t["present_precision"])
                        fp += t["present_answers"] - round(t["present_answers"] * t["present_precision"])
                    if t["absent_answers"]:
                        tn += round(t["absent_answers"] * t["absent_npv"])
                        fn += t["absent_answers"] - round(t["absent_answers"] * t["absent_npv"])
            n_ans = tp + fp + tn + fn
            res["test"] = {"answered_precision": round((tp + tn) / n_ans, 4) if n_ans else None, "answered": n_ans,
                           "coverage": round(n_ans / max(1, te.sum() * len(CLASSES[attr])), 4),
                           "wilson95_low": wilson_low(tp + tn, n_ans), "present_tp": tp, "present_fp": fp, "absent_tn": tn, "absent_fn": fn}
            models[attr] = {"kind": "multi", "classes": CLASSES[attr], "per_class": mdl_cls, "variants": vs, "emb": use_emb, "v1": use_v1,
                            "recipe_text": False, "selected": sel}
        out["attributes"][attr] = res
    import joblib
    joblib.dump(models, v2 / "dino_heads.joblib")
    out["models"] = {a: {k: v for k, v in m.items() if k not in ("model", "per_class")} for a, m in models.items()}
    out["prompt_sets_needed"] = sorted({(DINO_ATTR.get(a, a), v) for a, m in models.items() for v in m["variants"]})
    out["dino_spec"] = {"keys": sorted({(DINO_ATTR.get(a, a), v) for a, m in models.items() for v in m["variants"] if not v.endswith("_tiles")}),
                        "tiles": sorted({(DINO_ATTR.get(a, a), v[:-6]) for a, m in models.items() for v in m["variants"] if v.endswith("_tiles")}),
                        "emb": sorted({(DINO_ATTR.get(a, a), "mixed") for a, m in models.items() if m.get("emb")})}
    sp = out["dino_spec"]
    out["passes_per_object"] = len(set(map(tuple, sp["keys"])) | set(map(tuple, sp["emb"]))) + 4 * len(sp["tiles"])
    return out


def _map(Y, S):
    aps = [average_precision_score(Y[:, j], S[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    return round(float(np.mean(aps)), 4) if aps else None


def read_cal(v2: Path) -> dict:
    return json.loads((v2 / "dino_calibration.json").read_text(encoding="utf-8"))
