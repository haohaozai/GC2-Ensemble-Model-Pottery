"""Attributes v3: the six object attributes from Grounding DINO evidence, TRAINED on Claude's visual labels, with the
three most likely candidates per attribute.

Data (provenance claude_visual - Claude's labels from the object cut-outs, NOT human-verified):
  train  1,504 TRAIN objects labelled for v3 (work/attributes_v3/claude_labels_train_raw.txt)
  val      241 VALIDATION objects labelled for v2 (work/attributes_v2/claude_labels_raw.txt)
  test     240 TEST objects labelled for v2 - scored once, never used for a choice
The split is the project's recipe-group split (no recipe appears in two splits).

Features (all computed for every object, so the trained models can label the whole dataset):
  Grounding DINO evidence of the full run (work/attributes_v2/dino_all.npz): per phrase the best grounding score, box
  cover and box count, zoomed-tile scores for effects / defects, decoder embedding for form; + the earlier image
  heads' class probabilities (v1); + form words of the recipe text (form only).
  Grounding DINO itself stays pretrained (IDEA-Research/grounding-dino-base): what is trained here is the layer that
  turns its evidence into the six attributes (L2 logistic regression).

Training: fitted on TRAIN; the feature set and C are chosen by validation log-loss (good probabilities = good top-3
ranking and good soft search); the "answered" threshold is the smallest one with >= 98 % Laplace-smoothed precision on
VALIDATION predictions (out-of-sample); TEST is scored once at the end.

Top-3 candidates (per attribute and object): the classes ranked by the trained probability, kept when p >= 0.05 and
p >= 0.10 x the best one (never forced to three); each with its raw Grounding DINO class score (max phrase score) and the
box of its strongest phrase. Glaze effects / defects are multi-label: every "present" class is kept even beyond three.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from . import claude_labels as CL
from . import dino_attr as DA
from . import dino_calib as DC

FIELDS = ["form", "surface_finish", "transparency", "texture", "glaze_effect", "defect"]
GROUPS = ["surface_group", "transparency_group"]
GROUP_OF = {"surface_finish": "surface_group", "transparency": "transparency_group"}
TARGET = DC.TARGET
CAND_MIN, CAND_REL, TOPK = 0.05, 0.10, 3
CS = (0.01, 0.03, 0.1, 0.3, 1.0)
# prompt sets present in the full run (dino_all.npz) per DINO attribute - only these can be used for every object
FULL_RUN = {"form": (["mixed"], True), "surface_finish": (["cues", "names"], False), "transparency": (["mixed"], False),
            "texture": (["names"], False), "glaze_effect": (["mixed_tiles"], False), "defect": (["mixed", "mixed_tiles"], False)}
PROVENANCE = "claude_visual (Claude, from the object image only, guide V4; not human-verified)"


# ---------------------------------------------------------------- data
def load_labels(v1: Path, v2: Path, v3: Path) -> tuple[dict, dict]:
    split = {}
    with (v1 / "labels.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            split[r["object_id"]] = r["split"]
    lab = {}
    for p, want in ((v3 / "claude_labels_train_raw.txt", ("train",)), (v2 / "claude_labels_raw.txt", ("val", "test"))):
        for o, r in CL.parse(p).items():
            if split.get(o) in want:
                lab[o] = r
    return lab, split


def features(attr: str, D: dict, V: dict, FH: np.ndarray, use: str) -> np.ndarray:
    """use: 'dino' | 'dino+v1' | 'dino+v1+text' | 'v1'"""
    a = DC.DINO_ATTR.get(attr, attr)
    vs, emb = FULL_RUN[a]
    X = []
    if use.startswith("dino"):
        X.append(DC.dino_features(D, attr, vs, emb=emb))
    if "v1" in use and attr in V:
        X.append(np.nan_to_num(V[attr]))
    if use.endswith("text"):
        X.append(FH)
    return np.concatenate(X, 1).astype(np.float32)


def options(attr: str, V: dict) -> list[str]:
    o = ["dino"]
    if attr in V:
        o += ["dino+v1", "v1"]
        if attr == "form":
            o.append("dino+v1+text")
    return o


def _logloss(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))))


def recall_at(P, y, ks=(1, 2, 3)):
    order = np.argsort(-P, 1)
    return {f"R@{k}": round(float(np.mean([y[i] in order[i, :k] for i in range(len(y))])), 4) for k in ks}


def multi_recall_at(P, Y, ks=(1, 2, 3)):
    """multi-label Recall@K: over objects with >= 1 reference label, mean share of the reference labels found in the K
    highest-probability classes."""
    m = Y.sum(1) > 0
    order = np.argsort(-P[m], 1)
    Ym = Y[m]
    out = {}
    for k in ks:
        out[f"R@{k}"] = round(float(np.mean([Ym[i, order[i, :k]].sum() / Ym[i].sum() for i in range(len(Ym))])), 4) if m.any() else None
    out["n_with_labels"] = int(m.sum())
    out["n_none"] = int((~m).sum())
    return out


# ---------------------------------------------------------------- training
def train(v1: Path, v2: Path, v3: Path) -> tuple[dict, dict]:
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    lab, split = load_labels(v1, v2, v3)
    ids = sorted(lab)
    sp = np.array([split[o] for o in ids])
    D = DC.load_dino(v2 / "dino_all.npz", ids)
    V = DC.v1_probs(v1, ids)
    FH, _ = DC.form_hints(v1, ids)
    models, report = {}, {"objects": {s: int((sp == s).sum()) for s in ("train", "val", "test")}, "label_source": PROVENANCE,
                          "target_precision": TARGET, "attributes": {}}
    v2_heads = None
    try:
        import joblib
        v2_heads = joblib.load(v2 / "dino_heads.joblib")
    except Exception:  # noqa: BLE001
        pass
    for attr in FIELDS + GROUPS:
        cls = DC.CLASSES[attr]
        y, has = DC.labels(lab, ids, attr)
        tr, va, te = (sp == "train") & has, (sp == "val") & has, (sp == "test") & has
        res = {"classes": cls, "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}, "options": []}
        if attr not in DC.MULTI:
            best = None
            for use in options(attr, V):
                X = features(attr, D, V, FH, use)
                for C in CS:
                    sc = StandardScaler().fit(X[tr])
                    m = DC._lr(C).fit(sc.transform(X[tr]), y[tr])
                    Pv = DC.predict_single((sc, m), X[va], len(cls))
                    ll = _logloss(Pv, y[va])
                    res["options"].append({"features": use, "C": C, "val_logloss": round(ll, 4),
                                           "val_acc": round(float((Pv.argmax(1) == y[va]).mean()), 4), **recall_at(Pv, y[va])})
                    if best is None or ll < best[0]:
                        best = (ll, use, C, (sc, m), Pv, X)
            ll, use, C, mdl, Pv, X = best
            thr = DC.select_threshold(Pv.max(1), Pv.argmax(1) == y[va])
            Pt = DC.predict_single(mdl, X[te], len(cls))
            res.update({"selected": {"features": use, "C": C, "val_logloss": round(ll, 4)}, "threshold": thr,
                        "val": {**DC.single_report(y[va], Pv, thr), **recall_at(Pv, y[va])},
                        "test": {**DC.single_report(y[te], Pt, thr), **recall_at(Pt, y[te])},
                        "test_per_class": {cls[c]: {"n": int((y[te] == c).sum()),
                                                    "recall@1": round(float((Pt.argmax(1)[y[te] == c] == c).mean()), 3) if (y[te] == c).any() else None,
                                                    "recall@3": round(float(np.mean([c in np.argsort(-Pt[i])[:3] for i in np.nonzero(y[te] == c)[0]])), 3) if (y[te] == c).any() else None}
                                           for c in range(len(cls))}})
            # baselines on the same test objects: class frequency of TRAIN, and the v2 model (trained on the 241 val objects)
            freq = np.bincount(y[tr], minlength=len(cls)).astype(np.float32)
            res["test_baseline_frequency"] = recall_at(np.tile(freq, (te.sum(), 1)), y[te])
            if v2_heads and attr in v2_heads:
                res["test_v2"] = _v2_single(v2_heads[attr], attr, D, V, FH, te, y)
            models[attr] = {"kind": "single", "features": use, "model": mdl, "threshold": thr, "classes": cls}
        else:
            per, Pv_all, Pt_all = {}, np.zeros((va.sum(), len(cls))), np.zeros((te.sum(), len(cls)))
            best_use = None
            for use in options(attr, V):
                if use == "v1":
                    continue
                X = features(attr, D, V, FH, use)
                aps = []
                for j in range(len(cls)):
                    if y[tr][:, j].sum() < 5 or y[va][:, j].sum() < 1:
                        continue
                    sc = StandardScaler().fit(X[tr])
                    m = DC._lr(0.1).fit(sc.transform(X[tr]), y[tr][:, j])
                    aps.append(average_precision_score(y[va][:, j], m.predict_proba(sc.transform(X[va]))[:, 1]))
                score = float(np.mean(aps)) if aps else 0.0
                res["options"].append({"features": use, "C": 0.1, "val_mAP": round(score, 4)})
                if best_use is None or score > best_use[0]:
                    best_use = (score, use, X)
            _, use, X = best_use
            mdl_cls = {}
            for j, c in enumerate(cls):
                ytr, yv, yt = y[tr][:, j], y[va][:, j], y[te][:, j]
                info = {"train_pos": int(ytr.sum()), "val_pos": int(yv.sum()), "test_pos": int(yt.sum())}
                if ytr.sum() < 5:
                    info["status"] = "not modelled (fewer than 5 training positives)"
                    per[c] = info
                    continue
                bestc = None
                for C in CS:
                    sc = StandardScaler().fit(X[tr])
                    m = DC._lr(C).fit(sc.transform(X[tr]), ytr)
                    pv = m.predict_proba(sc.transform(X[va]))[:, 1]
                    key = average_precision_score(yv, pv) if 0 < yv.sum() < len(yv) else -_bce(pv, yv)
                    if bestc is None or key > bestc[0]:
                        bestc = (key, C, (sc, m), pv)
                _, C, (sc, m), pv = bestc
                pt = m.predict_proba(sc.transform(X[te]))[:, 1]
                hi = DC.select_threshold(pv, yv == 1)
                lo = DC.select_threshold(1 - pv, yv == 0)
                lo = None if lo is None else 1 - lo
                info.update({"C": C, "present_threshold": hi, "absent_threshold": lo,
                             "val_AP": round(float(average_precision_score(yv, pv)), 3) if 0 < yv.sum() < len(yv) else None,
                             "test_AP": round(float(average_precision_score(yt, pt)), 3) if 0 < yt.sum() < len(yt) else None,
                             "test": _multi_rep(pt, yt, hi, lo)})
                per[c] = info
                mdl_cls[c] = {"model": (sc, m), "present": hi, "absent": lo}
                Pv_all[:, j], Pt_all[:, j] = pv, pt
            freq = y[tr].mean(0)
            res.update({"selected": {"features": use}, "per_class": per,
                        "val": multi_recall_at(Pv_all, y[va]), "test": multi_recall_at(Pt_all, y[te]),
                        "test_baseline_frequency": multi_recall_at(np.tile(freq, (te.sum(), 1)), y[te]),
                        "test_mAP": DC._map(y[te], Pt_all)})
            models[attr] = {"kind": "multi", "features": use, "per_class": mdl_cls, "classes": cls}
        report["attributes"][attr] = res
    report["models"] = {a: {"kind": m["kind"], "features": m["features"], "threshold": m.get("threshold"),
                            "classes_modelled": sorted(m["per_class"]) if m["kind"] == "multi" else m["classes"]}
                        for a, m in models.items()}
    return models, report


def _bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _multi_rep(prob, yy, hi, lo):
    pres = prob >= hi if hi is not None else np.zeros(len(yy), bool)
    absn = prob <= lo if lo is not None else np.zeros(len(yy), bool)
    return {"present_answers": int(pres.sum()), "present_precision": round(float(yy[pres].mean()), 3) if pres.any() else None,
            "absent_answers": int(absn.sum()), "absent_npv": round(float(1 - yy[absn].mean()), 3) if absn.any() else None}


def _v2_single(m2: dict, attr: str, D: dict, V: dict, FH: np.ndarray, te: np.ndarray, y: np.ndarray) -> dict | None:
    """the v2 model (fitted on the 241 validation objects) on the same test objects, when its features exist in the full run."""
    try:
        from .dino_predict import _features
        X = _features(m2, attr, D, V, FH)
        P = DC.predict_single(m2["model"], X[te], len(m2["classes"]))
        return {"acc": round(float((P.argmax(1) == y[te]).mean()), 4), **recall_at(P, y[te])}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:120]}


# ---------------------------------------------------------------- labelling (dataset and query)
def _class_evidence(D: dict, attr: str, k: int, offset) -> dict:
    """class -> (raw DINO class score = max phrase score, phrase, box in object-image pixels) over the full-run prompt sets."""
    a = DC.DINO_ATTR.get(attr, attr)
    vs, _ = FULL_RUN[a]
    out = {}
    for v in vs:
        ph = DA.phrase_list(a, v)
        S, B = D[f"{a}|{v}|score"][k], D[f"{a}|{v}|box"][k]
        for j, (c, phrase) in enumerate(ph):
            s = float(S[j])
            if c not in out or s > out[c][0]:
                b = B[j]
                out[c] = (s, phrase, [round(float(b[0] + offset[0]), 1), round(float(b[1] + offset[1]), 1),
                                      round(float(b[2] + offset[0]), 1), round(float(b[3] + offset[1]), 1)])
    return out


def _candidates(probs: dict, ev: dict, keep=()) -> list[dict]:
    ranked = sorted(probs.items(), key=lambda t: -t[1])
    top = ranked[0][1] if ranked else 0
    out = []
    for c, p in ranked:
        if len(out) < TOPK and (not out or (p >= CAND_MIN and p >= CAND_REL * top)) or c in keep:
            e = ev.get(c)
            out.append({"rank": len(out) + 1, "label": c, "p": round(float(p), 3),
                        "dino": round(e[0], 3) if e else None, "phrase": e[1] if e else None, "box": e[2] if e else None})
    return out


def label_rows(models: dict, D: dict, V: dict, FH: np.ndarray, hints: list[str]) -> list[dict]:
    n = len(D["offset"])
    rows = [{} for _ in range(n)]
    for attr in FIELDS + GROUPS:
        m = models[attr]
        X = features(attr, D, V, FH, m["features"])
        if m["kind"] == "single":
            P = DC.predict_single(m["model"], X, len(m["classes"]))
            thr = m["threshold"]
            for k in range(n):
                j = int(P[k].argmax())
                conf = float(P[k, j])
                rec = {"value": m["classes"][j], "status": "answered" if thr is not None and conf >= thr else "uncertain",
                       "confidence": round(conf, 3), "probs": [round(float(x), 3) for x in P[k]]}
                if attr in FIELDS:
                    ev = _class_evidence(D, attr, k, D["offset"][k])
                    rec["candidates"] = _candidates(dict(zip(m["classes"], P[k])), ev)
                    rec["top"] = [[c["label"], c["p"]] for c in rec["candidates"]]
                    rec["evidence"] = [[c["phrase"], c["dino"], c["box"]] for c in rec["candidates"] if c["box"]]
                    if attr == "form" and hints[k]:
                        rec["hint"] = hints[k]
                rows[k][attr] = rec
        else:
            per = m["per_class"]
            probs = {c: per[c]["model"][1].predict_proba(per[c]["model"][0].transform(X))[:, 1] for c in per}
            for k in range(n):
                pres, absn, unc = [], [], []
                pk = {c: float(probs[c][k]) for c in per}
                for c, p in pk.items():
                    if per[c]["present"] is not None and p >= per[c]["present"]:
                        pres.append(c)
                    elif per[c]["absent"] is not None and p <= per[c]["absent"]:
                        absn.append(c)
                    else:
                        unc.append(c)
                ev = _class_evidence(D, attr, k, D["offset"][k])
                cand = _candidates(pk, ev, keep=set(pres))
                conf = [pk[c] for c in pres] + [1 - pk[c] for c in absn]
                rows[k][attr] = {"value": pres, "present": pres, "absent": absn, "uncertain": unc,
                                 "not_checked": [c for c in m["classes"] if c not in per],
                                 "status": "answered" if not unc else "uncertain", "candidates": cand,
                                 "top": [[c["label"], c["p"]] for c in cand], "probs": {c: round(p, 3) for c, p in pk.items()},
                                 "confidence": round(min(conf), 3) if conf else 0.0,
                                 "evidence": [[c["phrase"], c["dino"], c["box"]] for c in cand if c["box"]]}
    for r in rows:
        for fine, grp in GROUP_OF.items():
            g = r.pop(grp)
            r[fine]["group"] = {"value": g["value"], "status": g["status"], "confidence": g["confidence"]}
    return rows


def predict_all(v1: Path, v2: Path, v3: Path, models: dict, report: dict) -> dict:
    t0 = time.time()
    src = v2 / "dino_all.npz"
    ids = np.load(src)["ids"].tolist()
    D = DC.load_dino(src, ids)
    V = DC.v1_probs(v1, ids)
    FH, hints = DC.form_hints(v1, ids)
    rows = label_rows(models, D, V, FH, hints)
    out = dict(zip(ids, rows))
    lab, split = load_labels(v1, v2, v3)
    for oid, r in lab.items():
        if oid in out:
            out[oid]["claude"] = {f: r[f] for f in FIELDS}
            out[oid]["claude_split"] = split[oid]
    val = validation_summary(report)
    prio = {}
    for oid, o in out.items():
        n = sum(1 for f in FIELDS if o[f]["status"] != "answered" and (val.get(f) or {}).get("meets_target"))
        if n:
            prio[oid] = n
    return {"version": "attributes_v3", "created": time.strftime("%Y-%m-%d %H:%M"), "source": src.name,
            "label_source_for_training": PROVENANCE, "target_precision": TARGET, "validation": val,
            "candidate_rule": f"top {TOPK} by trained probability, p >= {CAND_MIN} and >= {CAND_REL} x best; effects / defects: + every present class",
            "review_priority": prio, "objects": out, "seconds": round(time.time() - t0, 1)}


def validation_summary(report: dict, min_answered: int = 20) -> dict:
    out = {}
    for attr, r in report["attributes"].items():
        t = r.get("test") or {}
        if attr in DC.MULTI:
            tp = fp = tn = fn = 0
            for c, info in r["per_class"].items():
                tt = info.get("test") or {}
                if tt.get("present_answers"):
                    k = round(tt["present_answers"] * tt["present_precision"])
                    tp, fp = tp + k, fp + tt["present_answers"] - k
                if tt.get("absent_answers"):
                    k = round(tt["absent_answers"] * tt["absent_npv"])
                    tn, fn = tn + k, fn + tt["absent_answers"] - k
            n = tp + tn + fp + fn
            prec = round((tp + tn) / n, 4) if n else None
            out[attr] = {"test_answered_precision": prec, "answered": n, "wilson95_low": DC.wilson_low(tp + tn, n),
                         "meets_target": bool(prec is not None and prec >= TARGET and n >= min_answered),
                         **{k: t.get(k) for k in ("R@1", "R@2", "R@3")}, "selected": r["selected"]["features"]}
        else:
            prec, n = t.get("answered_precision"), t.get("answered") or 0
            out[attr] = {"test_answered_precision": prec, "coverage": t.get("coverage"), "answered": n,
                         "wilson95_low": t.get("wilson95_low"), "test_accuracy": t.get("accuracy_forced"),
                         "meets_target": bool(prec is not None and prec >= TARGET and n >= min_answered),
                         **{k: t.get(k) for k in ("R@1", "R@2", "R@3")}, "selected": r["selected"]["features"]}
    return out


class QueryAttributesV3:
    """Grounding DINO attributes v3 for ONE object cut-out (the search app's query); shares a loaded Grounding DINO."""

    def __init__(self, v2: Path, v3: Path, dino_model=None, dino_proc=None):
        import joblib
        import json
        self.models = joblib.load(v3 / "attr_v3_models.joblib")
        rep = json.loads((v3 / "attr_v3_report.json").read_text(encoding="utf-8"))
        self.validation = validation_summary(rep)
        self.spec = DC.read_cal(v2)["dino_spec"]              # the full-run prompt sets (same evidence as the dataset)
        self.da = DA.DinoAttr(model=dino_model, proc=dino_proc)

    def __call__(self, rgba: np.ndarray, attr_full: dict) -> dict:
        from .dino_predict import QueryAttributes
        t0 = time.time()
        img, off = DA.cutout(rgba)
        r = self.da.score_spec([img], self.spec)[0]
        D = {f"{a}|{v}|{f}": d[f][None] for (a, v), d in r.items() for f in d}
        D["offset"] = np.array([off])
        V = QueryAttributes.v1_vectors(attr_full)
        FH = np.zeros((1, len(DC.CLASSES["form"])), np.float32)      # no recipe text for an uploaded photo
        row = label_rows(self.models, D, V, FH, [""])[0]
        return {"attributes": row, "validation": self.validation, "seconds": round(time.time() - t0, 2), "version": "attributes_v3"}
