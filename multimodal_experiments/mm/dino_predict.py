"""Auto-labels of the six object attributes from Grounding DINO (attributes_v2) - the file the review app and the search
app read (work/attributes_v2/attribute_predictions_v2.json), and the same labelling for ONE query object in the search
app (QueryAttributes - it shares the segmenter's Grounding DINO, so no extra model is loaded).

Per object and attribute:
  value       the label (single-label: always the best class, even when uncertain - object form is never "unknown")
  status      "answered" (confidence >= the threshold that reached the 98 % target on validation) or "uncertain"
  confidence  model probability of the value
  top         the three best classes with probabilities
  group       surface / transparency only: the group label (glossy / satin / matte, clear / opaque) and its status
  present / absent / uncertain / not_checked   multi-label (glaze effects, defects): per class decision
  evidence    the three strongest Grounding DINO phrases with their score and box (object-image pixels)
  hint        object form only: form words found in the recipe name / notes
Objects of the Claude-labelled sample also carry "claude" (Claude's visual labels) for comparison in the review app.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import joblib
import numpy as np

from . import claude_labels as CL
from . import dino_attr as DA
from . import dino_calib as DC

FIELDS = ["form", "surface_finish", "transparency", "texture", "glaze_effect", "defect"]
GROUP_OF = {"surface_finish": "surface_group", "transparency": "transparency_group"}
META_FIELD = {"surface_finish": "surface_meta", "transparency": "transparency_meta"}


def _features(m, attr, D, V, FH):
    X = [DC.dino_features(D, attr, m["variants"], emb=m.get("emb", False))]
    if m.get("v1"):
        X.append(np.nan_to_num(V[attr]))
    if m.get("recipe_text"):
        X.append(FH)
    return np.concatenate(X, 1).astype(np.float32)


def _evidence(D, attr, variants, k, offset, prefer=()):
    """strongest phrases of the attribute (classes in `prefer` first), boxes shifted to object-image pixels."""
    a = DC.DINO_ATTR.get(attr, attr)
    items = []
    for v in variants:
        ph = DA.phrase_list(a, v)
        S, B = D[f"{a}|{v}|score"][k], D[f"{a}|{v}|box"][k]
        for j, (cls, phrase) in enumerate(ph):
            b = B[j]
            items.append((cls in prefer, float(S[j]), phrase,
                          [round(float(b[0] + offset[0]), 1), round(float(b[1] + offset[1]), 1),
                           round(float(b[2] + offset[0]), 1), round(float(b[3] + offset[1]), 1)]))
    items.sort(key=lambda t: (not t[0], -t[1]))
    seen, out = set(), []
    for _, sc, phrase, box in items:
        if phrase not in seen:
            seen.add(phrase)
            out.append([phrase, round(sc, 3), box])
        if len(out) == 3:
            break
    return out


def label_rows(models: dict, D: dict, V: dict, FH: np.ndarray, hints: list[str]) -> list[dict]:
    """per object (rows of D): the six attributes (+ groups folded into surface / transparency)."""
    n = len(D["offset"])
    rows = [{} for _ in range(n)]
    for attr in FIELDS + ["surface_group", "transparency_group"]:
        m = models[attr]
        X = _features(m, attr, D, V, FH)
        if m["kind"] == "single":
            P = DC.predict_single(m["model"], X, len(m["classes"]))
            thr = m["threshold"]
            for k in range(n):
                j = int(P[k].argmax())
                conf = float(P[k, j])
                rec = {"value": m["classes"][j], "status": "answered" if thr is not None and conf >= thr else "uncertain",
                       "confidence": round(conf, 3), "top": [[m["classes"][i], round(float(P[k, i]), 3)] for i in np.argsort(-P[k])[:3]],
                       "probs": [round(float(x), 3) for x in P[k]]}
                if attr in FIELDS:
                    rec["evidence"] = _evidence(D, attr, m["variants"], k, D["offset"][k], prefer={rec["value"]})
                if attr == "form" and hints[k]:
                    rec["hint"] = hints[k]
                rows[k][attr] = rec
        else:
            per = m["per_class"]
            probs = {c: per[c]["model"][1].predict_proba(per[c]["model"][0].transform(X))[:, 1] for c in per}
            for k in range(n):
                pres, absn, unc, pl = [], [], [], []
                for c in m["classes"]:
                    if c not in per or (per[c]["present"] is None and per[c]["absent"] is None):
                        continue                       # no validated decision possible for this class: "not checked"
                    p = float(probs[c][k])
                    pl.append([c, round(p, 3)])
                    if per[c]["present"] is not None and p >= per[c]["present"]:
                        pres.append(c)
                    elif per[c]["absent"] is not None and p <= per[c]["absent"]:
                        absn.append(c)
                    else:
                        unc.append(c)
                pl.sort(key=lambda t: -t[1])
                conf = [p for c, p in pl if c in pres] + [1 - p for c, p in pl if c in absn]
                rows[k][attr] = {"value": pres, "present": pres, "absent": absn, "uncertain": unc,
                                 "not_checked": [c for c in m["classes"] if c not in per or (per[c]["present"] is None and per[c]["absent"] is None)],
                                 "status": "answered" if not unc else "uncertain", "top": pl[:4],
                                 "probs": {c: round(float(probs[c][k]), 3) for c in per},
                                 "confidence": round(min(conf), 3) if conf else 0.0,
                                 "evidence": _evidence(D, attr, m["variants"], k, D["offset"][k], prefer=set(pres) | set(unc))}
    for r in rows:
        for fine, grp in GROUP_OF.items():
            g = r.pop(grp)
            r[fine]["group"] = {"value": g["value"], "status": g["status"], "confidence": g["confidence"]}
    return rows


def predict(v1: Path, v2: Path, dino_npz: Path) -> dict:
    t0 = time.time()
    models = joblib.load(v2 / "dino_heads.joblib")
    cal = DC.read_cal(v2)
    ids = np.load(dino_npz)["ids"].tolist()
    D = DC.load_dino(dino_npz, ids)             # merges <name>_r2.npz (embeddings, tiles) when present
    V = DC.v1_probs(v1, ids)
    FH, hints = DC.form_hints(v1, ids)
    meta = {}
    with (v1 / "labels.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            meta[r["object_id"]] = r
    claude = CL.parse(v2 / "claude_labels_raw.txt")
    rows = label_rows(models, D, V, FH, hints)
    out = dict(zip(ids, rows))
    for oid in ids:
        if oid in claude:
            out[oid]["claude"] = {f: claude[oid][f] for f in FIELDS}
    # review priority: uncertain answers of attributes that are validated (DINO usually answers them) + disagreements
    # with Claude (10, confident answers only) and with the recipe metadata (5); each uncertain one counts 1. Attributes that are never validated are not counted (always uncertain).
    val = validation_summary(cal)
    prio = {}
    for oid in ids:
        o = out[oid]
        n = sum(1 for f in FIELDS if o[f]["status"] != "answered" and (val.get(f) or {}).get("meets_target"))
        cl = o.get("claude") or {}
        for f in FIELDS:                       # a CONFIDENT DINO answer that Claude contradicts = an error worth checking
            if cl.get(f) is None or (f not in ("glaze_effect", "defect") and o[f]["status"] != "answered"):
                continue
            if f in ("glaze_effect", "defect"):        # only the classes DINO decided (present / absent) can conflict
                wrong = any(c not in cl[f] for c in o[f]["present"]) or any(c in cl[f] for c in o[f]["absent"])
            else:
                wrong = cl[f] != o[f]["value"]
            if wrong:
                n += 10
        mr = meta.get(oid) or {}
        for f, col in META_FIELD.items():
            if mr.get(col) and o[f]["status"] == "answered" and mr[col] != o[f]["value"]:
                n += 5
        if n:
            prio[oid] = n
    return {"version": "attributes_v2", "created": time.strftime("%Y-%m-%d %H:%M"), "source": dino_npz.name,
            "label_source_for_validation": CL.PROVENANCE, "target_precision": DC.TARGET,
            "validation": val, "prompt_sets": cal.get("prompt_sets_needed"), "dino_spec": cal.get("dino_spec"),
            "review_priority": prio, "objects": out, "seconds": round(time.time() - t0, 1)}


def validation_summary(cal: dict, min_answered: int = 20) -> dict:
    """per attribute: TEST answered precision (Claude labels), coverage, Wilson 95 % lower bound and whether the
    98 % target is met (with at least `min_answered` test answers)."""
    out = {}
    for attr, r in cal["attributes"].items():
        t = r.get("test") or {}
        prec, n = t.get("answered_precision"), t.get("answered") or 0
        out[attr] = {"test_answered_precision": prec, "coverage": t.get("coverage"), "answered": n,
                     "wilson95_low": t.get("wilson95_low"), "selected": r.get("selected"),
                     "meets_target": bool(prec is not None and prec >= DC.TARGET and n >= min_answered)}
    return out


class QueryAttributes:
    """Grounding DINO attribute detection for ONE object cut-out (the search app's query); shares a loaded model."""

    def __init__(self, v2: Path, dino_model=None, dino_proc=None):
        self.models = joblib.load(v2 / "dino_heads.joblib")
        cal = DC.read_cal(v2)
        self.spec = cal["dino_spec"]
        self.validation = validation_summary(cal)
        self.da = DA.DinoAttr(model=dino_model, proc=dino_proc)

    @staticmethod
    def v1_vectors(attr_full: dict) -> dict:
        """ObjectEncoder full probability vectors -> CLASSES order of the calibration (as v1_probs does)."""
        V = {}
        for task, d in attr_full.items():
            if task not in DC.CLASSES:
                continue
            row = np.zeros((1, len(DC.CLASSES[task])), np.float32)
            for j, c in enumerate(DC.CLASSES[task]):
                if c in d["classes"]:
                    row[0, j] = d["p"][d["classes"].index(c)]
            V[task] = row
        return V

    def __call__(self, rgba: np.ndarray, attr_full: dict) -> dict:
        t0 = time.time()
        img, off = DA.cutout(rgba)
        r = self.da.score_spec([img], self.spec)[0]
        D = {f"{a}|{v}|{f}": d[f][None] for (a, v), d in r.items() for f in d}
        D["offset"] = np.array([off])
        V = self.v1_vectors(attr_full)
        FH = np.zeros((1, len(DC.CLASSES["form"])), np.float32)       # no recipe text for an uploaded photo
        row = label_rows(self.models, D, V, FH, [""])[0]
        return {"attributes": row, "validation": self.validation, "seconds": round(time.time() - t0, 2)}
