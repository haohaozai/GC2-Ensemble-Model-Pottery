"""Recall@K per feature (K = 3, 5, 10, 15, 20) for the search with Grounding DINO attributes (attributes_v2).

Protocol (no leakage): queries AND gallery are the Claude-labelled TEST objects (leave-one-out: every test object is a
query once, the other test objects are the gallery). The ranking uses only what the app has at search time: image
features of the balanced index, and the attributes PREDICTED by Grounding DINO for the query and for the gallery
(the DINO models were fitted on the validation objects only, so every label used here is out-of-sample). Relevance is
judged with Claude's visual labels:
  colour        the query's most important colour region within dE00 20 of a meaningful region of the object
  attribute f   same Claude label (effects / defects: at least one effect in common, or none on both)
  combination   every feature of the combination relevant
Rankings compared: balanced search without attributes (v1); balanced + the attributes of the combination, matched SOFTLY
(probability, from the DINO models, that the gallery object has the queried value), with the query values either
auto-detected by Grounding DINO or set correctly by the user (Claude's label of the query); random.
Recall@K = share of queries with at least one relevant object in the top K.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import balanced as B
from . import balanced_eval as BE
from . import claude_labels as CL

KS = (3, 5, 10, 15, 20)
ATTRS = ["form", "surface_finish", "transparency", "texture", "glaze_effect", "defect"]
MULTI = {"glaze_effect", "defect"}
COMBOS = [["colour"], *[[a] for a in ATTRS], ["colour", "form"], ["colour", "glaze_effect"], ["colour", "form", "glaze_effect"],
          ["form", "surface_finish", "transparency"], ["colour", "form", "surface_finish", "glaze_effect"]]
ATTR_SHARE = 0.40


def _match(f, a, b):
    if a is None or b is None:
        return None
    if f in MULTI:
        a, b = set(a), set(b)
        return (not b) if not a else bool(a & b)
    return a == b


def evaluate(v1: Path, v2: Path, seed: int = 0, pred_path: Path | None = None) -> dict:
    """pred_path: another prediction file with the same per-object format (attributes v3); its "candidates" add the
    method 'user picks from the top 3' (the user's value is the correct label when it is among the candidates, else #1)."""
    lab = CL.parse(v2 / "claude_labels_raw.txt")
    pj = json.loads((pred_path or (v2 / "attribute_predictions_v2.json")).read_text(encoding="utf-8"))
    pred, valid = pj["objects"], pj.get("validation", {})
    W = json.loads((v1 / "balanced_weights.json").read_text(encoding="utf-8"))["modes"]["balanced"]["weights"]
    split = {}
    with (v1 / "labels.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            split[r["object_id"]] = r["split"]
    I = dict(np.load(v1 / "balanced_index.npz", allow_pickle=False))
    pos = {o: k for k, o in enumerate(I["ids"].tolist())}
    ids = [o for o in sorted(lab) if split.get(o) == "test" and o in pos and o in pred and I["m_app"][pos[o]] and I["reg_valid"][pos[o]].any()]
    idx = np.array([pos[o] for o in ids])
    n = len(ids)
    # truth (Claude) and predictions (DINO) per attribute
    truth = {f: [lab[o][f] for o in ids] for f in ATTRS}
    dino = {f: [(pred[o][f]["present"] if f in MULTI else pred[o][f]["value"]) for o in ids] for f in ATTRS}
    has_cand = all("candidates" in pred[o][ATTRS[0]] for o in ids)
    picked = {}
    if has_cand:               # the user clicks the right candidate when it is shown, otherwise keeps #1
        for f in ATTRS:
            vals = []
            for a, o in enumerate(ids):
                cl = [c["label"] for c in pred[o][f]["candidates"]]
                t = truth[f][a]
                if f in MULTI:
                    hit = [c for c in (t or []) if c in cl]
                    vals.append(hit if hit else dino[f][a])
                else:
                    vals.append(t if t in cl else dino[f][a])
            picked[f] = vals
    from .dino_calib import CLASSES
    probs = {}
    for f in ATTRS:
        ci = {c: j for j, c in enumerate(CLASSES[f])}
        Pm = np.zeros((n, len(ci)), np.float32)
        for r, o in enumerate(ids):
            pr_ = pred[o][f].get("probs")
            if isinstance(pr_, dict):
                for c, q in pr_.items():
                    Pm[r, ci[c]] = q
            elif pr_ is not None:
                Pm[r, :len(pr_)] = pr_
        probs[f] = (Pm, ci)
    # colour truth: query's top-weighted region vs the object's regions (measured colours, dE00 <= 20)
    def top_colour(k):
        qlab, qpos, qarea, qauto, qhuman, qvalid = (I["reg_" + x][k] for x in ("lab", "pos", "area", "auto", "human", "valid"))
        w = B._weights(qarea, qauto, qhuman, qvalid, "reviewed")
        return qlab[int(np.argmax(w))]
    colour_rel = np.zeros((n, n), bool)
    for a in range(n):
        ql = top_colour(idx[a])
        for b in range(n):
            gv = I["reg_valid"][idx[b]]
            colour_rel[a, b] = bool(gv.any() and B.de2000(ql[None], I["reg_lab"][idx[b]][gv]).min() <= 20)
    rng = np.random.default_rng(seed)
    out = {"protocol": "leave-one-out over the Claude-labelled TEST objects; ranking with DINO-predicted attributes, relevance = Claude labels",
           "queries": n, "gallery_per_query": n - 1, "attribute_share": ATTR_SHARE, "combos": {}}
    for combo in COMBOS:
        res = {}
        methods = ["balanced (no attributes)", "balanced + attributes (auto-detected query)",
                   "balanced + attributes (as the app: validated answers only)", "balanced + attributes (user-set query)", "random"]
        if has_cand:
            methods.insert(2, "balanced + attributes (user picks from the top 3)")
        for method in methods:
            hits = {k: [] for k in KS}
            prec = {k: [] for k in KS}
            n_q = 0
            for a in range(n):
                gal_i = np.array([b for b in range(n) if b != a])
                gal = idx[gal_i]
                rel = []
                for b in gal_i:
                    vals = []
                    for f in combo:
                        vals.append(bool(colour_rel[a, b]) if f == "colour" else _match(f, truth[f][a], truth[f][b]))
                    rel.append(None if any(v is None for v in vals) else all(vals))
                if all(r is None for r in rel) or truth_missing(combo, truth, a):
                    continue
                rel = np.array([bool(r) for r in rel])
                if method == "random":
                    order = rng.permutation(len(gal_i))
                else:
                    q = BE.query_from_index(I, idx[a])
                    S = BE.component_scores(I, q, gal, "reviewed")
                    w = dict(W)
                    if "colour" not in combo:
                        w.pop("colour", None)
                    attrs = [f for f in combo if f != "colour"]
                    if "validated answers only" in method:      # the app auto-ticks only validated, answered attributes
                        attrs = [f for f in attrs if (valid.get(f) or {}).get("meets_target") and pred[ids[a]][f]["status"] == "answered"]
                    if method.startswith("balanced +") and attrs:
                        A = np.zeros(len(gal_i), np.float32)
                        for f in attrs:
                            qv = truth[f][a] if "user-set" in method else picked[f][a] if "top 3" in method else dino[f][a]
                            Pm, ci = probs[f]
                            G = Pm[gal_i]
                            if f in MULTI:
                                want = [ci[x] for x in (qv or []) if x in ci]
                                A += G[:, want].mean(1) if want else np.prod(1 - G, 1)
                            else:
                                A += G[:, ci[qv]] if qv in ci else 0
                        S["attributes"] = A / len(attrs)
                        tot = sum(w.values()) or 1
                        w = {c: v / tot * (1 - ATTR_SHARE) for c, v in w.items()}
                        w["attributes"] = ATTR_SHARE
                    F = B.fuse(S, w, np.ones(len(gal_i), bool))
                    order = np.argsort(-F)
                n_q += 1
                for k in KS:
                    top = rel[order[:k]]
                    hits[k].append(bool(top.any()))
                    prec[k].append(float(top.mean()))
            res[method] = {"queries": n_q, **{f"Recall@{k}": round(float(np.mean(hits[k])), 3) if hits[k] else None for k in KS},
                           **{f"P@{k}": round(float(np.mean(prec[k])), 3) if prec[k] else None for k in KS}}
        out["combos"][" + ".join(combo)] = res
    return out


def truth_missing(combo, truth, a) -> bool:
    return any(f != "colour" and truth[f][a] is None for f in combo)
