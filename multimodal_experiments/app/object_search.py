"""Balanced, object-level search for the Glaze Search app (attributes_v1).

Every segmented object of the 7k photos is a search result on its own (exact object image, e.g. 101_3.jpg - never
the parent photo instead). Modes: balanced (default), shape, glaze (glaze / surface), colour - weights tuned on
validation queries (work/attributes_v1/balanced_weights.json). Similarity scores and classifier confidences are
reported separately. Reviewed tile labels (review/tile_review.json) replace predictions and can be used as a filter.

Features (attributes_v2): the user ticks what must match - colour and / or the six attributes (object form, surface
finish, transparency, surface texture, glaze effects, possible defects). The query's attributes are detected by
Grounding DINO (editable); every gallery object's label is the best available one: human review > Claude's check >
Grounding DINO (answered > uncertain) > the earlier model. Ticked attributes add an "attributes" component to the
ranking, and Recall@K / P@K (K = 3, 5, 10, 15, 20) are reported per feature and for all ticked features together.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from mm import balanced as B
from mm import balanced_eval as BE
from mm.common import SEG, obj_file, read_csv, read_json

ATTR = Path(__file__).resolve().parents[1] / "work" / "attributes_v1"
V2 = ATTR.parent / "attributes_v2"
V3 = ATTR.parent / "attributes_v3"          # v3: trained on Claude's labels (1,504 train objects), top-3 candidates
BONUS_SEG = SEG.parent / "bonus_image_claude_process"     # bonus_image set (others_image photos)
BONUS_SRC = SEG.parent / "others_image"


def attr_labels_path() -> Path:
    """gallery attribute labels: v3 when present, else v2"""
    p3 = V3 / "attribute_predictions_v3.json"
    return p3 if p3.exists() else V2 / "attribute_predictions_v2.json"
ATTR_FEATS = ["form", "surface_finish", "transparency", "texture", "glaze_effect", "defect"]
MULTI_FEATS = {"glaze_effect", "defect"}
FEATURES = ["colour"] + ATTR_FEATS
FEATURE_LABEL = {"colour": "colour", "form": "object form", "surface_finish": "surface finish", "transparency": "transparency",
                 "texture": "surface texture", "glaze_effect": "glaze effects", "defect": "possible defects"}
KS = (3, 5, 10, 15, 20)
SOURCE_CONF = {"human": 1.0, "claude": 1.0, "dino": 1.0, "dino_uncertain": 0.6, "earlier model": 0.5, "metadata": 0.8}
ATTR_SHARE = 0.40          # share of the image part given to the ticked attributes (default; evaluated offline)
MODES = ("balanced", "shape", "glaze", "colour")
MODE_LABEL = {"balanced": "Balanced", "shape": "Shape priority", "glaze": "Glaze / surface priority", "colour": "Colour priority"}
COMP_LABEL = {"appearance": "glaze appearance", "shape": "shape", "surface": "surface / effects", "colour": "colour regions",
              "text": "text", "formula": "formula", "attributes": "attributes (ticked)"}
# Priority sliders (photo part of the ranking): three topics that always add up to 1. Glaze / surface = glaze appearance +
# surface / effects, split in the ratio of the chosen mode, so a mode's own slider values reproduce its validated weights.
PRIORITY = ("shape", "glaze", "colour")
# Evaluation (separate from ranking) - every search, the three criteria with the query object's values:
#   shape   same object form; glaze / surface: same surface group (glossy / satin / matte) [+ >= 1 glaze effect in common
#   when the user picked effects]; colour: the query's most important colour within dE2000 20 of a region of the object
#   (or the same Color Lab family when "same family" is ticked). Unknown labels are "?" - never a match.
EVAL_KS = (3, 5, 10, 15, 20)
CRIT_LABEL = {"shape": "shape (object form)", "glaze": "glaze / surface", "colour": "colour", "strict": "strict (all criteria)"}


class ObjectSearch:
    def __init__(self, engine, clip=None):
        from mm.object_query import ObjectEncoder
        t0 = time.time()
        self.eng = engine
        self.I = dict(np.load(ATTR / "balanced_index.npz", allow_pickle=False))
        self.W = read_json(ATTR / "balanced_weights.json")
        self.enc = ObjectEncoder(ATTR, engine, deep=getattr(engine, "deep", None), clip=clip)
        self.preds = read_json(ATTR / "attribute_predictions.json")["objects"]
        rev = {r["object_id"]: r for r in read_csv(SEG / "review" / "object_review.csv")}
        # bonus_image objects (mm/bonus_v3.py): same encoder as an uploaded photo; their recipes are not in the recipe bank
        self.bonus_meta, self.bonus_ids = {}, set()
        bp = V3 / "bonus_index.npz"
        if bp.exists():
            Ib = dict(np.load(bp, allow_pickle=False))
            n0 = len(self.I["ids"])
            for k, v in list(self.I.items()):
                if k in Ib and k != "forms" and len(v) == n0:
                    self.I[k] = np.concatenate([v, Ib[k].astype(v.dtype) if v.dtype.kind != "U" else Ib[k]])
            self.bonus_ids = set(Ib["ids"].tolist())
            self.bonus_meta = {rid: r["meta"] for rid, r in (read_json(V3 / "bonus_recipes.json", {}) or {}).items()}
            self.preds.update((read_json(V3 / "bonus_attr_v1.json", {}) or {}).get("objects", {}))
            bpr = BONUS_SEG / "review" / "object_review.csv"
            if bpr.exists():
                rev.update({r["object_id"]: r for r in read_csv(bpr)})
        self.csv_colours = self._colours_from_review(rev)
        self.ids = self.I["ids"].tolist()
        self.pos = {o: k for k, o in enumerate(self.ids)}
        self.ok = self.I["m_app"] & self.I["reg_valid"].any(1)
        self.status = {o: (rev.get(o) or {}).get("review_status", "") for o in self.ids}
        self.parent = {o: (rev.get(o) or {}).get("parent_filename", o.split("_")[0] + ".jpg") for o in self.ids}
        self._tiles, self._tiles_mtime = {}, None
        self._ga, self._ga_key, self._fam = None, None, {}
        self.cache = {}
        self.seconds_init = round(time.time() - t0, 1)

    # ------------------------------------------------------------ segmentation colour labels
    def _colours_from_review(self, rev: dict) -> int:
        """ColorThief palette (color1-3 hex / pct of review/object_review.csv - the same values as full_auto_tags.csv and
        tag_json) replaces the copy saved in the index (ct_lab / ct_pct), so the segmentation's colour labels are the
        ones searched and shown; the index of 2026-10-18 had no colours for 316 objects recoloured later.
        Returns the number of objects whose palette changed."""
        from mm.colorlab import hex2lab
        lab, pct = self.I["ct_lab"].copy(), self.I["ct_pct"].copy()
        for k, oid in enumerate(self.I["ids"].tolist()):
            r = rev.get(oid)
            if r is None:
                continue
            lab[k], pct[k] = 0, 0
            for j in range(3):
                h = r.get(f"color{j + 1}_hex") or ""
                if h.startswith("#") and len(h) == 7:
                    lab[k, j] = hex2lab(h)
                    pct[k, j] = float(r.get(f"color{j + 1}_pct") or 0) / 100
        changed = int(((lab != self.I["ct_lab"]).any((1, 2)) | (pct != self.I["ct_pct"]).any(1)).sum())
        self.I["ct_lab"], self.I["ct_pct"] = lab, pct
        return changed

    # ------------------------------------------------------------ reviewed tiles (sidecar, reloaded when it changes)
    def tiles(self) -> dict:
        p = SEG / "review" / "tile_review.json"
        m = p.stat().st_mtime if p.exists() else None
        if m != self._tiles_mtime:
            self._tiles = (read_json(p, {}) or {}).get("objects", {}) if p.exists() else {}
            self._tiles_mtime = m
        return self._tiles

    # ------------------------------------------------------------ gallery attribute labels (best available source)
    def gallery_attrs(self) -> dict:
        """attribute -> (values [N], sources [N], probabilities [N, C]); single-label values are strings ('' = none),
        multi-label frozensets. Probabilities (classes of mm.dino_calib.CLASSES): one-hot for human / Claude labels,
        Grounding DINO's model probabilities, else the earlier model's top classes."""
        from mm.dino_calib import CLASSES
        p2 = attr_labels_path()
        pb = V3 / "attribute_predictions_v3_bonus.json"
        tl = self.tiles()
        key = (p2.stat().st_mtime if p2.exists() else None, pb.stat().st_mtime if pb.exists() else None, self._tiles_mtime)
        if self._ga is not None and self._ga_key == key:
            return self._ga
        v2 = (read_json(p2, {}) or {}).get("objects", {}) if p2.exists() else {}
        if pb.exists() and self.bonus_ids:
            v2 = {**v2, **(read_json(pb, {}) or {}).get("objects", {})}
        out = {}
        for f in ATTR_FEATS:
            vals, src = [], []
            cls = CLASSES[f]
            ci = {c: j for j, c in enumerate(cls)}
            Pm = np.zeros((len(self.ids), len(cls)), np.float32)
            for row, oid in enumerate(self.ids):
                h = ((tl.get(oid) or {}).get("attributes") or {}).get(f) or {}
                hv = [v for v in (h.get("value") if isinstance(h.get("value"), list) else [h.get("value")]) if v and v not in ("unknown", "ambiguous")]
                d2 = v2.get(oid) or {}
                cl = (d2.get("claude") or {}).get(f)
                pr = (self.preds.get(oid) or {}).get(f + "_final") or {}
                if h.get("status") in ("confirmed", "corrected") and hv:
                    v, s_ = ([x for x in hv if x != "none"] if f in MULTI_FEATS else hv[0]), "human"
                elif cl is not None:
                    v, s_ = cl, "claude"
                elif f in d2:
                    v = d2[f]["present"] if f in MULTI_FEATS else d2[f]["value"]
                    s_ = "dino" if d2[f]["status"] == "answered" else "dino_uncertain"
                else:
                    v = pr.get("value")
                    s_ = "metadata" if str(pr.get("provenance", "")).startswith("glazy_meta") else "earlier model"
                if f in MULTI_FEATS:
                    v = frozenset(x for x in (v or []) if x not in ("none", "unknown"))
                else:
                    v = "" if v in (None, "unknown", "ambiguous") else v
                vals.append(v)
                src.append(s_)
                if s_ in ("human", "claude"):
                    for x in (v if f in MULTI_FEATS else [v]):
                        if x in ci:
                            Pm[row, ci[x]] = 1.0
                elif s_.startswith("dino"):
                    pr_ = d2[f].get("probs")
                    if isinstance(pr_, dict):
                        for c, q in pr_.items():
                            if c in ci:
                                Pm[row, ci[c]] = q
                    elif pr_ is not None:
                        Pm[row, :len(pr_)] = pr_
                else:
                    for c, q in ((self.preds.get(oid) or {}).get(f) or []):
                        if c in ci:
                            Pm[row, ci[c]] = q
            out[f] = (vals, src, Pm)
        self._ga, self._ga_key = out, key
        return out

    @staticmethod
    def attr_soft(f, q, Pm) -> np.ndarray:
        """probability that each gallery object has the queried value(s): single = P(value); effects / defects = mean
        P(each wanted effect), or P(no effect) = prod(1 - P) when the query is 'none'."""
        from mm.dino_calib import CLASSES
        ci = {c: j for j, c in enumerate(CLASSES[f])}
        if f in MULTI_FEATS:
            want = [ci[x] for x in (q or []) if x in ci]
            return Pm[:, want].mean(1) if want else np.prod(1 - Pm, 1)
        return Pm[:, ci[q]] if q in ci else np.zeros(len(Pm), np.float32)

    @staticmethod
    def attr_match(f, q, g) -> float:
        if f in MULTI_FEATS:
            q = frozenset(x for x in (q or []) if x not in ("none", "unknown"))
            if not q:
                return 1.0 if not g else 0.0
            return len(q & g) / len(q | g) if g else 0.0
        return 1.0 if q and g == q else 0.0

    @staticmethod
    def attr_relevant(f, q, g) -> bool | None:
        if f in MULTI_FEATS:
            q = frozenset(x for x in (q or []) if x not in ("none", "unknown"))
            return (not g) if not q else bool(q & g)
        return None if not q else g == q

    def info(self) -> dict:
        from mm.dino_calib import CLASSES
        rep3 = read_json(V3 / "attr_v3_report.json", {}) or {}
        if rep3:
            from mm.attr_v3 import validation_summary as vs3
            val, ver = vs3(rep3), "attributes_v3"
        else:
            v2 = read_json(V2 / "dino_calibration.json", {}) or {}
            from mm.dino_predict import validation_summary
            val, ver = (validation_summary(v2) if v2 else {}), "attributes_v2"
        return {"attr_vocab": {f: CLASSES[f] for f in ATTR_FEATS}, "validation": val, "attr_version": ver,
                "dino_labels": attr_labels_path().exists(),
                "objects": int(self.ok.sum()), "modes": {m: {"label": MODE_LABEL[m], "weights": self.W["modes"][m]["weights"],
                                                             "priority": {k: round(v, 4) for k, v in self.priority_of(m, None).items()},
                                                             "val_nDCG@10": self.W["modes"][m]["val"]["nDCG@10"]} for m in MODES},
                "reviewed_tiles": sum(1 for r in self.tiles().values() if r.get("category") == "review_tile"),
                "colours_updated_from_csv": self.csv_colours, "init_s": self.seconds_init}

    # ------------------------------------------------------------ query object
    def encode(self, key, rgba) -> dict:
        if key not in self.cache:
            self.cache[key] = self.enc.encode(rgba)
            while len(self.cache) > 40:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    @staticmethod
    def region_list(details) -> list:
        return [{k: r[k] for k in ("id", "hex", "name", "area_frac", "position", "distinctiveness_de00", "contrast_de00",
                                    "auto_importance", "flags", "meaningful")} for r in details["regions"].get("regions", [])]

    # ------------------------------------------------------------ search
    def search(self, p: dict, qf: dict | None, text_vec=None, formula_vec=None, qattr=None, q_attrs: dict | None = None) -> dict:
        t0 = time.time()
        mode = p.get("mode") if p.get("mode") in MODES else "balanced"
        I = self.I
        gal = np.nonzero(self.ok)[0]
        filters = []
        if (p.get("filters") or {}).get("reviewed_tiles"):
            tl = self.tiles()
            keep = np.array([tl.get(self.ids[k], {}).get("category") == "review_tile" for k in gal])
            gal = gal[keep]
            filters.append("reviewed tiles only (review_tile category)")
            if not len(gal):
                return {"results": [], "gallery": 0, "filters_applied": filters, "mode": mode, "mode_label": MODE_LABEL[mode],
                        "message": "No object is in the review_tile category yet - add objects with key T (or 'Tile ▸') in the "
                                   "review app, then search again (or switch the filter off)."}
        q = {}
        if qf is not None:
            q = {k: qf[k] for k in ("app", "geom", "sil", "attr")}
            imp = {str(k): v for k, v in (p.get("region_importance") or {}).items() if v in B.IMP_W}
            res = qf["details"]["regions"]
            q["reg"] = B.region_arrays(res, {"regions": {k: {"importance": v} for k, v in imp.items()}})
        colours = p.get("colours") or []
        if colours and (p.get("colour_source") == "typed" or qf is None):
            q["reg"] = B.colour_query(colours)
        if text_vec is not None:
            q["text"] = text_vec
        if formula_vec is not None:
            q["formula"] = formula_vec
        if not q and not any(f in (q_attrs or {}) for f in (p.get("features") or []) if f in ATTR_FEATS):
            return {"results": [], "message": "Nothing to search with: add a photo, colours, tags / notes, a formula or tick an attribute value.",
                    "gallery": int(len(gal))}
        ticked = [f for f in (p.get("features") or ["colour"]) if f in FEATURES]
        q_attrs = {f: v for f, v in (q_attrs or {}).items() if f in ticked and f in ATTR_FEATS}
        S = BE.component_scores(I, q, gal, "reviewed")
        GA = self.gallery_attrs() if q_attrs else None
        if q_attrs:                              # attribute component: mean match over the ticked attributes x label confidence
            A = np.zeros(len(gal), np.float32)
            for f, qv in q_attrs.items():
                A += self.attr_soft(f, qv, GA[f][2][gal])
            S["attributes"] = A / len(q_attrs)
        w_img = dict(self.W["modes"][mode]["weights"])
        prio_in = self.priority_of(mode, p.get("priority"))
        if qf is not None and isinstance(p.get("priority"), dict):
            w_img = self.priority_weights(mode, prio_in)
        if qf is None:                           # no photo: only typed / picked colours are image-side evidence
            w_img = {"colour": 1.0} if "reg" in q else {}
        if "colour" not in ticked:
            w_img.pop("colour", None)
        tot = sum(v for c, v in w_img.items() if c in S) or 1.0
        img_share = 0.40 * (1 - ATTR_SHARE) if "attributes" in S and w_img else 0.40
        weights = {c: v / tot * img_share for c, v in w_img.items() if c in S}
        if "attributes" in S:
            weights["attributes"] = 0.40 - img_share if w_img else 0.40
        if "text" in S:
            weights["text"] = 0.35
        if "formula" in S:
            weights["formula"] = 0.25
        if user_w := p.get("weights"):              # optional manual override from the page sliders
            weights = {c: float(user_w.get(c, weights.get(c, 0))) for c in S}
        mask = np.ones(len(gal), bool)
        F = B.fuse({c: np.nan_to_num(s, nan=-1.0) for c, s in S.items()}, weights, mask)
        top = np.argsort(-F)[:int(p.get("top", 20))]
        # contribution of every component to the final score = its weight share x its z-score (what fuse adds up)
        wsum = sum(weights.get(c, 0) for c in S) or 1.0
        Z = {c: B.zscore(np.nan_to_num(s, nan=-1.0), mask) for c, s in S.items() if weights.get(c, 0) > 0}
        out = []
        tl = self.tiles()
        for rank, j in enumerate(top, 1):
            k = gal[j]
            oid = self.ids[k]
            rid = str(I["recipe_id"][k])
            rp = self.eng.bank.pos.get(rid)
            meta = self.eng.meta_rows[rp] if rp is not None else self.bonus_meta.get(rid, {})
            pr = self.preds.get(oid) or {}
            attrs = {f: {"final": pr.get(f + "_final"), "top": pr.get(f)} for f in
                     ("form", "surface_finish", "transparency", "glaze_effect", "defect", "texture") if pr.get(f) is not None}
            row = {"rank": rank, "object_id": oid, "parent": self.parent.get(oid), "recipe_id": rid, "score": round(float(F[j]), 3),
                   "image_set": "bonus_image" if oid in self.bonus_ids else "7k_matched",
                   "similarity": {COMP_LABEL[c]: round(float(S[c][j]), 3) for c in S if weights.get(c, 0) > 0 and np.isfinite(S[c][j])},
                   "contribution": {COMP_LABEL[c]: round(float(weights[c] / wsum * Z[c][j]), 3) for c in Z},
                   "name": meta.get("name", ""), "subtype": meta.get("subtype", ""), "cone_lo": meta.get("cone_lo", ""),
                   "cone_hi": meta.get("cone_hi", ""), "surface": meta.get("surface", ""), "transparency": meta.get("transparency", ""),
                   "url": meta.get("url", ""), "attributes": attrs, "review_status": self.status.get(oid, ""),
                   "review_category": (tl.get(oid) or {}).get("category", ""), "reviewed": bool((tl.get(oid) or {}).get("reviewed_at")),
                   "palette": [[B_hex(I["ct_lab"][k][x]), round(float(I["ct_pct"][k][x]), 3)] for x in range(3) if I["ct_pct"][k][x] > 0]}
            if "reg" in q:
                row["colour_match"] = self.explain_colour(q["reg"], k)
            if q_attrs:
                row["attr_labels"] = {f: {"value": sorted(GA[f][0][k]) if f in MULTI_FEATS else GA[f][0][k], "source": GA[f][1][k]} for f in q_attrs}
            out.append(row)
        # automatic relevance (same rule as the report): colour dE00 <= 20 + the cone / surface / transparency entered
        auto = None
        if qattr is not None:
            auto = self.rule(qattr, [gal[j] for j in top])
            for r, a in zip(out, auto["per_result"]):
                r["relevant"] = a
        true = None
        tid = str(p.get("true_id") or "").strip()
        if tid:
            idx = np.nonzero(I["recipe_id"][gal] == tid)[0]
            if len(idx):
                best = idx[np.argmax(F[idx])]
                rk = int(1 + (F > F[best]).sum())
                true = {"id": tid, "rank": rk, "object": self.ids[gal[best]], **{f"Hit@{k}": rk <= k for k in (3, 5, 10)}}
            else:
                true = {"id": tid, "rank": None, "note": "no object of this recipe in the searched objects"}
        feats = self.feature_recall(ticked, q, q_attrs, out, p.get("colour_family", False))
        prio_used = {"shape": weights.get("shape", 0), "glaze": weights.get("appearance", 0) + weights.get("surface", 0),
                     "colour": weights.get("colour", 0)}
        ps = sum(prio_used.values())
        prio_used = {k: round(v / ps, 3) for k, v in prio_used.items()} if ps > 0 else None
        ev = self.evaluate(p, q, gal, F, prio_in, p.get("colour_family", False))
        for r_, e_ in zip(out, ev.pop("per_top")):
            r_["evaluation"] = e_
        custom = isinstance(p.get("priority"), dict) and any(abs(prio_in[k] - self.priority_of(mode, None)[k]) > 1e-3 for k in PRIORITY)
        return {"features": feats, "evaluation": ev, "priority": {"requested": {k: round(v, 3) for k, v in prio_in.items()},
                                                                   "used": prio_used, "custom": bool(custom),
                                                                   "colour_off": "colour" not in ticked},
                "mode": mode, "mode_label": MODE_LABEL[mode] + (" (custom priority)" if custom else ""),
                "weights_used": {COMP_LABEL[c]: round(v, 3) for c, v in weights.items() if v > 0},
                "gallery": int(len(gal)), "filters_applied": filters, "results": out, "true_recipe": true,
                "auto": None if auto is None else auto["summary"], "seconds": round(time.time() - t0, 2)}

    def feature_recall(self, ticked, q, q_attrs, out, family) -> dict:
        """relevance of every result per ticked feature + Hit@K / P@K (K = 3, 5, 10, 15, 20) per feature and combined.
        colour: the query's most important colour within dE00 20 of a region of the object (or, with 'colour family',
        the nearest named colours share the Color Lab V3 family); attributes: the object's best available label equals
        the query value (effects / defects: at least one in common; 'none' = no effect on both)."""
        from mm.colorlab import family_of, hex2lab
        per = {}
        for f in ticked:
            if f == "colour":
                if "reg" not in q:
                    continue
                rel = []
                for r in out:
                    cm = (r.get("colour_match") or [None])[0]
                    if cm is None:
                        rel.append(None)
                    elif family:
                        fq, fo = family_of([hex2lab(cm["query_hex"]), hex2lab(cm["object_hex"])])
                        rel.append(bool(cm["dE00"] <= 20 or fq == fo))
                    else:
                        rel.append(bool(cm["dE00"] <= 20))
                per[f] = rel
            elif f in q_attrs:
                per[f] = [self.attr_relevant(f, q_attrs[f], (frozenset(r["attr_labels"][f]["value"]) if f in MULTI_FEATS else r["attr_labels"][f]["value"]))
                          for r in out]
        if per:
            combo = []
            for k in range(len(out)):
                vals = [per[f][k] for f in per if per[f][k] is not None]
                combo.append(all(vals) if vals else None)
            per["all ticked"] = combo
        recall = {}
        for f, rel in per.items():
            r_ = [bool(x) for x in rel]
            recall[f] = {**{f"Hit@{k}": any(r_[:k]) for k in KS}, **{f"P@{k}": round(sum(r_[:k]) / k, 3) for k in KS}}
        src = {}
        if q_attrs:
            for r in out:
                for f, d in r["attr_labels"].items():
                    src.setdefault(f, {}).setdefault(d["source"], 0)
                    src[f][d["source"]] += 1
        return {"ticked": ticked, "query": {f: (sorted(v) if isinstance(v, (list, set, frozenset)) else v) for f, v in q_attrs.items()},
                "per_result": per, "recall": recall, "label_sources_top": src, "colour_family": bool(family)}

    def explain_colour(self, qreg, k) -> list:
        qlab, qpos, qarea, qauto, qhuman, qvalid = qreg[:6]
        w = B._weights(qarea, qauto, qhuman, qvalid, "reviewed")
        out = []
        for i in np.argsort(-w)[:3]:
            if not qvalid[i] or w[i] <= 0:
                continue
            gv = self.I["reg_valid"][k]
            d = B.de2000(qlab[i][None], self.I["reg_lab"][k][gv])
            j = int(np.argmin(d))
            gl = self.I["reg_lab"][k][gv][j]
            out.append({"query_hex": B_hex(qlab[i]), "weight": round(float(w[i]), 2), "object_hex": B_hex(gl),
                        "dE00": round(float(d[j]), 1), "object_area": round(float(self.I["reg_area"][k][gv][j]), 3)})
        return out

    # ------------------------------------------------------------ priority sliders
    def priority_of(self, mode, pr) -> dict:
        """the three slider values (shape / glaze / colour, adding up to 1): the user's, or the mode's own preset"""
        w = self.W["modes"][mode]["weights"]
        base = {"shape": w.get("shape", 0), "glaze": w.get("appearance", 0) + w.get("surface", 0), "colour": w.get("colour", 0)}
        src = base if not isinstance(pr, dict) else {k: max(0.0, float(pr.get(k, base[k]) or 0)) for k in PRIORITY}
        s = sum(src.values())
        return {k: v / s for k, v in src.items()} if s > 0 else {k: 1 / 3 for k in PRIORITY}

    def priority_weights(self, mode, prio: dict) -> dict:
        """slider values -> the four photo components; glaze / surface is split like the mode (Balanced 0.30 : 0.15)"""
        w = self.W["modes"][mode]["weights"]
        g = w.get("appearance", 0) + w.get("surface", 0)
        ra = w.get("appearance", 0) / g if g > 0 else 0.5
        return {"shape": prio["shape"], "appearance": prio["glaze"] * ra, "surface": prio["glaze"] * (1 - ra), "colour": prio["colour"]}

    # ------------------------------------------------------------ evaluation (separate from the ranking)
    def evaluate(self, p, q, gal, F, prio, family) -> dict:
        """per criterion (shape / glaze / colour) relevance of EVERY searched object (True / False / None = unknown), then
        Hit@K, P@K, Recall@K per criterion and for strict (all applicable criteria), weighted nDCG@K with the priority
        weights, and the true recipe's Recall@K. Labels: the best available one (human > Claude > Grounding DINO > earlier)."""
        from mm.claude_labels import SURFACE_GROUP
        ev = p.get("eval_values") or {}
        GA = self.gallery_attrs()
        n = len(gal)
        rel, rules, qv = {}, {}, {}
        # shape: object form
        qform = ev.get("form")
        if qform and qform not in ("unknown", "ambiguous"):
            vals = GA["form"][0]
            rel["shape"] = [None if not vals[k] else vals[k] == qform for k in gal]
            rules["shape"] = f"same object form: {qform}"
            qv["shape"] = qform
        # glaze / surface: surface group (+ effects the user picked)
        qs = ev.get("surface_finish")
        qg = SURFACE_GROUP.get(qs) if qs else None
        if qg:
            sv, effv = GA["surface_finish"][0], GA["glaze_effect"][0]
            eff_q = frozenset(x for x in (ev.get("glaze_effect") or []) if x not in ("none", "unknown")) if ev.get("effects_user") else frozenset()
            r = []
            for k in gal:
                g = SURFACE_GROUP.get(sv[k]) if sv[k] else None
                r.append(None if g is None else (g == qg and (not eff_q or bool(eff_q & effv[k]))))
            rel["glaze"] = r
            rules["glaze"] = f"same surface group: {qg} (your object: {qs})" + (f" + at least one of: {', '.join(sorted(eff_q))}" if eff_q else "")
            qv["glaze"] = qg + (" + " + ", ".join(sorted(eff_q)) if eff_q else "")
        # colour: the query's most important colour vs the object's closest colour region
        if "reg" in q:
            qlab, qpos, qarea, qauto, qhuman, qvalid = q["reg"][:6]
            wq = np.where(qvalid, B._weights(qarea, qauto, qhuman, qvalid, "reviewed"), -1)
            if qvalid.any() and wq.max() > 0:
                i = int(np.argmax(wq))
                L, V = self.I["reg_lab"][gal], self.I["reg_valid"][gal]
                d = np.where(V, B.de2000(np.asarray(qlab[i], np.float64)[None, None, :], L), np.inf)
                dmin, jmin = d.min(1), d.argmin(1)
                ok = np.isfinite(dmin)
                m = dmin <= 20
                if family:
                    from mm.colorlab import family_of
                    fq = family_of(np.asarray(qlab[i])[None])[0]
                    fo = np.array(family_of(L[np.arange(n), jmin]))
                    m = m | (fo == fq)
                rel["colour"] = [bool(m[t]) if ok[t] else None for t in range(n)]
                rules["colour"] = f"within ΔE2000 20 of {B_hex(qlab[i])}" + (" or the same colour family" if family else "")
                qv["colour"] = B_hex(qlab[i])
        crit = [c for c in PRIORITY if c in rel]
        # strict = every applicable criterion matches (a "?" is never a match); weighted = priority x matches
        if crit:
            strict = []
            for t in range(n):
                v = [rel[c][t] for c in crit]
                strict.append(True if all(x is True for x in v) else (False if any(x is False for x in v) else None))
            rel["strict"] = strict
            rules["strict"] = " AND ".join(crit)
        wsum = sum(prio[c] for c in crit) or 1.0
        wn = {c: prio[c] / wsum for c in crit}
        G = np.zeros(n, np.float32)
        for c in crit:
            G += wn[c] * np.array([x is True for x in rel[c]], np.float32)
        order = np.argsort(-F)
        metrics = {}
        for c, r in rel.items():
            arr = np.array([x is True for x in r])
            unk = np.array([x is None for x in r])
            yes = arr[order]
            n_rel = int(arr.sum())
            metrics[c] = {**{f"Hit@{k}": bool(yes[:k].any()) for k in EVAL_KS},
                          **{f"P@{k}": round(float(yes[:k].mean()), 3) for k in EVAL_KS},
                          **{f"Recall@{k}": (round(float(yes[:k].sum() / n_rel), 4) if n_rel else None) for k in EVAL_KS},
                          "relevant_in_gallery": n_rel, "unknown_in_gallery": int(unk.sum()), "unknown_top20": int(unk[order][:20].sum())}
        ndcg = {}
        if crit:
            disc = 1 / np.log2(np.arange(2, max(EVAL_KS) + 2))
            ideal = np.sort(G)[::-1]
            for k in EVAL_KS:
                idcg = float((ideal[:k] * disc[:k]).sum())
                ndcg[f"nDCG@{k}"] = round(float((G[order][:k] * disc[:k]).sum()) / idcg, 4) if idcg > 0 else None
        # the true recipe: its objects in the gallery are the complete relevant set -> real Recall@K
        true_recall = None
        tid = str(p.get("true_id") or "").strip()
        if tid:
            idx = np.nonzero(self.I["recipe_id"][gal] == tid)[0]
            if len(idx):
                pos = np.empty(n, int)
                pos[order] = np.arange(n)
                rk = pos[idx]
                true_recall = {"recipe": tid, "objects": int(len(idx)), "ranks": sorted(int(x) + 1 for x in rk),
                               **{f"Recall@{k}": round(float((rk < k).mean()), 3) for k in EVAL_KS}}
        sym = {True: "y", False: "n", None: "?"}
        per_top = [{**{c: sym[rel[c][j]] for c in rel}, "weighted": round(float(G[j]), 3)} for j in order[:int(p.get("top", 20))]]
        return {"criteria": crit, "rules": rules, "query_values": qv, "weights": {c: round(v, 3) for c, v in wn.items()},
                "metrics": metrics, "ndcg": ndcg, "true_recall": true_recall, "gallery": n, "per_top": per_top}

    def rule(self, qattr, ks) -> dict:
        I = self.I
        lab = qattr.get("lab")
        rel = []
        b = self.eng.bank
        for k in ks:
            rp = b.pos.get(str(I["recipe_id"][k]))
            comp, ok = 0, True
            if lab is None or np.isnan(lab).any() or I["ct_pct"][k][0] <= 0:
                rel.append(None)
                continue
            d = float(B.de2000(np.asarray(lab), I["ct_lab"][k][0]))
            colour_ok = d <= 20
            # recipe cone / surface / transparency: from the bank, or (bonus recipes) from the object's own index row
            if rp is not None:
                lo, hi, sf, tp = b.cone_lo[rp], b.cone_hi[rp], b.surf[rp], b.transp[rp]
            else:
                lo, hi, sf, tp = I["cone"][k][0], I["cone"][k][1], I["surf"][k], I["transp"][k]
            if qattr["lo"] >= 0 and lo >= 0:
                comp += 1
                ok &= bool(qattr["lo"] <= hi and lo <= qattr["hi"])
            if qattr["s"] >= 0 and sf >= 0:
                comp += 1
                ok &= bool(qattr["s"] == sf)
            if qattr["t"] >= 0 and tp >= 0:
                comp += 1
                ok &= bool(qattr["t"] == tp)
            rel.append(bool(colour_ok and ok and comp >= 1) if comp >= 1 else None)
        scored = [r for r in rel if r is not None]
        summ = None
        if len(scored) == len(rel) or len(scored) >= 3:
            arr = [bool(r) for r in rel]
            summ = {**{f"Hit@{k}": any(arr[:k]) for k in (3, 5, 10)}, **{f"P@{k}": round(sum(arr[:k]) / k, 3) for k in (3, 5, 10)}}
        return {"per_result": rel, "summary": summ}


def B_hex(lab) -> str:
    from mm.colorlab import lab2hex
    return lab2hex(lab)


def object_image(oid: str) -> Path:
    p = obj_file("smaller_object", f"{oid}.jpg")
    return p if p.exists() else obj_file("smaller_object", f"{oid}.jpg", root=BONUS_SEG)


def photo_file(pid: str) -> Path | None:
    """original photo of a 7k recipe or of a bonus photo (.jpg / .png)"""
    from mm.common import PHOTOS
    for p in (PHOTOS / f"{pid}.jpg", BONUS_SRC / f"{pid}.jpg", BONUS_SRC / f"{pid}.png"):
        if p.exists():
            return p
    return None
