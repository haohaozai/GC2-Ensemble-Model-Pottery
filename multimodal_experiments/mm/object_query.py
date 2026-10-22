"""Features of ONE object cut-out (RGBA), computed exactly as for the indexed objects (attributes_v1).

Used by the search app for new photos and by the synthetic-tile evaluation:
  app      multimodal image head (results/final_model.pt) on the object's DINOv2 (224, masked pooling) vector
  geom     silhouette geometry (mm/shape.py), standardised with the TRAINING statistics of the index, L2-normalised
  sil      DINOv2 CLS of the silhouette
  attr     attribute probabilities from the selected heads (work/attributes_v1/heads/*.joblib) + zero-shot form / texture
  reg      colour regions (mm/regions.py) -> region arrays for the colour similarity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import balanced as B
from . import regions as RG
from . import shape as SH


class ObjectEncoder:
    def __init__(self, work: Path, engine, deep=None, clip=None):
        import joblib
        from .attr_features import DinoExtras
        from .image_feats import Deep
        self.work, self.eng = work, engine
        self.deep = deep or getattr(engine, "deep", None) or Deep("dinov2")
        self.dx = DinoExtras(self.deep)
        self.clip = clip
        z = np.load(work / "balanced_index.npz", allow_pickle=False)
        tr = z["split"] == "train"
        g = np.load(work / "shape_geom.npz")
        gpos = {i: k for k, i in enumerate(g["ids"].tolist())}
        GX = g["X"]
        X = np.stack([GX[gpos[i]] for i in z["ids"][tr].tolist()])
        self.geom_mu, self.geom_sd = X.mean(0), X.std(0) + 1e-6
        self.heads = {p.stem: joblib.load(p) for p in (work / "heads").glob("*.joblib")}
        zt = np.load(work / "zeroshot_text.npz")
        self.zs = {k[:-3]: (zt[k[:-3] + "__classes"].tolist(), zt[k]) for k in zt.files if k.endswith("__E")}
        self.need = {h["features"] for h in self.heads.values()}

    # ------------------------------------------------------------ raw features
    def clip_embedding(self, rgba):
        from glazeprep.ceramic_tagger import CeramicTagger
        from .image_feats import padded_views
        if self.clip is None:
            from glazeprep.ceramic_clip import CeramicClip
            self.clip = CeramicClip()
        im = CeramicTagger.views(*padded_views(rgba))[0]
        return self.clip.image_embeddings([im]).float().cpu().numpy()[0]

    def feature_block(self, name, rgba, cache):
        if name in cache:
            return cache[name]
        if "+" in name:
            v = np.concatenate([self.feature_block(p, rgba, cache) for p in name.split("+")])
        elif name == "clip":
            v = self.clip_embedding(rgba)
        elif name == "dinov2":
            v = self.deep.encode([rgba])[0]
        elif name == "dinov2_hr":
            v = self.dx.hr([rgba])[0]
        elif name == "hand":
            from .image_feats import hand_features
            h = hand_features(rgba)
            st = self.heads_hand_stats()
            v = ((h - st[0]) / st[1] / np.sqrt(len(h))).astype(np.float32)
        else:
            raise ValueError(name)
        cache[name] = v
        return v

    def heads_hand_stats(self):
        if not hasattr(self, "_hand"):
            from .common import CACHE
            z = np.load(CACHE / "img_obj_hand.npz")
            X = z["X"]
            self._hand = (X.mean(0), X.std(0) + 1e-6)
        return self._hand

    # ------------------------------------------------------------ one object
    def encode(self, rgba: np.ndarray, importance: dict | None = None) -> dict:
        """rgba cut-out -> query dict for mm.balanced_eval.component_scores (+ readable details)."""
        from .attr_models import TASKS, zero_shot_scores
        from .query import _std
        mask = rgba[..., 3] >= 125
        cache = {}
        dv = self.feature_block("dinov2", rgba, cache)
        app = self.eng._head("image", _std((dv / max(1e-8, np.linalg.norm(dv)))[None], self.eng.img_mu, self.eng.img_sd))[0]
        geom = np.clip((SH.describe(mask) - self.geom_mu) / self.geom_sd, -5, 5)
        geom = geom / max(1e-6, np.linalg.norm(geom))
        sil = self.dx.silhouette([mask])[0]
        # attributes (+ full probability vectors, used by the Grounding DINO attribute models of attributes_v2)
        pred, full = {}, {}
        for task, h in self.heads.items():
            x = self.feature_block(h["features"], rgba, cache)[None]
            if h["type"] == "zero_shot":
                S = zero_shot_scores(x, h["E"], single=h["kind"] == "single")[0]
                cls = h["classes"]
                if task in ("surface_group", "transparency_group"):
                    continue
                order = [h["classes"].index(c) for c in TASKS[task][2]] if set(TASKS[task][2]) <= set(cls) else range(len(cls))
                P = S[list(order)]
            elif h["type"] in ("sklearn", "linear"):
                P = h["model"].predict_proba(x)[0]
                if h["kind"] == "single" and len(P) != len(h["classes"]):
                    full = np.zeros(len(h["classes"]))
                    full[h["model"].classes_] = P
                    P = full
            else:
                import torch
                with torch.no_grad():
                    o = h["net"](torch.tensor(x))
                    P = (torch.softmax(o, 1) if h["kind"] == "single" else torch.sigmoid(o)).numpy()[0]
            cls = TASKS[task][2]
            full[task] = {"classes": list(cls), "p": [float(v) for v in P]}
            pred[task] = [[cls[j], round(float(P[j]), 3)] for j in np.argsort(-P)[:4]]
            sel = h["selected"]
            if h["kind"] == "single":
                top = pred[task][0]
                pred[task + "_final"] = {"value": top[0] if top[1] >= sel.get("abstain_threshold", 0) else "unknown",
                                         "provenance": f"model ({sel['method']})"}
            else:
                thr = sel.get("thresholds", {})
                pred[task + "_final"] = {"value": [c for c, p in pred[task] if c in thr and p >= thr[c]], "provenance": f"model ({sel['method']})"}
        xc = self.feature_block("clip", rgba, cache)[None]
        for field in ("form", "texture"):
            cls, E = self.zs[field]
            S = zero_shot_scores(xc, E, single=True)[0]
            full[field] = {"classes": list(cls), "p": [float(v) for v in S]}
            pred[field] = [[cls[j], round(float(S[j]), 3)] for j in np.argsort(-S)[:3]]
            pred[field + "_final"] = {"value": pred[field][0][0], "provenance": "zero-shot CLIP"}
        attr = B.attr_vector(pred)
        # colour regions (+ the user's importance per region id)
        res, lbl = RG.analyse(rgba)
        rv = {"regions": {str(k): {"importance": v} for k, v in (importance or {}).items()}}
        reg = B.region_arrays(res, rv)
        return {"app": app, "geom": geom.astype(np.float32), "sil": sil, "attr": attr.astype(np.float32), "reg": reg,
                "details": {"regions": res, "labels": lbl, "attributes": pred, "attr_full": full}}
