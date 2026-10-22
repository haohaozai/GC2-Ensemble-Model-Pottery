"""Ceramic visual attribute tagger (v3) for the upgraded pipeline.

Zero-shot OpenCLIP with prompt ensembles, plus mask geometry for silhouette shape.
Taxonomy follows docs/CLAUDE_PIPELINE.md (derived from "craze documentaries.pdf").

What the scores mean
--------------------
* single-label attributes (texture, decoration, form, 3-D shape): softmax over the closed class set,
  using CLIP's own logit scale. This is a RELATIVE score among the offered descriptions, not a
  calibrated probability. A class is reported only if its relative score and its margin over the
  runner-up pass the thresholds below; otherwise the attribute is "Unknown" (abstained).
* multi-label attributes (glaze effect, defect): each class is contrasted with a neutral description
  ("a plain glaze without special effect"): presence = softmax([class, neutral])[0]. A class is
  reported when presence passes its threshold AND it ranks among the top classes of the attribute.
* nothing here is human-verified; `tags_human_verified` is always False in the outputs.

Intent rules (the PDF: crackle vs crazing and intentional vs accidental crawling cannot be told apart
from appearance alone): a visual crack network or crawling is reported as an effect or a defect only
when the recipe text (CSV Subtype / Description) states the intent; otherwise it is reported with the
qualifier "(intent ambiguous)".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

TEMPLATES = ["a photo of {}.", "a close-up photo of {}.", "{}."]

TEXTURE = {
    "Smooth": ["a ceramic piece with a smooth, even glaze surface",
               "smooth glossy glazed pottery with no bumps or grains",
               "a uniform smooth ceramic glaze"],
    "Rough": ["a rough, coarse ceramic surface",
              "an uneven rough stoneware surface",
              "rough unglazed clay"],
    "Granular": ["a granular glaze with visible grains",
                 "a sandy, grainy, stony ceramic texture",
                 "a speckled grainy matte stoneware surface"],
    "Dimpled": ["a dimpled glaze surface like orange peel",
                "a glaze covered with many small shallow dimples",
                "orange peel texture on a ceramic glaze"],
    "Ribbed": ["a ribbed ceramic surface with parallel ridges",
               "a test tile with parallel raised ridges and grooves",
               "throwing rings and grooves on pottery"],
    "Carved": ["a carved or incised ceramic surface",
               "engraved lines carved into pottery",
               "a stamped or carved relief pattern on ceramic"],
    "Pitted": ["a pitted glaze with many small pits and craters",
               "a cratered lava-like glaze surface",
               "a glaze surface full of tiny holes and craters"],
    "Wrinkled": ["a wrinkled glaze surface",
                 "a glaze with folds and ripples like wrinkled skin"],
    "Pebbled": ["a pebbled ceramic surface with rounded bumps",
                "a lumpy beaded glaze texture with raised droplets"],
}

GLAZE_EFFECT = {
    "Crystalline": ["a crystalline glaze with flower-like starburst crystals",
                    "large radiating crystals on a ceramic glaze",
                    "macrocrystalline glaze with crystal clusters"],
    "Oil Spot": ["an oil spot glaze with round metallic spots on a dark glaze",
                 "a dark tenmoku glaze covered with silvery oil spots"],
    "Speckled": ["a speckled glaze with tiny contrasting dots",
                 "a glaze sprinkled with small dark specks",
                 "a speckled stoneware glaze"],
    "Mottled": ["a mottled glaze with irregular patches of different colours",
                "a variegated glaze with cloud-like colour transitions"],
    "Dripping / Flowing": ["a glaze with downward drips and runs",
                           "a flowing glaze with streaks and rivulets",
                           "glaze running down and pooling at the bottom"],
    "Crackle": ["a crackle glaze with a network of fine cracks",
                "a crazed glaze with polygon-shaped crack lines",
                "fine interconnected crack lines across a glaze"],
    "Breaking": ["a glaze that changes colour on raised edges and ridges",
                 "a glaze breaking to a lighter colour over texture"],
    "Metallic / Iridescent": ["a metallic glaze with a metal-like sheen",
                              "an iridescent glaze with rainbow reflections",
                              "a gold or copper lustre glaze"],
    "Layered": ["two different glazes layered over each other",
                "a glaze combination with overlapping layers of colour"],
    "Marbled": ["a marbled glaze with swirling patterns",
                "swirled marble-like colours on ceramic"],
}
EFFECT_NEUTRAL = ["a plain glaze of a single even colour without any special effect",
                  "a simple uniform ceramic glaze"]

DEFECT = {
    "Pinholes": ["a glaze with small pinholes", "tiny round holes penetrating a glaze layer"],
    "Blistering": ["a glaze with raised bubbles and burst blisters", "a blistered glaze surface"],
    "Crawling": ["a crawled glaze pulled into islands exposing bare clay",
                 "a glaze that beaded up leaving bare clay patches"],
    "Crazing": ["a crazed glaze with a network of fine cracks",
                "fine crack lines across a glaze surface"],
    "Pitting": ["a glaze with small craters and pits", "a pitted glaze defect"],
    "Shivering": ["glaze flaking off the edge of a ceramic piece", "chipped glaze peeling away at the rim"],
    "Uneven Coverage": ["a glaze with uneven, patchy coverage and thin bare areas",
                        "an unevenly applied glaze with missing patches"],
    "Glaze Running": ["glaze that ran down and dripped off the bottom of the pot",
                      "excessive glaze run pooling at the foot"],
}
DEFECT_NEUTRAL = ["a flawless glaze without visible defects", "a clean, well-fired glaze surface"]

DECORATION = {
    # Test tiles very often carry a handwritten / stamped identification label (marker, pencil, oxide
    # numbers). That is not decoration; without these prompts CLIP called all such tiles "Calligraphy"
    # (16 of 16 sampled Calligraphy predictions were labels).
    "Plain": ["a plain undecorated ceramic piece with a single glaze",
              "a glaze test tile without any painted design",
              "pottery with natural glaze colour variation but no painted decoration",
              "a glaze test tile with a handwritten label or number written on it",
              "a ceramic test tile with marker writing on the bare clay",
              "a test tile with a stamped or scratched identification mark",
              # natural glaze phenomena that CLIP otherwise reads as Dotted / Striped / Abstract
              "a pottery glaze with natural dark speckles from the clay body",
              "a glazed pot with natural colour variation, runs and drips of glaze",
              "a glossy glaze with bright light reflections",
              "throwing lines and ridges on a plain glazed pot"],
    "Floral": ["pottery decorated with painted flowers", "a ceramic piece with a floral pattern"],
    "Geometric": ["pottery decorated with geometric patterns", "a ceramic piece with painted triangles and lines"],
    "Abstract": ["pottery with abstract painted brushstrokes", "a ceramic piece with an abstract painted design"],
    "Striped": ["pottery decorated with painted stripes", "a ceramic piece with bands of colour"],
    "Landscape": ["pottery painted with a landscape scene"],
    "Animal": ["pottery decorated with animal pictures", "a ceramic piece painted with birds or fish"],
    "Calligraphy": ["Chinese or Japanese calligraphy characters brush-painted on a ceramic surface",
                    "ornamental painted lettering as part of the decoration of pottery"],
    "Botanical": ["pottery decorated with painted leaves and plants"],
    "Figurative": ["pottery decorated with human figures"],
    "Dotted": ["pottery decorated with regular painted dots"],
    "Mixed": ["pottery with several kinds of painted decoration combined"],
}

FORM = {
    "Bowl": ["a ceramic bowl", "a pottery bowl"],
    "Vase": ["a ceramic vase", "a tall pottery vase"],
    "Mug": ["a ceramic mug with a handle"],
    "Cup": ["a ceramic cup without a handle", "a small tea cup"],
    "Plate": ["a ceramic plate", "a flat round pottery plate"],
    "Teapot": ["a ceramic teapot with a spout"],
    "Jar": ["a ceramic jar", "a lidded pottery jar"],
    "Pitcher": ["a ceramic pitcher with a spout and handle"],
    "Planter": ["a ceramic flower pot planter"],
    "Sculpture": ["a ceramic sculpture", "a sculptural ceramic figure"],
    "Tile": ["a flat square ceramic tile"],
    "Test Tile": ["a glaze test tile", "a small rectangular ceramic glaze sample tile",
                  "a vertical ceramic test tile with a hole for hanging"],
    "Dish": ["a shallow ceramic dish", "a small shallow pottery dish"],
    "Bottle": ["a ceramic bottle with a narrow neck"],
    "Vessel": ["a ceramic vessel", "a round pottery vessel"],
    "Fragment": ["a broken pottery shard", "a small piece of broken ceramic"],
}

SHAPE3D = {
    "Cylindrical": ["a cylindrical ceramic vessel with straight sides"],
    "Spherical": ["a round spherical ceramic vessel"],
    "Conical": ["a conical ceramic vessel that tapers"],
}
DEPTH = {
    "Shallow": ["a shallow, wide ceramic bowl"],
    "Deep": ["a deep ceramic bowl with high walls"],
}

DEFAULT_THRESHOLDS = {
    "single_min_score": {"texture": 0.35, "decoration": 0.45, "form": 0.30, "shape3d": 0.50, "depth": 0.60},
    "single_min_margin": {"texture": 0.10, "decoration": 0.15, "form": 0.08, "shape3d": 0.15, "depth": 0.20},
    # Presence thresholds (class vs neutral contrast) and rank limit for multi-label attributes.
    # These are fallbacks only: the pipeline loads configs/tagger_thresholds.json, whose per-class values
    # (weakly labelled classes) and defaults (median of those) were chosen on the dev half by
    # scripts/11_eval_tagger_weak_labels.py at the population prevalence of each effect (0.985-0.9985).
    "effect_presence": 0.95,
    "effect_presence_per_class": {},
    "effect_max_labels": 2,
    # painted decoration must beat "Plain" clearly: natural glaze variation (runs, speckle, colour
    # breaks) is easily read as decoration (the PDF asks to keep the two apart)
    "decoration_min_score_nonplain": 0.70,
    "decoration_min_gap_over_plain": 0.35,
    "defect_presence": 0.95,
    "defect_presence_per_class": {},
    "defect_max_labels": 2,
    "min_object_side_px": 48,
    "truncated_form_min_score": 0.50,
}

FLAT_FORMS = {"Plate", "Tile", "Test Tile", "Dish"}
DEPTH_FORMS = {"Bowl", "Dish", "Planter"}
VESSEL_FORMS = {"Vase", "Bottle", "Jar", "Cup", "Mug", "Pitcher", "Vessel", "Teapot", "Planter"}


def _norm(x):
    return torch.nn.functional.normalize(x.float(), dim=-1)


class CeramicTagger:
    def __init__(self, clip, thresholds_path: str | Path | None = None):
        """clip: glazeprep.ceramic_clip.CeramicClip (shared model instance)."""
        self.clip = clip
        self.th = json.loads(json.dumps(DEFAULT_THRESHOLDS))
        self.thresholds_source = "defaults (glazeprep/ceramic_tagger.py)"
        if thresholds_path and Path(thresholds_path).exists():
            loaded = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
            for k, v in loaded.get("thresholds", loaded).items():
                if isinstance(v, dict) and isinstance(self.th.get(k), dict):
                    self.th[k].update(v)
                else:
                    self.th[k] = v
            self.thresholds_source = str(thresholds_path)
        self.sets = {
            "texture": TEXTURE, "decoration": DECORATION, "form": FORM,
            "shape3d": SHAPE3D, "depth": DEPTH,
        }
        self.class_emb = {name: self._class_embeddings(spec) for name, spec in self.sets.items()}
        self.effect_emb = self._class_embeddings(GLAZE_EFFECT)
        self.effect_neutral = self._phrase_embedding(EFFECT_NEUTRAL)
        self.defect_emb = self._class_embeddings(DEFECT)
        self.defect_neutral = self._phrase_embedding(DEFECT_NEUTRAL)

    # ------------------------------------------------------------------ text side
    def _phrase_embedding(self, phrases):
        texts = [t.format(ph) for ph in phrases for t in TEMPLATES]
        e = self.clip.text_embeddings(texts)
        return _norm(e.mean(0, keepdim=True))[0]

    def _class_embeddings(self, spec: dict):
        names = list(spec)
        emb = torch.stack([self._phrase_embedding(spec[n]) for n in names])
        return names, emb

    # ------------------------------------------------------------------ image side
    @staticmethod
    def views(rgb: np.ndarray, mask: np.ndarray):
        """object view (object on white, tight crop) and texture view (largest interior square at
        native resolution, so fine crackle / speckle survive CLIP's 224 px input)."""
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = int(0.04 * max(y1 - y0, x1 - x0))
        H, W = mask.shape
        y0, y1, x0, x1 = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
        crop = rgb[y0:y1, x0:x1].copy()
        crop[~mask[y0:y1, x0:x1]] = 255
        obj = Image.fromarray(crop)
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        half = int(min(max(dist[cy, cx] / np.sqrt(2), 16), 256))
        ty0, ty1 = max(0, cy - half), min(H, cy + half)
        tx0, tx1 = max(0, cx - half), min(W, cx + half)
        tex = Image.fromarray(rgb[ty0:ty1, tx0:tx1].copy()) if (ty1 - ty0) >= 16 and (tx1 - tx0) >= 16 else obj
        return obj, tex

    # ------------------------------------------------------------------ scoring helpers
    def _single(self, name, img_emb):
        names, emb = self.class_emb[name]
        prob = (self.clip.logit_scale * img_emb @ emb.T).softmax(-1).cpu().numpy().reshape(-1)
        order = np.argsort(-prob)
        top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        return {"scores": {names[i]: round(float(prob[i]), 4) for i in order},
                "top": names[top], "top_score": float(prob[top]),
                "runner_up": names[second], "margin": float(prob[top] - prob[second])}

    def _multi(self, names_emb, neutral, img_emb):
        names, emb = names_emb
        s = (img_emb @ emb.T)[0]
        sn = (img_emb @ neutral[:, None])[0, 0]
        pres = torch.sigmoid(self.clip.logit_scale * (s - sn)).cpu().numpy()   # == 2-way softmax
        rel = (self.clip.logit_scale * s).softmax(-1).cpu().numpy()
        return {names[i]: {"presence": round(float(pres[i]), 4), "relative": round(float(rel[i]), 4)}
                for i in range(len(names))}

    def _decide_single(self, name, res):
        th = self.th
        ok = res["top_score"] >= th["single_min_score"][name] and res["margin"] >= th["single_min_margin"][name]
        return (res["top"] if ok else "Unknown"), ("predicted" if ok else
                f"abstained: top '{res['top']}' {res['top_score']:.2f}, margin {res['margin']:.2f} below threshold")

    def _decide_multi(self, scores, presence, per_class, max_labels):
        ranked = sorted(scores.items(), key=lambda kv: -kv[1]["presence"])
        out = []
        for cls, sc in ranked[:max_labels]:
            if sc["presence"] >= per_class.get(cls, presence):
                out.append(cls)
        return out

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def geometry(mask: np.ndarray):
        H, W = mask.shape
        ys, xs = np.where(mask)
        band = max(2, round(0.01 * min(H, W)))
        borders = int(mask[:band].any()) + int(mask[-band:].any()) + int(mask[:, :band].any()) + int(mask[:, -band:].any())
        m8 = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(c)) or float(mask.sum())
        hull = cv2.convexHull(c)
        solidity = area / max(1.0, cv2.contourArea(hull))
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        rectangularity = area / max(1.0, rw * rh)
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        ellipse_fit = 0.0
        if len(c) >= 5:
            (_, _), (ea, eb), _ = cv2.fitEllipse(c)
            ell = np.zeros_like(m8)
            cv2.ellipse(ell, cv2.fitEllipse(c), 1, -1)
            inter = float((ell.astype(bool) & mask).sum())
            union = float((ell.astype(bool) | mask).sum())
            ellipse_fit = inter / max(1.0, union)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        bh, bw = y1 - y0, x1 - x0
        sub = mask[y0:y1, x0:x1]
        mirror_iou = float((sub & sub[:, ::-1]).sum()) / max(1.0, float((sub | sub[:, ::-1]).sum()))

        def width_at(fr):
            row = sub[min(bh - 1, max(0, int(fr * bh)))]
            return float(row.sum())
        top_w, bottom_w = width_at(0.12), width_at(0.88)
        return {"touches_borders": borders, "solidity": round(solidity, 3),
                "rectangularity": round(rectangularity, 3), "aspect_ratio": round(aspect, 3),
                "ellipse_fit_iou": round(ellipse_fit, 3), "mirror_iou": round(mirror_iou, 3),
                "bbox_h_over_w": round(bh / max(1, bw), 3),
                "flare_ratio": round(top_w / max(1.0, bottom_w), 3),
                "bbox_px": [int(x0), int(y0), int(x1), int(y1)]}

    @staticmethod
    def silhouette_shape(g):
        out = []
        if g["rectangularity"] >= 0.85:
            out.append("Square" if g["aspect_ratio"] < 1.25 else "Rectangular")
        elif g["ellipse_fit_iou"] >= 0.92:
            out.append("Round" if g["aspect_ratio"] < 1.15 else "Oval")
        if g["bbox_h_over_w"] >= 1.8:
            out.append("Tall / Slender")
        elif g["bbox_h_over_w"] <= 0.62 or g["flare_ratio"] >= 1.3:
            out.append("Wide / Flared")
        if g["solidity"] < 0.85:
            out.append("Irregular")
        elif g["mirror_iou"] < 0.85 and not out[:1] == ["Rectangular"]:
            out.append("Asymmetrical")
        return out

    # ------------------------------------------------------------------ main
    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, mask: np.ndarray, context: dict | None = None) -> dict:
        """rgb: full image, mask: one object (full resolution). context: recipe fields + segmentation info."""
        context = context or {}
        th = self.th
        g = self.geometry(mask)
        x0, y0, x1, y1 = g["bbox_px"]
        small = min(x1 - x0, y1 - y0) < th["min_object_side_px"]
        truncated = g["touches_borders"] >= 2 or context.get("dominant_close_up", False)
        obj, tex = self.views(rgb, mask)
        e_obj, e_tex = self.clip.image_embeddings([obj, tex])
        e_obj, e_tex = e_obj[None], e_tex[None]
        # effects / defects use both views (best weak-label AUROC for 4 of 5 classes); texture uses the
        # object view (crater/pitted AUROC 0.835 vs 0.777 with both views, see scripts/11_...)
        e_surf = _norm(e_obj + e_tex)
        out = {"geometry": g, "object_truncated_by_frame": bool(truncated), "object_too_small": bool(small),
               "scores": {}, "status": {}}
        if small:
            for k in ("texture", "glaze_effect", "defect", "decoration", "form", "shape"):
                out[k] = ["Unknown"] if k in ("glaze_effect", "defect", "shape") else "Unknown"
                out["status"][k] = f"abstained: object smaller than {th['min_object_side_px']} px"
            return out

        # texture / decoration / form (single label)
        for name, emb in (("texture", e_obj), ("decoration", e_obj), ("form", e_obj)):
            res = self._single(name, emb)
            out["scores"][name] = res
            out[name], out["status"][name] = self._decide_single(name, res)
        dec = out["scores"]["decoration"]
        if out["decoration"] not in ("Plain", "Unknown"):
            plain = dec["scores"].get("Plain", 0.0)
            if (dec["top_score"] < th["decoration_min_score_nonplain"]
                    or dec["top_score"] - plain < th["decoration_min_gap_over_plain"]):
                out["status"]["decoration"] = (f"abstained: '{out['decoration']}' {dec['top_score']:.2f} vs Plain "
                                               f"{plain:.2f}; painted decoration vs natural glaze variation unclear")
                out["decoration"] = "Unknown"
        if out["decoration"] == "Calligraphy":
            out["status"]["decoration"] = ("predicted; written text is visible - on test pieces this is usually an "
                                           "identification label, not decoration (review)")
        if truncated and out["form"] != "Unknown":
            f = out["scores"]["form"]
            if not (out["form"] in ("Test Tile", "Tile") and f["top_score"] >= th["truncated_form_min_score"]):
                out["status"]["form"] = (f"abstained: object extends beyond the photo (top '{out['form']}' "
                                         f"{f['top_score']:.2f}); form not determinable from a partial view")
                out["form"] = "Unknown"

        # glaze effects and defects (multi label)
        eff = self._multi(self.effect_emb, self.effect_neutral, e_surf)
        dfc = self._multi(self.defect_emb, self.defect_neutral, e_surf)
        out["scores"]["glaze_effect"] = eff
        out["scores"]["defect"] = dfc
        effects = self._decide_multi(eff, th["effect_presence"], th["effect_presence_per_class"], th["effect_max_labels"])
        defects = self._decide_multi(dfc, th["defect_presence"], th["defect_presence_per_class"], th["defect_max_labels"])
        effects, defects, notes = self._apply_intent(effects, defects, context)
        out["glaze_effect"] = effects or ["None detected"]
        out["defect"] = defects or ["None detected"]
        out["defect_intent_notes"] = notes
        out["status"]["glaze_effect"] = "predicted (multi-label)"
        out["status"]["defect"] = "predicted (multi-label; defects always need review)"

        # shape: silhouette geometry + form-conditional 3-D descriptors
        shape = []
        if truncated:
            out["status"]["shape"] = "silhouette incomplete (object extends beyond the photo)"
        else:
            shape += self.silhouette_shape(g)
            out["status"]["shape"] = "predicted (silhouette geometry + CLIP 3-D descriptor)"
        form = out["form"]
        if form in FLAT_FORMS:
            shape.append("Flat")
        if form in DEPTH_FORMS:
            res = self._single("depth", e_obj)
            out["scores"]["depth"] = res
            lab, _ = self._decide_single("depth", res)
            if lab != "Unknown":
                shape.append(lab)
        if form in VESSEL_FORMS and not truncated:
            res = self._single("shape3d", e_obj)
            out["scores"]["shape3d"] = res
            lab, _ = self._decide_single("shape3d", res)
            if lab != "Unknown":
                shape.append(lab)
        out["shape"] = list(dict.fromkeys(shape)) or ["Unknown"]
        return out

    # ------------------------------------------------------------------ intent
    @staticmethod
    def _apply_intent(effects, defects, ctx):
        subtype = (ctx.get("subtype_csv") or "").lower()
        desc = (ctx.get("description_csv") or "").lower()
        notes = []
        crack_seen = "Crackle" in effects or "Crazing" in defects
        if crack_seen:
            effects = [e for e in effects if e != "Crackle"]
            defects = [d for d in defects if d != "Crazing"]
            if "crackle" in subtype or "crackle" in desc:
                effects.append("Crackle")
                notes.append("crack network: intentional crackle per recipe text")
            elif re.search(r"\bcraz", desc):
                defects.append("Crazing")
                notes.append("crack network: recipe text reports crazing")
            else:
                effects.append("Crackle")
                defects.append("Crazing (intent ambiguous)")
                notes.append("crack network visible; crackle vs crazing cannot be decided from the image")
        if "Crawling" in defects:
            defects = [d for d in defects if d != "Crawling"]
            if "crawl" in subtype or "crawl" in desc or "lichen" in desc:
                notes.append("crawling: intentional effect per recipe text (not reported as a defect)")
                effects.append("Crawling effect (intentional, per CSV)")
            else:
                defects.append("Crawling (intent ambiguous)")
                notes.append("crawling-like islands visible; intent unknown")
        if "Pitting" in defects and ("crater" in subtype or "crater" in desc or "lava" in desc):
            defects = [d for d in defects if d != "Pitting"]
            notes.append("pitting: intentional crater glaze per recipe text")
        return effects, defects, notes
