"""Attribute re-tagging rules (claude_v3.2-retag), applied after the segmentation + tagging run.

No model runs here; the inputs are the stored v3.1 outputs (tag_json), the recipe CSV fields and, for
Surface / Transparency fallbacks, a probability vector produced by scripts/16_retag_attributes.py.

Source policy (Glaze_Effects_Defects_Surface_Transparency_Reference_v2.pdf, sections 4 and 6-8):
  * Surface / Transparency: a valid CSV value is kept exactly and shown as "<value> (CSV)". When the CSV is
    blank, a value is auto-labelled only with visual evidence; recipe text can support it but never
    decides an optical property on its own. Otherwise "Unknown (CSV blank; inconclusive)".
  * Glaze effect / Defect: every object gets an explicit value with its source:
      "<class> (CSV subtype)"         the recipe family names the effect (intended appearance)
      "<class> (CSV description)"     the recipe text mentions it (not negated); may not show in this photo
      "Risk: <defect> (CSV description, conditional)"   the text names it as a risk ("if applied too thick")
      "<class> (visual)"              zero-shot score passed the strict v3.1 threshold
      "Possible: <class> (visual, low confidence)"   score in the top 3 % of that class over all objects
      "None visible (checked)"        image adequate, no evidence of any kind
      "Unknown (...)"                 image inadequate (object too small / no ceramic object)
  * intent rules (craze documentaries.pdf; reference v2 section 3): crackle vs crazing, crawling effect vs
    defect, crater vs pitting, half-dipped test tiles vs uneven coverage.
"""
from __future__ import annotations

import re

TAG_VERSION = "claude_v3.2-retag"

SURFACE_VOCAB = ["Glossy", "Semi-glossy", "Satin", "Satin-matte", "Semi-matte", "Smooth Matte", "Matte",
                 "Stony Matte", "Dry Matte"]
TRANSPARENCY_VOCAB = ["Transparent", "Translucent", "Semi-opaque", "Opaque"]
# fine CSV classes overlap visually (reference v2 section 6: "abstain rather than force a label"); the image
# fallback therefore predicts coarse groups only
SURFACE_GROUPS = {"G": ("Glossy / Semi-glossy", {"Glossy", "Semi-glossy"}),
                  "S": ("Satin / Semi-matte", {"Satin", "Satin-matte", "Semi-matte"}),
                  "M": ("Matte", {"Matte", "Smooth Matte", "Stony Matte", "Dry Matte"})}
TRANSPARENCY_GROUPS = {"T": ("Transparent / Translucent", {"Transparent", "Translucent"}),
                       "O": ("Semi-opaque / Opaque", {"Semi-opaque", "Opaque"})}


def surface_group(value):
    for k, (_, members) in SURFACE_GROUPS.items():
        if value in members:
            return k
    return None


def transparency_group(value):
    for k, (_, members) in TRANSPARENCY_GROUPS.items():
        if value in members:
            return k
    return None


CRATER_TEXTURE = {"Pitting", "Blistering", "Pinholes"}   # intended surface of crater / lava glazes

# --------------------------------------------------------------------------- recipe text evidence
NEGATION = re.compile(
    r"\b(no|not|non|never|without|free|doesn'?t|does not|won'?t|will not|didn'?t|did not|isn'?t|is not|"
    r"aren'?t|are not|wasn'?t|was not|hardly|barely|prevent\w*|avoid\w*|reduc\w*|eliminat\w*|fix\w*|"
    r"correct\w*|resist\w*|minimi[sz]\w*|less|zero|anti|instead of|cure\w*|stop\w*|solv\w*|remov\w*)"
    r"\b[^.;!?,]{0,15}$")
NEGATION_AFTER = re.compile(r"^[\w-]{0,3}[- ]?(free|resistant|proof)\b")
# "can cause pinholes if applied too thick": a risk described by the author, not an observation
CONDITIONAL = re.compile(r"\b(if|when|can|could|may|might|tends? to|prone|risk|careful|otherwise|too thick|"
                         r"too thin|unless)\b")

EFFECT_TEXT = {
    # "crystalline hold / schedule / firing" describes a firing programme, not a visible effect
    "Crystalline": r"\bcrystalline\b(?! (hold|schedule|firing|cycle|program|cool))|\b(macro|micro)[- ]?crystal\w*"
                   r"|\bcrystals\b|\bcrystal (growth|formation|clusters?)\b",
    "Oil Spot": r"\boil[- ]?spots?\b",
    "Speckled": r"\bspeckl(e|ed|es|ing)\b|\bflecks?\b|\bflecked\b",
    "Mottled": r"\bmottl(e|ed|ing)\b|\bvariegat(ed|ion)\b",
    "Dripping / Flowing": r"\b(runs|runny|running|drips|dripping|drippy|flows|flowing|flowy|fluid)\b",
    "Crackle": r"\bcrackl(e|ed|es|ing)\b",
    "Breaking": r"\bbreaking\b|\bbreaks\b[^.;]{0,30}\b(edges?|rims?|ridges?|texture|over|lighter|darker|brown|white)\b",
    "Metallic / Iridescent": r"\bmetallic\b|\blust(er|re)\b|\biridescen\w*",
    "Layered": r"\blayered\b|\blayering\b",
    "Marbled": r"\bmarbl(e|ed|ing)\b",
}
DEFECT_TEXT = {
    "Crazing": r"\bcraz(e|es|ed|ing)\b",
    "Crawling": r"\bcrawl(s|ed|ing)?\b",
    "Pinholes": r"\bpin[- ]?hol(e|es|ed|ing)\b",
    "Blistering": r"\bbliste?r(s|ed|ing)?\b",
    "Pitting": r"\bpitt(ed|ing)\b",
    "Shivering": r"\bshiver(s|ed|ing)?\b",
    "Glaze Running": r"\b(runs|ran|run|drips?|dripped) off\b|\b(onto|on to|stuck to) the (kiln )?shelf\b",
}
SURFACE_TEXT = {  # coarse surface group named in recipe text
    "G": r"\bglossy\b|\bgloss\b|\bshiny\b|\bsemi[- ]?gloss\w*",
    "S": r"\bsatin\w*|\bsilky\b|\bsemi[- ]?matte?\b",
    "M": r"(?<!semi[- ])(?<!satin[- ])\bmatte?\b|\bdry[- ]matte?\b|\bdry (surface|glaze)\b|\bstony\b",
}
# 'clear' is specific in a recipe name or Subtype ("Glaze - Clear"); in free text it often only names the
# base a coloured glaze was made from, so descriptions use the explicit words only
TRANSPARENCY_TEXT_SHORT = {"T": r"\bclear\b|\btransparent\b|\btranslucent\b", "O": r"\bopaque\b"}
TRANSPARENCY_TEXT_LONG = {"T": r"\btransparent\b|\btranslucent\b", "O": r"\bopaque\b"}


def find_mentions(text: str, pattern: str):
    """Matches of `pattern` in `text`: (positive, conditional, negated) snippet lists."""
    pos, cond, neg = [], [], []
    if not text:
        return pos, cond, neg
    low = text.lower()
    for m in re.finditer(pattern, low):
        before = low[max(0, m.start() - 60):m.start()]
        before = re.split(r"[.;!?,\n]", before)[-1]
        after = low[m.end():m.end() + 12]
        sent_after = re.split(r"[.;!?\n]", low[m.end():m.end() + 60])[0]
        snippet = text[max(0, m.start() - 30):m.end() + 30].replace("\n", " ").strip()
        if NEGATION.search(before) or NEGATION_AFTER.search(after):
            neg.append(snippet)
        elif CONDITIONAL.search(before) or CONDITIONAL.search(sent_after):
            cond.append(snippet)
        else:
            pos.append(snippet)
    return pos, cond, neg


def text_classes(text: str, table: dict, keep_conditional=False):
    """{class: snippet} for positive mentions; conditional mentions go to '_conditional' (or count as
    positive with keep_conditional), negated ones to '_negated'."""
    out = {}
    for cls, pat in table.items():
        pos, cond, neg = find_mentions(text, pat)
        if pos or (cond and keep_conditional):
            out[cls] = (pos or cond)[0]
        elif cond:
            out.setdefault("_conditional", {})[cls] = cond[0]
        elif neg:
            out.setdefault("_negated", {})[cls] = neg[0]
    return out


def subtype_effects(subtype: str):
    s = (subtype or "").lower()
    out = []
    if "crystalline" in s:
        out.append("Crystalline")
    if "oil spot" in s:
        out.append("Oil Spot")
    if "specialty - crackle" in s:
        out.append("Crackle")
    if "metallic" in s:
        out.append("Metallic / Iridescent")
    if "crawling" in s:
        out.append("Crawling effect")
    return out


def text_group(fields: dict, tables: dict):
    """Coarse group named in recipe text. tables: {"name": table, "subtype": table, "description": table}
    (Name / Subtype are short and specific; Description is free text). Returns (group or None, evidence)."""
    hits = {}
    for key in ("name", "subtype", "description"):
        found = text_classes(fields.get(key) or "", tables[key], keep_conditional=True)
        for g, snip in found.items():
            if g.startswith("_"):
                continue
            hits.setdefault(g, f"{key}: '{snip}'")
    groups = sorted(hits)
    if len(groups) == 1:
        return groups[0], hits[groups[0]]
    if len(groups) > 1:
        return None, "conflicting recipe text: " + " | ".join(hits[g] for g in groups)
    return None, ""


# --------------------------------------------------------------------------- surface / transparency
def resolve_optical(csv_value: str, vocab, groups, probs: dict | None, text_g, text_ev, th: dict):
    """Returns dict(value, label, source, evidence). probs: {group: probability} from the image classifier.
    th: {"visual": p_min for image-only, "agree": p_min when recipe text names the same group}."""
    v = (csv_value or "").strip()
    if v:
        return {"value": f"{v} (CSV)", "label": v, "source": "CSV",
                "evidence": "CSV value" + ("" if v in vocab else " (not in the standard vocabulary)")}
    if not probs:
        return {"value": "Unknown (CSV blank; no image evidence)", "label": "Unknown", "source": "UNKNOWN",
                "evidence": text_ev or "no usable image"}
    top = max(probs, key=probs.get)
    p = probs[top]
    pv = ", ".join(f"{groups[g][0]} {probs[g]:.2f}" for g in sorted(probs, key=lambda g: -probs[g]))
    ev = f"image classifier: {pv}"
    if text_ev:
        ev += f"; recipe text: {text_ev}"
    name = groups[top][0]
    if text_g == top and p >= th["agree"]:
        return {"value": f"{name} (auto: visual + recipe text; CSV blank)", "label": name,
                "source": "AUTO_VISUAL+TEXT", "evidence": ev}
    if text_g is not None and text_g != top:
        return {"value": f"Unknown (CSV blank; image suggests {name}, recipe text suggests {groups[text_g][0]})",
                "label": "Unknown", "source": "UNKNOWN_CONFLICT", "evidence": ev}
    if p >= th["visual"]:
        return {"value": f"{name} (auto: visual; CSV blank)", "label": name, "source": "AUTO_VISUAL", "evidence": ev}
    return {"value": f"Unknown (CSV blank; best image guess {name}, low confidence)", "label": "Unknown",
            "source": "UNKNOWN", "evidence": ev}


# --------------------------------------------------------------------------- effects / defects
def _fmt(items):
    return "; ".join(items)


def resolve_effects_defects(rec: dict, possible_th: dict, form: str):
    """rec: tag_json record (v3.1 fields). possible_th: {"effect": {cls: th}, "defect": {cls: th}}.
    Returns dict with display strings, label lists, sources and evidence for effect and defect."""
    subtype = rec.get("subtype_csv") or ""
    s_low = subtype.lower()
    desc = rec.get("description_csv") or ""
    name = rec.get("recipe_name") or ""
    text = f"{name}. {desc}"
    scores = rec.get("visual_tag_scores") or {}
    eff_sc = scores.get("glaze_effect") or {}
    def_sc = scores.get("defect") or {}
    v31_eff = [e for e in (rec.get("glaze_effect_list") or []) if e not in ("None detected", "Unknown")]
    v31_def = [d for d in (rec.get("defect_list") or []) if d not in ("None detected", "Unknown")]
    small = str(rec.get("texture_status", "")).startswith("abstained: object smaller")
    crawl_intended = "crawl" in s_low or bool(re.search(r"\bcrawl|\blichen", desc.lower()))
    crater_intended = "crater" in s_low or bool(re.search(r"\bcrater|\blava\b", desc.lower()))
    crackle_intended = "crackle" in s_low or bool(re.search(r"\bcrackl", desc.lower()))
    test_tile = form in ("Test Tile", "Tile")

    eff_items, eff_labels, eff_src, eff_ev = [], [], [], []
    def_items, def_labels, def_src, def_ev = [], [], [], []

    def add(items, labels, srcs, cls, src, display):
        if cls in labels:
            return
        items.append(display)
        labels.append(cls)
        srcs.append(src)

    # 1. recipe family (intended appearance)
    for cls in subtype_effects(subtype):
        disp = "Crawling effect (intended, CSV subtype)" if cls == "Crawling effect" else f"{cls} (CSV subtype)"
        add(eff_items, eff_labels, eff_src, cls, "CSV_SUBTYPE", disp)
        eff_ev.append(f"{cls}: Subtype '{subtype}'")

    # 2. confident visual (v3.1 decisions, intent rules already applied there)
    for e in v31_eff:
        cls = "Crawling effect" if e.startswith("Crawling effect") else e
        if cls in eff_labels:
            i = eff_labels.index(cls)
            eff_items[i] = eff_items[i][:-1] + " + visual)"
            eff_src[i] += "+VISUAL"
            continue
        add(eff_items, eff_labels, eff_src, cls, "VISUAL", f"{cls} (visual)")
        if cls in eff_sc:
            eff_ev.append(f"{cls}: presence {eff_sc[cls]['presence']:.4f} (strict threshold passed)")
    for d in v31_def:
        base = d.split(" (")[0]
        if base == "Uneven Coverage" and test_tile:
            def_ev.append("Uneven Coverage ignored: half-dipped / partly glazed test tiles are intended")
            continue
        if base in CRATER_TEXTURE and crater_intended:
            def_ev.append(f"{base} ignored: crater / lava glaze per CSV, pits and blisters are the intended texture")
            continue
        if base == "Crazing" and crawl_intended:
            def_ev.append("Crazing ignored: the crack network belongs to the intended crawl pattern (CSV)")
            continue
        qual = " (visual, intent ambiguous)" if "ambiguous" in d else " (visual)"
        if base in def_labels:
            continue
        add(def_items, def_labels, def_src, base, "VISUAL", base + qual)
        if base in def_sc:
            def_ev.append(f"{base}: presence {def_sc[base]['presence']:.4f} (strict threshold passed)")

    # 3. recipe description (may describe the recipe in general or other variants, not this photo).
    #    Conditional statements ("can pinhole if applied too thick") are risks, not observations.
    found = text_classes(text, EFFECT_TEXT)
    desc_eff = []
    for cls, snip in list(found.items()) + [(c, s) for c, s in found.get("_conditional", {}).items()]:
        if cls.startswith("_"):
            continue
        cond = cls in found.get("_conditional", {})
        if cls in eff_labels:
            continue
        desc_eff.append(cls)
        add(eff_items, eff_labels, eff_src, cls, "CSV_DESCRIPTION" + ("_CONDITIONAL" if cond else ""),
            f"{cls} (CSV description{', conditional' if cond else ''})")
        eff_ev.append(f"{cls}: '{snip}'")
    eff_ev += [f"negated mention ignored: {c}: '{s}'" for c, s in found.get("_negated", {}).items()]
    found = text_classes(text, DEFECT_TEXT)
    for cls, snip in list(found.items()) + [(c, s) for c, s in found.get("_conditional", {}).items()]:
        if cls.startswith("_"):
            continue
        cond = cls in found.get("_conditional", {})
        if cls == "Crawling" and crawl_intended:
            add(eff_items, eff_labels, eff_src, "Crawling effect", "CSV_DESCRIPTION", "Crawling effect (intended, CSV description)")
            eff_ev.append(f"Crawling effect: '{snip}'")
            continue
        if cls == "Crazing" and crackle_intended:
            continue
        if cls in CRATER_TEXTURE and crater_intended:
            continue
        if cond:
            add(def_items, def_labels, def_src, cls, "CSV_DESCRIPTION_CONDITIONAL", f"Risk: {cls} (CSV description, conditional)")
        else:
            add(def_items, def_labels, def_src, cls, "CSV_DESCRIPTION", f"{cls} (CSV description)")
        def_ev.append(f"{cls}: '{snip}'")
    def_ev += [f"negated mention ignored: {c}: '{s}'" for c, s in found.get("_negated", {}).items()]

    # 4. possible (top 3 % of the class over all objects): only for an attribute that has no other
    #    evidence, at most 2 labels; it fills an otherwise empty field, it does not add noise to a known one
    def possibles(sc, th, labels, kind):
        ranked = sorted(((c, v["presence"]) for c, v in sc.items() if c not in labels and v["presence"] >= th.get(c, 2)),
                        key=lambda kv: -kv[1])
        out = []
        for c, p in ranked:
            if kind == "effect":
                if c == "Crackle" and "Crackle" in labels:
                    continue
            else:
                if c == "Crawling" and (crawl_intended or "Crawling effect" in eff_labels):
                    continue
                if c in CRATER_TEXTURE and crater_intended:
                    continue
                if c == "Crazing" and (crackle_intended or crawl_intended or "Crackle" in eff_labels):
                    continue
                if c == "Uneven Coverage" and test_tile:
                    continue
            out.append((c, p))
            if len(out) == 2:
                break
        return out

    for c, p in (possibles(eff_sc, possible_th.get("effect", {}), eff_labels, "effect") if not eff_items else []):
        add(eff_items, eff_labels, eff_src, c, "VISUAL_POSSIBLE", f"Possible: {c} (visual, low confidence)")
        eff_ev.append(f"{c}: presence {p:.4f} (top 3 % of this class; below the strict threshold)")
    for c, p in (possibles(def_sc, possible_th.get("defect", {}), def_labels, "defect") if not def_items else []):
        amb = "; intent ambiguous" if c in ("Crazing", "Crawling") else ""
        add(def_items, def_labels, def_src, c, "VISUAL_POSSIBLE", f"Possible: {c} (visual, low confidence{amb})")
        def_ev.append(f"{c}: presence {p:.4f} (top 3 % of this class; below the strict threshold)")

    # 5. nothing found
    def empty(items, labels, srcs):
        if items:
            return
        if small or not scores:
            items.append("Unknown (object too small to judge)")
            labels.append("Unknown")
            srcs.append("UNKNOWN")
        else:
            items.append("None visible (checked)")
            labels.append("None")
            srcs.append("CHECKED_NONE")

    empty(eff_items, eff_labels, eff_src)
    empty(def_items, def_labels, def_src)
    # descriptions that list many variants ("With 3 rutile, ... With 1 cobalt, ...") would flood the field:
    # show the first 3 description-derived effects; all of them stay in *_labels and *_evidence
    if len(desc_eff) > 3:
        hidden = desc_eff[3:]
        eff_items = [it for it, lab in zip(eff_items, eff_labels) if lab not in hidden]
        eff_items.append(f"+{len(hidden)} more in CSV description")
    return {"glaze_effect": _fmt(eff_items), "glaze_effect_labels": " | ".join(eff_labels),
            "glaze_effect_source": " | ".join(eff_src), "glaze_effect_evidence": "; ".join(eff_ev),
            "defect": _fmt(def_items), "defect_labels": " | ".join(def_labels),
            "defect_source": " | ".join(def_src), "defect_evidence": "; ".join(def_ev)}


# --------------------------------------------------------------------------- frame truncation
def edge_contact(mask_crop, bbox_px, image_wh, band_frac=0.01):
    """Fraction of the object's side that lies on each photo border it touches.
    mask_crop: bool array of the object's bbox; bbox_px: [x0, y0, x1, y1] (full resolution)."""
    W, H = image_wh
    x0, y0, x1, y1 = bbox_px
    band = max(2, round(band_frac * min(H, W)))
    h, w = mask_crop.shape
    out = {}
    if y0 <= band:
        out["top"] = float(mask_crop[: max(1, band - y0 + 1)].any(0).mean())
    if y1 >= H - band:
        out["bottom"] = float(mask_crop[-max(1, band - (H - y1) + 1):].any(0).mean())
    if x0 <= band:
        out["left"] = float(mask_crop[:, : max(1, band - x0 + 1)].any(1).mean())
    if x1 >= W - band:
        out["right"] = float(mask_crop[:, -max(1, band - (W - x1) + 1):].any(1).mean())
    return out


def resolve_form_shape(rec: dict, contact: dict, min_contact=0.15):
    """v3.1 marked an object as cut by the frame only when it touched >= 2 borders; an object cut along one
    border (a second mug at the photo edge) was then classified as 'Fragment'. Here any border contact over
    `min_contact` of that side counts."""
    scores = (rec.get("visual_tag_scores") or {}).get("form") or {}
    form = rec.get("object_form_predicted") or "Unknown"
    shape = rec.get("shape_predicted") or "Unknown"
    trunc31 = bool(rec.get("object_truncated_by_frame"))
    cut_sides = sorted(k for k, v in contact.items() if v >= min_contact)
    truncated = trunc31 or bool(cut_sides)
    note = ""
    if truncated and not trunc31:
        keep_tile = form in ("Test Tile", "Tile") and float(scores.get("top_score", 0)) >= 0.5
        if form != "Unknown" and not keep_tile:
            note = f"form '{form}' withdrawn: object cut by the photo edge ({', '.join(cut_sides)})"
            form = "Unknown (object cut by photo edge)"
        flat = "Flat" in shape and keep_tile
        shape = "Flat" if flat else "Unknown (object cut by photo edge)"
    if truncated and form == "Unknown":
        form = "Unknown (object cut by photo edge)"
    if truncated and trunc31:
        if shape in ("Unknown", ""):
            shape = "Unknown (object cut by photo edge)"
    return {"object_form": form, "shape": shape, "object_truncated_by_frame_v32": truncated,
            "frame_contact": ", ".join(f"{k} {v:.2f}" for k, v in sorted(contact.items())), "form_note": note}
