"""Attribute labels per object, each with its PROVENANCE (attributes_v1).

Sources, in order of trust:
  human        review/tile_review.json (review_tile panel; status confirmed / corrected)        - verified
  claude_visual work/attributes_v1/form_gold_claude.csv (object form of a sample, labelled by Claude from the
               object images; NOT human-verified - to be confirmed in the review app)
  glazy_meta   Glazy recipe metadata (Surface, Transparency) from the original export          - recipe level
  glazy_text   glaze effects / defects named in the recipe subtype or description (non-conditional mentions,
               as parsed by the existing tagger: "(CSV subtype)" / "(CSV description)")         - recipe level, incomplete
  old_tagger   the existing OpenCLIP zero-shot tagger (tag_json visual_tag_scores)              - baseline predictions
Recipe-level labels are copied to every object of the photo (a photo can show several test tiles of variations,
so they are WEAK object labels). A missing text mention is not a reliable negative (descriptions are incomplete).
"""
from __future__ import annotations

import json
import re

from .common import SEG, read_csv, read_json

SURFACE = ["glossy", "semi-glossy", "satin", "satin-matte", "semi-matte", "matte", "smooth matte", "stony matte", "dry matte"]
SURFACE_GROUP = {"glossy": "glossy", "semi-glossy": "glossy", "satin": "satin", "satin-matte": "satin", "semi-matte": "satin",
                 "matte": "matte", "smooth matte": "matte", "stony matte": "matte", "dry matte": "matte"}
TRANSP = ["transparent", "translucent", "semi-opaque", "opaque"]
TRANSP_GROUP = {"transparent": "clear", "translucent": "clear", "semi-opaque": "opaque", "opaque": "opaque"}
EFFECTS = ["speckled", "oil spot", "crystalline", "mottled", "flowing", "breaking at edges", "metallic / iridescent",
           "visible crack network (crackle)"]
DEFECTS = ["pinholes", "blistering", "crawling", "crazing", "excessive glaze running"]
FORMS = ["bowl", "plate", "cup", "vase", "jar", "test tile", "sculpture", "figurine", "abstract ceramic artwork",
         "irregular ceramic form", "other vessel"]
TEXTURES = ["smooth", "rough", "granular", "dimpled"]
EFFECT_MAP = {"dripping / flowing": "flowing", "mottled": "mottled", "crystalline": "crystalline", "crackle": "visible crack network (crackle)",
              "oil spot": "oil spot", "speckled": "speckled", "breaking": "breaking at edges", "metallic / iridescent": "metallic / iridescent"}
DEFECT_MAP = {"crazing": "crazing", "pinholes": "pinholes", "blistering": "blistering", "glaze running": "excessive glaze running",
              "crawling": "crawling", "crawling effect": "crawling"}
OLD_FORM_MAP = {"test tile": "test tile", "bowl": "bowl", "plate": "plate", "cup": "cup", "mug": "cup", "vase": "vase",
                "jar": "jar", "sculpture": "sculpture", "figurine": "figurine", "tile": "test tile", "dish": "plate",
                "bottle": "vase", "teapot": "other vessel", "pitcher": "other vessel", "platter": "plate"}
OLD_TEXTURE_MAP = {"smooth": "smooth", "rough": "rough", "granular": "granular", "dimpled": "dimpled", "pitted": "dimpled",
                   "pebbled": "granular"}


def parse_tag_cell(cell: str):
    """'Crystalline (CSV subtype); Possible: Crackle (visual, low confidence)' -> [(name, source, conditional)]"""
    out = []
    for part in [p.strip() for p in re.split(r";\s*", cell or "") if p.strip()]:
        m = re.match(r"(?:Possible:\s*|Risk:\s*)?(.+?)\s*\((.+)\)$", part)
        if not m:
            continue
        name, src = m.group(1).strip().lower(), m.group(2)
        source = "visual" if "visual" in src else "csv" if "CSV" in src else "other"
        out.append((name, source, "conditional" in src or part.startswith("Risk:")))
    return out


def build(objs: list[dict], recipes: list[dict]) -> list[dict]:
    rec = {r["recipe_id"]: r for r in recipes}
    human = (read_json(SEG / "review" / "tile_review.json", {}) or {}).get("objects", {})
    rows = []
    for o in objs:
        oid = o["object_id"]
        r = rec.get(o["recipe_id"], {})
        row = {"object_id": oid, "recipe_id": o["recipe_id"], "split": r.get("split", ""), "group_id": r.get("group_id", ""),
               "surface_meta": r.get("surface", "") if r.get("surface", "") in SURFACE else "",
               "transparency_meta": r.get("transparency", "") if r.get("transparency", "") in TRANSP else ""}
        row["surface_group_meta"] = SURFACE_GROUP.get(row["surface_meta"], "")
        row["transparency_group_meta"] = TRANSP_GROUP.get(row["transparency_meta"], "")
        eff_text, eff_old = set(), set()
        for name, src, cond in parse_tag_cell(o.get("glaze_effect", "")):
            lab = EFFECT_MAP.get(name)
            if name == "crawling effect" and src == "csv" and not cond:
                row.setdefault("_def_text", set()).add("crawling")
            if not lab:
                continue
            if src == "csv" and not cond:
                eff_text.add(lab)
            elif src == "visual":
                eff_old.add(lab)
        def_text, def_old = set(row.pop("_def_text", set())), set()
        for name, src, cond in parse_tag_cell(o.get("defect", "")):
            lab = DEFECT_MAP.get(name)
            if not lab:
                continue
            if src == "csv" and not cond:
                def_text.add(lab)
            elif src == "visual":
                def_old.add(lab)
        row["effects_text"] = json.dumps(sorted(eff_text))
        row["defects_text"] = json.dumps(sorted(def_text))
        row["has_description"] = int(bool((r.get("desc_free") or "").strip()) or bool(r.get("subtype")))
        # old tagger (baseline): visual predictions + scores from tag_json
        tj = read_json(SEG / "tag_json" / f"{oid}.json") or read_json(SEG / "does_not_needed" / "tag_json" / f"{oid}.json") or {}
        sc = tj.get("visual_tag_scores") or {}
        row["old_form"] = OLD_FORM_MAP.get(str(tj.get("object_form_predicted", "")).lower(), "")
        row["old_texture"] = OLD_TEXTURE_MAP.get(str(tj.get("texture_predicted", "")).lower(), "")
        row["old_effects"] = json.dumps(sorted(eff_old))
        row["old_defects"] = json.dumps(sorted(def_old))
        row["old_scores"] = json.dumps({k: (v.get("scores") if isinstance(v, dict) else None) for k, v in sc.items()})
        ev = tj.get("surface_evidence", "") or ""
        m = re.search(r"image classifier: ([^0-9]+?) [0-9.]+", ev)
        row["old_surface_group"] = {"glossy / semi-glossy": "glossy", "satin / semi-matte": "satin", "matte": "matte"}.get(
            (m.group(1).strip().lower() if m else ""), "")
        ev = tj.get("transparency_evidence", "") or ""
        m = re.search(r"image classifier: ([^0-9]+?) [0-9.]+", ev)
        row["old_transparency_group"] = {"semi-opaque / opaque": "opaque", "transparent / translucent": "clear"}.get(
            (m.group(1).strip().lower() if m else ""), "")
        # human (review_tile) - verified values override everything else downstream
        h = human.get(oid, {}).get("attributes", {})
        row["human"] = json.dumps({f: a for f, a in h.items() if a.get("status") in ("confirmed", "corrected")})
        rows.append(row)
    return rows
