"""Claude's visual labels of the six attributes (attributes_v2) - the reference used to tune and validate Grounding DINO.

Source: work/attributes_v2/claude_labels_raw.txt, one line per object written while looking at the contact sheets
(mm/claude_sheets.py): oid|form|surface|transparency|texture|effects|defects with short codes. '?' = cannot be
judged from the photo (excluded from that attribute's scores). Provenance: claude_visual - NOT human-verified.
The judgement follows Ceramic_Real_Photo_Comparison_Guide_V4 (reflection sharpness / sheen for surface, visibility
of the clay body for transparency, visible cues for effects and defects).
"""
from __future__ import annotations

from pathlib import Path

FORM = {"tile": "test tile", "bowl": "bowl", "plate": "plate", "cup": "cup", "vase": "vase", "jar": "jar",
        "sculpt": "sculpture", "fig": "figurine", "abstract": "abstract ceramic artwork", "irreg": "irregular ceramic form",
        "vessel": "other vessel"}
SURFACE = {"g": "glossy", "sg": "semi-glossy", "sat": "satin", "satm": "satin-matte", "sm": "semi-matte", "m": "matte",
           "smm": "smooth matte", "stm": "stony matte", "dm": "dry matte"}
TRANSP = {"tr": "transparent", "tl": "translucent", "so": "semi-opaque", "op": "opaque"}
TEXTURE = {"s": "smooth", "r": "rough", "gr": "granular", "d": "dimpled"}
EFFECT = {"sp": "speckled", "os": "oil spot", "cr": "crystalline", "mo": "mottled", "fl": "flowing", "br": "breaking at edges",
          "me": "metallic / iridescent", "ck": "visible crack network (crackle)"}
DEFECT = {"ph": "pinholes", "bl": "blistering", "cw": "crawling", "cz": "crazing", "run": "excessive glaze running"}
SURFACE_GROUP = {"glossy": "glossy", "semi-glossy": "glossy", "satin": "satin", "satin-matte": "satin", "semi-matte": "matte",
                 "matte": "matte", "smooth matte": "matte", "stony matte": "matte", "dry matte": "matte"}
TRANSP_GROUP = {"transparent": "clear", "translucent": "clear", "semi-opaque": "opaque", "opaque": "opaque"}
SINGLE = {"form": FORM, "surface_finish": SURFACE, "transparency": TRANSP, "texture": TEXTURE}
MULTI = {"glaze_effect": EFFECT, "defect": DEFECT}
PROVENANCE = "claude_visual (Claude, from the object image only, guide V4; not human-verified)"


def parse(path: Path) -> dict[str, dict]:
    """oid -> {form, surface_finish, transparency, texture: str|None, glaze_effect, defect: list|None}"""
    out = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        oid, fo, su, tr, te, ef, de = (x.strip() for x in ln.split("|"))
        row = {}
        for key, code in (("form", fo), ("surface_finish", su), ("transparency", tr), ("texture", te)):
            row[key] = None if code == "?" else SINGLE[key][code]
        for key, code in (("glaze_effect", ef), ("defect", de)):
            row[key] = None if code == "?" else ([] if code == "none" else sorted(MULTI[key][c] for c in code.split(",")))
        row["surface_group"] = SURFACE_GROUP.get(row["surface_finish"]) if row["surface_finish"] else None
        row["transparency_group"] = TRANSP_GROUP.get(row["transparency"]) if row["transparency"] else None
        out[oid] = row
    return out
