"""Auto engine of card G: which recipe engine answers a query best, by what the query gives.

The choice was made on recipes no model trained on (scripts/eval_glaze_engines.py) and is written, with its version date and
the reason for each kind of query, in configs/glaze_decision.json; the dated decision document is
docs/manuals/CG2_Glaze_Search_Model_Decision_<version>.pdf. A new evaluation = a new version in that file, no code change.
  photo_words_formula  a photo object, typed words or a formula is given
  menus                only a colour and the cone / surface menus
  colour               only a colour
Engines: smart (the joint model), colour_match (the text-chemistry model as it is), colour_ranking (colour match with its
learned part switched off: CIEDE2000 colour + surface + cone, the weights in the config).
"""
from __future__ import annotations

import json

from .common import CODE

CONFIG = CODE / "configs" / "glaze_decision.json"
KINDS = ("photo_words_formula", "menus", "colour")
ENGINES = ("smart", "colour_match", "colour_ranking")


def read_config() -> dict:
    cfg = {k: v for k, v in json.loads(CONFIG.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    bad = [k for k in KINDS if cfg["engines"].get(k) not in ENGINES]
    if bad:
        raise ValueError(f"{CONFIG.name}: engines for {bad} must be one of {ENGINES}")
    return cfg


def input_kind(p: dict) -> str:
    """what the page sent (the smart-search body): typed text, a photo object, a formula, the menus, the colour"""
    if str(p.get("text") or "").strip() or (p.get("use_photo") and p.get("qid")) or p.get("oxides"):
        return "photo_words_formula"
    if str(p.get("cone") or "").strip() or str(p.get("surface") or "").strip():
        return "menus"
    return "colour"


def choose(p: dict, cfg: dict, smart_ready: bool) -> dict:
    kind = input_kind(p)
    engine = cfg["engines"][kind]
    note = None
    if engine == "smart" and not smart_ready:
        engine, note = "colour_match", "smart search is not installed: colour match answered instead"
    return {"version": cfg["version"], "document": cfg.get("document"), "input": kind, "engine": engine,
            "why": cfg["why"][kind], "note": note}
