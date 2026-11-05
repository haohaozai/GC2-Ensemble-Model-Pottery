"""7k reference data for the glaze recipe search (card G; the retrieval model is in mm/glaze_retrieval.py).

Kept apart from the model: recipe names, the ORIGINAL cone text, ingredients and the cone / surface choices of the page.
Nothing here changes a score except the original cone text, which the corrected cone scoring reads (the catalog lost
the leading zero of the low-fire Orton cones: 04 -> 4, 05 1/2 - 03 -> 5.5 - 3).

Reads 7k_matched_all_data*.csv and 7k_matched_ingredients*.csv (read-only) from the first folder that has them:
env CG2_7K_CSV_DIR, <CG2_DATA>\\csv 7k data, <CG2_ROOT> (D:\\NUS study\\CG2 on the first PC).
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import BASE, CG2
from .external import CONE_ORDER

# typed surface words -> the 9 surface values of the 7k recipes (the dropdown sends those exactly). Only spelling variants:
# Glossy / Semi-glossy, Matte / Semi-matte / Satin-matte ... are different finishes and are never merged.
SURFACE_SYNONYMS = {"gloss": "Glossy", "shiny": "Glossy", "semi gloss": "Semi-glossy", "semigloss": "Semi-glossy",
                    "matt": "Matte", "semi matt": "Semi-matte", "semimatte": "Semi-matte", "satin matt": "Satin-matte",
                    "satinmatte": "Satin-matte", "smooth matt": "Smooth Matte", "stony matt": "Stony Matte", "dry matt": "Dry Matte"}
ANY = ("", "any", "unknown", "none")
CONE_TOKEN = re.compile(r"(?<![\d.])0?\d+(?:\.\d+)?(?: 1/2)?")


def csv_dirs() -> list[Path]:
    env = os.environ.get("CG2_7K_CSV_DIR")
    return ([Path(env)] if env else []) + [BASE / "csv 7k data", CG2]


def find_7k(pattern: str) -> Path | None:
    for d in csv_dirs():
        hits = sorted(d.glob(pattern)) if d.is_dir() else []
        if hits:
            return hits[0]
    return None


def _key(s) -> str:
    return re.sub(r"[\s_-]+", " ", str(s).strip().lower())


def cone_tokens(cone) -> list[str]:
    """'05 1/2 - 03' -> ['05 1/2', '03']; '5.5 - 6' -> ['5 1/2', '6'] (7k spelling, leading zeros kept)."""
    s = str(cone).replace("½", " 1/2").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return [re.sub(r"^(0?\d+)\.5$", r"\1 1/2", t) for t in CONE_TOKEN.findall(s)]


def cone_order_key(token: str) -> tuple:
    return (0, CONE_ORDER.index(token)) if token in CONE_ORDER else (1, token)


def normalize_cone(cone) -> str | None:
    """page / API cone -> '6', '04' or a range '5 - 6' (low to high, 7k spelling); None = any cone. Every number must be an
    Orton cone (022 ... 01, 1 ... 14, half cones 05 1/2 and 5 1/2)."""
    if cone is None or _key(cone) in ANY:
        return None
    toks = cone_tokens(cone)
    bad = [t for t in toks if t not in CONE_ORDER]
    if not toks or bad or len(toks) > 2:
        raise ValueError(f"cone {cone!r} is not an Orton cone or range (e.g. 6, 04, 5 1/2, 5 - 6)")
    toks = sorted(set(toks), key=cone_order_key)
    return toks[0] if len(toks) == 1 else f"{toks[0]} - {toks[-1]}"


def normalize_surface(surface, choices: list[str]) -> str | None:
    """'glossy' / 'gloss' / 'Glossy' -> 'Glossy', 'semi gloss' -> 'Semi-glossy'; None = any surface; unknown words are refused."""
    if surface is None or _key(surface) in ANY:
        return None
    k = _key(surface)
    for c in choices:
        if _key(c) == k:
            return c
    if k in SURFACE_SYNONYMS:
        return SURFACE_SYNONYMS[k]
    raise ValueError(f"surface {surface!r} is not one of: {', '.join(choices)}")


class Reference:
    """recipe names, original cone text, ingredients (by recipe_id) and the cone / surface choices with their counts."""

    def __init__(self):
        self.recipes_path = find_7k("7k_matched_all_data*.csv")
        self.ingredients_path = find_7k("7k_matched_ingredients*.csv")
        self.names: dict[int, str] = {}
        self.cones: dict[int, str] = {}
        self.ingredients: dict[int, list[dict]] = {}
        self.surface_counts: Counter = Counter()
        self.cone_counts: Counter = Counter()
        if self.recipes_path:
            df = pd.read_csv(self.recipes_path, usecols=["recipe_id", "Name", "Cone", "Surface"],
                             dtype={"Name": str, "Cone": str, "Surface": str}, low_memory=False)   # str: keeps the 0 of 04
            for rid, name, cone in zip(df["recipe_id"], df["Name"], df["Cone"]):
                if isinstance(name, str) and name.strip():
                    self.names[int(rid)] = name.strip()
                if isinstance(cone, str) and cone.strip():
                    self.cones[int(rid)] = cone.strip()
            self.surface_counts = Counter(df["Surface"].dropna().str.strip())
            self.cone_counts = Counter(t for c in df["Cone"].dropna() for t in cone_tokens(c))
        if self.ingredients_path:
            ing = pd.read_csv(self.ingredients_path, usecols=["recipe_id", "Ingredient Name", "Percentage", "Additional"])
            for rid, name, pct, add in zip(ing["recipe_id"], ing["Ingredient Name"], ing["Percentage"], ing["Additional"]):
                self.ingredients.setdefault(int(rid), []).append(
                    {"name": None if pd.isna(name) else str(name), "percent": None if pd.isna(pct) else round(float(pct), 3),
                     "addition": str(add).strip().lower() == "yes"})

    def choices(self, catalog: pd.DataFrame) -> dict:
        """dropdown values: from the 7k recipes when the CSV is there, else from the catalog (its cones lack the zeros)."""
        surf, cones, src = self.surface_counts, self.cone_counts, "7k_matched_all_data.csv"
        if not self.recipes_path:
            surf = Counter(catalog["Surface"].dropna().astype(str).str.strip())
            cones = Counter(t for c in catalog["Cone"].dropna() for t in cone_tokens(c))
            src = "glaze_retrieval_catalog.csv (7k CSV not found: cones without their leading zero)"
        return {"source": src,
                "surfaces": [{"value": s, "count": n} for s, n in surf.most_common()],
                "cones": [{"value": c, "count": cones[c]} for c in sorted(cones, key=cone_order_key)]}

    def info(self) -> dict:
        return {"recipes_csv": str(self.recipes_path) if self.recipes_path else None,
                "ingredients_csv": str(self.ingredients_path) if self.ingredients_path else None,
                "names": len(self.names), "original_cones": len(self.cones), "recipes_with_ingredients": len(self.ingredients)}
