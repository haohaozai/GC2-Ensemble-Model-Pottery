"""Image filename -> Glazy recipe record + ingredient rows (7k matched CSVs).

Matching rule (verified on the 7k data, see docs/CLAUDE_PIPELINE.md):
  * the parent image stem must be all digits; it is the recipe_id (no substring or prefix guessing);
  * object files are named <parent stem>_<index>.<ext>; parse_object_name() recovers the parent stem
    from the LAST underscore, so object indices are never mistaken for recipe ids;
  * status: MATCHED | NO_RECIPE_MATCH | DUPLICATE_RECIPE_ID | INVALID_FILENAME, plus
    ingredient_status: OK | NO_INGREDIENT_ROWS | NAME_MISMATCH (ingredient 'Recipe Name' differs).
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

RECIPE_FIELDS = ["Name", "Type", "State", "Subtype", "Cone", "Surface", "Transparency", "Atmospheres",
                 "Country", "Description", "Ingredients", "Created By Name", "Date Created"]


def find_csv(folder: Path, pattern: str) -> Path:
    hits = sorted(Path(folder).glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No {pattern} in {folder}")
    return hits[0]


def parse_object_name(name: str):
    """'2977_1.jpg' -> ('2977', 1); '2977.jpg' -> ('2977', None)."""
    stem = Path(name).stem
    m = re.fullmatch(r"(\d+)_(\d+)", stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, None


class RecipeIndex:
    def __init__(self, csv_dir: Path):
        self.recipes_path = find_csv(csv_dir, "7k_matched_all_data*.csv")
        self.ingredients_path = find_csv(csv_dir, "7k_matched_ingredients*.csv")
        self.recipes = defaultdict(list)
        with self.recipes_path.open("r", newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                self.recipes[str(row.get("recipe_id", "")).strip()].append(row)
        self.ingredients = defaultdict(list)
        with self.ingredients_path.open("r", newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                self.ingredients[str(row.get("recipe_id", "")).strip()].append(row)

    def match(self, filename: str) -> dict:
        stem = Path(filename).stem
        rid = stem if re.fullmatch(r"\d+", stem) else ""
        out = {"filename": filename, "recipe_id": rid, "match_status": "", "recipe_name": "",
               "ingredient_count": 0, "ingredient_status": "", "notes": ""}
        if not rid:
            out["match_status"] = "INVALID_FILENAME"
            out["notes"] = "filename stem is not a plain numeric recipe id"
            return out
        rows = self.recipes.get(rid, [])
        if not rows:
            out["match_status"] = "NO_RECIPE_MATCH"
            return out
        if len(rows) > 1:
            out["match_status"] = "DUPLICATE_RECIPE_ID"
            out["notes"] = f"{len(rows)} recipe rows share this id; none chosen"
            return out
        rec = rows[0]
        out["match_status"] = "MATCHED"
        out["recipe_name"] = rec.get("Name", "")
        ing = self.ingredients.get(rid, [])
        out["ingredient_count"] = len(ing)
        if not ing:
            out["ingredient_status"] = "NO_INGREDIENT_ROWS"
        else:
            names = {r.get("Recipe Name", "").strip() for r in ing}
            if names and rec.get("Name", "").strip() not in names:
                out["ingredient_status"] = "NAME_MISMATCH"
                out["notes"] = (f"ingredient file recipe name {sorted(names)} differs from recipe name "
                                f"'{rec.get('Name', '')}' (spreadsheet conversion?)")
            else:
                out["ingredient_status"] = "OK"
        return out

    def record(self, rid: str) -> dict:
        rows = self.recipes.get(rid, [])
        return rows[0] if len(rows) == 1 else {}

    def ingredient_rows(self, rid: str) -> list:
        return [{"ingredient_id": r.get("Ingredient ID", ""), "ingredient_name": r.get("Ingredient Name", ""),
                 "percentage": r.get("Percentage", ""), "additional": r.get("Additional", "")}
                for r in self.ingredients.get(rid, [])]

    def recipe_context(self, rid: str) -> dict:
        rec = self.record(rid)
        return {"recipe_name": rec.get("Name", ""),
                "surface_csv": rec.get("Surface", ""), "transparency_csv": rec.get("Transparency", ""),
                "subtype_csv": rec.get("Subtype", ""), "cone_csv_raw": rec.get("Cone", ""),
                "atmospheres_csv": rec.get("Atmospheres", ""), "description_csv": rec.get("Description", ""),
                "type_csv": rec.get("Type", ""), "state_csv": rec.get("State", ""),
                "ingredients_text_csv": rec.get("Ingredients", "")}

    def summary(self) -> dict:
        return {"recipes_csv": str(self.recipes_path), "ingredients_csv": str(self.ingredients_path),
                "recipe_rows": sum(len(v) for v in self.recipes.values()),
                "recipe_ids": len(self.recipes),
                "duplicate_recipe_ids": sum(1 for v in self.recipes.values() if len(v) > 1),
                "ingredient_rows": sum(len(v) for v in self.ingredients.values()),
                "recipe_ids_with_ingredients": len(self.ingredients)}


def ingredients_json(rows) -> str:
    return json.dumps(rows, ensure_ascii=False)
