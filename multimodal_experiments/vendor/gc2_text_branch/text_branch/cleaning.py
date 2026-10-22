"""Clean Glazy recipe rows into validated ``RecipeTags`` records.

Policy: standardise whitespace and case, apply only the configured synonym
map, keep raw values, and flag anything malformed or contradictory instead of
fixing it. Invalid values become ``None`` (unknown) with an ``error`` flag;
contradictions keep their values with a ``warning`` flag. Nothing is inferred
from recipe names or descriptions.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .cones import ConeError, ConeScale
from .config import CleaningConfig
from .description import build_recipe_description
from .records import ERROR, INFO, WARNING, CleanRecipe, QualityFlag, RecipeTags

# Logical fields whose raw values are preserved on every record.
RAW_FIELDS = ("record_type", "subtype", "cone", "surface", "transparency",
              "atmospheres", "name", "state", "country", "other_names")


class UnrecognisedValueError(ValueError):
    pass


def clean_text(value: str | None) -> str | None:
    """NFC-normalise, collapse whitespace and strip; blank becomes ``None``."""
    if value is None:
        return None
    text = " ".join(unicodedata.normalize("NFC", value).split())
    return text or None


class TagNormaliser:
    """Applies a ``CleaningConfig`` to raw rows and to user filter values."""

    def __init__(self, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()
        cfg = self.config
        self.cones = ConeScale(cfg.cone_order, cfg.synonyms.get("cone"))
        self._vocab = {name: [v.lower() for v in values] for name, values in cfg.vocabularies.items()}
        self._synonyms = {
            name: {" ".join(k.split()).lower(): v.lower() for k, v in mapping.items()}
            for name, mapping in cfg.synonyms.items()
        }
        for name, mapping in self._synonyms.items():
            vocab = self._vocab.get(name)
            bad = [v for v in mapping.values() if vocab is not None and v not in vocab]
            if bad:
                raise ValueError(f"Synonyms for {name!r} map to unknown labels: {bad}")
        self._subtype_split = re.compile(cfg.subtype_separator)
        self._other_names_split = re.compile(cfg.other_names_separator)
        self._in_scope = {t.lower() for t in cfg.in_scope_types}

    # -- controlled vocabularies -------------------------------------------

    def vocabulary(self, name: str) -> list[str]:
        return list(self._vocab[name])

    def canonical(self, name: str, value: str) -> tuple[str, bool]:
        """Map ``value`` to its canonical label; returns (label, used_synonym).

        Raises ``UnrecognisedValueError`` if it is not in the vocabulary.
        """
        key = (clean_text(value) or "").lower()
        mapped = self._synonyms.get(name, {}).get(key, key)
        if mapped not in self._vocab[name]:
            raise UnrecognisedValueError(
                f"{value!r} is not a recognised {name} value; expected one of {self._vocab[name]}"
            )
        return mapped, mapped != key

    def _vocab_field(self, name: str, raw: str | None, flags: list[QualityFlag],
                     flag_field: str | None = None) -> str | None:
        text = clean_text(raw)
        if text is None:
            return None
        flag_field = flag_field or name
        try:
            label, mapped = self.canonical(name, text)
        except UnrecognisedValueError:
            flags.append(QualityFlag(flag_field, f"{name}_unrecognised", ERROR,
                                     f"{text!r} is not a recognised {name} value"))
            return None
        if mapped:
            flags.append(QualityFlag(flag_field, f"{name}_synonym_mapped", INFO,
                                     f"{text!r} mapped to {label!r}"))
        return label

    # -- individual fields -------------------------------------------------

    def _subtype(self, raw: str | None, flags: list[QualityFlag]) -> tuple[str, ...] | None:
        text = clean_text(raw)
        if text is None:
            return None
        levels = [level.strip() for level in self._subtype_split.split(text)]
        if any(not level or level.startswith("-") or level.endswith("-") for level in levels):
            flags.append(QualityFlag("subtype", "subtype_malformed", ERROR,
                                     f"{text!r} has an empty subtype level"))
            return None
        synonyms = self._synonyms.get("subtype", {})
        normalised = tuple(synonyms.get(level.lower(), level.lower()) for level in levels)
        if normalised[0] not in self._vocab["subtype_root"]:
            flags.append(QualityFlag("subtype", "subtype_root_unrecognised", WARNING,
                                     f"top-level subtype {normalised[0]!r} is not in the configured roots"))
        return normalised

    def _cone(self, raw: str | None, flags: list[QualityFlag]):
        text = clean_text(raw)
        if text is None:
            return None
        try:
            return self.cones.parse(text)
        except ConeError as exc:
            flags.append(QualityFlag("cone", exc.code, ERROR, exc.detail))
            return None

    def _atmospheres(self, raw: str | None, flags: list[QualityFlag]) -> tuple[str, ...] | None:
        text = clean_text(raw)
        if text is None:
            return None
        items = [item.strip() for item in text.split(self.config.atmosphere_separator)]
        if any(not item for item in items):
            flags.append(QualityFlag("atmospheres", "atmospheres_empty_item", WARNING,
                                     f"{text!r} contains an empty list item (ignored)"))
        items = [item for item in items if item]
        labels = [self._vocab_field("atmosphere", item, flags, flag_field="atmospheres") for item in items]
        if any(label is None for label in labels):
            # One unrecognised item makes the whole list unknown: a partial
            # list would misrepresent the recipe in strict filters.
            return None
        if len(set(labels)) != len(labels):
            flags.append(QualityFlag("atmospheres", "atmospheres_duplicate", WARNING,
                                     f"{text!r} lists an atmosphere more than once (deduplicated)"))
        order = self._vocab["atmosphere"]
        return tuple(sorted(set(labels), key=order.index)) or None

    def _other_names(self, raw: str | None) -> tuple[str, ...] | None:
        text = clean_text(raw)
        if text is None:
            return None
        parts = [p.strip() for p in self._other_names_split.split(text)]
        return tuple(dict.fromkeys(p for p in parts if p)) or None

    def _contradictions(self, tags: dict[str, Any], flags: list[QualityFlag]) -> None:
        subtype = tags["subtype"]
        if not subtype:
            return
        for rule in self.config.contradiction_rules:
            pattern = re.compile(rf"\b{re.escape(rule['subtype_keyword'].lower())}\b")
            value = tags.get(rule["field"])
            if value in rule["values"] and any(pattern.search(level) for level in subtype):
                flags.append(QualityFlag(
                    rule["field"], f"contradiction:{rule['id']}", WARNING,
                    f"subtype {' - '.join(subtype)!r} conflicts with {rule['field']} {value!r}",
                ))

    # -- whole record ------------------------------------------------------

    def normalise(self, row: Mapping[str, str | None]) -> RecipeTags:
        cols = self.config.columns
        raw = {name: clean_text(row.get(cols[name])) for name in RAW_FIELDS}
        flags: list[QualityFlag] = []
        recipe_id = clean_text(row.get(cols["recipe_id"]))
        if recipe_id is None:
            flags.append(QualityFlag("recipe_id", "recipe_id_missing", ERROR, "row has no recipe ID"))
        record_type = raw["record_type"]
        values: dict[str, Any] = {
            "subtype": self._subtype(raw["subtype"], flags),
            "cone": self._cone(raw["cone"], flags),
            "surface": self._vocab_field("surface", raw["surface"], flags),
            "transparency": self._vocab_field("transparency", raw["transparency"], flags),
            "atmospheres": self._atmospheres(raw["atmospheres"], flags),
        }
        self._contradictions(values, flags)
        return RecipeTags(
            recipe_id=recipe_id,
            record_type=record_type,
            in_scope=record_type is not None and record_type.lower() in self._in_scope,
            raw=raw,
            name=raw["name"],
            state=raw["state"],
            country=raw["country"],
            other_names=self._other_names(raw["other_names"]),
            flags=tuple(flags),
            **values,
        )

    def clean(self, row: Mapping[str, str | None]) -> CleanRecipe:
        tags = self.normalise(row)
        description = build_recipe_description(
            tags,
            min_semantic_fields=self.config.min_semantic_fields,
            metadata_fields=self.config.description_metadata_fields,
        )
        if tags.recipe_id is None:
            reason = "missing_recipe_id"
        elif not tags.in_scope:
            reason = "out_of_scope_type"
        elif description is None:
            reason = "no_usable_semantic_tags"
        else:
            reason = None
        return CleanRecipe(tags=tags, description=description, unavailable_reason=reason)


@functools.lru_cache(maxsize=1)
def _default_normaliser() -> TagNormaliser:
    return TagNormaliser(CleaningConfig())


def normalise_recipe_tags(record: Mapping[str, str | None],
                          normaliser: TagNormaliser | None = None) -> RecipeTags:
    """Clean one raw row (keyed by source column names) into ``RecipeTags``."""
    return (normaliser or _default_normaliser()).normalise(record)


def read_source_rows(path: str | Path, config: CleaningConfig) -> Iterator[dict[str, str]]:
    """Yield raw rows from the Glazy CSV export, checking required columns."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [col for col in config.columns.values() if col not in header]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; found {header}")
        yield from reader


def clean_rows(rows: Iterable[Mapping[str, str | None]],
               normaliser: TagNormaliser) -> list[CleanRecipe]:
    """Clean rows in source order; repeated IDs after the first are unusable."""
    recipes: list[CleanRecipe] = []
    seen: set[str] = set()
    for row in rows:
        recipe = normaliser.clean(row)
        rid = recipe.recipe_id
        if rid is not None and rid in seen:
            flag = QualityFlag("recipe_id", "recipe_id_duplicate", ERROR, f"recipe ID {rid!r} appears earlier")
            recipe = CleanRecipe(
                tags=dataclasses.replace(recipe.tags, flags=recipe.tags.flags + (flag,)),
                description=recipe.description,
                unavailable_reason="duplicate_recipe_id",
            )
        if rid is not None:
            seen.add(rid)
        recipes.append(recipe)
    return recipes


def write_clean_recipes(path: str | Path, recipes: Iterable[CleanRecipe]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for recipe in recipes:
            handle.write(json.dumps(recipe.to_json_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_clean_recipes(path: str | Path, cones: ConeScale) -> list[CleanRecipe]:
    with open(path, encoding="utf-8") as handle:
        return [CleanRecipe.from_json_dict(json.loads(line), cones) for line in handle if line.strip()]
