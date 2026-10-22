"""Validated recipe records: raw values, normalised values and review flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .cones import ConeRange, ConeScale
from .config import PREPROCESSING_VERSION

# Fields that may feed the semantic description, in description order.
SEMANTIC_FIELDS = ("subtype", "surface", "transparency", "cone", "atmospheres")
# Kept for display, filtering and alias lookup; never embedded by default.
METADATA_FIELDS = ("name", "state", "country", "other_names")

ERROR = "error"  # value unusable: normalised value is None (unknown)
WARNING = "warning"  # value kept, but needs human review
INFO = "info"  # explicit configured transformation (e.g. synonym mapped)


@dataclass(frozen=True)
class QualityFlag:
    field: str
    code: str
    severity: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass(frozen=True)
class RecipeTags:
    """One source row after cleaning. ``None`` always means unknown/missing."""

    recipe_id: str | None
    record_type: str | None
    in_scope: bool
    raw: Mapping[str, str | None]
    subtype: tuple[str, ...] | None = None  # hierarchy levels, root first
    cone: ConeRange | None = None
    surface: str | None = None
    transparency: str | None = None
    atmospheres: tuple[str, ...] | None = None  # canonical order, deduplicated
    name: str | None = None
    state: str | None = None
    country: str | None = None
    other_names: tuple[str, ...] | None = None
    flags: tuple[QualityFlag, ...] = field(default=())

    @property
    def semantic_fields_present(self) -> tuple[str, ...]:
        return tuple(f for f in SEMANTIC_FIELDS if getattr(self, f) is not None)

    @property
    def subtype_label(self) -> str | None:
        return " - ".join(self.subtype) if self.subtype else None


@dataclass(frozen=True)
class CleanRecipe:
    """Cleaned tags plus the text description used for embedding."""

    tags: RecipeTags
    description: str | None
    unavailable_reason: str | None  # None means available for text retrieval

    @property
    def recipe_id(self) -> str | None:
        return self.tags.recipe_id

    @property
    def text_available(self) -> bool:
        return self.unavailable_reason is None and self.description is not None

    def to_json_dict(self) -> dict[str, Any]:
        t = self.tags
        return {
            "recipe_id": t.recipe_id,
            "record_type": t.record_type,
            "in_scope": t.in_scope,
            "raw": dict(t.raw),
            "normalised": {
                "subtype": list(t.subtype) if t.subtype else None,
                "cone": t.cone.to_dict() if t.cone else None,
                "surface": t.surface,
                "transparency": t.transparency,
                "atmospheres": list(t.atmospheres) if t.atmospheres else None,
            },
            "metadata": {
                "name": t.name,
                "state": t.state,
                "country": t.country,
                "other_names": list(t.other_names) if t.other_names else None,
            },
            "flags": [f.to_dict() for f in t.flags],
            "description": self.description,
            "text_available": self.text_available,
            "unavailable_reason": self.unavailable_reason,
            "preprocessing_version": PREPROCESSING_VERSION,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any], cones: ConeScale) -> "CleanRecipe":
        version = data.get("preprocessing_version")
        if version != PREPROCESSING_VERSION:
            raise ValueError(
                f"Cleaned record uses preprocessing {version!r}, code expects "
                f"{PREPROCESSING_VERSION!r}; re-run `python -m text_branch preprocess`."
            )
        norm, meta = data["normalised"], data["metadata"]
        tags = RecipeTags(
            recipe_id=data["recipe_id"],
            record_type=data["record_type"],
            in_scope=data["in_scope"],
            raw=dict(data["raw"]),
            subtype=tuple(norm["subtype"]) if norm["subtype"] else None,
            cone=cones.from_dict(norm["cone"]) if norm["cone"] else None,
            surface=norm["surface"],
            transparency=norm["transparency"],
            atmospheres=tuple(norm["atmospheres"]) if norm["atmospheres"] else None,
            name=meta["name"],
            state=meta["state"],
            country=meta["country"],
            other_names=tuple(meta["other_names"]) if meta["other_names"] else None,
            flags=tuple(QualityFlag(**f) for f in data["flags"]),
        )
        return cls(tags=tags, description=data["description"], unavailable_reason=data["unavailable_reason"])
