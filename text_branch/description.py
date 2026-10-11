"""Deterministic text descriptions built only from validated semantic tags."""

from __future__ import annotations

from typing import Sequence

from .records import RecipeTags


def render_subtype(levels: Sequence[str]) -> str:
    """``("glaze", "iron", "celadon")`` -> ``"iron celadon glaze"``.

    Deeper levels qualify the root, so they are read first. The structured
    record keeps the full hierarchy; this is only the sentence wording.
    """
    return " ".join([*levels[1:], levels[0]])


def build_recipe_description(
    tags: RecipeTags,
    *,
    min_semantic_fields: int = 1,
    metadata_fields: Sequence[str] = (),
) -> str | None:
    """Return one sentence-per-field description, or ``None`` if unusable.

    Field order is fixed: subtype, surface, transparency, cone, atmosphere.
    Missing or invalid fields are omitted (they stay ``None`` in ``tags``).
    IDs, names, creators, timestamps and country are never included unless
    ``metadata_fields`` explicitly asks for country.
    """
    parts: list[str] = []
    if tags.subtype:
        parts.append(f"Subtype: {render_subtype(tags.subtype)}.")
    if tags.surface:
        parts.append(f"Surface: {tags.surface}.")
    if tags.transparency:
        parts.append(f"Transparency: {tags.transparency}.")
    if tags.cone:
        parts.append(f"Firing cone: {tags.cone.text}.")
    if tags.atmospheres:
        parts.append(f"Atmosphere: {', '.join(tags.atmospheres)}.")
    if len(parts) < min_semantic_fields:
        return None
    if "country" in metadata_fields and tags.country:
        parts.append(f"Country: {tags.country}.")
    return " ".join(parts)
