"""Text retrieval over recipe descriptions, with explicit structured filters.

Baseline mode ranks by cosine similarity between pretrained text embeddings.
Shared mode projects both query and recipes with a *trained* checkpoint.
Filters are applied before the top-K cut. Recipes whose value for a filtered
field is unknown are excluded unless ``include_unknown=True``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .artifacts import BASELINE_KIND, load_embedding_set
from .cleaning import TagNormaliser, read_clean_recipes
from .cones import ConeError, ConeRange
from .config import TextBranchConfig
from .description import render_subtype
from .encoder import RecipeEmbeddings, TextEncoder, encode_query
from .projection import (
    IncompatibleCheckpointError,
    ProjectionCheckpoint,
    UntrainedProjectionError,
    load_projection_checkpoint,
    project_embeddings,
)
from .records import INFO, CleanRecipe

CLEAN_RECIPES_FILE = "recipes_clean.jsonl"


def _as_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass(frozen=True)
class SearchFilters:
    """Explicit structured filters; each field matches if ANY listed value matches.

    - ``cone``: a cone (``"6"``, ``"06"``) or range (``"5 - 7"``). With
      ``cone_match="within"`` (default) the requested cone/range must lie
      inside the recipe's supported range; ``"overlap"`` accepts any overlap.
    - ``atmospheres``: recipe must list at least one requested atmosphere.
    - ``surfaces`` / ``transparencies``: recipe value must be one of these.
    - ``include_unknown``: also keep recipes whose filtered field is unknown.
    Values go through the same synonym map as the source data; unrecognised
    values raise ``ValueError`` rather than silently matching nothing.
    """

    cone: str | None = None
    atmospheres: str | Sequence[str] | None = None
    surfaces: str | Sequence[str] | None = None
    transparencies: str | Sequence[str] | None = None
    include_unknown: bool = False
    cone_match: str = "within"

    @property
    def is_empty(self) -> bool:
        return not (self.cone or _as_tuple(self.atmospheres) or _as_tuple(self.surfaces)
                    or _as_tuple(self.transparencies))


@dataclass(frozen=True)
class SearchResult:
    rank: int
    recipe_id: str
    score: float
    description: str
    name: str | None
    subtype: str | None
    cone: str | None
    surface: str | None
    transparency: str | None
    atmospheres: tuple[str, ...] | None
    state: str | None
    country: str | None
    other_names: tuple[str, ...] | None
    review_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NameMatch:
    recipe_id: str
    match_type: str  # "exact" (case/whitespace-insensitive) or "normalised"
    matched_field: str  # "name" or "other_names"
    matched_text: str
    name: str | None
    description: str | None
    text_available: bool


def name_key(text: str) -> str:
    """Loose key: accents and apostrophes dropped, case-folded, other punctuation -> spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = re.sub(r"['’`]", "", stripped.casefold())
    return " ".join(re.sub(r"[^0-9a-z]+", " ", stripped).split())


def _exact_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


class NameIndex:
    """Exact / normalised lookup over recipe names and "Other Names" aliases."""

    def __init__(self, recipes: Iterable[CleanRecipe]):
        self._exact: dict[str, list[tuple[CleanRecipe, str, str]]] = {}
        self._loose: dict[str, list[tuple[CleanRecipe, str, str]]] = {}
        for recipe in recipes:
            tags = recipe.tags
            if not tags.in_scope or tags.recipe_id is None:
                continue
            candidates = [("name", tags.name), ("other_names", tags.raw.get("other_names"))]
            candidates += [("other_names", alias) for alias in tags.other_names or ()]
            for table, make_key in ((self._exact, _exact_key), (self._loose, name_key)):
                seen: set[str] = set()
                for field_name, text in candidates:
                    key = make_key(text) if text else ""
                    if key and key not in seen:
                        seen.add(key)
                        table.setdefault(key, []).append((recipe, field_name, text))

    def lookup(self, query: str) -> list[NameMatch]:
        matches: list[NameMatch] = []
        found: set[str] = set()
        for match_type, table, key in (("exact", self._exact, _exact_key(query)),
                                       ("normalised", self._loose, name_key(query))):
            for recipe, field_name, text in table.get(key, []):
                if recipe.recipe_id in found:
                    continue
                found.add(recipe.recipe_id)
                matches.append(NameMatch(recipe.recipe_id, match_type, field_name, text,
                                         recipe.tags.name, recipe.description, recipe.text_available))
        return matches


class RecipeSearchIndex:
    """Searchable view of embedded recipes, aligned row-by-row with the vectors."""

    def __init__(
        self,
        recipes: Sequence[CleanRecipe],
        embeddings: RecipeEmbeddings,
        encoder: TextEncoder | None,
        normaliser: TagNormaliser,
        *,
        projection: ProjectionCheckpoint | None = None,
    ):
        if embeddings.space != "encoder":
            raise ValueError("Build the index from baseline (pretrained) embeddings; projection is applied here")
        by_id = {r.recipe_id: r for r in recipes if r.recipe_id is not None}
        missing = [rid for rid in embeddings.recipe_ids if rid not in by_id]
        if missing:
            raise ValueError(f"Embeddings reference unknown recipe IDs, e.g. {missing[:5]}; re-run embed")
        self._rows = [by_id[rid] for rid in embeddings.recipe_ids]
        stale = [r.recipe_id for r, d in zip(self._rows, embeddings.descriptions) if r.description != d]
        if stale:
            raise ValueError(f"{len(stale)} embeddings were built from different descriptions; re-run embed")
        if encoder is not None:
            problems = embeddings.encoder.incompatibilities(encoder.info)
            if problems:
                raise IncompatibleCheckpointError("Query encoder differs from recipe embeddings: " + "; ".join(problems))
        self.encoder = encoder
        self.normaliser = normaliser
        self.embeddings = embeddings
        self.projection = projection
        if projection is None:
            self.space = "baseline"
            self._matrix = embeddings.vectors
        else:
            if not projection.trained:
                raise UntrainedProjectionError("Shared-space retrieval requires a trained projection checkpoint")
            problems = projection.encoder.incompatibilities(embeddings.encoder)
            if problems:
                raise IncompatibleCheckpointError("Checkpoint encoder differs from embeddings: " + "; ".join(problems))
            self.space = "shared"
            self._matrix = project_embeddings(embeddings.vectors, projection)
        tags = [r.tags for r in self._rows]
        self._cone_low = np.array([t.cone.low_index if t.cone else -1 for t in tags])
        self._cone_high = np.array([t.cone.high_index if t.cone else -1 for t in tags])
        self._surface = np.array([t.surface or "" for t in tags], dtype=object)
        self._transparency = np.array([t.transparency or "" for t in tags], dtype=object)
        self._atmospheres = [frozenset(t.atmospheres or ()) for t in tags]
        self._names = NameIndex(recipes)

    @classmethod
    def from_outputs(
        cls,
        config: TextBranchConfig,
        *,
        encoder: TextEncoder | None = None,
        checkpoint: str | Path | None = None,
    ) -> "RecipeSearchIndex":
        """Load cleaned recipes and baseline embeddings written by the CLI."""
        out = config.output_dir
        normaliser = TagNormaliser(config.cleaning)
        recipes = read_clean_recipes(out / CLEAN_RECIPES_FILE, normaliser.cones)
        embeddings, _ = load_embedding_set(out / "baseline", expected_kind=BASELINE_KIND)
        encoder = encoder or TextEncoder(config.encoder)
        projection = None
        if checkpoint is not None:
            projection = load_projection_checkpoint(checkpoint, encoder=encoder.info, require_trained=True)
        return cls(recipes, embeddings, encoder, normaliser, projection=projection)

    def __len__(self) -> int:
        return len(self._rows)

    # -- filtering ---------------------------------------------------------

    def _vocab_values(self, name: str, values: str | Sequence[str] | None) -> set[str]:
        return {self.normaliser.canonical(name, v)[0] for v in _as_tuple(values)}

    def filter_mask(self, filters: SearchFilters | None) -> np.ndarray:
        """Boolean mask over rows; strict (unknown excluded) unless include_unknown."""
        mask = np.ones(len(self._rows), dtype=bool)
        if filters is None or filters.is_empty:
            return mask
        keep_unknown = filters.include_unknown
        if filters.cone:
            if filters.cone_match not in ("within", "overlap"):
                raise ValueError("cone_match must be 'within' or 'overlap'")
            try:
                wanted: ConeRange = self.normaliser.cones.parse(filters.cone)
            except ConeError as exc:
                raise ValueError(f"Invalid cone filter: {exc.detail}") from exc
            known = self._cone_low >= 0
            if filters.cone_match == "within":
                ok = (self._cone_low <= wanted.low_index) & (wanted.high_index <= self._cone_high)
            else:
                ok = (self._cone_low <= wanted.high_index) & (wanted.low_index <= self._cone_high)
            mask &= (known & ok) | (~known & keep_unknown)
        for name, column, values in (("surface", self._surface, filters.surfaces),
                                     ("transparency", self._transparency, filters.transparencies)):
            wanted_values = self._vocab_values(name, values)
            if wanted_values:
                ok = np.isin(column, list(wanted_values))
                mask &= ok | ((column == "") & keep_unknown)
        wanted_atm = self._vocab_values("atmosphere", filters.atmospheres)
        if wanted_atm:
            ok = np.array([bool(a & wanted_atm) if a else keep_unknown for a in self._atmospheres], dtype=bool)
            mask &= ok
        return mask

    # -- ranking -----------------------------------------------------------

    def search(self, query: str, top_k: int = 10, filters: SearchFilters | None = None) -> list[SearchResult]:
        if self.encoder is None:
            raise RuntimeError("This index has no query encoder; use search_vector()")
        if not query or not query.strip():
            raise ValueError("Query must be non-empty")
        return self.search_vector(encode_query(query, self.encoder, self.projection), top_k, filters)

    def search_vector(self, query_vector: np.ndarray, top_k: int = 10,
                      filters: SearchFilters | None = None) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.shape != (self._matrix.shape[1],):
            raise ValueError(f"Query vector has shape {query_vector.shape}, index expects ({self._matrix.shape[1]},)")
        candidates = np.flatnonzero(self.filter_mask(filters))  # filter BEFORE ranking
        scores = self._matrix[candidates] @ query_vector
        order = np.lexsort((candidates, -scores))[:top_k]  # score desc, then source order
        return [self._result(rank, int(candidates[i]), float(scores[i])) for rank, i in enumerate(order, 1)]

    def count_matching(self, filters: SearchFilters | None) -> int:
        return int(self.filter_mask(filters).sum())

    def count_with_description(self, description: str, filters: SearchFilters | None = None) -> int:
        """Filtered recipes sharing ``description`` (they tie exactly in ranking)."""
        mask = self.filter_mask(filters)
        return sum(1 for ok, d in zip(mask, self.embeddings.descriptions) if ok and d == description)

    def _result(self, rank: int, row: int, score: float) -> SearchResult:
        recipe = self._rows[row]
        t = recipe.tags
        return SearchResult(
            rank=rank,
            recipe_id=t.recipe_id,
            score=round(score, 6),
            description=recipe.description,
            name=t.name,
            subtype=render_subtype(t.subtype) if t.subtype else None,
            cone=t.cone.label if t.cone else None,
            surface=t.surface,
            transparency=t.transparency,
            atmospheres=t.atmospheres,
            state=t.state,
            country=t.country,
            other_names=t.other_names,
            review_flags=tuple(f.code for f in t.flags if f.severity != INFO),
        )

    def lookup_name(self, name: str) -> list[NameMatch]:
        """Separate exact / normalised lookup by recipe name or alias."""
        return self._names.lookup(name)


def search_recipes(
    query: str,
    top_k: int = 10,
    filters: SearchFilters | None = None,
    *,
    index: RecipeSearchIndex,
) -> list[SearchResult]:
    """Rank recipes for a free-text query; see ``RecipeSearchIndex.search``."""
    return index.search(query, top_k, filters)
