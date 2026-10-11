"""Retrieval demonstration (smoke test, not an evaluation).

The "tag agreement" columns compare results with tags hand-mapped from each
query's wording. Descriptions are generated from the same tags, so agreement
shows the pipeline is wired correctly; it is not evidence of real-world
retrieval quality, which needs manually reviewed relevance labels.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .search import RecipeSearchIndex, SearchFilters, SearchResult

Check = Callable[[SearchResult], bool]


def _surface_is(*values: str) -> Check:
    return lambda r: r.surface in values


def _surface_matte_family(r: SearchResult) -> bool:
    return r.surface is not None and "matte" in r.surface


def _transparency_is(*values: str) -> Check:
    return lambda r: r.transparency in values


def _subtype_has(word: str) -> Check:
    return lambda r: r.subtype is not None and word in r.subtype


@dataclass
class DemoCase:
    query: str
    filters: SearchFilters | None = None
    checks: dict[str, Check] = field(default_factory=dict)
    note: str = ""


DEMO_CASES = [
    DemoCase(
        "Shiny transparent celadon glaze.",
        checks={"surface glossy": _surface_is("glossy"), "transparent": _transparency_is("transparent"),
                "subtype celadon": _subtype_has("celadon")},
        note="'Shiny' never appears in descriptions; the encoder must relate it to 'glossy'.",
    ),
    DemoCase(
        "Matte opaque glaze.",
        checks={"surface *matte*": _surface_matte_family, "opaque": _transparency_is("opaque")},
    ),
    DemoCase(
        "Shiny transparent celadon glaze.",
        SearchFilters(cone="10", atmospheres=["reduction"]),
        checks={"surface glossy": _surface_is("glossy"), "transparent": _transparency_is("transparent"),
                "subtype celadon": _subtype_has("celadon")},
        note="Filters: cone 10 must lie within the recipe's cone range; reduction must be listed.",
    ),
    DemoCase(
        "Shiny transparent celadon glaze.",
        SearchFilters(cone="10", atmospheres=["reduction"], surfaces="glossy", transparencies="transparent"),
        checks={"surface glossy": _surface_is("glossy"), "transparent": _transparency_is("transparent"),
                "subtype celadon": _subtype_has("celadon")},
        note="Same query with the hard constraints passed as explicit filters (the recommended usage).",
    ),
    DemoCase(
        "Matte opaque glaze.",
        SearchFilters(cone="6", atmospheres="oxidation"),
        checks={"surface *matte*": _surface_matte_family, "opaque": _transparency_is("opaque")},
    ),
    DemoCase(
        "Glossy transparent glaze.",
        SearchFilters(cone="06"),
        checks={"surface glossy": _surface_is("glossy"), "transparent": _transparency_is("transparent")},
        note="Cone 06 (low fire) is a different cone from 6; compare with the next case.",
    ),
    DemoCase(
        "Glossy transparent glaze.",
        SearchFilters(cone="6"),
        checks={"surface glossy": _surface_is("glossy"), "transparent": _transparency_is("transparent")},
    ),
]

DEMO_NAME_LOOKUPS = ["Floating Green", "pier's pure lux deluxe revised"]


def _md(text: object) -> str:
    return str(text if text is not None else "").replace("|", "\\|")


def render_case(index: RecipeSearchIndex, case: DemoCase, top_k: int) -> list[str]:
    results = index.search(case.query, top_k=top_k, filters=case.filters)
    n_candidates = index.count_matching(case.filters)
    f = case.filters
    filt = "none" if f is None or f.is_empty else ", ".join(
        f"{k}={v}" for k, v in (("cone", f.cone), ("atmospheres", f.atmospheres), ("surfaces", f.surfaces),
                                 ("transparencies", f.transparencies)) if v)
    lines = [f"### `{case.query}` — filters: {filt}", ""]
    if case.note:
        lines += [f"_{case.note}_", ""]
    lines.append(f"Candidates after filtering: {n_candidates} of {len(index)} embedded recipes.")
    lines.append("")
    if not results:
        return lines + ["**No results** — no recipe satisfies the filters.", ""]
    head = ["#", "ID", "score", "name", "description"] + list(case.checks) + ["flags"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        marks = ["✓" if check(r) else "✗" for check in case.checks.values()]
        lines.append("| " + " | ".join(_md(x) for x in [
            r.rank, r.recipe_id, f"{r.score:.3f}", r.name, r.description, *marks, ", ".join(r.review_flags)]) + " |")
    lines.append("")
    def matches_all(r: SearchResult) -> bool:
        return all(c(r) for c in case.checks.values())

    all_ok = sum(matches_all(r) for r in results)
    lines.append(f"Tag agreement (synthetic smoke check): {all_ok}/{len(results)} results match every "
                 f"hand-mapped tag.")
    if all_ok < len(results):
        ranked = index.search(case.query, top_k=n_candidates, filters=case.filters)
        first = next((r for r in ranked if matches_all(r)), None)
        lines.append("First result matching every hand-mapped tag: " + (
            f"rank {first.rank} of {n_candidates} (score {first.score:.3f} vs top {results[0].score:.3f})."
            if first else "none among the candidates."))
    desc_counts = Counter(r.description for r in results)
    top_desc, top_n = desc_counts.most_common(1)[0]
    tied_total = index.count_with_description(results[0].description, case.filters)
    if tied_total > 1:
        lines.append(f"Rank 1's description is shared by {tied_total} filtered recipes, which tie exactly; "
                     "order among ties is source order (recipe ID), not relevance.")
    if top_n > 1:
        lines.append(f"{top_n} of the top {len(results)} results have the identical description `{top_desc}`.")
    return lines + [""]


def run_demo(index: RecipeSearchIndex, top_k: int = 5) -> str:
    info = index.embeddings.encoder
    lines = [
        "# Text-branch retrieval demonstration",
        "",
        "**Smoke test, not an evaluation.** Queries and tag checks are synthetic; tag agreement",
        "only shows the pipeline is wired correctly. No manually reviewed relevance labels exist",
        "yet, so no retrieval-quality claim is made.",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
        f"- Space: **{index.space}** ({'pretrained text embeddings' if index.space == 'baseline' else 'trained projection'})",
        f"- Encoder: `{info.model_name}` @ `{info.revision}` (dim {info.dim})",
        f"- Embedded recipes: {len(index)}",
        "",
    ]
    for case in DEMO_CASES:
        lines += render_case(index, case, top_k)
    lines += ["## Name / alias lookup (separate from semantic search)", ""]
    for name in DEMO_NAME_LOOKUPS:
        matches = index.lookup_name(name)
        lines.append(f"- `{name}` → " + (", ".join(
            f"{m.recipe_id} ({m.match_type} on {m.matched_field}: {m.matched_text!r})" for m in matches[:5]
        ) or "no match"))
    return "\n".join(lines) + "\n"
