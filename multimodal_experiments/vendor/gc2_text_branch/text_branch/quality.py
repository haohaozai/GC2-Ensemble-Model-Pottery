"""Data-quality report: missing fields, unrecognised values, unusable records."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from .records import ERROR, SEMANTIC_FIELDS, CleanRecipe

_REPORTED_FIELDS = (*SEMANTIC_FIELDS, "name", "country", "other_names")
_EXAMPLES = 5


def build_quality_report(recipes: Sequence[CleanRecipe], *, source: dict[str, Any]) -> dict[str, Any]:
    """Summarise cleaning results. Field statistics cover in-scope records only."""
    in_scope = [r for r in recipes if r.tags.in_scope]

    missing = {f: sum(r.tags.raw[f] is None for r in in_scope) for f in _REPORTED_FIELDS}
    invalid = {
        f: sum(r.tags.raw[f] is not None and getattr(r.tags, f) is None for r in in_scope)
        for f in SEMANTIC_FIELDS
    }

    unrecognised: dict[str, Counter] = defaultdict(Counter)
    flag_counts: Counter = Counter()
    flag_examples: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in in_scope:
        for flag in r.tags.flags:
            key = (flag.field, flag.code, flag.severity)
            flag_counts[key] += 1
            if len(flag_examples[key]) < _EXAMPLES:
                flag_examples[key].append(f"{r.recipe_id}: {flag.detail}")
            if flag.severity == ERROR and flag.field in SEMANTIC_FIELDS:
                unrecognised[f"{flag.field} ({flag.code})"][r.tags.raw[flag.field]] += 1

    reasons = Counter(r.unavailable_reason for r in in_scope if r.unavailable_reason)
    unusable_ids = defaultdict(list)
    for r in in_scope:
        if r.unavailable_reason:
            unusable_ids[r.unavailable_reason].append(r.recipe_id)

    available = [r for r in in_scope if r.text_available]
    descriptions = Counter(r.description for r in available)
    root_only = sum(
        r.tags.semantic_fields_present == ("subtype",) and len(r.tags.subtype) == 1 for r in available
    )

    return {
        "source": source,
        "records_total": len(recipes),
        "records_by_type": dict(Counter(r.tags.record_type or "<missing>" for r in recipes).most_common()),
        "records_in_scope": len(in_scope),
        "in_scope_missing_raw_values": missing,
        "in_scope_invalid_values": invalid,
        "unrecognised_or_malformed_values": {k: dict(v.most_common()) for k, v in sorted(unrecognised.items())},
        "flags": [
            {"field": f, "code": c, "severity": s, "count": n, "examples": flag_examples[(f, c, s)]}
            for (f, c, s), n in sorted(flag_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "semantic_fields_present_histogram": dict(sorted(Counter(
            len(r.tags.semantic_fields_present) for r in in_scope).items())),
        "unusable_in_scope_records": {reason: {"count": n, "recipe_ids": unusable_ids[reason]}
                                      for reason, n in reasons.most_common()},
        "text_available_records": len(available),
        "unique_descriptions": len(descriptions),
        "root_subtype_only_descriptions": root_only,
        "most_common_descriptions": [{"description": d, "count": n} for d, n in descriptions.most_common(10)],
    }


def render_quality_markdown(report: dict[str, Any]) -> str:
    lines = ["# Text branch data-quality report", ""]
    src = report["source"]
    lines += [
        f"- Source: `{src.get('path')}` (sha256 `{str(src.get('sha256'))[:12]}…`)",
        f"- Preprocessing version: `{src.get('preprocessing_version')}`",
        f"- Records: {report['records_total']} total; by type {report['records_by_type']}",
        f"- In scope (embedded if usable): {report['records_in_scope']}",
        f"- Available for text retrieval: {report['text_available_records']} "
        f"({report['unique_descriptions']} unique descriptions)",
        "",
        "## Missing and invalid values (in-scope records)",
        "",
        "| Field | Missing in source | Present but unusable |",
        "|---|---:|---:|",
    ]
    for field, n in report["in_scope_missing_raw_values"].items():
        invalid = report["in_scope_invalid_values"].get(field, "n/a (metadata)")
        lines.append(f"| {field} | {n} | {invalid} |")

    lines += ["", "## Unrecognised or malformed values", ""]
    if not report["unrecognised_or_malformed_values"]:
        lines.append("None.")
    for key, values in report["unrecognised_or_malformed_values"].items():
        shown = ", ".join(f"`{v}` ×{n}" for v, n in list(values.items())[:15])
        more = f" (+{len(values) - 15} more)" if len(values) > 15 else ""
        lines.append(f"- **{key}**: {shown}{more}")

    lines += ["", "## Review flags", "", "| Field | Code | Severity | Count | Example |", "|---|---|---|---:|---|"]
    for flag in report["flags"]:
        example = flag["examples"][0].replace("|", "\\|") if flag["examples"] else ""
        lines.append(f"| {flag['field']} | `{flag['code']}` | {flag['severity']} | {flag['count']} | {example} |")

    lines += ["", "## Unusable in-scope records", ""]
    if not report["unusable_in_scope_records"]:
        lines.append("None.")
    for reason, info in report["unusable_in_scope_records"].items():
        ids = ", ".join(map(str, info["recipe_ids"][:30]))
        more = " …" if info["count"] > 30 else ""
        lines.append(f"- `{reason}`: {info['count']} (IDs: {ids}{more})")

    lines += [
        "",
        "## Description coverage",
        "",
        f"- Semantic fields present per in-scope record: {report['semantic_fields_present_histogram']}",
        f"- Descriptions with only a root subtype (e.g. `Subtype: glaze.`): "
        f"{report['root_subtype_only_descriptions']} — low information, kept because the tag is real",
        "- Most common descriptions (identical descriptions get identical embeddings):",
        "",
    ]
    for item in report["most_common_descriptions"]:
        lines.append(f"  - {item['count']} × `{item['description']}`")
    return "\n".join(lines) + "\n"
