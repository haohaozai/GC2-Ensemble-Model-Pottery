"""Regression checks against the real Glazy export (skipped if it is absent)."""

import re
from collections import Counter

import pytest

from text_branch.cleaning import TagNormaliser, clean_rows, read_source_rows
from text_branch.config import CleaningConfig, PathsConfig

SOURCE = PathsConfig().resolve(PathsConfig().source_csv)

pytestmark = [
    pytest.mark.dataset,
    pytest.mark.skipif(not SOURCE.exists(), reason=f"{SOURCE} not present"),
]


@pytest.fixture(scope="module")
def recipes():
    config = CleaningConfig()
    return clean_rows(read_source_rows(SOURCE, config), TagNormaliser(config))


def test_every_row_is_kept_with_unique_ids(recipes):
    ids = [r.recipe_id for r in recipes]
    assert len(ids) == len(set(ids)) and None not in ids


def test_controlled_vocabularies_cover_the_export(recipes):
    flags = Counter(f.code for r in recipes if r.tags.in_scope for f in r.tags.flags)
    for code in ("surface_unrecognised", "transparency_unrecognised", "atmosphere_unrecognised",
                 "cone_unrecognised", "cone_malformed", "subtype_malformed", "subtype_root_unrecognised"):
        assert flags[code] == 0, code


def test_reversed_cone_ranges_are_flagged_not_fixed(recipes):
    reversed_ = [r for r in recipes if any(f.code == "cone_range_reversed" for f in r.tags.flags)]
    assert reversed_
    assert all(r.tags.cone is None and r.tags.raw["cone"] for r in reversed_)


def test_descriptions_only_for_in_scope_records_with_tags(recipes):
    allowed = {"Subtype", "Surface", "Transparency", "Firing cone", "Atmosphere"}
    for r in recipes:
        if r.text_available:
            assert r.tags.in_scope and r.tags.semantic_fields_present and r.description
            labels = re.findall(r"(?:^|\. )([A-Z][a-z]+(?: cone)?): ", r.description)
            assert labels and set(labels) <= allowed, r.description
        elif r.tags.in_scope and r.recipe_id:
            assert r.unavailable_reason in {"no_usable_semantic_tags", "duplicate_recipe_id"}
