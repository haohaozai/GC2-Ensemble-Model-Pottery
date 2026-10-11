import pytest

from conftest import SAMPLE_ROWS, make_row
from text_branch.cleaning import (
    TagNormaliser,
    clean_rows,
    normalise_recipe_tags,
    read_clean_recipes,
    read_source_rows,
    write_clean_recipes,
)
from text_branch.config import DEFAULT_CONFIG_PATH, CleaningConfig, TextBranchConfig, load_config
from text_branch.records import ERROR, INFO, WARNING


def codes(tags, severity=None):
    return {f.code for f in tags.flags if severity is None or f.severity == severity}


def test_missing_tags_stay_unknown_and_raw_is_preserved(normaliser):
    tags = normaliser.normalise(make_row("1", subtype="Glaze", cone="6"))
    assert tags.subtype == ("glaze",)
    assert tags.surface is None and tags.transparency is None and tags.atmospheres is None
    assert tags.raw["surface"] is None
    assert tags.semantic_fields_present == ("subtype", "cone")
    assert not tags.flags


def test_completely_empty_semantic_record_is_unavailable(normaliser):
    recipe = normaliser.clean(make_row("2", name="Celadon Blue Gloss Food Safe", country="Japan"))
    assert recipe.description is None
    assert not recipe.text_available
    assert recipe.unavailable_reason == "no_usable_semantic_tags"
    # Nothing is inferred from the name.
    assert recipe.tags.subtype is None and recipe.tags.surface is None and recipe.tags.transparency is None


def test_multiple_atmospheres_use_canonical_order(normaliser):
    tags = normaliser.normalise(make_row(atmospheres="Wood; Reduction; Oxidation"))
    assert tags.atmospheres == ("oxidation", "reduction", "wood")
    assert tags.raw["atmospheres"] == "Wood; Reduction; Oxidation"


def test_duplicate_atmosphere_is_flagged(normaliser):
    tags = normaliser.normalise(make_row(atmospheres="Oxidation; Oxidation"))
    assert tags.atmospheres == ("oxidation",)
    assert "atmospheres_duplicate" in codes(tags, WARNING)


def test_unrecognised_atmosphere_makes_list_unknown(normaliser):
    tags = normaliser.normalise(make_row(atmospheres="Oxidation; Pit fire"))
    assert tags.atmospheres is None
    assert "atmosphere_unrecognised" in codes(tags, ERROR)
    assert tags.raw["atmospheres"] == "Oxidation; Pit fire"


@pytest.mark.parametrize("raw, levels", [
    ("Glaze - Iron - Celadon - Blue", ("glaze", "iron", "celadon", "blue")),
    ("Glaze - Single-Fire", ("glaze", "single-fire")),
    ("Glaze - Iron - Kaki, Tomato Red", ("glaze", "iron", "kaki, tomato red")),
    ("Slip & Engobe - Terra Sigillata", ("slip & engobe", "terra sigillata")),
])
def test_hierarchical_subtypes(normaliser, raw, levels):
    tags = normaliser.normalise(make_row(subtype=raw))
    assert tags.subtype == levels
    assert tags.subtype_label == raw.lower()


@pytest.mark.parametrize("raw", ["Glaze -", "Glaze -  - Iron", "- Iron"])
def test_malformed_subtype_is_flagged(normaliser, raw):
    tags = normaliser.normalise(make_row(subtype=raw))
    assert tags.subtype is None
    assert "subtype_malformed" in codes(tags, ERROR)


def test_unknown_subtype_root_is_kept_but_flagged(normaliser):
    tags = normaliser.normalise(make_row(subtype="Kintsugi - Gold"))
    assert tags.subtype == ("kintsugi", "gold")
    assert "subtype_root_unrecognised" in codes(tags, WARNING)


def test_whitespace_and_case_are_standardised(normaliser):
    tags = normaliser.normalise(make_row(surface="  SEMI-GLOSSY ", transparency="semi-Opaque",
                                         subtype="  Glaze   -  Iron "))
    assert tags.surface == "semi-glossy"
    assert tags.transparency == "semi-opaque"
    assert tags.subtype == ("glaze", "iron")


def test_synonym_mapping_is_explicit_and_recorded(normaliser):
    tags = normaliser.normalise(make_row(surface="Semi gloss", atmospheres="Lustre; Oxidising"))
    assert tags.surface == "semi-glossy"
    assert tags.atmospheres == ("oxidation", "luster")
    assert {"surface_synonym_mapped", "atmosphere_synonym_mapped"} <= codes(tags, INFO)


def test_unrecognised_value_is_flagged_not_fixed(normaliser):
    tags = normaliser.normalise(make_row(surface="Shiny"))
    assert tags.surface is None
    assert tags.raw["surface"] == "Shiny"
    assert "surface_unrecognised" in codes(tags, ERROR)


def test_reversed_cone_is_flagged_not_swapped(normaliser):
    tags = normaliser.normalise(make_row(cone="10 - 6"))
    assert tags.cone is None
    assert tags.raw["cone"] == "10 - 6"
    assert "cone_range_reversed" in codes(tags, ERROR)


def test_contradiction_is_flagged_and_values_kept(normaliser):
    tags = normaliser.normalise(make_row(subtype="Glaze - Clear", transparency="Opaque"))
    assert tags.transparency == "opaque"
    assert "contradiction:clear_subtype_vs_opaque" in codes(tags, WARNING)
    tags = normaliser.normalise(make_row(subtype="Glaze - Matte - Magnesium Matte", surface="Glossy"))
    assert "contradiction:matte_subtype_vs_glossy_surface" in codes(tags, WARNING)


def test_country_and_other_names_are_metadata(normaliser):
    recipe = normaliser.clean(make_row("77", subtype="Glaze", country="Åland",
                                       other_names="Hesselberth Green, Floating Green; Frogpond Green"))
    assert recipe.tags.country == "Åland"
    assert recipe.tags.other_names == ("Hesselberth Green", "Floating Green", "Frogpond Green")
    assert recipe.description == "Subtype: glaze."
    for leaked in ("77", "Åland", "Green"):
        assert leaked not in recipe.description


def test_out_of_scope_types_are_not_available(normaliser):
    recipe = normaliser.clean({**make_row("900", subtype="Feldspar"), "Type": "Material"})
    assert not recipe.tags.in_scope
    assert recipe.unavailable_reason == "out_of_scope_type"


def test_duplicate_recipe_ids_are_flagged(normaliser):
    recipes = clean_rows([make_row("1", subtype="Glaze"), make_row("1", subtype="Glaze - Clear")], normaliser)
    assert recipes[0].text_available
    assert recipes[1].unavailable_reason == "duplicate_recipe_id"
    assert "recipe_id_duplicate" in codes(recipes[1].tags, ERROR)


def test_missing_recipe_id_is_unusable(normaliser):
    recipe = normaliser.clean(make_row("", subtype="Glaze"))
    assert recipe.unavailable_reason == "missing_recipe_id"


def test_module_level_function_uses_default_config():
    tags = normalise_recipe_tags(make_row(subtype="Glaze - Iron - Celadon", cone="9 - 10"))
    assert tags.subtype == ("glaze", "iron", "celadon")
    assert tags.cone.label == "9 - 10"


def test_jsonl_round_trip(tmp_path, normaliser, sample_recipes):
    path = tmp_path / "clean.jsonl"
    write_clean_recipes(path, sample_recipes)
    assert read_clean_recipes(path, normaliser.cones) == sample_recipes


def test_missing_source_column_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("ID,Name\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        list(read_source_rows(path, CleaningConfig()))


def test_synonyms_must_target_known_labels():
    config = CleaningConfig()
    config.synonyms["surface"]["shiny"] = "sparkly"
    with pytest.raises(ValueError, match="unknown labels"):
        TagNormaliser(config)


def test_repo_yaml_matches_python_defaults():
    assert load_config(DEFAULT_CONFIG_PATH) == TextBranchConfig()


def test_sample_rows_cover_expected_cases(sample_recipes):
    available = {r.recipe_id for r in sample_recipes if r.text_available}
    assert "13" not in available and "900" not in available
    assert len(sample_recipes) == len(SAMPLE_ROWS)
