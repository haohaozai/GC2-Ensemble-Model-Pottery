from conftest import make_row
from text_branch.cleaning import TagNormaliser
from text_branch.config import CleaningConfig
from text_branch.description import build_recipe_description


def test_matches_documented_format(normaliser):
    tags = normaliser.normalise(make_row(subtype="Glaze - Iron - Celadon", surface="Glossy",
                                         transparency="Transparent", cone="9 - 10", atmospheres="Reduction"))
    assert build_recipe_description(tags) == (
        "Subtype: iron celadon glaze. Surface: glossy. Transparency: transparent. "
        "Firing cone: 9 to 10. Atmosphere: reduction."
    )


def test_missing_fields_are_omitted_in_fixed_order(normaliser):
    tags = normaliser.normalise(make_row(atmospheres="Oxidation; Neutral", cone="06", transparency="Opaque"))
    assert build_recipe_description(tags) == (
        "Transparency: opaque. Firing cone: 06. Atmosphere: oxidation, neutral."
    )


def test_invalid_values_are_omitted(normaliser):
    tags = normaliser.normalise(make_row(subtype="Glaze", cone="10 - 6", surface="Shiny"))
    assert build_recipe_description(tags) == "Subtype: glaze."


def test_deterministic_across_source_ordering(normaliser):
    a = normaliser.normalise(make_row("1", subtype="Glaze", atmospheres="Wood; Reduction; Salt & Soda"))
    b = normaliser.normalise(make_row("2", subtype="Glaze", atmospheres="Salt & Soda; Reduction; Wood"))
    descriptions = {build_recipe_description(t) for t in (a, b, a, b)}
    assert descriptions == {"Subtype: glaze. Atmosphere: reduction, salt & soda, wood."}


def test_cone_06_and_6_render_differently(normaliser):
    low = build_recipe_description(normaliser.normalise(make_row(cone="06")))
    mid = build_recipe_description(normaliser.normalise(make_row(cone="6")))
    assert low == "Firing cone: 06." and mid == "Firing cone: 6."


def test_no_semantic_tags_gives_none(normaliser):
    assert build_recipe_description(normaliser.normalise(make_row(name="Shiny Celadon", country="Japan"))) is None


def test_country_only_when_configured_and_not_counted(normaliser):
    tags = normaliser.normalise(make_row(subtype="Glaze", country="Japan"))
    assert "Japan" not in build_recipe_description(tags)
    assert build_recipe_description(tags, metadata_fields=["country"]) == "Subtype: glaze. Country: Japan."
    country_only = normaliser.normalise(make_row(country="Japan"))
    assert build_recipe_description(country_only, metadata_fields=["country"]) is None


def test_min_semantic_fields_threshold():
    config = CleaningConfig(min_semantic_fields=2)
    recipe = TagNormaliser(config).clean(make_row(subtype="Glaze"))
    assert recipe.unavailable_reason == "no_usable_semantic_tags"
