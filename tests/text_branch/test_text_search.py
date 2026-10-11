import json

import numpy as np
import pytest

from text_branch.encoder import encode_query, encode_recipes
from text_branch.pipeline import UNTRAINED_DIR, run_embed, run_preprocess, run_project
from text_branch.projection import (
    UntrainedProjectionError,
    build_projection_head,
    load_projection_checkpoint,
    save_projection_checkpoint,
)
from text_branch.search import RecipeSearchIndex, SearchFilters, search_recipes

TRAINING = {"run_id": "test-run", "modalities": ["text", "image"]}


@pytest.fixture
def index(sample_recipes, fake_encoder, normaliser):
    emb = encode_recipes(sample_recipes, fake_encoder)
    return RecipeSearchIndex(sample_recipes, emb, fake_encoder, normaliser)


def ids(results):
    return [r.recipe_id for r in results]


def test_results_are_ranked_with_metadata(index, sample_recipes):
    recipe_4 = next(r for r in sample_recipes if r.recipe_id == "4")
    results = search_recipes(recipe_4.description, top_k=3, index=index)
    assert results[0].recipe_id == "4" and results[0].score == pytest.approx(1.0, abs=1e-5)
    assert [r.rank for r in results] == [1, 2, 3]
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))
    top = results[0]
    assert top.description.startswith("Subtype: iron celadon glaze.")
    assert top.cone == "9 - 11" and top.atmospheres == ("reduction",) and top.name


def test_filters_apply_before_top_k(index):
    query = "Subtype: iron celadon glaze. Surface: glossy. Transparency: transparent."
    filters = SearchFilters(atmospheres="oxidation")
    results = index.search(query, top_k=3, filters=filters)
    assert len(results) == 3  # still K results even though the best match was filtered out
    assert "4" not in ids(results)
    assert all("oxidation" in r.atmospheres for r in results)


def test_cone_filter_uses_recipe_range(index):
    assert set(ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="10")))) == {"4", "15"}
    assert set(ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="11")))) == {"4", "15"}
    assert "5" not in ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="10")))


def test_cone_06_does_not_match_6(index):
    low = ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="06")))
    mid = ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="6")))
    assert low == ["11"]
    assert "11" not in mid and {"10", "12", "16", "17"} <= set(mid)


def test_cone_range_filter_within_vs_overlap(index):
    within = ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="9 - 10")))
    overlap = ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="9 - 10", cone_match="overlap")))
    assert set(within) == {"4"}  # 9 - 11 contains 9 - 10; 10 - 12 does not
    assert {"4", "15", "5"} <= set(overlap)


def test_strict_filters_exclude_unknown_by_default(index):
    # Recipe 12 has cone 6 but no surface, transparency or atmosphere.
    strict = ids(index.search("glaze", top_k=20, filters=SearchFilters(surfaces="glossy")))
    assert "12" not in strict
    relaxed = ids(index.search("glaze", top_k=20, filters=SearchFilters(surfaces="glossy", include_unknown=True)))
    assert "12" in relaxed
    atm = ids(index.search("glaze", top_k=20, filters=SearchFilters(atmospheres="oxidation")))
    assert "12" not in atm
    # Recipe 14's reversed cone range was flagged and is unknown, so strict cone filters drop it.
    assert "14" not in ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="8")))
    assert "14" in ids(index.search("glaze", top_k=20, filters=SearchFilters(cone="8", include_unknown=True)))


def test_filter_values_use_synonyms_and_reject_unknown(index):
    assert ids(index.search("glaze", top_k=20, filters=SearchFilters(surfaces="Gloss", cone="cone 06"))) == ["11"]
    with pytest.raises(ValueError, match="surface"):
        index.search("glaze", filters=SearchFilters(surfaces="shiny"))
    with pytest.raises(ValueError, match="cone"):
        index.search("glaze", filters=SearchFilters(cone="6 - 04"))


def test_multiple_filter_values_match_any(index):
    results = index.search("glaze", top_k=20, filters=SearchFilters(transparencies=["opaque", "transparent"],
                                                                     atmospheres=["wood", "reduction"]))
    assert set(ids(results)) == {"4", "10", "15"}


def test_ties_are_broken_by_source_order(index):
    results = index.search("clear glaze cone 6 oxidation", top_k=20)
    twins = [r for r in results if r.recipe_id in {"16", "17"}]
    assert twins[0].score == twins[1].score
    assert ids(twins) == ["16", "17"]
    assert index.count_with_description(twins[0].description) == 2


def test_no_match_returns_empty(index):
    assert index.search("glaze", filters=SearchFilters(cone="022")) == []


def test_name_and_alias_lookup(index):
    exact = index.lookup_name("matte white")
    assert [(m.recipe_id, m.match_type, m.matched_field) for m in exact] == [("10", "exact", "name")]
    alias = index.lookup_name("floating   GREEN")
    assert alias[0].recipe_id == "10" and alias[0].matched_field == "other_names"
    loose = index.lookup_name("Piers Pure Lux Deluxe Revised")
    assert loose[0].recipe_id == "5" and loose[0].match_type == "normalised"
    # Recipes without semantic tags remain findable by name.
    assert index.lookup_name("Blue Celadon Food Safe")[0].text_available is False


def test_shared_space_requires_trained_projection(tmp_path, sample_recipes, fake_encoder, normaliser):
    emb = encode_recipes(sample_recipes, fake_encoder)
    head = build_projection_head(fake_encoder.info.dim)
    untrained_path = save_projection_checkpoint(tmp_path / "u.pt", head, encoder=fake_encoder.info, trained=False)
    untrained = load_projection_checkpoint(untrained_path, require_trained=False)
    with pytest.raises(UntrainedProjectionError):
        RecipeSearchIndex(sample_recipes, emb, fake_encoder, normaliser, projection=untrained)
    with pytest.raises(UntrainedProjectionError):
        encode_query("glossy", fake_encoder, untrained)

    trained_path = save_projection_checkpoint(tmp_path / "t.pt", head, encoder=fake_encoder.info,
                                              trained=True, training=TRAINING)
    trained = load_projection_checkpoint(trained_path, encoder=fake_encoder.info)
    shared = RecipeSearchIndex(sample_recipes, emb, fake_encoder, normaliser, projection=trained)
    assert shared.space == "shared"
    results = shared.search("glossy celadon", top_k=3)
    assert len(results) == 3
    assert encode_query("glossy", fake_encoder, trained).shape == (128,)


def test_stale_embeddings_are_detected(sample_recipes, fake_encoder, normaliser):
    emb = encode_recipes(sample_recipes, fake_encoder)
    emb.descriptions[0] = "Subtype: something else."
    with pytest.raises(ValueError, match="re-run embed"):
        RecipeSearchIndex(sample_recipes, emb, fake_encoder, normaliser)


def test_end_to_end_pipeline(tmp_config, fake_encoder):
    recipes, report = run_preprocess(tmp_config)
    out = tmp_config.output_dir
    assert (out / "recipes_clean.jsonl").exists() and (out / "data_quality_report.md").exists()
    assert report["unusable_in_scope_records"]["no_usable_semantic_tags"]["recipe_ids"] == ["13"]
    assert report["records_by_type"] == {"Recipe": 10, "Material": 1}

    emb = run_embed(tmp_config, encoder=fake_encoder)
    manifest = json.loads((out / "baseline" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "baseline_pretrained_text_embeddings" and manifest["space"] == "encoder"
    assert manifest["n_rows"] == len(emb.recipe_ids) == 9
    assert manifest["excluded_records"] == {"no_usable_semantic_tags": 1, "out_of_scope_type": 1}

    index = RecipeSearchIndex.from_outputs(tmp_config, encoder=fake_encoder)
    assert index.search("celadon glaze", top_k=1)[0].recipe_id in {"4", "14"}

    # Untrained projections are refused unless explicitly exported for development.
    head = build_projection_head(fake_encoder.info.dim)
    ckpt = save_projection_checkpoint(out / "untrained.pt", head, encoder=fake_encoder.info, trained=False)
    with pytest.raises(UntrainedProjectionError):
        run_project(tmp_config, ckpt)
    dev_dir = run_project(tmp_config, ckpt, allow_untrained=True)
    assert dev_dir.name == UNTRAINED_DIR
    dev_manifest = json.loads((dev_dir / "manifest.json").read_text(encoding="utf-8"))
    assert dev_manifest["trained"] is False and "UNTRAINED" in dev_manifest["warning"]
    with pytest.raises(UntrainedProjectionError):
        RecipeSearchIndex.from_outputs(tmp_config, encoder=fake_encoder, checkpoint=ckpt)

    trained = save_projection_checkpoint(out / "trained.pt", head, encoder=fake_encoder.info,
                                         trained=True, training=TRAINING)
    projected_dir = run_project(tmp_config, trained)
    assert projected_dir.name == "projected"
    vectors = np.load(projected_dir / "embeddings.npy")
    assert vectors.shape == (9, 128)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embed_refuses_stale_preprocessing(tmp_config, fake_encoder):
    run_preprocess(tmp_config)
    tmp_config.cleaning.min_semantic_fields = 2
    with pytest.raises(ValueError, match="re-run"):
        run_embed(tmp_config, encoder=fake_encoder)
