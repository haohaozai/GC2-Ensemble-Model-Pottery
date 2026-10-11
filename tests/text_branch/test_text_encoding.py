import dataclasses
import random

import numpy as np
import pytest
import torch

from conftest import FakeSentenceModel, make_row
from text_branch.artifacts import BASELINE_KIND, load_embedding_set, save_embedding_set
from text_branch.config import EncoderConfig
from text_branch.encoder import EmbeddingCache, TextEncoder, encode_query, encode_recipes


def test_recipe_ids_align_with_vector_rows(sample_recipes, fake_encoder, fake_model):
    shuffled = list(sample_recipes)
    random.Random(7).shuffle(shuffled)
    emb = encode_recipes(shuffled, fake_encoder)
    expected_ids = [r.recipe_id for r in shuffled if r.text_available]
    assert emb.recipe_ids == expected_ids
    by_id = {r.recipe_id: r for r in sample_recipes}
    for row, rid in enumerate(emb.recipe_ids):
        assert emb.row_of(rid) == row
        assert emb.descriptions[row] == by_id[rid].description
        reference = FakeSentenceModel()._embed([by_id[rid].description])[0]
        np.testing.assert_allclose(emb.vectors[row], reference, atol=1e-6)


def test_vectors_have_expected_shape_finite_and_unit_norm(sample_recipes, fake_encoder):
    emb = encode_recipes(sample_recipes, fake_encoder)
    assert emb.vectors.dtype == np.float32
    assert emb.vectors.shape == (sum(r.text_available for r in sample_recipes), fake_encoder.info.dim)
    assert np.isfinite(emb.vectors).all()
    np.testing.assert_allclose(np.linalg.norm(emb.vectors, axis=1), 1.0, atol=1e-5)


def test_unavailable_recipes_are_never_embedded(sample_recipes, fake_encoder, fake_model):
    emb = encode_recipes(sample_recipes, fake_encoder)
    assert "13" not in emb.recipe_ids and "900" not in emb.recipe_ids
    assert all(t.strip() for t in fake_model.encoded_texts())


def test_identical_descriptions_are_encoded_once(sample_recipes, fake_encoder, fake_model):
    emb = encode_recipes(sample_recipes, fake_encoder)
    encoded = fake_model.encoded_texts()
    assert len(encoded) == len(set(encoded)) == len(set(emb.descriptions))
    np.testing.assert_array_equal(emb.vectors[emb.row_of("16")], emb.vectors[emb.row_of("17")])


def test_cache_reuse_and_invalidation(tmp_path, normaliser, sample_recipes, fake_model):
    config = EncoderConfig(model_name="fake/hash-bow", revision="r1", device="cpu")
    encoder = TextEncoder(config, model=fake_model)
    first = encode_recipes(sample_recipes, encoder, cache=EmbeddingCache(tmp_path, encoder.info))
    n_first = len(fake_model.encoded_texts())

    # Same encoder + descriptions: everything comes from the cache.
    again = encode_recipes(sample_recipes, encoder, cache=EmbeddingCache(tmp_path, encoder.info))
    assert len(fake_model.encoded_texts()) == n_first
    np.testing.assert_array_equal(first.vectors, again.vectors)

    # A changed description misses the cache for that description only.
    changed = sample_recipes + [normaliser.clean(make_row("99", subtype="Glaze - Ash"))]
    encode_recipes(changed, encoder, cache=EmbeddingCache(tmp_path, encoder.info))
    assert fake_model.encoded_texts()[n_first:] == ["Subtype: ash glaze."]

    # A different encoder configuration uses a different cache file.
    other = TextEncoder(dataclasses.replace(config, document_prompt="passage: "), model=fake_model)
    assert other.info.fingerprint != encoder.info.fingerprint
    assert len(EmbeddingCache(tmp_path, other.info)) == 0
    other_rev = TextEncoder(dataclasses.replace(config, revision="r2"), model=fake_model)
    assert other_rev.info.fingerprint != encoder.info.fingerprint


def test_query_and_document_use_their_own_entry_points(fake_encoder, fake_model):
    fake_encoder.encode_documents(["Subtype: glaze."])
    vector = encode_query("glossy glaze", fake_encoder)
    assert [kind for kind, _ in fake_model.calls] == ["document", "query"]
    assert vector.shape == (fake_encoder.info.dim,)


def test_prompt_overrides_are_recorded(fake_model):
    encoder = TextEncoder(EncoderConfig(model_name="fake", revision=None, query_prompt="query: ",
                                        document_prompt="passage: "), model=fake_model)
    assert (encoder.info.query_prompt, encoder.info.document_prompt) == ("query: ", "passage: ")


def test_empty_query_is_rejected(fake_encoder):
    with pytest.raises(ValueError):
        encode_query("   ", fake_encoder)


def test_torch_encoder_is_frozen(fake_model):
    class TorchFake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)
            self.inner = fake_model

        def get_embedding_dimension(self):
            return self.inner.dim

    model = TorchFake().train()
    TextEncoder(EncoderConfig(model_name="fake", revision=None), model=model)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_embedding_set_round_trip_and_tamper_detection(tmp_path, sample_recipes, fake_encoder):
    emb = encode_recipes(sample_recipes, fake_encoder)
    save_embedding_set(tmp_path / "baseline", emb, kind=BASELINE_KIND)
    loaded, manifest = load_embedding_set(tmp_path / "baseline", expected_kind=BASELINE_KIND)
    assert loaded.recipe_ids == emb.recipe_ids and loaded.descriptions == emb.descriptions
    np.testing.assert_array_equal(loaded.vectors, emb.vectors)
    assert manifest["dim"] == fake_encoder.info.dim
    assert manifest["encoder"]["fingerprint"] == fake_encoder.info.fingerprint
    assert manifest["preprocessing_version"]

    vectors = np.load(tmp_path / "baseline" / "embeddings.npy")
    np.save(tmp_path / "baseline" / "embeddings.npy", vectors[::-1].copy())
    with pytest.raises(ValueError, match="manifest hash"):
        load_embedding_set(tmp_path / "baseline")
