import dataclasses

import numpy as np
import pytest
import torch

from text_branch.artifacts import BASELINE_KIND, save_embedding_set
from text_branch.config import ProjectionConfig
from text_branch.encoder import encode_recipes
from text_branch.projection import (
    CHECKPOINT_FORMAT,
    IncompatibleCheckpointError,
    TextEmbeddingStore,
    TextProjectionHead,
    UntrainedProjectionError,
    build_projection_head,
    load_projection_checkpoint,
    project_embeddings,
    save_projection_checkpoint,
)

TRAINING = {"run_id": "test-run", "modalities": ["text", "image"], "objective": "info_nce"}


@pytest.fixture
def store(sample_recipes, fake_encoder):
    emb = encode_recipes(sample_recipes, fake_encoder)
    return TextEmbeddingStore(emb.recipe_ids, emb.vectors, emb.descriptions, emb.encoder)


@pytest.mark.parametrize("hidden_dim", [None, 64])
def test_head_outputs_unit_vectors_of_shared_dim(hidden_dim):
    head = TextProjectionHead(32, output_dim=16, hidden_dim=hidden_dim)
    out = head(torch.randn(5, 32))
    assert out.shape == (5, 16)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(5))


def test_input_dim_is_inferred_from_encoder(fake_encoder):
    head = build_projection_head(fake_encoder.info.dim, ProjectionConfig(output_dim=8))
    assert (head.input_dim, head.output_dim) == (fake_encoder.info.dim, 8)
    with pytest.raises(ValueError):
        head(torch.randn(2, fake_encoder.info.dim + 1))


def test_seeded_initialisation_is_reproducible():
    a = build_projection_head(32, ProjectionConfig(seed=3))
    b = build_projection_head(32, ProjectionConfig(seed=3))
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)


def test_gradients_reach_head_but_not_frozen_embeddings(store):
    head = build_projection_head(store.dim, ProjectionConfig(output_dim=8))
    inputs = store.lookup(["4", "5", "10"])
    assert not inputs.requires_grad
    other_modality = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    logits = head(inputs) @ other_modality.T / 0.07
    torch.nn.functional.cross_entropy(logits, torch.arange(3)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert inputs.grad is None


def test_store_lookup_and_missing_ids(store):
    assert store.lookup(["4"]).shape == (1, store.dim)
    assert "13" not in store
    with pytest.raises(KeyError):
        store.lookup(["4", "13"])


def test_same_text_mask_marks_unreliable_negatives(store):
    mask = store.same_text_mask(["16", "17", "4"])
    assert mask.tolist() == [[True, True, False], [True, True, False], [False, False, True]]


def test_store_from_directory(tmp_path, sample_recipes, fake_encoder):
    emb = encode_recipes(sample_recipes, fake_encoder)
    save_embedding_set(tmp_path, emb, kind=BASELINE_KIND)
    loaded = TextEmbeddingStore.from_directory(tmp_path)
    assert loaded.recipe_ids == emb.recipe_ids and loaded.dim == fake_encoder.info.dim


@pytest.mark.parametrize("hidden_dim", [None, 16])
def test_checkpoint_round_trip(tmp_path, fake_encoder, hidden_dim):
    head = build_projection_head(fake_encoder.info.dim, ProjectionConfig(output_dim=8, hidden_dim=hidden_dim))
    path = save_projection_checkpoint(tmp_path / "head.pt", head, encoder=fake_encoder.info,
                                      trained=True, training=TRAINING)
    ckpt = load_projection_checkpoint(path, encoder=fake_encoder.info)
    assert ckpt.trained and ckpt.training == TRAINING and ckpt.sha256
    assert ckpt.head.architecture == head.architecture
    x = np.random.default_rng(0).normal(size=(4, fake_encoder.info.dim)).astype(np.float32)
    with torch.no_grad():
        expected = head.eval()(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(project_embeddings(x, ckpt), expected, atol=1e-6)


def test_trained_flag_requires_training_metadata(tmp_path, fake_encoder):
    head = build_projection_head(fake_encoder.info.dim)
    with pytest.raises(ValueError, match="training"):
        save_projection_checkpoint(tmp_path / "h.pt", head, encoder=fake_encoder.info, trained=True)


def test_untrained_checkpoint_is_refused_for_production(tmp_path, fake_encoder):
    head = build_projection_head(fake_encoder.info.dim)
    path = save_projection_checkpoint(tmp_path / "h.pt", head, encoder=fake_encoder.info, trained=False)
    with pytest.raises(UntrainedProjectionError):
        load_projection_checkpoint(path, encoder=fake_encoder.info)
    ckpt = load_projection_checkpoint(path, encoder=fake_encoder.info, require_trained=False)
    x = np.ones((1, fake_encoder.info.dim), np.float32)
    with pytest.raises(UntrainedProjectionError):
        project_embeddings(x, ckpt)
    assert project_embeddings(x, ckpt, allow_untrained=True).shape == (1, 128)


@pytest.mark.parametrize("change", [
    {"model_name": "other/model"},
    {"revision": "different-rev"},
    {"document_prompt": "passage: "},
])
def test_checkpoint_from_another_encoder_is_rejected(tmp_path, fake_model, fake_encoder, change):
    from text_branch.encoder import TextEncoder

    head = build_projection_head(fake_encoder.info.dim)
    path = save_projection_checkpoint(tmp_path / "h.pt", head, encoder=fake_encoder.info,
                                      trained=True, training=TRAINING)
    other = TextEncoder(dataclasses.replace(fake_encoder.config, **change), model=fake_model)
    with pytest.raises(IncompatibleCheckpointError):
        load_projection_checkpoint(path, encoder=other.info)


def test_dimension_mismatch_is_rejected(tmp_path, fake_encoder):
    head = TextProjectionHead(fake_encoder.info.dim + 1, 8)
    with pytest.raises(IncompatibleCheckpointError):
        save_projection_checkpoint(tmp_path / "h.pt", head, encoder=fake_encoder.info,
                                   trained=True, training=TRAINING)


def test_corrupted_or_foreign_checkpoints_are_rejected(tmp_path, fake_encoder):
    head = build_projection_head(fake_encoder.info.dim, ProjectionConfig(output_dim=8))
    path = save_projection_checkpoint(tmp_path / "h.pt", head, encoder=fake_encoder.info,
                                      trained=True, training=TRAINING)
    payload = torch.load(path, weights_only=True)
    payload["architecture"]["output_dim"] = 16  # weights no longer fit
    torch.save(payload, tmp_path / "bad_arch.pt")
    with pytest.raises(IncompatibleCheckpointError, match="architecture"):
        load_projection_checkpoint(tmp_path / "bad_arch.pt")

    payload = torch.load(path, weights_only=True)
    payload["preprocessing_version"] = "recipe-tags-v0"
    torch.save(payload, tmp_path / "old_prep.pt")
    with pytest.raises(IncompatibleCheckpointError, match="preprocessing"):
        load_projection_checkpoint(tmp_path / "old_prep.pt")

    torch.save({"format": "something-else"}, tmp_path / "foreign.pt")
    with pytest.raises(IncompatibleCheckpointError):
        load_projection_checkpoint(tmp_path / "foreign.pt")
    assert payload["format"] == CHECKPOINT_FORMAT
