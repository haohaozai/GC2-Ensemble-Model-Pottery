"""Real-model smoke test. Opt in with TEXT_BRANCH_MODEL_TESTS=1 (may download ~90 MB).

These checks confirm the real encoder loads and behaves sanely. They are not
a retrieval-quality evaluation.
"""

import os

import numpy as np
import pytest

from text_branch.config import EncoderConfig
from text_branch.encoder import TextEncoder

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(os.environ.get("TEXT_BRANCH_MODEL_TESTS") != "1",
                       reason="set TEXT_BRANCH_MODEL_TESTS=1 to load the real sentence-transformers model"),
]


@pytest.fixture(scope="module")
def encoder():
    return TextEncoder(EncoderConfig(device="cpu"))


def test_real_encoder_metadata(encoder):
    info = encoder.info
    assert info.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert info.dim == 384
    assert info.resolved_revision == EncoderConfig().revision
    assert info.sentence_transformers_version
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_real_encoder_outputs_are_normalised_and_reproducible(encoder):
    texts = ["Subtype: iron celadon glaze. Surface: glossy.", "Subtype: glaze. Firing cone: 06."]
    a = encoder.encode_documents(texts)
    b = encoder.encode_documents(list(reversed(texts)))[::-1]
    assert a.shape == (2, 384) and np.isfinite(a).all()
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_real_encoder_sanity_ordering(encoder):
    query = encoder.encode_queries(["glossy transparent celadon glaze"])[0]
    docs = encoder.encode_documents([
        "Subtype: iron celadon glaze. Surface: glossy. Transparency: transparent.",
        "Subtype: stoneware throwing clay body. Firing cone: 10.",
    ])
    scores = docs @ query
    assert scores[0] > scores[1]
