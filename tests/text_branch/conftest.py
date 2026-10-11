"""Shared fixtures for text-branch tests. No model downloads happen here."""

from __future__ import annotations

import csv
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from text_branch.cleaning import TagNormaliser, clean_rows
from text_branch.config import DEFAULT_COLUMNS, CleaningConfig, EncoderConfig, TextBranchConfig
from text_branch.encoder import TextEncoder

FAKE_DIM = 32


class FakeSentenceModel:
    """Deterministic hashed bag-of-words stand-in for a SentenceTransformer."""

    def __init__(self, dim: int = FAKE_DIM):
        self.dim = dim
        self.prompts = {"query": "", "document": ""}
        self.max_seq_length = 128
        self.calls: list[tuple[str, list[str]]] = []

    def get_embedding_dimension(self) -> int:
        return self.dim

    def _embed(self, texts, prompt=None):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", f"{prompt or ''}{text}".lower()):
                out[i, zlib.crc32(token.encode()) % self.dim] += 1.0
            out[i, -1] += 0.01  # never all-zero
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def encode_query(self, texts, prompt=None, **kwargs):
        self.calls.append(("query", list(texts)))
        return self._embed(texts, prompt)

    def encode_document(self, texts, prompt=None, **kwargs):
        self.calls.append(("document", list(texts)))
        return self._embed(texts, prompt)

    def encoded_texts(self, kind: str = "document") -> list[str]:
        return [t for k, texts in self.calls if k == kind for t in texts]


def make_row(recipe_id: str = "1", **fields: str) -> dict[str, str]:
    """A raw CSV row keyed by Glazy column names; unspecified fields are blank."""
    row = {column: "" for column in DEFAULT_COLUMNS.values()}
    row["ID"] = recipe_id
    row["Type"] = "Recipe"
    for logical, value in fields.items():
        row[DEFAULT_COLUMNS[logical]] = value
    return row


SAMPLE_ROWS = [
    make_row("4", name="Celadon-type glaze David Pier", subtype="Glaze - Iron - Celadon", cone="9 - 11",
             surface="Glossy", transparency="Transparent", atmospheres="Reduction"),
    make_row("5", name="Pier's Pure Lux-Deluxe Revised", subtype="Glaze - Clear", cone="8 - 9",
             surface="Glossy", transparency="Transparent", atmospheres="Oxidation"),
    make_row("10", name="Matte White", subtype="Glaze - Matte", cone="6", surface="Matte",
             transparency="Opaque", atmospheres="Oxidation; Reduction", country="Canada",
             other_names="Hesselberth Green, Floating Green"),
    make_row("11", name="Low Fire Clear", subtype="Glaze - Clear", cone="06", surface="Glossy",
             transparency="Transparent", atmospheres="Oxidation"),
    make_row("12", name="Mystery Glaze", subtype="Glaze", cone="6"),  # surface/atm unknown
    make_row("13", name="Blue Celadon Food Safe"),  # no semantic tags at all
    make_row("14", name="Backwards", subtype="Glaze - Iron - Celadon", cone="10 - 6", surface="Glossy"),
    make_row("15", name="Wood Shino", subtype="Glaze - Shino", cone="10 - 12", surface="Satin",
             transparency="Opaque", atmospheres="Wood; Reduction; Salt & Soda"),
    make_row("16", name="Twin Clear A", subtype="Glaze - Clear", cone="6", surface="Glossy",
             transparency="Transparent", atmospheres="Oxidation"),
    make_row("17", name="Twin Clear B", subtype="Glaze - Clear", cone="6", surface="Glossy",
             transparency="Transparent", atmospheres="Oxidation"),
    {**make_row("900", subtype="Feldspar"), "Type": "Material"},
]


@pytest.fixture
def normaliser() -> TagNormaliser:
    return TagNormaliser(CleaningConfig())


@pytest.fixture
def sample_recipes(normaliser):
    return clean_rows(SAMPLE_ROWS, normaliser)


@pytest.fixture
def fake_model() -> FakeSentenceModel:
    return FakeSentenceModel()


@pytest.fixture
def fake_encoder(fake_model) -> TextEncoder:
    return TextEncoder(EncoderConfig(model_name="fake/hash-bow", revision="test-rev", device="cpu"),
                       model=fake_model)


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    header = list(dict.fromkeys(["ID", *DEFAULT_COLUMNS.values(), "Description"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**{h: "" for h in header}, **row})
    return path


@pytest.fixture
def tmp_config(tmp_path) -> TextBranchConfig:
    config = TextBranchConfig()
    config.paths.source_csv = str(write_csv(tmp_path / "glazy.csv", SAMPLE_ROWS))
    config.paths.output_dir = str(tmp_path / "out")
    config.encoder.model_name = "fake/hash-bow"
    config.encoder.revision = "test-rev"
    return config
