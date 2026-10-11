"""Save and load embedding sets: vectors, recipe-ID row map and manifest."""

from __future__ import annotations

import csv
import datetime as dt
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import PREPROCESSING_VERSION
from .encoder import EncoderInfo, RecipeEmbeddings, _package_version, description_key
from .projection import file_sha256

BASELINE_KIND = "baseline_pretrained_text_embeddings"
PROJECTED_KIND = "projected_text_embeddings"

EMBEDDINGS_FILE = "embeddings.npy"
ID_MAP_FILE = "id_map.csv"
MANIFEST_FILE = "manifest.json"


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_embedding_set(
    directory: str | Path,
    embeddings: RecipeEmbeddings,
    *,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``embeddings.npy`` (row i), ``id_map.csv`` (row i -> recipe ID) and ``manifest.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vectors = np.ascontiguousarray(embeddings.vectors, dtype=np.float32)
    np.save(directory / EMBEDDINGS_FILE, vectors, allow_pickle=False)
    with open(directory / ID_MAP_FILE, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "recipe_id", "description_sha256", "description"])
        for row, (rid, desc) in enumerate(zip(embeddings.recipe_ids, embeddings.descriptions)):
            writer.writerow([row, rid, description_key(desc), desc])
    manifest = {
        "kind": kind,
        "space": embeddings.space,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "dtype": "float32",
        "l2_normalised": True,
        "similarity": "cosine (dot product of unit vectors)",
        "preprocessing_version": PREPROCESSING_VERSION,
        "encoder": embeddings.encoder.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "sentence_transformers": _package_version("sentence-transformers"),
        },
        "files": {
            "embeddings": EMBEDDINGS_FILE,
            "id_map": ID_MAP_FILE,
            "embeddings_sha256": file_sha256(directory / EMBEDDINGS_FILE),
            "id_map_sha256": file_sha256(directory / ID_MAP_FILE),
        },
        **dict(extra or {}),
    }
    write_json(directory / MANIFEST_FILE, manifest)
    return directory


def load_embedding_set(
    directory: str | Path, *, expected_kind: str | None = None
) -> tuple[RecipeEmbeddings, dict[str, Any]]:
    """Load an embedding set, verifying file hashes, row order and shapes."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"No embedding manifest at {manifest_path}; run `python -m text_branch embed`")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_kind and manifest.get("kind") != expected_kind:
        raise ValueError(f"{directory} holds {manifest.get('kind')!r}, expected {expected_kind!r}")
    files = manifest["files"]
    for name in ("embeddings", "id_map"):
        actual = file_sha256(directory / files[name])
        if actual != files[f"{name}_sha256"]:
            raise ValueError(f"{directory / files[name]} does not match its manifest hash; regenerate it")
    vectors = np.load(directory / files["embeddings"], allow_pickle=False)
    with open(directory / files["id_map"], encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(r["row"]) for r in rows] != list(range(len(rows))) or len(rows) != len(vectors):
        raise ValueError(f"{directory}: id_map rows do not line up with embeddings")
    if vectors.shape != (manifest["n_rows"], manifest["dim"]):
        raise ValueError(f"{directory}: embeddings shape {vectors.shape} != manifest")
    embeddings = RecipeEmbeddings(
        recipe_ids=[r["recipe_id"] for r in rows],
        descriptions=[r["description"] for r in rows],
        vectors=vectors.astype(np.float32),
        encoder=EncoderInfo.from_dict(manifest["encoder"]),
        space=manifest["space"],
    )
    return embeddings, manifest
