"""Frozen pretrained sentence encoder with a content-addressed embedding cache."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .config import EncoderConfig
from .records import CleanRecipe

if TYPE_CHECKING:
    from .projection import ProjectionCheckpoint

log = logging.getLogger(__name__)

NORM_TOLERANCE = 1e-3
_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


class EncoderError(RuntimeError):
    pass


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def check_unit_vectors(vectors: np.ndarray, dim: int, what: str = "embeddings") -> None:
    """Raise unless ``vectors`` is ``[n, dim]``, finite and L2-normalised."""
    if vectors.ndim != 2 or vectors.shape[1] != dim:
        raise EncoderError(f"{what} have shape {vectors.shape}, expected [n, {dim}]")
    if not np.isfinite(vectors).all():
        raise EncoderError(f"{what} contain non-finite values")
    if len(vectors):
        norms = np.linalg.norm(vectors, axis=1)
        if np.abs(norms - 1.0).max() > NORM_TOLERANCE:
            raise EncoderError(f"{what} are not L2-normalised (norms {norms.min():.4f}..{norms.max():.4f})")


def description_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class EncoderInfo:
    """Everything needed to tell whether two sets of text vectors are comparable."""

    model_name: str
    requested_revision: str | None
    resolved_revision: str | None
    dim: int
    max_seq_length: int | None
    query_prompt: str
    document_prompt: str
    sentence_transformers_version: str | None
    normalised: bool = True

    @property
    def revision(self) -> str | None:
        return self.resolved_revision or self.requested_revision

    @property
    def fingerprint(self) -> str:
        """Cache key: changes with model, revision, prompts or library minor version."""
        st = self.sentence_transformers_version
        payload = {
            "model_name": self.model_name,
            "revision": self.revision,
            "dim": self.dim,
            "max_seq_length": self.max_seq_length,
            "query_prompt": self.query_prompt,
            "document_prompt": self.document_prompt,
            "sentence_transformers": ".".join(st.split(".")[:2]) if st else None,
            "normalised": self.normalised,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EncoderInfo":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def incompatibilities(self, other: "EncoderInfo") -> list[str]:
        """Differences that make vectors from ``self`` and ``other`` incomparable."""
        problems = []
        if self.model_name != other.model_name:
            problems.append(f"model {self.model_name!r} != {other.model_name!r}")
        if self.revision and other.revision and self.revision != other.revision:
            problems.append(f"revision {self.revision!r} != {other.revision!r}")
        for attr in ("dim", "query_prompt", "document_prompt", "normalised"):
            if getattr(self, attr) != getattr(other, attr):
                problems.append(f"{attr} {getattr(self, attr)!r} != {getattr(other, attr)!r}")
        return problems


class TextEncoder:
    """Frozen pretrained Sentence Transformers encoder.

    Uses ``encode_query`` / ``encode_document`` so each model's own prompt
    conventions apply; ``EncoderConfig.query_prompt`` / ``document_prompt``
    override them. Output vectors are float32 and L2-normalised. ``model``
    can be injected (tests use a download-free fake).
    """

    def __init__(self, config: EncoderConfig | None = None, *, model: Any = None):
        self.config = config or EncoderConfig()
        self.device = resolve_device(self.config.device)
        is_injected = model is not None
        self.model = model if is_injected else self._load()
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
        prompts = getattr(self.model, "prompts", None) or {}
        self.info = EncoderInfo(
            model_name=self.config.model_name,
            requested_revision=self.config.revision,
            resolved_revision=None if is_injected else self._resolved_revision(),
            dim=self._infer_dim(),
            max_seq_length=getattr(self.model, "max_seq_length", None),
            query_prompt=self._prompt("query", prompts),
            document_prompt=self._prompt("document", prompts),
            sentence_transformers_version=None if is_injected else _package_version("sentence-transformers"),
        )

    def _load(self) -> Any:
        from sentence_transformers import SentenceTransformer  # heavy; import lazily

        torch.manual_seed(self.config.seed)
        name, revision = self.config.model_name, self.config.revision
        log.info("Loading %s (revision %s) on %s", name, revision, self.device)
        if revision and _COMMIT_HASH.fullmatch(revision):
            # A pinned commit is immutable: use the local copy without network calls if present.
            try:
                return SentenceTransformer(name, revision=revision, device=self.device, local_files_only=True)
            except (OSError, ValueError):
                log.info("Pinned revision not cached locally; downloading")
        return SentenceTransformer(name, revision=revision, device=self.device)

    def _resolved_revision(self) -> str | None:
        """Commit hash of the loaded Hub snapshot, or None for local paths."""
        card = getattr(self.model, "model_card_data", None)
        revision = getattr(card, "base_model_revision", None)
        if revision:
            return revision
        if Path(self.config.model_name).exists():
            return None
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(self.config.model_name, revision=self.config.revision, local_files_only=True)
        except (OSError, ValueError):
            return None
        name = Path(path).name  # .../snapshots/<commit hash>
        return name if _COMMIT_HASH.fullmatch(name) else None

    def _prompt(self, kind: str, model_prompts: Mapping[str, str]) -> str:
        override = self.config.query_prompt if kind == "query" else self.config.document_prompt
        return override if override is not None else (model_prompts.get(kind) or "")

    def _infer_dim(self) -> int:
        for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            getter = getattr(self.model, attr, None)
            if getter is not None and getter():
                return int(getter())
        probe = self._raw_encode("document", ["dimension probe"])
        return int(probe.shape[1])

    def _raw_encode(self, kind: str, texts: list[str]) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        override = self.config.query_prompt if kind == "query" else self.config.document_prompt
        if override is not None:
            kwargs["prompt"] = override
        method = getattr(self.model, f"encode_{kind}", None)
        if method is None:  # sentence-transformers < 5: plain encode + explicit prompt
            method = self.model.encode
        with torch.inference_mode():
            return np.asarray(method(texts, **kwargs), dtype=np.float32)

    def _encode(self, kind: str, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.info.dim), dtype=np.float32)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Cannot encode empty or non-string text")
        vectors = self._raw_encode(kind, texts)
        check_unit_vectors(vectors, self.info.dim, f"{kind} embeddings")
        return vectors

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode("document", texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode("query", texts)


class EmbeddingCache:
    """Document-embedding cache keyed by encoder fingerprint + description hash.

    One ``.npz`` file per encoder fingerprint, so a new model, revision,
    prompt or library version starts empty; an edited description misses
    because its sha256 changes.
    """

    def __init__(self, directory: str | Path, info: EncoderInfo):
        self.info = info
        self.path = Path(directory) / f"{info.fingerprint[:24]}.npz"
        self._table: dict[str, np.ndarray] = {}
        self._dirty = False
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                keys, vectors = data["keys"], data["vectors"]
            if vectors.ndim == 2 and vectors.shape[1] == info.dim:
                self._table = dict(zip(keys.tolist(), vectors.astype(np.float32)))
            else:
                log.warning("Ignoring cache %s with unexpected shape %s", self.path, vectors.shape)

    def __len__(self) -> int:
        return len(self._table)

    def get(self, key: str) -> np.ndarray | None:
        return self._table.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._table[key] = np.asarray(vector, dtype=np.float32)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._table)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            np.savez(handle, keys=np.array(keys, dtype=str), vectors=np.stack([self._table[k] for k in keys]))
        tmp.replace(self.path)
        self.path.with_suffix(".json").write_text(json.dumps(self.info.to_dict(), indent=2), encoding="utf-8")
        self._dirty = False


@dataclass
class RecipeEmbeddings:
    """Row ``i`` of ``vectors`` belongs to ``recipe_ids[i]``."""

    recipe_ids: list[str]
    descriptions: list[str]
    vectors: np.ndarray
    encoder: EncoderInfo
    space: str = "encoder"  # "encoder" = pretrained baseline, "shared" = projected

    def __post_init__(self) -> None:
        if not (len(self.recipe_ids) == len(self.descriptions) == len(self.vectors)):
            raise ValueError("recipe_ids, descriptions and vectors must have equal length")
        if len(set(self.recipe_ids)) != len(self.recipe_ids):
            raise ValueError("recipe_ids must be unique")
        self._rows = {rid: i for i, rid in enumerate(self.recipe_ids)}

    def row_of(self, recipe_id: str) -> int:
        return self._rows[recipe_id]


def encode_recipes(
    recipes: Iterable[CleanRecipe],
    encoder: TextEncoder,
    *,
    cache: EmbeddingCache | None = None,
) -> RecipeEmbeddings:
    """Embed every text-available recipe, in input order, with the frozen encoder.

    Identical descriptions are encoded once. Recipes that are unavailable for
    text retrieval are skipped, never embedded as empty text.
    """
    selected = [r for r in recipes if r.text_available]
    recipe_ids = [r.recipe_id for r in selected]
    descriptions = [r.description for r in selected]
    unique = sorted(set(descriptions))
    table: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for text in unique:
        cached = cache.get(description_key(text)) if cache is not None else None
        if cached is None:
            missing.append(text)
        else:
            table[text] = cached
    log.info("%d recipes, %d unique descriptions, %d from cache, %d to encode",
             len(selected), len(unique), len(unique) - len(missing), len(missing))
    if missing:
        torch.manual_seed(encoder.config.seed)
        for text, vector in zip(missing, encoder.encode_documents(missing)):
            table[text] = vector
            if cache is not None:
                cache.put(description_key(text), vector)
        if cache is not None:
            cache.save()
    dim = encoder.info.dim
    vectors = np.stack([table[d] for d in descriptions]) if descriptions else np.zeros((0, dim), np.float32)
    check_unit_vectors(vectors, dim, "recipe embeddings")
    return RecipeEmbeddings(recipe_ids, descriptions, vectors.astype(np.float32), encoder.info)


def encode_query(
    text: str,
    encoder: TextEncoder,
    projection: "ProjectionCheckpoint | None" = None,
) -> np.ndarray:
    """Encode a user query; with ``projection``, map it into the shared space.

    The projection must be trained and built for this encoder; untrained
    projections are rejected (see ``project_embeddings``).
    """
    vector = encoder.encode_queries([text])[0]
    if projection is None:
        return vector
    from .projection import IncompatibleCheckpointError, project_embeddings

    problems = projection.encoder.incompatibilities(encoder.info)
    if problems:
        raise IncompatibleCheckpointError("Query encoder does not match checkpoint: " + "; ".join(problems))
    return project_embeddings(vector[None, :], projection)[0]
