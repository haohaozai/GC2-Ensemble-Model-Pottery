"""Trainable text projection head: frozen encoder embeddings -> shared space.

A freshly built head is randomly initialised. Matching the shared dimension
does NOT align text with the image or formula branches; only joint
contrastive training does. Untrained heads are therefore refused wherever a
shared-space result is produced, unless a caller explicitly opts in for
development and labels the output as untrained.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import PREPROCESSING_VERSION, ProjectionConfig
from .encoder import EncoderInfo

CHECKPOINT_FORMAT = "gc2-text-projection-head"
CHECKPOINT_FORMAT_VERSION = 1


class UntrainedProjectionError(RuntimeError):
    """Raised when an untrained projection would be used as a shared space."""


class IncompatibleCheckpointError(ValueError):
    """Checkpoint does not match this encoder, preprocessing or architecture."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TextProjectionHead(nn.Module):
    """Linear layer (or 2-layer MLP) followed by L2 normalisation.

    ``forward`` takes frozen encoder embeddings ``[batch, input_dim]`` and
    returns unit vectors ``[batch, output_dim]``. Gradients flow through the
    head only; the encoder is never part of this module.
    """

    def __init__(self, input_dim: int, output_dim: int = 128,
                 hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)
        self.dropout = float(dropout)
        if self.hidden_dim is None:
            self.net: nn.Module = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

    @property
    def architecture(self) -> dict[str, Any]:
        return {
            "type": "linear" if self.hidden_dim is None else "mlp",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    @classmethod
    def from_architecture(cls, arch: Mapping[str, Any]) -> "TextProjectionHead":
        return cls(arch["input_dim"], arch["output_dim"], arch.get("hidden_dim"), arch.get("dropout", 0.0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"Expected encoder embeddings of size {self.input_dim}, got {embeddings.shape[-1]}")
        return F.normalize(self.net(embeddings), p=2, dim=-1)


def build_projection_head(encoder_dim: int, config: ProjectionConfig | None = None) -> TextProjectionHead:
    """New (untrained) head; input size comes from the encoder, never hard-coded."""
    config = config or ProjectionConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return TextProjectionHead(encoder_dim, config.output_dim, config.hidden_dim, config.dropout)


@dataclass
class ProjectionCheckpoint:
    head: TextProjectionHead
    trained: bool
    encoder: EncoderInfo
    preprocessing_version: str
    training: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    sha256: str | None = None


def save_projection_checkpoint(
    path: str | Path,
    head: TextProjectionHead,
    *,
    encoder: EncoderInfo,
    trained: bool,
    training: Mapping[str, Any] | None = None,
) -> Path:
    """Save weights plus architecture, encoder and preprocessing metadata.

    ``trained=True`` requires ``training`` metadata describing the joint run
    (e.g. run id, modalities, objective, data split, epochs, metrics) so a
    random head cannot be labelled as trained by accident.
    """
    training = dict(training or {})
    if trained and not training:
        raise ValueError("A trained checkpoint must include `training` metadata describing the joint run")
    json.dumps(training)  # must be plain JSON-like data so weights_only loading works
    if head.input_dim != encoder.dim:
        raise IncompatibleCheckpointError(f"Head input {head.input_dim} != encoder dim {encoder.dim}")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": head.architecture,
        "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "trained": bool(trained),
        "encoder": encoder.to_dict(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "training": training,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_projection_checkpoint(
    path: str | Path,
    *,
    encoder: EncoderInfo | None = None,
    require_trained: bool = True,
    map_location: str = "cpu",
) -> ProjectionCheckpoint:
    """Load and validate a checkpoint.

    Raises ``IncompatibleCheckpointError`` on format, architecture, encoder or
    preprocessing mismatch, and ``UntrainedProjectionError`` when
    ``require_trained`` and the checkpoint was not produced by training.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise IncompatibleCheckpointError(f"{path} is not a text projection checkpoint")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise IncompatibleCheckpointError(f"Unsupported checkpoint format_version {payload.get('format_version')}")
    head = TextProjectionHead.from_architecture(payload["architecture"])
    try:
        head.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise IncompatibleCheckpointError(f"Weights do not match the stored architecture: {exc}") from exc
    head.eval()
    stored_encoder = EncoderInfo.from_dict(payload["encoder"])
    problems = []
    if stored_encoder.dim != head.input_dim:
        problems.append(f"head input {head.input_dim} != stored encoder dim {stored_encoder.dim}")
    if payload["preprocessing_version"] != PREPROCESSING_VERSION:
        problems.append(f"preprocessing {payload['preprocessing_version']!r} != {PREPROCESSING_VERSION!r}")
    if encoder is not None:
        problems += stored_encoder.incompatibilities(encoder)
    if problems:
        raise IncompatibleCheckpointError(f"{path}: " + "; ".join(problems))
    trained = bool(payload["trained"])
    if require_trained and not trained:
        raise UntrainedProjectionError(
            f"{path} is an untrained projection; shared-space inference needs a checkpoint "
            "produced by joint training"
        )
    return ProjectionCheckpoint(
        head=head,
        trained=trained,
        encoder=stored_encoder,
        preprocessing_version=payload["preprocessing_version"],
        training=dict(payload.get("training") or {}),
        path=Path(path),
        sha256=file_sha256(path),
    )


def project_embeddings(
    embeddings: np.ndarray,
    checkpoint: ProjectionCheckpoint,
    *,
    allow_untrained: bool = False,
    batch_size: int = 8192,
) -> np.ndarray:
    """Map pretrained text embeddings into the shared space (inference only).

    Refuses untrained checkpoints unless ``allow_untrained`` is set, which is
    for development exports that must then be labelled untrained.
    """
    if not checkpoint.trained and not allow_untrained:
        raise UntrainedProjectionError("Refusing to project with an untrained checkpoint")
    x = np.asarray(embeddings, dtype=np.float32)
    head = checkpoint.head.eval()
    if x.ndim != 2 or x.shape[1] != head.input_dim:
        raise ValueError(f"Expected embeddings of shape [n, {head.input_dim}], got {x.shape}")
    device = next(head.parameters()).device
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            chunks.append(head(batch).cpu().numpy())
    out = np.concatenate(chunks) if chunks else np.zeros((0, head.output_dim), np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Projection produced non-finite values")
    return out.astype(np.float32)


class TextEmbeddingStore:
    """Frozen baseline text embeddings by recipe ID, for joint training.

    The encoder is frozen, so its outputs are precomputed once and looked up
    per batch; only the projection head receives gradients.
    """

    def __init__(self, recipe_ids: Sequence[str], vectors: np.ndarray,
                 descriptions: Sequence[str], encoder: EncoderInfo):
        if not (len(recipe_ids) == len(vectors) == len(descriptions)):
            raise ValueError("recipe_ids, vectors and descriptions must align")
        self.encoder = encoder
        self.recipe_ids = list(recipe_ids)
        self._rows = {rid: i for i, rid in enumerate(self.recipe_ids)}
        if len(self._rows) != len(self.recipe_ids):
            raise ValueError("recipe_ids must be unique")
        self._vectors = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32))
        groups: dict[str, int] = {}
        self._text_group = torch.tensor([groups.setdefault(d, len(groups)) for d in descriptions])

    @classmethod
    def from_directory(cls, directory: str | Path) -> "TextEmbeddingStore":
        from .artifacts import BASELINE_KIND, load_embedding_set

        embeddings, _ = load_embedding_set(directory, expected_kind=BASELINE_KIND)
        return cls(embeddings.recipe_ids, embeddings.vectors, embeddings.descriptions, embeddings.encoder)

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1])

    def __contains__(self, recipe_id: object) -> bool:
        return recipe_id in self._rows

    def __len__(self) -> int:
        return len(self.recipe_ids)

    def _rows_for(self, recipe_ids: Sequence[str]) -> torch.Tensor:
        missing = [rid for rid in recipe_ids if rid not in self._rows]
        if missing:
            raise KeyError(f"No text embedding for recipe IDs {missing[:10]} (unavailable for text retrieval)")
        return torch.tensor([self._rows[rid] for rid in recipe_ids], dtype=torch.long)

    def lookup(self, recipe_ids: Sequence[str], device: str | torch.device | None = None) -> torch.Tensor:
        """``[batch, encoder_dim]`` frozen embeddings (no grad) for the given IDs."""
        out = self._vectors[self._rows_for(recipe_ids)]
        return out.to(device) if device is not None else out

    def same_text_mask(self, recipe_ids: Sequence[str]) -> torch.Tensor:
        """``[batch, batch]`` bool, True where two recipes have identical descriptions.

        Identical descriptions give identical text vectors, so off-diagonal
        True entries are not reliable negatives in a contrastive loss.
        """
        groups = self._text_group[self._rows_for(recipe_ids)]
        return groups[:, None] == groups[None, :]
