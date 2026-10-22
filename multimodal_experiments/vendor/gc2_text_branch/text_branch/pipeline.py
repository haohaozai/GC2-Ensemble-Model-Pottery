"""End-to-end steps behind the CLI: preprocess, embed, project."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .artifacts import BASELINE_KIND, PROJECTED_KIND, load_embedding_set, save_embedding_set, write_json
from .cleaning import TagNormaliser, clean_rows, read_clean_recipes, read_source_rows, write_clean_recipes
from .config import PREPROCESSING_VERSION, REPO_ROOT, TextBranchConfig, config_digest
from .encoder import EmbeddingCache, RecipeEmbeddings, TextEncoder, encode_recipes
from .projection import file_sha256, load_projection_checkpoint, project_embeddings
from .quality import build_quality_report, render_quality_markdown
from .records import CleanRecipe
from .search import CLEAN_RECIPES_FILE

log = logging.getLogger(__name__)

PREPROCESS_MANIFEST_FILE = "preprocess_manifest.json"
UNTRAINED_DIR = "projected_UNTRAINED_DEV_ONLY"


def _display_path(path: Path) -> str:
    try:
        return Path(os.path.relpath(path, REPO_ROOT)).as_posix()
    except ValueError:  # different drive on Windows
        return str(path)


def run_preprocess(config: TextBranchConfig, *, source: str | Path | None = None
                   ) -> tuple[list[CleanRecipe], dict[str, Any]]:
    """Clean the source CSV; write cleaned JSONL, quality report and manifest."""
    src = Path(source) if source else config.paths.resolve(config.paths.source_csv)
    normaliser = TagNormaliser(config.cleaning)
    recipes = clean_rows(read_source_rows(src, config.cleaning), normaliser)
    out = config.output_dir
    write_clean_recipes(out / CLEAN_RECIPES_FILE, recipes)
    source_info = {
        "path": _display_path(src),
        "sha256": file_sha256(src),
        "rows": len(recipes),
        "preprocessing_version": PREPROCESSING_VERSION,
        "cleaning_config_sha256": config_digest(config.cleaning),
    }
    report = build_quality_report(recipes, source=source_info)
    write_json(out / "data_quality_report.json", report)
    (out / "data_quality_report.md").write_text(render_quality_markdown(report), encoding="utf-8")
    write_json(out / PREPROCESS_MANIFEST_FILE, {
        **source_info,
        "outputs": {"clean_recipes": CLEAN_RECIPES_FILE, "quality_report": "data_quality_report.md"},
        "text_available": sum(r.text_available for r in recipes),
        "embedded_fields": ["subtype", "surface", "transparency", "cone", "atmospheres"],
        "description_metadata_fields": list(config.cleaning.description_metadata_fields),
        "metadata_only_fields": ["recipe_id", "name", "record_type", "state", "country", "other_names"],
    })
    log.info("Preprocessed %d rows -> %s", len(recipes), out / CLEAN_RECIPES_FILE)
    return recipes, report


def load_preprocessed(config: TextBranchConfig) -> tuple[list[CleanRecipe], dict[str, Any]]:
    out = config.output_dir
    manifest_path = out / PREPROCESS_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; run `python -m text_branch preprocess` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["cleaning_config_sha256"] != config_digest(config.cleaning):
        raise ValueError("Cleaning config changed since preprocessing; re-run `python -m text_branch preprocess`")
    normaliser = TagNormaliser(config.cleaning)
    return read_clean_recipes(out / CLEAN_RECIPES_FILE, normaliser.cones), manifest


def run_embed(config: TextBranchConfig, *, encoder: TextEncoder | None = None) -> RecipeEmbeddings:
    """Embed text-available recipes with the frozen pretrained encoder."""
    recipes, prep_manifest = load_preprocessed(config)
    encoder = encoder or TextEncoder(config.encoder)
    cache = EmbeddingCache(config.output_dir / "cache", encoder.info) if config.encoder.use_cache else None
    embeddings = encode_recipes(recipes, encoder, cache=cache)
    excluded = Counter(r.unavailable_reason for r in recipes if not r.text_available)
    save_embedding_set(config.output_dir / "baseline", embeddings, kind=BASELINE_KIND, extra={
        "note": "Pretrained text-encoder space for the text-only baseline. NOT the shared multimodal space.",
        "source": {k: prep_manifest[k] for k in ("path", "sha256", "rows", "cleaning_config_sha256")},
        "encoder_config": dataclasses.asdict(config.encoder),
        "device": encoder.device,
        "excluded_records": dict(excluded.most_common()),
        "unique_descriptions": len(set(embeddings.descriptions)),
    })
    log.info("Saved %d x %d baseline embeddings", *embeddings.vectors.shape)
    return embeddings


def run_project(config: TextBranchConfig, checkpoint: str | Path, *, allow_untrained: bool = False) -> Path:
    """Export shared-space text vectors using a projection checkpoint.

    Untrained checkpoints are refused unless ``allow_untrained``; such dev
    exports go to a separate, clearly named directory and manifest.
    """
    out = config.output_dir
    baseline, baseline_manifest = load_embedding_set(out / "baseline", expected_kind=BASELINE_KIND)
    ckpt = load_projection_checkpoint(checkpoint, encoder=baseline.encoder, require_trained=not allow_untrained)
    vectors = project_embeddings(baseline.vectors, ckpt, allow_untrained=allow_untrained)
    projected = RecipeEmbeddings(baseline.recipe_ids, baseline.descriptions, vectors, baseline.encoder, space="shared")
    target = out / ("projected" if ckpt.trained else UNTRAINED_DIR)
    extra: dict[str, Any] = {
        "trained": ckpt.trained,
        "checkpoint": {
            "path": _display_path(Path(checkpoint).resolve()),
            "sha256": ckpt.sha256,
            "architecture": ckpt.head.architecture,
            "training": ckpt.training,
        },
        "baseline_embeddings_sha256": baseline_manifest["files"]["embeddings_sha256"],
    }
    if not ckpt.trained:
        extra["warning"] = ("UNTRAINED random projection exported for development only. "
                            "These vectors are not aligned with any other modality.")
    save_embedding_set(target, projected, kind=PROJECTED_KIND, extra=extra)
    return target
