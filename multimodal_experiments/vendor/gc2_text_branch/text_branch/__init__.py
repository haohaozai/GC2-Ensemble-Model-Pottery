"""Recipe tags -> text encoder -> text projection head -> shared-space vector.

Text branch of the multi-modal ceramic glaze retrieval engine. See
docs/text_branch.md for the handoff notes.
"""

from .cleaning import TagNormaliser, normalise_recipe_tags
from .config import PREPROCESSING_VERSION, TextBranchConfig, load_config
from .cones import ConeRange, ConeScale
from .description import build_recipe_description
from .encoder import EncoderInfo, RecipeEmbeddings, TextEncoder, encode_query, encode_recipes
from .projection import (
    IncompatibleCheckpointError,
    ProjectionCheckpoint,
    TextEmbeddingStore,
    TextProjectionHead,
    UntrainedProjectionError,
    build_projection_head,
    load_projection_checkpoint,
    project_embeddings,
    save_projection_checkpoint,
)
from .records import CleanRecipe, RecipeTags
from .search import RecipeSearchIndex, SearchFilters, SearchResult, search_recipes

__all__ = [
    "PREPROCESSING_VERSION",
    "CleanRecipe",
    "ConeRange",
    "ConeScale",
    "EncoderInfo",
    "IncompatibleCheckpointError",
    "ProjectionCheckpoint",
    "RecipeEmbeddings",
    "RecipeSearchIndex",
    "RecipeTags",
    "SearchFilters",
    "SearchResult",
    "TagNormaliser",
    "TextBranchConfig",
    "TextEmbeddingStore",
    "TextEncoder",
    "TextProjectionHead",
    "UntrainedProjectionError",
    "build_projection_head",
    "build_recipe_description",
    "encode_query",
    "encode_recipes",
    "load_config",
    "load_projection_checkpoint",
    "normalise_recipe_tags",
    "project_embeddings",
    "save_projection_checkpoint",
    "search_recipes",
]
