"""Configuration for the recipe-tag text branch.

Python defaults are the source of truth; ``configs/text_branch.yaml`` mirrors
them and may override any field. Relative paths resolve against the repo root.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Bump whenever cleaning or description logic changes in a way that alters
# descriptions. Stored in every output manifest and projection checkpoint.
PREPROCESSING_VERSION = "recipe-tags-v1"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "text_branch.yaml"

# Orton pyrometric cones, coolest to hottest, covering every label in the
# Glazy export (022 .. 14). Labels are strings: "06" (~1000 C) and "6"
# (~1220 C) are different cones. "05 1/2" sits between 06 and 05; "5 1/2"
# between 5 and 6. Anything outside this list is flagged, not guessed.
ORTON_CONE_ORDER = [
    "022", "021", "020", "019", "018", "017", "016", "015", "014", "013",
    "012", "011", "010", "09", "08", "07", "06", "05 1/2", "05", "04", "03",
    "02", "01", "1", "2", "3", "4", "5", "5 1/2", "6", "7", "8", "9", "10",
    "11", "12", "13", "14",
]

# Logical field -> column header in the Glazy CSV export.
DEFAULT_COLUMNS = {
    "recipe_id": "ID",
    "name": "Name",
    "record_type": "Type",
    "state": "State",
    "subtype": "Subtype",
    "cone": "Cone",
    "surface": "Surface",
    "transparency": "Transparency",
    "atmospheres": "Atmospheres",
    "country": "Country",
    "other_names": "Other Names",
}

# Controlled vocabularies (lower-case canonical labels, in Glazy's order).
DEFAULT_VOCABULARIES = {
    "surface": [
        "glossy", "semi-glossy", "satin", "satin-matte", "semi-matte",
        "smooth matte", "matte", "dry matte", "stony matte",
    ],
    "transparency": ["transparent", "translucent", "semi-opaque", "opaque"],
    "atmosphere": [
        "oxidation", "neutral", "reduction", "salt & soda", "wood", "raku", "luster",
    ],
    # Top level of the subtype hierarchy. Unknown roots are kept but flagged.
    "subtype_root": [
        "glaze", "clay body", "slip & engobe", "underglaze", "overglaze",
        "clay", "feldspar", "flux", "frit", "primitive", "colorant", "silica",
        "alumina", "ash", "miscellaneous", "opacifier",
    ],
}

# Verified spelling variants / synonyms only: hyphenation, British spelling
# and adjective forms of the exact Glazy labels. Keys are compared after
# whitespace collapsing and lower-casing. Applied to source data and to
# user-supplied search filters alike.
DEFAULT_SYNONYMS = {
    "surface": {
        "gloss": "glossy",
        "semi gloss": "semi-glossy",
        "semi-gloss": "semi-glossy",
        "semi glossy": "semi-glossy",
        "satin matte": "satin-matte",
        "semi matte": "semi-matte",
    },
    "transparency": {
        "semi opaque": "semi-opaque",
    },
    "atmosphere": {
        "oxidising": "oxidation",
        "oxidizing": "oxidation",
        "reducing": "reduction",
        "salt and soda": "salt & soda",
        "salt/soda": "salt & soda",
        "lustre": "luster",
    },
    "subtype": {},
    # Cone label aliases beyond the built-in handling of "cone 6", "^6",
    # "5½" and "5.5" (see cones.py).
    "cone": {},
}

# Flag (never fix) tag combinations that contradict the recipe's own subtype.
# A rule fires when ``subtype_keyword`` appears as a word in any subtype level
# and ``field`` holds one of ``values``.
DEFAULT_CONTRADICTION_RULES = [
    {"id": "clear_subtype_vs_opaque", "subtype_keyword": "clear",
     "field": "transparency", "values": ["opaque"]},
    {"id": "matte_subtype_vs_glossy_surface", "subtype_keyword": "matte",
     "field": "surface", "values": ["glossy", "semi-glossy"]},
    {"id": "glossy_subtype_vs_matte_surface", "subtype_keyword": "glossy",
     "field": "surface",
     "values": ["satin-matte", "semi-matte", "smooth matte", "matte", "dry matte", "stony matte"]},
]


@dataclass
class PathsConfig:
    source_csv: str = "data/glazy_20260531.csv"
    output_dir: str = "outputs/text_branch"

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class CleaningConfig:
    columns: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COLUMNS))
    in_scope_types: list[str] = field(default_factory=lambda: ["Recipe"])
    subtype_separator: str = r"\s+-\s+"  # regex; keeps "Single-Fire" intact
    atmosphere_separator: str = ";"
    other_names_separator: str = r"[;,]"  # regex, for alias lookup only
    cone_order: list[str] = field(default_factory=lambda: list(ORTON_CONE_ORDER))
    vocabularies: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_VOCABULARIES.items()})
    synonyms: dict[str, dict[str, str]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_SYNONYMS.items()})
    contradiction_rules: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(r) for r in DEFAULT_CONTRADICTION_RULES])
    # A recipe needs at least this many usable semantic fields to be embedded.
    min_semantic_fields: int = 1
    # Metadata appended to descriptions. Empty by default; only "country" is
    # supported. Never counts towards min_semantic_fields.
    description_metadata_fields: list[str] = field(default_factory=list)


@dataclass
class EncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Pinned Hugging Face commit for reproducibility. Set to null when
    # changing model_name, or pin the new model's commit.
    revision: str | None = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    device: str | None = None  # None = cuda if available, else mps, else cpu
    batch_size: int = 64
    # None = use the prompts the model itself defines for encode_query /
    # encode_document (empty for all-MiniLM-L6-v2). Set explicitly for models
    # whose card documents a prefix but does not configure it.
    query_prompt: str | None = None
    document_prompt: str | None = None
    seed: int = 0
    use_cache: bool = True


@dataclass
class ProjectionConfig:
    output_dim: int = 128  # team shared-space dimension (initial default)
    hidden_dim: int | None = None  # None = single linear layer; int = 2-layer MLP
    dropout: float = 0.0
    seed: int = 0


@dataclass
class TextBranchConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    @property
    def output_dir(self) -> Path:
        return self.paths.resolve(self.paths.output_dir)


def _apply_overrides(target: Any, overrides: dict[str, Any], prefix: str = "") -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown config key: {prefix}{key}")
        current = getattr(target, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value, f"{prefix}{key}.")
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None = None) -> TextBranchConfig:
    """Return defaults overridden by ``path`` (or the repo YAML if it exists)."""
    config = TextBranchConfig()
    if path is None and DEFAULT_CONFIG_PATH.exists():
        path = DEFAULT_CONFIG_PATH
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _apply_overrides(config, data)
    validate_config(config)
    return config


def validate_config(config: TextBranchConfig) -> None:
    cleaning = config.cleaning
    missing = set(DEFAULT_COLUMNS) - set(cleaning.columns)
    if missing:
        raise ValueError(f"cleaning.columns is missing logical fields: {sorted(missing)}")
    for field_name in ("surface", "transparency", "atmosphere", "subtype_root"):
        if field_name not in cleaning.vocabularies:
            raise ValueError(f"cleaning.vocabularies is missing {field_name!r}")
    unsupported = set(cleaning.description_metadata_fields) - {"country"}
    if unsupported:
        raise ValueError(f"Unsupported description_metadata_fields: {sorted(unsupported)}")
    if cleaning.min_semantic_fields < 1:
        raise ValueError("cleaning.min_semantic_fields must be >= 1")
    if config.projection.output_dim < 1:
        raise ValueError("projection.output_dim must be >= 1")


def config_digest(section: Any) -> str:
    """Stable sha256 of a config dataclass, used in manifests."""
    payload = json.dumps(dataclasses.asdict(section), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
