"""Central paths + cache redirection.

IMPORT THIS MODULE FIRST (before torch / transformers / unreflectanything).
It points every model cache at D: so nothing large is written to C:.
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Roots
# ----------------------------------------------------------------------------
CODE_ROOT = Path(__file__).resolve().parents[1]                    # the project folder (code_project / the GitHub repo)


def _load_env_file(p: Path) -> None:
    """optional <project>/.env with KEY=VALUE lines (see .env.example); variables already set in Windows win"""
    if p.is_file():
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env_file(CODE_ROOT / ".env")
# default: the folder that holds the project (D:\NUS study\CG2 on the first PC); the online Space sets CG2_ROOT itself
CG2_ROOT = Path(os.environ.get("CG2_ROOT") or CODE_ROOT.parent)
CG2_DATA = Path(os.environ.get("CG2_DATA") or CG2_ROOT / "new image data")         # 7k photos + segmentation outputs
MODEL_CACHE = Path(os.environ.get("CG2_MODEL_CACHE") or CG2_ROOT / "model_cache")  # model weights (SETUP.bat)
THIRD_PARTY = CODE_ROOT / "third_party"

IMAGE_ROOT = CG2_ROOT / "image"
SOURCE_DIR = IMAGE_ROOT / "all_images" / "all_images"          # ORIGINAL - NEVER MODIFY
RECIPES_CSV = IMAGE_ROOT / "glazy_20260531.csv"
INGREDIENTS_CSV = IMAGE_ROOT / "glazy_20260531_ingredients.csv"

SEG_ROOT = IMAGE_ROOT / "segmentation"
SEG_DIRS = {
    "masks": SEG_ROOT / "masks",                       # binary PNG, 255 = glaze object
    "transparent": SEG_ROOT / "transparent",           # RGBA cut-out
    "white_background": SEG_ROOT / "white_background", # object on pure white
    "preview": SEG_ROOT / "preview",                   # 4-panel QA sheet
    "meta": SEG_ROOT / "meta",                         # per-image JSON (boxes, scores, flags)
}
EXP_ROOT = IMAGE_ROOT / "preprocessing_experiment"   # 8 variants for the 470-photo evaluation subset
CONVERT_ROOT = IMAGE_ROOT / "convert_image"           # 8 variants for ALL 9,965 photos (script 06)
RESULTS_ROOT = IMAGE_ROOT / "results"
CACHE_ROOT = IMAGE_ROOT / "_work_cache"                # depth maps etc. (safe to delete)

# ----------------------------------------------------------------------------
# Cache redirection  (C: has ~13 GB free -> everything goes to D:)
# ----------------------------------------------------------------------------
_CACHE_ENV = {
    "HF_HOME": MODEL_CACHE / "huggingface",
    "HF_HUB_CACHE": MODEL_CACHE / "huggingface" / "hub",
    "TORCH_HOME": MODEL_CACHE / "torch",
    "XDG_CACHE_HOME": MODEL_CACHE,
    "TEMP": MODEL_CACHE / "tmp",
    "TMP": MODEL_CACHE / "tmp",
}
for _k, _v in _CACHE_ENV.items():
    _v.mkdir(parents=True, exist_ok=True)
    os.environ[_k] = str(_v)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ----------------------------------------------------------------------------
# Model identifiers / checkpoint locations (all on D:)
# ----------------------------------------------------------------------------
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-base"
SAM2_ID = "facebook/sam2.1-hiera-large"

# Chosen by scripts/03a_choose_phasr_checkpoint.py on 24 pilot images (unsupervised metrics only):
#   Ambient6K: blurs glaze texture (local contrast x0.51) and paints colour blotches
#   WSRD:      most colour-faithful but barely corrects shading
#   ISTD+:     corrects shading almost as much as Ambient6K, keeps texture (x1.08), no blotches
PHASR_CKPT = MODEL_CACHE / "phasr" / "ISTDp_model_best.pth"
PHASR_WIN_SIZE = 16           # 8 for Ambient6K / INS checkpoints, 16 for ISTD / ISTD+ / WSRD
DEPTH_ANYTHING_CKPT = MODEL_CACHE / "depth_anything" / "depth_anything_v2_vitl.pth"
UNREFLECT_WEIGHTS_DIR = MODEL_CACHE / "unreflectanything"

# ----------------------------------------------------------------------------
# The 8 experimental variants:  bit order = (Retinex, PhaSR, UnReflect)
# Applied in the fixed order Retinex -> PhaSR -> UnReflectAnything.
# ----------------------------------------------------------------------------
VARIANTS = {
    "000_segmented_only":    (0, 0, 0),
    "100_retinex":           (1, 0, 0),
    "010_phasr":             (0, 1, 0),
    "001_unreflect":         (0, 0, 1),
    "110_retinex_phasr":     (1, 1, 0),
    "101_retinex_unreflect": (1, 0, 1),
    "011_phasr_unreflect":   (0, 1, 1),
    "111_all":               (1, 1, 1),
}

# Working resolution for the enhancement stage: the longest side of the
# object crop is resized to this so every variant is compared at the same scale
# and PhaSR / UnReflectAnything fit in 8 GB VRAM.
WORK_MAX_SIDE = 1024
CROP_PAD_FRAC = 0.06          # context kept around the object bbox before enhancement

DEVICE = "cuda"


def ensure_output_dirs() -> None:
    for d in SEG_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        (EXP_ROOT / name).mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
