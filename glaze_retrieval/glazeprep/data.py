"""Image I/O + Glazy metadata + weak labels used for evaluation.

The image filename stem IS the Glazy recipe ID (e.g. 352742.jpg -> recipe 352742):
7,188 of the 9,965 images match a row of glazy_20260531.csv. That link lets us
derive *weak labels* from metadata, so the 8 preprocessing variants can be
compared without any manual labelling.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from . import config as C

Image.MAX_IMAGE_PIXELS = None

# ----------------------------------------------------------------------------
# Image I/O
# ----------------------------------------------------------------------------

def load_rgb(path: Path | str) -> np.ndarray:
    """uint8 HxWx3 RGB, EXIF orientation applied (phone photos are often rotated)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB")).copy()


def save_png(arr: np.ndarray, path: Path | str) -> None:
    """Atomic write (temp file + rename): a run killed mid-write never leaves a broken PNG
    that the resume logic would mistake for a finished image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(arr).save(tmp, format="PNG", optimize=False, compress_level=3)
    os.replace(tmp, path)


def list_source_images() -> list[Path]:
    return sorted(p for p in C.SOURCE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def work_crop(img_id: str, src_path: Path | str):
    """Object bbox (+ context padding) of the raw photo with its fixed stage-1 mask,
    resized so the longest side = C.WORK_MAX_SIDE (the shared working resolution).
    Returns (rgb uint8, mask bool, [y0,y1,x0,x1] in the original photo, scale)."""
    rgb = load_rgb(src_path)
    mask = np.asarray(Image.open(C.SEG_DIRS["masks"] / f"{img_id}.png")) > 127
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = int(C.CROP_PAD_FRAC * max(y1 - y0, x1 - x0))
    H, W = mask.shape
    y0, y1, x0, x1 = max(0, y0 - p), min(H, y1 + p), max(0, x0 - p), min(W, x1 + p)
    crop, cm = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    s = min(1.0, C.WORK_MAX_SIDE / max(crop.shape[:2]))
    if s < 1.0:
        size = (max(8, round(crop.shape[1] * s)), max(8, round(crop.shape[0] * s)))
        crop = np.asarray(Image.fromarray(crop).resize(size, Image.LANCZOS))
        cm = np.asarray(Image.fromarray(cm.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127
    return crop, cm, [int(y0), int(y1), int(x0), int(x1)], s


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
OXIDE_COLS = ["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "K2O", "Na2O", "Li2O", "ZnO", "SrO", "BaO",
              "P2O5", "Fe2O3", "TiO2", "CoO", "CuO", "Cr2O3", "MnO", "MnO2", "NiO", "ZrO2", "SnO2", "V2O5"]
COLORANT_COLS = ["CoO", "CuO", "Cr2O3", "Fe2O3", "MnO", "MnO2", "NiO", "TiO2", "ZrO2", "SnO2", "V2O5"]


def load_recipes() -> pd.DataFrame:
    df = pd.read_csv(C.RECIPES_CSV, low_memory=False)
    df["image_id"] = df["ID"].astype(str)
    for c in OXIDE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def colour_family(subtype: str) -> str | None:
    """Coarse visual colour family from Glazy's Subtype string (None = unknown)."""
    if not isinstance(subtype, str) or not subtype.startswith("Glaze"):
        return None
    s = subtype.lower()
    if "celadon" in s:
        return "celadon"
    if "turquoise" in s:
        return "turquoise"
    if "blue" in s:
        return "blue"
    if "green" in s:
        return "green"
    if "purple" in s:
        return "purple"
    if "kaki" in s or "tomato" in s:
        return "brown_iron"
    if "red" in s:
        return "red_pink"
    if "yellow" in s:
        return "yellow"
    if "black" in s:
        return "black"
    if "grey" in s or "gray" in s or "charcoal" in s:
        return "grey"
    if "white" in s:
        return "white"
    if "clear" in s:
        return "clear"
    if "iron" in s:          # tenmoku, tea dust, oil spot, amber, hare's fur ...
        return "brown_iron"
    return None


def colorant_class(row: pd.Series) -> str:
    """Dominant colouring oxide from chemistry (wt%), strongest colourant first."""
    if row["CoO"] >= 0.10:
        return "cobalt"
    if row["Cr2O3"] >= 0.20:
        return "chrome"
    if row["CuO"] >= 0.50:
        return "copper"
    if row["NiO"] >= 0.50:
        return "nickel"
    if row["MnO"] + row["MnO2"] >= 1.0:
        return "manganese"
    if row["Fe2O3"] >= 4.0:
        return "iron_high"
    if row["Fe2O3"] >= 1.0:
        return "iron_mid"
    if row["TiO2"] >= 3.0:
        return "titanium"
    if row["ZrO2"] >= 4.0 or row["SnO2"] >= 3.0:
        return "opacified"
    return "none"


def surface_coarse(s) -> str | None:
    if not isinstance(s, str):
        return None
    return "glossy" if s in {"Glossy", "Semi-glossy"} else "matte"


def transparency_coarse(s) -> str | None:
    if not isinstance(s, str):
        return None
    return "opaque" if s in {"Opaque", "Semi-opaque"} else "see_through"


def chem_vector(df: pd.DataFrame) -> np.ndarray:
    """log1p colourant/opacifier oxides -> the chemistry that drives glaze colour."""
    x = df[COLORANT_COLS].to_numpy(dtype=np.float32).copy()
    x[:, COLORANT_COLS.index("MnO")] += x[:, COLORANT_COLS.index("MnO2")]
    x = np.delete(x, COLORANT_COLS.index("MnO2"), axis=1)
    return np.log1p(x)


def build_labelled_index() -> pd.DataFrame:
    """One row per image that has a recipe, with all weak labels attached."""
    stems = {p.stem: p for p in list_source_images()}
    df = load_recipes()
    df = df[df["image_id"].isin(stems)].copy()
    df["path"] = df["image_id"].map(lambda s: str(stems[s]))
    df["colour_family"] = df["Subtype"].map(colour_family)
    df["colorant"] = df.apply(colorant_class, axis=1)
    df["surface"] = df["Surface"].map(surface_coarse)
    df["transparency"] = df["Transparency"].map(transparency_coarse)
    return df.reset_index(drop=True)
