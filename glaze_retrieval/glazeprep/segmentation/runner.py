"""Segment one photo and write all Stage-1 outputs (shared by scripts 02 and 06)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config as C
from ..data import load_rgb, save_png
from ..viz import composite, seg_preview


def bbox(mask, pad_frac=0.03):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = int(pad_frac * max(y1 - y0, x1 - x0))
    H, W = mask.shape
    return max(0, y0 - p), min(H, y1 + p), max(0, x0 - p), min(W, x1 + p)


def meta_path(img_id: str) -> Path:
    return C.SEG_DIRS["meta"] / f"{img_id}.json"


def segment_and_save(seg, img_id: str, path: Path, overwrite: bool = False) -> dict:
    """Run GroundedSAM on one photo; write mask / transparent / white_background / preview / meta.
    Returns the meta dict (read from disk if this photo was already segmented)."""
    mp = meta_path(img_id)
    if mp.exists() and not overwrite:
        return json.loads(mp.read_text())
    t0 = time.time()
    rgb = load_rgb(path)
    res = seg(rgb)
    meta = {"image_id": img_id, "height": rgb.shape[0], "width": rgb.shape[1], **res.meta(),
            "seconds": round(time.time() - t0, 2)}
    mask = res.mask
    save_png((mask * 255).astype(np.uint8), C.SEG_DIRS["masks"] / f"{img_id}.png")
    if mask.any():
        y0, y1, x0, x1 = bbox(mask)
        crop, cm = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        save_png(np.dstack([crop, (cm * 255).astype(np.uint8)]), C.SEG_DIRS["transparent"] / f"{img_id}.png")
        Image.fromarray(composite(crop, cm, "white")).save(C.SEG_DIRS["white_background"] / f"{img_id}.jpg", quality=95)
    title = f"{img_id}  objects={len(res.boxes)}  area={res.area_frac:.2f}  flags={','.join(res.flags) or 'ok'}"
    seg_preview(rgb, mask, res.boxes, title).save(C.SEG_DIRS["preview"] / f"{img_id}.jpg", quality=88)
    mp.write_text(json.dumps(meta, indent=1))
    return meta
