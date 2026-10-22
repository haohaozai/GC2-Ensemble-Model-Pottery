"""Form check set: a stratified sample of objects whose FORM is labelled from the object images.

The labels are Claude-assisted (labelled by Claude from numbered contact sheets of the object cut-outs), recorded
with provenance "claude_visual" and NOT treated as human-verified ground truth; they can be confirmed or corrected
in the review app (review_tile panel, "Object form"). Sample: 120 random objects + up to 120 objects that the old
tagger called anything other than "test tile" (forms other than tiles are rare and would otherwise be missing).
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from .common import SEG, obj_file, read_csv, read_json

SEED = 20261018


def sample(objs: list[dict], n_random=120, n_other=120) -> list[str]:
    rng = np.random.default_rng(SEED)
    ids = [o["object_id"] for o in objs]
    rand = sorted(rng.choice(ids, n_random, replace=False).tolist())
    other = []
    for oid in ids:
        if oid in rand:
            continue
        tj = read_json(SEG / "tag_json" / f"{oid}.json") or {}
        f = str(tj.get("object_form_predicted", "")).lower()
        if f and f not in ("test tile", "unknown", "tile"):
            other.append(oid)
    other = sorted(rng.choice(other, min(n_other, len(other)), replace=False).tolist())
    return rand + other


def sheets(ids: list[str], out_dir, per=30, cols=6, cell=210):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(ids), per):
        chunk = ids[s:s + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for k, oid in enumerate(chunk):
            im = Image.open(obj_file("smaller_object", f"{oid}.jpg")).convert("RGB")
            im.thumbnail((cell - 8, cell - 8), Image.LANCZOS)
            x, y = (k % cols) * cell, (k // cols) * (cell + 18)
            sheet.paste(im, (x + (cell - im.width) // 2, y + 18 + (cell - 8 - im.height) // 2))
            d.text((x + 4, y + 2), f"{s + k + 1}. {oid}", fill=(0, 0, 0))
        p = out_dir / f"form_sheet_{s // per + 1:02d}.jpg"
        sheet.save(p, quality=88)
        paths.append(p)
    return paths
