"""Contact sheets for Claude's visual attribute check (attributes_v2).

A stratified sample of VALIDATION and TEST objects (recipe group split) is drawn so that rare forms, every surface
group and objects with named effects are represented. Each sheet shows 8 object cut-outs (no recipe text, no
metadata: the check is made from the image only, so it can be compared with the metadata later). Claude labels the
six attributes per object following Ceramic_Real_Photo_Comparison_Guide_V4; '?' = cannot be judged from the photo.
The labels are stored with provenance claude_visual (NOT human-verified).
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import obj_file, read_csv, write_csv
from .dino_attr import cutout

PANEL, COLS, ROWS = 400, 4, 2
FORCE = ["2152_1", "206_1"]


def sample(labels_csv: Path, preds: dict, n_per_split=240, seed=7) -> list[dict]:
    rows = read_csv(labels_csv)
    rng = random.Random(seed)
    out = []
    for split in ("val", "test"):
        pool = [r for r in rows if r["split"] == split]
        def stratum(r):
            f = (preds.get(r["object_id"], {}).get("form_final") or {}).get("value") or "?"
            return (f if f != "test tile" else "tile", r["surface_group_meta"] or "-", r["effects_text"] != "[]")
        cnt = Counter(stratum(r) for r in pool)
        # weight ~ 1 / sqrt(stratum size): rare strata over-represented, common ones still present
        w = [1.0 / cnt[stratum(r)] ** 0.5 for r in pool]
        chosen, seen = [], set()
        while len(chosen) < n_per_split:
            r = rng.choices(pool, weights=w, k=1)[0]
            if r["object_id"] not in seen:
                seen.add(r["object_id"])
                chosen.append({"object_id": r["object_id"], "split": split, "stratum": "|".join(map(str, stratum(r)))})
        out += chosen
    have = {o["object_id"] for o in out}
    out += [{"object_id": o, "split": "extra", "stratum": "example"} for o in FORCE if o not in have]
    return out


def sample_split(labels_csv: Path, preds: dict, split: str, n: int, seed: int, exclude: set[str]) -> list[dict]:
    """attributes_v3: n more objects of one split (TRAIN for the enlarged Claude training set), drawn with the same
    stratified weighting as sample() (rare form / surface group / named-effect strata over-represented), never an object
    in `exclude` (the 481 already labelled). The val / test objects of sample() stay untouched."""
    rows = [r for r in read_csv(labels_csv) if r["split"] == split and r["object_id"] not in exclude]
    rng = random.Random(seed)

    def stratum(r):
        f = (preds.get(r["object_id"], {}).get("form_final") or {}).get("value") or "?"
        return (f if f != "test tile" else "tile", r["surface_group_meta"] or "-", r["effects_text"] != "[]")
    cnt = Counter(stratum(r) for r in rows)
    w = [1.0 / cnt[stratum(r)] ** 0.5 for r in rows]
    chosen, seen = [], set()
    while len(chosen) < min(n, len(rows)):
        r = rng.choices(rows, weights=w, k=1)[0]
        if r["object_id"] not in seen:
            seen.add(r["object_id"])
            chosen.append({"object_id": r["object_id"], "split": split, "stratum": "|".join(map(str, stratum(r)))})
    return chosen


def _font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(objs: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    manifest = []
    font = _font(22)
    for s in range(0, len(objs), per):
        sheet_no = s // per + 1
        canvas = Image.new("RGB", (COLS * PANEL, ROWS * (PANEL + 34)), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for k, o in enumerate(objs[s:s + per]):
            a = np.asarray(Image.open(obj_file("smaller_object_rgba_raw", f"{o['object_id']}.png")).convert("RGBA"))
            im = Image.fromarray(cutout(a, pad=0.03)[0])
            im.thumbnail((PANEL - 8, PANEL - 8), Image.LANCZOS)
            x, y = (k % COLS) * PANEL, (k // COLS) * (PANEL + 34)
            canvas.paste(im, (x + (PANEL - im.width) // 2, y + 34 + (PANEL - im.height) // 2))
            d.text((x + 8, y + 5), f"{k + 1}. {o['object_id']}", fill=(0, 0, 0), font=font)
            d.rectangle([x, y, x + PANEL - 1, y + PANEL + 33], outline=(170, 170, 170))
            manifest.append({**o, "sheet": sheet_no, "pos": k + 1})
        canvas.save(out_dir / f"sheet_{sheet_no:03d}.jpg", quality=90)
    write_csv(out_dir / "manifest.csv", manifest)
    return manifest
