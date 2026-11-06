"""Stage 2 - produce the 8 variants for a list of photos into any output root.

Shared by scripts/03_run_8_variants.py (root = image/preprocessing_experiment) and
scripts/06_convert_all_images.py (root = image/convert_image), so the experiment and the
full-dataset conversion run exactly the same code.

Three GPU phases, each loading only its own models (8 GB VRAM):
    A1  crop -> 000, Retinex -> 100, Depth-Anything-V2 depth (uint8 .npy cache)
    A2  PhaSR (+DINOv2):  010 = P(000),  110 = P(100)
    B   UnReflectAnything: 001 = U(000), 101 = U(100), 011 = U(010), 111 = U(110)
Every phase is resumable (existing outputs are skipped) and a failing photo is logged and
skipped instead of stopping a long run.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .. import config as C
from ..data import save_png, work_crop
from ..viz import variant_grid
from .retinex import msrcp

PHASE_A2 = {"000_segmented_only": "010_phasr", "100_retinex": "110_retinex_phasr"}
PHASE_B = {"000_segmented_only": "001_unreflect", "100_retinex": "101_retinex_unreflect",
           "010_phasr": "011_phasr_unreflect", "110_retinex_phasr": "111_all"}


class VariantStore:
    """<root>/<variant>/<id>.png (RGBA, alpha = fixed mask), <root>/_grids, <root>/_crop_meta."""

    def __init__(self, root: Path, depth_dir: Path):
        self.root, self.depth_dir = Path(root), Path(depth_dir)
        for v in C.VARIANTS:
            (self.root / v).mkdir(parents=True, exist_ok=True)
        for d in (self.root / "_grids", self.root / "_crop_meta", self.depth_dir):
            d.mkdir(parents=True, exist_ok=True)

    def path(self, variant: str, img_id: str) -> Path:
        return self.root / variant / f"{img_id}.png"

    def depth(self, img_id: str) -> Path:
        return self.depth_dir / f"{img_id}.npy"

    def grid(self, img_id: str) -> Path:
        return self.root / "_grids" / f"{img_id}.jpg"

    def load(self, variant: str, img_id: str):
        a = np.asarray(Image.open(self.path(variant, img_id)))
        return a[..., :3].copy(), a[..., 3] > 127

    def save(self, variant: str, img_id: str, rgb, mask):
        save_png(np.dstack([rgb, (mask * 255).astype(np.uint8)]), self.path(variant, img_id))

    def complete(self, img_id: str) -> bool:
        return all(self.path(v, img_id).exists() for v in C.VARIANTS)


def _fail(failures, phase, img_id):
    failures.append({"image_id": img_id, "phase": phase, "error": traceback.format_exc(limit=2)[-400:]})


def _needs_a1(store: VariantStore, img_id: str) -> bool:
    base_done = all(store.path(v, img_id).exists() for v in PHASE_A2)                 # 000 and 100
    phasr_done = all(store.path(v, img_id).exists() for v in PHASE_A2.values())       # 010 and 110
    return not base_done or (not phasr_done and not store.depth(img_id).exists())


def phase_a1(items, store: VariantStore, overwrite=False, failures=None):
    """crop -> 000, Retinex -> 100, depth (cached) for PhaSR."""
    failures = [] if failures is None else failures
    todo = [(i, p) for i, p in items if overwrite or _needs_a1(store, i)]
    rows = []
    if not todo:
        return rows
    from .phasr import DepthGeometry
    dg = DepthGeometry()
    for img_id, path in tqdm(todo, desc="phase A1: crop + Retinex + depth"):
        try:
            t = {}
            t0 = time.time(); x, m, box, scale = work_crop(img_id, path); t["crop_s"] = time.time() - t0
            store.save("000_segmented_only", img_id, x, m)
            t0 = time.time(); r = msrcp(x, m); t["retinex_s"] = time.time() - t0
            store.save("100_retinex", img_id, r, m)
            t0 = time.time(); np.save(store.depth(img_id), dg(x)); torch.cuda.synchronize(); t["depth_s"] = time.time() - t0
            (store.root / "_crop_meta" / f"{img_id}.json").write_text(json.dumps({"box_y0y1x0x1": box, "scale": scale}))
            rows.append({"image_id": img_id, **{k: round(v, 3) for k, v in t.items()}})
        except Exception:
            _fail(failures, "A1", img_id)
    del dg
    torch.cuda.empty_cache()
    return rows


def phase_a2(items, store: VariantStore, overwrite=False, failures=None):
    """PhaSR on 000 -> 010 and on 100 -> 110 (same depth: lighting does not change geometry)."""
    failures = [] if failures is None else failures
    todo = [(i, p) for i, p in items if store.depth(i).exists()
            and (overwrite or not all(store.path(d, i).exists() for d in PHASE_A2.values()))]
    rows = []
    if not todo:
        return rows
    from .phasr import PhaSRRunner
    ph = PhaSRRunner()
    for img_id, _ in tqdm(todo, desc="phase A2: PhaSR"):
        try:
            depth = np.load(store.depth(img_id))
            for src, dst in PHASE_A2.items():
                if store.path(dst, img_id).exists() and not overwrite:
                    continue
                rgb, m = store.load(src, img_id)
                t0 = time.time(); p = ph(rgb, depth); torch.cuda.synchronize()
                store.save(dst, img_id, p, m)
                rows.append({"image_id": img_id, "input": src, "phasr_s": round(time.time() - t0, 3)})
        except Exception:
            _fail(failures, "A2", img_id)
    del ph
    torch.cuda.empty_cache()
    return rows


def phase_b(items, store: VariantStore, overwrite=False, failures=None):
    """UnReflectAnything on the four phase-A outputs."""
    failures = [] if failures is None else failures
    todo = [(i, p) for i, p in items if all(store.path(s, i).exists() for s in PHASE_B)
            and (overwrite or not all(store.path(d, i).exists() for d in PHASE_B.values()))]
    rows = []
    if not todo:
        return rows
    from .unreflect import UnReflectRunner
    ur = UnReflectRunner()
    for img_id, _ in tqdm(todo, desc="phase B: UnReflectAnything"):
        try:
            for src, dst in PHASE_B.items():
                if store.path(dst, img_id).exists() and not overwrite:
                    continue
                rgb, m = store.load(src, img_id)
                t0 = time.time(); u, hm = ur(rgb); torch.cuda.synchronize()
                store.save(dst, img_id, u, m)
                rows.append({"image_id": img_id, "input": src, "unreflect_s": round(time.time() - t0, 3),
                             "highlight_frac_in_mask": round(float((hm[m] > 0.5).mean()) if m.any() else 0.0, 5)})
        except Exception:
            _fail(failures, "B", img_id)
    del ur
    torch.cuda.empty_cache()
    return rows


def make_grids(items, store: VariantStore, overwrite=False, failures=None):
    failures = [] if failures is None else failures
    for img_id, _ in tqdm(items, desc="8-variant grids"):
        if not store.complete(img_id) or (store.grid(img_id).exists() and not overwrite):
            continue
        try:
            imgs, mask = {}, None
            for v in C.VARIANTS:
                imgs[v], mask = store.load(v, img_id)
            variant_grid(imgs, mask, title=f"{img_id}").save(store.grid(img_id), quality=88)
        except Exception:
            _fail(failures, "grid", img_id)
