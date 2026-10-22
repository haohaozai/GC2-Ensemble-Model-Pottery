"""Rebuild the cut-outs of the Tile review panel (smaller_object_rgba_raw/<object>.png) after the data download.

They are not in the download (12 GB): each one is the original photo cropped to the object's box
(meta/<photo>.json -> segmentation.instances[index].box_xyxy = x0, y0, x1, y1 in photo pixels, end excluded, after the
photo's EXIF rotation) with the object's raw mask (smaller_object_masks_raw/<object>.png) as transparency - pixel for
pixel the files of the first PC (all objects checked with --verify on 2026-10-22; objects that did not match are in the
download as they are).
Image sets: full_segmentation_claude_process (photos 7k_matched_images/7k_matched_images) and bonus_image_claude_process
(photos others_image), the kept objects and does_not_needed/. Existing cut-outs are left alone.

usage: python scripts/rebuild_tile_cutouts.py [--verify] [--workers N] [--list-mismatches FILE]
  --verify   compare with the existing cut-outs instead of writing them (used before packaging the data)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glazeprep.config import CG2_DATA  # noqa: E402  (env CG2_DATA / CG2_ROOT or .env; default: next to the project)

SETS = (("full_segmentation_claude_process", "7k_matched_images/7k_matched_images"),
        ("bonus_image_claude_process", "others_image"))
NOT_NEEDED = "does_not_needed"


def tasks():
    """one task per photo: (photo file, meta file, [(object index, mask file, cut-out file)])"""
    by_photo = {}
    for out_name, photos_name in SETS:
        out, photos = CG2_DATA / out_name, CG2_DATA / photos_name
        for base in (out, out / NOT_NEEDED):
            mdir = base / "smaller_object_masks_raw"
            if not mdir.is_dir():
                continue
            for m in mdir.glob("*.png"):
                pid, _, k = m.stem.rpartition("_")
                key = (out_name, pid)
                if key not in by_photo:
                    by_photo[key] = [out / "meta" / f"{pid}.json", photos, []]
                by_photo[key][2].append((int(k), str(m), str(base / "smaller_object_rgba_raw" / f"{m.stem}.png")))
    return [(str(meta), str(photos), objs) for meta, photos, objs in by_photo.values()]


def cutout(img: np.ndarray, box, mask_file: str) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    mask = np.array(Image.open(mask_file))
    crop = img[y0:y1, x0:x1]
    if crop.shape[:2] != mask.shape[:2]:
        raise ValueError(f"box {box} does not fit the mask {mask.shape[::-1]}")
    return np.dstack([crop, mask])


def run(task, verify: bool):
    meta_file, photos, objs = task
    res = {"written": 0, "kept": 0, "same": 0, "different": [], "failed": []}
    todo = objs if verify else [o for o in objs if not Path(o[2]).exists()]
    res["kept"] = len(objs) - len(todo)
    if not todo:
        return res
    try:
        meta = json.loads(Path(meta_file).read_text(encoding="utf-8"))
        boxes = {int(i["index"]): i["box_xyxy"] for i in meta["segmentation"]["instances"]}
        with Image.open(Path(photos) / meta["parent_filename"]) as im:
            img = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        res["failed"] += [f"{Path(o[1]).stem}: {type(exc).__name__}: {exc}" for o in todo]
        return res
    for k, mask_file, out_file in todo:
        oid = Path(mask_file).stem
        try:
            rgba = cutout(img, boxes[k], mask_file)
            if verify:
                if not Path(out_file).exists():
                    continue
                ref = np.array(Image.open(out_file).convert("RGBA"))
                if ref.shape == rgba.shape and np.array_equal(ref, rgba):
                    res["same"] += 1
                else:
                    res["different"].append(oid)
            else:
                Path(out_file).parent.mkdir(parents=True, exist_ok=True)
                tmp = out_file + ".tmp.png"
                Image.fromarray(rgba, "RGBA").save(tmp)
                os.replace(tmp, out_file)
                res["written"] += 1
        except Exception as exc:  # noqa: BLE001
            res["failed"].append(f"{oid}: {type(exc).__name__}: {exc}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--list-mismatches", type=Path, default=None)
    args = ap.parse_args()
    t0 = time.time()
    tl = tasks()
    n = sum(len(t[2]) for t in tl)
    print(f"{'Checking' if args.verify else 'Rebuilding'} the Tile cut-outs of {n:,} objects ({len(tl):,} photos) in {CG2_DATA} ...", flush=True)
    tot = {"written": 0, "kept": 0, "same": 0, "different": [], "failed": []}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(run, tl, [args.verify] * len(tl), chunksize=8), 1):
            for k in ("written", "kept", "same"):
                tot[k] += r[k]
            tot["different"] += r["different"]
            tot["failed"] += r["failed"]
            if i % 500 == 0:
                print(f"  {i:,}/{len(tl):,} photos ...", flush=True)
    if args.verify:
        print(f"identical: {tot['same']:,} | different: {len(tot['different'])} | failed: {len(tot['failed'])}", flush=True)
    else:
        print(f"written: {tot['written']:,} | already there: {tot['kept']:,} | failed: {len(tot['failed'])}", flush=True)
    for f in tot["failed"][:20]:
        print("  FAILED", f, flush=True)
    if args.list_mismatches:
        args.list_mismatches.write_text(json.dumps({"different": tot["different"], "failed": tot["failed"]}, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 1 if tot["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
