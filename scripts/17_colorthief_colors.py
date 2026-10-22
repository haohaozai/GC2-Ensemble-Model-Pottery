"""Three dominant colours of every object with ColorThief (same extractor as the team's colour report).

For each object the transparent cut-out (smaller_object_rgba_raw/<object>.png) is read, so ColorThief only
sees object pixels: it keeps pixels with alpha >= 125 and skips near-white pixels (r, g and b > 250).
  * palette: ColorThief MMCQ (colorthief 0.2.1), in ColorThief's own order (like the team's report, so the
    1st colour can cover less area than the 2nd)
  * percentage: share of the object's ColorThief pixels whose nearest palette colour (RGB distance) it is
  * name: nearest of the 87 reference shades in glazeprep/appearance.py (CIEDE2000), the names already used
    in full_auto_tags.csv; ColorThief itself does not name colours
Results go into tag_json/<object>.json as "colorthief_palette"; scripts/16_retag_attributes.py writes them
to full_auto_tags.csv (color1_* .. color3_*) and draws them on the detailed_preview cards.
Objects moved to does_not_needed/ by the review app are processed in place.

usage: python scripts/17_colorthief_colors.py [--output DIR] [--workers 8] [--force]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from glazeprep.config import CG2_DATA as BASE  # noqa: E402  (env CG2_DATA / CG2_ROOT; default: next to the project)

PARAMS = {"library": "colorthief 0.2.1", "color_count": 6, "quality": 5, "max_side_px": 600,
          "naming": "nearest reference shade, CIEDE2000 (glazeprep/appearance.py)"}
PARAMS_KEY = json.dumps(PARAMS, sort_keys=True)


def obj_file(out: Path, sub: str, name: str) -> Path:
    """Object files may have been moved to does_not_needed/<sub>/ by the review app."""
    p = out / sub / name
    if p.exists():
        return p
    q = out / "does_not_needed" / sub / name
    return q if q.exists() else p


def colorthief_palette(rgba: Image.Image):
    from colorthief import ColorThief
    from glazeprep.appearance import nearest_color
    im = rgba.convert("RGBA")
    im.thumbnail((PARAMS["max_side_px"], PARAMS["max_side_px"]), Image.LANCZOS)
    a = np.asarray(im)
    valid = (a[..., 3] >= 125) & ~((a[..., 0] > 250) & (a[..., 1] > 250) & (a[..., 2] > 250))
    px = a[..., :3][valid].astype(np.float32)
    note = ""
    if len(px) < 50:
        # ColorThief has nothing to quantise (object almost entirely near-white): report the mean colour
        opaque = a[..., :3][a[..., 3] >= 125].astype(np.float32)
        if len(opaque) == 0:
            return [], "no opaque object pixels"
        pal = [tuple(int(round(v)) for v in opaque.mean(0))]
        px = opaque
        note = "object almost entirely near-white (ColorThief skips pixels above 250); mean colour reported"
    else:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        try:
            pal = ColorThief(buf).get_palette(color_count=PARAMS["color_count"], quality=PARAMS["quality"])
        except Exception as exc:  # noqa: BLE001  (MMCQ fails on single-colour inputs)
            pal = [tuple(int(round(v)) for v in px.mean(0))]
            note = f"ColorThief failed ({type(exc).__name__}); mean colour reported"
    P = np.array(pal, np.float32)
    nearest = np.argmin(((px[:, None, :] - P[None, :, :]) ** 2).sum(-1), axis=1)
    share = np.bincount(nearest, minlength=len(P)) / max(1, len(px))
    out = []
    for i, rgb in enumerate(pal[:3]):
        m = nearest_color(rgb)
        out.append({"hex": "#%02X%02X%02X" % tuple(int(v) for v in rgb), "name": m["name"], "family": m["family"],
                    "fraction": round(float(share[i]), 4), "delta_e00": m["delta_e00"],
                    "name_uncertain": m["name_uncertain"]})
    return out, note


def work(task):
    out, oid, force = task
    out = Path(out)
    tp = obj_file(out, "tag_json", f"{oid}.json")
    try:
        t = json.loads(tp.read_text(encoding="utf-8"))
        if not force and t.get("colorthief_params") == PARAMS_KEY and t.get("colorthief_palette") is not None:
            return oid, "cached"
        pal, note = colorthief_palette(Image.open(obj_file(out, "smaller_object_rgba_raw", f"{oid}.png")))
        t["colorthief_palette"] = pal
        t["colorthief_note"] = note
        t["colorthief_params"] = PARAMS_KEY
        tmp = tp.with_name(tp.name + ".tmp")
        tmp.write_text(json.dumps(t, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, tp)
        return oid, "done" if not note else "note"
    except Exception as exc:  # noqa: BLE001
        return oid, f"error: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=BASE / "full_segmentation_claude_process")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    oids = []
    for mp in sorted((out / "meta").glob("*.json"), key=lambda p: (len(p.stem), p.stem)):
        rec = json.loads(mp.read_text(encoding="utf-8"))
        if rec.get("complete"):
            oids += rec.get("objects", [])
    print("=" * 66, flush=True)
    print(f"COLORTHIEF DOMINANT COLOURS | objects {len(oids)} | workers {args.workers}", flush=True)
    print("=" * 66, flush=True)
    t0, counts, errors = time.time(), {}, []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (oid, st) in enumerate(ex.map(work, [(str(out), o, args.force) for o in oids], chunksize=32), 1):
            key = st.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "error":
                errors.append(f"{oid}: {st}")
            if i % 1000 == 0 or i == len(oids):
                el = time.time() - t0
                print(f"[colorthief] {i}/{len(oids)} | {el / 60:.1f} min | ETA {(len(oids) - i) * el / i / 60:.1f} min | {counts}", flush=True)
    for e in errors[:20]:
        print("  " + e, flush=True)
    summary = {"objects": len(oids), "counts": counts, "errors": errors, "params": PARAMS,
               "minutes": round((time.time() - t0) / 60, 1)}
    p = out / "evaluation" / "retag" / "colorthief_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
