"""Local review app: browse every photo with its object crops and ColorThief colours, mark objects as
"Not needed" or "Needs review", then Save & download the review CSV.

On Save (one click in the page):
  1. review/object_review.csv is rewritten (one row per object: recipe, 3 ColorThief colours, review_status,
     review_note, reviewed_at, object_location) - this file is the master record of the decisions;
  2. every object marked "not_needed" is MOVED (cut, not copied or deleted) with its 4 companion files
     smaller_object/, smaller_object_masks_raw/, smaller_object_rgba_raw/, detailed_preview/, tag_json/
     -> does_not_needed/<same subfolder>/ ; an object whose mark is removed is moved back;
  3. full_auto_tags.csv gets review_status / review_note / reviewed_at / object_location;
  4. every move is appended to review/move_log.csv;
  5. the page downloads object_review.csv.
Objects in does_not_needed/ are hidden the next time the page opens (filter "Not needed" shows them, to restore).

Overall view: the uncropped photo carries the outline + number of every object of the photo (from
instance_masks/, cached in review_thumbs/outlines/), and the Overview window shows the whole photo with all its
objects, the pipeline's 4-panel preview and, for manually corrected photos, the preview from before the correction.

Bonus images: the photos of others_image (recipes of the 7k CSV whose photo was not in 7k_matched_images),
segmented by the same pipeline into bonus_image_claude_process, are listed after the 7k photos with a "Bonus"
badge and their own filter. Their decisions, moves and CSV updates stay in bonus_image_claude_process; the
downloaded CSV holds both sets with an image_set column (7k_matched / bonus_image).

Runs only on this computer (127.0.0.1). Start it with OPEN_REVIEW_APP.bat or:
  .venv\\Scripts\\python.exe scripts\\18_review_app.py [--output DIR] [--bonus DIR | --no-bonus] [--port 8765] [--no-browser]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import os
import re
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from glazeprep.config import CG2_DATA as BASE  # noqa: E402  (env CG2_DATA / CG2_ROOT or .env; default: next to the project)

APP_HTML = CODE / "review_app" / "index.html"
OBJ_DIRS = ("smaller_object", "smaller_object_masks_raw", "smaller_object_rgba_raw", "detailed_preview", "tag_json")
OBJ_EXT = {"smaller_object": ".jpg", "smaller_object_masks_raw": ".png", "smaller_object_rgba_raw": ".png",
           "detailed_preview": ".jpg", "tag_json": ".json"}
NOT_NEEDED = "does_not_needed"
# completed = corrected in round 1 (19_manual_corrections.py), completed_v2 = corrected in round 2 (--round 2)
STATUSES = {"", "not_needed", "needs_review", "completed", "completed_v2"}
REVIEW_COLUMNS = ["object_id", "object_filename", "parent_filename", "object_index", "object_count", "recipe_id",
                  "recipe_name", "subtype_csv", "surface", "transparency", "glaze_effect", "defect",
                  "color1_name", "color1_hex", "color1_pct", "color2_name", "color2_hex", "color2_pct",
                  "color3_name", "color3_hex", "color3_pct", "review_status", "review_note", "reviewed_at",
                  "object_location", "tag_review_required", "segmentation_status", "correction", "correction_v2",
                  "previous_object_id"]
HISTORY = ("review_status", "review_note", "reviewed_at", "correction", "correction_v2", "previous_object_id")
THUMB = {"orig": 420, "obj": 420, "big": 1280}
ID_RE = re.compile(r"^\d+(_\d+)?$")
PHOTO_RE = re.compile(r"^\d+$")
OUTLINE_SIDE = 640          # outlines are traced on the instance mask scaled to this longest side
BACKUP = "manual_corrections_backup"
BACKUP_V2 = "manual_corrections_backup_v2"
POOL = ThreadPoolExecutor(4)
# review_tile (category, not a status): sidecar review/tile_review.json via glazeprep.review_tile; machine labels of
# the six attributes (Grounding DINO v2, v1 heads) are read from the versioned attribute folders, never written
ATTR_DIR = CODE / "multimodal_experiments" / "work" / "attributes_v1"
ATTR_V2 = CODE / "multimodal_experiments" / "work" / "attributes_v2"     # Grounding DINO attribute labels (validated by Claude)
ATTR_V3 = CODE / "multimodal_experiments" / "work" / "attributes_v3"     # v3: trained on Claude's labels, top-3 candidates


def attr_labels_path() -> Path:
    p3 = ATTR_V3 / "attribute_predictions_v3.json"
    return p3 if p3.exists() else ATTR_V2 / "attribute_predictions_v2.json"
OID_RE = re.compile(r"^\d+_\d+$")
# image sets: the 7k matched photos plus the bonus_image photos (others_image, segmented by the same claude_v3.1
# pipeline into their own output folder). Each set keeps its own files, review CSV and move log; bonus photos are
# listed after the 7k photos so the 7k page numbers do not move.
MAIN_SET, BONUS_SET = "7k_matched", "bonus_image"
IMG_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


class ImageSet:
    """One segmentation output folder and the folder of its original photos."""

    def __init__(self, name: str, out: Path, source: Path):
        self.name, self.out, self.source = name, out, source
        self.review_csv = out / "review" / "object_review.csv"
        self.move_log = out / "review" / "move_log.csv"
        self.thumbs = out / "review_thumbs"


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def atomic_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    os.replace(tmp, path)


class Store:
    """In-memory view of the output folder + decisions; all writes go through save()."""

    def __init__(self, out: Path, source: Path, extra_sets=()):
        """out / source = the 7k matched set; extra_sets = [(name, out, source)] listed after it (bonus_image)."""
        self.sets = [ImageSet(MAIN_SET, out, source)] + [ImageSet(n, o, s) for n, o, s in extra_sets]
        self.out, self.source = out, source                  # the main set (tile reviews, defaults)
        self.lock = threading.Lock()
        self.review_csv = self.sets[0].review_csv
        self.move_log = self.sets[0].move_log
        self.thumbs = self.sets[0].thumbs
        self.load()

    # ------------------------------------------------------------------ loading
    def set_of(self, ident) -> ImageSet:
        """Image set of a photo id or object id (the main set when unknown)."""
        ps = getattr(self, "photo_set", None)
        s = ps.get(str(ident).split("_")[0]) if ps else None
        return s or ImageSet(MAIN_SET, self.out, getattr(self, "source", self.out))

    def photo_file(self, ident) -> Path:
        """Original photo file (the bonus photos are .jpg or .png)."""
        s = self.set_of(ident)
        ph = getattr(self, "photo_by_id", {}).get(ident)
        return s.source / (ph["file"] if ph else f"{ident}.jpg")

    def location(self, oid):
        out = self.set_of(oid).out
        if (out / "smaller_object" / f"{oid}.jpg").exists():
            return "smaller_object"
        if (out / NOT_NEEDED / "smaller_object" / f"{oid}.jpg").exists():
            return NOT_NEEDED
        return "missing"

    def load(self):
        t0 = time.time()
        self.objects, self.photo_set, plist, skipped = {}, {}, [], 0
        for s in self.sets:
            with (s.out / "full_auto_tags.csv").open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
            if s is self.sets[0]:
                self.csv_columns = list(rows[0].keys()) if rows else []
            photos = {}
            for r in rows:
                pf = r.get("parent_filename") or ""
                stem = Path(pf).stem
                if self.photo_set.get(stem, s) is not s:    # same photo id in an earlier set: keep that one
                    skipped += 1
                    continue
                self.photo_set[stem] = s
                ph = photos.setdefault(stem, {"id": stem, "file": pf, "recipe": r.get("recipe_name", ""),
                                              "subtype": r.get("subtype_csv", ""), "objects": []})
                if s.name != MAIN_SET:
                    ph["set"] = s.name
                oid = r.get("object_id")
                if not oid:
                    continue
                o = {k: r.get(k, "") for k in REVIEW_COLUMNS}
                if not o.get("color1_hex"):        # ColorThief columns missing: fall back to the k-means palette
                    o.update({"color1_name": r.get("primary_color_name", ""), "color1_hex": r.get("primary_hex", ""),
                              "color1_pct": "", "color2_name": r.get("secondary_color_name", ""),
                              "color2_hex": r.get("secondary_hex", ""), "color3_name": r.get("third_color_name", ""),
                              "color3_hex": r.get("third_hex", "")})
                self.objects[oid] = o
                ph["objects"].append(oid)

            def backed_up(name):
                bk = s.out / name
                return {p.name for p in bk.iterdir() if (p / "preview.jpg").exists()} if bk.exists() else set()
            backups, backups2 = backed_up(BACKUP), backed_up(BACKUP_V2)
            for ph in photos.values():
                ph["objects"].sort(key=lambda o: int(self.objects[o].get("object_index") or 0))
                ph["bk"] = ph["id"] in backups       # preview from before the round-1 correction is available
                ph["bk2"] = ph["id"] in backups2     # preview from before the round-2 correction is available
            plist += sorted(photos.values(), key=lambda p: (len(p["id"]), p["id"]))
            # decisions: the set's review CSV is the master record (full_auto_tags.csv may have been rebuilt since)
            if s.review_csv.exists():
                with s.review_csv.open(encoding="utf-8-sig", newline="") as fh:
                    for r in csv.DictReader(fh):
                        o = self.objects.get(r.get("object_id"))
                        if o is not None and self.set_of(r.get("object_id")) is s:
                            for k in HISTORY:
                                o[k] = r.get(k, "")
        self.photos = plist
        self.photo_by_id = {ph["id"]: ph for ph in plist}
        for oid, o in self.objects.items():
            o["object_location"] = self.location(oid)
        per_set = ", ".join(f"{s.name}: {sum(1 for p in plist if self.photo_set[p['id']] is s)} photos" for s in self.sets)
        print(f"[review] loaded {len(self.photos)} photos ({per_set}), {len(self.objects)} objects in "
              f"{time.time() - t0:.1f} s" + (f" | {skipped} rows skipped (photo id already in an earlier set)" if skipped else ""),
              flush=True)

    def export_csv(self) -> bytes:
        """The saved review decisions of every set in one CSV (image_set = 7k_matched / bonus_image)."""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=REVIEW_COLUMNS + ["image_set"], extrasaction="ignore", lineterminator="\r\n")
        w.writeheader()
        for ph in self.photos:
            for oid in ph["objects"]:
                o = self.objects[oid]
                w.writerow({**{c: ("" if o.get(c) is None else o.get(c)) for c in REVIEW_COLUMNS},
                            "image_set": ph.get("set", MAIN_SET)})
        return ("﻿" + buf.getvalue()).encode("utf-8")

    def data(self):
        objs = {oid: {"k": o["object_index"], "n": o["object_count"],
                      "c": [[o[f"color{i}_name"], o[f"color{i}_hex"], o[f"color{i}_pct"]] for i in (1, 2, 3)],
                      "st": o["review_status"], "note": o["review_note"], "loc": o["object_location"],
                      "tr": o["tag_review_required"] == "True", "seg": o["segmentation_status"],
                      "sf": o["surface"], "tp": o["transparency"], "ef": o["glaze_effect"], "df": o["defect"],
                      "corr": (o.get("correction") or "").split(": ", 1)[-1],
                      "corr2": (o.get("correction_v2") or "").split(": ", 1)[-1].split(" | reviewer note")[0],
                      "prev": o.get("previous_object_id") or ""}
                for oid, o in self.objects.items()}
        st = [o["review_status"] for o in self.objects.values()]
        counts = {"photos": len(self.photos), "objects": len(self.objects),
                  "not_needed": st.count("not_needed"), "needs_review": st.count("needs_review"),
                  "completed": st.count("completed"), "completed_v2": st.count("completed_v2"),
                  "in_does_not_needed": sum(1 for o in self.objects.values() if o["object_location"] == NOT_NEEDED),
                  "bonus_photos": sum(1 for p in self.photos if p.get("set") == BONUS_SET),
                  "bonus_objects": sum(len(p["objects"]) for p in self.photos if p.get("set") == BONUS_SET)}
        return {"photos": self.photos, "objects": objs, "counts": counts}

    # ------------------------------------------------------------------ moving
    def _move_object(self, oid, to_not_needed: bool, log):
        """Move the object's 5 files inside its own set's folder; roll back this object's moves if one fails."""
        done = []
        base = self.set_of(oid).out
        try:
            for sub in OBJ_DIRS:
                name = f"{oid}{OBJ_EXT[sub]}"
                a, b = base / sub / name, base / NOT_NEEDED / sub / name
                src, dst = (a, b) if to_not_needed else (b, a)
                if not src.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():   # never overwrite: keep the older copy under a new name
                    dst.rename(dst.with_name(f"{dst.stem}.superseded_{int(time.time())}{dst.suffix}"))
                os.replace(src, dst)
                done.append((src, dst))
            for src, dst in done:
                log.append({"time": now_iso(), "object_id": oid,
                            "action": "move_to_does_not_needed" if to_not_needed else "restore",
                            "from": str(src.relative_to(base)), "to": str(dst.relative_to(base))})
            return None
        except Exception as exc:  # noqa: BLE001
            for src, dst in reversed(done):
                try:
                    os.replace(dst, src)
                except Exception:  # noqa: BLE001
                    pass
            return f"{oid}: {exc}"

    # ------------------------------------------------------------------ saving
    def save(self, changes: dict):
        with self.lock:
            errors, moved, restored = [], 0, 0
            stamp = now_iso()
            for oid, ch in changes.items():
                o = self.objects.get(oid)
                st = (ch.get("status") or "").strip()
                if o is None or st not in STATUSES:
                    errors.append(f"{oid}: unknown object or status '{st}'")
                    continue
                o["review_status"] = st
                o["review_note"] = (ch.get("note") or "").replace("\n", " ").strip()[:500]
                o["reviewed_at"] = stamp if (st or o["review_note"]) else ""
            log = []
            for oid, o in self.objects.items():
                loc = self.location(oid)
                if o["review_status"] == "not_needed" and loc == "smaller_object":
                    err = self._move_object(oid, True, log)
                    moved += err is None
                elif o["review_status"] != "not_needed" and loc == NOT_NEEDED:
                    err = self._move_object(oid, False, log)
                    restored += err is None
                else:
                    err = None
                if err:
                    errors.append(err)
                o["object_location"] = self.location(oid)
            for s in self.sets:          # each set keeps its own review CSV, move log and full_auto_tags.csv
                rows = [self.objects[oid] for ph in self.photos if self.photo_set[ph["id"]] is s for oid in ph["objects"]]
                atomic_csv(s.review_csv, rows, REVIEW_COLUMNS)
                slog = [e for e in log if self.set_of(e["object_id"]) is s]
                if slog:
                    new = not s.move_log.exists()
                    with s.move_log.open("a", newline="", encoding="utf-8-sig") as fh:
                        w = csv.DictWriter(fh, fieldnames=["time", "object_id", "action", "from", "to"])
                        if new:
                            w.writeheader()
                        w.writerows(slog)
                self._update_full_csv(s.out)
            res = {"saved_at": stamp, "changes": len(changes), "moved_to_does_not_needed": moved,
                   "restored": restored, "errors": errors, "counts": self.data()["counts"]}
            print(f"[review] save: {json.dumps({k: v for k, v in res.items() if k != 'counts'})}", flush=True)
            return res

    def _update_full_csv(self, out: Path):
        p = out / "full_auto_tags.csv"
        with p.open(encoding="utf-8-sig", newline="") as fh:
            rd = csv.DictReader(fh)
            cols = list(rd.fieldnames)
            rows = list(rd)
        for c in ("review_status", "review_note", "reviewed_at", "object_location"):
            if c not in cols:
                cols.append(c)
        for r in rows:
            o = self.objects.get(r.get("object_id"))
            if o is not None:
                for c in ("review_status", "review_note", "reviewed_at", "object_location"):
                    r[c] = o[c]
        atomic_csv(p, rows, cols)

    # ------------------------------------------------------------------ thumbnails
    def thumb(self, kind, ident):
        if not ID_RE.match(ident):
            return None
        s = self.set_of(ident)
        dst = s.thumbs / kind / f"{ident}.jpg"
        if kind in ("orig", "big"):
            src = self.photo_file(ident)
        else:
            src = s.out / "smaller_object" / f"{ident}.jpg"
            if not src.exists():
                src = s.out / NOT_NEEDED / "smaller_object" / f"{ident}.jpg"
        if not src.exists():
            return None
        # a re-segmented photo reuses object ids: rebuild the thumbnail when the crop is newer
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst
        im = Image.open(src)
        im.draft("RGB", (THUMB[kind] * 2, THUMB[kind] * 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((THUMB[kind], THUMB[kind]), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f".{threading.get_ident()}.tmp")
        im.save(tmp, format="JPEG", quality=84)
        os.replace(tmp, dst)
        return dst

    # ------------------------------------------------------------------ outlines
    def outlines(self, ident):
        """Outline polygons of every object of a photo, from its instance mask (pixel value = object index).

        Returns {"w", "h", "o": {index: {"p": [flat x,y list per contour], "a": [x, y] label point,
        "f": area fraction}}} in the coordinates of the mask scaled to OUTLINE_SIDE; holes are included."""
        if not PHOTO_RE.match(ident):
            return None
        s = self.set_of(ident)
        src = s.out / "instance_masks" / f"{ident}.png"
        if not src.exists():
            return None
        dst = s.thumbs / "outlines" / f"{ident}.json"
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return json.loads(dst.read_text(encoding="utf-8"))
        full = np.array(Image.open(src).convert("L"))
        H, W = full.shape
        s = min(1.0, OUTLINE_SIDE / max(H, W))
        w, h = max(1, round(W * s)), max(1, round(H * s))
        lab = cv2.resize(full, (w, h), interpolation=cv2.INTER_NEAREST) if s < 1 else full
        present = set(int(v) for v in np.unique(lab))
        objs = {}
        for k in (int(v) for v in np.unique(full)):
            if k == 0:
                continue
            if k not in present:       # too small to survive the scaling: show its box instead
                ys, xs = np.nonzero(full == k)
                x0, y0, x1, y1 = (int(v) for v in (xs.min() * s, ys.min() * s, xs.max() * s + 1, ys.max() * s + 1))
                objs[str(k)] = {"p": [[x0, y0, x1, y0, x1, y1, x0, y1]], "a": [(x0 + x1) // 2, (y0 + y1) // 2],
                                "f": round(len(xs) / (H * W), 5)}
                continue
            m = (lab == k).astype(np.uint8)
            area = int(m.sum())
            cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            big = max(abs(cv2.contourArea(c)) for c in cs)
            polys = []
            for c in cs:
                if abs(cv2.contourArea(c)) < max(4.0, 0.01 * big) and len(cs) > 1:
                    continue
                c = cv2.approxPolyDP(c, 0.7, True).reshape(-1)
                if len(c) >= 6:
                    polys.append([int(v) for v in c])
            dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
            y, x = np.unravel_index(int(dist.argmax()), dist.shape)
            objs[str(k)] = {"p": polys, "a": [int(x), int(y)], "f": round(area / (w * h), 5)}
        data = {"w": w, "h": h, "o": objs}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, dst)
        return data

    def outlines_safe(self, ident):
        try:
            return self.outlines(ident)
        except Exception as exc:  # noqa: BLE001
            print(f"[review] outlines {ident} failed: {exc}", flush=True)
            return None


class TileData:
    """Read-only access to the machine labels shown in the attribute (tile) review panel: Grounding DINO labels v2
    (with Claude's check and the validation summary), the earlier v1 predictions and the recipe metadata."""

    def __init__(self, store: Store):
        self.store = store
        self._v1, self._v1_mtime, self._v2, self._v2_mtime = {}, None, {}, None
        self._b1, self._b1_mtime, self._b3, self._b3_mtime = {}, None, {}, None      # bonus_image objects (attributes v3)

    def bonus(self) -> tuple[dict, dict]:
        """v1 predictions and v3 labels of the bonus_image objects (mm/bonus_v3.py), empty until that run exists"""
        self._b1, self._b1_mtime = self._cached(ATTR_V3 / "bonus_attr_v1.json", self._b1, self._b1_mtime)
        self._b3, self._b3_mtime = self._cached(ATTR_V3 / "attribute_predictions_v3_bonus.json", self._b3, self._b3_mtime)
        return self._b1.get("objects") or {}, self._b3.get("objects") or {}

    @staticmethod
    def _cached(p: Path, cache: dict, mtime):
        if not p.exists():
            return {}, None
        m = p.stat().st_mtime
        if m != mtime:
            return json.loads(p.read_text(encoding="utf-8")), m
        return cache, mtime

    def v1(self) -> dict:
        self._v1, self._v1_mtime = self._cached(ATTR_DIR / "attribute_predictions.json", self._v1, self._v1_mtime)
        return self._v1

    def v2(self) -> dict:
        """Grounding DINO attribute labels: v3 (trained on Claude's labels, top-3 candidates) when present, else v2."""
        self._v2, self._v2_mtime = self._cached(attr_labels_path(), self._v2, self._v2_mtime)
        return self._v2

    def predictions(self, oid) -> dict | None:
        return (self.v1().get("objects") or {}).get(oid) or self.bonus()[0].get(oid)

    def dino(self, oid) -> dict | None:
        return (self.v2().get("objects") or {}).get(oid) or self.bonus()[1].get(oid)

    def uncertain_counts(self) -> dict:
        """object -> priority for the "attributes to check" filter: uncertain attributes + disagreements with Claude"""
        return (self.v2().get("review_priority") or {})

    def rgba_path(self, oid) -> Path:
        out = self.store.set_of(oid).out
        p = out / "smaller_object_rgba_raw" / f"{oid}.png"
        return p if p.exists() else out / NOT_NEEDED / "smaller_object_rgba_raw" / f"{oid}.png"


def make_handler(store: Store, tiles=None):
    tdata = TileData(store)

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: only errors are printed
            pass

        def _send(self, body: bytes, ctype: str, status=200, extra=None, cache=False):
            gz = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 2048 and ctype.startswith(("application/json", "text/"))
            if gz:
                body = gzip.compress(body, 5)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            if gz:
                self.send_header("Content-Encoding", "gzip")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            path = unquote(u.path)
            try:
                if path in ("/", "/index.html"):
                    return self._send(APP_HTML.read_bytes(), "text/html; charset=utf-8")
                if path == "/api/data":
                    return self._send(json.dumps(store.data(), ensure_ascii=False).encode("utf-8"), "application/json")
                if path == "/api/export.csv":
                    with store.lock:
                        body = store.export_csv()
                    return self._send(body, "text/csv; charset=utf-8",
                                      extra={"Content-Disposition": 'attachment; filename="object_review.csv"'})
                if path == "/api/outlines":
                    ids = [i for i in (parse_qs(u.query).get("ids") or [""])[0].split(",") if PHOTO_RE.match(i)][:200]
                    res = dict(zip(ids, POOL.map(store.outlines_safe, ids)))
                    return self._send(json.dumps(res, separators=(",", ":")).encode("utf-8"), "application/json")
                # ---- review_tile (sidecar category + tile review panel)
                if path == "/api/tiles":
                    v2 = tdata.v2()
                    res = {"assigned": tiles.categories() if tiles else {}, "priority": tdata.uncertain_counts(),
                           "enabled": tiles is not None, "dino_ready": bool(v2), "validation": v2.get("validation", {})}
                    return self._send(json.dumps(res, separators=(",", ":")).encode("utf-8"), "application/json")
                m = re.match(r"^/api/tile/(\d+_\d+)$", path)
                if m:
                    oid = m.group(1)
                    o = store.objects.get(oid)
                    if o is None:
                        return self._send(b"not found", "text/plain", 404)
                    from glazeprep.review_tile import ATTR_STATUS, MULTI, VOCAB
                    v2 = tdata.v2()
                    res = {"object_id": oid, "csv": {k: o.get(k, "") for k in ("recipe_id", "recipe_name", "subtype_csv",
                           "surface", "transparency", "glaze_effect", "defect", "review_status")},
                           "predictions": tdata.predictions(oid), "dino": tdata.dino(oid),
                           "validation": v2.get("validation", {}), "dino_version": v2.get("version", ""),
                           "review": tiles.get(oid) if tiles else {}, "history": tiles.history_count(oid) if tiles else 0,
                           "vocab": VOCAB, "multi": sorted(MULTI), "attr_status": ATTR_STATUS}
                    return self._send(json.dumps(res, ensure_ascii=False).encode("utf-8"), "application/json")
                m = re.match(r"^/full/rgba/(\d+_\d+)\.png$", path)
                if m:
                    p = tdata.rgba_path(m.group(1))
                    if not p.exists():
                        return self._send(b"not found", "text/plain", 404)
                    return self._send(p.read_bytes(), "image/png")
                m = re.match(r"^/thumb/(orig|obj|big)/([0-9_]+)\.jpg$", path)
                if m:
                    p = store.thumb(m.group(1), m.group(2))
                    if p is None:
                        return self._send(b"not found", "text/plain", 404)
                    return self._send(p.read_bytes(), "image/jpeg", extra={"Last-Modified": str(int(p.stat().st_mtime))})
                m = re.match(r"^/full/(orig|obj|card|preview|before|before2)/([0-9_]+)\.jpg$", path)
                if m:
                    kind, ident = m.groups()
                    out = store.set_of(ident).out
                    if kind == "orig":
                        p = store.photo_file(ident)
                    elif kind == "preview":
                        p = out / "preview" / f"{ident}.jpg"
                    elif kind in ("before", "before2"):
                        p = out / (BACKUP if kind == "before" else BACKUP_V2) / ident / "preview.jpg"
                    else:
                        sub = "smaller_object" if kind == "obj" else "detailed_preview"
                        p = out / sub / f"{ident}.jpg"
                        if not p.exists():
                            p = out / NOT_NEEDED / sub / f"{ident}.jpg"
                    if not p.exists():
                        return self._send(b"not found", "text/plain", 404)
                    return self._send(p.read_bytes(), IMG_TYPES.get(p.suffix.lower(), "image/jpeg"))
                return self._send(b"not found", "text/plain", 404)
            except Exception as exc:  # noqa: BLE001
                print(f"[review] GET {path} failed: {exc}", flush=True)
                return self._send(str(exc).encode(), "text/plain", 500)

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/save" and not path.startswith("/api/tile"):
                return self._send(b"not found", "text/plain", 404)
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}",
                                                   f"http://localhost:{self.server.server_port}"):
                return self._send(b"forbidden", "text/plain", 403)
            if path.startswith("/api/tile"):
                return self._tile_post(path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                res = store.save(body.get("changes") or {})
                return self._send(json.dumps(res).encode("utf-8"), "application/json")
            except Exception as exc:  # noqa: BLE001
                print(f"[review] save failed: {exc}", flush=True)
                return self._send(json.dumps({"error": str(exc)}).encode(), "application/json", 500)

        def _tile_post(self, path):
            """/api/tile/<oid> save an attribute review · /api/tile_category {oid, category}"""
            if tiles is None:
                return self._send(json.dumps({"error": "tile review is not enabled"}).encode(), "application/json", 400)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                m = re.match(r"^/api/tile/(\d+_\d+)$", path)
                if m and m.group(1) in store.objects:
                    rec = tiles.save(m.group(1), body)
                    print(f"[review] tile review saved: {m.group(1)} ({rec.get('category') or 'no category'})", flush=True)
                    return self._send(json.dumps({"ok": True, "review": rec}).encode("utf-8"), "application/json")
                oid = str(body.get("oid", ""))
                if not OID_RE.match(oid) or oid not in store.objects:
                    return self._send(json.dumps({"error": f"unknown object {oid!r}"}).encode(), "application/json", 400)
                if path == "/api/tile_category":
                    rec = tiles.set_category(oid, body.get("category", ""), body.get("source", "human"))
                    return self._send(json.dumps({"ok": True, "review": rec}).encode("utf-8"), "application/json")
                return self._send(b"not found", "text/plain", 404)
            except ValueError as exc:
                return self._send(json.dumps({"error": str(exc)}).encode(), "application/json", 400)
            except Exception as exc:  # noqa: BLE001
                print(f"[review] tile request failed: {exc}", flush=True)
                return self._send(json.dumps({"error": str(exc)}).encode(), "application/json", 500)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=BASE / "full_segmentation_claude_process")
    ap.add_argument("--source", type=Path, default=BASE / "7k_matched_images" / "7k_matched_images")
    ap.add_argument("--bonus", type=Path, default=BASE / "bonus_image_claude_process",
                    help="output folder of the bonus_image photos (shown after the 7k photos)")
    ap.add_argument("--bonus-source", type=Path, default=BASE / "others_image", help="original bonus photos")
    ap.add_argument("--no-bonus", action="store_true", help="show the 7k matched photos only")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    extra = []
    if not args.no_bonus:
        if (args.bonus / "full_auto_tags.csv").exists():
            extra.append((BONUS_SET, args.bonus.resolve(), args.bonus_source.resolve()))
        else:
            print(f"[review] bonus_image set not shown: {args.bonus / 'full_auto_tags.csv'} does not exist", flush=True)
    store = Store(args.output.resolve(), args.source.resolve(), extra)
    sys.path.insert(0, str(CODE))
    from glazeprep.review_tile import TileReviewStore
    tiles = TileReviewStore(store.out / "review")
    print(f"[review] tile review (review_tile category): {len(tiles.data['objects'])} record(s) in "
          f"{tiles.path} | Grounding DINO attribute labels: "
          f"{attr_labels_path().name if attr_labels_path().exists() else 'not computed yet (run_attributes_v3.py)'}", flush=True)
    srv = None
    for port in range(args.port, args.port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, tiles))
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit("no free port found")
    url = f"http://127.0.0.1:{srv.server_port}/"
    print("=" * 66, flush=True)
    print("CERAMIC OBJECT REVIEW APP", flush=True)
    print(f"Open: {url}", flush=True)
    print("Keep this window open while you review. Close it (or Ctrl+C) to stop the app.", flush=True)
    print("=" * 66, flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
