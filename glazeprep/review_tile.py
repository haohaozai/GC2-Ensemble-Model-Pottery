"""review_tile: a review CATEGORY (not a status) for ceramic objects whose colour regions / surfaces need a closer look.

Coexistence with the existing review (scripts/18_review_app.py):
  * review/object_review.csv keeps its columns and its workflow field `review_status`
    ("" / not_needed / needs_review / completed / completed_v2) - untouched by this module;
  * the category and the tile review live in a SIDECAR file next to it: review/tile_review.json
    {"version": 1, "objects": {object_id: record}} ; every save is appended to review/tile_review_history.jsonl
    (before / after) and the first save of a session backs the sidecar up to review/history/;
  * machine predictions (regions, attribute predictions) stay in their own versioned files and are never
    modified by a review - the human record is stored separately, so originals and corrections are both kept.

record = {
  "category": "review_tile" | "",                      # the review category
  "category_source": "human" | "auto_suggestion_accepted",
  "regions": {region_id: {"importance": "high|normal|ignore", "color_name", "surface", "material", "note"}},
  "custom_regions": [{"id": "c1", "rect": [x0, y0, x1, y1] (object pixels), "importance", "color_name",
                      "surface", "material", "note", "measured": {hex, lab, area_frac}}],
  "attributes": {field: {"status": "confirmed|rejected|corrected|unknown", "value": str | [str]}},
  "note": str, "reviewed_at": iso, "regions_version": "regions_v1"
}
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import threading
from pathlib import Path

CATEGORIES = ("", "review_tile")
IMPORTANCE = ("high", "normal", "ignore")
MATERIALS = ("unknown", "glaze", "slip / engobe", "clay body (exposed)", "underglaze", "oxide / stain wash",
             "glaze over slip", "other")
VOCAB = {
    "form": ["bowl", "plate", "cup", "vase", "jar", "test tile", "sculpture", "figurine", "abstract ceramic artwork",
             "irregular ceramic form", "other vessel", "unknown"],
    "surface_finish": ["glossy", "semi-glossy", "satin", "satin-matte", "semi-matte", "matte", "smooth matte",
                       "stony matte", "dry matte", "ambiguous", "unknown"],
    "transparency": ["transparent", "translucent", "semi-opaque", "opaque", "ambiguous", "unknown"],
    "texture": ["smooth", "rough", "granular", "dimpled", "ambiguous", "unknown"],
    "glaze_effect": ["speckled", "oil spot", "crystalline", "mottled", "flowing", "breaking at edges",
                     "metallic / iridescent", "visible crack network (crackle)", "none", "ambiguous", "unknown"],
    "defect": ["pinholes", "blistering", "crawling", "crazing", "excessive glaze running", "none", "ambiguous", "unknown"],
}
MULTI = {"surface_finish", "texture", "glaze_effect", "defect"}       # multi-label fields (a list of values)
ATTR_STATUS = ("confirmed", "rejected", "corrected", "unknown")


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


class TileReviewStore:
    def __init__(self, review_dir: Path):
        self.dir = Path(review_dir)
        self.path = self.dir / "tile_review.json"
        self.hist = self.dir / "tile_review_history.jsonl"
        self.lock = threading.Lock()
        self.backed_up = False
        self.data = {"version": 1, "objects": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self.data.setdefault("objects", {})

    def get(self, oid: str) -> dict:
        return self.data["objects"].get(oid, {})

    def categories(self) -> dict:
        return {oid: {"cat": r.get("category", ""), "rev": bool(r.get("reviewed_at"))}
                for oid, r in self.data["objects"].items() if r.get("category") or r.get("reviewed_at")}

    def history_count(self, oid: str) -> int:
        if not self.hist.exists():
            return 0
        n = 0
        with self.hist.open(encoding="utf-8") as fh:
            for line in fh:
                if f'"object_id": "{oid}"' in line:
                    n += 1
        return n

    # ------------------------------------------------------------ validation
    @staticmethod
    def clean(rec: dict) -> dict:
        out = {}
        cat = rec.get("category", "")
        if cat not in CATEGORIES:
            raise ValueError(f"unknown category {cat!r}")
        out["category"] = cat
        out["category_source"] = rec.get("category_source") if rec.get("category_source") in ("human", "auto_suggestion_accepted") else "human"
        regs = {}
        for rid, r in (rec.get("regions") or {}).items():
            imp = r.get("importance", "normal")
            mat = r.get("material", "unknown")
            if imp not in IMPORTANCE or mat not in MATERIALS:
                raise ValueError(f"region {rid}: bad importance / material")
            sf = r.get("surface", "")
            if sf and sf not in VOCAB["surface_finish"]:
                raise ValueError(f"region {rid}: unknown surface {sf!r}")
            regs[str(rid)] = {"importance": imp, "color_name": str(r.get("color_name", ""))[:60], "surface": sf,
                              "material": mat, "note": str(r.get("note", ""))[:300]}
        out["regions"] = regs
        cust = []
        for k, c in enumerate(rec.get("custom_regions") or []):
            rect = [int(v) for v in c.get("rect", [])][:4]
            if len(rect) != 4:
                continue
            cust.append({"id": c.get("id") or f"c{k + 1}", "rect": rect, "importance": c.get("importance", "high")
                         if c.get("importance", "high") in IMPORTANCE else "high",
                         "color_name": str(c.get("color_name", ""))[:60], "surface": c.get("surface", ""),
                         "material": c.get("material", "unknown") if c.get("material", "unknown") in MATERIALS else "unknown",
                         "note": str(c.get("note", ""))[:300], "measured": c.get("measured") or {}})
        out["custom_regions"] = cust
        attrs = {}
        for f, a in (rec.get("attributes") or {}).items():
            if f not in VOCAB:
                raise ValueError(f"unknown attribute {f!r}")
            st = a.get("status", "unknown")
            if st not in ATTR_STATUS:
                raise ValueError(f"{f}: bad status {st!r}")
            val = a.get("value", [] if f in MULTI else "")
            vals = val if isinstance(val, list) else [val]
            bad = [v for v in vals if v and v not in VOCAB[f]]
            if bad:
                raise ValueError(f"{f}: unknown value(s) {bad}")
            attrs[f] = {"status": st, "value": [v for v in vals if v] if f in MULTI else (vals[0] if vals else "")}
        out["attributes"] = attrs
        out["note"] = str(rec.get("note", ""))[:1000]
        out["regions_version"] = str(rec.get("regions_version", ""))[:40]
        return out

    # ------------------------------------------------------------ writing
    def _write(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.backed_up and self.path.exists():
            bk = self.dir / "history"
            bk.mkdir(exist_ok=True)
            shutil.copy2(self.path, bk / f"tile_review_{dt.datetime.now():%Y%m%d_%H%M%S}.json")
        self.backed_up = True
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def _log(self, oid, before, after, action):
        with self.hist.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": now_iso(), "object_id": oid, "action": action, "before": before,
                                 "after": after}, ensure_ascii=False) + "\n")

    def save(self, oid: str, rec: dict) -> dict:
        with self.lock:
            before = self.data["objects"].get(oid)
            after = self.clean(rec)
            after["reviewed_at"] = now_iso()
            self.data["objects"][oid] = after
            self._write()
            self._log(oid, before, after, "save")
            return after

    def import_record(self, oid: str, rec: dict, saved_at: str, saved_by: str = "", source: str = "online") -> dict:
        """A review saved elsewhere (the online review page, via scripts/22_sync_online_reviews.py): checked like
        save(), its own review / category times kept, who and when it was saved online recorded, logged as an import."""
        with self.lock:
            before = self.data["objects"].get(oid)
            after = self.clean(rec)
            for k in ("reviewed_at", "category_at"):
                if rec.get(k):
                    after[k] = str(rec[k])[:40]
            after["saved_by"], after["saved_at"], after["source"] = str(saved_by)[:120], str(saved_at)[:40], source
            self.data["objects"][oid] = after
            self._write()
            self._log(oid, before, after, f"import_{source}")
            return after

    def set_category(self, oid: str, category: str, source="human") -> dict:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        with self.lock:
            before = self.data["objects"].get(oid)
            after = dict(before or {"regions": {}, "custom_regions": [], "attributes": {}, "note": ""})
            after["category"] = category
            after["category_source"] = source
            after["category_at"] = now_iso()
            self.data["objects"][oid] = after
            self._write()
            self._log(oid, before, after, "category")
            return after
