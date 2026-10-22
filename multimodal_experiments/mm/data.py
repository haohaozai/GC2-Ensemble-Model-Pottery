"""Canonical recipe-level dataset: one row per Glazy recipe with a photo, aligned by recipe ID.

* recipe ID is the key: the photo filename stem equals the Glazy recipe ID (verified for all 7,063 photos)
* text  = tag description from the existing text branch (TagNormaliser) + cleaned free-text Description
* image = usable segmented objects of the photo (master review CSV: not_needed / needs_review excluded)
* formula = oxide analysis (weight %) + ingredients
* leakage groups (union-find): identical oxide analysis, near-duplicate photos (dHash), same name + creator
* split by group BEFORE anything is fitted: 70 / 15 / 15
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .common import (CACHE, CSV7K, GLAZY_EXPORT, ING7K, PHOTOS, REVIEW, SEED, SEG, log, obj_file, read_csv,
                     read_json, sha256_file, write_csv, write_json)

OXIDE_START = "SiO2"
SURFACE_GROUP = {"glossy": "glossy", "semi-glossy": "glossy", "satin": "satin", "satin-matte": "satin",
                 "semi-matte": "satin", "smooth matte": "matte", "matte": "matte", "dry matte": "matte",
                 "stony matte": "matte"}
TRANSP_GROUP = {"transparent": "clear", "translucent": "clear", "semi-opaque": "opaque", "opaque": "opaque"}
USABLE = {"", "completed", "completed_v2"}


def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", html.unescape(s or ""))
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def oxide_columns(header: list[str]) -> list[str]:
    return header[header.index(OXIDE_START):]


def text_normaliser():
    from text_branch.cleaning import TagNormaliser
    from text_branch.config import CleaningConfig
    return TagNormaliser(CleaningConfig())


def clean_tags(rows: list[dict], id_col: str = "recipe_id") -> dict[str, dict]:
    """Text-branch cleaning of each row -> tags + deterministic description (unchanged text-branch rules)."""
    norm = text_normaliser()
    out = {}
    for r in rows:
        row = dict(r)
        row["ID"] = r[id_col]
        cr = norm.clean(row)
        t = cr.tags
        out[r[id_col]] = {
            "desc_tags": cr.description or "",
            "subtype": " - ".join(t.subtype) if t.subtype else "",
            "subtype_l2": t.subtype[1] if t.subtype and len(t.subtype) > 1 else "",
            "cone_lo": t.cone.low if t.cone else "", "cone_hi": t.cone.high if t.cone else "",
            "cone_lo_i": t.cone.low_index if t.cone else -1, "cone_hi_i": t.cone.high_index if t.cone else -1,
            "surface": t.surface or "", "transparency": t.transparency or "",
            "atmospheres": ";".join(t.atmospheres) if t.atmospheres else "",
            "tag_flags": ";".join(f.code for f in t.flags),
        }
    return out


def dhash(path, size=8) -> int:
    with Image.open(path) as im:
        im.draft("L", (size * 16, size * 16))
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def photo_hashes(ids: list[str]) -> dict[str, int]:
    p = CACHE / "photo_dhash.json"
    cache = read_json(p, {})
    todo = [i for i in ids if i not in cache]
    if todo:
        with ThreadPoolExecutor(8) as ex:
            for i, h in zip(todo, ex.map(lambda i: dhash(PHOTOS / f"{i}.jpg"), todo)):
                cache[i] = h
        write_json(p, cache)
    return {i: int(cache[i]) for i in ids}


class UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dominant_colours(oids_by_recipe: dict[str, list[tuple[str, int]]]) -> dict[str, list]:
    """Per recipe: ColorThief palette (hex, fraction) of its usable objects, area-weighted, from tag_json."""
    out = {}
    for rid, objs in oids_by_recipe.items():
        acc = defaultdict(float)
        for oid, area in objs:
            t = read_json(obj_file("tag_json", f"{oid}.json"), {}) or {}
            for c in (t.get("colorthief_palette") or [])[:3]:
                acc[c["hex"]] += area * float(c.get("fraction") or 0)
        tot = sum(acc.values()) or 1.0
        out[rid] = [[h, round(v / tot, 4)] for h, v in sorted(acc.items(), key=lambda kv: -kv[1])[:3]]
    return out


def build(work, smoke_n: int = 0):
    """Write work/data/{recipes.csv, objects.csv, manifest.json}. Smoke: a group-complete random subset."""
    t_rows = read_csv(CSV7K)
    header = list(t_rows[0].keys())
    oxides = oxide_columns(header)
    photos = {p.stem for p in PHOTOS.glob("*.jpg")}
    # The 7k CSV went through Excel: 3,338 cone ranges became dates ("9 - 10" -> "10-Sep") and 2 names "#NAME?".
    # The original Glazy export (same recipes, identical oxides) is the source of every recipe field; the 7k CSV
    # only defines which recipes have a photo. Nothing is modified on disk.
    export = {e["ID"]: e for e in read_csv(GLAZY_EXPORT)}
    rows = {}
    bad_cone = 0
    for r in t_rows:
        rid = r["recipe_id"]
        if rid in photos:
            e = export.get(rid)
            bad_cone += bool(re.search(r"[A-Za-z]{3}", r.get("Cone") or ""))
            rows[rid] = {**e, "recipe_id": rid} if e else r
    log().info(f"[data] 7k CSV {len(t_rows)} rows, photos {len(photos)}, recipes with photo {len(rows)} | fields from the "
               f"Glazy export (7k CSV cone fields damaged by Excel: {bad_cone})")

    # usable objects from the master review CSV
    review = read_csv(REVIEW)
    objs = defaultdict(list)
    status = Counter()
    for r in review:
        status[r["review_status"] or "(blank)"] += 1
        if r["review_status"] in USABLE and r["object_location"] == "smaller_object":
            objs[r["parent_filename"][:-4]].append(r["object_id"])
    obj_rows = []
    obj_area = {}
    for rid, oids in objs.items():
        for oid in oids:
            m = obj_file("smaller_object_masks_raw", f"{oid}.png")
            with Image.open(m) as im:
                a = int((np.asarray(im) > 127).sum())
            obj_area[oid] = a
            obj_rows.append({"object_id": oid, "recipe_id": rid, "area_px": a})

    tags = clean_tags(list(rows.values()))
    ing = defaultdict(list)
    for r in read_csv(ING7K):
        if r["recipe_id"] in rows:
            ing[r["recipe_id"]].append(r)
    colours = dominant_colours({rid: [(o, obj_area[o]) for o in objs.get(rid, [])] for rid in rows})

    recs = []
    for rid in sorted(rows, key=lambda x: int(x)):
        r = rows[rid]
        ox = [float(r[c]) if (r.get(c) or "").strip() else 0.0 for c in oxides]
        tg = tags[rid]
        desc_free = strip_html(r.get("Description", ""))
        recs.append({
            "recipe_id": rid, "name": r["Name"], "creator_id": r["Created By ID"], "creator": r["Created By Name"],
            "state": r["State"], **tg, "desc_free": desc_free[:1500],
            "has_text": int(bool(tg["desc_tags"])), "has_image": int(bool(objs.get(rid))),
            "has_formula": int(sum(1 for v in ox if v > 0) >= 3),
            "n_objects": len(objs.get(rid, [])), "n_ingredients": len(ing.get(rid, [])),
            "oxides_json": json.dumps({c: v for c, v in zip(oxides, ox) if v > 0}),
            "ingredients_json": json.dumps([[x["Ingredient ID"], x["Ingredient Name"], float(x["Percentage"] or 0),
                                             x["Additional"] == "Yes"] for x in ing.get(rid, [])]),
            "palette_json": json.dumps(colours.get(rid, [])),
            "glazy_url": f"https://glazy.org/recipes/{rid}",
        })

    # leakage groups
    uf = UF([r["recipe_id"] for r in recs])
    by_formula = defaultdict(list)
    for r in recs:
        ox = json.loads(r["oxides_json"])
        if len(ox) >= 3:
            by_formula[tuple(sorted((k, round(v, 1)) for k, v in ox.items()))].append(r["recipe_id"])
    n_formula_dup = 0
    for ids in by_formula.values():
        if len(ids) > 1:
            n_formula_dup += len(ids)
            for i in ids[1:]:
                uf.union(ids[0], i)
    hashes = photo_hashes([r["recipe_id"] for r in recs])
    keys = list(hashes)
    H = np.array([hashes[k] for k in keys], dtype=np.uint64)
    n_photo_dup = 0
    for a in range(len(keys)):
        x = np.bitwise_xor(H[a + 1:], H[a])
        d = np.array([bin(int(v)).count("1") for v in x]) if len(x) else np.array([])
        for b in np.nonzero(d <= 2)[0]:     # <= 2 of 64 bits: true near-duplicates (<= 4 chains 224 photos)
            uf.union(keys[a], keys[a + 1 + b])
            n_photo_dup += 1
    by_name = defaultdict(list)
    for r in recs:
        nm = re.sub(r"[^a-z0-9]+", " ", r["name"].lower()).strip()
        if nm:
            by_name[(nm, r["creator_id"])].append(r["recipe_id"])
    n_name_dup = 0
    for ids in by_name.values():
        if len(ids) > 1:
            n_name_dup += len(ids)
            for i in ids[1:]:
                uf.union(ids[0], i)
    for r in recs:
        r["group_id"] = uf.find(r["recipe_id"])
    groups = defaultdict(list)
    for r in recs:
        groups[r["group_id"]].append(r)

    # split by group: 70 / 15 / 15 of recipes
    rng = np.random.default_rng(SEED)
    gids = sorted(groups)
    rng.shuffle(gids)
    n = len(recs)
    acc, cut1, cut2 = 0, 0.70 * n, 0.85 * n
    for g in gids:
        part = "train" if acc < cut1 else "val" if acc < cut2 else "test"
        for r in groups[g]:
            r["split"] = part
        acc += len(groups[g])

    gsize = sorted((len(v) for v in groups.values()), reverse=True)
    if smoke_n:
        keep = []
        for part, frac in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            acc = 0
            for g in gids:
                if groups[g][0]["split"] == part and acc < frac * smoke_n:
                    keep.append(g)
                    acc += len(groups[g])
        ks = set(keep)
        recs = [r for r in recs if r["group_id"] in ks]
        keep_ids = {r["recipe_id"] for r in recs}
        obj_rows = [o for o in obj_rows if o["recipe_id"] in keep_ids]

    d = work / "data"
    write_csv(d / "recipes.csv", recs)
    write_csv(d / "objects.csv", obj_rows)
    sc = Counter(r["split"] for r in recs)
    man = {"recipes": len(recs), "objects": len(obj_rows), "split": dict(sc),
           "modalities": {m: sum(int(r[f"has_{m}"]) for r in recs) for m in ("text", "image", "formula")},
           "all_three": sum(1 for r in recs if r["has_text"] and r["has_image"] and r["has_formula"]),
           "groups": len({r["group_id"] for r in recs}),
           "largest_groups_full_data": gsize[:10], "groups_of_size_1_full_data": sum(1 for s in gsize if s == 1),
           "leakage": {"recipes_in_identical_formula_sets": n_formula_dup, "near_duplicate_photo_pairs": n_photo_dup,
                       "recipes_sharing_name_and_creator": n_name_dup},
           "review_status_counts": dict(status), "smoke_n": smoke_n,
           "csv7k_cone_fields_damaged_by_excel": bad_cone,
           "sources": {"csv7k": str(CSV7K), "csv7k_sha256": sha256_file(CSV7K), "glazy_export": str(GLAZY_EXPORT),
                       "glazy_export_sha256": sha256_file(GLAZY_EXPORT), "review": str(REVIEW),
                       "review_sha256": sha256_file(REVIEW), "photos": str(PHOTOS), "segmentation": str(SEG)},
           "oxide_columns": oxides}
    write_json(d / "manifest.json", man)
    log().info(f"[data] {man['recipes']} recipes, {man['objects']} usable objects, split {dict(sc)}, "
               f"modalities {man['modalities']}, all three {man['all_three']}, groups {man['groups']}, "
               f"leakage {man['leakage']}")
    return man


def load(work) -> list[dict]:
    return read_csv(work / "data" / "recipes.csv")


def load_objects(work) -> list[dict]:
    return read_csv(work / "data" / "objects.csv")
