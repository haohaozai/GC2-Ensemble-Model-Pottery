"""External test data: public Glazy recipes with a photo that are NOT among our 7,063 photos.

Access rules followed: only public recipe pages + the public sitemap (robots.txt allows them; /api/ and /export/
are not touched), honest user agent, >= 4 s between requests, everything cached so a re-run does not fetch
again. Glazy content is CC BY-NC-SA 4.0 and signals "ai-train=no, use=reference": these recipes are used ONLY
as test queries (never for training, model selection or tuning) and are not published anywhere.

Contamination checks: recipe ID in our photo set / in the Glazy export, near-duplicate photo (dHash <= 3 vs all
7,063 photos), identical rounded oxide analysis, same normalised name + creator. Near-duplicate photos are
excluded from the metrics; the other flags are reported.
"""
from __future__ import annotations

import html as H
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import data as D
from .common import CACHE, CODE, GLAZY_EXPORT, PHOTOS, SEED, log, read_csv, read_json, write_csv, write_json

UA = {"User-Agent": "CG2-NUS-student-research/1.0 (evaluation only; low rate; no training use)"}
DELAY = 4.0
EXPORT_MAX_ID = 819941
CONE_ORDER = ["022", "021", "020", "019", "018", "017", "016", "015", "014", "013", "012", "011", "010", "09", "08",
              "07", "06", "05 1/2", "05", "04", "03", "02", "01", "1", "2", "3", "4", "5", "5 1/2", "6", "7", "8", "9",
              "10", "11", "12", "13", "14"]


class Fetcher:
    def __init__(self, cache_dir: Path):
        import requests
        self.s = requests.Session()
        self.s.headers.update(UA)
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.last = 0.0
        self.n = 0

    def get(self, url: str, name: str, binary=False):
        p = self.cache / name
        if p.exists():
            return p.read_bytes() if binary else p.read_text(encoding="utf-8")
        wait = DELAY - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        r = self.s.get(url, timeout=60)
        self.last = time.time()
        self.n += 1
        if r.status_code != 200:
            log().info(f"[external] HTTP {r.status_code} {url}")
            return None
        if binary:
            p.write_bytes(r.content)
            return r.content
        p.write_text(r.text, encoding="utf-8")
        return r.text


def _js_obj(text: str, key: str) -> str | None:
    i = text.find(key)
    if i < 0:
        return None
    j = text.find("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j:k + 1]
    return None


def parse_recipe(page: str, rid: str) -> dict | None:
    m = re.search(r"materialId:%s,cdnId:\"([^\"]+)\"" % rid, page)
    img = m.group(1) if m else None
    def num(k):
        x = re.search(k + r":(\-?\d+|null)", page)
        return None if not x or x.group(1) == "null" else int(x.group(1))
    head = page[:page.find("components:")] if "components:" in page else page
    out = {"recipe_id": rid, "image_cdn": img,
           "from_cone_id": num("fromOrtonConeId"), "to_cone_id": num("toOrtonConeId"),
           "surface_id": num("surfaceTypeId"), "transparency_id": num("transparencyTypeId"),
           "material_type_id": num("materialTypeId"), "base_type_id": num("baseTypeId")}
    at = re.search(r"atmospheres:\$R\[\d+\]=\[([^\]]*)\]", page)
    out["atmosphere_ids"] = [int(x) for x in at.group(1).split(",") if x.strip().isdigit()] if at else []
    cr = re.search(r'createdAt:"([^"]+)"', page)
    out["created_at"] = cr.group(1) if cr else ""
    u = re.search(r'user:\$R\[\d+\]=\{id:(\d+),name:"([^"]*)"', page)
    out["creator_id"], out["creator"] = (u.group(1), u.group(2)) if u else ("", "")
    comps = re.findall(r'\{id:\d+,amount:([\d.]+),isAdd:(!0|!1),material:\$R\[\d+\]=\{id:(\d+),name:"([^"]*)"', page)
    out["components"] = [[mid, H.unescape(nm), float(a), add == "!0"] for a, add, mid, nm in comps]
    an = _js_obj(page, "analysis:$R")
    ox = {}
    if an:
        for k, v in re.findall(r"([A-Za-z0-9]+):(-?[\d.]+)", an):
            if k not in ("loi", "thermalExpansion", "weight", "oxideWeight") and float(v) > 0:
                ox[k] = float(v)
        te = re.search(r"thermalExpansion:([\d.]+)", an)
        out["glazy_thermal_expansion"] = float(te.group(1)) if te else None
    out["oxides"] = ox
    tr = re.search(r'translations:\$R\[\d+\]=\[\$R\[\d+\]=\{id:\d+,locale:"en",name:"((?:[^"\\]|\\.)*)",otherNames:(null|"(?:[^"\\]|\\.)*"),description:(null|"(?:[^"\\]|\\.)*")', page)
    if tr:
        out["name"] = H.unescape(tr.group(1).encode().decode("unicode_escape", "ignore"))
        desc = tr.group(3)
        out["description"] = D.strip_html(desc[1:-1].encode().decode("unicode_escape", "ignore")) if desc != "null" else ""
    else:
        t = re.search(r"<title>(.*?) \| Glazy</title>", page)
        out["name"], out["description"] = (H.unescape(t.group(1)) if t else ""), ""
    nav = re.search(r'data-testid="material-type-trail">(.*?)</nav>', page, re.S)
    out["subtype"] = " - ".join(H.unescape(x) for x in re.findall(r">([^<>]+)</a>", nav.group(1))) if nav else ""
    sf = re.search(r"Surface<!-- -->:.{0,40}?<span[^>]*>([^<]+)</span>", page)
    tp = re.search(r"Transparency<!-- -->:.{0,40}?<span[^>]*>([^<]+)</span>", page)
    out["surface"] = sf.group(1).strip() if sf else ""
    out["transparency"] = tp.group(1).strip() if tp else ""
    return out


def learn_maps(fetch: Fetcher, known: list[dict], n=30) -> dict:
    """cone / atmosphere id -> label learned from recipes we already have (labels known from the export)."""
    maps = {"cone": {}, "atmosphere": {}}
    rng = np.random.default_rng(SEED)
    pool = [r for r in known if r["Cone"] and r["Atmospheres"]]
    by_atm = {}
    for r in pool:
        by_atm.setdefault(r["Atmospheres"], []).append(r)
    pick = []
    for k, rows in sorted(by_atm.items(), key=lambda kv: -len(kv[1]))[:12]:
        pick.append(rows[int(rng.integers(len(rows)))])
    while len(pick) < n:
        pick.append(pool[int(rng.integers(len(pool)))])
    cone_ok, tag_ok = [], []
    for r in pick:
        page = fetch.get(f"https://glazy.org/recipes/{r['recipe_id']}", f"page_{r['recipe_id']}.html")
        if not page:
            continue
        p = parse_recipe(page, r["recipe_id"])
        cones = [c.strip() for c in re.split(r"\s+-\s+", r["Cone"])]
        if p["from_cone_id"] is not None and cones and all(c in CONE_ORDER for c in cones):
            maps["cone"][str(p["from_cone_id"])] = cones[0]
            maps["cone"][str(p["to_cone_id"])] = cones[-1]
            cone_ok.append(CONE_ORDER.index(cones[0]) + 1 == p["from_cone_id"]
                           and CONE_ORDER.index(cones[-1]) + 1 == p["to_cone_id"])
        atms = [a.strip() for a in r["Atmospheres"].split(";")]
        if len(atms) == len(p["atmosphere_ids"]) == 1:
            maps["atmosphere"][str(p["atmosphere_ids"][0])] = atms[0]
        tag_ok.append((p["surface"] or "") == (r["Surface"] or "") and (p["transparency"] or "") == (r["Transparency"] or ""))
    maps["cone_checks"] = len(cone_ok)
    maps["cone_rule_index_plus_1_holds"] = bool(cone_ok) and all(cone_ok)
    maps["tag_text_agreement"] = round(float(np.mean(tag_ok)), 3) if tag_ok else None
    return maps


def cone_label(i, maps):
    if i is None:
        return None
    if str(i) in maps["cone"]:
        return maps["cone"][str(i)]
    return CONE_ORDER[i - 1] if maps.get("cone_rule_index_plus_1_holds") and 1 <= i <= len(CONE_ORDER) else None


def sitemap_ids(fetch: Fetcher) -> list[int]:
    ids = []
    for k in range(1, 5):
        t = fetch.get(f"https://glazy.org/sitemap-recipes-{k}.xml", f"sitemap-recipes-{k}.xml")
        if t:
            ids += [int(x) for x in re.findall(r"glazy\.org/recipes/(\d+)", t)]
    return sorted(set(ids))


def run(work: Path, bank, n_target: int = 120) -> dict:
    ext = work / "external"
    fetch = Fetcher(CACHE / "glazy_http")     # shared by smoke and full runs: a page is fetched once
    have = {p.stem for p in PHOTOS.glob("*.jpg")}
    export_rows = read_csv(GLAZY_EXPORT)
    export_ids = {r["ID"] for r in export_rows}
    # our 7k photo recipes with their ORIGINAL export fields (the 7k CSV has Excel-damaged cone ranges)
    known7k = [{**r, "recipe_id": r["ID"]} for r in export_rows if r["ID"] in have]
    maps = read_json(ext / "label_maps.json")
    if not maps:
        maps = learn_maps(fetch, known7k)
        write_json(ext / "label_maps.json", maps)
    log().info(f"[external] label maps: {len(maps['cone'])} cone ids, {len(maps['atmosphere'])} atmosphere ids, "
               f"cone rule holds {maps['cone_rule_index_plus_1_holds']}, tag agreement {maps['tag_text_agreement']}")
    ids = sitemap_ids(fetch)
    cand_new = [i for i in ids if i > EXPORT_MAX_ID and str(i) not in have]
    cand_old = [i for i in ids if i <= EXPORT_MAX_ID and str(i) not in have]       # flagged if in the export
    rng = np.random.default_rng(SEED)
    rng.shuffle(cand_new)
    rng.shuffle(cand_old)
    order = cand_new[: int(n_target * 8)] + cand_old[: int(n_target * 4)]     # many recipes have no photo; stops at the target
    log().info(f"[external] sitemap {len(ids)} recipes | new after export (not ours) {len(cand_new)} | "
               f"older, not in our photos {len(cand_old)}")
    hashes7k = np.array(list(D.photo_hashes(sorted(have, key=int)).values()), dtype=np.uint64)
    ox_sig = {}
    for r in known7k:
        ox = {c: float(r[c]) for c in D.oxide_columns(list(r.keys())) if (r.get(c) or "").strip() and float(r[c]) > 0}
        if len(ox) >= 3:
            ox_sig[tuple(sorted((k, round(v, 1)) for k, v in ox.items() if round(v, 1) > 0))] = r["recipe_id"]
    names7k = {(re.sub(r"[^a-z0-9]+", " ", r["Name"].lower()).strip(), r["Created By ID"]) for r in known7k}
    accepted = read_json(ext / "accepted.json", [])
    seen = {a["recipe_id"] for a in accepted}
    stats = {"pages": 0, "no_photo": 0, "not_glaze": 0, "too_few_components": 0, "no_analysis": 0, "near_dup_photo": 0}
    img_dir = ext / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    new_target = int(n_target * 0.75)
    for i in order:
        n_new = sum(1 for a in accepted if a["after_export"])
        if len(accepted) >= n_target:
            break
        if i > EXPORT_MAX_ID and n_new >= new_target and len(cand_old):
            continue
        rid = str(i)
        if rid in seen:
            continue
        page = fetch.get(f"https://glazy.org/recipes/{rid}", f"page_{rid}.html")
        stats["pages"] += 1
        if not page:
            continue
        p = parse_recipe(page, rid)
        if not p["image_cdn"]:
            stats["no_photo"] += 1
            continue
        if not p["subtype"].lower().startswith("glaze"):
            stats["not_glaze"] += 1
            continue
        if len(p["components"]) < 3:
            stats["too_few_components"] += 1
            continue
        if len(p["oxides"]) < 3:
            stats["no_analysis"] += 1
            continue
        raw = fetch.get(f"https://cdn.glazy.org/public/uploads{p['image_cdn']}", f"img_{rid}.jpg", binary=True)
        if not raw:
            continue
        im = Image.open(CACHE / "glazy_http" / f"img_{rid}.jpg").convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600), Image.LANCZOS)
        fn = img_dir / f"x{rid}.jpg"
        im.save(fn, quality=93)
        h = D.dhash(fn)
        dist = int(min(bin(int(x)).count("1") for x in np.bitwise_xor(hashes7k, np.uint64(h))))
        sig = tuple(sorted((k, round(v, 1)) for k, v in p["oxides"].items() if round(v, 1) > 0))   # traces < 0.05 ignored
        nm = (re.sub(r"[^a-z0-9]+", " ", (p.get("name") or "").lower()).strip(), p["creator_id"])
        p.update({"after_export": i > EXPORT_MAX_ID, "in_glazy_export": rid in export_ids,
                  "min_dhash_distance_to_7k": dist, "near_duplicate_photo": dist <= 3,
                  "identical_formula_in_7k": ox_sig.get(sig, ""), "same_name_creator_in_7k": nm in names7k,
                  "cone": cone_label(p["from_cone_id"], maps), "cone_hi": cone_label(p["to_cone_id"], maps),
                  "atmospheres": ";".join(maps["atmosphere"].get(str(a), "") for a in p["atmosphere_ids"]
                                          if str(a) in maps["atmosphere"]),
                  "glazy_url": f"https://glazy.org/recipes/{rid}", "licence": "CC BY-NC-SA 4.0 (Glazy user content)",
                  "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        if p["near_duplicate_photo"]:
            stats["near_dup_photo"] += 1
        accepted.append(p)
        seen.add(rid)
        write_json(ext / "accepted.json", accepted)
        if len(accepted) % 10 == 0:
            log().info(f"[external] accepted {len(accepted)}/{n_target} (pages fetched {fetch.n})")
    write_json(ext / "fetch_stats.json", {**stats, "accepted": len(accepted), "requests": fetch.n})
    log().info(f"[external] accepted {len(accepted)} recipes; {stats}")
    seg = segment(ext)
    rows, objs = to_recipes(ext, accepted, seg)
    write_csv(ext / "recipes_ext.csv", rows)
    write_csv(ext / "objects_ext.csv", objs)
    return {"accepted": len(accepted), "after_export": sum(1 for a in accepted if a["after_export"]),
            "near_duplicate_photo": sum(1 for a in accepted if a["near_duplicate_photo"]),
            "identical_formula": sum(1 for a in accepted if a["identical_formula_in_7k"]),
            "objects": len(objs), "requests": fetch.n}


def segment(ext: Path) -> Path:
    """existing segmentation pipeline (Grounding DINO + SAM 2.1 + CLIP checks) on the new photos only."""
    out = ext / "seg"
    # one GPU job at a time: release this process's cached encoders before the segmentation subprocess starts
    from . import text_feats as TF
    TF._ENC.clear()
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    cmd = [sys.executable, str(CODE / "scripts" / "10_claude_ceramic_pipeline.py"), "--stage", "full", "--source", str(ext / "images"),
           "--output", str(out), "--no-tags", "--before-after", "none", "--display-every", "20"]
    log().info(f"[external] segmentation: {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=str(CODE))
    if r.returncode not in (0,):
        log().info(f"[external] segmentation exit code {r.returncode} (incomplete images are skipped)")
    return out


def to_recipes(ext: Path, accepted: list[dict], seg: Path):
    """External recipes in the same schema as data/recipes.csv; objects = segmented objects (CLIP p_ceramic >= 0.35;
    the largest object if none passes). No human review exists for these objects."""
    from . import image_feats as IF
    rows, objs = [], []
    raw_rows = []
    for a in accepted:
        raw_rows.append({"recipe_id": "x" + a["recipe_id"], "Name": a.get("name", ""), "Type": "Recipe",
                         "State": "", "Subtype": a.get("subtype", ""),
                         "Cone": (a["cone"] + (" - " + a["cone_hi"] if a.get("cone_hi") and a["cone_hi"] != a["cone"] else ""))
                         if a.get("cone") else "", "Surface": a.get("surface", ""),
                         "Transparency": a.get("transparency", ""), "Atmospheres": a.get("atmospheres", ""),
                         "Country": "", "Other Names": ""})
    tags = D.clean_tags(raw_rows)
    for a in accepted:
        rid = "x" + a["recipe_id"]
        meta = read_json(seg / "meta" / f"{rid}.json", {}) or {}
        inst = (meta.get("segmentation") or {}).get("instances", [])
        keep = [(k, o) for k, o in enumerate(meta.get("objects", []))
                if k < len(inst) and (inst[k].get("p_ceramic") or 0) >= 0.35]
        if not keep and meta.get("objects"):
            keep = [(0, meta["objects"][0])]
        pal_acc = {}
        for k, oid in keep:
            mp = seg / "smaller_object_masks_raw" / f"{oid}.png"
            if not mp.exists():
                continue
            area = int((np.asarray(Image.open(mp)) > 127).sum())
            objs.append({"object_id": oid, "recipe_id": rid, "area_px": area, "root": str(seg)})
            for hx, fr in IF.palette_lab(IF.load_rgba(seg / "smaller_object_rgba_raw" / f"{oid}.png")):
                pal_acc[hx] = pal_acc.get(hx, 0.0) + area * fr
        tot = sum(pal_acc.values()) or 1.0
        pal = [[h, round(v / tot, 4)] for h, v in sorted(pal_acc.items(), key=lambda kv: -kv[1])[:3]]
        t = tags[rid]
        ox = a["oxides"]
        rows.append({"recipe_id": rid, "name": a.get("name", ""), "creator_id": a.get("creator_id", ""),
                     "creator": a.get("creator", ""), "state": "", **t, "desc_free": (a.get("description") or "")[:1500],
                     "has_text": int(bool(t["desc_tags"])), "has_image": int(bool(keep)),
                     "has_formula": int(len(ox) >= 3), "n_objects": len(keep), "n_ingredients": len(a["components"]),
                     "oxides_json": json.dumps(ox), "ingredients_json": json.dumps(a["components"]),
                     "palette_json": json.dumps(pal), "glazy_url": a["glazy_url"], "group_id": rid, "split": "ext",
                     "after_export": int(a["after_export"]), "near_duplicate_photo": int(a["near_duplicate_photo"]),
                     "identical_formula_in_7k": a["identical_formula_in_7k"],
                     "same_name_creator_in_7k": int(a["same_name_creator_in_7k"]), "created_at": a["created_at"]})
    return rows, objs
