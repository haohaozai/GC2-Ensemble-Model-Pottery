"""Local glaze search server - http://127.0.0.1:8770 (this PC only, works offline).

Start: double-click ..\\..\\RUN_GLAZE_SEARCH.bat  (or: .venv\\Scripts\\python.exe app\\server.py)
Keeps the segmentation models (Grounding DINO + SAM 2.1 + CLIP), DINOv2 and the final retrieval model on the GPU.
The page (app/index.html) sends a photo / tags / formula and gets the top database recipes back.
Uploads, cut-outs and every search are saved under work/queries/ (session log: work/queries/session_log.json).
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import os  # noqa: E402

# every model is in the local cache: never contact the Hugging Face Hub (works without internet, no waiting)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from mm.common import PHOTOS, log, other_gpu_jobs, read_json, setup_logging, work_dir, write_json  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

WORK = work_dir(False)
QDIR = WORK / "queries"
SITE = WORK / "results" / "site"
PAGE = Path(__file__).resolve().parent / "index.html"
# static reference of every attribute value (build_reference_library.py): /lib/<attribute>/<value>/describe.json + PNGs
LIBRARY = Path(__file__).resolve().parents[2] / "reference_library"
LIB_TYPES = {".json": "application/json; charset=utf-8", ".png": "image/png", ".md": "text/markdown; charset=utf-8"}
LOG_FILE = QDIR / "session_log.json"
LOCK = threading.Lock()          # one GPU job at a time
S: dict = {"queries": {}}        # engine, segmenter, per-upload objects
COLOURS = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (138, 92, 246), (224, 69, 123), (201, 162, 39),
           (20, 160, 190), (120, 120, 120)]
MAX_UPLOAD = 30 * 1024 * 1024


# ---------------------------------------------------------------- image helpers
def decode_image(data: bytes) -> np.ndarray:
    from mm.query import MAX_SIDE
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        s = MAX_SIDE / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        return np.asarray(im).copy()


def fetch_url(url: str) -> bytes:
    if not re.match(r"^https?://", url or ""):
        raise ValueError("only http(s) image links can be fetched")
    if os.environ.get("PUBLIC_SPACE"):                 # online: never fetch addresses inside the host's network
        import ipaddress
        import socket
        from urllib.parse import urlparse
        for info in socket.getaddrinfo(urlparse(url).hostname or "", None):
            if not ipaddress.ip_address(info[4][0]).is_global:
                raise ValueError("this link points to a private address")
    req = urllib.request.Request(url, headers={"User-Agent": "CG2-glaze-search/1.0 (local research tool, single image)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise ValueError("image larger than 30 MB")
    return data


def draw_overlay(rgb: np.ndarray, objs: list[dict], path: Path):
    import cv2
    H, W = rgb.shape[:2]
    s = min(1.0, 1100 / max(H, W))
    im = np.asarray(Image.fromarray(rgb).resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)).copy()
    dim = (im * 0.55 + 255 * 0.45).astype(np.uint8)
    union = np.zeros(im.shape[:2], bool)
    for o in objs:
        m = np.asarray(Image.fromarray(o["mask"].astype(np.uint8) * 255).resize((im.shape[1], im.shape[0]), Image.NEAREST)) > 127
        o["_m_small"] = m
        union |= m
    out = np.where(union[..., None], im, dim).astype(np.uint8)
    for k, o in enumerate(objs):
        c = COLOURS[k % len(COLOURS)]
        cs, _ = cv2.findContours(o.pop("_m_small").astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, c, max(2, round(3 * s * 1.6)))
        x0, y0 = int(o["bbox"][0] * s), int(o["bbox"][1] * s)
        label = str(o["index"])
        org = (x0 + 6, max(22, y0 + 26))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
    Image.fromarray(out).save(path, quality=88)


def save_label_map(rgb: np.ndarray, objs: list[dict], path: Path) -> bool:
    """object map for hovering / clicking on the photo: same size as overlay.jpg, pixel value = object index (0 = none).
    Larger objects are painted first, so a small object lying on a big one stays clickable. Segmentation is not rerun."""
    if not objs or max(o["index"] for o in objs) > 255:
        return False
    H, W = rgb.shape[:2]
    s = min(1.0, 1100 / max(H, W))
    w, h = max(1, round(W * s)), max(1, round(H * s))
    lab = np.zeros((h, w), np.uint8)
    for o in sorted(objs, key=lambda o: -o["area_px"]):
        m = np.asarray(Image.fromarray(o["mask"].astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
        lab[m] = o["index"]
    Image.fromarray(lab).save(path)
    return True


def thumb(rgba: np.ndarray, path: Path, side=200):
    im = Image.fromarray(rgba)
    im.thumbnail((side, side), Image.LANCZOS)
    im.save(path)


# ---------------------------------------------------------------- query state
def new_qid() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def load_query(qid: str) -> dict | None:
    """objects of an upload (memory, else re-read from work/queries/<qid>/)."""
    if qid in S["queries"]:
        return S["queries"][qid]
    meta = read_json(QDIR / qid / "meta.json")
    if not meta:
        return None
    objs = []
    for o in meta["objects"]:
        a = np.asarray(Image.open(QDIR / qid / f"obj_{o['index']}.png").convert("RGBA"))
        objs.append({**o, "rgba": a})
    q = {"objects": objs, "meta": meta, "vec": {}}
    S["queries"][qid] = q
    return q


def segment(rgb: np.ndarray, source: str, dup: dict | None = None) -> dict:
    from mm import image_feats as IF
    from mm.query import default_selection
    qid = new_qid()
    d = QDIR / qid
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Image.fromarray(rgb).save(d / "original.jpg", quality=95)
    with LOCK:
        res = S["seg"](rgb)
    t_seg = time.time() - t0
    objs = res["objects"]
    sel = default_selection(objs)
    H, W = rgb.shape[:2]
    for o in objs:
        Image.fromarray(o["rgba"]).save(d / f"obj_{o['index']}.png")
        thumb(o["rgba"], d / f"obj_{o['index']}_t.png")
        o["palette"] = IF.palette_lab(o["rgba"])
    draw_overlay(rgb, objs, d / "overlay.jpg")
    has_labels = save_label_map(rgb, objs, d / "labels.png")
    pub = [{"index": o["index"], "p_ceramic": None if o["p_ceramic"] is None else round(o["p_ceramic"], 3),
            "area_px": o["area_px"], "area_frac": round(o["area_px"] / (H * W), 4), "bbox": o["bbox"],
            "palette": o["palette"], "primary": o["primary"], "flags": o["flags"],
            "selected": o["index"] in sel, "thumb": f"/q/{qid}/obj_{o['index']}_t.png"} for o in objs]
    meta = {"qid": qid, "source": source, "width": W, "height": H, "mode": res["mode"], "flags": res["flags"],
            "database_duplicate": dup, "labels": f"/q/{qid}/labels.png" if has_labels else None,
            "objects": pub, "default_selection": sel, "seconds_segmentation": round(t_seg, 2),
            "seconds_total": round(time.time() - t0, 2)}
    write_json(d / "meta.json", meta)
    S["queries"][qid] = {"objects": [{**p, "rgba": o["rgba"]} for p, o in zip(pub, objs)], "meta": meta, "vec": {}}
    while len(S["queries"]) > 30:
        S["queries"].pop(next(iter(S["queries"])))
    log().info(f"[app] segmented {qid} ({source}): {len(objs)} objects, mode {res['mode']}, {meta['seconds_total']} s")
    return {**meta, "overlay": f"/q/{qid}/overlay.jpg"}


def run_search(p: dict) -> dict:
    from mm.query import parse_ingredients, parse_oxides, umf_to_wt
    eng = S["engine"]
    t0 = time.time()
    use = p.get("use") or {}
    info = {}
    # ---- image
    img_vec, palette, sel = None, [], []
    q = load_query(p["qid"]) if p.get("qid") else None
    if q is not None:
        sel = [int(x) for x in (p.get("selected") or q["meta"]["default_selection"])]
        objs = [o for o in q["objects"] if o["index"] in sel]
        acc, tot = {}, 0.0
        for o in objs:
            for hx, fr in o["palette"]:
                acc[hx] = acc.get(hx, 0.0) + o["area_px"] * fr
                tot += o["area_px"] * fr
        palette = [[h, round(v / (tot or 1), 4)] for h, v in sorted(acc.items(), key=lambda kv: -kv[1])[:3]]
        if objs and use.get("image", True):
            key = tuple(sorted(sel))
            if key not in q["vec"]:
                with LOCK:
                    q["vec"][key] = eng.image_vector(objs)
            img_vec = q["vec"][key]
    # ---- text
    tags = p.get("tags") or {}
    text, cleaned = eng.compose_text(tags, p.get("notes") or "")
    has_tags = any(tags.get(k) for k in ("subtype", "cone_lo", "cone_hi", "surface", "transparency")) or bool(tags.get("atmospheres"))
    txt_vec = eng.text_vector(text) if (use.get("text", True) and (has_tags or (p.get("notes") or "").strip())) else None
    info["text_used"] = text if txt_vec is not None else ""
    # ---- formula
    ox = {k: float(v) for k, v in (p.get("oxides") or {}).items() if v not in ("", None) and float(v) > 0}
    if p.get("oxide_paste"):
        ox.update(parse_oxides(p["oxide_paste"]))
    if ox and p.get("oxide_unit") == "umf":
        ox = umf_to_wt(ox)
    ings = parse_ingredients(p.get("ingredients") or "")
    umf_vec = ing_vec = umf = None
    if ox and use.get("formula", True):
        umf_vec, ing_vec, umf = eng.formula_parts(ox, ings)
        if umf_vec is None:
            info["formula_note"] = "formula ignored: needs at least 3 oxides including a flux (R2O / RO)"
    with LOCK:
        Zq, finfo = eng.embed_query(img_vec, txt_vec, umf_vec, ing_vec)
    info.update(finfo)
    if umf:
        info["umf"] = {k: round(v, 3) for k, v in sorted(umf.items(), key=lambda kv: -kv[1])}
        si, al = umf.get("SiO2", 0), umf.get("Al2O3", 0)
        info["si_al"] = round(si / al, 2) if al else None
    if ings:
        info["ingredients_parsed"] = len(ings)
    qattr = eng.query_attributes(cleaned, palette)
    res = eng.search(Zq, p.get("weights") or {}, qattr, p.get("filters") or {}, (p.get("true_id") or "").strip() or None)
    res.update({"query": info, "palette": palette, "selected": sel, "cleaned_tags": {k: cleaned[k] for k in ("subtype", "cone_lo", "cone_hi", "surface", "transparency", "atmospheres")},
                "seconds": round(time.time() - t0, 2)})
    # ---- log
    dup = (q or {}).get("meta", {}).get("database_duplicate") if q is not None and "image" in res.get("query_modalities", []) else None
    res["database_photo"] = dup
    entry = {"entry": uuid.uuid4().hex[:8], "time": time.strftime("%Y-%m-%d %H:%M:%S"), "qid": p.get("qid") or "",
             "db_photo": (dup or {}).get("recipe_id"),
             "modalities": res.get("query_modalities", []), "text": info["text_used"], "oxides": len(ox),
             "ingredients": len(ings), "filters": res.get("filters_applied", []), "gallery": res.get("gallery"),
             "top10": [r["id"] for r in res.get("results", [])], "auto": res.get("auto"), "true": res.get("true_recipe"),
             "manual": None, "seconds": res["seconds"]}
    with LOCK:
        lg = read_json(LOG_FILE, []) or []
        lg.append(entry)
        write_json(LOG_FILE, lg)
    if p.get("qid"):
        write_json(QDIR / p["qid"] / f"search_{entry['entry']}.json", {"request": {k: v for k, v in p.items() if k != "image"}, "result": res})
    res["entry"] = entry["entry"]
    return res


def text_formula(p: dict):
    """text / formula part of a query (shared by the recipe and the object search) -> (text emb, formula emb, info, cleaned)."""
    from mm.query import parse_ingredients, parse_oxides, umf_to_wt
    eng = S["engine"]
    info = {}
    tags = p.get("tags") or {}
    text, cleaned = eng.compose_text(tags, p.get("notes") or "")
    has_tags = any(tags.get(k) for k in ("subtype", "cone_lo", "cone_hi", "surface", "transparency")) or bool(tags.get("atmospheres"))
    txt_vec = eng.text_vector(text) if (has_tags or (p.get("notes") or "").strip()) else None
    info["text_used"] = text if txt_vec is not None else ""
    ox = {k: float(v) for k, v in (p.get("oxides") or {}).items() if v not in ("", None) and float(v) > 0}
    if p.get("oxide_paste"):
        ox.update(parse_oxides(p["oxide_paste"]))
    if ox and p.get("oxide_unit") == "umf":
        ox = umf_to_wt(ox)
    ings = parse_ingredients(p.get("ingredients") or "")
    umf_vec = ing_vec = None
    if ox:
        umf_vec, ing_vec, umf = eng.formula_parts(ox, ings)
        if umf_vec is None:
            info["formula_note"] = "formula ignored: needs at least 3 oxides including a flux (R2O / RO)"
        elif umf:
            info["umf"] = {k: round(v, 3) for k, v in sorted(umf.items(), key=lambda kv: -kv[1])}
    with LOCK:
        Zq, finfo = eng.embed_query(None, txt_vec, umf_vec, ing_vec)
    info.update(finfo)
    if "text" in Zq:
        # vague text (e.g. only subtype "glaze"): identical to the text of many recipes, which then all get text
        # similarity 1.00 - with text weight 0.35 those recipes (usually the ones with the LEAST information) flood the top
        z = Zq["text"][0][0]
        Zt, Mt = eng.Z["text"]
        s = Zt[Mt] @ z / (np.linalg.norm(Zt[Mt], axis=1) * np.linalg.norm(z) + 1e-9)
        n_same = int((s >= 0.999).sum())
        info["text_identical_recipes"] = n_same
        if n_same >= 25:
            info["text_warning"] = (f"Your text \"{text}\" is word for word the text of {n_same} recipes: all of them get text similarity "
                                    f"1.00, and text weighs 0.35, so recipes with almost no information can push the right answer down. "
                                    f"Add cone / surface / transparency or notes - or leave card 2 empty when you search with a photo.")
    tv = Zq["text"][0][0] if "text" in Zq else None
    fv = Zq["formula"][0][0] if "formula" in Zq else None
    return tv, fv, info, cleaned, len(ox), len(ings)


def query_object(qid: str, index: int) -> dict:
    """the selected query object: colour regions (label map saved for the page's overlay) + the six attributes detected
    by Grounding DINO (the segmenter's model, shared) with the validation summary of each attribute."""
    q = load_query(qid)
    if q is None:
        raise ValueError("unknown upload")
    o = next((x for x in q["objects"] if x["index"] == int(index)), None)
    if o is None:
        raise ValueError("unknown object")
    osr = S["objects"]
    with LOCK:
        qf = osr.encode((qid, o["index"]), o["rgba"])
        lp = QDIR / qid / f"regions_{o['index']}.png"
        if not lp.exists():
            Image.fromarray(np.asarray(qf["details"]["labels"]).astype(np.uint8)).save(lp)
        det = None
        if S.get("qattr") is not None:
            key = ("qattr", qid, o["index"])
            det = q["vec"].get(key)
            if det is None:
                det = S["qattr"](o["rgba"], qf["details"].get("attr_full", {}))
                q["vec"][key] = det
    reg = qf["details"]["regions"]
    return {"index": o["index"], "regions": osr.region_list(qf["details"]), "region_size": reg.get("size"),
            "labels": f"/q/{qid}/regions_{o['index']}.png", "object_image": f"/q/{qid}/obj_{o['index']}.png",
            "attributes": qf["details"]["attributes"], "dino": det,
            "dino_available": S.get("qattr") is not None}


def run_object_search(p: dict) -> dict:
    """Balanced / shape / glaze / colour search over the segmented OBJECTS (exact object images)."""
    from mm.balanced import colour_query  # noqa: F401  (typed colours are converted inside ObjectSearch)
    eng, osr = S["engine"], S["objects"]
    t0 = time.time()
    q = load_query(p["qid"]) if p.get("qid") else None
    qf, oi, palette = None, None, []
    if q is not None and q["objects"] and p.get("use_photo", True):
        oi = int(p.get("object") or (q["meta"]["default_selection"] or [q["objects"][0]["index"]])[0])
        o = next((x for x in q["objects"] if x["index"] == oi), q["objects"][0])
        with LOCK:
            qf = osr.encode((p["qid"], o["index"]), o["rgba"])
        palette = o["palette"]
    tv, fv, info, cleaned, n_ox, n_ing = text_formula(p)
    colours = p.get("colours") or []
    if not palette and colours:
        palette = [[c["hex"], 1.0] for c in colours[:1]]
    qattr = eng.query_attributes(cleaned, palette)
    res = osr.search(p, qf, tv, fv, qattr, q_attrs=p.get("query_attrs") or {})
    if qf is not None:
        res["query_object"] = {"index": oi, "regions": osr.region_list(qf["details"]), "attributes": qf["details"]["attributes"]}
    res.update({"query": info, "palette": palette, "seconds": round(time.time() - t0, 2)})
    dup = (q or {}).get("meta", {}).get("database_duplicate") if qf is not None else None
    res["database_photo"] = dup
    mods = (["image"] if qf is not None else []) + (["colours"] if colours else []) + (["text"] if tv is not None else []) + (["formula"] if fv is not None else [])
    entry = {"entry": uuid.uuid4().hex[:8], "time": time.strftime("%Y-%m-%d %H:%M:%S"), "qid": p.get("qid") or "",
             "db_photo": (dup or {}).get("recipe_id"), "search": "objects · " + res.get("mode", ""),
             "modalities": mods, "text": info.get("text_used", ""), "oxides": n_ox, "ingredients": n_ing,
             "filters": res.get("filters_applied", []), "gallery": res.get("gallery"),
             "top10": [r["object_id"] for r in res.get("results", [])[:10]], "auto": res.get("auto"), "true": res.get("true_recipe"),
             "features": (res.get("features") or {}).get("per_result"), "ticked": (res.get("features") or {}).get("ticked"),
             "query_attrs": (res.get("features") or {}).get("query"),
             "manual": None, "seconds": res["seconds"]}
    ev = res.get("evaluation")
    if ev:                                       # evaluation of this search (shape / glaze / colour / strict, weighted nDCG)
        keep = [f"{m}@{k}" for m in ("Hit", "P", "Recall") for k in (3, 5, 10, 15, 20)] + ["relevant_in_gallery", "unknown_top20"]
        entry["eval"] = {"rules": ev.get("rules"), "values": ev.get("query_values"), "weights": ev.get("weights"),
                         "metrics": {c: {k: m.get(k) for k in keep} for c, m in (ev.get("metrics") or {}).items()},
                         "ndcg": ev.get("ndcg"), "true_recall": ev.get("true_recall"), "mode": res.get("mode_label"),
                         "priority": (res.get("priority") or {}).get("used")}
    with LOCK:
        lg = read_json(LOG_FILE, []) or []
        lg.append(entry)
        write_json(LOG_FILE, lg)
    if p.get("qid"):
        write_json(QDIR / p["qid"] / f"search_{entry['entry']}.json", {"request": p, "result": res})
    res["entry"] = entry["entry"]
    return res


def mark(p: dict) -> dict:
    marks = {str(k): bool(v) for k, v in (p.get("marks") or {}).items()}
    with LOCK:
        lg = read_json(LOG_FILE, []) or []
        for e in lg:
            if e["entry"] == p.get("entry"):
                rel = [marks.get(i, False) for i in e["top10"]]
                e["manual"] = {**{f"Hit@{k}": any(rel[:k]) for k in (3, 5, 10)},
                               **{f"P@{k}": round(sum(rel[:k]) / k, 3) for k in (3, 5, 10)}, "marked": marks}
        write_json(LOG_FILE, lg)
    return summary()


def summary() -> dict:
    """session statistics over the log (never rewritten). Hit Rate@K = share of searches with >= 1 relevant result in the
    top K (it was called "Recall@K" before 2026-10-20); P@K = share of the top K that is relevant; Recall@K = share of ALL
    relevant objects that reach the top K - only meaningful for the true recipe (its objects are the complete set)."""
    lg = read_json(LOG_FILE, []) or []
    use = [e for e in lg if not e.get("db_photo")]
    out = {"entries": lg[-200:][::-1], "n": len(lg), "excluded_db_photo": len(lg) - len(use)}
    for src in ("auto", "manual", "true"):
        rows = [e[src] for e in use if e.get(src) and e[src].get("Hit@3") is not None]
        out[src] = {"n": len(rows), **{f"HitRate@{k}": (round(sum(r[f"Hit@{k}"] for r in rows) / len(rows), 3) if rows else None)
                                        for k in (3, 5, 10)}}
    feats = {}
    for e in use:
        if not e.get("features"):
            continue
        for f, rel in e["features"].items():
            feats.setdefault(f, []).append([bool(x) for x in rel])
    out["features"] = {f: {"n": len(R), **{f"HitRate@{k}": round(sum(any(r[:k]) for r in R) / len(R), 3) for k in (3, 5, 10, 15, 20)},
                           **{f"P@{k}": round(sum(sum(r[:k]) / k for r in R) / len(R), 3) for k in (3, 5, 10, 15, 20)}}
                       for f, R in feats.items()}
    # evaluation of every search since 2026-10-20 (shape / glaze / colour / strict), weighted nDCG and true-recipe Recall@K
    ev, nd, tr = {}, [], []
    for e in use:
        x = e.get("eval")
        if not x:
            continue
        for c, m in (x.get("metrics") or {}).items():
            ev.setdefault(c, []).append(m)
        if (x.get("ndcg") or {}).get("nDCG@10") is not None:
            nd.append(x["ndcg"])
        if x.get("true_recall"):
            tr.append(x["true_recall"])
    KS = (3, 5, 10, 15, 20)
    mean = lambda v: round(sum(v) / len(v), 3) if v else None  # noqa: E731
    out["eval"] = {c: {"n": len(M), **{f"HitRate@{k}": mean([bool(m.get(f"Hit@{k}")) for m in M]) for k in KS},
                       **{f"P@{k}": mean([m.get(f"P@{k}") or 0 for m in M]) for k in KS}} for c, M in ev.items()}
    out["ndcg"] = {"n": len(nd), **{f"nDCG@{k}": mean([d.get(f"nDCG@{k}") for d in nd if d.get(f"nDCG@{k}") is not None]) for k in KS}}
    out["true_recall"] = {"n": len(tr), **{f"Recall@{k}": mean([d.get(f"Recall@{k}") for d in tr]) for k in KS}}
    return out


def log_csv() -> bytes:
    lg = read_json(LOG_FILE, []) or []
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "entry", "qid", "database_photo_of", "modalities", "gallery", "filters", "auto_Hit@3", "auto_Hit@5", "auto_Hit@10",
                "auto_P@5", "manual_Hit@3", "manual_Hit@5", "manual_Hit@10", "manual_P@5", "true_id", "true_rank",
                "top10", "text", "eval_mode", "eval_priority"]
               + [f"{c}_{m}@{k}" for c in ("shape", "glaze", "colour", "strict") for m in ("Hit", "P") for k in (3, 10, 20)]
               + ["eval_nDCG@10", "true_Recall@10", "eval_rules"])
    for e in lg:
        a, m, t = e.get("auto") or {}, e.get("manual") or {}, e.get("true") or {}
        x = e.get("eval") or {}
        M = x.get("metrics") or {}
        w.writerow([e["time"], e["entry"], e["qid"], e.get("db_photo") or "", "+".join(e["modalities"]), e["gallery"], "; ".join(e["filters"]),
                    a.get("Hit@3"), a.get("Hit@5"), a.get("Hit@10"), a.get("P@5"), m.get("Hit@3"), m.get("Hit@5"),
                    m.get("Hit@10"), m.get("P@5"), t.get("id"), t.get("rank"), " ".join(e["top10"]), e["text"],
                    x.get("mode", ""), json.dumps(x.get("priority")) if x.get("priority") else ""]
                   + [(M.get(c) or {}).get(f"{mm}@{k}", "") for c in ("shape", "glaze", "colour", "strict") for mm in ("Hit", "P") for k in (3, 10, 20)]
                   + [(x.get("ndcg") or {}).get("nDCG@10", ""), (x.get("true_recall") or {}).get("Recall@10", ""),
                      "; ".join(f"{c}: {r}" for c, r in (x.get("rules") or {}).items())])
    return ("﻿" + buf.getvalue()).encode("utf-8")


# ---------------------------------------------------------------- HTTP
class H(BaseHTTPRequestHandler):
    server_version = "CG2GlazeSearch/1.0"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode("utf-8"))

    def _file(self, p: Path, ctype):
        if not p.is_file():
            return self._send(404, b"not found", "text/plain")
        self._send(200, p.read_bytes(), ctype)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._file(PAGE, "text/html; charset=utf-8")
        if path == "/api/meta":
            eng = S["engine"]
            return self._json({**eng.options(), "ready": True, "oxide_only": read_json(WORK / "app" / "oxide_only_choice.json", {}),
                               "verify": S.get("verify"), "timings": S.get("timings"),
                               "object_search": S["objects"].info() if S.get("objects") else None})
        if path == "/api/color_words":
            from urllib.parse import parse_qs, urlparse
            from mm.colorlab import search_words
            qs = parse_qs(urlparse(self.path).query).get("q", [""])[0]
            return self._json({"results": search_words(qs, 10) if qs.strip() else []})
        m = re.fullmatch(r"/obj/(\d+_\d+)\.jpg", path)
        if m:
            from object_search import object_image
            return self._file(object_image(m.group(1)), "image/jpeg")
        if path == "/api/log":
            return self._json(summary())
        if path == "/api/related_colours":
            from urllib.parse import parse_qs, urlparse
            from mm.colorlab import related
            qs = parse_qs(urlparse(self.path).query)
            hx = qs.get("hex", [""])[0]
            if not re.fullmatch(r"#?[0-9A-Fa-f]{6}", hx):
                return self._json({"error": "hex like #A0522D"}, 400)
            return self._json(related("#" + hx.lstrip("#"), qs.get("mode", ["both"])[0]))
        if path == "/api/log.csv":
            return self._send(200, log_csv(), "text/csv; charset=utf-8",
                              {"Content-Disposition": "attachment; filename=glaze_search_log.csv"})
        m = re.fullmatch(r"/thumb/(\d+)\.jpg", path)
        if m:
            return self._file(SITE / "thumbs" / f"{m.group(1)}.jpg", "image/jpeg")
        m = re.fullmatch(r"/photo/(\d+)\.jpg", path)
        if m:
            from object_search import photo_file          # 7k photo, or a bonus photo (.jpg / .png)
            p = photo_file(m.group(1)) or PHOTOS / f"{m.group(1)}.jpg"
            return self._file(p, "image/png" if p.suffix.lower() == ".png" else "image/jpeg")
        m = re.fullmatch(r"/q/([0-9a-f-]{20,24})/([a-z0-9_]+\.(?:jpg|png))", path)
        if m:
            return self._file(QDIR / m.group(1) / m.group(2), "image/png" if m.group(2).endswith(".png") else "image/jpeg")
        if path.startswith("/lib/"):                       # static reference library (read-only, files inside it only)
            from urllib.parse import unquote
            p = (LIBRARY / unquote(path[5:])).resolve()
            if LIBRARY.resolve() in p.parents and p.suffix.lower() in LIB_TYPES:
                return self._file(p, LIB_TYPES[p.suffix.lower()])
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD * 1.4:
            return self._json({"error": "request too large (max 30 MB image)"}, 413)
        try:
            p = json.loads(self.rfile.read(n) or b"{}")
            path = self.path.split("?")[0]
            if path == "/api/segment":
                if p.get("url"):
                    data, src = fetch_url(p["url"]), "link"
                else:
                    b64 = p.get("image", "")
                    data, src = base64.b64decode(b64.split(",", 1)[-1]), p.get("source", "upload")
                from mm.query import database_duplicate
                return self._json(segment(decode_image(data), src, database_duplicate(data)))
            if path == "/api/search":
                return self._json(run_search(p))
            if path == "/api/search_objects":
                if not S.get("objects"):
                    return self._json({"error": "object search is not available (run run_attributes.py first)"}, 400)
                return self._json(run_object_search(p))
            if path == "/api/query_object":
                if not S.get("objects"):
                    return self._json({"error": "object search is not available (run run_attributes.py first)"}, 400)
                return self._json(query_object(p.get("qid", ""), int(p.get("object", 0))))
            if path == "/api/parse_colours":
                from mm.colorlab import parse_phrase
                return self._json(parse_phrase(p.get("text", "")))
            if path == "/api/mark":
                return self._json(mark(p))
            return self._json({"error": "unknown endpoint"}, 404)
        except Exception as e:  # noqa: BLE001
            log().exception("[app] request failed")
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 in the online Space (Docker); this PC only by default")
    args = ap.parse_args()
    setup_logging(WORK / "logs" / "app.log")
    url = f"http://127.0.0.1:{args.port}/"
    import socket
    import webbrowser
    with socket.socket() as sk:
        sk.settimeout(0.5)
        if sk.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"Glaze search is already running: {url}")
            if not args.no_browser:
                webbrowser.open(url)
            return 0
    busy = other_gpu_jobs()
    if busy:
        print("Another GPU job is running (only one at a time):", busy)
        return 2
    t0 = time.time()
    print("Loading the retrieval model and database (about 10 s)...", flush=True)
    from mm.query import Engine, Segmenter
    eng = Engine()
    S["verify"] = eng.verify()
    bad = [m for m, v in S["verify"].items() if v["min_cos"] < 0.999 or not v["mask_equal"]]
    if bad:
        print("ERROR: rebuilt embeddings differ from results/db_embeddings.npz for", bad)
        return 3
    if eng.oxide_head is None:
        print("Fitting the oxide-only formula head (train / validation recipes, about 10 s)...", flush=True)
        eng.fit_oxide_head()
    S["engine"] = eng
    t1 = time.time()
    print("Loading segmentation models (Grounding DINO, SAM 2.1, OpenCLIP)...", flush=True)
    S["seg"] = Segmenter()
    # warm-up on one database photo so the first real query is not slower
    warm = next(iter(sorted(PHOTOS.glob("*.jpg"))), None)
    t2 = time.time()
    if warm is not None:
        rgb = decode_image(warm.read_bytes())
        res = S["seg"](rgb)
        from mm.query import default_selection
        objs = [o for o in res["objects"] if o["index"] in default_selection(res["objects"])]
        if objs:
            eng.image_vector(objs)
    import torch
    t3 = time.time()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from object_search import ObjectSearch
        print("Loading the object index (balanced search, colour regions, attributes)...", flush=True)
        S["objects"] = ObjectSearch(eng, clip=S["seg"].clip)
    except Exception as exc:  # noqa: BLE001
        S["objects"] = None
        print(f"Object search not available ({type(exc).__name__}: {exc}); recipe search still works.", flush=True)
    S["qattr"] = None
    v2, v3 = WORK / "attributes_v2", WORK / "attributes_v3"
    if S.get("objects") and (v3 / "attr_v3_models.joblib").exists() and (v2 / "dino_calibration.json").exists():
        try:          # v3: models trained on Claude's labels, top-3 candidates per attribute
            from mm.attr_v3 import QueryAttributesV3
            S["qattr"] = QueryAttributesV3(v2, v3, dino_model=S["seg"].seg.dino, dino_proc=S["seg"].seg.dino_proc)
            print("Grounding DINO attribute detection v3 ready (six attributes, top-3 candidates, trained on Claude's labels).", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Attribute detection v3 not available ({type(exc).__name__}: {exc}); trying v2.", flush=True)
    if S["qattr"] is None and S.get("objects") and (v2 / "dino_heads.joblib").exists() and (v2 / "dino_calibration.json").exists():
        try:
            from mm.dino_predict import QueryAttributes
            S["qattr"] = QueryAttributes(v2, dino_model=S["seg"].seg.dino, dino_proc=S["seg"].seg.dino_proc)
            print("Grounding DINO attribute detection ready (six attributes, validated against Claude's check).", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Attribute detection not available ({type(exc).__name__}: {exc}).", flush=True)
    S["timings"] = {"engine_s": round(t1 - t0, 1), "segmenter_s": round(t2 - t1, 1), "warmup_s": round(t3 - t2, 1),
                    "object_index_s": round(time.time() - t3, 1),
                    "gpu_GB": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0}
    QDIR.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), H)
    log().info(f"[app] ready at {url} | {S['timings']}")
    print("=" * 60, f"\nGlaze search is ready: {url}\nKeep this window open while you use the page; close it to stop.\n" + "=" * 60, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
