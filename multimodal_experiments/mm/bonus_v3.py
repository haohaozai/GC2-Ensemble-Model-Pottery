"""The bonus_image set (others_image -> bonus_image_claude_process, 125 photos) in Glaze Search and in attributes v3.

For every bonus object still in use (object_location = smaller_object, review status not "not_needed"):
  search row   the same encoder as an uploaded photo (mm/object_query.ObjectEncoder, checked against stored 7k index rows:
               shape / silhouette identical, appearance cosine >= 0.987): appearance, shape, silhouette, attribute vector,
               colour regions; ColorThief palette from the bonus full_auto_tags.csv
  recipe       the bonus recipes are not in the recipe bank (it holds the 7,063 recipes with a 7k photo): their text and
               formula embeddings are built exactly as for the bank - Glazy export fields -> cleaned tags + description ->
               TF-IDF -> text head; oxides (UMF) + ingredients -> formula head (only when both exist, as in the bank)
  attributes   Grounding DINO evidence with the full-run prompt sets + the earlier image heads (live) -> the v3 models ->
               labels and top-3 candidates (attribute_predictions_v3_bonus.json)
Outputs in work/attributes_v3/: bonus_index.npz, bonus_recipes.json, bonus_attr_v1.json, attribute_predictions_v3_bonus.json.
Nothing in the 7k index, the 7k predictions or the bonus segmentation folder is changed.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from . import balanced as B
from . import data as D
from . import dino_attr as DA
from . import dino_calib as DC
from .common import BASE, GLAZY_EXPORT, ING7K, read_csv

BONUS = BASE / "bonus_image_claude_process"
NOT_NEEDED = "does_not_needed"
SG = {"glossy": 0, "satin": 1, "matte": 2}
TG = {"clear": 0, "opaque": 1}


def bonus_file(sub: str, name: str) -> Path:
    p = BONUS / sub / name
    return p if p.exists() else BONUS / NOT_NEEDED / sub / name


def bonus_objects() -> list[tuple[str, str]]:
    """(object id, recipe id) of the bonus objects in use"""
    p = BONUS / "review" / "object_review.csv"
    rows = read_csv(p) if p.exists() else read_csv(BONUS / "full_auto_tags.csv")
    out = []
    for r in rows:
        oid = r.get("object_id")
        if not oid or r.get("review_status") == "not_needed":
            continue
        if (BONUS / "smaller_object" / f"{oid}.jpg").exists():
            out.append((oid, Path(r["parent_filename"]).stem))
    return out


def recipe_embeddings(eng, rids: list[str]) -> dict:
    """recipe id -> {meta, text z, formula z} built like the recipe bank"""
    ex = {e["ID"]: e for e in read_csv(GLAZY_EXPORT)}
    header = list(next(iter(ex.values())).keys())
    oxides = D.oxide_columns(header)
    ing = {}
    for x in read_csv(ING7K):
        if x["recipe_id"] in rids:
            ing.setdefault(x["recipe_id"], []).append(x)
    from . import text_feats as TF
    out = {}
    for rid in rids:
        e = ex.get(rid)
        if e is None:
            continue
        row = {**e, "recipe_id": rid}
        tg = D.clean_tags([row])[rid]
        desc_free = D.strip_html(e.get("Description", ""))[:1500]
        text = TF.texts_for(eng.cfg["text"], [{"desc_tags": tg["desc_tags"], "desc_free": desc_free}])[0]
        ox = {c: float(e[c]) for c in oxides if (e.get(c) or "").strip() and float(e[c]) > 0}
        ings = [[x["Ingredient ID"], x["Ingredient Name"], float(x["Percentage"] or 0), x["Additional"] == "Yes"] for x in ing.get(rid, [])]
        tv = eng.text_vector(text) if text else None
        umf_vec = ing_vec = None
        if len(ox) >= 3 and ings:
            umf_vec, ing_vec, _ = eng.formula_parts(ox, ings)
            if ing_vec is None:
                umf_vec = None
        Z, _ = eng.embed_query(None, tv, umf_vec, ing_vec)
        out[rid] = {"meta": {"id": rid, "name": e.get("Name", ""), "subtype": tg["subtype"], "cone_lo": tg["cone_lo"],
                             "cone_hi": tg["cone_hi"], "surface": tg["surface"], "transparency": tg["transparency"],
                             "atmospheres": tg["atmospheres"], "url": f"https://glazy.org/recipes/{rid}", "split": "bonus",
                             "image_set": "bonus_image"},
                    "cone": [int(tg.get("cone_lo_i") or -1), int(tg.get("cone_hi_i") or -1)],
                    "surface_group": D.SURFACE_GROUP.get((tg["surface"] or "").lower(), ""),
                    "transparency_group": D.TRANSP_GROUP.get((tg["transparency"] or "").lower(), ""),
                    "form_text": (e.get("Name", "") + " " + desc_free).lower(),
                    "text": Z["text"][0][0] if "text" in Z else None, "formula": Z["formula"][0][0] if "formula" in Z else None}
    return out


def colorthief(oids: set[str]) -> dict:
    from .colorlab import hex2lab
    out = {}
    with (BONUS / "full_auto_tags.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("object_id") in oids:
                lab, pc = np.zeros((3, 3), np.float32), np.zeros(3, np.float32)
                for i in range(3):
                    h = r.get(f"color{i + 1}_hex") or ""
                    if h:
                        lab[i] = hex2lab(h)
                        pc[i] = float(r.get(f"color{i + 1}_pct") or 0) / 100
                out[r["object_id"]] = (lab, pc)
    return out


def pad_aspect(img: np.ndarray, off, max_ratio=1.6):
    """pad a cut-out with the neutral background (as around every cut-out) so its long side <= max_ratio x short side;
    the offset is moved so evidence boxes still map to object-image pixels."""
    h, w = img.shape[:2]
    if max(h, w) <= max_ratio * min(h, w):
        return img, off, 0
    if w > h:
        add = int(np.ceil(w / max_ratio)) - h
        top = add // 2
        out = np.full((h + add, w, 3), DA.BG, np.uint8)
        out[top:top + h] = img
        return out, (off[0], off[1] - top, off[2], off[3] + add - top), add
    add = int(np.ceil(h / max_ratio)) - w
    left = add // 2
    out = np.full((h, w + add, 3), DA.BG, np.uint8)
    out[:, left:left + w] = img
    return out, (off[0] - left, off[1], off[2] + add - left, off[3]), add


def form_hint_row(text: str) -> tuple[np.ndarray, str]:
    import re
    cls = DC.CLASSES["form"]
    h = np.zeros(len(cls), np.float32)
    found = [w for w in DC.FORM_WORDS if re.search(r"\b" + w + r"s?\b", text)]
    for w in found:
        h[cls.index(DC.FORM_WORDS[w])] = 1
    return h, ", ".join(f"'{w}'" for w in found)


def build(v1: Path, v2: Path, v3: Path, log=print, limit: int = 0) -> dict:
    import joblib
    import torch
    from PIL import Image
    from . import attr_v3 as A3
    from .dino_predict import QueryAttributes
    from .object_query import ObjectEncoder
    from .query import Engine
    t0 = time.time()
    objs = bonus_objects()
    if limit:                               # ramp test: first N objects, nothing written
        objs = objs[:limit]
    rids = sorted({r for _, r in objs})
    log(f"[bonus] {len(objs)} objects in use of {len(rids)} photos / recipes")
    guard = DA.GpuGuard(vram_cap_gb=5.0, duty_pause=0.02)
    eng = Engine()
    enc = ObjectEncoder(v1, eng)
    da = DA.DinoAttr(guard=guard)
    spec = DC.read_cal(v2)["dino_spec"]
    models = joblib.load(v3 / "attr_v3_models.joblib")
    recs = recipe_embeddings(eng, rids)
    log(f"[bonus] recipes embedded: {len(recs)} | text {sum(r['text'] is not None for r in recs.values())} | "
        f"formula {sum(r['formula'] is not None for r in recs.values())} | {time.time() - t0:.0f} s")
    I0 = np.load(v1 / "balanced_index.npz", allow_pickle=False)
    forms = I0["forms"].tolist()
    dim = I0["app"].shape[1]
    ct = colorthief({o for o, _ in objs})
    rows = {k: [] for k in ("ids", "recipe_id", "split", "app", "m_app", "text", "m_text", "formula", "m_formula", "geom", "sil", "attr",
                            "attr_pred", "reg_lab", "reg_pos", "reg_area", "reg_auto", "reg_human", "reg_valid", "surf", "transp", "eff",
                            "eff_has", "cone", "form", "ct_lab", "ct_pct")}
    dino, v1full, preds_v1, offs, hints, FH = [], [], {}, [], [], []
    for n, (oid, rid) in enumerate(objs, 1):
        rgba = np.asarray(Image.open(bonus_file("smaller_object_rgba_raw", f"{oid}.png")).convert("RGBA"))
        with torch.no_grad():
            qf = enc.encode(rgba)
        img, off = DA.cutout(rgba)
        try:
            r = da.score_spec([img], spec)[0]
        except RuntimeError as exc:          # very long, thin cut-out: too few image tokens for the 900 queries
            img, off, n_pad = pad_aspect(img, off)
            log(f"[bonus] {oid}: {str(exc)[:60]} -> cut-out padded with the neutral background to aspect <= 1.6 ({n_pad} px)")
            r = da.score_spec([img], spec)[0]
        dino.append(r)
        offs.append(off)
        v1full.append(qf["details"]["attr_full"])
        preds_v1[oid] = qf["details"]["attributes"]
        rec = recs.get(rid) or {}
        h, ht = form_hint_row(rec.get("form_text", ""))
        FH.append(h)
        hints.append(ht)
        top = (qf["details"]["attributes"].get("form") or [])
        lab, pc = ct.get(oid, (np.zeros((3, 3), np.float32), np.zeros(3, np.float32)))
        vals = {"ids": oid, "recipe_id": rid, "split": "bonus", "app": qf["app"].astype(np.float32), "m_app": True,
                "text": rec.get("text") if rec.get("text") is not None else np.zeros(dim, np.float32), "m_text": rec.get("text") is not None,
                "formula": rec.get("formula") if rec.get("formula") is not None else np.zeros(dim, np.float32),
                "m_formula": rec.get("formula") is not None, "geom": qf["geom"], "sil": qf["sil"].astype(np.float32),
                "attr": qf["attr"], "attr_pred": qf["attr"], "surf": SG.get(rec.get("surface_group", ""), -1),
                "transp": TG.get(rec.get("transparency_group", ""), -1), "eff": np.zeros(I0["eff"].shape[1], bool), "eff_has": False,
                "cone": np.array(rec.get("cone", [-1, -1])), "form": forms.index(top[0][0]) if top and top[0][1] >= 0.5 and top[0][0] in forms else -1,
                "ct_lab": lab, "ct_pct": pc}
        for k, a in zip(("reg_lab", "reg_pos", "reg_area", "reg_auto", "reg_human", "reg_valid"), qf["reg"]):
            vals[k] = a
        for k in rows:
            rows[k].append(vals[k])
        guard.after_object()
        if n % 50 == 0 or n == len(objs):
            st = guard.gpu_state()
            log(f"[bonus] {n}/{len(objs)} objects | {(time.time() - t0) / 60:.1f} min | peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
                f"GPU {st.get('temp_C', '?')} C {st.get('vram_used_MB', '?')} MB")
    I = {k: np.stack([np.asarray(x) for x in v]) for k, v in rows.items()}
    if limit:
        return {"ramp_test_objects": len(objs), "s_per_object": round((time.time() - t0) / len(objs), 2),
                "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "text": int(I["m_text"].sum()), "formula": int(I["m_formula"].sum())}
    I["forms"] = I0["forms"]
    for k in I0.files:                      # same dtypes as the 7k index
        if k in I and I0[k].dtype.kind != "U":
            I[k] = I[k].astype(I0[k].dtype)
    np.savez_compressed(v3 / "bonus_index.npz", **I)
    # attributes v3 of the bonus objects
    Dd = {f"{a}|{v}|{f}": np.stack([d[(a, v)][f] for d in dino]) for (a, v) in dino[0] for f in dino[0][(a, v)]}
    Dd["offset"] = np.array(offs)
    V = {}
    for full in v1full:
        for task, row in QueryAttributes.v1_vectors(full).items():
            V.setdefault(task, []).append(row[0])
    V = {k: np.stack(v) for k, v in V.items()}
    lab_rows = A3.label_rows(models, Dd, V, np.stack(FH), hints)
    rep = json.loads((v3 / "attr_v3_report.json").read_text(encoding="utf-8"))
    pj = {"version": "attributes_v3", "image_set": "bonus_image", "created": time.strftime("%Y-%m-%d %H:%M"),
          "validation": A3.validation_summary(rep), "objects": {o: r for (o, _), r in zip(objs, lab_rows)}}
    (v3 / "attribute_predictions_v3_bonus.json").write_text(json.dumps(pj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (v3 / "bonus_attr_v1.json").write_text(json.dumps({"objects": preds_v1}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (v3 / "bonus_recipes.json").write_text(json.dumps({rid: {k: v for k, v in r.items() if k in ("meta", "cone", "surface_group", "transparency_group")}
                                                       for rid, r in recs.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"objects": len(objs), "recipes": len(recs), "with_text": int(I["m_text"].sum()), "with_formula": int(I["m_formula"].sum()),
            "with_regions": int(I["reg_valid"].any(1).sum()), "minutes": round((time.time() - t0) / 60, 1),
            "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
