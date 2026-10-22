"""Stage F: attribute re-tagging (claude_v3.2-retag) of a finished claude_v3 output folder.

Segmentation, object files, colours and recipe links are NOT recomputed. For every object this updates:
  Surface / Transparency   CSV value kept and marked "(CSV)"; when the CSV is blank, a coarse group is
                           auto-labelled from the image (+ recipe text) only above a precision-calibrated
                           threshold, otherwise "Unknown (CSV blank; ...)"
  Glaze effect / Defect    never blank: CSV subtype / CSV description / visual / "Possible" (low confidence) /
                           "None visible (checked)" / "Unknown (...)", each with its source
  Form / Shape             objects cut by one photo edge no longer get a form or silhouette shape
and rewrites full_auto_tags.csv (new columns, old ones kept), tag_json/<object>.json (new keys, v3.1 keys
kept) and detailed_preview/<object>.jpg. Rules: glazeprep/retag.py.

Phases (default: all):
  embed  CLIP embeddings (object + texture view, from smaller_object_rgba_raw) and highlight statistics for
         every object -> evaluation/retag/object_features.npz (cached; GPU if available)
  fit    image classifier for Surface / Transparency groups (logistic regression on frozen CLIP features,
         trained on objects whose recipe HAS a CSV value; dev / test split by md5(recipe_id)); thresholds chosen
         on dev for a target precision; held-out test metrics -> evaluation/retag/surface_transparency_eval.*
  apply  backups (first run only), new fields, CSV, tag_json, cards, report, run_manifest entry

usage: python scripts/16_retag_attributes.py [--output DIR] [--phases embed,fit,apply] [--no-cards]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glazeprep.config  # noqa: F401,E402  (D: model caches)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from glazeprep import retag as R  # noqa: E402

from glazeprep.config import CG2_DATA as BASE  # noqa: E402  (env CG2_DATA / CG2_ROOT; default: next to the project)

CODE = Path(__file__).resolve().parents[1]
CONFIG_OUT = CODE / "configs" / "retag_thresholds.json"
TARGET_PRECISION = {"surface": 0.80, "transparency": 0.85}
POSSIBLE_QUANTILE = 0.97
FEATURE_VERSION = "v2: v3.1 views with the 4 % margin rebuilt, float32"
NEW_COLUMNS = ["glazy_url", "surface", "surface_source", "surface_evidence", "transparency", "transparency_source",
               "transparency_evidence", "glaze_effect", "glaze_effect_labels", "glaze_effect_source",
               "glaze_effect_evidence", "defect", "defect_labels", "defect_source", "defect_evidence",
               "object_form", "shape", "object_truncated_by_frame_v32", "frame_contact", "form_note", "tag_version"]
# ColorThief colours (scripts/17_colorthief_colors.py) and the review app's decisions (scripts/18_review_app.py)
COLOR_COLUMNS = [f"color{i}_{k}" for i in (1, 2, 3) for k in ("name", "hex", "pct")] + ["top3_colors_colorthief", "color_method"]
REVIEW_COLUMNS = ["review_status", "review_note", "reviewed_at", "object_location"]
NOT_NEEDED_DIR = "does_not_needed"
REVIEW_CSV = Path("review") / "object_review.csv"
SURFACE_PROMPTS = {"G": ["a glossy ceramic glaze with sharp mirror-like reflections", "a shiny glazed ceramic surface with bright highlights"],
                   "S": ["a satin ceramic glaze with a soft silky sheen", "a semi-matte ceramic glaze with soft broad highlights"],
                   "M": ["a matte ceramic glaze with no reflections", "a dry, chalky, stony matte ceramic surface"]}
TRANSPARENCY_PROMPTS = {"T": ["a transparent clear glaze showing the clay and carving underneath", "a translucent glaze like jade or frosted glass"],
                        "O": ["an opaque glaze that completely hides the clay body", "a solid opaque coloured ceramic glaze"]}


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    atomic_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def is_dev(rid) -> bool:
    return int(hashlib.md5(str(rid).encode()).hexdigest(), 16) % 2 == 0


# --------------------------------------------------------------------------- inventory
def inventory(out: Path):
    """Complete images in meta/ order; returns (objects [(oid, meta)], no-object metas)."""
    objs, empty = [], []
    metas = sorted((out / "meta").glob("*.json"), key=lambda p: (len(p.stem), p.stem))
    for mp in metas:
        rec = json.loads(mp.read_text(encoding="utf-8"))
        if not rec.get("complete"):
            continue
        if rec.get("objects"):
            objs += [(oid, rec) for oid in rec["objects"]]
        else:
            empty.append(rec)
    return objs, empty


def obj_file(out: Path, sub: str, name: str) -> Path:
    """Path of an object file; objects marked "not needed" live in does_not_needed/<sub>/."""
    p = out / sub / name
    if p.exists():
        return p
    q = out / NOT_NEEDED_DIR / sub / name
    return q if q.exists() else p


def load_tag(out: Path, oid: str):
    return json.loads(obj_file(out, "tag_json", f"{oid}.json").read_text(encoding="utf-8"))


def color_fields(t: dict) -> dict:
    pal = t.get("colorthief_palette") or []
    out = {}
    for i in range(3):
        c = pal[i] if i < len(pal) else {}
        out[f"color{i + 1}_name"] = c.get("name", "")
        out[f"color{i + 1}_hex"] = c.get("hex", "")
        out[f"color{i + 1}_pct"] = round(float(c["fraction"]) * 100, 1) if c else ""
    out["top3_colors_colorthief"] = "; ".join(f"{c['name']} {c['hex']} {float(c['fraction']) * 100:.1f}%" for c in pal[:3])
    out["color_method"] = ("ColorThief palette (colorthief 0.2.1) on the object cut-out; % = share of object pixels; "
                           "names = nearest reference shade (CIEDE2000)") if pal else ""
    return out


def load_review(out: Path) -> dict:
    p = out / REVIEW_CSV
    if not p.exists():
        return {}
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return {r["object_id"]: r for r in csv.DictReader(fh) if r.get("object_id")}


# --------------------------------------------------------------------------- phase: embed
def highlight_features(rgb: np.ndarray, mask: np.ndarray):
    """Gloss cues inside the mask at <= 640 px: highlight area, value statistics, saturation, roughness."""
    h, w = mask.shape
    s = 640 / max(h, w)
    if s < 1:
        im = Image.fromarray(rgb).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        mk = Image.fromarray(mask.astype(np.uint8) * 255).resize(im.size, Image.NEAREST)
        rgb, mask = np.asarray(im), np.asarray(mk) > 127
    if mask.sum() < 30:
        return np.zeros(6, np.float32)
    a = rgb.astype(np.float32) / 255
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    v = mx[mask]
    vmed = float(np.median(v))
    hl = ((mx >= 0.92) & (sat <= 0.25) & (mx - vmed >= 0.25) & mask).sum() / mask.sum()
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = np.abs(4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1] - lum[1:-1, :-2] - lum[1:-1, 2:])
    inner = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return np.array([hl, vmed, float(np.percentile(v, 99.5)), float(np.std(v)), float(np.median(sat[mask])),
                     float(lap[inner].mean()) if inner.any() else 0.0], np.float32)


def phase_embed(out: Path, objs, device, workers, batch):
    import torch
    from glazeprep.ceramic_clip import CeramicClip
    from glazeprep.ceramic_tagger import CeramicTagger
    cache = out / "evaluation" / "retag" / "object_features.npz"
    have, old = {}, None
    if cache.exists():
        # read into memory and close: Windows cannot replace a file that is still open
        with np.load(cache, allow_pickle=False) as z:
            if "version" in z.files and str(z["version"]) == FEATURE_VERSION:
                old = {k: z[k] for k in ("ids", "e_obj", "e_tex", "hf")}
                have = {oid: i for i, oid in enumerate(old["ids"].tolist())}
            else:
                print("[embed] cached features were made with another feature version; recomputing all", flush=True)
    if not [oid for oid, _ in objs if oid not in have]:
        print(f"[embed] all {len(have)} objects cached", flush=True)
        return
    todo = [oid for oid, _ in objs if oid not in have]
    print(f"[embed] objects {len(objs)} | cached {len(have)} | to compute {len(todo)}", flush=True)
    ids, E_obj, E_tex, HF = [], [], [], []
    if todo:
        clip = CeramicClip(device=device)

        geo = {oid: meta for oid, meta in objs}

        def prep(oid):
            p = obj_file(out, "smaller_object_rgba_raw", f"{oid}.png")
            try:
                a = np.asarray(Image.open(p).convert("RGBA"))
                t = load_tag(out, oid)
            except Exception:  # noqa: BLE001
                return oid, None
            rgb, mask = np.ascontiguousarray(a[..., :3]), a[..., 3] > 127
            if mask.sum() < 50:
                return oid, None
            # the crop is the tight mask box; CeramicTagger.views adds a 4 % margin (clipped at the photo
            # edge) that is white outside the mask, so rebuild exactly that margin to get the v3.1 views
            bb = (t.get("object_geometry") or {}).get("bbox_px")
            if bb:
                W, H = geo[oid]["image_size_wh"]
                x0, y0, x1, y1 = bb
                pad = int(0.04 * max(y1 - y0, x1 - x0))
                pl, pt, pr, pb = min(pad, x0), min(pad, y0), min(pad, W - x1), min(pad, H - y1)
                rgb = np.pad(rgb, ((pt, pb), (pl, pr), (0, 0)), constant_values=255)
                mask = np.pad(mask, ((pt, pb), (pl, pr)), constant_values=False)
            obj, tex = CeramicTagger.views(rgb, mask)
            return oid, (obj, tex, highlight_features(rgb, mask))

        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            for start in range(0, len(todo), batch):
                chunk = list(ex.map(prep, todo[start:start + batch]))
                good = [(o, v) for o, v in chunk if v is not None]
                if good:
                    with torch.inference_mode():
                        e = clip.image_embeddings([v[0] for _, v in good] + [v[1] for _, v in good]).float().cpu().numpy()
                    n = len(good)
                    ids += [o for o, _ in good]
                    E_obj.append(e[:n])
                    E_tex.append(e[n:])
                    HF.append(np.stack([v[2] for _, v in good]))
                done = min(len(todo), start + batch)
                if done % (batch * 10) == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"[embed] {done}/{len(todo)} | {el / 60:.1f} min | ETA {(len(todo) - done) * el / max(1, done) / 60:.1f} min", flush=True)
    if have:
        keep = [oid for oid in have]
        ids = keep + ids
        E_obj = [old["e_obj"]] + E_obj
        E_tex = [old["e_tex"]] + E_tex
        HF = [old["hf"]] + HF
    E_obj = np.concatenate(E_obj).astype(np.float32) if E_obj else np.zeros((0, 512), np.float32)
    E_tex = np.concatenate(E_tex).astype(np.float32) if E_tex else np.zeros((0, 512), np.float32)
    HF = np.concatenate(HF).astype(np.float32) if HF else np.zeros((0, 6), np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name("object_features.tmp.npz")
    np.savez_compressed(tmp, ids=np.array(ids), e_obj=E_obj, e_tex=E_tex, hf=HF, version=np.array(FEATURE_VERSION))
    os.replace(tmp, cache)
    print(f"[embed] saved {len(ids)} feature rows -> {cache}", flush=True)


def reproducibility_check(out: Path, objs, device, n=40):
    """The cached embeddings must reproduce the stored v3.1 effect scores (same views, same model)."""
    import torch
    from glazeprep.ceramic_clip import CeramicClip
    from glazeprep.ceramic_tagger import CeramicTagger
    z = np.load(out / "evaluation" / "retag" / "object_features.npz")
    idx = {o: i for i, o in enumerate(z["ids"].tolist())}
    clip = CeramicClip(device=device)
    tagger = CeramicTagger(clip, CODE / "configs" / "tagger_thresholds.json")
    diffs = []
    for oid, _ in objs[:: max(1, len(objs) // n)][:n]:
        if oid not in idx:
            continue
        t = load_tag(out, oid)
        st = (t.get("visual_tag_scores") or {}).get("glaze_effect")
        if not st:
            continue
        i = idx[oid]
        e = torch.nn.functional.normalize(torch.tensor(z["e_obj"][i] + z["e_tex"][i]).float()[None].to(clip.device), dim=-1)
        sc = tagger._multi(tagger.effect_emb, tagger.effect_neutral, e)
        diffs += [abs(sc[c]["presence"] - st[c]["presence"]) for c in st]
    res = {"objects_checked": n, "max_abs_presence_diff": round(float(max(diffs)), 5) if diffs else None,
           "mean_abs_presence_diff": round(float(np.mean(diffs)), 6) if diffs else None}
    print(f"[check] recomputed v3.1 effect scores from cached features: {res}", flush=True)
    return res


# --------------------------------------------------------------------------- phase: fit
def text_fields(t):
    return {"name": t.get("recipe_name") or "", "subtype": t.get("subtype_csv") or "", "description": t.get("description_csv") or ""}


TEXT_TABLES = {"surface": {"name": R.SURFACE_TEXT, "subtype": R.SURFACE_TEXT, "description": R.SURFACE_TEXT},
               "transparency": {"name": R.TRANSPARENCY_TEXT_SHORT, "subtype": R.TRANSPARENCY_TEXT_SHORT,
                                "description": R.TRANSPARENCY_TEXT_LONG}}


def feature_matrix(z, rows_idx):
    eo = z["e_obj"][rows_idx].astype(np.float32)
    et = z["e_tex"][rows_idx].astype(np.float32)
    return np.hstack([eo, et, z["hf"][rows_idx]])


DISABLED = 1.01   # a probability threshold that can never be reached


def precision_threshold(p_top, correct, target, min_n=30):
    """Smallest threshold whose selected set reaches `target` precision (with >= min_n samples);
    DISABLED when no threshold reaches it."""
    for th in np.round(np.arange(0.34, 0.995, 0.01), 2):
        sel = p_top >= th
        if sel.sum() >= min_n and correct[sel].mean() >= target:
            return float(th)
    return DISABLED


def phase_fit(out: Path, objs, device):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    z = np.load(out / "evaluation" / "retag" / "object_features.npz")
    idx = {o: i for i, o in enumerate(z["ids"].tolist())}
    recs = []
    for oid, _ in objs:
        if oid in idx:
            t = load_tag(out, oid)
            recs.append({"oid": oid, "i": idx[oid], "rid": str(t.get("recipe_id")), "surface": t.get("surface_csv") or "",
                         "transparency": t.get("transparency_csv") or "", "text": text_fields(t)})
    report, models, thresholds = {}, {}, {}
    zs = zero_shot_probs(z, device)
    for attr, groups, grp_fn in (("surface", R.SURFACE_GROUPS, R.surface_group),
                                 ("transparency", R.TRANSPARENCY_GROUPS, R.transparency_group)):
        lab = [r for r in recs if grp_fn(r[attr])]
        keys = sorted(groups)
        y = np.array([keys.index(grp_fn(r[attr])) for r in lab])
        X = feature_matrix(z, [r["i"] for r in lab])
        rid = np.array([r["rid"] for r in lab])
        dev = np.array([is_dev(x) for x in rid])
        tg = [R.text_group(r["text"], TEXT_TABLES[attr])[0] for r in lab]
        tg = np.array([keys.index(g) if g in keys else -1 for g in tg])
        # model selection on dev only (grouped CV by recipe)
        best_c, best_acc, oof_best = None, -1, None
        for c in (0.03, 0.1, 0.3, 1.0):
            oof = np.zeros((dev.sum(), len(keys)))
            for tr, va in GroupKFold(5).split(X[dev], y[dev], rid[dev]):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000))
                m.fit(X[dev][tr], y[dev][tr])
                oof[va] = m.predict_proba(X[dev][va])
            acc = float((oof.argmax(1) == y[dev]).mean())
            print(f"[fit] {attr}: C={c} grouped-CV accuracy on dev {acc:.3f}", flush=True)
            if acc > best_acc:
                best_c, best_acc, oof_best = c, acc, oof
        p_top, pred = oof_best.max(1), oof_best.argmax(1)
        correct = pred == y[dev]
        th_vis = precision_threshold(p_top, correct, TARGET_PRECISION[attr])
        agree = tg[dev] == pred
        th_agree = precision_threshold(p_top[agree], correct[agree], TARGET_PRECISION[attr]) if agree.sum() >= 30 else th_vis
        th_agree = min(th_agree, th_vis)
        # held-out test: model trained on dev only, thresholds from dev
        m_dev = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000))
        m_dev.fit(X[dev], y[dev])
        pt = m_dev.predict_proba(X[~dev])
        test = evaluate_rule(pt, y[~dev], tg[~dev], keys, groups, th_vis, th_agree)
        # zero-shot baseline on the same test objects
        zp = zs[attr][[r["i"] for r in lab]][~dev]
        zs_acc = float((zp.argmax(1) == y[~dev]).mean())
        text_cov = float((tg[~dev] >= 0).mean())
        text_prec = float((tg[~dev][tg[~dev] >= 0] == y[~dev][tg[~dev] >= 0]).mean()) if (tg[~dev] >= 0).any() else None
        report[attr] = {"groups": {k: groups[k][0] for k in keys}, "labelled_objects": len(lab),
                        "image_only_reached_target_on_dev": th_vis < DISABLED,
                        "image_plus_text_reached_target_on_dev": th_agree < DISABLED,
                        "dev_objects": int(dev.sum()), "test_objects": int((~dev).sum()),
                        "class_counts": {groups[k][0]: int((y == i).sum()) for i, k in enumerate(keys)},
                        "chosen_C": best_c, "dev_grouped_cv_accuracy_all": round(best_acc, 3),
                        "threshold_visual": th_vis, "threshold_visual_plus_text": th_agree,
                        "target_precision": TARGET_PRECISION[attr], "test": test,
                        "test_accuracy_classifier_all_objects": round(float((pt.argmax(1) == y[~dev]).mean()), 3),
                        "test_accuracy_zero_shot_prompts_all_objects": round(zs_acc, 3),
                        "test_majority_class_rate": round(float(np.bincount(y[~dev]).max() / (~dev).sum()), 3),
                        "test_recipe_text_coverage": round(text_cov, 3),
                        "test_recipe_text_precision": None if text_prec is None else round(text_prec, 3)}
        # production model: all labelled objects, same C; thresholds from dev
        m_all = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000))
        m_all.fit(X, y)
        models[attr] = {"model": m_all, "keys": keys}
        thresholds[attr] = {"visual": th_vis, "agree": th_agree}
        print(f"[fit] {attr}: thresholds visual {th_vis} / visual+text {th_agree} | test {test}", flush=True)
    # possible-tier thresholds: per-class quantile of the stored presence scores over all objects
    pres = {"effect": {}, "defect": {}}
    for oid, _ in objs:
        sc = load_tag(out, oid).get("visual_tag_scores") or {}
        for kind, key in (("effect", "glaze_effect"), ("defect", "defect")):
            for c, v in (sc.get(key) or {}).items():
                pres[kind].setdefault(c, []).append(v["presence"])
    possible = {k: {c: round(float(np.quantile(v, POSSIBLE_QUANTILE)), 4) for c, v in d.items()} for k, d in pres.items()}
    cfg = {"tag_version": R.TAG_VERSION, "created": now_iso(), "surface": thresholds["surface"],
           "transparency": thresholds["transparency"], "possible_quantile": POSSIBLE_QUANTILE,
           "possible_thresholds": possible, "target_precision": TARGET_PRECISION,
           "note": "thresholds chosen on the dev half (md5(recipe_id) parity) with grouped CV; see evaluation/retag"}
    atomic_json(CONFIG_OUT, cfg)
    ev = out / "evaluation" / "retag"
    atomic_json(ev / "retag_thresholds.json", cfg)
    atomic_json(ev / "surface_transparency_eval.json", report)
    with (ev / "surface_transparency_model.pkl").open("wb") as fh:
        pickle.dump(models, fh)
    return report


def zero_shot_probs(z, device):
    import torch
    from glazeprep.ceramic_clip import CeramicClip
    clip = CeramicClip(device=device)
    out = {}
    e = torch.nn.functional.normalize(torch.tensor(z["e_obj"].astype(np.float32) + z["e_tex"].astype(np.float32)), dim=-1).to(clip.device)
    for attr, prompts in (("surface", SURFACE_PROMPTS), ("transparency", TRANSPARENCY_PROMPTS)):
        keys = sorted(prompts)
        te = []
        for k in keys:
            t = clip.text_embeddings([f"a photo of {p}." for p in prompts[k]] + [f"{p}." for p in prompts[k]])
            te.append(torch.nn.functional.normalize(t.float().mean(0, keepdim=True), dim=-1)[0])
        te = torch.stack(te)
        out[attr] = (clip.logit_scale * e @ te.T).softmax(-1).cpu().numpy()
    return out


def evaluate_rule(prob, y, tg, keys, groups, th_vis, th_agree):
    """Apply the CSV-blank decision rule to labelled test objects (pretending the CSV were blank)."""
    pred, p = prob.argmax(1), prob.max(1)
    resolved = np.zeros(len(y), bool)
    for i in range(len(y)):
        if tg[i] >= 0 and tg[i] != pred[i]:
            continue
        if (tg[i] == pred[i] and p[i] >= th_agree) or p[i] >= th_vis:
            resolved[i] = True
    out = {"coverage": round(float(resolved.mean()), 3),
           "precision": round(float((pred[resolved] == y[resolved]).mean()), 3) if resolved.any() else None,
           "per_group": {}}
    for k_i, k in enumerate(keys):
        sel = resolved & (pred == k_i)
        out["per_group"][groups[k][0]] = {
            "predicted": int(sel.sum()),
            "precision": round(float((y[sel] == k_i).mean()), 3) if sel.any() else None,
            "recall_of_true": round(float((resolved & (y == k_i) & (pred == k_i)).sum() / max(1, (y == k_i).sum())), 3)}
    return out


# --------------------------------------------------------------------------- phase: apply
def first_backup(out: Path):
    b_csv = out / "full_auto_tags_v3.1_backup.csv"
    b_dir = out / "tag_json_v3.1_backup"
    if not b_csv.exists():
        shutil.copy2(out / "full_auto_tags.csv", b_csv)
        print(f"[apply] backup {b_csv.name}", flush=True)
    if not b_dir.exists():
        tmp = out / "tag_json_v3.1_backup.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(out / "tag_json", tmp)
        os.replace(tmp, b_dir)
        print(f"[apply] backup {b_dir.name}/ ({sum(1 for _ in b_dir.iterdir())} files)", flush=True)
    return b_csv


def review_reasons(t, new, v31):
    """Tag review reasons recomputed from the v3.1 reasons that still apply plus the new sources."""
    keep = []
    for r in (v31.get("tag_review_reasons") or "").split("; "):
        # v3.1 joined reasons with "; " and one reason itself contained "; " -> its tail is dropped too
        if not r or r.startswith(("glaze effect predicted", "defect predicted", "recipe subtype names",
                                  "low measured precision")):
            continue
        keep.append(r)
    out = list(keep)
    es, ds = new["glaze_effect_source"], new["defect_source"]
    if "VISUAL" in es.replace("VISUAL_POSSIBLE", ""):
        out.append("glaze effect from the image (zero-shot, low measured precision)")
    if "VISUAL_POSSIBLE" in es or "VISUAL_POSSIBLE" in ds:
        out.append("possible effect / defect (low-confidence image score)")
    if "VISUAL" in ds.replace("VISUAL_POSSIBLE", ""):
        out.append("defect predicted from the image (defects always need review)")
    if "CSV_DESCRIPTION" in ds:
        out.append("defect named in the recipe description (may not be visible in this photo)")
    # a blank CSV value that stays "Unknown" is a data gap, not a prediction to verify: not a review trigger
    for a in ("surface", "transparency"):
        if new[f"{a}_source"].startswith("AUTO"):
            out.append(f"{a} inferred from the image (CSV blank)")
    return list(dict.fromkeys(out))


def mask_crop(out: Path, oid: str):
    try:
        return np.asarray(Image.open(obj_file(out, "smaller_object_masks_raw", f"{oid}.png")).convert("L")) > 127
    except Exception:  # noqa: BLE001
        return None


def phase_apply(out: Path, objs, empty, make_cards: bool, workers: int):
    from glazeprep.recipes import RecipeIndex
    cfg = json.loads(CONFIG_OUT.read_text(encoding="utf-8"))
    with (out / "evaluation" / "retag" / "surface_transparency_model.pkl").open("rb") as fh:
        models = pickle.load(fh)
    z = np.load(out / "evaluation" / "retag" / "object_features.npz")
    idx = {o: i for i, o in enumerate(z["ids"].tolist())}
    probs = {}
    if len(idx):
        X = feature_matrix(z, list(range(len(idx))))
        for attr in ("surface", "transparency"):
            probs[attr] = models[attr]["model"].predict_proba(X)
    b_csv = first_backup(out)
    with b_csv.open(encoding="utf-8-sig", newline="") as fh:
        v31_columns = csv.DictReader(fh).fieldnames
    k = v31_columns.index("description_csv") + 1
    added = NEW_COLUMNS + COLOR_COLUMNS + REVIEW_COLUMNS
    columns = v31_columns[:k] + added + [c for c in v31_columns[k:] if c not in added]
    review = load_review(out)
    started = time.time()
    counters = {a: Counter() for a in ("surface", "transparency", "glaze_effect", "defect", "form")}

    def one(item):
        oid, meta = item
        t = load_tag(out, oid)
        v31 = t.get("v31_original") or {key: t.get(key) for key in ("tag_review_required", "tag_review_reasons", "review_reasons")}
        new = {"glazy_url": f"https://glazy.org/recipes/{t.get('recipe_id')}" if t.get("recipe_id") else "",
               "tag_version": R.TAG_VERSION}
        tf = text_fields(t)
        for attr, vocab, groups in (("surface", R.SURFACE_VOCAB, R.SURFACE_GROUPS),
                                    ("transparency", R.TRANSPARENCY_VOCAB, R.TRANSPARENCY_GROUPS)):
            pv = None
            if oid in idx:
                keys = models[attr]["keys"]
                pv = {g: float(probs[attr][idx[oid]][i]) for i, g in enumerate(keys)}
            tg, tev = R.text_group(tf, TEXT_TABLES[attr])
            res = R.resolve_optical(t.get(f"{attr}_csv"), vocab, groups, pv, tg, tev, cfg[attr])
            new[attr], new[f"{attr}_source"], new[f"{attr}_evidence"] = res["value"], res["source"], res["evidence"]
        crop = mask_crop(out, oid)
        g = t.get("object_geometry") or {}
        contact = R.edge_contact(crop, g["bbox_px"], meta["image_size_wh"]) if crop is not None and g.get("bbox_px") else {}
        fs = R.resolve_form_shape(t, contact)
        new.update({"object_form": fs["object_form"], "shape": fs["shape"],
                    "object_truncated_by_frame_v32": fs["object_truncated_by_frame_v32"],
                    "frame_contact": fs["frame_contact"], "form_note": fs["form_note"]})
        new.update(R.resolve_effects_defects(t, cfg["possible_thresholds"], fs["object_form"]))
        reasons = review_reasons(t, new, v31)
        seg_reasons = [r for r in (v31.get("review_reasons") or "").split("; ") if r and r not in (v31.get("tag_review_reasons") or "")]
        t.update({kk: new[kk] for kk in NEW_COLUMNS})
        t.update(color_fields(t))
        rv = review.get(oid, {})
        t.update({"review_status": rv.get("review_status", ""), "review_note": rv.get("review_note", ""),
                  "reviewed_at": rv.get("reviewed_at", ""),
                  "object_location": NOT_NEEDED_DIR if obj_file(out, "tag_json", f"{oid}.json").parent.parent.name == NOT_NEEDED_DIR
                  else "smaller_object"})
        t["v31_original"] = v31
        t["tag_review_required"] = bool(reasons)
        t["tag_review_reasons"] = "; ".join(reasons)
        t["review_reasons"] = "; ".join(dict.fromkeys(seg_reasons + reasons))
        t["retag"] = {"tag_version": R.TAG_VERSION, "retagged_at": now_iso(), "thresholds": str(CONFIG_OUT.name)}
        atomic_json(obj_file(out, "tag_json", f"{oid}.json"), t)
        return t, new

    rows = []
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for t, new in ex.map(one, objs):
            r = dict(t)
            r["visual_tag_scores_json"] = json.dumps(t.get("visual_tag_scores", {}), ensure_ascii=False)
            rows.append(r)
            counters["surface"][new["surface_source"]] += 1
            counters["transparency"][new["transparency_source"]] += 1
            for s in new["glaze_effect_source"].split(" | "):
                counters["glaze_effect"][s] += 1
            for s in new["defect_source"].split(" | "):
                counters["defect"][s] += 1
            counters["form"]["withdrawn_cut_by_edge" if new["form_note"] else "unchanged"] += 1
            done += 1
            if done % 2000 == 0 or done == len(objs):
                el = time.time() - started
                print(f"[apply] fields {done}/{len(objs)} objects | {el / 60:.1f} min", flush=True)
    if make_cards:
        # PIL font objects are shared inside one process, so cards are drawn in separate processes
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(str(out), oid, meta.get("diagnostics", {}).get("review_reasons", [])[:3]) for oid, meta in objs]
        t1, n = time.time(), 0
        with ProcessPoolExecutor(max(1, min(workers, (os.cpu_count() or 2) - 1))) as ex:
            for _ in ex.map(draw_card_task, tasks, chunksize=64):
                n += 1
                if n % 2000 == 0 or n == len(tasks):
                    el = time.time() - t1
                    print(f"[apply] cards {n}/{len(tasks)} | {el / 60:.1f} min | ETA {(len(tasks) - n) * el / n / 60:.1f} min", flush=True)
    recipes = RecipeIndex(BASE / "csv 7k data")
    for rec in empty:
        r = rec.get("recipe", {})
        ctx = recipes.recipe_context(r.get("recipe_id")) if r.get("match_status") == "MATCHED" else {}
        row = {"parent_filename": rec["parent_filename"], "recipe_id": r.get("recipe_id", ""),
               "recipe_match_status": r.get("match_status", ""), "recipe_name": r.get("recipe_name", ""),
               "object_count": 0, "segmentation_status": rec.get("segmentation_status"),
               "segmentation_review_required": True, "original_image_path": rec.get("original_image_path"),
               "config_hash": rec.get("config_hash"), "pipeline_version": rec.get("schema_version"),
               "processed_at": rec.get("processed_at"), "tags_human_verified": False, "tag_review_required": True,
               "surface_csv": ctx.get("surface_csv", ""), "transparency_csv": ctx.get("transparency_csv", ""),
               "subtype_csv": ctx.get("subtype_csv", ""), "glazy_url": f"https://glazy.org/recipes/{r.get('recipe_id')}",
               "tag_version": R.TAG_VERSION, "glaze_effect": "Unknown (no ceramic object found)",
               "defect": "Unknown (no ceramic object found)", "object_form": "Unknown (no ceramic object found)",
               "shape": "Unknown (no ceramic object found)",
               "review_reasons": "; ".join(rec.get("diagnostics", {}).get("review_reasons", []))}
        for attr in ("surface", "transparency"):
            v = row[f"{attr}_csv"]
            row[attr] = f"{v} (CSV)" if v else "Unknown (CSV blank; no ceramic object found)"
            row[f"{attr}_source"] = "CSV" if v else "UNKNOWN"
        rows.append(row)
    rows.sort(key=lambda r: (len(Path(r["parent_filename"]).stem), Path(r["parent_filename"]).stem, int(r.get("object_index") or 0)))
    tmp = out / "full_auto_tags.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    os.replace(tmp, out / "full_auto_tags.csv")
    summary = {"tag_version": R.TAG_VERSION, "finished_at": now_iso(), "objects": len(objs),
               "photos_without_object": len(empty), "minutes": round((time.time() - started) / 60, 1),
               "cards_regenerated": make_cards, "source_counts": {k: dict(v) for k, v in counters.items()},
               "csv_columns_added": NEW_COLUMNS, "backup_csv": b_csv.name, "backup_tag_json": "tag_json_v3.1_backup/"}
    atomic_json(out / "evaluation" / "retag" / "retag_summary.json", summary)
    print(f"[apply] wrote full_auto_tags.csv ({len(rows)} rows, {len(columns)} columns)", flush=True)
    return summary


def draw_card_task(task):
    out, oid, seg_reasons = task
    from glazeprep.previews_v3 import detail_card
    out = Path(out)
    t = load_tag(out, oid)
    draw_card(out, t, seg_reasons, detail_card)
    return oid


def draw_card(out: Path, t: dict, seg_reasons, detail_card):
    oid = t["object_id"]
    img = Image.open(obj_file(out, "smaller_object", f"{oid}.jpg")).convert("RGB")
    status = t.get("segmentation_status")
    entries = [("Recipe", t.get("recipe_name") or "No linked recipe"),
               ("Recipe match", t.get("recipe_match_status")),
               ("Recipe page", (t.get("glazy_url") or "").replace("https://", "")),
               ("Surface", t["surface"]), ("Transparency", t["transparency"]),
               ("Subtype (CSV)", t.get("subtype_csv") or "Not provided in CSV"),
               ("Texture (visual)", t.get("texture_predicted")), ("Glaze effect", t["glaze_effect"]),
               ("Defect", t["defect"]), ("Decoration", t.get("decoration_predicted")),
               ("Form", t["object_form"]), ("Shape", t["shape"]),
               ("Object", f"{t.get('object_index')} of {t.get('object_count')} | area {float(t.get('mask_area_fraction') or 0):.1%} "
                          f"of photo | segmentation {status} (quality {t.get('segmentation_quality_score')})")]
    reasons = [r for r in (t.get("tag_review_reasons") or "").split("; ") if r]
    lines = [f"Tags: {'REVIEW REQUIRED' if t.get('tag_review_required') else 'no review trigger'} (not human-verified)"]
    lines += ["- " + r for r in reasons[:5]]
    if t.get("segmentation_review_required"):
        lines.append("Segmentation: REVIEW REQUIRED - " + "; ".join(seg_reasons))
    if t.get("form_note"):
        lines.append("Note: " + t["form_note"])
    lines.append("Sources: (CSV) = recipe database value | (CSV subtype / description) = recipe text | (visual) = "
                 "image model | Possible = low-confidence image score | None visible = checked, nothing found")
    if t.get("colorthief_palette"):
        colors, title = t["colorthief_palette"], "TOP 3 COLORS (ColorThief palette, % of object pixels)"
    else:
        colors, title = t.get("color_palette") or [], "TOP 3 CERAMIC COLORS (mask pixels, CIELAB k-means)"
    card = detail_card(img, f"{oid}.jpg", entries, colors, lines, colors_title=title)
    dst = obj_file(out, "detailed_preview", f"{oid}.jpg")
    tmp = dst.with_name(dst.name + ".tmp")
    card.save(tmp, format="JPEG", quality=90)
    os.replace(tmp, dst)


# --------------------------------------------------------------------------- checks + report
def run_checks(out: Path, objs, started_ts):
    with (out / "full_auto_tags.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("object_id")]
    probs = Counter()
    for r in rows:
        for c in ("surface", "transparency", "glaze_effect", "defect", "object_form", "shape"):
            if not (r.get(c) or "").strip():
                probs[f"blank {c}"] += 1
        for a in ("surface", "transparency"):
            csv_v = (r.get(f"{a}_csv") or "").strip()
            if csv_v and r.get(a) != f"{csv_v} (CSV)":
                probs[f"{a}: CSV value not kept exactly"] += 1
            if not csv_v and r.get(a, "").endswith("(CSV)"):
                probs[f"{a}: '(CSV)' shown although the CSV is blank"] += 1
        if str(r.get("tags_human_verified")).lower() not in ("false", "0", ""):
            probs["tags_human_verified not False"] += 1
        if r.get("recipe_id") != Path(r.get("parent_filename", "")).stem:
            probs["recipe_id != parent stem"] += 1
    stale = sum(1 for oid, _ in objs if obj_file(out, "detailed_preview", f"{oid}.jpg").stat().st_mtime < started_ts)
    res = {"object_rows": len(rows), "objects_expected": len(objs), "problems": dict(probs),
           "cards_not_regenerated": stale, "passed": not probs and len(rows) == len(objs) and stale == 0}
    atomic_json(out / "evaluation" / "retag" / "retag_checks.json", res)
    print(f"[check] {res}", flush=True)
    return res


def write_report(out: Path, report: dict | None, summary: dict, checks: dict, repro: dict | None):
    ev = out / "evaluation" / "retag"
    if report is None and (ev / "surface_transparency_eval.json").exists():
        report = json.loads((ev / "surface_transparency_eval.json").read_text(encoding="utf-8"))
    cfg = json.loads(CONFIG_OUT.read_text(encoding="utf-8"))
    L = ["# Attribute re-tagging report (claude_v3.2-retag)", "",
         f"Generated {now_iso()} by scripts/16_retag_attributes.py. Segmentation, object files, colours and recipe links "
         "were not changed. No pixel or attribute ground truth exists; every number below is agreement with CSV labels "
         "typed by recipe authors (weak labels), not accuracy.", "",
         "## Source of each value", "",
         "| field | rule |", "|---|---|",
         "| Surface, Transparency | CSV value kept exactly and shown as `<value> (CSV)`. CSV blank: coarse group from the image classifier only above a threshold chosen for the target precision on the dev half; recipe text (name / subtype / description) may support it but never decides alone; otherwise `Unknown (CSV blank; ...)` |",
         "| Glaze effect, Defect | `(CSV subtype)`, `(CSV description)` (non-negated mention), `(visual)` (strict v3.1 threshold), `Possible: ... (visual, low confidence)` (score in the top 3 % of that class over all objects), `None visible (checked)`, `Unknown (...)` |",
         "| Form, Shape | objects cut along one photo edge (>= 15 % of that side) no longer receive a form or silhouette shape |",
         ""]
    if report:
        L += ["## Surface / Transparency fallback (when the CSV is blank)", "",
              "Image classifier: logistic regression on frozen OpenCLIP ViT-B-32 features (object + texture view) plus 6 "
              "highlight statistics, trained on objects whose recipe has a CSV value (weak labels from many authors). "
              "Dev / test split by md5(recipe_id); C and thresholds chosen on dev with grouped CV; the test half was used "
              "once, for the numbers below. The production model is refitted on dev + test with the same settings.", "",
              "| attribute | groups | labelled objects | test accuracy (all) | zero-shot prompts (all) | majority class | threshold image / image+text | test coverage | test precision on covered |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a, r in report.items():
            L.append(f"| {a} | {', '.join(r['groups'].values())} | {r['labelled_objects']} | {r['test_accuracy_classifier_all_objects']} | "
                     f"{r['test_accuracy_zero_shot_prompts_all_objects']} | {r['test_majority_class_rate']} | "
                     f"{r['threshold_visual']} / {r['threshold_visual_plus_text']} | {r['test']['coverage']} | {r['test']['precision']} |")
        L += ["", "A threshold of 1.01 means the rule is switched off: no threshold reached the target precision "
              "(surface 0.80, transparency 0.85) on the dev half. Recipe text alone on the test half: "
              + "; ".join(f"{a} coverage {r['test_recipe_text_coverage']}, precision {r['test_recipe_text_precision']}"
                          for a, r in report.items()) + " (text alone never decides; it is listed for comparison).",
              "", "Per group on the test half (covered objects):", ""]
        for a, r in report.items():
            for g, v in r["test"]["per_group"].items():
                L.append(f"- {a} / {g}: predicted {v['predicted']}, precision {v['precision']}, recall {v['recall_of_true']}")
        L += ["", "Caveat: recipes with a blank CSV value may differ from labelled ones (other authors, more "
              "generic 'Glaze' subtypes), so precision on them can be lower than measured here.", ""]
    L += ["## Possible-tier thresholds (97th percentile of each class over all objects)", "",
          "Effects: " + ", ".join(f"{c} {v}" for c, v in cfg["possible_thresholds"]["effect"].items()),
          "", "Defects: " + ", ".join(f"{c} {v}" for c, v in cfg["possible_thresholds"]["defect"].items()), "",
          "Measured against the recipe Subtype as a weak label over all objects, a score in the top 3 % agreed with the "
          "Subtype in only about 3-13 % of cases (crystalline 13 %, pitting/crater 10 %, oil spot 4 %, crackle 4 %, "
          "metallic 3 %, crawling 3 %). The weak labels are incomplete, so true precision is higher but unknown: "
          "'Possible' means 'worth a look', not 'present'.", "",
          "## Counts by source", ""]
    for k, v in summary["source_counts"].items():
        L.append(f"- {k}: " + ", ".join(f"{s} {n}" for s, n in sorted(v.items(), key=lambda kv: -kv[1])))
    L += ["", "## Automatic checks", "", "```", json.dumps(checks, indent=2), "```", ""]
    if repro:
        L += [f"Reproducibility: v3.1 effect scores recomputed from the cached features differ by at most "
              f"{repro['max_abs_presence_diff']} (mean {repro['mean_abs_presence_diff']}).", ""]
    L += ["## Rollback", "", "`full_auto_tags_v3.1_backup.csv` and `tag_json_v3.1_backup/` hold the v3.1 outputs. "
          "The v3.1 values are also kept in every new record (`*_predicted` columns, `v31_original`).", ""]
    atomic_text(ev / "retag_report.md", "\n".join(L))


def manifest_entry(out: Path, summary, checks):
    p = out / "run_manifest.json"
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    m.setdefault("retag_runs", []).append({"tag_version": R.TAG_VERSION, "finished_at": summary["finished_at"],
                                            "objects": summary["objects"], "checks_passed": checks["passed"],
                                            "rules": "glazeprep/retag.py", "script": "scripts/16_retag_attributes.py",
                                            "thresholds": str(CONFIG_OUT)})
    atomic_json(p, m)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=BASE / "full_segmentation_claude_process")
    ap.add_argument("--phases", default="embed,fit,apply")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-cards", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    phases = {p.strip() for p in args.phases.split(",")}
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 66, flush=True)
    print(f"STAGE F - ATTRIBUTE RE-TAGGING ({R.TAG_VERSION}) | device {device}", flush=True)
    print("=" * 66, flush=True)
    objs, empty = inventory(out)
    print(f"complete photos with objects: {len({m['parent_filename'] for _, m in objs})} | objects {len(objs)} | "
          f"photos without object {len(empty)}", flush=True)
    report = repro = None
    if "embed" in phases:
        phase_embed(out, objs, device, args.workers, args.batch)
        repro = reproducibility_check(out, objs, device)
    if "fit" in phases:
        report = phase_fit(out, objs, device)
    if "apply" in phases:
        t0 = time.time()
        summary = phase_apply(out, objs, empty, not args.no_cards, args.workers)
        checks = run_checks(out, objs, t0 - 1 if not args.no_cards else 0)
        write_report(out, report, summary, checks, repro)
        manifest_entry(out, summary, checks)
        print("report:", out / "evaluation" / "retag" / "retag_report.md", flush=True)
        return 0 if checks["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
