"""Train the smart search model (one joint space for free text + photo + glaze formula) and test it on held-out recipes.

Needs the recipe bank and the DINOv2 object cache of DOWNLOAD_DATA.bat (multimodal_experiments\\work\\data,
multimodal_experiments\\cache) and MiniLM in model_cache (SETUP.bat). INSTALL_GLAZE_MODEL.bat runs it; about 2 - 5 min
with an NVIDIA GPU. Writes multimodal_experiments\\models\\glaze_joint\\ (not in git):
  heads.pt          the three trained projection heads (weights_only)      gallery.npz   recipe embeddings, families
  text_tags.joblib  free text -> implied tags (logistic regression)        meta.json     settings + every test result
Every number in meta.json is measured on the TEST split, which training and model selection never see.
A trained model is kept (INSTALL_GLAZE_MODEL.bat can be run again safely); --retrain trains a new one.
usage: python scripts/train_glaze_joint.py [--device cuda|cpu] [--seed 1] [--retrain]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "multimodal_experiments"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mm import glaze_joint as J  # noqa: E402  (mm.common points the model cache at <CG2>\model_cache)
from mm import glaze_text as GT  # noqa: E402

HYPER = {"hidden": 512, "dim": 256, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "batch": 256, "temperature": 0.07,
         "max_epochs": 150, "patience": 12, "p_tags": 0.25, "p_free": 0.20, "n_noisy": 8}


def tag_model(D, log):
    from sklearn.metrics import f1_score
    from glazeprep.appearance import _REFERENCES
    fam = {n: f for f, n, _ in _REFERENCES}
    lab = {"surface": D["surface"], "transparency": D["transparency"], "atmosphere": np.array([a.split(";")[0] for a in D["atmospheres"]]),
           "family": np.array([fam.get(w, "") for w in D["colour_word"]])}
    tr = np.flatnonzero((D["split"] == "train") & D["MF"])
    te = np.flatnonzero((D["split"] == "test") & D["MF"])
    texts = list(D["free"])
    tm = GT.TagModel.fit([texts[i] for i in tr], {f: list(y[tr]) for f, y in lab.items()})
    Xte = tm.transform([texts[i] for i in te])
    res = {}
    for f, m in tm.models.items():
        y = lab[f][te]
        ok = np.array([bool(v) for v in y])
        if not ok.any():
            continue
        pred = m.predict(Xte[ok])
        maj = Counter(lab[f][tr][[bool(v) for v in lab[f][tr]]]).most_common(1)[0][0]
        res[f] = {"n": int(ok.sum()), "accuracy": round(float(np.mean(pred == y[ok])), 4), "macro_F1": round(float(f1_score(y[ok], pred, average="macro")), 4),
                  "majority_accuracy": round(float(np.mean(y[ok] == maj)), 4), "majority_class": maj, "classes": int(len(m.classes_))}
        log(f"[tags] {f:12s} test accuracy {res[f]['accuracy']:.3f} (always '{maj}' {res[f]['majority_accuracy']:.3f}), macro-F1 {res[f]['macro_F1']:.3f}")
    return tm, res


def extraction_check(D, seed, log):
    """the text reader on fresh noisy sentences of TEST recipes: when it reads a value, is it the recipe's value?"""
    from skimage.color import deltaE_ciede2000
    from mm.glaze_retrieval import cone_overlap, cone_range_orton
    te = np.flatnonzero(D["split"] == "test")
    hit, seen, n = Counter(), Counter(), 0
    examples = []
    for j, i in enumerate(te):
        f = D["facts"][i]
        s = GT.noisy_variants(f, 1, 900001 + seed * 7919 + j, typo_p=0.15)[0]
        u = GT.extract(s)
        n += 1
        if u["surface"] and f["surface"]:
            seen["surface"] += 1
            hit["surface"] += u["surface"]["value"].lower() == f["surface"].lower()
        if u["cone"] and f["cone_lo"]:
            seen["cone"] += 1
            hit["cone"] += cone_overlap(u["cone"]["value"], J.cone_text(f["cone_lo"], f["cone_hi"]), cone_range_orton) == 1.0
        if u["atmospheres"] and f["atmospheres"]:
            seen["atmosphere"] += 1
            hit["atmosphere"] += bool({a["value"].lower() for a in u["atmospheres"]} & {a.lower() for a in f["atmospheres"]})
        if u["colour"] and D["top_hex"][i]:
            seen["colour"] += 1
            a, b = J.lab_of([u["colour"]["hex"]]), J.lab_of([D["top_hex"][i]])
            hit["colour"] += float(deltaE_ciede2000(a, b)[0]) <= 20
        if len(examples) < 12 and j % 37 == 0:
            examples.append({"text": s, "recipe": int(D["ids"][i]), "colour": (u["colour"] or {}).get("words"), "hex": (u["colour"] or {}).get("hex"),
                             "cone": (u["cone"] or {}).get("value"), "surface": (u["surface"] or {}).get("value"),
                             "atmospheres": [a["value"] for a in u["atmospheres"]], "corrections": u["corrections"]})
    res = {k: {"read": seen[k], "correct": round(hit[k] / max(1, seen[k]), 4)} for k in ("colour", "cone", "surface", "atmosphere")}
    res["sentences"] = n
    log("[read] noisy test sentences: " + ", ".join(f"{k} {v['correct']:.1%} correct ({v['read']} read)" for k, v in res.items() if k != "sentences"))
    return res, examples


def smart_check(js, D, seed, log, n=300):
    """P@10 of the full search for noisy test sentences: a result is relevant when its colour is within dE00 20 of the
    query recipe's colour and its surface and cone match the query recipe's (when known). Three ways of scoring."""
    from skimage.color import deltaE_ciede2000
    from mm.glaze_retrieval import cone_overlap, cone_range_orton
    rng = np.random.default_rng(seed)
    te = rng.choice(np.flatnonzero((D["split"] == "test") & (D["top_hex"] != "")), n, replace=False)
    row = {int(r): k for k, r in enumerate(D["ids"])}
    zero = {k: 0.0 for k in J.PARTS}
    modes = {"text model only": {**zero, "text": 1.0}, "read values only": {**js.cfg["weights"], "text": 0.0}, "both (smart search)": js.cfg["weights"]}
    res = {}
    for name, wts in modes.items():
        p10 = []
        for j, i in enumerate(te):
            f = D["facts"][i]
            s = GT.noisy_variants(f, 1, 700001 + j)[0]
            try:
                r = js.search({"text": s, "weights": wts, "top_k": 10})
            except ValueError:
                continue
            qa = J.lab_of([D["top_hex"][i]])
            rel = 0
            for x in r["weighted_results"]:
                k = row[x["recipe_id"]]
                ok = bool(D["top_hex"][k]) and float(deltaE_ciede2000(qa, J.lab_of([D["top_hex"][k]]))[0]) <= 20
                if f["surface"]:
                    ok &= D["surface"][k] == f["surface"]
                if f["cone_lo"]:
                    ok &= cone_overlap(J.cone_text(f["cone_lo"], f["cone_hi"]), J.cone_text(D["cone_lo"][k], D["cone_hi"][k]) or "", cone_range_orton) == 1.0
                rel += ok
            p10.append(rel / 10)
        res[name] = {"queries": len(p10), "P@10": round(float(np.mean(p10)), 4)}
        log(f"[smart] {name:22s} P@10 {res[name]['P@10']:.3f} ({len(p10)} noisy test sentences)")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--retrain", action="store_true", help="train again although a trained model exists (its test numbers change slightly)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log = lambda m: print(m, flush=True)  # noqa: E731
    t0 = time.time()
    cfg = {**HYPER, "seed": args.seed}
    out = J.CODE / J.read_config()["model_dir"]
    if (out / "meta.json").is_file() and (out / "heads.pt").is_file() and not args.retrain:
        m = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        log(f"[joint] already trained: model {m.get('model_version')} ({m.get('created')}) in {out} - kept. "
            "Train again with: .venv\\Scripts\\python.exe scripts\\train_glaze_joint.py --retrain")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    enc = J.TextEncoder(args.device)
    D = J.build_data(enc, cfg["n_noisy"], args.seed, log)
    log(f"[joint] {len(D['ids']):,} recipes · split {dict(Counter(D['split']))} · photo {D['MI'].sum():,} · formula {D['MC'].sum():,} · description {D['MF'].sum():,}")
    heads, info = J.train(D, cfg, args.device, log)
    dev = torch.device(args.device)
    ev = {"val": J.evaluate(heads, D, "val", dev, seed=args.seed + 7), "test": J.evaluate(heads, D, "test", dev, seed=args.seed + 11, baseline_encode=True)}
    for t, m in ev["test"].items():
        log(f"[test] {t:55s} R@1 {m['R@1']:.3f} R@5 {m['R@5']:.3f} R@10 {m['R@10']:.3f} median rank {m['median_rank']} (n {m['n']}, gallery {m['gallery']})")
    Z = J.embed(heads, D, dev)
    M = {"text": np.ones(len(D["ids"]), bool), "photo": D["MI"], "formula": D["MC"]}
    fused = J.fuse(Z, M, ("text", "photo", "formula")).astype(np.float32)
    fam, fam_info = J.families(fused, np.flatnonzero(D["split"] == "train"), args.seed, log)
    tm, tag_eval = tag_model(D, log)
    import joblib
    joblib.dump(tm, out / "text_tags.joblib")
    torch.save({"heads": {m: {k: v.cpu() for k, v in h.state_dict().items()} for m, h in heads.items()},
                "in_dims": {"text": int(D["TS"].shape[1]), "photo": int(D["IM"].shape[1]), "formula": int(D["CH"].shape[1])},
                "hidden": cfg["hidden"], "dim": cfg["dim"]}, out / "heads.pt")
    txt = lambda a: np.asarray(a).astype(str)  # noqa: E731  (plain text arrays: the app loads with allow_pickle=False)
    np.savez_compressed(out / "gallery.npz", ids=txt(D["ids"]), fused=fused, family=fam, P_lab=D["P_lab"], P_frac=D["P_frac"],
                        top_hex=txt(D["top_hex"]), colour_word=txt(D["colour_word"]), name=txt(D["name"]), cone_lo=txt(D["cone_lo"]),
                        cone_hi=txt(D["cone_hi"]), surface=txt(D["surface"]), transparency=txt(D["transparency"]), atmospheres=txt(D["atmospheres"]))
    h = hashlib.sha256()
    for f in ("heads.pt", "gallery.npz"):
        h.update(hashlib.sha256((out / f).read_bytes()).digest())
    meta = {"model_version": h.hexdigest()[:12], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "device": args.device,
            "encoders": {"text": f"{J.TEXT_MODEL} (frozen, mean pooled, 384-d, max {J.TEXT_MAXLEN} tokens)",
                         "photo": "DINOv2 ViT-L/14 (frozen, 1024-d, area-weighted mean of the recipe's objects)",
                         "formula": "26 oxides -> fractions -> square root (26-d)"},
            "hyper": cfg, "training": info, "families": fam_info, "eval": ev, "tags_eval": tag_eval,
            "counts": {"recipes": int(len(D["ids"])), "photo": int(D["MI"].sum()), "formula": int(D["MC"].sum()), "description": int(D["MF"].sum())}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    js = J.JointSearch(out, text_encoder=enc)
    meta["extraction_eval"], meta["extraction_examples"] = extraction_check(D, args.seed, log)
    meta["smart_search_eval"] = smart_check(js, D, args.seed, log)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"[joint] model {meta['model_version']} saved to {out} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
