"""Glaze recipe engines side by side, on recipes that NO model trained on.

Engines (card G of the glaze search page, plus two hybrids and a baseline):
  rules        0.80 colour + 0.10 surface + 0.10 cone: no learned model
  colour       colour match as in the app: the team's text-chemistry model (test_similarity.zip), 0.40 learned match +
               0.40 colour + 0.10 surface + 0.10 cone, Orton cones
  smart        smart search as in the app (configs/glaze_joint.json)
  hybrid       smart search + the colour-match learned match as one more score part
  colour+img   colour match + a NEW image head: a cut-out object (DINOv2) -> the colour-match 256-d space, trained on the
               colour-match TRAIN recipes against its frozen chemistry vectors (the team's model is not changed)
The two hybrid weights are cross-fitted: chosen on one half of the test recipes (split by composition group) and scored on
the other half, then the halves swap. No number is tuned on the recipes it is scored on.

Test recipes: in the colour-match TEST split and in the joint model's VAL or TEST split (neither model trained on them).
Searchable: every recipe of both catalogs. Each test recipe is asked six ways:
  colour  its colour only
  menus   its colour + its cone and surface chosen in the page's menus, no words (what the page sends without text)
  words   colour word + cone + surface (the fields colour match was built for) + its colour
  typed   a typed sentence with typos (glaze_text.noisy_variants: never seen in training) + its colour
  photo   its largest cut-out object + its colour. The object is taken out of the recipe's own gallery embedding
          (leave-one-object-out), so no engine has seen the query photo
  all     photo + typed sentence + colour
"its colour" = the recipe's measured glaze colour moved by a random CIELAB offset (sd --jitter per channel), as a
new photo of the same glaze would be.
Metrics: rank of the true recipe (R@1, R@10, MRR); same chemistry in the top 10 (a recipe whose square-root oxide fractions
have cosine >= 0.99 with the true one); chemistry closeness of the top 10 (mean of that cosine); colour precision of the
top 10 (CIEDE2000 <= 10 to the true colour), surface precision, and look precision (same colour AND same surface). Paired bootstrap (2,000) for the differences.
"same_hex" repeats the summary on the test recipes whose colour is the same in both catalogs: the query colour comes from
the joint model's measured palette, so there no engine has a home advantage on colour.

Read only: loads the installed models, writes one JSON. Usage:
  eval_glaze_engines.py [--out multimodal_experiments/work/eval/glaze_engines_<date>.json] [--seed 0] [--jitter 3]"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "multimodal_experiments"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import mm.common  # noqa: E402,F401  (model cache paths, as the app)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from skimage.color import deltaE_ciede2000, rgb2lab  # noqa: E402

from glazeprep.appearance import nearest_color  # noqa: E402
from mm import glaze_joint as J  # noqa: E402
from mm import glaze_reference as GR  # noqa: E402
from mm import glaze_retrieval as GRt  # noqa: E402
from mm import glaze_text as GT  # noqa: E402
from mm.common import EXP  # noqa: E402

SCEN = ("colour", "menus", "words", "typed", "photo", "all")
ENGINES = ("rules", "colour", "smart", "hybrid", "colour+img")
W_HYBRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60)
W_IMAGE = (0.0, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.20)
SAME_CHEM = 0.99


def log(*a):
    print(*a, flush=True)


def hex_lab(h):
    return rgb2lab(GRt.parse_hex(h).reshape(1, 1, 3)).reshape(3)


# ---------------------------------------------------------------- the image head (colour+img)
class ImageHead(nn.Module):
    """the colour-match heads' design: one linear layer, L2-normalised output"""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


def train_image_head(X, y_rows, C, groups, seed, epochs=300, patience=20, temp=0.07, lr=1e-3, batch=256):
    """X: object vectors, y_rows: their recipe's row in C (frozen chemistry vectors). 10 % of the recipes (by composition
    group) stop the training early (object -> recipe R@10 among those recipes). Same-chemistry pairs are no negatives."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    val_g = set(rng.choice(ug, max(1, len(ug) // 10), replace=False))
    va = np.array([g in val_g for g in groups])
    Xt, Ct = torch.tensor(X, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)
    head = ImageHead(X.shape[1], C.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    tr = np.flatnonzero(~va)
    vrec = np.unique(y_rows[va])
    best, state, bad, hist = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        head.train()
        perm = rng.permutation(tr)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            zi, zc = head(Xt[b]), Ct[y_rows[b]]
            same = (zc @ zc.T >= SAME_CHEM) & ~torch.eye(len(b), dtype=torch.bool)
            logits = (zi @ zc.T / temp).masked_fill(same, float("-inf"))
            yy = torch.arange(len(b))
            loss = (F.cross_entropy(logits, yy) + F.cross_entropy(logits.T, yy)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        head.eval()
        with torch.no_grad():
            S = head(Xt[va]).numpy() @ C[vrec].T
        pos = {r: k for k, r in enumerate(vrec)}
        own = np.array([pos[r] for r in y_rows[va]])
        rank = (S > S[np.arange(len(own)), own][:, None]).sum(1)
        r10 = float(np.mean(rank < 10))
        hist.append({"epoch": ep, "loss": round(tot / nb, 4), "val_R@10": round(r10, 4)})
        if r10 > best + 1e-4:
            best, bad, state = r10, 0, {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(state)
    head.eval()
    return head, {"objects_train": int((~va).sum()), "objects_val": int(va.sum()), "recipes_val": int(len(vrec)),
                  "epochs_run": ep, "best_val_R@10": round(best, 4), "val_random_R@10": round(10 / len(vrec), 4),
                  "history": hist[::5] + [hist[-1]]}


# ---------------------------------------------------------------- metrics
def rank_of(s, t):
    """1 = first; ties with other recipes count half"""
    return 1.0 + float(np.sum(s > s[t])) + 0.5 * float(np.sum(s == s[t]) - 1)


def query_metrics(s, t, chem, has_chem, lab, has_lab, surf):
    top = np.argsort(-s, kind="stable")[:10]
    r = rank_of(s, t)
    m = {"rank": r, "R@1": float(r <= 1), "R@10": float(r <= 10), "MRR": 1.0 / r}
    if has_chem[t]:
        ok = top[has_chem[top]]
        cs = chem[ok] @ chem[t]
        m["same_chem@10"] = float((cs >= SAME_CHEM).any())
        m["chem@10"] = float(cs.mean()) if len(cs) else np.nan
    if has_lab[t]:
        ok = top[has_lab[top]]
        de = deltaE_ciede2000(np.broadcast_to(lab[t], (len(ok), 3)), lab[ok])
        m["colour_P@10"] = float(np.mean(de <= 10)) if len(ok) else np.nan
    if surf[t]:
        m["surface_P@10"] = float(np.mean(surf[top] == surf[t]))
        if has_lab[t]:                                            # the look: same colour AND same surface
            de = np.full(len(top), np.inf)
            de[has_lab[top]] = deltaE_ciede2000(np.broadcast_to(lab[t], (int(has_lab[top].sum()), 3)), lab[top[has_lab[top]]])
            m["look_P@10"] = float(np.mean((de <= 10) & (surf[top] == surf[t])))
    return m


def summarise(rows):
    df = pd.DataFrame(rows)
    out = {"n": int(len(df)), "median_rank": float(df["rank"].median())}
    for k in ("R@1", "R@10", "MRR", "same_chem@10", "chem@10", "colour_P@10", "surface_P@10", "look_P@10"):
        if k in df:
            out[k] = round(float(df[k].mean()), 4)
    return out


def bootstrap(a, b, n=2000, seed=0):
    """paired: mean(a - b) and its 95 % interval"""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "better": bool(np.percentile(bs, 2.5) > 0), "worse": bool(np.percentile(bs, 97.5) < 0)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    today = dt.date.today().isoformat()
    ap.add_argument("--out", default=str(EXP / "work" / "eval" / f"glaze_engines_{today}.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=3.0)
    a = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    rng = np.random.default_rng(a.seed)

    log("loading colour match and smart search (CPU) ...")
    CM = GRt.GlazeRetriever(cfg={**GRt.read_config(), "device": "cpu"})
    js = J.JointSearch(text_encoder=J.TextEncoder("cpu", CM.tokenizer, CM.encoder))
    cat = CM.catalog
    crow = {str(r): i for i, r in enumerate(cat["recipe_id"])}
    jrow = {str(r): i for i, r in enumerate(js.g["ids"])}
    D = pd.read_csv(EXP / "work" / "data" / "recipes.csv", dtype=str, keep_default_na=False)
    D = D[D["recipe_id"].isin(crow.keys()) & D["recipe_id"].isin(jrow.keys())].reset_index(drop=True)
    pool = D["recipe_id"].tolist()
    n = len(pool)
    csel = np.array([crow[r] for r in pool])
    jsel = np.array([jrow[r] for r in pool])
    split_cm = cat["split"].to_numpy()[csel]

    # truth for judging (the joint model's recipe bank: measured palette, oxides, surface)
    top_hex = [json.loads(p)[0][0] if p else "" for p in D["palette_json"]]
    has_lab = np.array([bool(h) for h in top_hex])
    lab = np.zeros((n, 3))
    lab[has_lab] = rgb2lab(np.stack([GRt.parse_hex(h) for h in top_hex if h]).reshape(-1, 1, 3)).reshape(-1, 3)
    chem = np.zeros((n, len(J.OXIDES)), np.float32)
    has_chem = np.zeros(n, bool)
    for i, s in enumerate(D["oxides_json"]):
        v = J.chem_input(json.loads(s)) if s else None
        if v is not None:
            chem[i], has_chem[i] = v / np.linalg.norm(v), True
    surf = D["surface"].str.lower().to_numpy()

    test = np.flatnonzero((split_cm == "test") & D["split"].isin(["val", "test"]).to_numpy() & has_lab)
    log(f"pool {n:,} recipes (in both catalogs) · test {len(test)} (colour-match test AND joint val/test, with a measured colour)")

    # objects (DINOv2 cache of the cut-out objects, as the joint model was trained)
    with np.load(EXP / "cache" / "img_obj_dinov2.npz", allow_pickle=False) as z:
        X_obj, obj_ids = z["X"].astype(np.float32), z["ids"]
    opos = {str(o): k for k, o in enumerate(obj_ids)}
    O = pd.read_csv(EXP / "work" / "data" / "objects.csv", dtype=str)
    prow = {r: i for i, r in enumerate(pool)}
    objs = {}
    for oid, rid, area in zip(O["object_id"], O["recipe_id"], O["area_px"]):
        if oid in opos and rid in prow:
            objs.setdefault(prow[rid], []).append((opos[oid], float(area)))

    # ---- the image head for colour+img: colour-match TRAIN recipes only
    tr_rows = [i for i in np.flatnonzero(split_cm == "train") if i in objs]
    Xi, yi, gi = [], [], []
    for i in tr_rows:
        for k, _ in objs[i]:
            Xi.append(X_obj[k] / np.linalg.norm(X_obj[k]))
            yi.append(csel[i])
            gi.append(D["group_id"].iat[i])
    log(f"training the image head on {len(tr_rows):,} colour-match train recipes ({len(Xi):,} objects) ...")
    t0 = time.time()
    head, head_info = train_image_head(np.stack(Xi), np.array(yi), CM.vectors.astype(np.float32), np.array(gi), a.seed)
    head_info["seconds"] = round(time.time() - t0, 1)
    log(f"  best val R@10 {head_info['best_val_R@10']} (random {head_info['val_random_R@10']}) after {head_info['epochs_run']} epochs, {head_info['seconds']} s")

    # ---- leave-one-object-out: the target recipe's joint embedding without the query object
    tags_of = lambda i: (f"{js.g['colour_word'][jsel[i]]} glaze. {D['desc_tags'].iat[i]}".strip() if js.g["colour_word"][jsel[i]] else D["desc_tags"].iat[i])  # noqa: E731
    TS = js.text([tags_of(i) for i in test])

    @torch.no_grad()
    def fused_without(i, ts, drop):
        parts = [js.heads["text"](torch.tensor(ts[None]))[0].numpy()]
        rest = [(k, ar) for k, ar in objs.get(i, []) if k != drop]
        if rest:
            w = np.array([ar for _, ar in rest], np.float32)
            v = (X_obj[[k for k, _ in rest]] * (w / w.sum())[:, None]).sum(0)
            parts.append(js.heads["photo"](torch.tensor((v / max(1e-8, np.linalg.norm(v)))[None]))[0].numpy())
        if has_chem[i]:
            ox = J.chem_input(json.loads(D["oxides_json"].iat[i]))
            parts.append(js.heads["formula"](torch.tensor(ox[None]))[0].numpy())
        f = np.sum(parts, 0)
        return f / max(1e-8, np.linalg.norm(f))

    check = [float(fused_without(i, TS[k], -1) @ js.g["fused"][jsel[i]]) for k, i in enumerate(test[:50])]
    log(f"check: rebuilt joint embeddings vs the installed gallery, min cosine {min(check):.6f} (1 = same)")

    # ---- queries
    queries = []
    for k, i in enumerate(test):
        word = nearest_color(tuple(int(top_hex[i][j:j + 2], 16) for j in (1, 3, 5)))["name"]
        jl = lab[i] + rng.normal(0, a.jitter, 3) if a.jitter > 0 else lab[i]
        hq = GRt.lab_hex(jl)
        cone = J.cone_text(D["cone_lo"].iat[i], D["cone_hi"].iat[i])
        surface = D["surface"].iat[i] or None
        m = re.search(r"Subtype: (.*?)\.", D["desc_tags"].iat[i])
        facts = {"colour": word, "surface": surface, "transparency": D["transparency"].iat[i] or None,
                 "cone_lo": D["cone_lo"].iat[i] or None, "cone_hi": D["cone_hi"].iat[i] or None,
                 "atmospheres": [x for x in D["atmospheres"].iat[i].split(";") if x],
                 "subtype": re.sub(r"\bglaze\b", "", m.group(1)).strip() if m else ""}
        typed = GT.noisy_variants(facts, 1, 10_000 + int(D["recipe_id"].iat[i]) % 100_000)[0]
        o = max(objs[i], key=lambda x: x[1])[0] if i in objs else None
        queries.append({"row": int(i), "recipe_id": D["recipe_id"].iat[i], "word": word, "hex": hq,
                        "dE_jitter": round(float(deltaE_ciede2000(lab[i], jl)), 2), "cone": cone, "surface": surface,
                        "typed": typed, "object": o, "ts": TS[k]})

    def cm_fields(text, hq, word=None, cone=None, surface=None):
        """what colour match receives: the typed words read by the page's reader (glaze_text.extract), or the given fields"""
        if text is not None:
            u = GT.extract(text)
            cone = u["cone"]["value"] if u["cone"] else None
            surface = u["surface"]["value"] if u["surface"] else None
            word = CM.color_name(u["colour"]["hex"])["name"] if u["colour"] else None
        try:
            surface = GR.normalize_surface(surface, CM.surfaces)
        except ValueError:
            surface = None
        try:
            GR.normalize_cone(cone)
        except ValueError:
            cone = None
        q = CM.query({"hex": hq, "color_name": word or "", "cone": cone, "surface": surface})
        rk = CM.rank(q["color_name"], q["surface_text"], q["cone_text"], q["hex"]).sort_index()
        return {"final": rk["final_score"].to_numpy()[csel], "latent": rk["latent_score"].to_numpy()[csel],
                "rules": (0.8 * rk["color_score"].fillna(0) + 0.1 * rk["surface_match"].fillna(0) + 0.1 * rk["cone_match"].fillna(0)).to_numpy()[csel],
                "model_text": f"color: {q['color_name']}, Cone {q['cone_text']}, {q['surface_text']}, unknown, unknown"}

    def smart(p, q, photo):
        r = js.score(p, photo)
        t = q["row"]
        S = {k: v[jsel].copy() for k, v in r["S"].items()}
        if photo is not None or "text" in S:                        # the target recipe's embedding without the query photo
            f = fused_without(t, q["ts"], q["object"] if photo is not None else -1)
            if "text" in S:
                S["text"][t] = (f @ js._q("text", js.text([r["query"]["text"]])[0]) + 1) / 2
            if photo is not None:
                S["photo"][t] = (f @ js._q("photo", photo) + 1) / 2
        num = sum(r["w"][k] * S[k] for k in S)
        return num, r["tot"]

    results = {s: {e: [] for e in ENGINES} for s in SCEN}
    raw = {s: [] for s in SCEN}                                   # per query: what the hybrids need, to tune their weights
    t0 = time.time()
    for qi, q in enumerate(queries):
        t = q["row"]
        photo = None
        if q["object"] is not None:
            photo = X_obj[q["object"]] / np.linalg.norm(X_obj[q["object"]])
            with torch.no_grad():
                img_lat = (CM.vectors[csel] @ head(torch.tensor(photo[None]))[0].numpy() + 1) / 2
        for s in SCEN:
            if s in ("photo", "all") and photo is None:
                continue
            if s == "colour":
                p = {"hex": q["hex"]}
                c = cm_fields(None, q["hex"])
            elif s == "menus":
                p = {"hex": q["hex"], "cone": q["cone"], "surface": q["surface"]}
                c = cm_fields(None, q["hex"], None, q["cone"], q["surface"])
            elif s == "words":
                text = " ".join(x for x in (q["word"], (q["surface"] or "") + " glaze", f"cone {q['cone']}" if q["cone"] else "") if x.strip())
                p = {"text": text, "hex": q["hex"], "cone": q["cone"], "surface": q["surface"]}
                c = cm_fields(None, q["hex"], q["word"], q["cone"], q["surface"])
            elif s == "typed":
                p = {"text": q["typed"], "hex": q["hex"]}
                c = cm_fields(q["typed"], q["hex"])
            elif s == "photo":
                p = {"hex": q["hex"]}
                c = cm_fields(None, q["hex"])
            else:
                p = {"text": q["typed"], "hex": q["hex"]}
                c = cm_fields(q["typed"], q["hex"])
            num, tot = smart(p, q, photo if s in ("photo", "all") else None)
            raw[s].append({"q": qi, "t": t, "num": num, "tot": tot, "lat": c["latent"], "cm": c["final"],
                           "img": img_lat if s in ("photo", "all") else None})
            for e, sc in (("rules", c["rules"]), ("colour", c["final"]), ("smart", num / tot)):
                results[s][e].append({"q": qi, **query_metrics(sc, t, chem, has_chem, lab, has_lab, surf)})
            if qi == 0:
                q.setdefault("model_text", {})[s] = c["model_text"]
        if (qi + 1) % 50 == 0:
            log(f"  {qi + 1}/{len(queries)} queries, {time.time() - t0:.0f} s")

    # ---- the hybrids: cross-fitted weights (2 halves by composition group)
    grp = D["group_id"].to_numpy()
    half = {q["row"]: zlib.crc32(str(grp[q["row"]]).encode()) & 1 for q in queries}   # a whole composition group in one half

    def hyb(rr, w):
        return (rr["num"] + w * rr["lat"]) / (rr["tot"] + w)

    def img(rr, w):
        return (rr["cm"] + w * rr["img"]) / (1 + w)

    def mrr_of(fn, w, rows):
        return float(np.mean([1.0 / rank_of(fn(rr, w), rr["t"]) for rr in rows]))

    tuned, curves = {}, {}
    for name, fn, grid, scen in (("hybrid", hyb, W_HYBRID, SCEN), ("colour+img", img, W_IMAGE, ("photo", "all"))):
        curves[name] = {s: [round(mrr_of(fn, w, raw[s]), 4) for w in grid] for s in scen}
        tuned[name] = {}
        for h in (0, 1):
            tune_rows = {s: [rr for rr in raw[s] if half[rr["t"]] == h] for s in scen}
            score = [np.mean([mrr_of(fn, w, tune_rows[s]) for s in scen]) for w in grid]
            w_best = grid[int(np.argmax(score))]
            tuned[name][f"chosen_on_half_{h}"] = w_best
            for s in scen:
                for rr in raw[s]:
                    if half[rr["t"]] != h:                         # scored on the OTHER half
                        results[s][name].append({"q": rr["q"], **query_metrics(fn(rr, w_best), rr["t"], chem, has_chem, lab, has_lab, surf)})
        all_rows = {s: raw[s] for s in scen}
        tuned[name]["all_test_best"] = grid[int(np.argmax([np.mean([mrr_of(fn, w, all_rows[s]) for s in scen]) for w in grid]))]

    # ---- summaries and paired comparisons
    summary = {s: {e: summarise(results[s][e]) for e in ENGINES if results[s][e]} for s in SCEN}
    her_hex = cat["hex_color"].fillna("").str.upper().to_numpy()[csel]
    same = {qi for qi, q in enumerate(queries) if her_hex[q["row"]] == top_hex[q["row"]].upper()}
    summary_same_hex = {s: {e: summarise([r for r in results[s][e] if r["q"] in same]) for e in ENGINES if results[s][e]} for s in SCEN}
    compare = {}
    for s in SCEN:
        by = {e: pd.DataFrame(results[s][e]).set_index("q").sort_index() for e in ENGINES if results[s][e]}
        compare[s] = {}
        for ref in ("smart", "colour"):
            for e in by:
                if e == ref:
                    continue
                common = by[e].index.intersection(by[ref].index)
                compare[s][f"{e} vs {ref}"] = {m: bootstrap(by[e].loc[common, m], by[ref].loc[common, m], seed=a.seed)
                                               for m in ("MRR", "R@10", "same_chem@10", "chem@10", "colour_P@10", "surface_P@10", "look_P@10") if m in by[e]}
    for s in SCEN:
        log(f"\n[{s}] n={summary[s]['smart']['n']}")
        log(f"  {'engine':11s} {'R@1':>6s} {'R@10':>6s} {'MRR':>6s} {'chem10':>7s} {'samechem':>8s} {'colourP':>8s} {'surfP':>6s} {'lookP':>6s}")
        for e, m in summary[s].items():
            log(f"  {e:11s} {m['R@1']:6.3f} {m['R@10']:6.3f} {m['MRR']:6.3f} {m.get('chem@10', float('nan')):7.3f} "
                f"{m.get('same_chem@10', float('nan')):8.3f} {m.get('colour_P@10', float('nan')):8.3f} {m.get('surface_P@10', float('nan')):6.3f} {m.get('look_P@10', float('nan')):6.3f}")
    rand = {"R@10": round(10 / n, 4), "chem@10": round(float(np.mean(chem[has_chem][rng.integers(0, has_chem.sum(), 4000)]
                                                                 @ chem[has_chem][rng.integers(0, has_chem.sum(), 4000)].T)), 4)}
    out = {"date": dt.date.today().isoformat(), "seconds": round(time.time() - t_start), "seed": a.seed, "colour_jitter_sd": a.jitter,
           "models": {"colour_match": CM.version, "smart_search": js.version},
           "pool": n, "test_recipes": int(len(test)), "photo_queries": int(sum(q["object"] is not None for q in queries)),
           "median_dE_of_jitter": float(np.median([q["dE_jitter"] for q in queries])),
           "check_rebuilt_embedding_min_cos": round(min(check), 6), "random": rand,
           "image_head": head_info, "hybrid_weights": tuned, "weight_grids": {"hybrid": W_HYBRID, "colour+img": W_IMAGE}, "mrr_curves_all_test": curves,
           "summary": summary, "compare": compare, "same_hex_queries": len(same), "summary_same_hex": summary_same_hex,
           "examples": [{k: v for k, v in q.items() if k != "ts"} for q in queries[:5]]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
    torch.save({"state_dict": head.state_dict(), "in_dim": int(X_obj.shape[1]), "shared_dim": int(CM.vectors.shape[1]),
                "trained_on": "colour-match train recipes", "colour_match_version": CM.version, "date": out["date"], "info": head_info},
               Path(a.out).with_suffix(".image_head.pt"))
    log(f"\nsaved {a.out} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
