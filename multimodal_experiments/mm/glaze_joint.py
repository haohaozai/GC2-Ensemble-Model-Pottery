"""Smart search: one joint model for free text + photo + glaze formula (card G; POST /api/glaze/smart_search).

Frozen pretrained encoders -> trained projection heads -> one 256-d space (L2-normalised):
  text     all-MiniLM-L6-v2, mean pooled (384-d). Recipe text = colour word + cleaned tags (Orton cones kept: 04 != 4),
           noisy sentences (glaze_text.noisy_variants: typos, shorthand, filler words) and the recipe's own description
  photo    DINOv2 ViT-L/14 (1024-d): area-weighted mean of the recipe's segmented objects (same as the object search)
  formula  26 oxides -> fractions -> square root (26-d)
Training (scripts/train_glaze_joint.py): symmetric InfoNCE for text-photo, text-formula and photo-formula on the TRAIN
split of the recipe bank (composition groups); recipes of the same group are no negatives; early stop on VAL.
Search: each given query part (text / photo / formula) is compared with the recipe's fused embedding, plus CIEDE2000
to the recipe's palette (best colour with >= 10 % of the area), Orton cone overlap and surface / transparency /
atmosphere matches from the understood text. Weights: configs/glaze_joint.json, renormalised over the parts given.
Families = K-Means on the fused embeddings (k by silhouette, train split) -> "Explore formulations".
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import glaze_text as GT
from .common import CODE, EXP
from .glaze_retrieval import GLAZY_CDN, GLAZY_RECIPE, cone_range_orton, parse_hex, rgb_hex, thumb_url

OXIDES = GT.OXIDES
CONFIG = CODE / "configs" / "glaze_joint.json"
DEFAULTS = {"model_dir": "multimodal_experiments/models/glaze_joint",
            "weights": {"text": 0.35, "photo": 0.35, "formula": 0.25, "colour": 0.30, "cone": 0.10, "surface": 0.10,
                        "transparency": 0.05, "atmosphere": 0.05},
            "max_per_family": 2, "top_k": 10, "device": "cpu"}
TEXT_MODEL, TEXT_MAXLEN = "sentence-transformers/all-MiniLM-L6-v2", 128
PARTS = ("text", "photo", "formula", "colour", "cone", "surface", "transparency", "atmosphere")


def read_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG.is_file():
        cfg.update({k: v for k, v in json.loads(CONFIG.read_text(encoding="utf-8")).items() if not k.startswith("_")})
    return cfg


class Head(nn.Module):
    def __init__(self, d_in: int, hidden: int = 512, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TextEncoder:
    """frozen MiniLM, mean pooled (the same encoder as the colour-match model; one instance can be shared)."""

    def __init__(self, device="cpu", tokenizer=None, encoder=None):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(device)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.enc = (encoder or AutoModel.from_pretrained(TEXT_MODEL)).to(self.device).eval()
        self.lock = threading.Lock()

    @torch.no_grad()
    def __call__(self, texts: list[str], batch: int = 256) -> np.ndarray:
        out = []
        with self.lock:
            for i in range(0, len(texts), batch):
                b = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=TEXT_MAXLEN, return_tensors="pt").to(self.device)
                tok = self.enc(**b)[0]
                m = b["attention_mask"].unsqueeze(-1).float()
                out.append(((tok * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def chem_input(oxides: dict) -> np.ndarray | None:
    """oxide weight % -> sqrt of the fractions over the 26 oxides; None with fewer than 3 oxides."""
    v = np.array([max(0.0, float(oxides.get(o, 0) or 0)) for o in OXIDES], np.float64)
    if (v > 0).sum() < 3 or v.sum() <= 0:
        return None
    return np.sqrt(v / v.sum()).astype(np.float32)


def lab_of(hexes) -> np.ndarray:
    from skimage.color import rgb2lab
    return rgb2lab(np.stack([parse_hex(h) for h in hexes]).reshape(-1, 1, 3)).reshape(-1, 3)


def cone_text(lo, hi) -> str | None:
    lo, hi = (str(lo) if isinstance(lo, str) and lo else ""), (str(hi) if isinstance(hi, str) and hi else "")
    return None if not lo else lo if not hi or hi == lo else f"{lo} - {hi}"


# ---------------------------------------------------------------- training data
def build_data(encode: TextEncoder, n_noisy: int = 8, seed: int = 0, log=print) -> dict:
    """recipe bank (work/data) + DINOv2 object cache -> arrays for training, evaluation and the search gallery."""
    import pandas as pd
    from glazeprep.appearance import nearest_color
    R = pd.read_csv(EXP / "work" / "data" / "recipes.csv", dtype=str, keep_default_na=False)
    n = len(R)
    ids = R["recipe_id"].tolist()
    # colour: the recipe's palette (measured on its segmented photo objects)
    pal = [json.loads(p) if p else [] for p in R["palette_json"]]
    top = [p[0][0] if p else None for p in pal]
    words = [nearest_color(tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)))["name"] if h else "" for h in top]
    P_lab = np.full((n, 3, 3), np.nan, np.float32)
    P_frac = np.zeros((n, 3), np.float32)
    for i, p in enumerate(pal):
        for k, (h, fr) in enumerate(p[:3]):
            P_lab[i, k] = lab_of([h])[0]
            P_frac[i, k] = fr
    # text views
    sub = [re.sub(r"\bglaze\b", "", m.group(1)).strip() if (m := re.search(r"Subtype: (.*?)\.", t)) else "" for t in R["desc_tags"]]
    atm = [[a.strip() for a in re.split(r"[;,]", s) if a.strip()] for s in R["atmospheres"]]
    tags = [f"{w} glaze. {t}".strip() if w else t for w, t in zip(words, R["desc_tags"])]
    facts = [{"colour": w, "surface": s, "transparency": tr, "cone_lo": lo, "cone_hi": hi, "atmospheres": a, "subtype": sb}
             for w, s, tr, lo, hi, a, sb in zip(words, R["surface"], R["transparency"], R["cone_lo"], R["cone_hi"], atm, sub)]
    free = [f"{nm}. {d}".strip(". ") if d.strip() else "" for nm, d in zip(R["name"], R["desc_free"])]
    t0 = time.time()
    TS = encode(tags)
    noisy = [GT.noisy_variants(f, n_noisy, seed * 100003 + i) for i, f in enumerate(facts)]
    TN = encode([s for v in noisy for s in v]).reshape(n, n_noisy, -1)
    MF = np.array([bool(t) for t in free])
    TF = np.zeros_like(TS)
    TF[MF] = encode([t for t in free if t])
    log(f"[joint] text views encoded in {time.time() - t0:.0f} s: {n} tag texts, {n * n_noisy} noisy sentences, {MF.sum()} descriptions")
    # photo: area-weighted mean of the recipe's object DINOv2 vectors, L2-normalised
    with np.load(EXP / "cache" / "img_obj_dinov2.npz", allow_pickle=False) as z:   # read once: z[...] reloads the array
        X_obj, obj_ids = z["X"], z["ids"]
    pos = {str(o): k for k, o in enumerate(obj_ids)}
    O = pd.read_csv(EXP / "work" / "data" / "objects.csv", dtype=str)
    IM = np.zeros((n, X_obj.shape[1]), np.float32)
    MI = np.zeros(n, bool)
    row = {r: i for i, r in enumerate(ids)}
    acc: dict[int, list] = {}
    for oid, rid, area in zip(O["object_id"], O["recipe_id"], O["area_px"]):
        if oid in pos and rid in row:
            acc.setdefault(row[rid], []).append((pos[oid], float(area)))
    for i, lst in acc.items():
        w = np.array([a for _, a in lst], np.float32)
        IM[i] = (X_obj[[k for k, _ in lst]] * (w / w.sum())[:, None]).sum(0)
        IM[i] /= max(1e-8, float(np.linalg.norm(IM[i])))
        MI[i] = True
    # formula
    CH = np.zeros((n, len(OXIDES)), np.float32)
    MC = np.zeros(n, bool)
    for i, s in enumerate(R["oxides_json"]):
        v = chem_input(json.loads(s)) if s else None
        if v is not None:
            CH[i], MC[i] = v, True
    return {"ids": np.array(ids), "split": R["split"].to_numpy(), "group": R["group_id"].to_numpy(), "name": R["name"].to_numpy(),
            "TS": TS, "TN": TN, "TF": TF, "MF": MF, "IM": IM, "MI": MI, "CH": CH, "MC": MC, "P_lab": P_lab, "P_frac": P_frac,
            "colour_word": np.array(words), "top_hex": np.array([h or "" for h in top]), "facts": facts, "free": free, "noisy": noisy,
            "cone_lo": R["cone_lo"].to_numpy(), "cone_hi": R["cone_hi"].to_numpy(), "surface": R["surface"].to_numpy(),
            "transparency": R["transparency"].to_numpy(), "atmospheres": np.array([";".join(a) for a in atm])}


# ---------------------------------------------------------------- training
def info_nce(a, b, g, temp):
    logits = a @ b.T / temp
    same = (g[:, None] == g[None, :]) & ~torch.eye(len(g), dtype=torch.bool, device=a.device)
    logits = logits.masked_fill(same, float("-inf"))
    y = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


def train(D: dict, cfg: dict, device: str, log=print) -> tuple[dict, dict]:
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = torch.device(device)
    heads = {"text": Head(D["TS"].shape[1], cfg["hidden"], cfg["dim"], cfg["dropout"]),
             "photo": Head(D["IM"].shape[1], cfg["hidden"], cfg["dim"], cfg["dropout"]),
             "formula": Head(D["CH"].shape[1], cfg["hidden"], cfg["dim"], cfg["dropout"])}
    for h in heads.values():
        h.to(dev)
    opt = torch.optim.AdamW([p for h in heads.values() for p in h.parameters()], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tr = np.flatnonzero(D["split"] == "train")
    gid = {g: k for k, g in enumerate(np.unique(D["group"]))}
    G = torch.tensor([gid[g] for g in D["group"]], device=dev)
    T = {k: torch.tensor(D[k], device=dev) for k in ("TS", "TN", "TF", "IM", "CH")}
    M = {"text": torch.ones(len(G), dtype=torch.bool, device=dev), "photo": torch.tensor(D["MI"], device=dev), "formula": torch.tensor(D["MC"], device=dev)}
    MF = torch.tensor(D["MF"], device=dev)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, cfg["max_epochs"] + 1):
        for h in heads.values():
            h.train()
        perm, tot, nb = rng.permutation(tr), 0.0, 0
        for s in range(0, len(perm), cfg["batch"]):
            b = torch.tensor(perm[s:s + cfg["batch"]], device=dev)
            r = torch.rand(len(b), device=dev)
            k = torch.randint(0, T["TN"].shape[1], (len(b),), device=dev)
            xt = torch.where((r < cfg["p_tags"])[:, None], T["TS"][b], T["TN"][b, k])
            use_free = (r >= cfg["p_tags"]) & (r < cfg["p_tags"] + cfg["p_free"]) & MF[b]
            xt = torch.where(use_free[:, None], T["TF"][b], xt)
            z = {"text": heads["text"](xt), "photo": heads["photo"](T["IM"][b]), "formula": heads["formula"](T["CH"][b])}
            loss = 0.0
            for a, c in (("text", "photo"), ("text", "formula"), ("photo", "formula")):
                m = M[a][b] & M[c][b]
                if m.sum() > 2:
                    loss = loss + info_nce(z[a][m], z[c][m], G[b][m], cfg["temperature"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss), nb + 1
        ev = evaluate(heads, D, "val", dev, seed=cfg["seed"] + 7)
        score = float(np.mean([ev[t]["R@10"] for t in ("noisy text -> recipe", "photo -> recipe", "formula -> recipe")]))
        hist.append({"epoch": ep, "loss": round(tot / max(1, nb), 4), "val_mean_R@10": round(score, 4)})
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {m: {k: v.detach().cpu().clone() for k, v in h.state_dict().items()} for m, h in heads.items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            log(f"[joint] epoch {ep:3d} loss {tot / max(1, nb):.3f} val mean R@10 {score:.3f}{' *' if bad == 0 else ''}")
        if bad >= cfg["patience"]:
            break
    for m, h in heads.items():
        h.load_state_dict(best_state[m])
        h.eval()
    return heads, {"epochs_run": ep, "best_val_mean_R@10": round(best, 4), "history": hist}


@torch.no_grad()
def embed(heads, D, dev, rows=None, text_key="TS") -> dict:
    rows = np.arange(len(D["ids"])) if rows is None else rows
    f = lambda m, X: heads[m](torch.tensor(X[rows], device=dev)).cpu().numpy()  # noqa: E731
    return {"text": f("text", D[text_key]), "photo": f("photo", D["IM"]), "formula": f("formula", D["CH"])}


def fuse(Z: dict, masks: dict, parts) -> np.ndarray:
    S = sum(Z[p] * masks[p][:, None] for p in parts)
    return S / np.linalg.norm(S, axis=1, keepdims=True).clip(1e-8)


def ranks(Q, Gal, gq, gg) -> np.ndarray:
    """rank of the query's own recipe in the gallery (1 = first); a recipe of the same composition group counts as found."""
    order = np.argsort(-(Q @ Gal.T), axis=1)
    return np.array([int(np.flatnonzero(gg[o] == t)[0]) + 1 for o, t in zip(order, gq)])


def _summary(rk, n_gal) -> dict:
    return {"n": int(len(rk)), "gallery": int(n_gal), **{f"R@{k}": round(float(np.mean(rk <= k)), 4) for k in (1, 5, 10)},
            "median_rank": int(np.median(rk)), "random_R@10": round(10 / n_gal, 4)}


@torch.no_grad()
def evaluate(heads, D, split, dev, seed=123, baseline_encode=None) -> dict:
    for h in heads.values():
        h.eval()
    rows = np.flatnonzero(D["split"] == split)
    g = D["group"][rows]
    Z = embed(heads, D, dev, rows)
    M = {"text": np.ones(len(rows), bool), "photo": D["MI"][rows], "formula": D["MC"][rows]}
    noisy = np.stack([D["TN"][r, (seed + i) % D["TN"].shape[1]] for i, r in enumerate(rows)])
    Zn = heads["text"](torch.tensor(noisy, device=dev)).cpu().numpy()
    tasks = {"tag text -> recipe": (Z["text"], fuse(Z, M, ("photo", "formula")), np.ones(len(rows), bool)),
             "noisy text -> recipe": (Zn, fuse(Z, M, ("photo", "formula")), np.ones(len(rows), bool)),
             "description -> recipe": (heads["text"](torch.tensor(D["TF"][rows], device=dev)).cpu().numpy(), fuse(Z, M, ("photo", "formula")), D["MF"][rows]),
             "photo -> recipe": (Z["photo"], fuse(Z, M, ("text", "formula")), M["photo"]),
             "formula -> recipe": (Z["formula"], fuse(Z, M, ("text", "photo")), M["formula"])}
    out = {name: _summary(ranks(Q[ok], Gal, g[ok], g), len(rows)) for name, (Q, Gal, ok) in tasks.items() if ok.any()}
    if baseline_encode:                                   # pretrained MiniLM only: noisy sentence vs the recipe's tag text
        Tn = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
        Ts = D["TS"][rows] / np.linalg.norm(D["TS"][rows], axis=1, keepdims=True)
        out["noisy text -> tag text (pretrained MiniLM only, no training)"] = _summary(ranks(Tn, Ts, g, g), len(rows))
        out["noisy text -> recipe (trained, all parts)"] = _summary(ranks(Zn, fuse(Z, M, ("text", "photo", "formula")), g, g), len(rows))
    return out


def families(F_all: np.ndarray, train_rows: np.ndarray, seed: int = 0, log=print) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    X = F_all[train_rows]
    smp = rng.choice(len(X), min(2000, len(X)), replace=False)
    diag = []
    for k in (8, 10, 12, 15, 20):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        diag.append({"k": k, "silhouette": round(float(silhouette_score(X[smp], km.labels_[smp])), 4)})
    k = max(diag, key=lambda d: (d["silhouette"], -d["k"]))["k"]
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    hd = HDBSCAN(min_cluster_size=15, algorithm="brute").fit(X)            # tree search is very slow in 256-d
    info = {"kmeans_k": k, "kmeans_candidates": diag, "hdbscan_min_cluster_size": 15,
            "hdbscan_clusters": int(len(set(hd.labels_)) - (1 if -1 in hd.labels_ else 0)), "hdbscan_noise": round(float(np.mean(hd.labels_ == -1)), 4)}
    log(f"[joint] families: K-Means k={k} (silhouette {max(d['silhouette'] for d in diag)}); HDBSCAN for comparison: "
        f"{info['hdbscan_clusters']} clusters, {info['hdbscan_noise']:.0%} noise")
    return km.predict(F_all).astype(np.int16), info


# ---------------------------------------------------------------- search
class JointSearch:
    """the trained heads + recipe gallery, loaded once; understand() and search() are thread safe."""

    def __init__(self, folder=None, text_encoder: TextEncoder | None = None, catalog_urls: dict | None = None, names: dict | None = None,
                 ingredients: dict | None = None):
        t0 = time.time()
        self.cfg = read_config()
        self.dir = Path(folder) if folder else (CODE / self.cfg["model_dir"])
        if not (self.dir / "heads.pt").is_file():
            raise FileNotFoundError(f"smart search model missing in {self.dir} - run INSTALL_GLAZE_MODEL.bat (it trains it)")
        ck = torch.load(self.dir / "heads.pt", map_location="cpu", weights_only=True)
        self.heads = {}
        for m, sd in ck["heads"].items():
            h = Head(int(ck["in_dims"][m]), int(ck["hidden"]), int(ck["dim"]), 0.0)
            h.load_state_dict(sd)
            self.heads[m] = h.eval()
        with np.load(self.dir / "gallery.npz", allow_pickle=False) as z:
            self.g = {k: z[k] for k in z.files}
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.tags = None
        if (self.dir / "text_tags.joblib").is_file():
            import joblib
            self.tags = joblib.load(self.dir / "text_tags.joblib")
        self.text = text_encoder or TextEncoder(self.cfg.get("device", "cpu"))
        self.urls = catalog_urls or {}
        self.names = names or {}
        self.ingredients = ingredients or {}
        self.n = len(self.g["ids"])
        lo = [cone_range_orton(cone_text(a, b)) if cone_text(a, b) else None for a, b in zip(self.g["cone_lo"], self.g["cone_hi"])]
        self.cone_lo = np.array([r[0] if r else np.nan for r in lo])
        self.cone_hi = np.array([r[1] if r else np.nan for r in lo])
        self.version = self.meta.get("model_version", "?")
        self.load_s = round(time.time() - t0, 2)

    def describe(self) -> str:
        return f"{self.n:,} recipes, joint model {self.version}, {self.load_s} s"

    @torch.no_grad()
    def _q(self, part: str, x: np.ndarray) -> np.ndarray:
        return self.heads[part](torch.tensor(x[None].astype(np.float32))).numpy()[0]

    def understand(self, text: str, colour_name=None) -> dict:
        u = GT.extract(text)
        if u["colour"] and colour_name:
            u["colour"]["model_word"] = colour_name(u["colour"]["hex"])
        u["inferred"] = self.tags.predict(text) if (self.tags and text.strip()) else {}
        return u

    def search(self, p: dict, photo_vec: np.ndarray | None = None) -> dict:
        from skimage.color import deltaE_ciede2000
        from .glaze_reference import normalize_cone
        t0 = time.perf_counter()
        w = {**self.cfg["weights"], **{k: float(v) for k, v in (p.get("weights") or {}).items() if k in PARTS}}
        text = re.sub(r"\s+", " ", str(p.get("text") or "")).strip()[:500]
        u = GT.extract(text) if text else None
        S, used, q = {}, {}, {"text": text or None}
        G = self.g["fused"]
        if text:
            S["text"] = (G @ self._q("text", self.text([text])[0]) + 1) / 2
        if photo_vec is not None:
            S["photo"] = (G @ self._q("photo", photo_vec) + 1) / 2
        ox = p.get("oxides") or (u["oxides"] if u and u["oxides"] else None)
        cv = chem_input(ox) if ox else None
        if cv is not None:
            S["formula"] = (G @ self._q("formula", cv) + 1) / 2
            q["oxides"] = {k: v for k, v in ox.items() if float(v or 0) > 0}
        hx = p.get("hex") or (u["colour"]["hex"] if u and u["colour"] else None)
        if hx and parse_hex(hx) is not None:
            hx = rgb_hex(parse_hex(hx))
            ql = lab_of([hx])[0]
            P, fr = self.g["P_lab"], self.g["P_frac"]
            de = deltaE_ciede2000(np.broadcast_to(ql, P.shape).reshape(-1, 3), np.nan_to_num(P.reshape(-1, 3))).reshape(P.shape[:2])
            de[(fr < 0.10) | np.isnan(P[..., 0])] = np.inf
            best = de.min(1)
            S["colour"] = np.where(np.isfinite(best), np.exp(-best / 20.0), 0.0)
            used["delta_e"] = best
            q["hex"] = hx
        cone = normalize_cone(p.get("cone") or (u["cone"]["value"] if u and u["cone"] else None))
        if cone:
            a = cone_range_orton(cone)
            known = ~np.isnan(self.cone_lo)
            c = np.zeros(self.n)
            c[known] = (np.maximum(a[0], self.cone_lo[known]) <= np.minimum(a[1], self.cone_hi[known])).astype(float)
            S["cone"], q["cone"] = c, cone
        for part, key, src in (("surface", "surface", "surface"), ("transparency", "transparency", "transparency")):
            val = p.get(key) or (u[src]["value"] if u and u[src] else None)
            if val:
                S[part] = (np.char.lower(self.g[key].astype(str)) == str(val).lower()).astype(float)
                q[key] = val
        atms = p.get("atmospheres") or ([a["value"] for a in u["atmospheres"]] if u else [])
        if atms:
            al = {a.lower() for a in atms}
            S["atmosphere"] = np.array([float(bool(al & set(x.split(";")))) for x in np.char.lower(self.g["atmospheres"].astype(str))])
            q["atmospheres"] = atms
        if not S:
            raise ValueError("give at least a text, a colour, a photo object or a formula")
        tot = sum(w[k] for k in S)
        if tot <= 0:
            raise ValueError("no part of this query has a weight above 0")
        score = sum(w[k] * S[k] for k in S) / tot
        order = np.lexsort((-S.get("text", np.zeros(self.n)), -score))
        k, mpf = int(p.get("top_k") or self.cfg["top_k"]), int(self.cfg["max_per_family"])
        best = order[:k]
        chosen, cnt = [], {}
        for i in order:
            f = int(self.g["family"][i])
            if cnt.get(f, 0) < mpf:
                chosen.append(i)
                cnt[f] = cnt.get(f, 0) + 1
            if len(chosen) == k:
                break
        card = lambda i, r: self._card(i, r, S, w, tot, score, used)  # noqa: E731
        return {"query": q, "understood": u, "parts": {k2: round(w[k2] / tot, 4) for k2 in S}, "model_version": self.version,
                "weighted_results": [card(i, r) for r, i in enumerate(best, 1)], "diverse_results": [card(i, r) for r, i in enumerate(chosen, 1)],
                "max_per_cluster": mpf, "searched": self.n, "warnings": [],
                "timings": {"retrieval_ms": round(1000 * (time.perf_counter() - t0), 1), "total_ms": round(1000 * (time.perf_counter() - t0), 1)}}

    def _card(self, i, rank, S, w, tot, score, used) -> dict:
        rid = int(self.g["ids"][i])
        img = self.urls.get(rid)
        hx = str(self.g["top_hex"][i]) or None
        de = used.get("delta_e")
        return {"rank": rank, "recipe_id": rid, "recipe_name": self.names.get(rid) or str(self.g["name"][i]) or None,
                "recipe_url": GLAZY_RECIPE.format(rid), "image_url": img, "thumb_url": thumb_url(img) or f"/thumb/{rid}.jpg",
                "hex": hx, "color_name": str(self.g["colour_word"][i]) or None, "cone": cone_text(self.g["cone_lo"][i], self.g["cone_hi"][i]),
                "surface": str(self.g["surface"][i]).capitalize() or None, "transparency": str(self.g["transparency"][i]).capitalize() or None,
                "atmospheres": str(self.g["atmospheres"][i]) or None, "similarity_score": round(float(score[i]), 6),
                "score_components": {k: round(float(v[i]), 4) for k, v in S.items()} | ({"delta_e_2000": round(float(de[i]), 2)} if de is not None and np.isfinite(de[i]) else {}),
                "contributions": {k: round(float(w[k] * v[i] / tot), 4) for k, v in S.items()},
                "cluster_id": int(self.g["family"][i]), "ingredients": self.ingredients.get(rid)}

    def options(self) -> dict:
        return {"ready": True, "model_version": self.version, "recipes": self.n, "families": int(self.meta.get("families", {}).get("kmeans_k", 0)),
                "weights": self.cfg["weights"], "eval": self.meta.get("eval", {}).get("test"), "parts": list(PARTS)}
