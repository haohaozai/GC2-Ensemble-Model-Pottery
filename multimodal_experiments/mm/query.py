"""Query engine for NEW inputs - a photo from anywhere, typed tags / notes, an oxide analysis - used by app/server.py.

Rebuilds every fitted transform of the final model exactly as phase E did (training rows only, the selected seed),
loads results/final_model.pt and embeds new queries into the same shared space as the database recipes:

  photo    -> Grounding DINO + SAM 2.1 + CLIP segmenter (the claude_v3 pipeline, same parameters) -> objects
              (CLIP p_ceramic >= 0.35, else the first object, as for the external test) -> DINOv2 masked pooling ->
              area-weighted mean -> train standardisation -> image head
  tags     -> vendor text-branch cleaning (unchanged) -> "Subtype: ... Firing cone: ..." + notes -> TF-IDF/SVD -> text head
  formula  -> oxide wt % (or UMF converted to wt %) -> UMF vector -> (+ ingredient SVD) -> formula head

`verify()` checks that the rebuilt transforms reproduce results/db_embeddings.npz. Nothing here trains the
retrieval model; the optional oxide-only formula head (fit_oxide_head) is fitted on training recipes and selected on
validation recipes, never on test or external data.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .common import PHOTOS, Timer, log, read_json, work_dir, write_json   # first: model caches -> D:
from . import data as D
from . import formula as F
from . import image_feats as IF
from . import metrics as MT
from . import model as MD
from . import text_feats as TF
from .bank import _deltaE_lab, load_bank
from .external import CONE_ORDER

MODS = ("text", "image", "formula")
SURFACES = ["Glossy", "Semi-glossy", "Satin", "Satin-matte", "Semi-matte", "Smooth Matte", "Matte", "Stony Matte", "Dry Matte"]
TRANSPARENCIES = ["Transparent", "Translucent", "Semi-opaque", "Opaque"]
ATMOSPHERES = ["Oxidation", "Reduction", "Neutral", "Salt & Soda", "Wood", "Raku", "Luster"]
P_CERAMIC_KEEP = 0.35          # same object rule as the external test (mm/external.to_recipes)
MAX_SIDE = 2048                # very large uploads are reduced before segmentation


def _musd(X, M, fit_idx):
    fi = np.asarray(fit_idx)
    fi = fi[M[fi]]
    return X[fi].mean(0), X[fi].std(0) + 1e-6


def _std(X, mu, sd):
    return np.clip((X - mu) / sd, -8, 8).astype(np.float32)


def _std_masked(X, M, mu, sd):
    Z = np.zeros_like(X, dtype=np.float32)
    Z[M] = _std(X[M], mu, sd)
    return Z


def umf_to_wt(umf: dict) -> dict:
    """UMF (moles) -> weight % (moles x molar mass, normalised to 100)."""
    g = {k: v * F.molar_mass(k) for k, v in umf.items() if v > 0 and F.molar_mass(k)}
    tot = sum(g.values()) or 1.0
    return {k: 100.0 * v / tot for k, v in g.items()}


def parse_oxides(text: str) -> dict:
    """'SiO2 64.2  Al2O3: 13.1, CaO 9' (or a table copied from Glazy) -> {oxide: value}."""
    import re
    out = {}
    for k, v in re.findall(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)", text or ""):
        if F.molar_mass(k) and "O" in k or k == "F":
            out[k] = float(v.replace(",", "."))
    return {k: v for k, v in out.items() if v > 0}


def parse_ingredients(text: str) -> list:
    """lines 'Custer Feldspar 41' / '+ Red Iron Oxide 8' (+ or 'add' = addition) -> [[id, name, amount, is_add]]."""
    import re
    items = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        add = s.startswith("+") or bool(re.search(r"\badd(ition(al)?)?\b", s, re.I))
        s = re.sub(r"^\+\s*", "", s)
        s = re.sub(r"\(?\badd(ition(al)?)?\b\)?", "", s, flags=re.I).strip()
        m = re.match(r"(.+?)[\s:,=]+(\d+(?:[.,]\d+)?)\s*%?$", s)
        if m:
            items.append(["", m.group(1).strip(), float(m.group(2).replace(",", ".")), add])
    return items


# ---------------------------------------------------------------- segmentation of a new photo
class Segmenter:
    def __init__(self, device=None):
        import torch
        from glazeprep.ceramic_clip import CeramicClip
        from glazeprep.segmentation.ceramic_v3 import CeramicSegmenterV3, V3Params
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip = CeramicClip(device=dev)
        self.seg = CeramicSegmenterV3(params=V3Params(), device=dev, clip=self.clip)

    def __call__(self, rgb: np.ndarray) -> dict:
        """-> {"objects": [{index, rgba, mask, bbox, area_px, p_ceramic, primary, flags}], "mode", "union", "flags"}"""
        res = self.seg(rgb)
        objs = []
        for k, inst in enumerate(res.instances, 1):
            m = inst.mask
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop, cm = rgb[y0:y1, x0:x1], m[y0:y1, x0:x1]
            objs.append({"index": k, "rgba": np.dstack([crop, (cm * 255).astype(np.uint8)]), "mask": m,
                         "bbox": [int(x0), int(y0), int(x1), int(y1)], "area_px": int(m.sum()),
                         "p_ceramic": None if inst.p_ceramic is None else float(inst.p_ceramic),
                         "primary": bool(inst.primary), "flags": list(inst.flags)})
        return {"objects": objs, "mode": res.mode, "union": res.union, "flags": list(res.flags)}


def database_duplicate(data: bytes, max_dist: int = 3) -> dict | None:
    """Is this photo (almost) the same as a database photo? Same dHash as the leakage check (mm/data.dhash, computed
    on the file bytes). A database photo finds its own recipe at rank 1 - that says nothing about a new photo."""
    import io
    from .common import CACHE
    hashes = read_json(CACHE / "photo_dhash.json", {}) or {}
    if not hashes:
        return None
    try:
        h = D.dhash(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        return None
    ids = list(hashes)
    H = np.array([int(hashes[i]) for i in ids], dtype=np.uint64)
    d = np.array([bin(int(x)).count("1") for x in np.bitwise_xor(H, np.uint64(h))])
    # the 64-bit dHash alone gave false matches (2026-10-20: a 147 px contact-sheet crop of 20812_1 at distance 2 from the
    # unrelated photo 666131), so every close hash is confirmed on the pixels before a photo is called a database photo
    for k in np.argsort(d)[:5]:
        if d[k] > max_dist:
            break
        conf = same_photo(data, ids[k])
        if conf is not None:
            return {"recipe_id": ids[k], "distance": int(d[k]), **conf}
    return None


def same_photo(data: bytes, rid: str, side: int = 32) -> dict | None:
    """pixel check of a dHash candidate: same aspect ratio (within 6 %), grey 32 x 32 correlation >= 0.95 and mean colour
    difference <= 14 / 255 -> {"pixel_corr", "colour_diff"}; otherwise None (not the same photo)."""
    import io
    from PIL import Image, ImageOps
    from .common import PHOTOS
    p = PHOTOS / f"{rid}.jpg"
    if not p.exists():
        return None
    try:
        with Image.open(io.BytesIO(data)) as a, Image.open(p) as b:
            a, b = ImageOps.exif_transpose(a).convert("RGB"), ImageOps.exif_transpose(b).convert("RGB")
            ra, rb = a.width / a.height, b.width / b.height
            if abs(ra - rb) / rb > 0.06:
                return None
            A = np.asarray(a.resize((side, side), Image.BILINEAR), np.float32)
            Bm = np.asarray(b.resize((side, side), Image.BILINEAR), np.float32)
    except Exception:  # noqa: BLE001
        return None
    ga, gb = A.mean(2).ravel(), Bm.mean(2).ravel()
    corr = float(np.corrcoef(ga, gb)[0, 1]) if ga.std() > 1e-3 and gb.std() > 1e-3 else 0.0
    diff = float(np.abs(A - Bm).mean())
    return {"pixel_corr": round(corr, 3), "colour_diff": round(diff, 1)} if corr >= 0.95 and diff <= 14 else None


def default_selection(objs: list[dict]) -> list[int]:
    keep = [o["index"] for o in objs if (o["p_ceramic"] or 0) >= P_CERAMIC_KEEP]
    return keep or ([objs[0]["index"]] if objs else [])


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, work=None, load_image_model=True):
        import torch
        t = Timer()
        self.work = Path(work or work_dir(False))
        self.res = self.work / "results"
        ck = torch.load(self.res / "final_model.pt", map_location="cpu", weights_only=False)
        if not ck.get("trained"):
            raise RuntimeError("final_model.pt is not marked as trained")
        self.cfg, self.weights = ck["cfg"], ck["weights"]
        seed = self.cfg["seed"]
        b = self.bank = load_bank(self.work)                 # the database recipes (no external rows)
        tr = self.tr = b.idx("train")
        # ---- image
        Xi, Mi = b.image_raw(self.cfg["image"])
        self.img_mu, self.img_sd = _musd(Xi, Mi, tr)
        # ---- text
        texts, Mt = b.text_raw(self.cfg["text"])
        self.tfidf, _ = TF.fit_tfidf([texts[i] for i in tr if Mt[i]], seed=seed)
        Xt = self.tfidf(texts)
        Xt[~Mt] = 0
        self.txt_mu, self.txt_sd = _musd(Xt, Mt, tr)
        # ---- formula: UMF part
        U, _, Mf, self.cols = b.chem(tr)
        self.u_mu, self.u_sd = _musd(U, Mf, tr)
        # ---- formula: ingredient part (same steps as Bank.ingredients)
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD
        bags = [F.ingredient_bag(r["ingredients_json"]) for r in b.R]
        fit_set = set(tr.tolist())
        df = Counter(k for i, bg in enumerate(bags) if i in fit_set for k in bg)
        self.ing_vocab = {k: j for j, (k, c) in enumerate(df.most_common()) if c >= 3}
        A = self._bag_matrix(bags)
        Mg = np.asarray(A.getnnz(axis=1) > 0)
        fi = np.array([i for i in tr if Mg[i]])
        self.ing_svd = TruncatedSVD(n_components=min(64, len(self.ing_vocab) - 1), random_state=seed).fit(A[fi])
        Xg = self.ing_svd.transform(A).astype(np.float32)
        self.ing_mu, self.ing_sd = _musd(Xg, Mg, tr)
        # ---- database features, exactly as Data(bank, cfg, train) built them
        Us = _std_masked(U, Mf, self.u_mu, self.u_sd)
        Gs = _std_masked(Xg, Mg, self.ing_mu, self.ing_sd)
        Mb = Mf & Mg
        Xf = np.zeros((b.n, Us.shape[1] + Gs.shape[1]), np.float32)
        Xf[Mb] = np.concatenate([Us[Mb], Gs[Mb]], 1)
        self.feats = {"image": (_std_masked(Xi, Mi, self.img_mu, self.img_sd), Mi),
                      "text": (_std_masked(Xt, Mt, self.txt_mu, self.txt_sd), Mt),
                      "formula": (Xf, Mb)}
        self.U_std, self.U_mask = Us, Mf
        # ---- model + database embeddings
        self.model = MD.MultiModal(ck["dims"], self.cfg).to(MD.DEV)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.Z = {m: (self._head(m, X, M), M) for m, (X, M) in self.feats.items()}
        self.oxide_head = None
        p = self.work / "app" / "oxide_head.pt"
        if p.exists():
            self._load_oxide_head(p)
        # ---- attributes for the result cards
        self.meta_rows = [{"id": r["recipe_id"], "name": r["name"], "subtype": r["subtype"], "cone_lo": r["cone_lo"],
                           "cone_hi": r["cone_hi"], "surface": r["surface"], "transparency": r["transparency"],
                           "atmospheres": r["atmospheres"], "url": r["glazy_url"], "split": r["split"],
                           "palette": json.loads(r["palette_json"] or "[]")} for r in b.R]
        self.deep = IF.Deep(self.cfg["image"]) if load_image_model else None
        log().info(f"[app] engine ready in {t()} s: {b.n} database recipes, model seed {seed}, weights {self.weights}")

    # ------------------------------------------------------------ helpers
    def _bag_matrix(self, bags):
        from scipy import sparse
        rows, cols, vals = [], [], []
        for i, bg in enumerate(bags):
            for k, v in bg.items():
                if k in self.ing_vocab:
                    rows.append(i), cols.append(self.ing_vocab[k]), vals.append(np.log1p(v))
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(bags), len(self.ing_vocab)))

    def _head(self, m, X, M=None):
        import torch
        out = np.zeros((len(X), self.model.dim), np.float32)
        rows = np.nonzero(M)[0] if M is not None else np.arange(len(X))
        with torch.no_grad():
            for s in range(0, len(rows), 2048):
                r = rows[s:s + 2048]
                out[r] = self.model(m, torch.tensor(X[r], device=MD.DEV)).cpu().numpy()
        return out

    def verify(self) -> dict:
        """rebuilt database embeddings vs the saved results/db_embeddings.npz (same model, same transforms)."""
        z = np.load(self.res / "db_embeddings.npz")
        pos = {i: k for k, i in enumerate(z["ids"].tolist())}
        rows = np.array([pos[i] for i in self.bank.ids])
        out = {}
        for m in MODS:
            Z, M = self.Z[m]
            Zs, Ms = z["Z_" + m][rows], z["M_" + m][rows]
            both = M & Ms
            cos = (Z[both] * Zs[both]).sum(1)
            out[m] = {"mask_equal": bool((M == Ms).all()), "rows": int(both.sum()),
                      "min_cos": round(float(cos.min()), 6), "mean_cos": round(float(cos.mean()), 6)}
        return out

    # ------------------------------------------------------------ query encoders
    def image_vector(self, objs: list[dict]):
        """objs: [{"rgba", "area_px"}] -> raw pooled vector (as Bank.image_raw) or None."""
        if not objs:
            return None
        V = self.deep.encode([o["rgba"] for o in objs])
        w = np.array([o["area_px"] for o in objs], np.float32)
        v = (V * (w / w.sum())[:, None]).sum(0)
        return v / max(1e-8, float(np.linalg.norm(v)))

    def compose_text(self, tags: dict, notes: str = "") -> tuple[str, dict]:
        """UI tags (raw Glazy values) -> the text the model was trained on + cleaned attributes."""
        cone = ""
        lo, hi = (tags.get("cone_lo") or "").strip(), (tags.get("cone_hi") or "").strip()
        if lo and hi and lo != hi:
            cone = f"{lo} - {hi}"
        elif lo or hi:
            cone = lo or hi
        atm = tags.get("atmospheres") or []
        raw = {"recipe_id": "query", "Name": "", "Type": "Recipe", "State": "", "Subtype": tags.get("subtype") or "",
               "Cone": cone, "Surface": tags.get("surface") or "", "Transparency": tags.get("transparency") or "",
               "Atmospheres": "; ".join(atm) if isinstance(atm, list) else str(atm), "Country": "", "Other Names": ""}
        t = D.clean_tags([raw])["query"]
        text = TF.texts_for(self.cfg["text"], [{"desc_tags": t["desc_tags"], "desc_free": (notes or "").strip()}])[0]
        return text, t

    def text_vector(self, text: str):
        if not text:
            return None
        return self.tfidf([text])[0]

    def formula_parts(self, oxides_wt: dict, ingredients: list):
        """-> (UMF-part raw vector or None, ingredient-part std vector or None, computed UMF dict)."""
        cv = F.chem_vector(json.dumps(oxides_wt), self.cols) if oxides_wt else None
        u = F.umf(oxides_wt) if oxides_wt else None
        ing = None
        if ingredients:
            bag = F.ingredient_bag(json.dumps(ingredients))
            A = self._bag_matrix([bag])
            if A.getnnz() > 0:
                ing = _std(self.ing_svd.transform(A).astype(np.float32), self.ing_mu, self.ing_sd)[0]
        return (None if cv is None else cv[0]), ing, u

    def embed_query(self, img=None, text=None, umf_vec=None, ing_vec=None, oxide_mode="auto") -> tuple[dict, dict]:
        """-> ({modality: (z [1, d], mask [1])}, info). oxide_mode for oxides without ingredients: 'mean' fills the
        ingredient part with the training average (0 after standardisation), 'head' uses the oxide-only head."""
        Z, info = {}, {}
        if img is not None:
            x = _std(img[None], self.img_mu, self.img_sd)
            Z["image"] = (self._head("image", x), np.ones(1, bool))
        if text is not None:
            x = _std(text[None], self.txt_mu, self.txt_sd)
            Z["text"] = (self._head("text", x), np.ones(1, bool))
        if umf_vec is not None:
            us = _std(umf_vec[None], self.u_mu, self.u_sd)
            if ing_vec is not None:
                x = np.concatenate([us, ing_vec[None]], 1)
                Z["formula"] = (self._head("formula", x), np.ones(1, bool))
                info["formula_path"] = "oxides + ingredients (final formula head)"
            else:
                mode = oxide_mode
                if mode == "auto":
                    mode = (read_json(self.work / "app" / "oxide_only_choice.json", {}) or {}).get("choice", "mean")
                if mode == "head" and self.oxide_head is not None:
                    Z["formula"] = (self._oxide(us), np.ones(1, bool))
                    info["formula_path"] = "oxides only (oxide-only head, chosen on validation)"
                else:
                    x = np.concatenate([us, np.zeros((1, self.ing_mu.shape[0]), np.float32)], 1)
                    Z["formula"] = (self._head("formula", x), np.ones(1, bool))
                    info["formula_path"] = "oxides only (ingredient part = training average)"
        return Z, info

    # ------------------------------------------------------------ oxide-only head (no ingredient list given)
    def _oxide(self, us):
        import torch
        with torch.no_grad():
            return self.oxide_head(torch.tensor(us, device=MD.DEV)).cpu().numpy()

    def _load_oxide_head(self, p):
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        h = MD.Head(ck["d_in"], self.model.dim, self.cfg["head"], self.cfg["hidden"], self.cfg["dropout"]).to(MD.DEV)
        h.load_state_dict(ck["state_dict"])
        self.oxide_head = h.eval()

    def fit_oxide_head(self, seed=0) -> dict:
        """Formula head on the UMF part only, trained on TRAINING recipes to land on the final model's (frozen)
        text and image embeddings of the same recipe; early stopping and the mean-fill vs head choice on VALIDATION."""
        import torch
        import torch.nn.functional as Fn
        torch.manual_seed(seed)
        b, tr, va = self.bank, self.tr, self.bank.idx("val")
        Us, Mu = self.U_std, self.U_mask
        Zt, Mt = self.Z["text"]
        Zi, Mi = self.Z["image"]
        tgt_rows = [i for i in tr if Mu[i] and Mt[i] and Mi[i]]
        h = MD.Head(Us.shape[1], self.model.dim, self.cfg["head"], self.cfg["hidden"], self.cfg["dropout"]).to(MD.DEV)
        opt = torch.optim.AdamW(h.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        scale = float(self.model.scale())
        X = torch.tensor(Us, device=MD.DEV)
        T, I = torch.tensor(Zt, device=MD.DEV), torch.tensor(Zi, device=MD.DEV)
        fg = torch.tensor(b.form_gid, device=MD.DEV)
        rows = torch.tensor(np.array(tgt_rows), device=MD.DEV)

        def val_mrr(fn):
            V = fn(Us[va])
            return self._val_identity(V, va)["MRR"]

        best, state, bad, hist = -1, None, 0, []
        for ep in range(150):
            h.train()
            perm = rows[torch.randperm(len(rows), device=MD.DEV)]
            for s in range(0, len(perm), self.cfg["batch"]):
                r = perm[s:s + self.cfg["batch"]]
                z = h(X[r])
                same = fg[r][:, None] == fg[r][None, :]
                loss = sum(MD.info_nce(z, tg[r], scale, same) for tg in (T, I)) / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
            h.eval()
            with torch.no_grad():
                v = val_mrr(lambda A: h(torch.tensor(A, device=MD.DEV)).cpu().numpy())
            hist.append(round(v, 4))
            if v > best + 1e-4:
                best, bad, state = v, 0, {k: t.clone() for k, t in h.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        h.load_state_dict(state)
        self.oxide_head = h.eval()
        p = self.work / "app" / "oxide_head.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": h.state_dict(), "d_in": Us.shape[1], "trained": True,
                    "split": "train, early stop on validation", "epochs": len(hist)}, p)
        # validation comparison: F -> T+I (gallery = validation recipes) for the three ways of encoding a formula
        Xf, Mb = self.feats["formula"]
        full = self._val_identity(self._head("formula", Xf[va]), va, rows_ok=Mb[va])
        zero = np.concatenate([Us[va], np.zeros((len(va), self.ing_mu.shape[0]), np.float32)], 1)
        mean_fill = self._val_identity(self._head("formula", zero), va, rows_ok=Mb[va])
        head = self._val_identity(self._oxide(Us[va]), va, rows_ok=Mb[va])
        choice = "head" if head["MRR"] > mean_fill["MRR"] else "mean"
        out = {"validation_F->T+I": {"oxides + ingredients (reference)": full,
                                     "oxides only, ingredient part = training average": mean_fill,
                                     "oxides only, oxide-only head": head},
               "choice": choice, "epochs": len(hist), "history_val_MRR": hist,
               "rule": "oxide-only queries use the variant with the higher validation MRR"}
        write_json(self.work / "app" / "oxide_only_choice.json", out)
        log().info(f"[app] oxide-only formula: {json.dumps({k: v['R@5'] for k, v in out['validation_F->T+I'].items()})} -> {choice}")
        return out

    def _val_identity(self, Vf, va, rows_ok=None):
        """F -> T+I same-recipe retrieval inside the validation recipes (Vf: formula embeddings of va)."""
        Zq = {"formula": (Vf, self.U_mask[va] if rows_ok is None else rows_ok)}
        Zg = {m: (self.Z[m][0][va], self.Z[m][1][va]) for m in ("text", "image")}
        Q, qm = MT.fuse(Zq, ["formula"], self.weights)
        G, gm = MT.fuse(Zg, ["text", "image"], self.weights)
        rows = np.nonzero(qm & gm)[0]
        S = Q[rows] @ G[gm].T
        gpos = {g: k for k, g in enumerate(np.nonzero(gm)[0])}
        tgt = np.array([gpos[r] for r in rows])
        rk = 1 + (S > S[np.arange(len(rows)), tgt][:, None]).sum(1)
        return {"n": int(len(rows)), "R@5": round(float((rk <= 5).mean()), 4), "R@10": round(float((rk <= 10).mean()), 4),
                "MRR": round(float((1 / rk).mean()), 4)}

    # ------------------------------------------------------------ search
    def search(self, Zq: dict, weights: dict, qattr: dict | None, filters: dict | None, true_id: str | None,
               top: int = 10) -> dict:
        b = self.bank
        w = {m: float(weights.get(m, self.weights[m])) for m in MODS}
        qmods = [m for m in MODS if m in Zq]
        if not qmods:
            return {"gallery": int(b.n), "filters_applied": [], "query_modalities": [], "weights_used": {}, "results": [],
                    "auto": None, "message": "Nothing to search with: add a photo, some tags / notes or a formula."}
        Q, qm = MT.fuse(Zq, qmods, w)
        G, gm = MT.fuse(self.Z, MODS, w)
        allow = gm.copy()
        applied = []
        if filters and qattr:
            if filters.get("cone") and qattr["lo"] >= 0:
                allow &= (b.cone_lo >= 0) & (b.cone_lo <= qattr["hi"]) & (qattr["lo"] <= b.cone_hi)
                applied.append("cone range overlaps")
            if filters.get("surface") and qattr["s"] >= 0:
                allow &= b.surf == qattr["s"]
                applied.append("same surface group")
            if filters.get("transparency") and qattr["t"] >= 0:
                allow &= b.transp == qattr["t"]
                applied.append("same transparency group")
            if filters.get("subtype") and qattr.get("subtype"):
                allow &= np.array([r["subtype"].startswith(qattr["subtype"]) for r in b.R])
                applied.append(f"subtype starts with '{qattr['subtype']}'")
        gi = np.nonzero(allow)[0]
        out = {"gallery": int(len(gi)), "filters_applied": applied, "query_modalities": qmods,
               "weights_used": {m: w[m] for m in qmods}}
        if not qm.any() or not len(gi):
            out["results"] = []
            return out
        S = (G[gi] @ Q[0])
        order = np.argsort(-S)
        topi = gi[order[:top]]
        rel = gain = dE = None
        if qattr is not None:
            A = {"lo": np.array([qattr["lo"]]), "hi": np.array([qattr["hi"]]), "s": np.array([qattr["s"]]),
                 "t": np.array([qattr["t"]]), "lab": qattr["lab"][None]}
            rel, gain, dE = MT.relevance(b, np.array([0]), gi, A)
            rel, gain, dE = rel[0], gain[0], dE[0]
        rows = []
        for k, j in enumerate(order[:top]):
            i = gi[j]
            r = dict(self.meta_rows[i])
            r.update({"rank": k + 1, "score": round(float(S[j]), 4),
                      "dE": None if dE is None or np.isnan(dE[j]) else round(float(dE[j]), 1),
                      "relevant": None if rel is None else bool(rel[j]),
                      "gain": None if gain is None else round(float(gain[j]), 2)})
            rows.append(r)
        out["results"] = rows
        comparable = qattr is not None and not np.isnan(qattr["lab"]).any() and \
            (qattr["lo"] >= 0 or qattr["s"] >= 0 or qattr["t"] >= 0)
        if comparable and rel is not None:
            R = rel[order]
            out["auto"] = {f"Hit@{k}": bool(R[:k].any()) for k in (3, 5, 10)}
            out["auto"].update({f"P@{k}": round(float(R[:k].mean()), 3) for k in (3, 5, 10)})
            out["auto"]["relevant_in_gallery"] = int(rel.sum())
        else:
            out["auto"] = None
        if true_id and true_id in b.pos:
            ti = b.pos[true_id]
            if allow[ti]:
                rk = int(1 + (S > float(G[ti] @ Q[0])).sum())
                out["true_recipe"] = {"id": true_id, "rank": rk, **{f"Hit@{k}": rk <= k for k in (3, 5, 10)}}
            else:
                out["true_recipe"] = {"id": true_id, "rank": None, "note": "excluded by the hard filters"}
        elif true_id:
            out["true_recipe"] = {"id": true_id, "rank": None, "note": "not one of the database recipes"}
        return out

    def query_attributes(self, cleaned: dict, palette: list) -> dict:
        """cleaned tags (compose_text) + dominant query colour -> relevance attributes (same rule as the report)."""
        sg = {"glossy": 0, "satin": 1, "matte": 2}
        tg = {"clear": 0, "opaque": 1}
        lab = _deltaE_lab([palette[0][0] if palette else ""])[0]
        return {"lo": int(cleaned["cone_lo_i"]), "hi": int(cleaned["cone_hi_i"]),
                "s": sg.get(D.SURFACE_GROUP.get(cleaned["surface"], ""), -1),
                "t": tg.get(D.TRANSP_GROUP.get(cleaned["transparency"], ""), -1), "lab": lab,
                "subtype": cleaned.get("subtype", "")}

    def options(self) -> dict:
        exp_sub = Counter(r["subtype"] for r in self.bank.R if r["subtype"])
        return {"subtypes": [s for s, _ in exp_sub.most_common()], "cones": CONE_ORDER, "surfaces": SURFACES,
                "transparencies": TRANSPARENCIES, "atmospheres": ATMOSPHERES, "oxides": self.cols,
                "weights": self.weights, "gallery": self.bank.n, "model": {k: self.cfg[k] for k in ("text", "image", "formula", "dim", "seed")}}


def palette_of(objs: list[dict]) -> list:
    """area-weighted ColorThief palette over the selected objects (same as the external recipes)."""
    acc, tot = {}, 0.0
    for o in objs:
        for hx, fr in IF.palette_lab(o["rgba"]):
            acc[hx] = acc.get(hx, 0.0) + o["area_px"] * fr
            tot += o["area_px"] * fr
    tot = tot or 1.0
    return [[h, round(v / tot, 4)] for h, v in sorted(acc.items(), key=lambda kv: -kv[1])[:3]]
