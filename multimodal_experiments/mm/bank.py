"""Feature bank: all recipes of a run, aligned by recipe ID, with per-modality matrices + availability masks.
Every learned transform (scaler, PCA, TF-IDF, autoencoders, ingredient SVD, BiLSTM vocabulary) is fitted on
the rows passed as `fit_idx` only (training split or training fold), never on validation / test rows."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import data as D
from . import formula as F
from . import image_feats as IF
from . import text_feats as TF
from .common import log

MODS = ("text", "image", "formula")


def _deltaE_lab(hexes):
    from skimage.color import rgb2lab
    out = np.full((len(hexes), 3), np.nan, np.float32)
    for i, h in enumerate(hexes):
        if h:
            rgb = np.array([[[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]]], np.float32) / 255.0
            out[i] = rgb2lab(rgb)[0, 0]
    return out


class Bank:
    def __init__(self, recipes: list[dict], objects: list[dict], obj_source=None):
        self.R = recipes
        self.ids = [r["recipe_id"] for r in recipes]
        self.pos = {r: i for i, r in enumerate(self.ids)}
        self.n = len(recipes)
        self.split = np.array([r.get("split", "ext") for r in recipes])
        gid = {g: k for k, g in enumerate(sorted({r.get("group_id", r["recipe_id"]) for r in recipes}))}
        self.group = np.array([gid[r.get("group_id", r["recipe_id"])] for r in recipes])
        self.objects = objects
        self.obj_source = obj_source or IF.internal_objects
        self._img, self._fix, self._fit = {}, {}, {}
        # identical-content groups (unreliable negatives / ties)
        tg = {}
        self.text_gid = np.array([tg.setdefault(r["desc_tags"], len(tg)) if r["desc_tags"] else -1 - i
                                  for i, r in enumerate(recipes)])
        fg = {}
        self.form_gid = np.array([fg.setdefault(tuple(sorted((k, round(v, 1)) for k, v in json.loads(r["oxides_json"]).items())), len(fg))
                                  if r["has_formula"] in ("1", 1) else -1 - i for i, r in enumerate(recipes)])
        # attributes for relevance labels
        self.cone_lo = np.array([int(r["cone_lo_i"]) for r in recipes])
        self.cone_hi = np.array([int(r["cone_hi_i"]) for r in recipes])
        sg = {"glossy": 0, "satin": 1, "matte": 2}
        tg2 = {"clear": 0, "opaque": 1}
        self.surf = np.array([sg.get(D.SURFACE_GROUP.get(r["surface"], ""), -1) for r in recipes])
        self.transp = np.array([tg2.get(D.TRANSP_GROUP.get(r["transparency"], ""), -1) for r in recipes])
        pal = [json.loads(r["palette_json"] or "[]") for r in recipes]
        self.lab = _deltaE_lab([p[0][0] if p else "" for p in pal])
        self.palette = pal

    # ------------------------------------------------------------ helpers
    def idx(self, part) -> np.ndarray:
        return np.nonzero(self.split == part)[0] if isinstance(part, str) else np.asarray(part)

    @staticmethod
    def _key(name, fit_idx):
        return name + ":" + hashlib.md5(np.asarray(fit_idx).tobytes()).hexdigest()[:12]

    @staticmethod
    def _standardise(X, M, fit_idx):
        fi = np.asarray(fit_idx)
        fi = fi[M[fi]]
        mu, sd = X[fi].mean(0), X[fi].std(0) + 1e-6
        Z = np.zeros_like(X)
        Z[M] = (X[M] - mu) / sd
        return np.clip(Z, -8, 8).astype(np.float32)

    # ------------------------------------------------------------ image
    def image_raw(self, enc):
        if enc not in self._img:
            objs = self.obj_source(self.objects)
            vecs = IF.extract(enc, objs) if not hasattr(self, "obj_vecs") else self.obj_vecs(enc, objs)
            X, M = IF.pool_recipes(vecs, self.objects, self.ids, normalise=enc != "hand")
            self._img[enc] = (X, M)
        return self._img[enc]

    # ------------------------------------------------------------ text
    def text_raw(self, enc):
        if enc not in self._fix:
            texts = TF.texts_for(enc, self.R)
            M = np.array([bool(t) for t in texts]) if enc in ("minilm_full", "tfidf", "bilstm") \
                else np.array([r["has_text"] in ("1", 1) for r in self.R])
            self._fix[enc] = (texts, M)
        return self._fix[enc]

    # ------------------------------------------------------------ formula
    def chem(self, fit_idx):
        k = self._key("chemcols", fit_idx)
        if k not in self._fit:
            fit_ids = {self.ids[i] for i in fit_idx}
            cols = F.chem_columns(self.R, fit_ids)
            U = np.zeros((self.n, len(cols) + 5), np.float32)
            W = np.zeros((self.n, len(cols)), np.float32)
            M = np.zeros(self.n, bool)
            for i, r in enumerate(self.R):
                v = F.chem_vector(r["oxides_json"], cols) if r["has_formula"] in ("1", 1) else None
                if v is not None:
                    U[i], W[i], M[i] = v[0], v[1], True
            self._fit[k] = (U, W, M, cols)
        return self._fit[k]

    # ------------------------------------------------------------ main accessor
    def get(self, mod: str, enc: str, fit_idx, seed: int = 0):
        """-> (X float32 [n, d] or int64 tokens, mask [n] bool, info dict)."""
        k = self._key(f"{mod}:{enc}:{seed}", fit_idx)
        if k in self._fit:
            return self._fit[k]
        fit_idx = np.asarray(fit_idx)
        info = {}
        if mod == "image":
            X, M = self.image_raw(enc)
            out = (self._standardise(X, M, fit_idx), M, info)
        elif mod == "text":
            texts, M = self.text_raw(enc)
            if enc in ("minilm_tags", "minilm_full", "clip_text"):
                X = TF.encode_fixed(enc, texts)
                out = (self._standardise(X, M, fit_idx), M, info)
            elif enc == "tfidf":
                fn, info = TF.fit_tfidf([texts[i] for i in fit_idx if M[i]], seed=seed)
                X = fn(texts)
                X[~M] = 0
                out = (self._standardise(X, M, fit_idx), M, info)
            elif enc == "bilstm":
                vocab = TF.build_vocab([texts[i] for i in fit_idx if M[i]])
                out = (TF.to_tokens(texts, vocab), M, {"vocab": len(vocab) + 2})
            else:
                raise ValueError(enc)
        elif mod == "formula":
            U, W, M, cols = self.chem(fit_idx)
            Us = self._standardise(U, M, fit_idx)
            info = {"oxides": len(cols)}
            if enc == "umf":
                out = (Us, M, info)
            elif enc == "wt":
                out = (self._standardise(W, M, fit_idx), M, info)
            elif enc == "pca":
                from sklearn.decomposition import PCA
                fi = fit_idx[M[fit_idx]]
                p = PCA(n_components=16, random_state=seed).fit(Us[fi])
                X = np.zeros((self.n, 16), np.float32)
                X[M] = p.transform(Us[M])
                rec = ((p.inverse_transform(p.transform(Us[fi])) - Us[fi]) ** 2).mean()
                info.update({"explained_variance": round(float(p.explained_variance_ratio_.sum()), 4),
                             "train_recon_mse": round(float(rec), 4)})
                out = (self._standardise(X, M, fit_idx), M, info)
            elif enc in ("ae", "dae", "vae"):
                fi = fit_idx[M[fit_idx]]
                rng = np.random.default_rng(seed)
                perm = rng.permutation(fi)
                cut = max(1, int(0.1 * len(perm)))
                enc_fn, info2 = F.fit_autoencoder(Us[perm[cut:]], Us[perm[:cut]], enc, seed=seed)
                X = np.zeros((self.n, 16), np.float32)
                X[M] = enc_fn(Us[M])
                info.update(info2)
                out = (self._standardise(X, M, fit_idx), M, info)
            elif enc in ("ing", "umf_ing"):
                Xi, Mi = self.ingredients(fit_idx, seed)
                if enc == "ing":
                    out = (Xi, Mi, {"ingredient_svd": 64})
                else:
                    Mb = M & Mi
                    X = np.zeros((self.n, Us.shape[1] + Xi.shape[1]), np.float32)
                    X[Mb] = np.concatenate([Us[Mb], Xi[Mb]], 1)
                    out = (X, Mb, info)
            else:
                raise ValueError(enc)
        else:
            raise ValueError(mod)
        self._fit[k] = out
        return out

    def ingredients(self, fit_idx, seed=0):
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD
        bags = [F.ingredient_bag(r["ingredients_json"]) for r in self.R]
        fit_set = set(np.asarray(fit_idx).tolist())
        from collections import Counter
        df = Counter(k for i, b in enumerate(bags) if i in fit_set for k in b)
        vocab = {k: j for j, (k, c) in enumerate(df.most_common()) if c >= 3}
        rows, cols, vals = [], [], []
        for i, b in enumerate(bags):
            for k, v in b.items():
                if k in vocab:
                    rows.append(i), cols.append(vocab[k]), vals.append(np.log1p(v))
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, len(vocab)))
        M = np.asarray(A.getnnz(axis=1) > 0)
        fi = np.array([i for i in fit_idx if M[i]])
        svd = TruncatedSVD(n_components=min(64, len(vocab) - 1), random_state=seed).fit(A[fi])
        X = svd.transform(A).astype(np.float32)
        return self._standardise(X, M, fit_idx), M


def load_bank(work, include_external: bool = False) -> Bank:
    recs, objs = D.load(work), D.load_objects(work)
    ext = work / "external"
    if include_external and (ext / "recipes_ext.csv").exists():
        from .common import read_csv
        recs = recs + read_csv(ext / "recipes_ext.csv")
        objs = objs + read_csv(ext / "objects_ext.csv")
    b = Bank(recs, objs)
    log().info(f"[bank] {b.n} recipes | split {dict(zip(*np.unique(b.split, return_counts=True)))}")
    return b
