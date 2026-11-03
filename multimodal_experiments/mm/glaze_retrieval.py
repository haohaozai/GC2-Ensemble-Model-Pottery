"""Glaze recipe search with Claire's model (card G of the glaze search page, POST /api/glaze/search).

Port of the retrieval cell of glaze_kmeans_color_weighted_full_catalog.ipynb (Claire, test_similarity.zip): same artifacts,
same preprocessing, same scoring - the notebook's results are reproduced (tests/test_glaze_retrieval.py).
  query text "color: {name}, Cone {cone}, {surface}, unknown, unknown" -> frozen MiniLM (mean pooled) -> Claire's text head
  -> cosine with the precomputed chemistry vectors of the catalog recipes (oxides -> PCA-16 -> chemistry head, same 256-d space)
  final = 0.40 latent (cos + 1) / 2 + 0.40 colour exp(-CIEDE2000 / 20) + 0.10 surface (same word) + 0.10 cone (ranges overlap)
Both lists come from ONE scoring pass: "Best matches" = the top of the ranking; "Explore formulations" = the same ranking
walked in order with at most max_per_cluster recipes per K-Means chemistry cluster (K-Means picks, it never changes a score).

One deliberate difference, configs/glaze_retrieval.json "cone_scoring":
  "orton" (default)  04 is LOW fire (-4, Orton: 022 ... 01, 1 ... 14) and the original 7k cone text is used - the catalog
                     lost the leading zeros (56 of its 60 "Cone 4" recipes are cone 04)
  "notebook"         Claire's parser and catalog cones exactly (04 -> 4); the parity test uses it
The latent part was trained on the zero-less text, so only a retrained model fixes that part.
Weights, max_per_cluster, top_k and the model folder are in configs/glaze_retrieval.json; a new model from Claire is
installed with INSTALL_GLAZE_MODEL.bat (scripts/install_glaze_model.py) - the page does not change.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

from . import glaze_reference as GR
from .common import CODE

CONFIG = CODE / "configs" / "glaze_retrieval.json"
FILES = {"catalog": "glaze_retrieval_catalog.csv", "vectors": "glaze_retrieval_vectors.npz", "heads": "shared_space_heads.pt"}
CATALOG_COLS = ["recipe_id", "image_path", "color_name", "hex_color", "Cone", "Surface", "kmeans_cluster"]
WEIGHT_KEYS = ("latent", "color", "surface", "cone")
DEFAULTS = {"model_dir": "multimodal_experiments/models/glaze_retrieval", "weights": {"latent": 0.40, "color": 0.40, "surface": 0.10, "cone": 0.10},
            "max_per_cluster": 2, "top_k": 10, "cone_scoring": "orton", "allowed_splits": None, "device": "cpu"}
GLAZY_CDN = "https://cdn.glazy.org/"
GLAZY_RECIPE = "https://glazy.org/recipes/{}"
MAX_TOP_K = 50


# ---------------------------------------------------------------- Claire's notebook (same code, same results)
def parse_hex(h):
    if not isinstance(h, str):
        return None
    h = h.strip().lstrip("#")
    if len(h) == 3 and re.fullmatch(r"[0-9a-fA-F]{3}", h):
        h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        return None
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0


def cone_range_notebook(s):
    """the notebook's cone_overlap parser: '5 1/2 - 6' -> (5.5, 6.0), but also '04' -> (4.0, 4.0)."""
    s = str(s).replace("½", ".5").replace("–", "-").replace("—", "-")
    s = re.sub(r"(\d+)\s+1/2", lambda m: str(float(m.group(1)) + 0.5), s)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    return (min(nums), max(nums)) if nums else None


def cone_range_orton(s):
    """Orton-aware: a leading zero is a LOW-fire cone, so it is negative - '04' -> (-4, -4), '05 1/2 - 03' -> (-5.5, -3),
    '04 - 3' -> (-4, 3) (contains cone 1); '5 1/2' and '5.5' -> 5.5. Same overlap rule as the notebook afterwards."""
    s = str(s).replace("½", " 1/2").replace("–", "-").replace("—", "-")
    vals = []
    for zero, num, half in re.findall(r"(?<![\d.])(0?)(\d+(?:\.\d+)?)(\s*1/2)?", s):
        v = float(num) + (0.5 if half else 0.0)
        vals.append(-v if zero else v)
    return (min(vals), max(vals)) if vals else None


def cone_overlap(query, actual, parse=cone_range_notebook):
    a, b = parse(query), parse(actual)
    return np.nan if a is None or b is None else float(max(a[0], b[0]) <= min(a[1], b[1]))


def mean_pooling(model_output, attention_mask):
    tokens = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
    return (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class TextProjection(nn.Module):
    """Linear map from 384-d MiniLM embeddings into the shared space, L2-normalised."""
    def __init__(self, in_dim=384, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)


def check_weights(w: dict) -> dict:
    try:
        w = {k: float(w[k]) for k in WEIGHT_KEYS}
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"weights need {', '.join(WEIGHT_KEYS)}") from None
    a = np.array(list(w.values()))
    if not np.isfinite(a).all() or (a < 0).any() or not np.isclose(a.sum(), 1.0):
        raise ValueError("Ranking weights must be nonnegative and sum to 1")
    return w


def diversify(ranked: pd.DataFrame, top_k: int, max_per_cluster: int) -> pd.DataFrame:
    """the notebook's K-Means diversification: walk the ranking, keep at most max_per_cluster recipes per cluster until
    top_k are chosen; if the clusters run out, the rest of the ranking fills up. Scores are not changed."""
    if top_k < 1 or max_per_cluster < 1:
        raise ValueError("top_k and max_per_cluster must be >= 1")
    selected, counts = [], {}
    for idx, cluster in zip(ranked.index, ranked["kmeans_cluster"]):
        cluster = int(cluster)
        if counts.get(cluster, 0) < max_per_cluster:
            selected.append(idx)
            counts[cluster] = counts.get(cluster, 0) + 1
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        chosen = set(selected)
        selected.extend(idx for idx in ranked.index if idx not in chosen)
    return ranked.loc[selected].head(top_k)


# ---------------------------------------------------------------- artifacts
def read_config(path: Path = CONFIG) -> dict:
    cfg = dict(DEFAULTS)
    if path.is_file():
        cfg.update({k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")})
    return cfg


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else CODE / p


def load_heads(path: Path) -> dict:
    """Claire's checkpoint with weights_only=True: only tensors, numpy arrays and plain containers are unpickled."""
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:                                                     # numpy < 2
        from numpy.core.multiarray import _reconstruct
    allow = [(_reconstruct, "numpy.core.multiarray._reconstruct"), (_reconstruct, "numpy._core.multiarray._reconstruct"),
             np.ndarray, np.dtype] + [type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32")]
    with torch.serialization.safe_globals(allow):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_artifacts(folder: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """catalog + vectors + checkpoint with the notebook's consistency checks (raises with a readable message)."""
    folder = Path(folder)
    miss = [f for f in FILES.values() if not (folder / f).is_file()]
    if miss:
        raise FileNotFoundError(f"glaze model files missing in {folder}: {', '.join(miss)} - run INSTALL_GLAZE_MODEL.bat")
    catalog = pd.read_csv(folder / FILES["catalog"])
    lack = [c for c in CATALOG_COLS if c not in catalog.columns]
    if lack:
        raise ValueError(f"{FILES['catalog']} lacks the columns {lack}")
    with np.load(folder / FILES["vectors"], allow_pickle=False) as v:
        vectors, ids = v["vectors"], v["recipe_ids"]
    if len(catalog) != len(vectors) or not np.array_equal(catalog["recipe_id"].astype(str).to_numpy(), ids.astype(str)):
        raise ValueError("catalog and vectors are not the same recipes in the same order")
    if not catalog["recipe_id"].is_unique:
        raise ValueError("recipe_id must be unique")
    ckpt = load_heads(folder / FILES["heads"])
    if "text_head" not in ckpt or int(ckpt.get("shared_dim", -1)) != vectors.shape[1]:
        raise ValueError(f"{FILES['heads']} has no text head of the vectors' size {vectors.shape[1]}")
    return catalog, vectors, ckpt


def model_version(folder: Path) -> str:
    h = hashlib.sha256()
    for f in FILES.values():
        h.update(hashlib.sha256((Path(folder) / f).read_bytes()).digest())
    return h.hexdigest()[:12]


def rgb_hex(rgb) -> str:
    return "#" + "".join(f"{round(v * 255):02X}" for v in rgb)


def lab_hex(lab) -> str:
    rgb = np.clip(lab2rgb(np.asarray(lab, np.float64).reshape(1, 1, 3)).reshape(3), 0, 1)
    return "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgb)


def thumb_url(url: str | None) -> str | None:
    """Glazy stores l_ (large: the catalog's link), m_ and s_ sizes of every photo; cards load m_ (about 1/3 of the bytes)
    and fall back to the catalog's l_ link when it fails."""
    if not url:
        return None
    t = re.sub(r"/l_([^/]+)$", r"/m_\1", url)
    return t if t != url else None


def _num(v, nd=6):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def _str(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or not str(v).strip() else str(v).strip()


# ---------------------------------------------------------------- search
class GlazeRetriever:
    """loads the artifacts once (about 5 s, CPU); search() is thread safe."""

    def __init__(self, model_dir=None, cfg: dict | None = None, reference: GR.Reference | None = None):
        from transformers import AutoModel, AutoTokenizer
        t0 = time.time()
        self.cfg = {**DEFAULTS, **(cfg if cfg is not None else read_config())}
        self.weights = check_weights(self.cfg["weights"])
        self.cone_scoring = self.cfg["cone_scoring"]
        if self.cone_scoring not in ("orton", "notebook"):
            raise ValueError('cone_scoring must be "orton" or "notebook"')
        self.dir = resolve(model_dir or self.cfg["model_dir"])
        self.catalog, self.vectors, ckpt = load_artifacts(self.dir)
        self.version = model_version(self.dir)
        self.ref = reference if reference is not None else GR.Reference()
        cat = self.catalog
        # recipe colours (notebook): recipes without a valid HEX (the 'clear' glazes) get no colour score
        rgb = [parse_hex(h) for h in cat["hex_color"]]
        self.valid_hex = np.array([v is not None for v in rgb])
        self.lab = np.full((len(cat), 3), np.nan)
        if self.valid_hex.any():
            self.lab[self.valid_hex] = rgb2lab(np.stack([v for v in rgb if v is not None]).reshape(-1, 1, 3)).reshape(-1, 3)
        # recipe cone ranges, parsed once (the notebook parses them inside every search - same values)
        if self.cone_scoring == "orton":
            self.cone_text = [self.ref.cones.get(int(i), c) for i, c in zip(cat["recipe_id"], cat["Cone"])]
            self._parse = cone_range_orton
        else:
            self.cone_text, self._parse = list(cat["Cone"]), cone_range_notebook
        rng = [self._parse(c) for c in self.cone_text]
        self.cone_lo = np.array([r[0] if r else np.nan for r in rng])
        self.cone_hi = np.array([r[1] if r else np.nan for r in rng])
        # colour words the text head was trained on (the catalog's names = the segmentation pipeline's reference shades)
        from glazeprep.appearance import _REFERENCES
        swatch = {n: h.upper() for _, n, h in _REFERENCES}
        self.names = []
        for n, c in cat["color_name"].dropna().value_counts().items():
            h = swatch.get(n)
            if h is None:                                   # not a reference shade: median colour of its recipes, if any
                labs = self.lab[(cat["color_name"] == n).to_numpy() & self.valid_hex]
                if not len(labs):
                    continue                                # e.g. 'clear': no HEX, cannot be searched by colour
                h = lab_hex(np.median(labs, 0))
            self.names.append({"name": str(n), "hex": h, "recipes": int(c)})
        self.name_lab = rgb2lab(np.stack([parse_hex(x["hex"]) for x in self.names]).reshape(-1, 1, 3)).reshape(-1, 3)
        self.choices = self.ref.choices(cat)
        self.surfaces = [s["value"] for s in self.choices["surfaces"]]
        # MiniLM + Claire's text head on the CPU (a query is one short sentence; the GPU stays with the segmentation models)
        self.device = torch.device(self.cfg.get("device") or "cpu")
        name = ckpt.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.encoder = AutoModel.from_pretrained(name).to(self.device).eval()
        self.text_head = TextProjection(out_dim=int(ckpt["shared_dim"])).to(self.device).eval()
        self.text_head.load_state_dict(ckpt["text_head"])
        self.max_len = int(ckpt.get("max_len", 64))
        self.lock = threading.Lock()
        self.load_s = round(time.time() - t0, 2)

    @classmethod
    def from_config(cls) -> "GlazeRetriever":
        return cls()

    def describe(self) -> str:
        return (f"{len(self.catalog):,} recipes, model {self.version}, cone scoring {self.cone_scoring}, "
                f"{len(self.ref.names):,} names from the 7k CSV, {self.load_s} s")

    def embed(self, text: str) -> np.ndarray:
        batch = self.tokenizer([text], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        with self.lock, torch.no_grad():
            return self.text_head(mean_pooling(self.encoder(**batch), batch["attention_mask"])).cpu().numpy()[0]

    def rank(self, color, surface, cone, query_hex, weights: dict | None = None, allowed_splits=None) -> pd.DataFrame:
        """the notebook's retrieve_glazes up to its sort: scores EVERY eligible recipe (no latent shortlist)."""
        w = check_weights(weights or self.weights)
        latent_weight, color_weight, surface_weight, cone_weight = (w[k] for k in WEIGHT_KEYS)
        rgb_q = parse_hex(query_hex)
        if rgb_q is None:
            raise ValueError("query_hex must be a valid #RRGGBB or #RGB color")
        lab_q = rgb2lab(rgb_q.reshape(1, 1, 3)).reshape(3)
        q = self.embed(f"color: {color}, Cone {cone}, {surface}, unknown, unknown")
        similarity = self.vectors @ q
        catalog = self.catalog
        eligible = (np.arange(len(catalog)) if allowed_splits is None else
                    np.flatnonzero(catalog["split"].isin(allowed_splits).to_numpy()))
        if len(eligible) == 0:
            raise ValueError("No eligible recipes")
        ranked = catalog.iloc[eligible].copy()
        ranked["latent_similarity"] = similarity[eligible]
        delta = np.full(len(eligible), np.nan)
        ok = self.valid_hex[eligible]
        if ok.any():
            delta[ok] = deltaE_ciede2000(
                np.broadcast_to(lab_q, (ok.sum(), 3)).reshape(-1, 1, 3),
                self.lab[eligible[ok]].reshape(-1, 1, 3)
            ).reshape(-1)
        ranked["delta_e_2000"] = delta
        ranked["color_score"] = np.exp(-delta / 20.0)
        actual_surface = ranked["Surface"].fillna("").astype(str).str.strip().str.casefold()
        ranked["surface_match"] = np.where(
            actual_surface.isin(["", "unknown", "nan", "none"]), np.nan,
            (actual_surface == str(surface).strip().casefold()).astype(float)
        )
        qr = self._parse(cone)                    # = cone_overlap(cone, v) for every recipe, with the ranges parsed at start-up
        lo, hi = self.cone_lo[eligible], self.cone_hi[eligible]
        cone_match = np.full(len(eligible), np.nan)
        if qr is not None:
            known = ~np.isnan(lo)
            cone_match[known] = (np.maximum(qr[0], lo[known]) <= np.minimum(qr[1], hi[known])).astype(float)
        ranked["cone_match"] = cone_match
        ranked["cone_text"] = [self.cone_text[i] for i in eligible]
        fields = ["color_score", "surface_match", "cone_match"]
        ranked["metadata_coverage"] = ranked[fields].notna().mean(axis=1)
        ranked["latent_score"] = (ranked["latent_similarity"].clip(-1, 1) + 1) / 2
        ranked["final_score"] = (
            latent_weight * ranked["latent_score"] +
            color_weight * ranked["color_score"].fillna(0) +
            surface_weight * ranked["surface_match"].fillna(0) +
            cone_weight * ranked["cone_match"].fillna(0)
        )
        return ranked.sort_values(["final_score", "latent_similarity"], ascending=False, kind="stable")

    def color_name(self, hx: str) -> dict:
        """nearest of the model's colour words (CIEDE2000 to their reference swatch)."""
        rgb = parse_hex(hx)
        if rgb is None:
            raise ValueError("hex must look like #3054A5")
        lab = rgb2lab(rgb.reshape(1, 1, 3)).reshape(1, 3)
        d = deltaE_ciede2000(np.broadcast_to(lab, self.name_lab.shape), self.name_lab)
        i = int(np.argmin(d))
        return {"hex": rgb_hex(rgb), "name": self.names[i]["name"],
                "swatch": self.names[i]["hex"], "delta_e00": round(float(d[i]), 2)}

    def query(self, p: dict) -> dict:
        """the normalized query of both input methods (colour / photo): hex, colour word, cone, surface (+ optional settings)."""
        auto = self.color_name(str(p.get("hex") or ""))
        name = re.sub(r"\s+", " ", str(p.get("color_name") or "")).strip()[:40]
        cone = GR.normalize_cone(p.get("cone"))
        surface = GR.normalize_surface(p.get("surface"), self.surfaces)
        given = lambda k: self.cfg[k] if p.get(k) in (None, "") else p[k]  # noqa: E731  (0 is a value, not "missing")
        top_k, mpc = int(given("top_k")), int(given("max_per_cluster"))
        if not 1 <= top_k <= MAX_TOP_K or mpc < 1:
            raise ValueError(f"top_k must be 1 - {MAX_TOP_K}, max_per_cluster >= 1")
        return {"hex": auto["hex"], "color_name": name or auto["name"],
                "color_name_source": "you" if name else f"nearest model colour word (ΔE00 {auto['delta_e00']})",
                "cone": cone, "surface": surface, "top_k": top_k, "max_per_cluster": mpc,
                "weights": check_weights(p["weights"]) if p.get("weights") else self.weights,
                # text for the model: Claire's spelling (5 1/2 -> 5.5); 'unknown' is her word for a missing field
                "cone_text": cone.replace(" 1/2", ".5") if cone else "unknown", "surface_text": surface or "unknown"}

    def search(self, p: dict) -> dict:
        t0 = time.perf_counter()
        q = self.query(p)
        ranked = self.rank(q["color_name"], q["surface_text"], q["cone_text"], q["hex"], q["weights"], self.cfg.get("allowed_splits"))
        t1 = time.perf_counter()
        best, explore = ranked.head(q["top_k"]), diversify(ranked, q["top_k"], q["max_per_cluster"])
        rows = lambda df: [self.row(r, k, q["weights"]) for k, r in enumerate(df.itertuples(index=False), 1)]   # noqa: E731
        out = {"query": {k: q[k] for k in ("hex", "color_name", "color_name_source", "cone", "surface", "top_k")},
               "model_text": f"color: {q['color_name']}, Cone {q['cone_text']}, {q['surface_text']}, unknown, unknown",
               "weighted_results": rows(best), "diverse_results": rows(explore),
               "model_version": self.version, "weights": q["weights"], "max_per_cluster": q["max_per_cluster"],
               "cone_scoring": self.cone_scoring, "searched": len(ranked), "warnings": []}
        if not self.ref.recipes_path:
            out["warnings"].append("7k_matched_all_data.csv not found: no recipe names, cones are Claire's catalog values")
        out["timings"] = {"retrieval_ms": round(1000 * (t1 - t0), 1), "total_ms": round(1000 * (time.perf_counter() - t0), 1)}
        return out

    def row(self, r, rank: int, w: dict) -> dict:
        rid = int(r.recipe_id)
        img = r.image_path if isinstance(r.image_path, str) and r.image_path.startswith(GLAZY_CDN) else None
        rgb = parse_hex(r.hex_color)
        comp = {k: _num(getattr(r, k)) for k in ("latent_similarity", "latent_score", "delta_e_2000", "color_score",
                                                 "surface_match", "cone_match", "metadata_coverage")}
        part = {"latent": comp["latent_score"], "color": comp["color_score"], "surface": comp["surface_match"], "cone": comp["cone_match"]}
        return {"rank": rank, "recipe_id": rid, "recipe_name": self.ref.names.get(rid), "recipe_url": GLAZY_RECIPE.format(rid),
                "image_url": img, "thumb_url": thumb_url(img), "hex": None if rgb is None else rgb_hex(rgb), "color_name": _str(r.color_name),
                "cone": _str(r.cone_text), "cone_catalog": _str(r.Cone), "surface": _str(r.Surface),
                "transparency": _str(getattr(r, "Transparency", None)), "atmospheres": _str(getattr(r, "Atmospheres", None)),
                "similarity_score": _num(r.final_score), "score_components": comp,
                "contributions": {k: round(w[k] * (part[k] or 0.0), 6) for k in WEIGHT_KEYS},
                "cluster_id": int(r.kmeans_cluster), "split": _str(getattr(r, "split", None)),
                "ingredients": self.ref.ingredients.get(rid)}

    def options(self) -> dict:
        return {"ready": True, "model_version": self.version, "recipes": len(self.catalog),
                "clusters": int(self.catalog["kmeans_cluster"].nunique()), "weights": self.weights,
                "max_per_cluster": int(self.cfg["max_per_cluster"]), "top_k": int(self.cfg["top_k"]), "cone_scoring": self.cone_scoring,
                "cones": self.choices["cones"], "surfaces": self.choices["surfaces"], "dictionary_source": self.choices["source"],
                "color_names": self.names, "reference": self.ref.info(), "load_s": self.load_s}
