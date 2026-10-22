"""Text branch features. Reuses the existing text branch (vendor/gc2_text_branch) for cleaning, the
deterministic tag description and the pinned frozen MiniLM encoder.

  minilm_tags  MiniLM-L6-v2 (text_branch.TextEncoder) on the tag description          (existing design)
  minilm_full  MiniLM on tag description + cleaned free-text Description
  clip_text    OpenCLIP ViT-B/32 text tower (cached) on the tag description            (zero-shot I<->T baseline)
  tfidf        TF-IDF (1-2 grams) + TruncatedSVD(256), fitted on training recipes only  (lexical baseline)
  bilstm       token ids for a BiLSTM trained jointly in the contrastive model           (RNN baseline, free text)
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .common import CACHE, log

ENCODERS = ["minilm_tags", "minilm_full", "clip_text", "tfidf", "bilstm"]


def texts_for(enc: str, recipes: list[dict]) -> list[str]:
    out = []
    for r in recipes:
        tags, free = r.get("desc_tags", ""), r.get("desc_free", "")
        if enc in ("minilm_tags", "clip_text"):
            out.append(tags)
        else:
            words = free.split()
            free = " ".join(words[:180])
            out.append((tags + (" Notes: " + free if free else "")).strip())
    return out


def _key(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:24]


def _cached_encode(name: str, texts: list[str], fn) -> np.ndarray:
    p = CACHE / f"text_{name}.npz"
    have = {}
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            have = dict(zip(z["keys"].tolist(), z["X"]))
    todo = sorted({t for t in texts if t and _key(t) not in have})
    if todo:
        log().info(f"[text:{name}] encoding {len(todo)} new texts ({len(have)} cached)")
        V = fn(todo)
        for t, v in zip(todo, V):
            have[_key(t)] = v
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = list(have)
        np.savez_compressed(p.with_name(p.stem + ".tmp.npz"), keys=np.array(keys), X=np.stack([have[k] for k in keys]))
        p.with_name(p.stem + ".tmp.npz").replace(p)
    d = len(next(iter(have.values())))
    return np.stack([have[_key(t)] if t else np.zeros(d, np.float32) for t in texts]).astype(np.float32)


_ENC = {}


def _minilm(texts):
    if "minilm" not in _ENC:
        from text_branch.config import EncoderConfig
        from text_branch.encoder import TextEncoder
        _ENC["minilm"] = TextEncoder(EncoderConfig())
    return _ENC["minilm"].encode_documents(texts)


def _minilm_query(texts):
    _minilm(["warm up"]) if "minilm" not in _ENC else None
    return _ENC["minilm"].encode_queries(texts)


def _clip_text(texts):
    if "clip" not in _ENC:
        from glazeprep.ceramic_clip import CeramicClip
        _ENC["clip"] = CeramicClip()
    out = []
    for i in range(0, len(texts), 256):
        out.append(_ENC["clip"].text_embeddings(texts[i:i + 256]).cpu().numpy())
    return np.concatenate(out)


def encode_fixed(enc: str, texts: list[str]) -> np.ndarray:
    """Frozen pretrained encoders (cached by text)."""
    if enc in ("minilm_tags", "minilm_full"):
        return _cached_encode("minilm", texts, _minilm)
    if enc == "clip_text":
        return _cached_encode("clip_text", texts, _clip_text)
    raise ValueError(enc)


def fit_tfidf(train_texts: list[str], dim: int = 256, seed: int = 0):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    A = vec.fit_transform([t for t in train_texts if t])
    svd = TruncatedSVD(n_components=min(dim, A.shape[1] - 1), random_state=seed).fit(A)

    def transform(texts):
        X = svd.transform(vec.transform([t or "" for t in texts])).astype(np.float32)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(n, 1e-8, None)
    return transform, {"vocab": len(vec.vocabulary_), "svd_dim": svd.n_components,
                       "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4)}


TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(t: str) -> list[str]:
    return TOKEN.findall((t or "").lower())


def build_vocab(train_texts: list[str], min_freq: int = 2, max_size: int = 20000) -> dict[str, int]:
    from collections import Counter
    c = Counter(w for t in train_texts for w in tokenize(t))
    words = [w for w, n in c.most_common(max_size) if n >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}                     # 0 = pad, 1 = unknown


def to_tokens(texts: list[str], vocab: dict[str, int], max_len: int = 96) -> np.ndarray:
    X = np.zeros((len(texts), max_len), np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tokenize(t)][:max_len]
        X[i, :len(ids)] = ids
    return X
