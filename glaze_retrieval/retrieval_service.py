"""Recipe retrieval extracted from test_similarity/retrieval.ipynb (same ranking)."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import rgb2lab, deltaE_ciede2000
from transformers import AutoTokenizer, AutoModel
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 64
SHARED_DIM = 256

def mean_pooling(model_output, attention_mask):
    tokens = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
    return (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

class TextProjection(nn.Module):
    def __init__(self, in_dim=384, out_dim=SHARED_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)


import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from skimage.color import rgb2lab, deltaE_ciede2000
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent / "retrieval_assets"
catalog = pd.read_csv(ROOT / "glaze_retrieval_catalog.csv")
vectors = np.load(ROOT / "glaze_retrieval_vectors.npz", allow_pickle=False)
assert len(catalog) == len(vectors["vectors"])
assert np.array_equal(catalog.recipe_id.astype(str).to_numpy(), vectors["recipe_ids"].astype(str))
chem_vectors = vectors["vectors"]

def parse_hex(h):
    if not isinstance(h, str):
        return None
    h = h.strip().lstrip("#")
    if len(h) == 3 and re.fullmatch(r"[0-9a-fA-F]{3}", h):
        h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        return None
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

rgb = [parse_hex(h) for h in catalog["hex_color"]]
valid_hex = np.array([v is not None for v in rgb])
lab = np.full((len(catalog), 3), np.nan)
if valid_hex.any():
    lab[valid_hex] = rgb2lab(np.stack([v for v in rgb if v is not None]).reshape(-1, 1, 3)).reshape(-1, 3)

def cone_overlap(query, actual):
    def parse(s):
        s = str(s).replace("½", ".5").replace("–", "-").replace("—", "-")
        s = re.sub(r"(\d+)\s+1/2", lambda m: str(float(m.group(1)) + 0.5), s)
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
        return (min(nums), max(nums)) if nums else None
    a, b = parse(query), parse(actual)
    return np.nan if a is None or b is None else float(max(a[0], b[0]) <= min(a[1], b[1]))

checkpoint = torch.load(ROOT / "shared_space_heads.pt", map_location="cpu", weights_only=True)  # Tensor-only checkpoint; restricted deserialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint.get("model_name", MODEL_NAME))
encoder = AutoModel.from_pretrained(checkpoint.get("model_name", MODEL_NAME)).to(device).eval()
text_head = TextProjection(out_dim=checkpoint["shared_dim"]).to(device).eval()
text_head.load_state_dict(checkpoint["text_head"])

def retrieve_glazes(color, surface, cone, query_hex, top_k=10,
                    diversify=False, max_per_cluster=2, allowed_splits=None,
                    latent_weight=0.40, color_weight=0.40,
                    surface_weight=0.10, cone_weight=0.10):
    """Score ALL eligible recipes, then optionally diversify with K-Means.

    Default weights: 40% shared latent similarity, 40% CIEDE2000 color,
    10% surface and 10% cone. Missing metadata contributes zero, and is
    flagged by metadata_coverage. K-Means only affects selection, not score.
    """
    weights = np.array([latent_weight, color_weight, surface_weight, cone_weight], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Ranking weights must be nonnegative and sum to 1")
    if top_k < 1 or max_per_cluster < 1:
        raise ValueError("top_k and max_per_cluster must be >= 1")
    rgb_q = parse_hex(query_hex)
    if rgb_q is None:
        raise ValueError("query_hex must be a valid #RRGGBB or #RGB color")
    lab_q = rgb2lab(rgb_q.reshape(1, 1, 3)).reshape(3)
    query = f"color: {color}, Cone {cone}, {surface}, unknown, unknown"
    batch = tokenizer([query], padding=True, truncation=True,
                      max_length=checkpoint.get("max_len", MAX_LEN), return_tensors="pt").to(device)
    with torch.no_grad():
        q = text_head(mean_pooling(encoder(**batch), batch["attention_mask"])).cpu().numpy()[0]
    similarity = chem_vectors @ q
    eligible = (np.arange(len(catalog)) if allowed_splits is None else
                np.flatnonzero(catalog["split"].isin(allowed_splits).to_numpy()))
    if len(eligible) == 0:
        raise ValueError("No eligible recipes")
    # No latent top-200 bottleneck: every eligible recipe gets a color score.
    ranked = catalog.iloc[eligible].copy()
    ranked["latent_similarity"] = similarity[eligible]
    delta = np.full(len(eligible), np.nan)
    ok = valid_hex[eligible]
    if ok.any():
        delta[ok] = deltaE_ciede2000(
            np.broadcast_to(lab_q, (ok.sum(), 3)).reshape(-1, 1, 3),
            lab[eligible[ok]].reshape(-1, 1, 3)
        ).reshape(-1)
    ranked["delta_e_2000"] = delta
    ranked["color_score"] = np.exp(-delta / 20.0)
    actual_surface = ranked["Surface"].fillna("").astype(str).str.strip().str.casefold()
    ranked["surface_match"] = np.where(
        actual_surface.isin(["", "unknown", "nan", "none"]), np.nan,
        (actual_surface == str(surface).strip().casefold()).astype(float)
    )
    ranked["cone_match"] = ranked["Cone"].map(lambda v: cone_overlap(cone, v))
    fields = ["color_score", "surface_match", "cone_match"]
    ranked["metadata_coverage"] = ranked[fields].notna().mean(axis=1)
    ranked["latent_score"] = (ranked["latent_similarity"].clip(-1, 1) + 1) / 2
    ranked["final_score"] = (
        latent_weight * ranked["latent_score"] +
        color_weight * ranked["color_score"].fillna(0) +
        surface_weight * ranked["surface_match"].fillna(0) +
        cone_weight * ranked["cone_match"].fillna(0)
    )
    ranked = ranked.sort_values(["final_score", "latent_similarity"],
                                ascending=False, kind="stable")
    if diversify:
        selected, counts = [], {}
        for idx, row in ranked.iterrows():
            cluster = int(row["kmeans_cluster"])
            if counts.get(cluster, 0) < max_per_cluster:
                selected.append(idx)
                counts[cluster] = counts.get(cluster, 0) + 1
            if len(selected) == top_k:
                break
        if len(selected) < top_k:
            selected.extend(idx for idx in ranked.index if idx not in selected)
        ranked = ranked.loc[selected]
    return ranked.head(top_k).reset_index(drop=True)


