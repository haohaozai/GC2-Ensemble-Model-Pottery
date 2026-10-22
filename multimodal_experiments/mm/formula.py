"""Chemical formula branch: oxide analysis (weight %) -> UMF, ingredient bag, and fitted encoders.

UMF (unity molecular formula) is COMPUTED from the oxide weight % with standard molar masses: moles of every
oxide divided by the moles of the fluxes (R2O + RO) so that the fluxes sum to 1. Nothing is invented: recipes
without at least 3 non-zero oxides have no formula vector. Fitted encoders (scaler, PCA, autoencoders, ingredient
SVD) are fitted on TRAINING recipes only.
"""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np

ATOMIC = {"O": 15.999, "Si": 28.085, "Al": 26.982, "Ca": 40.078, "K": 39.098, "Na": 22.990, "Fe": 55.845,
          "Mg": 24.305, "Ti": 47.867, "B": 10.81, "P": 30.974, "Zn": 65.38, "Cu": 63.546, "Co": 58.933,
          "Sr": 87.62, "Li": 6.94, "Zr": 91.224, "Sn": 118.71, "Ba": 137.33, "Cr": 51.996, "Mn": 54.938,
          "Ni": 58.693, "Pb": 207.2, "F": 18.998, "V": 50.942, "Ce": 140.12, "Mo": 95.95, "Bi": 208.98,
          "Cd": 112.41, "As": 74.922, "Nd": 144.24, "Pr": 140.91, "Er": 167.26, "Sb": 121.76, "Ho": 164.93,
          "Hf": 178.49, "Ag": 107.87, "U": 238.03, "W": 183.84, "La": 138.91, "Be": 9.0122, "Nb": 92.906,
          "Yb": 173.05, "Y": 88.906, "Se": 78.971, "Sm": 150.36, "Eu": 151.96, "Dy": 162.50, "Tb": 158.93,
          "Tm": 168.93, "Lu": 174.97}
R2O = {"Li2O", "Na2O", "K2O", "Cu2O", "Ag2O"}
RO = {"BeO", "MgO", "CaO", "SrO", "BaO", "ZnO", "PbO", "CdO", "FeO", "MnO", "NiO", "CuO", "CoO"}
FLUX = R2O | RO
SYN = {"kaolien": "kaolin", "caolin": "kaolin", "kaolin epk": "ep kaolin", "epk": "ep kaolin",
       "e p k": "ep kaolin", "flint": "silica", "quartz": "silica", "silica 325": "silica", "silica 200": "silica",
       "calcium carbonate": "whiting", "nepheline syenite 270": "nepheline syenite", "neph sy": "nepheline syenite",
       "red iron oxide": "red iron oxide", "rio": "red iron oxide"}


def molar_mass(formula: str) -> float | None:
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not parts or "".join(e + n for e, n in parts) != formula:
        return None
    try:
        return sum(ATOMIC[e] * (int(n) if n else 1) for e, n in parts)
    except KeyError:
        return None


def umf(oxides_wt: dict[str, float]) -> dict[str, float] | None:
    mol = {}
    for k, v in oxides_wt.items():
        mm = molar_mass(k)
        if mm and v > 0:
            mol[k] = v / mm
    flux = sum(v for k, v in mol.items() if k in FLUX)
    if flux <= 0:
        return None
    return {k: v / flux for k, v in mol.items()}


def chem_vector(oxides_json: str, cols: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """[log1p(UMF) for cols] + ratios, and normalised weight % for cols. None if no usable analysis."""
    ox = json.loads(oxides_json) if oxides_json else {}
    if sum(1 for v in ox.values() if v > 0) < 3:
        return None
    u = umf(ox)
    tot = sum(v for v in ox.values() if v > 0) or 1.0
    wt = np.array([100.0 * ox.get(c, 0.0) / tot for c in cols], np.float32)
    if u is None:
        return None
    r2o = sum(v for k, v in u.items() if k in R2O)
    si, al = u.get("SiO2", 0.0), u.get("Al2O3", 0.0)
    extra = [np.log1p(si / al) if al > 0 else np.log1p(si), r2o, u.get("B2O3", 0.0) > 0,
             np.log1p(si), np.log1p(al)]
    vec = np.concatenate([np.log1p(np.array([u.get(c, 0.0) for c in cols], np.float32)), np.array(extra, np.float32)])
    return vec, wt


def norm_ingredient(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9#]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return SYN.get(s, s)


def ingredient_bag(ingredients_json: str) -> dict[str, float]:
    """name -> parts per 100 of the base recipe (additions keep their own amount)."""
    items = json.loads(ingredients_json) if ingredients_json else []
    base = sum(a for _, _, a, add in items if not add) or 1.0
    bag = {}
    for _, name, amount, add in items:
        k = norm_ingredient(name)
        if k:
            bag[k] = bag.get(k, 0.0) + 100.0 * amount / base
    return bag


def chem_columns(recipes: list[dict], train_ids: set[str], min_frac: float = 0.005) -> list[str]:
    """Oxides present in at least min_frac of the training recipes (fixed order)."""
    cnt = {}
    n = 0
    for r in recipes:
        if r["recipe_id"] in train_ids and r["has_formula"] == "1":
            n += 1
            for k in json.loads(r["oxides_json"]):
                cnt[k] = cnt.get(k, 0) + 1
    cols = [k for k, v in cnt.items() if v >= max(3, min_frac * n) and molar_mass(k)]
    order = ["SiO2", "Al2O3", "B2O3", "Li2O", "Na2O", "K2O", "MgO", "CaO", "SrO", "BaO", "ZnO", "PbO",
             "Fe2O3", "FeO", "TiO2", "MnO", "MnO2", "CuO", "Cu2O", "CoO", "Cr2O3", "NiO", "SnO2", "ZrO2", "P2O5",
             "V2O5", "F"]
    return sorted(cols, key=lambda c: (order.index(c) if c in order else 99, c))


# ---------------------------------------------------------------- fitted encoders (train recipes only)
def fit_autoencoder(X: np.ndarray, Xval: np.ndarray, kind: str, latent: int = 16, epochs: int = 300, seed: int = 0):
    """MLP autoencoder (ae), denoising (dae) or variational (vae). Returns (encode_fn, info)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = X.shape[1]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
            self.mu = nn.Linear(64, latent)
            self.lv = nn.Linear(64, latent)
            self.dec = nn.Sequential(nn.Linear(latent, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, d))

        def forward(self, x):
            h = self.enc(x)
            mu = self.mu(h)
            if kind == "vae":
                lv = self.lv(h).clamp(-8, 8)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
                return self.dec(z), mu, lv
            return self.dec(mu), mu, None

    m = AE().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    xt, xv = torch.tensor(X, device=dev), torch.tensor(Xval, device=dev)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 256):
            xb = xt[perm[i:i + 256]]
            inp = xb
            if kind == "dae":
                inp = xb * (torch.rand_like(xb) > 0.15) + 0.1 * torch.randn_like(xb)
            rec, mu, lv = m(inp)
            loss = ((rec - xb) ** 2).mean()
            if kind == "vae":
                loss = loss + 1e-3 * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = ((m(xv)[0] - xv) ** 2).mean().item()
        if v < best - 1e-5:
            best, bad, best_state = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    m.load_state_dict(best_state)
    m.eval()

    def encode(A):
        with torch.no_grad():
            return m(torch.tensor(A, device=dev))[1].cpu().numpy().astype(np.float32)
    return encode, {"kind": kind, "latent": latent, "val_mse": round(best, 5), "epochs": ep + 1}
