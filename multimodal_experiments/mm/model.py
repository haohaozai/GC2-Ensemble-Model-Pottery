"""Projection heads into the shared latent space + multi-modal symmetric InfoNCE training.

Frozen encoder features (bank) -> per-modality projection head (linear or 2-layer MLP; BiLSTM tower for raw
tokens) -> D-dim, L2-normalised. Loss: for every modality pair (text-image, text-formula, image-formula) the
symmetric InfoNCE over the batch rows that have both modalities; the same recipe is the positive; recipes with
identical tag text / identical formula are removed from the negatives (they are not reliable negatives).
Missing modalities are skipped, never filled with placeholders.
"""
from __future__ import annotations

import copy
import itertools
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import metrics as MT
from .common import gpu_peak_gb, gpu_peak_reset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PAIRS_ALL = [("text", "image"), ("text", "formula"), ("image", "formula")]


class Head(nn.Module):
    def __init__(self, d_in, dim, head, hidden, dropout):
        super().__init__()
        if head == "linear":
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, dim))
        else:
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x):
        return Fn.normalize(self.net(x), dim=-1)


class LSTMHead(nn.Module):
    def __init__(self, vocab, dim, head, hidden, dropout, emb=128, rnn=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, rnn, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = Head(2 * rnn, dim, head, hidden, dropout)

    def forward(self, tok):
        m = (tok > 0).float()
        h, _ = self.rnn(self.drop(self.emb(tok)))
        v = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        return self.head(v)


class MultiModal(nn.Module):
    def __init__(self, dims: dict, cfg: dict):
        super().__init__()
        self.dim = cfg["dim"]
        self.towers = nn.ModuleDict()
        for m, d in dims.items():
            if isinstance(d, tuple) and d[0] == "tokens":
                self.towers[m] = LSTMHead(d[1], cfg["dim"], cfg["head"], cfg["hidden"], cfg["dropout"])
            else:
                self.towers[m] = Head(d, cfg["dim"], cfg["head"], cfg["hidden"], cfg["dropout"])
        t0 = cfg["temperature"]
        init = math.log(1 / (0.07 if t0 == "learn" else float(t0)))
        self.logit_scale = nn.Parameter(torch.tensor(init), requires_grad=(t0 == "learn"))

    def scale(self):
        return self.logit_scale.clamp(max=math.log(100.0)).exp()

    def forward(self, m, x):
        return self.towers[m](x)


def default_cfg(**kw) -> dict:
    cfg = {"text": "minilm_tags", "image": "clip", "formula": "umf", "modalities": ["text", "image", "formula"],
           "head": "mlp", "hidden": 512, "dim": 128, "dropout": 0.1, "temperature": "learn", "lr": 1e-3,
           "weight_decay": 1e-4, "batch": 256, "max_epochs": 150, "patience": 12, "seed": 0}
    cfg.update(kw)
    return cfg


class Data:
    """Tensors of one feature configuration on the GPU (rows = all bank recipes)."""

    def __init__(self, bank, cfg, fit_idx):
        self.X, self.M, self.dims, self.info = {}, {}, {}, {}
        for m in cfg["modalities"]:
            X, M, info = bank.get(m, cfg[m], fit_idx, seed=cfg["seed"])
            self.M[m] = torch.tensor(M, device=DEV)
            if X.dtype == np.int64:
                self.X[m] = torch.tensor(X, device=DEV)
                self.dims[m] = ("tokens", int(info["vocab"]))
            else:
                self.X[m] = torch.tensor(X, device=DEV)
                self.dims[m] = X.shape[1]
            self.info[m] = info
        self.text_gid = torch.tensor(bank.text_gid, device=DEV)
        self.form_gid = torch.tensor(bank.form_gid, device=DEV)


def info_nce(za, zb, scale, same):
    logits = scale * za @ zb.T
    if same is not None:
        eye = torch.eye(len(za), dtype=torch.bool, device=za.device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))
    t = torch.arange(len(za), device=za.device)
    return (Fn.cross_entropy(logits, t) + Fn.cross_entropy(logits.T, t)) / 2


@torch.no_grad()
def embed(model, data, idx) -> dict:
    model.eval()
    out = {}
    it = torch.tensor(np.asarray(idx), device=DEV)
    for m in model.towers:
        Z = torch.zeros((len(it), model.dim), device=DEV)
        mk = data.M[m][it]
        rows = it[mk]
        for s in range(0, len(rows), 2048):
            Z[torch.nonzero(mk)[s:s + 2048, 0]] = model(m, data.X[m][rows[s:s + 2048]])
        out[m] = (Z.cpu().numpy(), mk.cpu().numpy())
    return out


def val_score(model, data, idx, bank) -> dict:
    Z = embed(model, data, idx)
    res = MT.pairwise_same_recipe(Z, bank, idx, ks=(1, 5, 10))
    mrrs = [v["MRR"] for v in res.values()]
    return {"mean_MRR": float(np.mean(mrrs)) if mrrs else 0.0, "directions": res}


def train(bank, cfg: dict, fit_idx, val_idx, log_every: int = 0, logger=None):
    """Train on fit_idx, early-stop on val_idx (mean MRR over modality directions). Returns (model, data, history)."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    gpu_peak_reset()
    t0 = time.time()
    data = Data(bank, cfg, fit_idx)
    model = MultiModal(data.dims, cfg).to(DEV)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pairs = [p for p in PAIRS_ALL if p[0] in cfg["modalities"] and p[1] in cfg["modalities"]]
    fit = np.asarray(fit_idx)
    hist, best, best_state, bad = [], -1.0, None, 0
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = torch.tensor(np.random.permutation(fit), device=DEV)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), cfg["batch"]):
            b = perm[s:s + cfg["batch"]]
            loss = 0.0
            used = 0
            for a, c in pairs:
                rows = b[data.M[a][b] & data.M[c][b]]
                if len(rows) < 4:
                    continue
                za, zc = model(a, data.X[a][rows]), model(c, data.X[c][rows])
                same = None
                if "text" in (a, c):
                    g = data.text_gid[rows]
                    same = g[:, None] == g[None, :]
                if "formula" in (a, c):
                    g = data.form_gid[rows]
                    sf = g[:, None] == g[None, :]
                    same = sf if same is None else (same | sf)
                loss = loss + info_nce(za, zc, model.scale(), same)
                used += 1
            if used:
                loss = loss / used
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
        vs = val_score(model, data, val_idx, bank)
        hist.append({"epoch": ep + 1, "train_loss": round(tot / max(1, nb), 4), "val_mean_MRR": round(vs["mean_MRR"], 4),
                     "temperature": round(1 / float(model.scale()), 4)})
        if logger and log_every and (ep + 1) % log_every == 0:
            logger(f"    epoch {ep + 1}: loss {hist[-1]['train_loss']:.4f} val MRR {vs['mean_MRR']:.4f}")
        if vs["mean_MRR"] > best + 1e-4:
            best, bad, best_state = vs["mean_MRR"], 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    meta = {"best_val_mean_MRR": round(best, 4), "epochs": len(hist), "best_epoch": len(hist) - bad,
            "seconds": round(time.time() - t0, 1), "peak_vram_GB": gpu_peak_gb(), "feature_info": data.info,
            "params": sum(p.numel() for p in model.parameters())}
    return model, data, hist, meta


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]
