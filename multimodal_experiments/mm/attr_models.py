"""Surface / transparency / glaze-effect / defect / form recognition: model comparison on frozen features.

Tasks (labels from mm/attr_labels.py; object split = the recipe's group split of the multimodal experiments):
  surface_group   3 classes (glossy / satin / matte)      Glazy metadata, single label
  surface_finish  9 classes                               Glazy metadata, single label
  transparency    4 classes                               Glazy metadata, single label
  transparency_group 2 classes (clear / opaque)           Glazy metadata, single label
  glaze_effect    8 labels, multi-label                   recipe text (incomplete: a missing mention is not a
  defect          5 labels, multi-label                    reliable negative -> precision is a lower bound)
Methods: old tagger (baseline, where comparable) · zero-shot CLIP with guide prompts · logistic regression on
frozen features (C on validation) · MLP head (BCE vs asymmetric loss for multi-label, CE for single-label).
Multi-label thresholds per class on validation (max F1); single-label abstention threshold on validation
(Unknown when the top probability is below it). Test is scored once for the selected method of every task.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import attr_labels as AL
from .common import CACHE, log

TASKS = {"surface_group": ("single", "surface_group_meta", ["glossy", "satin", "matte"]),
         "surface_finish": ("single", "surface_meta", AL.SURFACE),
         "transparency": ("single", "transparency_meta", AL.TRANSP),
         "transparency_group": ("single", "transparency_group_meta", ["clear", "opaque"]),
         "glaze_effect": ("multi", "effects_text", AL.EFFECTS),
         "defect": ("multi", "defects_text", AL.DEFECTS)}
ZS_FIELD = {"surface_group": "surface_finish", "surface_finish": "surface_finish", "transparency": "transparency",
            "transparency_group": "transparency", "glaze_effect": "glaze_effect", "defect": "defect"}


# ---------------------------------------------------------------- data
def _load_npz(p, ids):
    z = np.load(p, allow_pickle=False)
    pos = {i: k for k, i in enumerate(z["ids"].tolist())}
    ZX = z["X"]                       # read once (each z[...] access decompresses the whole array)
    X = np.zeros((len(ids), ZX.shape[1]), np.float32)
    M = np.zeros(len(ids), bool)
    for k, i in enumerate(ids):
        j = pos.get(i)
        if j is not None:
            X[k], M[k] = ZX[j], True
    return X, M


def feature_sets(work, ids) -> dict:
    fs = {}
    for name in ("clip", "dinov2", "resnet50", "effb0", "hand"):
        p = CACHE / f"img_obj_{name}.npz"
        if p.exists():
            fs[name] = _load_npz(p, ids)
    for name, f in (("dinov2_hr", "feat_dinov2_hr.npz"),):
        if (work / f).exists():
            fs[name] = _load_npz(work / f, ids)
    if "hand" in fs:          # standardise the handcrafted block (other blocks are L2-normalised embeddings)
        X, M = fs["hand"]
        mu, sd = X[M].mean(0), X[M].std(0) + 1e-6
        fs["hand"] = (np.where(M[:, None], (X - mu) / sd / np.sqrt(X.shape[1]), 0).astype(np.float32), M)
    combos = [("clip+dinov2", ["clip", "dinov2"]), ("clip+dinov2_hr+hand", ["clip", "dinov2_hr", "hand"])]
    for name, parts in combos:
        if all(p in fs for p in parts):
            fs[name] = (np.concatenate([fs[p][0] for p in parts], 1), np.all([fs[p][1] for p in parts], 0))
    return fs


def targets(rows, task):
    kind, col, classes = TASKS[task]
    idx = {c: i for i, c in enumerate(classes)}
    n = len(rows)
    if kind == "single":
        y = np.full(n, -1)
        for k, r in enumerate(rows):
            if r[col] in idx:
                y[k] = idx[r[col]]
        return y, y >= 0
    Y = np.zeros((n, len(classes)), np.float32)
    has = np.zeros(n, bool)
    for k, r in enumerate(rows):
        labs = json.loads(r[col] or "[]")
        for l in labs:
            if l in idx:
                Y[k, idx[l]] = 1
        has[k] = r["has_description"] == "1"       # objects whose recipe has a subtype / description
    return Y, has


# ---------------------------------------------------------------- metrics
def single_metrics(y, P, classes, thr=0.0):
    from sklearn.metrics import confusion_matrix, f1_score
    pred = P.argmax(1)
    conf = P.max(1)
    keep = conf >= thr
    out = {"n": int(len(y)), "accuracy": round(float((pred == y).mean()), 4),
           "macro_F1": round(float(f1_score(y, pred, labels=range(len(classes)), average="macro", zero_division=0)), 4),
           "per_class_F1": {c: round(float(f), 4) for c, f in zip(classes, f1_score(y, pred, labels=range(len(classes)), average=None, zero_division=0))},
           "confusion": confusion_matrix(y, pred, labels=range(len(classes))).tolist()}
    if thr > 0:
        out.update({"abstain_threshold": round(float(thr), 3), "unknown_rate": round(float(1 - keep.mean()), 4),
                    "accuracy_when_answered": round(float((pred[keep] == y[keep]).mean()), 4) if keep.any() else None})
    return out


def multi_metrics(Y, S, thr, classes):
    from sklearn.metrics import average_precision_score
    Pb = (S >= thr[None, :]).astype(int)
    tp = (Pb * Y).sum(0)
    fp = (Pb * (1 - Y)).sum(0)
    fn = ((1 - Pb) * Y).sum(0)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)
    mtp, mfp, mfn = tp.sum(), fp.sum(), fn.sum()
    micro = 2 * mtp / max(1, 2 * mtp + mfp + mfn)
    ap = [float(average_precision_score(Y[:, j], S[:, j])) if Y[:, j].sum() > 0 else None for j in range(Y.shape[1])]
    return {"n": int(len(Y)), "micro_F1": round(float(micro), 4), "macro_F1": round(float(f1.mean()), 4),
            "mAP": round(float(np.mean([a for a in ap if a is not None])), 4) if any(a is not None for a in ap) else None,
            "exact_match": round(float((Pb == Y).all(1).mean()), 4), "hamming_loss": round(float((Pb != Y).mean()), 4),
            "per_class": {c: {"positives": int(Y[:, j].sum()), "precision": round(float(prec[j]), 4), "recall": round(float(rec[j]), 4),
                              "F1": round(float(f1[j]), 4), "AP": None if ap[j] is None else round(ap[j], 4)} for j, c in enumerate(classes)}}


def tune_thresholds(Y, S):
    thr = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        best, bt = -1, 0.5
        for t in np.unique(np.quantile(S[:, j], np.linspace(0.5, 0.995, 80))):
            p = S[:, j] >= t
            tp, fp, fn = (p & (Y[:, j] == 1)).sum(), (p & (Y[:, j] == 0)).sum(), (~p & (Y[:, j] == 1)).sum()
            f = 2 * tp / max(1, 2 * tp + fp + fn)
            if f > best:
                best, bt = f, t
        thr[j] = bt
    return thr


def abstain_threshold(y, P, target=0.9):
    """smallest confidence threshold reaching `target` accuracy on validation (0 if never)."""
    conf, ok = P.max(1), P.argmax(1) == y
    for t in np.linspace(0, 0.95, 96):
        k = conf >= t
        if k.sum() >= 20 and ok[k].mean() >= target:
            return float(t)
    return 0.0


# ---------------------------------------------------------------- models
def zero_shot_scores(X_clip, E, single=True, scale=100.0):
    S = X_clip @ E.T
    if single:
        S = np.exp((S - S.max(1, keepdims=True)) * scale)
        return S / S.sum(1, keepdims=True)
    return S


class LinearLR:
    """L2-regularised, class-balanced logistic regression (softmax for single-label, one sigmoid per class for
    multi-label = one-vs-rest), fitted with full-batch L-BFGS on the GPU. Objective = mean weighted log-loss
    + ||W||^2 / (2 C n), i.e. scikit-learn's LogisticRegression objective divided by C n. Predicts on the CPU."""

    def __init__(self, kind, C):
        self.kind, self.C = kind, C

    def fit(self, X, y):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Xt = torch.tensor(X, device=dev, dtype=torch.float32)
        n, d = Xt.shape
        if self.kind == "single":
            nC = int(y.max()) + 1
            self.classes_ = np.arange(nC)
            cnt = np.bincount(y, minlength=nC).astype(np.float64)
            w = torch.tensor(np.where(cnt > 0, n / (nC * np.maximum(cnt, 1)), 0), device=dev, dtype=torch.float32)
            yt = torch.tensor(y, device=dev, dtype=torch.long)
            lossf = lambda o: torch.nn.functional.cross_entropy(o, yt, weight=w, reduction="sum") / w[yt].sum()  # noqa: E731
        else:
            nC = y.shape[1]
            self.classes_ = np.arange(nC)
            pos = y.mean(0).clip(1e-4, 1 - 1e-4)
            yt = torch.tensor(y, device=dev, dtype=torch.float32)
            pw = torch.tensor((1 - pos) / pos, device=dev, dtype=torch.float32)
            lossf = lambda o: torch.nn.functional.binary_cross_entropy_with_logits(o, yt, pos_weight=pw)  # noqa: E731
        W = torch.zeros((d, nC), device=dev, requires_grad=True)
        b = torch.zeros(nC, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=300, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = lossf(Xt @ W + b) + (W * W).sum() / (2 * self.C * n)
            loss.backward()
            return loss
        opt.step(closure)
        self.W, self.b = W.detach().cpu().numpy(), b.detach().cpu().numpy()
        return self

    def predict_proba(self, X):
        o = np.asarray(X, np.float32) @ self.W + self.b
        if self.kind == "single":
            o = np.exp(o - o.max(1, keepdims=True))
            return o / o.sum(1, keepdims=True)
        return 1 / (1 + np.exp(-o))


def fit_lr(X, y, kind, C):
    return LinearLR(kind, C).fit(X, y)


def predict(m, X, kind, n_classes):
    P = m.predict_proba(X)
    if kind == "single" and P.shape[1] != n_classes:     # a class missing from training
        full = np.zeros((len(X), n_classes))
        full[:, m.classes_] = P
        P = full
    return P


def asl_loss(logits, y, gp=4.0, gn=1.0, clip=0.05):
    """asymmetric loss (Ridnik et al. 2021) for multi-label with many negatives."""
    import torch
    p = torch.sigmoid(logits)
    pn = (p - clip).clamp(min=0)
    lp = y * (1 - p) ** gn * torch.log(p.clamp(min=1e-8))
    ln = (1 - y) * pn ** gp * torch.log((1 - pn).clamp(min=1e-8))
    return -(lp + ln).mean()


def fit_mlp(Xtr, Ytr, Xva, Yva, kind, n_classes, loss="bce", epochs=60, seed=0, hidden=512, dropout=0.3, lr=1e-3, wd=1e-3):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    xt, xv = torch.tensor(Xtr, device=dev), torch.tensor(Xva, device=dev)
    if kind == "single":
        yt = torch.tensor(Ytr, device=dev, dtype=torch.long)
        cnt = np.bincount(Ytr, minlength=n_classes).astype(np.float32)
        w = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / n_classes, device=dev)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        yt = torch.tensor(Ytr, device=dev)
        pos = Ytr.mean(0).clip(1e-3, None)
        crit = (lambda o, t: asl_loss(o, t)) if loss == "asl" else nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev).clamp(max=50))
    best, state, bad = -1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for s in range(0, len(xt), 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            crit(net(xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            o = net(xv)
            P = (torch.softmax(o, 1) if kind == "single" else torch.sigmoid(o)).cpu().numpy()
        if kind == "single":
            from sklearn.metrics import f1_score
            score = f1_score(Yva, P.argmax(1), average="macro", zero_division=0)
        else:
            from sklearn.metrics import average_precision_score
            score = np.mean([average_precision_score(Yva[:, j], P[:, j]) for j in range(n_classes) if Yva[:, j].sum() > 0])
        if score > best + 1e-4:
            best, bad, state = score, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(state)
    net.eval()
    net.cpu()
    dev = "cpu"

    def prob(X):
        with torch.no_grad():
            o = net(torch.tensor(X, device=dev))
            return (torch.softmax(o, 1) if kind == "single" else torch.sigmoid(o)).cpu().numpy()
    return prob, float(best), ep + 1, net


def val_score(kind, y, P):
    from sklearn.metrics import average_precision_score, f1_score
    if kind == "single":
        return float(f1_score(y, P.argmax(1), average="macro", zero_division=0))
    return float(np.mean([average_precision_score(y[:, j], P[:, j]) for j in range(y.shape[1]) if y[:, j].sum() > 0]))


def run_task(task, rows, fs, zs, split, log_fn=None) -> dict:
    """compare methods on one task; returns results + the selected predictor's probabilities for ALL objects."""
    kind, col, classes = TASKS[task]
    y, lab = targets(rows, task)
    tr, va, te = (split == "train") & lab, (split == "val") & lab, (split == "test") & lab
    nC = len(classes)
    res = {"task": task, "kind": kind, "classes": classes, "label_source": col,
           "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}, "methods": []}
    if tr.sum() < 30 or va.sum() < 10:
        res["skipped"] = "too few labels"
        return res, None, None
    if kind == "multi":
        keep = [j for j in range(nC) if y[tr, j].sum() >= 5 and y[va, j].sum() >= 2]
        res["classes_used"] = [classes[j] for j in keep]
        res["classes_skipped_too_few_labels"] = [classes[j] for j in range(nC) if j not in keep]
    cand = []      # (val score, name, prob_fn over all rows, extra)
    # zero-shot
    if "clip" in fs and ZS_FIELD[task] in zs:
        zc, E = zs[ZS_FIELD[task]]
        Xc, Mc = fs["clip"]
        S = zero_shot_scores(Xc, E, single=kind == "single")
        if task == "surface_group":
            grp = [AL.SURFACE_GROUP[c] for c in zc]
            S = np.stack([S[:, [i for i, g in enumerate(grp) if g == c]].sum(1) for c in classes], 1)
        elif task == "transparency_group":
            grp = [AL.TRANSP_GROUP[c] for c in zc]
            S = np.stack([S[:, [i for i, g in enumerate(grp) if g == c]].sum(1) for c in classes], 1)
        else:
            S = S[:, [zc.index(c) for c in classes]]
        cand.append((val_score(kind, y[va & Mc], S[va & Mc]), "zero-shot CLIP (guide prompts)", lambda S=S: S, {"features": "clip"},
                     {"type": "zero_shot", "field": ZS_FIELD[task], "classes": zc, "E": E}))
    # logistic regression on each feature set
    for fname, (X, M) in fs.items():
        best = None
        for C in (0.1, 1.0, 10.0):
            t0 = time.time()
            m = fit_lr(X[tr & M], y[tr & M], kind, C)
            P = predict(m, X[va & M], kind, nC)
            sc = val_score(kind, y[va & M], P)
            if best is None or sc > best[0]:
                best = (sc, C, m, round(time.time() - t0, 1))
        sc, C, m, secs = best
        cand.append((sc, f"logistic regression on {fname}", lambda m=m, X=X: predict(m, X, kind, nC), {"features": fname, "C": C, "fit_s": secs},
                     {"type": "linear", "model": m}))
        if log_fn:
            log_fn(f"[heads] {task} | LR {fname}: val {sc:.4f} (C={C})")
    # MLP heads on the two best feature sets of the LR comparison
    lr_sorted = sorted([c for c in cand if c[1].startswith("logistic")], key=lambda c: -c[0])[:2]
    for _, _, _, ex, _ in lr_sorted:
        X, M = fs[ex["features"]]
        for loss in (("ce",) if kind == "single" else ("bce", "asl")):
            prob, sc, eps, net = fit_mlp(X[tr & M], y[tr & M], X[va & M], y[va & M], kind, nC, loss=loss)
            cand.append((sc, f"MLP ({loss.upper()}) on {ex['features']}", lambda prob=prob, X=X: prob(X), {"features": ex["features"], "loss": loss, "epochs": eps},
                         {"type": "torch_mlp", "net": net}))
            if log_fn:
                log_fn(f"[heads] {task} | MLP {loss} {ex['features']}: val {sc:.4f}")
    for sc, name, fn, ex, _ in cand:
        res["methods"].append({"method": name, "val_score": round(sc, 4), **ex})
    sc, name, fn, ex, model = max(cand, key=lambda c: c[0])
    P = fn()
    M = fs[ex["features"]][1]
    sel = {"method": name, "val_score": round(sc, 4), **ex}
    if kind == "single":
        thr = abstain_threshold(y[va & M], P[va & M], 0.9 if task.endswith("group") else 0.8)
        sel["test"] = single_metrics(y[te & M], P[te & M], classes, thr)
        sel["test_no_abstain"] = single_metrics(y[te & M], P[te & M], classes)
        sel["abstain_threshold"] = thr
    else:
        thr = tune_thresholds(y[va & M], P[va & M])
        keepc = [classes.index(c) for c in res["classes_used"]]
        sel["thresholds"] = {classes[j]: round(float(thr[j]), 4) for j in keepc}
        sel["test"] = multi_metrics(y[te & M][:, keepc], P[te & M][:, keepc], thr[keepc], res["classes_used"])
    res["selected"] = sel
    res["_model"] = {**model, "features": ex["features"], "kind": kind, "classes": classes}
    return res, P, M
