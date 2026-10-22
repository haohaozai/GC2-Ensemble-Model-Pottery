"""Test B: fine-tuning Grounding DINO's own weights on the six attributes (Claude's labels), compared with approach A.

Grounding DINO base (IDEA-Research/grounding-dino-base) is trained as a phrase-grounding detector: for an object cut-out
and the prompt of one attribute (one phrase per class, the "names" prompt set, e.g. "glossy glaze . satin glaze . ..."),
the target is the object's box labelled with Claude's class (effects / defects: one box per present effect; none = no
box, so the model also learns to say "nothing"). The Swin image backbone, the BERT text encoder and the feature
enhancer stay frozen; the decoder and the box head are trained (HF GroundingDinoForObjectDetection loss:
Hungarian matching, focal alignment loss, L1 + GIoU box loss).

Data: the 1,504 Claude-labelled TRAIN objects; the epoch is chosen on the VALIDATION objects; TEST is scored once.
Score of a class after fine-tuning = the best grounding score of its phrase (max over the 900 queries), like the raw
pretrained evidence. GPU: one attribute prompt per step, batch 1, 512 px, bf16 autocast, VRAM capped (OOM raises an
error instead of filling the card), temperature guard (mm/dino_attr.GpuGuard).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from . import claude_labels as CL
from . import dino_attr as DA
from . import dino_calib as DC

ATTRS = ["form", "surface_finish", "transparency", "texture", "glaze_effect", "defect"]
VARIANT = "names"          # one phrase per class


def target_box(rgba: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """cut-out (as Grounding DINO sees it) + the object's box in it, normalised (cx, cy, w, h)"""
    img, (x0, y0, x1, y1) = DA.cutout(rgba)
    ys, xs = np.nonzero(rgba[..., 3] >= 125)
    H, W = img.shape[:2]
    if not len(ys):
        return img, [0.5, 0.5, 1.0, 1.0]
    bx0, by0, bx1, by1 = xs.min() - x0, ys.min() - y0, xs.max() + 1 - x0, ys.max() + 1 - y0
    return img, [(bx0 + bx1) / 2 / W, (by0 + by1) / 2 / H, (bx1 - bx0) / W, (by1 - by0) / H]


def class_index(attr: str) -> dict:
    return {c: j for j, (c, _) in enumerate(DA.phrase_list(attr, VARIANT))}


def targets(lab: dict, attr: str) -> list[int] | None:
    """class indices of the object's label (None = Claude could not judge -> not used)"""
    v = lab.get(attr)
    if v is None:
        return None
    ci = class_index(attr)
    return [ci[x] for x in v if x in ci] if isinstance(v, list) else ([ci[v]] if v in ci else None)


class Trainer:
    def __init__(self, load_rgba, vram_cap_gb=4.5, lr=2e-5, seed=0, log=print):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        DA.sys_path_glazeprep()
        from glazeprep import config as C
        self.torch, self.log, self.load_rgba = torch, log, load_rgba
        self.guard = DA.GpuGuard(vram_cap_gb=vram_cap_gb, duty_pause=0.03)
        self.dev = "cuda"
        self.proc = AutoProcessor.from_pretrained(C.GROUNDING_DINO_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(C.GROUNDING_DINO_ID).to(self.dev)
        m = self.model.model
        # train only the decoder and the box head: no gradient flows through the encoder / backbones, so their
        # activations are not stored (the first ramp test with the encoder trainable hit the 4.5 GB cap)
        for p in self.model.parameters():
            p.requires_grad = False
        for mod in (m.decoder, self.model.bbox_embed):
            for p in mod.parameters():
                p.requires_grad = True
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.n_train = sum(p.numel() for p in self.params)
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-4)
        self.rng = random.Random(seed)
        self.da = DA.DinoAttr(model=self.model, proc=self.proc)      # scoring with the model being trained
        self.cache = {}

    def sample(self, oid):
        if oid not in self.cache:
            from PIL import Image
            img, box = target_box(self.load_rgba(oid))
            pil = self.da._resize(Image.fromarray(img))
            self.cache[oid] = (pil, box)
            if len(self.cache) > 600:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[oid]

    def step(self, attr: str, items: list[tuple[str, list[int]]]) -> float:
        torch = self.torch
        self.model.train()
        pils, labels = [], []
        for oid, cls in items:
            pil, box = self.sample(oid)
            pils.append(pil)
            labels.append({"class_labels": torch.tensor(cls, dtype=torch.long, device=self.dev),
                           "boxes": torch.tensor([box] * len(cls), dtype=torch.float32, device=self.dev).reshape(-1, 4)})
        text = DA.prompt_text(attr, VARIANT)
        inp = self.proc(images=pils, text=[text] * len(pils), return_tensors="pt", do_resize=False, padding=True).to(self.dev)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(**inp, labels=labels)
        # decoder losses only (final + auxiliary layers), weights as in the HF config: alignment 1, L1 box 5, GIoU 2.
        # The two-stage encoder-proposal terms (*_enc) are left out: that stage is frozen here and its alignment term is
        # numerically broken for these prompts (~2e5 on the ramp test, while the decoder terms are ~1).
        W = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        loss = sum(W[k.rsplit("_", 1)[0] if k[-1].isdigit() else k] * v for k, v in out.loss_dict.items()
                   if "enc" not in k and (k.rsplit("_", 1)[0] if k[-1].isdigit() else k) in W)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 0.1)
        self.opt.step()
        self.guard.after_pass(time.time() - t0)
        return float(loss.detach())

    def scores(self, oids: list[str], attr: str) -> np.ndarray:
        """[N, C] class scores (best grounding score of each class phrase) with the current weights"""
        torch = self.torch
        self.model.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for oid in oids:
                pil, _ = self.sample(oid)
                r = self.da._pass(pil, (attr, VARIANT))
                out.append(r["score"])
        return np.stack(out).astype(np.float32)


def recall_single(S, y, ks=(1, 2, 3)):
    order = np.argsort(-S, 1)
    return {f"R@{k}": round(float(np.mean([y[i] in order[i, :k] for i in range(len(y))])), 4) for k in ks}


def recall_multi(S, Y, ks=(1, 2, 3)):
    m = Y.sum(1) > 0
    order = np.argsort(-S[m], 1)
    return {f"R@{k}": round(float(np.mean([Y[m][i, order[i, :k]].sum() / Y[m][i].sum() for i in range(m.sum())])), 4) if m.any() else None
            for k in ks}


def evaluate(tr: Trainer, ids: list[str], lab: dict, attrs=ATTRS) -> dict:
    res = {}
    for a in attrs:
        ok = [o for o in ids if targets(lab[o], a) is not None]
        S = tr.scores(ok, a)
        C = len(class_index(a))
        if a in ("glaze_effect", "defect"):
            Y = np.zeros((len(ok), C))
            for i, o in enumerate(ok):
                Y[i, targets(lab[o], a)] = 1
            res[a] = {"n": len(ok), **recall_multi(S, Y), "mAP": DC._map(Y, S)}
        else:
            y = np.array([targets(lab[o], a)[0] for o in ok])
            res[a] = {"n": len(ok), **recall_single(S, y)}
    return res


def run(v1: Path, v2: Path, v3: Path, load_rgba, epochs=3, batch=1, limit_steps=0, log=print) -> dict:
    split = {}
    import csv
    with (v1 / "labels.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            split[r["object_id"]] = r["split"]
    lab = {**{o: r for o, r in CL.parse(v3 / "claude_labels_train_raw.txt").items() if split.get(o) == "train"},
           **{o: r for o, r in CL.parse(v2 / "claude_labels_raw.txt").items() if split.get(o) in ("val", "test")}}
    tr_ids = sorted(o for o in lab if split[o] == "train")
    va_ids = sorted(o for o in lab if split[o] == "val")
    te_ids = sorted(o for o in lab if split[o] == "test")
    T = Trainer(load_rgba, log=log)
    torch = T.torch
    log(f"[dino_ft] trainable parameters {T.n_train / 1e6:.1f} M (decoder + box head; backbones and encoder frozen) | train {len(tr_ids)} val {len(va_ids)} test {len(te_ids)}")
    out = {"method": __doc__.strip().splitlines()[0], "objects": {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)},
           "trainable_M": round(T.n_train / 1e6, 1), "epochs": [], "label_source": CL.PROVENANCE}
    t0 = time.time()
    out["val_epoch0"] = evaluate(T, va_ids, lab) if not limit_steps else None       # pretrained, names prompt
    best = (-1, None, None)
    for ep in range(1, epochs + 1):
        work = [(a, o, targets(lab[o], a)) for o in tr_ids for a in ATTRS if targets(lab[o], a) is not None]
        T.rng.shuffle(work)
        by_attr = {a: [(o, c) for aa, o, c in work if aa == a] for a in ATTRS}
        steps = []
        for a, items in by_attr.items():
            steps += [(a, items[i:i + batch]) for i in range(0, len(items), batch)]
        T.rng.shuffle(steps)
        losses, ts = [], time.time()
        for s, (a, items) in enumerate(steps, 1):
            losses.append(T.step(a, items))
            if s % 200 == 0 or s == len(steps) or (limit_steps and s == limit_steps):
                peak = torch.cuda.max_memory_allocated() / 1e9
                st = T.guard.gpu_state()
                log(f"[dino_ft] epoch {ep} step {s}/{len(steps)} | loss {np.mean(losses[-200:]):.3f} | {(time.time() - ts) / s:.2f} s/step | "
                    f"peak VRAM {peak:.2f} GB | GPU {st.get('temp_C', '?')} C {st.get('vram_used_MB', '?')} MB | {(time.time() - t0) / 60:.1f} min")
                T.guard.after_object()
            if limit_steps and s >= limit_steps:
                return {"ramp_test_steps": s, "loss_first": round(float(np.mean(losses[:10])), 3), "loss_last": round(float(np.mean(losses[-10:])), 3),
                        "s_per_step": round((time.time() - ts) / s, 2), "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        val = evaluate(T, va_ids, lab)
        mean_r1 = float(np.mean([v["R@1"] for v in val.values() if v["R@1"] is not None]))
        out["epochs"].append({"epoch": ep, "train_loss": round(float(np.mean(losses)), 4), "val": val, "val_mean_R@1": round(mean_r1, 4),
                              "minutes": round((time.time() - t0) / 60, 1)})
        log(f"[dino_ft] epoch {ep} done | val mean R@1 {mean_r1:.4f} | " + json.dumps({a: v['R@1'] for a, v in val.items()}))
        if mean_r1 > best[0]:
            best = (mean_r1, ep, {k: v.detach().to("cpu", copy=True) for k, v in T.model.state_dict().items() if "backbone" not in k})
    out["selected_epoch"] = best[1]
    T.model.load_state_dict(best[2], strict=False)
    out["test"] = evaluate(T, te_ids, lab)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    torch.save(best[2], v3 / "dino_finetuned_heads.pt")          # trained (non-backbone) weights of the selected epoch
    return out
