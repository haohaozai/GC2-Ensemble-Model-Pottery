"""Image branch features, computed per segmented OBJECT and pooled per photo (area-weighted).

Encoders (all frozen):
  hand      handcrafted: CIELAB / HSV statistics, ColorThief palette (Lab), colour variation, LBP + GLCM texture,
            edge density, gloss highlight statistics                                  (measured, not inferred)
  clip      OpenCLIP ViT-B/32 laion2b (cached) on the object view (object on white, 4 % margin)
  dinov2    DINOv2 ViT-L/14 (cached): patch tokens averaged inside the mask (masked spatial pooling)
  resnet50  torchvision ResNet-50 ImageNet: layer4 map averaged inside the mask
  effb0     torchvision EfficientNet-B0 ImageNet: last feature map averaged inside the mask
Object features are cached in cache/img_obj_<encoder>.npz (resumable, shared by smoke / full / external runs).
"""
from __future__ import annotations

import importlib.util
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .common import CACHE, CODE, SEG, Timer, log, obj_file

ENCODERS = ["hand", "clip", "dinov2", "resnet50", "effb0"]
_MODS = {}


def _script(name, file):
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location(name, CODE / "scripts" / file)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[name] = m
    return _MODS[name]


def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def padded_views(a: np.ndarray):
    """rgb, mask with a 4 % white margin (rebuilds the object view used by the existing CLIP tagger)."""
    rgb, mask = np.ascontiguousarray(a[..., :3]), a[..., 3] > 127
    pad = int(0.04 * max(mask.shape))
    rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    mask = np.pad(mask, ((pad, pad), (pad, pad)), constant_values=False)
    return rgb, mask


# ---------------------------------------------------------------- handcrafted
def hand_features(a: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2lab, deltaE_cie76
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, canny
    import cv2
    rgb_full, mask_full = padded_views(a)
    hf = _script("retag16", "16_retag_attributes.py").highlight_features(rgb_full, mask_full)     # 6 gloss stats
    im = Image.fromarray(a)
    s = 256 / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    b = np.asarray(im)
    rgb, m = b[..., :3].astype(np.float32) / 255.0, b[..., 3] > 127
    if m.sum() < 30:
        m = np.ones(m.shape, bool)
    lab = rgb2lab(rgb)
    px = lab[m]
    stats = []
    for c in range(3):
        v = px[:, c]
        stats += [v.mean(), v.std(), *np.percentile(v, [10, 50, 90])]
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    h, sat, val = hsv[..., 0][m] / 180.0, hsv[..., 1][m] / 255.0, hsv[..., 2][m] / 255.0
    hh, _ = np.histogram(h, bins=12, range=(0, 1), weights=sat)
    hh = hh / max(1e-6, hh.sum())
    chroma = np.sqrt(px[:, 1] ** 2 + px[:, 2] ** 2)
    var = deltaE_cie76(px, px.mean(0, keepdims=True)).mean()
    L = lab[..., 0]
    Lm = np.where(m, L, L[m].mean())
    g8 = np.clip(Lm / 100 * 31, 0, 31).astype(np.uint8)
    lbp = local_binary_pattern(g8, 8, 1, method="uniform")
    lh, _ = np.histogram(lbp[m], bins=10, range=(0, 10))
    lh = lh / max(1, lh.sum())
    glcm = graycomatrix(g8, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                        symmetric=True, normed=True)
    gl = [graycoprops(glcm, p).mean(axis=1) for p in ("contrast", "homogeneity", "energy", "correlation")]
    er = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edges = canny(L / 100.0, sigma=1.5)
    ed = float(edges[er].mean()) if er.sum() > 20 else 0.0
    out = np.concatenate([np.array(stats), hh, [sat.mean(), sat.std(), val.mean(), val.std()],
                          [chroma.mean(), chroma.std(), var], lh, np.concatenate(gl), [ed], hf]).astype(np.float32)
    return np.nan_to_num(out)


def palette_lab(a: np.ndarray) -> list:
    """ColorThief palette (same function as scripts/17) -> [[hex, fraction], ...]."""
    pal, _ = _script("colorthief17", "17_colorthief_colors.py").colorthief_palette(Image.fromarray(a))
    return [[c["hex"], c["fraction"]] for c in pal[:3]]


# ---------------------------------------------------------------- deep encoders
class Deep:
    def __init__(self, name: str):
        import torch
        self.torch, self.name = torch, name
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        if name == "clip":
            from glazeprep.ceramic_clip import CeramicClip
            self.clip = CeramicClip(device=self.dev)
            self.dim = 512
        elif name == "dinov2":
            import os
            repo = Path(os.environ["TORCH_HOME"]) / "hub" / "facebookresearch_dinov2_main"
            self.m = torch.hub.load(str(repo), "dinov2_vitl14", source="local", pretrained=True).to(self.dev).eval()
            self.dim, self.size = 1024, 224
        elif name == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.m = torch.nn.Sequential(*list(m.children())[:-2]).to(self.dev).eval()
            self.dim, self.size = 2048, 224
        elif name == "effb0":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            self.m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features.to(self.dev).eval()
            self.dim, self.size = 1280, 224
        else:
            raise ValueError(name)

    def prep(self, a: np.ndarray):
        """square canvas, neutral grey outside the mask; returns (CHW float, mask HxW)."""
        rgb, m = a[..., :3], a[..., 3] > 127
        h, w = m.shape
        S = max(h, w)
        can = np.full((S, S, 3), 128, np.uint8)
        mm = np.zeros((S, S), np.uint8)
        y0, x0 = (S - h) // 2, (S - w) // 2
        can[y0:y0 + h, x0:x0 + w] = np.where(m[..., None], rgb, 128)
        mm[y0:y0 + h, x0:x0 + w] = m
        im = np.asarray(Image.fromarray(can).resize((self.size, self.size), Image.BICUBIC), np.float32) / 255.0
        mk = np.asarray(Image.fromarray(mm * 255).resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.0
        return im.transpose(2, 0, 1), mk

    def encode(self, arrays: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        if self.name == "clip":
            from glazeprep.ceramic_tagger import CeramicTagger
            objs = [CeramicTagger.views(*padded_views(a))[0] for a in arrays]
            with torch.inference_mode():
                return self.clip.image_embeddings(objs).float().cpu().numpy()
        ims, mks = zip(*(self.prep(a) for a in arrays))
        x = torch.tensor(np.stack(ims), device=self.dev)
        x = (x - self.mean) / self.std
        mk = torch.tensor(np.stack(mks), device=self.dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            if self.name == "dinov2":
                f = self.m.forward_features(x)["x_norm_patchtokens"].float()           # B, 256, 1024
                g = int(f.shape[1] ** 0.5)
                w = torch.nn.functional.adaptive_avg_pool2d(mk[:, None], g).flatten(1)  # B, 256
            else:
                fm = self.m(x).float()                                                  # B, C, 7, 7
                w = torch.nn.functional.adaptive_avg_pool2d(mk[:, None], fm.shape[-1]).flatten(1)
                f = fm.flatten(2).transpose(1, 2)                                       # B, 49, C
            w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
            v = (f * w[..., None]).sum(1)
        return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()


# ---------------------------------------------------------------- cache + extraction
def cache_path(enc: str, tag: str = "") -> Path:
    return CACHE / f"img_obj_{enc}{('_' + tag) if tag else ''}.npz"


def load_cache(enc: str, tag: str = ""):
    p = cache_path(enc, tag)
    if not p.exists():
        return {}
    with np.load(p, allow_pickle=False) as z:
        return {i: v for i, v in zip(z["ids"].tolist(), z["X"])}


def save_cache(enc: str, d: dict, tag: str = ""):
    p = cache_path(enc, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    ids = list(d)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, ids=np.array(ids), X=np.stack([d[i] for i in ids]).astype(np.float32))
    tmp.replace(p)


def extract(enc: str, objects: list[tuple[str, Path]], tag: str = "", batch: int = 64, workers: int = 6) -> dict:
    """objects: [(object_id, rgba path)] -> {object_id: vector}; only missing objects are computed."""
    have = load_cache(enc, tag)
    todo = [(o, p) for o, p in objects if o not in have]
    log().info(f"[image:{enc}] objects {len(objects)} | cached {len(objects) - len(todo)} | to compute {len(todo)}")
    if not todo:
        return {o: have[o] for o, _ in objects}
    t = Timer()
    if enc == "hand":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            for i, (o, v) in enumerate(zip([o for o, _ in todo], ex.map(_hand_job, [str(p) for _, p in todo], chunksize=16)), 1):
                have[o] = v
                if i % 2000 == 0:
                    save_cache(enc, have, tag)
                    log().info(f"[image:hand] {i}/{len(todo)} | {t() / 60:.1f} min")
    else:
        model = Deep(enc)
        with ThreadPoolExecutor(workers) as ex:
            for s in range(0, len(todo), batch):
                chunk = todo[s:s + batch]
                arrays = list(ex.map(lambda op: load_rgba(op[1]), chunk))
                vecs = model.encode(arrays)
                for (o, _), v in zip(chunk, vecs):
                    have[o] = v
                done = s + len(chunk)
                if done % (batch * 40) == 0 or done == len(todo):
                    save_cache(enc, have, tag)
                    log().info(f"[image:{enc}] {done}/{len(todo)} | {t() / 60:.1f} min")
        del model
        import torch
        torch.cuda.empty_cache()
    save_cache(enc, have, tag)
    return {o: have[o] for o, _ in objects}


def _hand_job(path: str) -> np.ndarray:
    return hand_features(load_rgba(Path(path)))


def internal_objects(objects_csv_rows) -> list[tuple[str, Path]]:
    """(object_id, RGBA path); external objects carry their own segmentation root."""
    return [(r["object_id"], Path(r["root"]) / "smaller_object_rgba_raw" / f"{r['object_id']}.png" if r.get("root")
             else obj_file("smaller_object_rgba_raw", f"{r['object_id']}.png")) for r in objects_csv_rows]


def pool_recipes(obj_vecs: dict, objects_rows, recipe_ids: list[str], normalise: bool) -> tuple[np.ndarray, np.ndarray]:
    """area-weighted mean of a recipe's object vectors -> [n_recipes, d], mask."""
    by = {}
    for r in objects_rows:
        if r["object_id"] in obj_vecs:
            by.setdefault(r["recipe_id"], []).append((obj_vecs[r["object_id"]], float(r["area_px"])))
    d = len(next(iter(obj_vecs.values())))
    X = np.zeros((len(recipe_ids), d), np.float32)
    M = np.zeros(len(recipe_ids), bool)
    for i, rid in enumerate(recipe_ids):
        if rid in by:
            V = np.stack([v for v, _ in by[rid]])
            w = np.array([a for _, a in by[rid]], np.float32)
            X[i] = (V * (w / w.sum())[:, None]).sum(0)
            M[i] = True
    if normalise:
        X[M] /= np.linalg.norm(X[M], axis=1, keepdims=True).clip(1e-8)
    return X, M
