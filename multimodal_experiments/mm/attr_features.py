"""Image features for attribute recognition and shape association (attributes_v1), computed once per object.

  dinov2_hr   DINOv2 ViT-L/14 at 448 px (32 x 32 patch tokens instead of 16 x 16): masked mean + masked GeM (p = 3)
              of the patch tokens -> 2048-d. The higher resolution is meant for small structures (speckles,
              pinholes, crystals, crackle) - the "high-resolution crop" evidence of the comparison guide.
  dinov2_sil  DINOv2 ViT-L/14 CLS token of the SILHOUETTE only (black mask on white, 224 px) -> 1024-d shape embedding
              with no colour / glaze information.
  clip text   OpenCLIP ViT-B/32 text embeddings of guide-based prompt ensembles, for zero-shot classification of
              form, surface finish, transparency, texture, glaze effects and defects (image side: the cached
              CLIP object embeddings of the multimodal experiments, cache/img_obj_clip.npz).
Prompts follow the "key difference" sentences of Ceramic_Real_Photo_Comparison_Guide_V4 (the guide's photos are
NOT used for training).
"""
from __future__ import annotations

import numpy as np

TEMPLATES = ["a close-up photo of {}.", "a photo of a ceramic glaze test tile with {}.", "a photo of pottery with {}."]
FORM_TEMPLATES = ["a photo of {}.", "a product photo of {}.", "a photo of {} on a table."]
PROMPTS = {
    "form": {"bowl": "a ceramic bowl", "plate": "a flat ceramic plate or dish", "cup": "a ceramic cup or mug",
             "vase": "a ceramic vase or bottle", "jar": "a ceramic jar or lidded pot",
             "test tile": "a flat ceramic glaze test tile", "sculpture": "a ceramic sculpture",
             "figurine": "a small ceramic figurine of a person or animal", "abstract ceramic artwork": "an abstract ceramic art piece",
             "irregular ceramic form": "an irregular hand-built ceramic form", "other vessel": "a ceramic teapot or pitcher"},
    "surface_finish": {"glossy": "a glossy glaze with sharp bright reflections", "semi-glossy": "a semi-glossy glaze with softer highlights",
                       "satin": "a satin glaze with a broad gentle sheen", "satin-matte": "a satin-matte glaze with a faint sheen",
                       "semi-matte": "a semi-matte glaze with very little shine", "matte": "a matte glaze with low reflectivity",
                       "smooth matte": "an even, smooth, low-gloss matte glaze", "stony matte": "a granular, mineral-looking stony matte glaze",
                       "dry matte": "a dry, chalky matte glaze"},
    "transparency": {"transparent": "a transparent glaze where the clay and details underneath are clearly visible",
                     "translucent": "a translucent glaze with diffuse, partial visibility of what is underneath",
                     "semi-opaque": "a semi-opaque glaze that mostly hides the clay underneath",
                     "opaque": "an opaque glaze that completely conceals the material underneath"},
    "texture": {"smooth": "a smooth glaze surface", "rough": "a rough, uneven glaze surface",
                "granular": "a granular, sandy glaze surface", "dimpled": "a repeated dimpled pattern of tiny depressions in the glaze"},
    "glaze_effect": {"speckled": "small dispersed contrasting speckles", "oil spot": "larger rounded, often metallic-looking oil spots",
                     "crystalline": "branching or flower-shaped starburst crystal clusters in the glaze",
                     "mottled": "irregular mottled colour variation within the glaze",
                     "flowing": "gravity-aligned streaks and drips of flowing glaze",
                     "breaking at edges": "glaze that changes colour where it thins over raised edges and ridges",
                     "metallic / iridescent": "a metal-like or iridescent, angle-dependent coloured sheen",
                     "visible crack network (crackle)": "a fine interconnected network of cracks in the glaze"},
    "defect": {"pinholes": "tiny pin-sized holes through the glaze surface", "blistering": "raised or burst bubbles in the glaze",
               "crawling": "glaze pulled into separate islands that expose bare clay",
               "crazing": "crazing: a network of fine cracks caused by glaze fit",
               "excessive glaze running": "glaze that ran excessively and pooled thickly at the base"},
}


def text_embeddings(clip) -> dict:
    """field -> (classes, [C, 512] normalised mean over templates)"""
    import torch
    out = {}
    for field, cls in PROMPTS.items():
        tpl = FORM_TEMPLATES if field == "form" else TEMPLATES
        E = []
        for c, desc in cls.items():
            e = clip.text_embeddings([t.format(desc) for t in tpl]).mean(0)
            E.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        out[field] = (list(cls), np.stack(E).astype(np.float32))
    return out


def _gem(f, w, p=3.0):
    import torch
    f = f.clamp_min(1e-6) if (f >= 0).all() else f - f.min(dim=1, keepdim=True).values + 1e-6
    return ((f.pow(p) * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6)).pow(1 / p)


class DinoExtras:
    """high-resolution and silhouette DINOv2 features (shares the frozen ViT-L/14 of the multimodal experiments)."""

    def __init__(self, deep=None):
        from .image_feats import Deep
        self.deep = deep or Deep("dinov2")
        self.torch = self.deep.torch

    def hr(self, arrays: list[np.ndarray], size=448) -> np.ndarray:
        torch, d = self.torch, self.deep
        old = d.size
        d.size = size
        try:
            ims, mks = zip(*(d.prep(a) for a in arrays))
        finally:
            d.size = old
        x = (torch.tensor(np.stack(ims), device=d.dev) - d.mean) / d.std
        mk = torch.tensor(np.stack(mks), device=d.dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=d.dev == "cuda"):
            f = d.m.forward_features(x)["x_norm_patchtokens"].float()
        g = int(f.shape[1] ** 0.5)
        w = torch.nn.functional.adaptive_avg_pool2d(mk[:, None], g).flatten(1)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        mean = (f * w[..., None]).sum(1)
        gem = _gem(f, w)
        v = torch.cat([torch.nn.functional.normalize(mean, dim=-1), torch.nn.functional.normalize(gem, dim=-1)], 1)
        return v.cpu().numpy().astype(np.float32)

    def silhouette(self, masks: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        torch, d = self.torch, self.deep
        ims = []
        for m in masks:
            ys, xs = np.nonzero(m)
            m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else m
            S = int(max(m.shape) * 1.1) + 2
            can = np.full((S, S), 255, np.uint8)
            y0, x0 = (S - m.shape[0]) // 2, (S - m.shape[1]) // 2
            can[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m] = 0
            im = np.asarray(Image.fromarray(can).resize((224, 224), Image.BILINEAR), np.float32) / 255.0
            ims.append(np.repeat(im[None], 3, 0))
        x = (torch.tensor(np.stack(ims), device=d.dev) - d.mean) / d.std
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=d.dev == "cuda"):
            cls = d.m.forward_features(x)["x_norm_clstoken"].float()
        return torch.nn.functional.normalize(cls, dim=-1).cpu().numpy().astype(np.float32)
