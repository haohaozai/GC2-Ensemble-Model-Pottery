"""Visual embeddings used to compare the 8 variants downstream.

1. ResNet-50 + Masked Spatial Pooling  (the CV pathway of the proposal, Stream Gamma)
   background pixels are set to the ImageNet mean colour AND excluded from pooling, so
   only glaze pixels contribute to the 2048-d vector.
2. Masked CIELAB colour histogram (8 L* x 12 a* x 12 b* bins, Hellinger-normalised)
   - a transparent, training-free colour descriptor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from skimage.color import rgb2lab

from .. import config as C

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _letterbox(rgb: np.ndarray, mask: np.ndarray, side: int):
    """Pad to square with the ImageNet mean colour (keeps aspect ratio), resize to side."""
    h, w = mask.shape
    s = max(h, w)
    canvas = np.empty((s, s, 3), np.float32)
    canvas[:] = _MEAN
    m = np.zeros((s, s), np.float32)
    y, x = (s - h) // 2, (s - w) // 2
    canvas[y:y + h, x:x + w] = rgb
    m[y:y + h, x:x + w] = mask
    t = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(side, side), mode="bilinear", antialias=True, align_corners=False)
    mt = F.interpolate(torch.from_numpy(m)[None, None], size=(side, side), mode="area")
    return t, mt


class ResNetMaskedPool:
    def __init__(self, device: str = C.DEVICE, side: int = 448):
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET2K_V2)
        self.body = torch.nn.Sequential(*list(net.children())[:-2]).to(device).eval()  # -> layer4 maps
        self.device, self.side = device, side
        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        if mask is None:                         # reference: raw photo, global average pooling
            mask = np.ones(rgb.shape[:2], bool)
        else:                                    # segmented: background -> mean colour
            x = np.where(mask[..., None], x, _MEAN)
        t, mt = _letterbox(x, mask.astype(np.float32), self.side)
        feats = self.body((t.to(self.device) - self.mean) / self.std)           # (1, 2048, 14, 14)
        mt = F.interpolate(mt.to(self.device), size=feats.shape[-2:], mode="area")
        v = (feats * mt).sum((2, 3)) / mt.sum((2, 3)).clamp_min(1e-6)             # masked spatial pooling
        return F.normalize(v, dim=1)[0].cpu().numpy()


def lab_histogram(rgb: np.ndarray, mask: np.ndarray | None, bins=(8, 12, 12)) -> np.ndarray:
    lab = rgb2lab(rgb[::2, ::2])
    m = np.ones(lab.shape[:2], bool) if mask is None else mask[::2, ::2]
    pix = lab[m]
    h, _ = np.histogramdd(pix, bins=bins, range=[(0, 100), (-64, 64), (-64, 64)])
    h = np.sqrt(h.ravel() / max(h.sum(), 1))                                     # Hellinger
    return (h / (np.linalg.norm(h) + 1e-12)).astype(np.float32)
