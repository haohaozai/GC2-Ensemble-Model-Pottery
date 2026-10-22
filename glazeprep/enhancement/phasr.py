"""PhaSR (CVPR 2026) - shadow removal / ambient-lighting normalisation.

PhaSR needs three inputs per image (reproduced exactly from the official test.py):
    1. RGB image in [0, 1]
    2. DINOv2 ViT-L/14 intermediate features (semantic prior)
    3. point map + normal map derived from Depth-Anything-V2 ViT-L (geometric prior)

Checkpoint: ISTD+ (window 16), chosen over Ambient6K and WSRD by
scripts/03a_choose_phasr_checkpoint.py - on pottery photos Ambient6K blurred the
glaze texture and painted colour blotches, WSRD barely corrected shading.

Memory: Depth-Anything-V2-L + DINOv2-L + PhaSR do not fit in 8 GB together at
768 px (Windows then silently spills to system RAM and slows ~20x). So depth is a
separate model (`DepthGeometry`) run in its own pass and released before PhaSR.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C

_PHASR_DIR = C.THIRD_PARTY / "PhaSR-main"
_DAV2_DIR = C.THIRD_PARTY / "Depth-Anything-V2-main"
for _p in (_PHASR_DIR, _DAV2_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DPS = 14  # DINOv2 patch size; PhaSR upsamples the input by 14/8 before DINOv2


def _normal_from_depth(depth_u8: np.ndarray, ksize: int = 5) -> np.ndarray:
    """calculate_depth_normal.py::calculate_normal_map, returns HxWx3 uint8."""
    depth = depth_u8.astype(np.float32)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=ksize)
    dz = np.ones_like(dx) * -1
    n = np.stack((dx, dy, dz), axis=-1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)
    return ((n + 1) / 2 * 255).astype("uint8")


def geometry_from_depth(depth_u8: np.ndarray):
    """uint8 depth -> (point map, normal map) exactly as PhaSR test.py::load_geo."""
    from utils import depthToPoint, process_normal  # PhaSR utils
    normal = process_normal(_normal_from_depth(depth_u8))
    point = depthToPoint(60, depth_u8)
    point = point / (2 * point[:, :, 2].mean() + 1e-8)
    return point, normal


class DepthGeometry:
    """Depth-Anything-V2 ViT-L -> min-max normalised uint8 depth (PhaSR's preprocessing).
    Geometry does not change with illumination, so it is computed once per image on the
    raw crop and re-used for the Retinex-corrected input."""

    def __init__(self, device: str = C.DEVICE):
        from depth_anything_v2.dpt import DepthAnythingV2
        self.model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        self.model.load_state_dict(torch.load(C.DEPTH_ANYTHING_CKPT, map_location="cpu"))
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        depth = self.model.infer_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0).astype(np.uint8)


class PhaSRRunner:
    def __init__(self, device: str = C.DEVICE, win_size: int = C.PHASR_WIN_SIZE, ckpt=C.PHASR_CKPT):
        from utils import load_checkpoint  # PhaSR utils
        from model import PhaSR

        self.device, self.win = device, win_size
        hub_dir = C.MODEL_CACHE / "torch" / "hub" / "facebookresearch_dinov2_main"
        self.dino = torch.hub.load(str(hub_dir), "dinov2_vitl14", source="local").to(device).eval()
        self.model = PhaSR(img_size=256, embed_dim=32, win_size=win_size,
                           token_projection="linear", token_mlp="leff")
        load_checkpoint(self.model, str(ckpt))
        self.model = self.model.to(device).eval()

    def _dino_feats(self, x):
        xu = F.interpolate(x, size=(int(x.shape[2] * DPS / 8), int(x.shape[3] * DPS / 8)),
                           mode="bilinear", align_corners=False)
        return self.dino.get_intermediate_layers(xu, 4, True)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, depth_u8: np.ndarray) -> np.ndarray:
        """rgb uint8 HxWx3 + its uint8 depth -> shadow/ambient-normalised uint8 HxWx3."""
        point, normal = geometry_from_depth(depth_u8)
        dev = self.device
        it = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        pt = torch.from_numpy(point).permute(2, 0, 1)[None].float().to(dev)
        nm = torch.from_numpy(normal).permute(2, 0, 1)[None].float().to(dev)

        m = 8 * self.win
        h, w = it.shape[2:]
        H, W = ((h + m) // m) * m, ((w + m) // m) * m
        xi = F.pad(it, (0, W - w, 0, H - h), "reflect")
        pp = F.pad(pt, (0, W - w, 0, H - h), "reflect")
        nn_ = F.pad(nm, (0, W - w, 0, H - h), "reflect")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = self.model(xi, self._dino_feats(xi), pp, nn_)
        o = o.float().clamp(0, 1)[:, :, :h, :w]
        return (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
