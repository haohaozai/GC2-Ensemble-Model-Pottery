"""UnReflectAnything (CVPR 2026) - RGB-only specular-highlight removal.

Glazes are glossy, so studio lights create white specular blobs that a CNN
would read as "white glaze". UnReflectAnything predicts a highlight map and
inpaints those regions in DINOv3 feature space.

We composite at full crop resolution: pixels outside the predicted (dilated)
highlight mask are kept *exactly* as input, so only true highlights change.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C


def _remap_state_dict(sd: dict, model_keys: set) -> dict:
    """Checkpoint was saved with DDP ('module.') and an older transformers DINOv3
    layout ('dinov3.dinov3.layer.*'); transformers>=5 nests it under '.model.'."""
    out = {}
    for k, v in sd.items():
        k = k.removeprefix("module.")
        if k not in model_keys and k.startswith("dinov3.dinov3."):
            alt = k.replace("dinov3.dinov3.", "dinov3.dinov3.model.", 1)
            if alt in model_keys:
                k = alt
        out[k] = v
    return out


class UnReflectRunner:
    def __init__(self, device: str = C.DEVICE, threshold: float = 0.3, dilation: int = 40):
        from unreflectanything._model_builder import create_model_from_config_minimal
        from unreflectanything.model_ import UnReflectModel

        ckpt = torch.load(C.UNREFLECT_WEIGHTS_DIR / "weights" / "full_model_weights.pt",
                          map_location="cpu", weights_only=False)
        cfg = UnReflectModel._resolve_config(None, None, ckpt, False)
        inner = create_model_from_config_minimal(cfg, torch.device("cpu"))
        sd = _remap_state_dict(UnReflectModel._extract_state_dict(ckpt), set(inner.state_dict().keys()))
        inner.load_state_dict(sd, strict=True)
        del ckpt, sd
        self.model = inner.to(device).eval()
        self.size = int(cfg.MODEL.RGB_ENCODER.IMAGE_SIZE)        # 896
        self.device = device
        self.threshold = threshold
        self.dilation = dilation

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rgb uint8 HxWx3 -> (diffuse uint8 HxWx3, highlight_mask float HxW in [0,1])."""
        H, W = rgb.shape[:2]
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        xs = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        # bf16 autocast: ~2x faster; vs fp32 the mean pixel difference is 0.02/255 and the
        # highlight-mask IoU 0.985 (checked on a pilot image)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            out = self.model({"rgb": xs, "inpaint_mask_threshold": self.threshold,
                              "inpaint_mask_dilation": self.dilation})
        diffuse = out["diffuse"].float().clamp(0, 1)
        m = out.get("highlight_mask")
        if m is None:   # fall back to the soft highlight map
            m = (out["highlight"].float() > self.threshold).float()
        m = F.avg_pool2d(m.float(), kernel_size=5, stride=1, padding=2)        # feather edges
        diffuse = F.interpolate(diffuse, size=(H, W), mode="bilinear", align_corners=False)
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
        y = m * diffuse + (1 - m) * x                                          # identity outside highlights
        y = (y[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        return y, m[0, 0].cpu().numpy()
