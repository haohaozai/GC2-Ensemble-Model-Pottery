"""Adapter for the project's original Grounding DINO + SAM 2.1 v3 segmenter."""
from __future__ import annotations
import numpy as np
from PIL import Image


def load_segmenter():
    from glazeprep.segmentation.ceramic_v3 import CeramicSegmenterV3
    return CeramicSegmenterV3()


def extract_objects(photo: Image.Image, segmenter):
    rgb = np.asarray(photo.convert("RGB"))
    result = segmenter(rgb)
    objects = []
    for index, instance in enumerate(result.instances, 1):
        mask = np.asarray(instance.mask, dtype=bool)
        if mask.shape != rgb.shape[:2] or not mask.any():
            continue
        ys, xs = np.where(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        rgba = np.dstack((rgb[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)))
        objects.append({"index": index, "image": Image.fromarray(rgba, "RGBA"),
                        "area": float(mask.mean()), "primary": bool(instance.primary),
                        "flags": list(instance.flags), "box": [int(x0), int(y0), int(x1), int(y1)]})
    return objects, result.meta()
