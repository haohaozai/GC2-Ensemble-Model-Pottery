"""Small visual helpers for QA sheets (segmentation previews, 8-variant grids)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit(arr: np.ndarray, h: int) -> Image.Image:
    im = Image.fromarray(arr)
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def checkerboard(h, w, s=16):
    yy, xx = np.mgrid[:h, :w]
    c = (((yy // s) + (xx // s)) % 2).astype(np.uint8)
    return np.stack([200 + 40 * c] * 3, -1).astype(np.uint8)


def composite(rgb: np.ndarray, mask: np.ndarray, bg: str = "white") -> np.ndarray:
    m = mask[..., None].astype(np.float32)
    back = np.full_like(rgb, 255) if bg == "white" else checkerboard(*rgb.shape[:2])
    return (rgb * m + back * (1 - m)).astype(np.uint8)


def seg_preview(rgb, mask, boxes, title="", h=360) -> Image.Image:
    """4 panels: original+boxes | overlay | cut-out on checkerboard | white background."""
    boxed = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(boxed)
    lw = max(2, rgb.shape[1] // 300)
    for b in boxes:
        d.rectangle(b, outline=(255, 40, 40), width=lw)
    overlay = rgb.copy()
    overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)
    edge = mask ^ np.pad(mask, 1, mode="edge")[2:, 1:-1]
    overlay[edge] = (0, 255, 0)
    panels = [_fit(np.asarray(boxed), h), _fit(overlay, h),
              _fit(composite(rgb, mask, "checker"), h), _fit(composite(rgb, mask, "white"), h)]
    W = sum(p.width for p in panels) + 10 * (len(panels) - 1)
    sheet = Image.new("RGB", (W, h + 30), (255, 255, 255))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 30))
        x += p.width + 10
    ImageDraw.Draw(sheet).text((6, 5), title, fill=(0, 0, 0), font=_font(18))
    return sheet


def variant_grid(images: dict[str, np.ndarray], mask: np.ndarray, title="", h=240) -> Image.Image:
    """2x4 grid of the 8 variants, each shown segmented on white."""
    tiles = []
    for name, rgb in images.items():
        t = _fit(composite(rgb, mask, "white"), h)
        canvas = Image.new("RGB", (t.width, h + 24), (255, 255, 255))
        canvas.paste(t, (0, 24))
        ImageDraw.Draw(canvas).text((4, 3), name, fill=(0, 0, 0), font=_font(16))
        tiles.append(canvas)
    tw = max(t.width for t in tiles)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 8), rows * (h + 32) + 30), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (tw + 8), 30 + (i // cols) * (h + 32)))
    ImageDraw.Draw(sheet).text((6, 5), title, fill=(0, 0, 0), font=_font(18))
    return sheet
