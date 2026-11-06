"""Diagnostic images for the upgraded pipeline (never used as model input).

* four_panel():   original + boxes | mask overlay | transparent cut-out | white background, with a
                  bottom white strip holding the filename and the diagnostic summary (legacy layout).
* before_after(): original | legacy mask | new instances, for review / evaluation sets.
* detail_card():  per-object recipe + appearance card (layout of the legacy detailed_preview).
"""
from __future__ import annotations

import textwrap

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from scipy import ndimage as ndi

COLORS = [(0, 230, 0), (255, 60, 60), (60, 160, 255), (255, 210, 0), (255, 0, 255), (0, 230, 230),
          (255, 140, 0), (150, 255, 150), (180, 120, 255), (255, 255, 255)]
_FONT_CACHE = {}


def font(size=18, bold=False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        f = None
        for name in (("arialbd.ttf" if bold else "arial.ttf"), ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
            try:
                f = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        _FONT_CACHE[key] = f or ImageFont.load_default()
    return _FONT_CACHE[key]


def _fit(img: Image.Image, h: int) -> Image.Image:
    return img.resize((max(1, round(img.width * h / img.height)), h), Image.LANCZOS)


def _checker(h, w, s=16):
    yy, xx = np.mgrid[:h, :w]
    c = (((yy // s) + (xx // s)) % 2).astype(np.uint8)
    return np.stack([200 + 40 * c] * 3, -1).astype(np.uint8)


def _composite(rgb, mask, bg="white"):
    m = mask[..., None].astype(np.float32)
    back = np.full_like(rgb, 255) if bg == "white" else _checker(*rgb.shape[:2])
    return (rgb * m + back * (1 - m)).astype(np.uint8)


def _outline(mask, t):
    return mask & ~ndi.binary_erosion(mask, iterations=max(1, t))


def overlay(rgb, masks, darken=0.30):
    """Darken outside the union; instance outlines in distinct colours."""
    union = np.zeros(rgb.shape[:2], bool)
    for m in masks:
        union |= m
    ov = rgb.copy()
    ov[~union] = (ov[~union] * darken).astype(np.uint8)
    t = max(1, rgb.shape[1] // 400)
    for k, m in enumerate(masks):
        ov[_outline(m, t)] = COLORS[k % len(COLORS)]
    return ov


def _wrap(draw, text, fnt, max_w):
    lines = []
    for para in str(text).split("\n"):
        words, line = para.split(), ""
        for w in words:
            cand = (line + " " + w).strip()
            if line and draw.textbbox((0, 0), cand, font=fnt)[2] > max_w:
                lines.append(line)
                line = w
            else:
                line = cand
        lines.append(line)
    return lines


def _strip(width, filename, info_lines, title_size=22, info_size=15):
    """Bottom white strip: filename (bold, centred) + wrapped diagnostic lines."""
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    fi = font(info_size)
    wrapped = []
    for ln in info_lines:
        wrapped += _wrap(tmp, ln, fi, width - 24)
    h = 14 + title_size + 10 + len(wrapped) * (info_size + 5) + 10
    s = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(s)
    d.text((width // 2, 8 + title_size // 2), filename, font=font(title_size, True), anchor="mm", fill="black")
    y = 14 + title_size + 6
    for ln in wrapped:
        d.text((12, y), ln, font=fi, fill=(40, 40, 40))
        y += info_size + 5
    return s


def _labelled_panels(panels, labels, h):
    ims = [_fit(p if isinstance(p, Image.Image) else Image.fromarray(p), h) for p in panels]
    W = sum(i.width for i in ims) + 10 * (len(ims) - 1)
    sheet = Image.new("RGB", (W, h + 28), "white")
    d = ImageDraw.Draw(sheet)
    x = 0
    for im, lab in zip(ims, labels):
        sheet.paste(im, (x, 28))
        d.text((x + 4, 5), lab, font=font(15, True), fill=(30, 30, 30))
        x += im.width + 10
    return sheet


def four_panel(rgb, inst_masks, inst_boxes, filename, info_lines, h=360):
    union = np.zeros(rgb.shape[:2], bool)
    for m in inst_masks:
        union |= m
    boxed = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(boxed)
    lw = max(2, rgb.shape[1] // 300)
    fs = max(14, rgb.shape[1] // 45)
    for k, b in enumerate(inst_boxes):
        col = COLORS[k % len(COLORS)]
        d.rectangle(b, outline=col, width=lw)
        d.text((b[0] + lw + 2, b[1] + lw), str(k + 1), font=font(fs, True), fill=col)
    panels = [boxed, overlay(rgb, inst_masks), _composite(rgb, union, "checker"), _composite(rgb, union, "white")]
    labels = ["1. Original + object boxes", "2. Mask overlay", "3. Transparent background", "4. White background"]
    sheet = _labelled_panels(panels, labels, h)
    strip = _strip(sheet.width, filename, info_lines)
    out = Image.new("RGB", (sheet.width, sheet.height + strip.height), "white")
    out.paste(sheet, (0, 0))
    out.paste(strip, (0, sheet.height))
    return out


def before_after(rgb, legacy_mask, inst_masks, filename, info_lines, h=360):
    panels = [rgb, overlay(rgb, [legacy_mask]) if legacy_mask is not None else np.zeros_like(rgb),
              overlay(rgb, inst_masks)]
    labels = ["Original", "BEFORE: legacy full_segmentation", "AFTER: upgraded (v3) objects"]
    sheet = _labelled_panels(panels, labels, h)
    strip = _strip(sheet.width, filename, info_lines)
    out = Image.new("RGB", (sheet.width, sheet.height + strip.height), "white")
    out.paste(sheet, (0, 0))
    out.paste(strip, (0, sheet.height))
    return out


def detail_card(obj_img: Image.Image, name: str, entries, colors, review_lines, width=None,
                colors_title="TOP 3 CERAMIC COLORS (mask pixels, CIELAB k-means)"):
    """entries: list of (label, value); colors: palette dicts (hex/name/fraction); review_lines: list[str]."""
    obj_img = obj_img.convert("RGB")
    w = width or max(820, min(1100, obj_img.width + 80))
    pic_h = min(620, int(obj_img.height * min(1.0, (w - 40) / obj_img.width)))
    pic_w = max(1, int(obj_img.width * pic_h / max(1, obj_img.height)))
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    fv, fl = font(15), font(16, True)
    label_w = 230
    rows = [(lab, _wrap(tmp, val if str(val).strip() else "-", fv, w - label_w - 30)) for lab, val in entries]
    rlines = []
    for r in review_lines:
        rlines += _wrap(tmp, r, font(14), w - 40)
    h = (53 + pic_h + 14 + 38 + 12 + sum(max(30, 20 * len(ls) + 10) for _, ls in rows) + 8
         + 36 + 12 + 43 * max(1, len(colors[:3])) + 36 + 12 + 20 * max(1, len(rlines)) + 24)
    out = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, w, 43), fill="#17435f")
    d.text((20, 11), f"CERAMIC ATTRIBUTE PREVIEW  |  {name}", font=font(19, True), fill="white")
    out.paste(obj_img.resize((pic_w, pic_h)), ((w - pic_w) // 2, 53))
    y = 53 + pic_h + 14
    d.rectangle((0, y, w, y + 38), fill="#17435f")
    d.text((18, y + 8), "RECIPE AND APPEARANCE", font=font(19, True), fill="white")
    y += 50
    for i, (lab, lines) in enumerate(rows):
        rh = max(30, 20 * len(lines) + 10)
        if i % 2 == 0:
            d.rectangle((0, y - 4, w, y - 4 + rh), fill="#f1f5f8")
        d.text((18, y), lab, font=fl, fill="#25485d")
        for k, ln in enumerate(lines):
            d.text((label_w, y + 20 * k), ln, font=fv, fill="#273640")
        y += rh
    y += 8
    d.rectangle((0, y, w, y + 36), fill="#17435f")
    d.text((18, y + 7), colors_title, font=font(18, True), fill="white")
    y += 48
    if not colors:
        d.text((22, y + 4), "No usable colour estimate", font=font(16), fill="#273640")
        y += 43
    for c in colors[:3]:
        hexc = c.get("hex", "#888888")
        try:
            ImageColor.getrgb(hexc)
        except ValueError:
            hexc = "#888888"
        d.rectangle((22, y, 82, y + 29), fill=hexc, outline="#555555")
        pct = float(c.get("fraction", 0) or 0) * 100
        unc = "  (name uncertain)" if c.get("name_uncertain") else ""
        d.text((100, y + 4), f"{c.get('name', 'Unknown')}  |  {hexc}  |  {pct:.1f}%{unc}", font=font(17), fill="#273640")
        y += 43
    d.rectangle((0, y, w, y + 36), fill="#7a4a12")
    d.text((18, y + 7), "REVIEW STATUS (automatic predictions, not human-verified)", font=font(18, True), fill="white")
    y += 48
    for ln in rlines or ["No review flags"]:
        d.text((20, y), ln, font=font(14), fill="#5a3410")
        y += 20
    return out.crop((0, 0, w, min(h, y + 20)))


def label_strip(image: Image.Image, text: str) -> Image.Image:
    """Image + white strip with the name (display copies only)."""
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 48), "white")
    canvas.paste(image, (0, 0))
    ImageDraw.Draw(canvas).text((image.width // 2, image.height + 24), text, font=font(22, True),
                                anchor="mm", fill="black")
    return canvas


def textwrap_join(items, width=110):
    return textwrap.wrap("; ".join(items), width=width) if items else []
