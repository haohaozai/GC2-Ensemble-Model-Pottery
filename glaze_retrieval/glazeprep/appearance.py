"""Conservative appearance tagging from source RGB image and segmentation mask.

Predictions are NOT ground truth. Cone and atmosphere require recipe metadata.
Object masks are not necessarily glaze-only masks.
"""
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SHADES = {
    'red': {'crimson red':'#DC143C','scarlet red':'#FF2400','brick red':'#B55239','rust red':'#A34A28','burgundy red':'#800020','cherry red':'#C2183A','coral red':'#E66B5B','terracotta red':'#B65F43'},
    'orange': {'burnt orange':'#CC5500','tangerine orange':'#F28500','peach orange':'#FFCBA4','amber orange':'#D99032','apricot orange':'#FBCEB1','copper orange':'#B87333','rust orange':'#B7410E','pumpkin orange':'#D9772B'},
    'yellow': {'lemon yellow':'#FFF44F','golden yellow':'#FFD700','mustard yellow':'#C9A227','ochre yellow':'#CC9A32','honey yellow':'#E6B84C','straw yellow':'#E4D69A','pale yellow':'#F5E6A8','saffron yellow':'#F4C430'},
    'green': {'olive green':'#737D39','sage green':'#97A88A','celadon green':'#A0BE9E','mint green':'#91D9B5','forest green':'#285D3D','emerald green':'#289162','seafoam green':'#7EC4B2','teal green':'#2B7477','moss green':'#78835A','jade green':'#58A68B'},
    'blue': {'cobalt blue':'#2455A4','navy blue':'#1D3557','sky blue':'#87CEEB','powder blue':'#B0D0E6','slate blue':'#657C91','steel blue':'#4682B4','cerulean blue':'#2A91B5','turquoise blue':'#40BFC1','midnight blue':'#191970','denim blue':'#527BA3'},
    'purple': {'lavender purple':'#B8A2D1','violet purple':'#8F5AB5','plum purple':'#74436A','amethyst purple':'#9966CC','mauve purple':'#B784A7','eggplant purple':'#614051','lilac purple':'#C8A2C8','deep purple':'#55286F'},
    'pink': {'rose pink':'#D98A9A','blush pink':'#E8B4B8','salmon pink':'#FA8072','dusty pink':'#C58E96','coral pink':'#F2A19A','hot pink':'#FF69B4','pale pink':'#F5D5DA','magenta pink':'#C64B8C'},
    'brown': {'chocolate brown':'#5D4037','coffee brown':'#6F4E37','walnut brown':'#76533D','chestnut brown':'#954535','earth brown':'#8B6F47','umber brown':'#635147','caramel brown':'#B77B45','tan brown':'#C19A6B','dark brown':'#49352A'},
    'black': {'jet black':'#101010','charcoal black':'#30343B','soft black':'#252525','blue black':'#1B2633','brown black':'#30251F','graphite black':'#383B40'},
    'white': {'pure white':'#FFFFFF','snow white':'#FAFAF5','ivory white':'#F4F0DF','porcelain white':'#F2F2EE','warm white':'#F5F0E8','cool white':'#E9F0F2'},
    'gray': {'light gray':'#D3D3D3','medium gray':'#969696','dark gray':'#555555','slate gray':'#708090','warm gray':'#A39B91','cool gray':'#8C9CA6','ash gray':'#B2BEB5','blue gray':'#6D8490','green gray':'#89958B'},
    'cream': {'cream':'#FFFDD0','beige':'#DCC7A1','off white':'#F3EDE0','sand':'#CDB58B','oatmeal':'#D8C9AD','pale ivory':'#F1E5C8','warm cream':'#F2DEB5','buff':'#E0C59A'},
}


def _rgb(hex_code):
    s = hex_code.lstrip('#')
    return [int(s[i:i+2], 16) for i in (0, 2, 4)]


_REFERENCES = [(family, name, code) for family, shades in SHADES.items() for name, code in shades.items()]
_REF_RGB = np.array([_rgb(code) for _, _, code in _REFERENCES], dtype=np.float32) / 255.0
_REF_LAB = cv2.cvtColor(_REF_RGB.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0]


def _lab_from_rgb(rgb):
    arr = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0]


def nearest_color(rgb, max_delta_e=18.0):
    """Return closest reference shade, with CIEDE2000 uncertainty indicator.

    OpenCV float LAB uses L*=0..100, a*/b* signed, appropriate for CIEDE2000.
    """
    from skimage.color import deltaE_ciede2000
    target = _lab_from_rgb(rgb)
    distances = deltaE_ciede2000(_REF_LAB.reshape(1, -1, 3), target.reshape(1, 1, 3))[0]
    idx = int(np.argmin(distances))
    family, name, reference = _REFERENCES[idx]
    distance = float(distances[idx])
    return {'family': family, 'name': name, 'reference_hex': reference,
            'delta_e00': round(distance, 2), 'name_uncertain': bool(distance > max_delta_e)}


def connected_objects(mask, min_frac=0.008):
    """Connected components of an existing mask; cannot recover missing pixels."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    threshold = max(100, int(binary.sum() * min_frac))
    comps = [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= threshold]
    return sorted(comps, key=lambda x: int(x.sum()), reverse=True)


def palette(rgb, mask, k=10):
    """Cluster sampled object-mask pixels in LAB; fractions sum approximately to one."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    interior = cv2.erode(binary, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if interior.sum() < 100:
        interior = binary.astype(bool)
    pixels = rgb[interior]
    if len(pixels) < 30:
        return []
    hsv = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
    # Keep most valid pixels. Removing extreme highlights can distort white glazes,
    # so reject only near-clipped values and report the uncertainty elsewhere.
    valid = (hsv[:, 2] > 8) & (hsv[:, 2] < 253)
    if valid.sum() >= max(80, int(0.5 * len(pixels))):
        pixels = pixels[valid]
    if len(pixels) > 12000:
        idx = np.random.default_rng(42).choice(len(pixels), 12000, replace=False)
        pixels = pixels[idx]
    lab = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.float32) / 255.0,
                       cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    kk = max(1, min(int(k), len(lab)))
    cv2.setRNGSeed(42)
    _, labels, centers = cv2.kmeans(lab, kk, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.4),
        3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=kk)
    out = []
    for i in np.argsort(-counts):
        center = centers[i].reshape(1, 1, 3).astype(np.float32)
        color = np.clip(cv2.cvtColor(center, cv2.COLOR_LAB2RGB)[0, 0] * 255, 0, 255).round().astype(np.uint8)
        match = nearest_color(color)
        out.append({'name': match['name'], 'family': match['family'],
                    'hex': '#%02X%02X%02X' % tuple(map(int, color)),
                    'reference_hex': match['reference_hex'],
                    'delta_e00': match['delta_e00'], 'name_uncertain': match['name_uncertain'],
                    'fraction': round(float(counts[i] / len(labels)), 4)})
    return out


def _aggregate(objects, max_colors=10):
    """Aggregate palettes weighted by object pixel area, not equal per object."""
    family_weights = defaultdict(float)
    shade_weights = defaultdict(float)
    all_colors = []
    total = sum(o['pixel_area'] for o in objects)
    for obj in objects:
        weight = obj['pixel_area'] / total if total else 0
        for color in obj['color_candidates']:
            contribution = weight * color['fraction']
            family_weights[color['family']] += contribution
            shade_weights[(color['family'], color['name'])] += contribution
            all_colors.append({**color, 'image_fraction': round(contribution, 4),
                               'object_id': obj['object_id']})
    family_dist = [{'family': key, 'fraction': round(value, 4)} for key, value in
                   sorted(family_weights.items(), key=lambda kv: -kv[1])]
    shade_dist = [{'family': key[0], 'name': key[1], 'fraction': round(value, 4)}
                  for key, value in sorted(shade_weights.items(), key=lambda kv: -kv[1])]
    colors = sorted(all_colors, key=lambda c: -c['image_fraction'])[:max_colors]
    return family_dist, shade_dist, colors


def analyze(rgb, mask, metadata=None):
    metadata = metadata or {}
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask) > 0
    if mask.shape != rgb.shape[:2]:
        raise ValueError('mask and image dimensions must match')
    if not mask.any():
        return {'status': 'NO_OBJECT', 'objects': [], 'review_reasons': ['empty_segmentation_mask'],
                'dominant_family': None, 'family_distribution': [], 'shade_distribution': [],
                'group_palette': [], 'cone': metadata.get('cone'),
                'atmospheres': metadata.get('atmospheres')}
    components = connected_objects(mask)
    if not components:
        components = [mask]
    objects = []
    h, w = mask.shape
    for i, component in enumerate(components, 1):
        ys, xs = np.where(component)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        colors = palette(rgb, component)
        touches_border = x0 <= 1 or y0 <= 1 or x1 >= w-1 or y1 >= h-1
        objects.append({'object_id': i, 'bbox_xyxy': [x0, y0, x1, y1],
                        'pixel_area': int(component.sum()),
                        'area_fraction': round(float(component.mean()), 4),
                        'touches_image_border': touches_border,
                        'color_candidates': colors, 'glaze_region_verified': False,
                        'surface': None, 'transparency': None})
    family_dist, shade_dist, group_colors = _aggregate(objects)
    reasons = ['glaze_region_not_verified', 'surface_and_transparency_require_review']
    if len(objects) > 1:
        reasons.append('multiple_objects')
    if mask.mean() > 0.9:
        reasons.append('mask_may_include_background')
    if any(o['touches_image_border'] for o in objects):
        reasons.append('object_touches_image_border_check_crop')
    if any(c['name_uncertain'] for o in objects for c in o['color_candidates'][:3]):
        reasons.append('uncertain_color_name')
    reasons.extend(str(v) for v in metadata.get('flags', []))
    return {'status': 'REVIEW_REQUIRED', 'objects': objects,
            'dominant_family': family_dist[0]['family'] if family_dist else None,
            'family_distribution': family_dist, 'shade_distribution': shade_dist,
            'group_palette': group_colors, 'review_reasons': sorted(set(reasons)),
            'cone': metadata.get('cone'), 'atmospheres': metadata.get('atmospheres'),
            'note': 'Object-mask colors are predictions; may include unglazed clay or background.'}


def _font(size, bold=False):
    candidates = ['C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def preview(rgb, mask, analysis, filename):
    """Annotated original photograph followed by a readable, dynamic summary table.

    Bounding boxes are derived from the supplied mask. They are NOT corrected here.
    Unknown glaze, surface, and transparency attributes are never inferred as facts.
    """
    from PIL import ImageColor
    import textwrap

    photo = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGB')
    scale = min(1.0, 1100 / max(photo.size))
    if scale < 1:
        photo = photo.resize((max(1, round(photo.width * scale)),
                              max(1, round(photo.height * scale))), Image.Resampling.LANCZOS)
    width = max(photo.width, 780)
    left_offset = (width - photo.width) // 2
    # Render labels on the original photo (no generated or substituted imagery).
    photo_draw = ImageDraw.Draw(photo)
    objects = analysis.get('objects', [])
    for obj in objects:
        x0, y0, x1, y1 = [round(float(v) * scale) for v in obj['bbox_xyxy']]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(photo.width - 1, x1 - 1), min(photo.height - 1, y1 - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        photo_draw.rectangle((x0, y0, x1, y1), outline='#00CB51', width=4)
        label = 'Object ' + str(obj['object_id'])
        label_font = _font(17, True)
        bounds = photo_draw.textbbox((0, 0), label, font=label_font)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        label_y = y0 + 3 if y0 + th + 10 < photo.height else max(0, y0 - th - 8)
        photo_draw.rectangle((x0, label_y, min(photo.width-1, x0+tw+12), label_y+th+7), fill='#007D32')
        photo_draw.text((x0+6, label_y+2), label, fill='white', font=label_font)

    heading = _font(24, True)
    body = _font(18)
    bold = _font(18, True)
    small = _font(16)
    margin = 18
    label_width = 205
    line_height = 25
    max_chars = max(35, (width - label_width - margin * 3) // 10)

    def display(value):
        if value is None or value == '' or value == []:
            return 'Unknown / not verified'
        if isinstance(value, (list, tuple)):
            return ', '.join(map(str, value)) or 'Unknown / not verified'
        return str(value)

    family = analysis.get('dominant_family')
    shades = analysis.get('shade_distribution', [])
    palette_colors = analysis.get('group_palette', [])
    if shades:
        primary = shades[0]['name'] + ' (' + shades[0]['family'] + ')'
        secondary = ', '.join(x['name'] for x in shades[1:5]) or 'None prominent'
    else:
        primary = 'Unknown / no usable segmented color'
        secondary = 'Unknown'
    surfaces = sorted({str(o['surface']) for o in objects if o.get('surface')})
    transparencies = sorted({str(o['transparency']) for o in objects if o.get('transparency')})
    rows = [
        ('Objects detected', str(len(objects)) + ' (from existing mask)'),
        ('Primary color', primary + ' - candidate, not ground truth'),
        ('Secondary colors', secondary),
        ('Dominant family', display(family)),
        ('Surface', ', '.join(surfaces) if surfaces else 'Unknown - visual review required'),
        ('Transparency', ', '.join(transparencies) if transparencies else 'Unknown - visual review required'),
        ('Cone', display(analysis.get('cone'))),
        ('Atmospheres', display(analysis.get('atmospheres'))),
        ('Glaze region', 'NOT VERIFIED - may include exposed clay'),
    ]
    def wrap_value(v):
        return textwrap.wrap(str(v), width=max_chars, break_long_words=False,
                             break_on_hyphens=False) or ['']
    row_specs = [(label, wrap_value(value)) for label, value in rows]
    reasons = analysis.get('review_reasons', [])
    notes = '; '.join(str(x).replace('_', ' ') for x in reasons) or 'No review flags'
    notes_lines = wrap_value(notes)
    palette_rows = palette_colors[:10]
    object_rows = objects[:12]
    obj_specs = []
    for o in object_rows:
        cands = o.get('color_candidates', [])[:4]
        detail = ', '.join(f"{c.get('name','?')} {float(c.get('fraction',0))*100:.0f}%" for c in cands)
        obj_specs.append((f"Object {o['object_id']}", wrap_value(detail or 'No colors available')))
    row_heights = [max(34, len(lines)*line_height + 12) for _, lines in row_specs]
    notes_h = max(42, len(notes_lines)*line_height + 12)
    object_heights = [max(34, len(lines)*line_height + 10) for _, lines in obj_specs]
    panel_height = (64 + 43 + sum(row_heights) + 43 + 36*len(palette_rows)
                    + 43 + sum(object_heights) + 43 + notes_h + 34)
    out = Image.new('RGB', (width, photo.height + panel_height), 'white')
    out.paste(photo, (left_offset, 0))
    d = ImageDraw.Draw(out)
    y = photo.height

    def section(title):
        nonlocal y
        d.rectangle((0, y, width, y+43), fill='#184C79')
        d.text((margin, y+9), title, fill='white', font=heading)
        y += 43

    d.rectangle((0, y, width, y+64), fill='#EAF0F6')
    d.text((margin, y+8), filename + ' | ' + analysis.get('status', 'UNKNOWN'),
           fill='#17334D', font=heading)
    d.text((margin, y+37), 'GREEN = detected ceramic-object box; NOT a verified glaze-only region',
           fill='#80511E', font=small)
    y += 64
    section('APPEARANCE AND RECIPE ATTRIBUTES')
    for idx, ((label, lines), height) in enumerate(zip(row_specs, row_heights)):
        d.rectangle((0, y, width, y+height), fill='#F0F5F9' if idx % 2 == 0 else 'white')
        d.line((0, y+height-1, width, y+height-1), fill='#CCD6DF', width=1)
        d.line((label_width, y, label_width, y+height), fill='#CCD6DF', width=1)
        d.text((margin, y+7), label, fill='#22364B', font=bold)
        for j, line in enumerate(lines):
            d.text((label_width+margin, y+7+j*line_height), line, fill='#26394A', font=body)
        y += height
    section('COLOR PALETTE - SHARE OF SEGMENTED PIXELS')
    if not palette_rows:
        d.text((margin, y+8), 'No usable color palette', fill='#26394A', font=body)
        y += 36
    for c in palette_rows:
        hex_color = c.get('hex', '#CCCCCC')
        try:
            ImageColor.getrgb(hex_color)
        except ValueError:
            hex_color = '#CCCCCC'
        d.rectangle((margin, y+5, margin+43, y+30), fill=hex_color, outline='#65717C')
        label = (f"{c.get('name','Unknown')} | {c.get('family','?')} | {hex_color} | "
                 f"{float(c.get('image_fraction', c.get('fraction',0)))*100:.1f}%")
        if c.get('name_uncertain'):
            label += ' | uncertain shade'
        d.text((margin+56, y+7), label[:max_chars+20], fill='#253747', font=small)
        y += 36
    section('INDIVIDUAL OBJECTS')
    if not obj_specs:
        d.text((margin, y+8), 'No segmented objects detected', fill='#26394A', font=body)
        y += 34
    for idx, ((label, lines), height) in enumerate(zip(obj_specs, object_heights)):
        d.rectangle((0, y, width, y+height), fill='#F0F5F9' if idx % 2 == 0 else 'white')
        d.text((margin, y+6), label, fill='#17334D', font=bold)
        for j, line in enumerate(lines):
            d.text((label_width+margin, y+6+j*line_height), line, fill='#26394A', font=body)
        y += height
    section('REVIEW NOTES')
    d.rectangle((0, y, width, y+notes_h), fill='#FFF3E8')
    for j, line in enumerate(notes_lines):
        d.text((margin, y+8+j*line_height), line, fill='#88431D', font=small)
    y += notes_h
    d.text((margin, y+8), 'Predictions only. Cone/atmosphere require linked recipe metadata.',
           fill='#66717A', font=small)
    return out
