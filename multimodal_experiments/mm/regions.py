"""Multi-region colour analysis of one segmented ceramic object (regions_v1).

Why: a dominant-colour palette ranks colours by pixel count, so a large beige area hides a smaller reddish band
that may be what makes the glaze recognisable (2152_1.jpg: sand 60 % / chestnut brown 37 %). Here every object is
split into spatially coherent colour REGIONS, and area, distinctiveness, contrast and position are kept as separate
features - colour area is never used as colour importance.

Algorithm (all on the object cut-out, transparent background excluded):
  1. resize to <= 320 px, CIELAB (D65);
  2. SLIC superpixels inside the mask (compactness 10, ~220 segments);
  3. region-adjacency-graph merge of neighbouring superpixels while their mean Lab colours are closer than
     MERGE_DE (Euclidean Lab = CIE76; spatially coherent regions, a colour can form several regions);
  4. regions below MIN_REGION of the object are absorbed into their most similar neighbour (noise, JPEG blocks);
  5. per region: median Lab colour -> HEX + reference shade name (CIEDE2000, glazeprep/appearance.py, the same
     names as the review CSV), area fraction, centroid / position, share on the object edge, contrast (area-weighted
     CIEDE2000 to the other regions), distinctiveness (CIEDE2000 to the mean colour of the rest of the object),
     lightness texture, flags for possible specular highlight / possible shadow (NOT removed - flagged);
  6. small high-contrast spots (speckles) measured separately (local lightness contrast, connected components);
  7. boundaries between regions: shared length and sharpness (mean lightness gradient on the boundary).

auto_importance = 0.45 * min(1, distinctiveness / 25) + 0.35 * sqrt(min(1, area / 0.25)) + 0.20 * min(1, contrast / 30),
x 0.3 for regions flagged as highlight / shadow. It is a transparent heuristic used only when no human importance flag
exists; the reviewed importance (review_tile) always replaces it. Material (glaze / slip / clay body / ...) is NOT
inferred from colour: it stays "unknown" until a reviewer sets it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

VERSION = "regions_v1"
MAX_SIDE = 320
N_SEGMENTS = 220
MERGE_DE = 11.0          # CIE76 distance below which neighbouring superpixels merge
MIN_REGION = 0.012       # regions smaller than 1.2 % of the object are absorbed (noise)
MEANINGFUL = 0.04        # a region counts as a colour region of the object from 4 % area
SUGGEST_DE = 25.0        # review_tile suggestion: secondary region distinct by dE2000 >= 25 ...
SUGGEST_AREA = (0.05, 0.45)   # ... covering 5-45 % of the object ...
SUGGEST_MAX_REGIONS = 3  # ... on an object with 2-3 meaningful regions (no shadow / highlight region >= 5 %)
PARAMS = {"version": VERSION, "max_side": MAX_SIDE, "n_segments": N_SEGMENTS, "merge_de76": MERGE_DE,
          "min_region": MIN_REGION, "meaningful_region": MEANINGFUL,
          "suggest_review_tile": "2-3 meaningful regions, no shadow/highlight >=5 %, secondary region 5-45 % area with dE00>=25",
          "review_priority": "min(1,dE00/40)*(1-area) of the most distinctive secondary region",
          "importance": "0.45*min(1,dist/25)+0.35*sqrt(min(1,area/0.25))+0.20*min(1,contrast/30); x0.3 if highlight/shadow"}


def _lab2hex(lab):
    from skimage.color import lab2rgb
    rgb = np.clip(lab2rgb(np.asarray(lab, np.float64).reshape(1, 1, 3))[0, 0], 0, 1)
    return "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgb), rgb


def _de00(a, b):
    from skimage.color import deltaE_ciede2000
    return float(deltaE_ciede2000(np.asarray(a, np.float64).reshape(1, 3), np.asarray(b, np.float64).reshape(1, 3))[0])


def _position(cx, cy, edge):
    v = "top" if cy < 0.36 else "bottom" if cy > 0.64 else "middle"
    h = "left" if cx < 0.36 else "right" if cx > 0.64 else "center"
    pos = "center" if (v, h) == ("middle", "center") else (h if v == "middle" else (v if h == "center" else f"{v}-{h}"))
    return pos + (" (edge)" if edge > 0.6 else "")


def analyse(rgba: np.ndarray, name_fn=None) -> tuple[dict, np.ndarray]:
    """rgba (H, W, 4) uint8 cut-out -> (result dict, region label map uint8 at the analysis size)."""
    from PIL import Image
    from scipy import ndimage as ndi
    from skimage import graph
    from skimage.color import rgb2lab
    from skimage.segmentation import slic
    if name_fn is None:
        from glazeprep.appearance import nearest_color as name_fn
    im = Image.fromarray(rgba)
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    a = np.asarray(im)
    mask = a[..., 3] >= 125
    H, W = mask.shape
    n_obj = int(mask.sum())
    if n_obj < 30:
        return {"version": VERSION, "error": "object too small", "regions": []}, np.zeros((H, W), np.uint8)
    rgb = a[..., :3].astype(np.float64) / 255.0
    lab = rgb2lab(rgb)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    # ---- superpixels + RAG merge
    sp = slic(rgb, n_segments=N_SEGMENTS, compactness=10, mask=mask, start_label=1, channel_axis=-1)
    rag = graph.rag_mean_color(lab, sp, mode="distance")

    def _w(g, src, dst, n):
        diff = g.nodes[dst]["mean color"] - g.nodes[n]["mean color"]
        return {"weight": float(np.linalg.norm(diff))}

    def _merge(g, src, dst):
        g.nodes[dst]["total color"] += g.nodes[src]["total color"]
        g.nodes[dst]["pixel count"] += g.nodes[src]["pixel count"]
        g.nodes[dst]["mean color"] = g.nodes[dst]["total color"] / g.nodes[dst]["pixel count"]

    lbl = graph.merge_hierarchical(sp, rag, thresh=MERGE_DE, rag_copy=False, in_place_merge=True,
                                   merge_func=_merge, weight_func=_w)
    lbl = np.where(mask, lbl + 1, 0)
    # ---- absorb tiny regions into the most similar neighbour
    for _ in range(4):
        ids, counts = np.unique(lbl[mask], return_counts=True)
        small = [i for i, c in zip(ids, counts) if c < MIN_REGION * n_obj]
        if not small or len(ids) == 1:
            break
        means = {i: np.median(lab[lbl == i], 0) for i in ids}
        for i in small:
            ring = ndi.binary_dilation(lbl == i, iterations=1) & (lbl != i) & mask
            nb = np.unique(lbl[ring])
            nb = [j for j in nb if j != 0 and j != i]
            if not nb:
                continue
            j = min(nb, key=lambda j: np.linalg.norm(means[i] - means[j]))
            lbl[lbl == i] = j
    # relabel 1..k by area
    ids, counts = np.unique(lbl[mask], return_counts=True)
    order = ids[np.argsort(-counts)]
    remap = np.zeros(lbl.max() + 1, np.int32)
    for k, i in enumerate(order, 1):
        remap[i] = k
    lbl = remap[lbl].astype(np.uint8 if len(order) < 255 else np.uint16)
    # ---- geometry helpers
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    dist = ndi.distance_transform_edt(mask)
    edge_band = mask & (dist <= max(2.0, 0.08 * min(bw, bh)))
    gy, gx = np.gradient(L)
    grad = np.hypot(gx, gy)
    # ---- regions
    regions = []
    med = {}
    for k in range(1, len(order) + 1):
        m = lbl == k
        n = int(m.sum())
        med[k] = np.median(lab[m], 0)
    tot_lab = np.median(lab[mask], 0)
    for k in range(1, len(order) + 1):
        m = lbl == k
        n = int(m.sum())
        area = n / n_obj
        ry, rx = np.nonzero(m)
        cx, cy = float((rx.mean() - x0) / bw), float((ry.mean() - y0) / bh)
        edge = float((m & edge_band).sum() / n)
        hexc, rgbf = _lab2hex(med[k])
        nm = name_fn(tuple(int(round(v * 255)) for v in rgbf))
        others = [(j, (lbl == j).sum()) for j in range(1, len(order) + 1) if j != k]
        wsum = sum(c for _, c in others)
        contrast = sum(_de00(med[k], med[j]) * c for j, c in others) / wsum if wsum else 0.0
        rest = mask & ~m
        distinct = _de00(med[k], np.median(lab[rest], 0)) if rest.sum() > 20 else 0.0
        hl = float(((L[m] > 90) & (C[m] < 8)).mean())
        flags = []
        if hl > 0.5:
            flags.append("possible_highlight")
        # possible shadow: much darker than a neighbour of similar hue, on the object edge
        ring = ndi.binary_dilation(m, iterations=2) & ~m & mask
        for j in np.unique(lbl[ring]):
            if j in (0, k):
                continue
            dl = med[j][0] - med[k][0]
            dhue = math.hypot(med[j][1] - med[k][1], med[j][2] - med[k][2])
            if dl > 22 and dhue < 12 and edge > 0.35:
                flags.append("possible_shadow")
                break
        if n < 60:
            flags.append("few_pixels")
        imp = 0.45 * min(1.0, distinct / 25) + 0.35 * math.sqrt(min(1.0, area / 0.25)) + 0.20 * min(1.0, contrast / 30)
        if "possible_highlight" in flags or "possible_shadow" in flags:
            imp *= 0.3
        ncomp = int(ndi.label(m)[1])
        regions.append({"id": k, "hex": hexc, "lab": [round(float(v), 2) for v in med[k]], "name": nm["name"],
                        "family": nm["family"], "name_uncertain": bool(nm["name_uncertain"]),
                        "area_frac": round(area, 4), "centroid": [round(cx, 3), round(cy, 3)],
                        "position": _position(cx, cy, edge), "edge_share": round(edge, 3),
                        "bbox": [int(rx.min()), int(ry.min()), int(rx.max()) + 1, int(ry.max()) + 1],
                        "components": ncomp, "contrast_de00": round(contrast, 2),
                        "distinctiveness_de00": round(distinct, 2), "lightness_std": round(float(lab[m][:, 0].std()), 2),
                        "auto_importance": round(imp, 3), "flags": flags, "material": "unknown",
                        "meaningful": area >= MEANINGFUL})
    # ---- spots / speckles (small local contrast blobs, measured separately from regions)
    Lm = np.where(mask, L, np.median(L[mask]))
    loc = ndi.median_filter(Lm, size=9)
    spot = mask & (np.abs(L - loc) > 14) & (dist > 2)
    sl, ns = ndi.label(spot)
    sizes = ndi.sum(spot, sl, range(1, ns + 1)) if ns else np.array([])
    keep = np.isin(sl, 1 + np.nonzero((sizes >= 2) & (sizes <= 0.004 * n_obj + 6))[0]) if ns else spot
    spots = {"count": int(((sizes >= 2) & (sizes <= 0.004 * n_obj + 6)).sum()) if ns else 0,
             "density": round(float(keep.sum() / n_obj), 4)}
    if keep.sum() >= 5:
        sl_lab = np.median(lab[keep], 0)
        hx, rgbf = _lab2hex(sl_lab)
        spots.update({"hex": hx, "name": name_fn(tuple(int(round(v * 255)) for v in rgbf))["name"],
                      "contrast_de00": round(_de00(sl_lab, tot_lab), 2),
                      "darker": bool(sl_lab[0] < tot_lab[0])})
    # ---- boundaries
    bounds = []
    K = len(order)
    if K > 1:
        right = (lbl[:, :-1] != lbl[:, 1:]) & (lbl[:, :-1] > 0) & (lbl[:, 1:] > 0)
        down = (lbl[:-1, :] != lbl[1:, :]) & (lbl[:-1, :] > 0) & (lbl[1:, :] > 0)
        pairs = {}
        for (ya, xa) in zip(*np.nonzero(right)):
            p = tuple(sorted((int(lbl[ya, xa]), int(lbl[ya, xa + 1]))))
            pairs.setdefault(p, []).append(grad[ya, xa])
        for (ya, xa) in zip(*np.nonzero(down)):
            p = tuple(sorted((int(lbl[ya, xa]), int(lbl[ya + 1, xa]))))
            pairs.setdefault(p, []).append(grad[ya, xa])
        per = 2 * (bw + bh)
        for (i, j), g in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
            sharp = float(np.mean(g))
            bounds.append({"a": i, "b": j, "length_frac": round(len(g) / per, 3), "sharpness": round(sharp, 2),
                           "type": "sharp" if sharp > 4.0 else "gradual", "de00": round(_de00(med[i], med[j]), 2)})
    res = {"version": VERSION, "size": [W, H], "object_px": n_obj, "object_mean_hex": _lab2hex(tot_lab)[0],
           "regions": regions, "spots": spots, "boundaries": bounds}
    res["summary"] = summarize(res)
    return res, lbl


def summarize(res: dict) -> dict:
    """region summary + review_tile SUGGESTION (a queue for a human, not an error flag).

    Suggested: a clear object of 2-3 meaningful colour regions, no shadow / highlight region >= 5 %, and a secondary
    region (5-45 % of the object) that differs from the rest by dE2000 >= 25 - the case where a dominant-colour
    palette hides a distinctive part (e.g. 2152_1: sand 62 % + chestnut brown 36 %). priority = min(1, dE/40) x
    (1 - area) of that region orders the queue. Objects with many regions are multi-colour, not suspicious."""
    regions = res.get("regions") or []
    mean_reg = [r for r in regions if r["meaningful"]]
    lighting = [r for r in regions if ({"possible_shadow", "possible_highlight"} & set(r["flags"])) and r["area_frac"] >= 0.05]
    sec = [r for r in mean_reg[1:] if not ({"possible_shadow", "possible_highlight"} & set(r["flags"]))
           and SUGGEST_AREA[0] <= r["area_frac"] <= SUGGEST_AREA[1]]
    best = max(sec, key=lambda r: r["distinctiveness_de00"], default=None)
    priority = round(min(1.0, best["distinctiveness_de00"] / 40) * (1 - best["area_frac"]), 3) if best else 0.0
    multi = any(_de00(a["lab"], b["lab"]) >= 15 for i, a in enumerate(mean_reg) for b in mean_reg[i + 1:])
    reasons = []
    if best and best["distinctiveness_de00"] >= SUGGEST_DE:
        reasons.append(f"secondary distinctive region {best['id']} ({best['name']}, {100 * best['area_frac']:.0f} % area, "
                       f"dE00 {best['distinctiveness_de00']:.0f} to the rest): a dominant-colour palette may hide it")
    if lighting:
        reasons.append("possible shadow / highlight region >= 5 % (lighting, not glaze?) - not suggested")
    suggest = bool(best and best["distinctiveness_de00"] >= SUGGEST_DE and 2 <= len(mean_reg) <= SUGGEST_MAX_REGIONS and not lighting)
    return {"n_regions": len(regions), "n_meaningful": len(mean_reg), "multi_colour": multi,
            "max_distinctiveness": max([r["distinctiveness_de00"] for r in mean_reg] or [0]),
            "review_priority": priority, "suggest_review_tile": suggest, "reasons": reasons}


def save(res: dict, lbl: np.ndarray, out_dir: Path, oid: str):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{oid}.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    Image.fromarray(lbl.astype(np.uint8)).save(out_dir / f"{oid}_labels.png", optimize=False, compress_level=6)
