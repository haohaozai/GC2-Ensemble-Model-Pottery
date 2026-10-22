"""Source of configs/manual_corrections_v2.json (run this file to regenerate the JSON).

Round 2 of the manual corrections: one entry per photo with objects marked "Needs review" in
object_review (8).csv (2026-10-16). Recipes start from the CURRENT segmentation (object numbers as reviewed).
Objects are listed in their final order: unchanged objects keep their place, a combined object takes the place of
its first part, new objects come last. Part syntax and the round-2 fields (from / fixed / unresolved / drop /
verified) are described in scripts/19_manual_corrections.py. Coordinates are fractions of the photo width /
height, read off the inspection sheets' 0.1 grid (manual_corrections_sheets_v2/).
"""
import json
from pathlib import Path

R = {}


def keep(*ks, **kw):
    return {"parts": [{"keep": list(ks)}], **kw}


def box(b, pos=None, neg=None, pick="best"):
    p = {"box": b, "pick": pick}
    if pos:
        p["pos"] = pos
    if neg:
        p["neg"] = neg
    return p


def obj(*parts, **kw):
    return {"parts": list(parts), **kw}


def same(*ks):
    """Unchanged objects: status, note and history carried over; their pixels are never taken by new regions."""
    return [keep(k, exact=True, label="unchanged") for k in ks]


def ok(*ks, why="unchanged; the photo's missing objects were added"):
    """Objects flagged only because of a photo-level problem that round 2 fixed: kept as they are, completed_v2."""
    return [keep(k, fixed=True, exact=True, label=why) for k in ks]


def col(b, label):
    """A standing test tile in the box b (fractions), separated from its neighbours: SAM box + points, kept in b."""
    x0, y0, x1, y1 = b
    cx, cy, h = (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0
    return obj(box(b, pos=[[cx, y0 + 0.3 * h], [cx, y0 + 0.7 * h]], neg=[[max(0.0, x0 - 0.02), cy], [min(1.0, x1 + 0.02), cy],
                                                                          [cx, min(1.0, y1 + 0.05)]]),
               clip=b, fill_holes=True, fixed=True, label=label)


def merge(*ks, extra=(), close=15, label=None, **kw):
    return obj({"keep": list(ks)}, *extra, close=close, fixed=True,
               label=label or "combined " + " + ".join(f"_{k}" for k in ks), **kw)


def new(b, pos=None, neg=None, pick="best", **kw):
    return obj(box(b, pos, neg, pick), fixed=True, **kw)


def split(k, rect, label, **kw):
    """One piece of current object k (the part inside rect)."""
    return obj({"keep": [k]}, clip=rect, fixed=True, label=label, **kw)


def cell(r, label, **kw):
    """A tile filling the cell r of a regular grid: SAM box on the cell, kept inside it."""
    cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    return obj(box(r, pos=[[cx, cy]]), clip=r, fill_holes=True, fixed=True, label=label, **kw)


# --- 900: three front cups: purple (kept), green, pink on the left (round 1 had kept the purple one only)
R["900"] = {"objects": [keep(1, fixed=True, exact=True, label="purple cup (unchanged)"),
                        new([0.34, 0.43, 0.65, 0.87], pos=[[0.48, 0.62], [0.45, 0.78]], neg=[[0.75, 0.65], [0.25, 0.65], [0.6, 0.36]],
                            label="green cup (front middle)"),
                        new([0.05, 0.4, 0.36, 0.86], pos=[[0.2, 0.62], [0.2, 0.78]], neg=[[0.48, 0.62], [0.36, 0.35]],
                            label="pink cup (front left)")]}
# --- 2479: only the white test tile in the middle, not the page around it
R["2479"] = {"objects": [obj(box([0.17, 0.0, 0.81, 0.64], pos=[[0.5, 0.45], [0.45, 0.15], [0.3, 0.55]], neg=[[0.5, 0.8], [0.05, 0.4], [0.95, 0.4]]),
                             within=[1], fill_holes=True, fixed=True, label="middle tile only (page removed)")]}
# --- 8415: blue tile (with its white foot) without the wooden deck and the paper label
R["8415"] = {"objects": [obj(box([0.23, 0.0, 0.62, 0.79], pos=[[0.42, 0.4], [0.45, 0.65], [0.42, 0.74]], neg=[[0.1, 0.4], [0.85, 0.4], [0.5, 0.9]]),
                             within=[1], fill_holes=True, fixed=True, label="blue tile, background removed")]}
# --- 20093: _6 and _7 are the glazed part of one tile
R["20093"] = {"objects": same(1, 2, 3, 4, 5) + [merge(6, 7)]}
# --- 22420: the whole left tile (glazed part _2, its two stripes incl. _3 and the unglazed top)
R["22420"] = {"objects": same(1) + [obj({"keep": [2, 3]}, box([0.02, 0.02, 0.48, 0.92], pos=[[0.25, 0.3], [0.25, 0.7], [0.15, 0.55]],
                                                             neg=[[0.25, 0.97], [0.5, 0.5]]),
                                        close=15, fixed=True, label="whole left tile with both stripes")]}
# --- 27803: composite of 20 tiles (4 columns x 5 rows); _1 held two tiles; _2.._6 unchanged
C4 = [(0.0, 0.248), (0.252, 0.497), (0.5, 0.75), (0.755, 1.0)]
R5 = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
_have = {(0, 0), (0, 1), (0, 3), (1, 1), (1, 2), (2, 3), (3, 3)}
# (the old _1 mask had holes inside both tiles: filled after applying, see the pipeline's MISS_INNER_PART flag)
R["27803"] = {"objects": [split(1, [0.755, 0.4, 1.0, 0.6], "tile row 3 col 4 (split from _1)", fill_holes=True)]
              + same(2, 3, 4, 5, 6)
              + [split(1, [0.755, 0.6, 1.0, 0.8], "tile row 4 col 4 (split from _1)", fill_holes=True)]
              + [cell([C4[c][0], R5[r][0], C4[c][1], R5[r][1]], f"tile row {r + 1} col {c + 1}")
                 for r in range(5) for c in range(4) if (r, c) not in _have]}
# --- 30789: the four pitchers separately, the outer (front, dark green) one first; yellow one kept
R["30789"] = {"objects": [new([0.34, 0.35, 0.86, 0.98], pos=[[0.62, 0.6], [0.65, 0.85], [0.75, 0.42]], neg=[[0.47, 0.45], [0.33, 0.52]],
                              label="outer (front) dark green pitcher"),
                          keep(1, fixed=True, exact=True, label="yellow pitcher (unchanged)"),
                          new([0.22, 0.2, 0.6, 0.62], pos=[[0.48, 0.24], [0.39, 0.35], [0.25, 0.47]], neg=[[0.5, 0.42], [0.33, 0.2], [0.62, 0.7]],
                              label="sage green pitcher"),
                          new([0.16, 0.14, 0.5, 0.5], pos=[[0.33, 0.2], [0.19, 0.36]], neg=[[0.46, 0.28], [0.3, 0.5]],
                              label="purple pitcher (back)")]}
# --- 36071: the 4th tile on the right was missing
R["36071"] = {"objects": ok(1, 2, 3) + [new([0.76, 0.23, 0.96, 0.81], pos=[[0.86, 0.5], [0.8, 0.35]], neg=[[0.7, 0.5], [0.98, 0.5], [0.86, 0.9]],
                                           label="4th tile (added)")]}
# --- 67781: _2 and _3 are one tile
R["67781"] = {"objects": same(1) + [merge(2, 3)]}
# --- 68637: two tiles -> two objects
R["68637"] = {"objects": [split(1, [0.0, 0.0, 0.505, 1.0], "left tile", fill_holes=True),
                          split(1, [0.505, 0.0, 1.0, 1.0], "right tile", fill_holes=True)]}
# --- 80339: the bowl and the speckled tile separately (the blue piece at the left edge belongs to neither)
R["80339"] = {"objects": [obj(box([0.39, 0.11, 1.0, 0.76], pos=[[0.65, 0.45], [0.5, 0.3]], neg=[[0.25, 0.45], [0.6, 0.85]]),
                              within=[1], fixed=True, label="mauve bowl"),
                          *same(2),
                          obj(box([0.09, 0.11, 0.41, 0.79], pos=[[0.25, 0.5], [0.2, 0.2]], neg=[[0.04, 0.5], [0.6, 0.45]]),
                              within=[1], fixed=True, label="speckled tile")]}
# --- 138162: _5 + _9 = glazed part of the middle tile (_6 carried the same note; it is fine as it is)
R["138162"] = {"objects": same(1, 2, 3, 4) + [merge(5, 9), keep(6, fixed=True, exact=True, label="unchanged (its note asked to combine _5 and _9: done)")]
               + same(7, 8)}
# --- 141251 / 141252: keep the black stripe on the right (convex outline of the square tile)
R["141251"] = {"objects": [keep(1, hull=True, fixed=True, label="whole tile incl. the black stripe")]}
R["141252"] = {"objects": [keep(1, hull=True, fixed=True, label="whole tile incl. the black stripe")]}
# --- 200118: panels _1.._9 -> the tile only (grey studio background removed); _10.._14 unchanged
T200118 = {1: [0.81, 0.1, 0.915, 0.28], 2: [0.105, 0.1, 0.215, 0.28], 3: [0.57, 0.1, 0.685, 0.28], 4: [0.1, 0.39, 0.21, 0.565],
           5: [0.33, 0.1, 0.445, 0.28], 6: [0.1, 0.67, 0.21, 0.855], 7: [0.575, 0.67, 0.685, 0.855], 8: [0.81, 0.38, 0.935, 0.565],
           9: [0.805, 0.67, 0.93, 0.86]}
R["200118"] = {"objects": [obj(box(b, pos=[[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]]), within=[k], fill_holes=True, fixed=True,
                               label="tile only, grey background removed") for k, b in sorted(T200118.items())]
               + same(10, 11, 12, 13, 14)}
# --- 210411: the plate and the 3 tiles on it, separately
_t210411 = [box([0.24, 0.325, 0.42, 0.61], pos=[[0.33, 0.42], [0.34, 0.56]], neg=[[0.23, 0.45], [0.5, 0.45]]),
            box([0.39, 0.29, 0.58, 0.57], pos=[[0.48, 0.4], [0.52, 0.53]], neg=[[0.62, 0.38], [0.36, 0.45]]),
            box([0.52, 0.27, 0.7, 0.54], pos=[[0.62, 0.36], [0.63, 0.5]], neg=[[0.5, 0.42]])]
R["210411"] = {"objects": [obj({"keep": [1]}, minus=_t210411, fixed=True, label="plate (tiles separated)")]
               + [obj(p, fixed=True, label=f"{n} tile on the plate") for p, n in zip(_t210411, ("left", "middle", "right"))]}
# --- 229652: cup without the grey background at the top right
R["229652"] = {"objects": [obj(box([0.0, 0.24, 1.0, 1.0], pos=[[0.5, 0.7], [0.5, 0.33], [0.2, 0.6], [0.9, 0.36]], neg=[[0.7, 0.1], [0.9, 0.2], [0.55, 0.05]]),
                               within=[1], largest=True, fixed=True, label="cup, grey background removed")]}
# --- 248684: only the green cup (no basket)
R["248684"] = {"objects": [obj(box([0.24, 0.0, 0.7, 0.26], pos=[[0.4, 0.1], [0.45, 0.2], [0.62, 0.04]], neg=[[0.15, 0.15], [0.76, 0.15], [0.63, 0.2], [0.2, 0.04]]),
                               within=[1], fixed=True, label="green cup, basket removed")]}
# --- 285462: two pieces: the right tile (kept) and the tall left one
R["285462"] = {"objects": [keep(1, fixed=True, exact=True, label="right tile (unchanged)"),
                           new([0.21, 0.07, 0.51, 0.87], pos=[[0.35, 0.3], [0.36, 0.6], [0.4, 0.75]], neg=[[0.15, 0.5], [0.56, 0.5], [0.3, 0.92]],
                               label="tall left tile (added)")]}
# --- 311328: two tiles
R["311328"] = {"objects": [split(1, [0.0, 0.0, 0.502, 1.0], "left tile"), split(1, [0.502, 0.0, 1.0, 1.0], "right tile")]}
# --- 320494: 12 pieces: _1.._9 fine, add the two small strips and the yellow tile at the bottom right
R["320494"] = {"objects": ok(*range(1, 10))
               + [new([0.255, 0.67, 0.47, 0.765], pos=[[0.36, 0.72]], neg=[[0.36, 0.64]], label="small strip, left (added)"),
                  new([0.505, 0.67, 0.7, 0.765], pos=[[0.6, 0.72]], neg=[[0.6, 0.64]], label="small strip, right (added)"),
                  new([0.49, 0.77, 0.81, 0.97], pos=[[0.65, 0.87], [0.55, 0.8]], neg=[[0.45, 0.87], [0.85, 0.87]], label="yellow tile (added)")]}
# --- 360618: also the whole left tile with the blue stripe (its parts _2 and _3 had been split off)
R["360618"] = {"objects": [keep(1, fixed=True, exact=True, label="right blue tile (unchanged)"),
                           obj({"keep": [2, 3]}, box([0.02, 0.15, 0.48, 0.87], pos=[[0.25, 0.45], [0.25, 0.8], [0.25, 0.22]],
                                                     neg=[[0.6, 0.5], [0.25, 0.95], [0.2, 0.08]]),
                               close=15, fixed=True, label="whole left tile with the blue stripe")]
               + same(4, 5)}
# --- 362096: _1 + _2 (right tile), _3 + _4 + _5 (left tile)
R["362096"] = {"objects": [merge(1, 2, label="right tile (_1 + _2)"), merge(3, 4, 5, label="left tile (_3 + _4 + _5)")]}
# --- 392555: _1 and _2 fine; the 3 centre tiles were split in halves (_7+_3, _6+_8, _5+_4)
_c392555 = {(7, 3): [0.235, 0.325, 0.428, 0.67], (6, 8): [0.425, 0.33, 0.61, 0.665], (5, 4): [0.598, 0.33, 0.79, 0.67]}
R["392555"] = {"objects": ok(1, 2, why="unchanged; the 3 centre tiles were rebuilt")
               + [merge(*ks, extra=[box(b, pos=[[(b[0] + b[2]) / 2, 0.42], [(b[0] + b[2]) / 2, 0.6]])], clip=b,
                        label=f"centre tile (_{ks[0]} + _{ks[1]})") for ks, b in _c392555.items()]}
# --- 423649: _2 and _3 are one tile
R["423649"] = {"objects": same(1) + [merge(2, 3)]}
# --- 424501: _5 + _7 = the right tile
R["424501"] = {"objects": same(1, 2, 3, 4) + [merge(5, 7)] + same(6)}
# --- 429348: _3 + _5 + _6 = glazed part of the left tile
R["429348"] = {"objects": same(1, 2) + [merge(3, 5, 6)] + same(4)}
# --- 449044: the vase without the flowers
R["449044"] = {"objects": [obj(box([0.21, 0.44, 0.78, 0.98], pos=[[0.5, 0.85], [0.5, 0.58], [0.27, 0.7], [0.7, 0.7]], neg=[[0.4, 0.25], [0.5, 0.42], [0.6, 0.38]]),
                               within=[1], clip=[0.0, 0.452, 1.0, 1.0], largest=True, fixed=True, label="vase, flowers removed")]}
# --- 465580: the white tile on the left was missing
R["465580"] = {"objects": ok(1, 2, 3) + [new([0.1, 0.31, 0.27, 0.84], pos=[[0.18, 0.55], [0.18, 0.78]], neg=[[0.05, 0.6], [0.3, 0.55], [0.18, 0.9]],
                                            label="white tile on the left (added)")]}
# --- 515280: the black spots of the outer band were holes in the mask
R["515280"] = {"objects": [keep(1, fill_holes=True, fixed=True, label="plate incl. the black spots of the outer band")]}
# --- 577052: _2 is the bottom of the bowl
R["577052"] = {"objects": [merge(1, 2, fill_holes=True, label="whole bowl (_1 + _2)")]}
# --- 608842: two cups
R["608842"] = {"objects": [obj(box([0.225, 0.33, 0.5, 0.71], pos=[[0.36, 0.5], [0.36, 0.67]], neg=[[0.62, 0.5]]), within=[1], fixed=True, label="left cup"),
                           obj(box([0.49, 0.32, 0.795, 0.68], pos=[[0.6, 0.56], [0.72, 0.36]], neg=[[0.4, 0.5]]), within=[1], fixed=True, label="right cup")]}
# --- 614728: disc tile without the brick background (and the hanging hole)
R["614728"] = {"objects": [obj(box([0.13, 0.07, 0.92, 0.87], pos=[[0.5, 0.5], [0.3, 0.7], [0.75, 0.35]], neg=[[0.42, 0.24], [0.05, 0.5], [0.96, 0.5], [0.5, 0.95], [0.9, 0.68]]),
                               largest=True, fixed=True, label="disc tile, brick background removed", **{"from": [1]})]}
# --- 619078: _3 + _4 + _5 = the middle tile
R["619078"] = {"objects": same(1, 2) + [merge(3, 4, 5, label="middle tile (_3 + _4 + _5)")]}
# --- 621014: _4 was only the stamp: the whole right tile (its foot _3 included, like the other two tiles)
R["621014"] = {"objects": same(1, 2) + [obj({"keep": [4, 3]}, box([0.635, 0.1, 0.975, 0.865], pos=[[0.8, 0.45], [0.8, 0.8], [0.7, 0.2]], neg=[[0.55, 0.45], [0.8, 0.95]]),
                                             close=15, fixed=True, label="whole right tile")]}
# --- 624285: two pieces (front white, back dark)
R["624285"] = {"objects": [obj(box([0.23, 0.14, 0.56, 0.81], pos=[[0.4, 0.3], [0.4, 0.7]], neg=[[0.63, 0.45]]), within=[1], fixed=True, label="front white piece"),
                           obj(box([0.5, 0.28, 0.73, 0.71], pos=[[0.62, 0.45], [0.65, 0.62]], neg=[[0.45, 0.5]]), within=[1], fixed=True, label="back dark piece")]}
# --- 626759: the vase without the grey background
R["626759"] = {"objects": [obj(box([0.34, 0.29, 0.62, 0.97], pos=[[0.47, 0.5], [0.47, 0.85], [0.58, 0.45]], neg=[[0.31, 0.7], [0.66, 0.75], [0.47, 0.99]]),
                               within=[1], largest=True, fixed=True, label="vase, grey background removed")]}
# --- 632730: the big F brick was missing
R["632730"] = {"objects": ok(1, 2) + [new([0.02, 0.47, 0.995, 0.89], pos=[[0.5, 0.65], [0.2, 0.7], [0.85, 0.65]], neg=[[0.5, 0.95], [0.3, 0.35]],
                                         label="F brick (added)")]}
# --- 633192: _6 + _8 = tile BL5
R["633192"] = {"objects": same(1, 2, 3, 4, 5) + [merge(6, 8, label="tile BL5 (_6 + _8)")] + same(7)}
# --- 634621: the vase without the flowers
R["634621"] = {"objects": [obj(box([0.35, 0.33, 0.62, 0.9], pos=[[0.5, 0.7], [0.48, 0.45], [0.52, 0.37]], neg=[[0.47, 0.29], [0.5, 0.15]]),
                               within=[1], clip=[0.0, 0.335, 1.0, 1.0], largest=True, fixed=True, label="vase, flowers removed")]}
# --- 635771: "check the square again": the whole square tile (ridged strips _2.._4 were split off)
R["635771"] = {"objects": [obj({"keep": [1, 2, 3, 4]}, box([0.01, 0.08, 0.98, 0.79], pos=[[0.5, 0.55], [0.5, 0.2]], neg=[[0.5, 0.9]]),
                               close=10, clip=[0.0, 0.07, 1.0, 0.8], fixed=True, label="whole square tile")] + same(5)}
# --- 643412: the oval pendant only (not the felt)
R["643412"] = {"objects": [obj(box([0.34, 0.26, 0.69, 0.65], pos=[[0.51, 0.5], [0.45, 0.4], [0.55, 0.6]], neg=[[0.37, 0.28], [0.2, 0.5], [0.8, 0.5]]),
                               within=[1], fixed=True, label="oval pendant")]}
# --- 647666: the vase without the feather
R["647666"] = {"objects": [obj(box([0.18, 0.25, 0.8, 0.9], pos=[[0.5, 0.6], [0.45, 0.35], [0.3, 0.55], [0.7, 0.55]], neg=[[0.76, 0.33], [0.1, 0.4], [0.88, 0.45], [0.5, 0.2]]),
                               within=[1], largest=True, fixed=True, label="vase, feather removed")]}
# --- 656336: the inner glazed disc as its own object; the outer plate kept separately
R["656336"] = {"objects": [obj(box([0.3, 0.39, 0.81, 0.74], pos=[[0.55, 0.56], [0.4, 0.65]], neg=[[0.2, 0.5], [0.55, 0.85]]),
                               within=[1], fill_holes=True, fixed=True, label="inner glazed disc"),
                           keep(1, fixed=True, label="outer plate (without the inner disc)")]}   # not exact: gives up the disc
# --- 656393: _2 is the bottom of the bowl
R["656393"] = {"objects": [merge(1, 2, fill_holes=True, label="whole bowl (_1 + _2)")]}
# --- 661382: only the bottle, no flowers / wall / hand
R["661382"] = {"objects": [obj(box([0.15, 0.46, 0.85, 0.97], pos=[[0.5, 0.8], [0.5, 0.55], [0.3, 0.75], [0.7, 0.75]], neg=[[0.2, 0.5], [0.8, 0.52], [0.5, 0.42], [0.86, 0.68], [0.5, 0.995]]),
                               clip=[0.1, 0.46, 0.9, 1.0], largest=True, fixed=True, label="bottle only", **{"from": [1]})]}
# --- 664662: precise outline of the tile (no board around it)
R["664662"] = {"objects": [obj(box([0.15, 0.11, 0.91, 0.945], pos=[[0.5, 0.5], [0.5, 0.88], [0.3, 0.2]], neg=[[0.07, 0.5], [0.95, 0.5], [0.5, 0.04], [0.5, 0.98]]),
                               within=[1], fill_holes=True, fixed=True, label="tile, precise outline")]}
# --- 680711 / 680717: _4 (base) + _5 (top) with the middle part _6 that joins them
R["680711"] = {"objects": same(1, 2, 3) + [merge(4, 5, 6, label="whole piece (_4 + _5 + the middle _6)")]}
R["680717"] = {"objects": same(1, 2, 3) + [merge(4, 5, 6, fill_holes=True, label="whole piece (_4 + _5, inner bit _6)")]}
# --- 680730: _3 + _5
R["680730"] = {"objects": same(1, 2) + [merge(3, 5)] + same(4)}
# --- 680766: _2 + _4
R["680766"] = {"objects": same(1) + [merge(2, 4)] + same(3, 5, 6)}
# --- 687304: _3 + the middle strip + _2 = the whole left tile
R["687304"] = {"objects": same(1) + [merge(2, 3, extra=[box([0.04, 0.285, 0.505, 0.885], pos=[[0.25, 0.5], [0.25, 0.8], [0.12, 0.5], [0.4, 0.5]], neg=[[0.25, 0.2], [0.53, 0.5]])],
                                           clip=[0.03, 0.28, 0.51, 0.89], label="whole left tile (_3 + middle + _2)")]}
# --- 695240: _1 + _2 + the blue right edge that neither covered = the whole tile
R["695240"] = {"objects": [merge(1, 2, extra=[box([0.19, 0.14, 0.71, 0.85], pos=[[0.45, 0.3], [0.45, 0.65], [0.65, 0.7]], neg=[[0.1, 0.5], [0.8, 0.5]])],
                                 clip=[0.18, 0.13, 0.72, 0.86], label="whole tile (_1 + _2)")]}

# --- 696154: _2 + _3 + the five texture bumps = the right tile; the 20 not-needed fragments stay as they are
R["696154"] = {"objects": same(1) + [merge(2, 3, 12, 13, 20, 21, 24, close=10, label="right tile (_2 + _3 + bumps _12 _13 _20 _21 _24)")]
               + same(4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27, 28)}
# --- 696741 / 696983 / 696984: two halves of one tile
R["696741"] = {"objects": same(1, 2) + [merge(3, 4)] + same(5)}
R["696983"] = {"objects": same(1) + [merge(2, 3)] + same(4)}
R["696984"] = {"objects": same(1) + [merge(2, 3)]}
# --- 702101: _1 + _2 (+ the small glaze chip _3 in its corner) = the whole tile
R["702101"] = {"objects": [merge(1, 2, 3, label="whole tile (_1 + _2, corner chip _3 included)")]}
# --- 7024xx AGC tiles: glazed upper parts were split in bands; the bisque feet (not needed) stay separate
R["702420"] = {"objects": [merge(1, 2, label="right tile, glazed part (_1 + _2)"),
                           merge(3, 4, extra=[box([0.115, 0.17, 0.48, 0.668], pos=[[0.3, 0.25], [0.3, 0.44], [0.3, 0.6]], neg=[[0.3, 0.75], [0.05, 0.4]])],
                                 minus=[{"keep": [6]}], clip=[0.11, 0.16, 0.49, 0.668], label="left tile, glazed part (_3 + middle band + _4)")]
               + same(5, 6)}
R["702425"] = {"objects": [merge(1, 2, label="right tile, glazed part (_1 + _2)"),
                           merge(3, 4, extra=[box([0.095, 0.18, 0.415, 0.655], pos=[[0.25, 0.25], [0.25, 0.42], [0.25, 0.58]], neg=[[0.25, 0.75], [0.04, 0.4]])],
                                 minus=[{"keep": [5]}], clip=[0.09, 0.17, 0.42, 0.66], label="left tile, glazed part (_3 + middle band + _4)")]
               + same(5, 6)}
R["702431"] = {"objects": [merge(1, 4, extra=[box([0.49, 0.15, 0.87, 0.72], pos=[[0.7, 0.2], [0.7, 0.37], [0.7, 0.55]], neg=[[0.7, 0.8], [0.95, 0.4]])],
                                 minus=[{"keep": [5]}, {"keep": [6]}], clip=[0.48, 0.14, 0.88, 0.725], label="right tile, glazed part (_4 + middle band + _1)"),
                           merge(3, 2, extra=[box([0.105, 0.13, 0.48, 0.74], pos=[[0.3, 0.25], [0.3, 0.45], [0.3, 0.62]], neg=[[0.3, 0.8], [0.05, 0.4]])],
                                 clip=[0.1, 0.12, 0.49, 0.745], label="left tile, glazed part (_3 + middle band + _2)")]
               + same(5, 6)}
R["702434"] = {"objects": [merge(1, 4, label="right tile, glazed part (_1 + _4)"),
                           obj({"keep": [2]}, box([0.11, 0.14, 0.445, 0.685], pos=[[0.28, 0.25], [0.28, 0.42], [0.28, 0.6]], neg=[[0.28, 0.78], [0.05, 0.4]]),
                               minus=[{"keep": [3]}], clip=[0.1, 0.13, 0.45, 0.69], close=15, fixed=True, label="left tile, glazed part incl. its upper part")]
               + same(3, 5)}
R["702441"] = {"objects": [merge(1, 4, extra=[box([0.095, 0.1, 0.455, 0.725], pos=[[0.28, 0.2], [0.28, 0.36], [0.28, 0.55]], neg=[[0.28, 0.78], [0.05, 0.4]])],
                                 minus=[{"keep": [6]}], clip=[0.09, 0.09, 0.46, 0.725], label="left tile, glazed part (_4 + middle band + _1)"),
                           merge(2, 3, label="right tile, glazed part (_3 + _2)")]
               + same(5, 6)}
R["702443"] = {"objects": same(1) + [merge(2, 3, label="right tile, glazed part (_2 + _3)")] + same(4, 5)}
R["702444"] = {"objects": same(1) + [merge(2, 4, label="left tile, glazed part (_2 + _4)")] + same(3)}
R["702446"] = {"objects": [merge(1, 3, label="right tile, glazed part (_1 + _3)"), merge(2, 4, label="left tile, glazed part (_2 + _4)")] + same(5, 6)}
R["702447"] = {"objects": [merge(1, 2, 7, label="right tile (_2 + _7 + _1)"), merge(3, 6, 4, label="left tile, glazed part (_3 + _6 + _4)")] + same(5)}
R["702449"] = {"objects": same(1) + [obj({"keep": [2]}, box([0.5, 0.12, 0.85, 0.72], pos=[[0.68, 0.2], [0.68, 0.38], [0.68, 0.6]], neg=[[0.68, 0.78], [0.95, 0.4]]),
                                         minus=[{"keep": [3]}], clip=[0.48, 0.11, 0.86, 0.73], close=15, fixed=True,
                                         label="right tile, glazed part incl. its upper part")] + same(3)}
R["702452"] = {"objects": same(1) + [merge(2, 4, extra=[box([0.485, 0.12, 0.865, 0.675], pos=[[0.68, 0.2], [0.68, 0.43], [0.68, 0.58]], neg=[[0.68, 0.75], [0.95, 0.4]])],
                                           minus=[{"keep": [3]}], clip=[0.48, 0.11, 0.87, 0.68], label="right tile, glazed part (_2 + middle part + _4)")] + same(3)}
# --- 702578: two tiles
R["702578"] = {"objects": [split(1, [0.395, 0.0, 1.0, 1.0], "LUCA tile (right)", fill_holes=True),
                           split(1, [0.0, 0.0, 0.395, 1.0], "small tile (left)", fill_holes=True)]}
# --- 702586: the LUCA tile on the right was missing
R["702586"] = {"objects": ok(1) + [new([0.38, 0.045, 0.97, 0.97], pos=[[0.65, 0.5], [0.7, 0.2], [0.6, 0.85]], neg=[[0.2, 0.4], [0.99, 0.5]],
                                      label="LUCA tile (added)")]}
# --- 702761: the green glazed part of the left tile was missing (its unglazed top _3 stays not needed)
R["702761"] = {"objects": same(1) + ok(2) + same(3, 4)
               + [obj(box([0.06, 0.435, 0.445, 0.945], pos=[[0.25, 0.7], [0.25, 0.5], [0.25, 0.9]], neg=[[0.25, 0.3], [0.52, 0.7]]),
                      minus=[{"keep": [3, 4]}], fixed=True, label="left tile, green glazed part (added)")]}
# --- 703095: _2 + _4 = glazed part of the left tile
R["703095"] = {"objects": same(1) + [merge(2, 4, label="left tile, glazed part (_4 + _2)")] + same(3)}
# --- 703195: two cups
R["703195"] = {"objects": [obj(box([0.12, 0.05, 0.53, 0.85], pos=[[0.35, 0.45], [0.16, 0.5]], neg=[[0.7, 0.45]]), within=[1], fixed=True, label="left cup (blue rim)"),
                           obj(box([0.51, 0.07, 0.87, 0.85], pos=[[0.7, 0.45]], neg=[[0.35, 0.45]]), within=[1], fixed=True, label="right cup")]}
# --- 703627: _2 + _3 = the whole left tile
R["703627"] = {"objects": same(1) + [merge(2, 3, label="left tile (_2 + _3)")] + same(4)}
# --- 703820: only the black saucer with its cup (the old mask was the background)
R["703820"] = {"objects": [obj(box([0.1, 0.22, 0.92, 0.61], pos=[[0.5, 0.45], [0.2, 0.42], [0.85, 0.42], [0.5, 0.25]], neg=[[0.5, 0.7], [0.5, 0.1], [0.05, 0.42]]),
                               fill_holes=True, largest=True, fixed=True, label="black saucer with cup", **{"from": [1]})]}
# --- 704195: _2 + _3 = the right tile
R["704195"] = {"objects": same(1) + [merge(2, 3, label="right tile (_3 + _2)")]}
# --- 704883: _2 with the upper part and the down part (_3) = the whole right tile; the hole _4 stays separate
R["704883"] = {"objects": same(1) + [merge(2, 3, extra=[box([0.46, 0.15, 0.84, 0.88], pos=[[0.65, 0.25], [0.65, 0.6], [0.65, 0.8]], neg=[[0.4, 0.5], [0.9, 0.5]])],
                                           clip=[0.45, 0.14, 0.85, 0.89], label="whole right tile (_2 + upper part + down part _3)")] + same(4)}
# --- 704884: _3 + middle + _1 = left tile; _2 + its upper part = right tile (hole _4 separate)
R["704884"] = {"objects": [merge(1, 3, extra=[box([0.09, 0.06, 0.45, 0.84], pos=[[0.27, 0.12], [0.27, 0.27], [0.27, 0.5], [0.27, 0.75]], neg=[[0.27, 0.9], [0.5, 0.35]])],
                                 clip=[0.08, 0.05, 0.46, 0.86], label="left tile (_3 + middle + _1)"),
                           obj({"keep": [2]}, box([0.5, 0.1, 0.85, 0.84], pos=[[0.67, 0.15], [0.67, 0.3], [0.67, 0.5], [0.67, 0.75]], neg=[[0.67, 0.9], [0.45, 0.3]]),
                               clip=[0.49, 0.09, 0.86, 0.86], close=15, fixed=True, label="right tile incl. its upper part")]
               + same(4)}
# --- 704885: _1 + _3 (left tile), _2 + _4 + _5 + _6 (right tile)
R["704885"] = {"objects": [merge(1, 3, label="left tile (_3 + _1)"), merge(2, 4, 5, 6, fill_holes=True, label="right tile (_5 + _6 + _4 + _2)")]}
# --- 706345 / 706351: close-ups of the glaze: the whole photo is the object
for _s in ("706345", "706351"):
    R[_s] = {"objects": [obj({"full": True}, fixed=True, label="whole photo (close-up, no segmentation needed)", **{"from": [1]})]}
# --- 708085: two tiles side by side (the left one's ragged edge restored)
R["708085"] = {"objects": [obj(box([0.0, 0.0, 0.5, 1.0], pos=[[0.25, 0.4], [0.25, 0.9], [0.45, 0.7]], neg=[[0.75, 0.5]]), clip=[0.0, 0.0, 0.499, 1.0],
                               fill_holes=True, fixed=True, label="left tile", **{"from": [1]}),
                           obj(box([0.5, 0.0, 1.0, 1.0], pos=[[0.75, 0.4], [0.75, 0.9]], neg=[[0.25, 0.5]]), clip=[0.499, 0.0, 1.0, 1.0],
                               fill_holes=True, fixed=True, label="right tile", **{"from": [1]})]}
# --- 708124: _2 + _4 + _3 + _5 = the left mould tile
R["708124"] = {"objects": same(1) + [merge(2, 3, 4, 5, label="left tile (_2 + _4 + _3 + _5)")]}
# --- 708135: 2 pieces: green-glazed left piece (_1 + _7 + _8) and red-glazed right piece (_5 + _2)
R["708135"] = {"objects": [merge(1, 7, 8, label="left glazed piece (_1 + _7 + _8)"), merge(2, 5, label="right glazed piece (_5 + _2)")]
               + same(3, 4, 6, 9, 10, 11)}
# --- 708147: the whole left slab = _2 + _3 + _4 + _5 + _6 + _7 (+ the unsegmented parts between them)
R["708147"] = {"objects": same(1) + [merge(2, 3, 4, 5, 6, 7, extra=[box([0.0, 0.255, 0.52, 0.99], pos=[[0.25, 0.4], [0.1, 0.6], [0.3, 0.75], [0.3, 0.95]],
                                                                       neg=[[0.75, 0.6], [0.25, 0.15]])],
                                           clip=[0.0, 0.25, 0.52, 0.99], fill_holes=True, label="whole left slab (_2 .. _7)")]}
# --- 709660: _3 + _7 = glazed part of the right tile
R["709660"] = {"objects": same(1, 2) + [merge(3, 7, label="right tile, glazed part (_7 + _3)")] + same(4, 5, 6, 8)}
# --- 710757: the black glazed part of the left tile was missing (same extent as _1 on the right tile)
R["710757"] = {"objects": ok(1) + [new([0.1, 0.17, 0.43, 0.46], pos=[[0.25, 0.3], [0.3, 0.4]], neg=[[0.3, 0.55], [0.05, 0.3], [0.47, 0.3]],
                                      label="left tile, black glazed part (added)")]}
# --- 714396: 4 tiles: also "No Opacifier" and "2 dips"
R["714396"] = {"objects": ok(1, 2) + [col([0.02, 0.07, 0.23, 0.81], "'No Opacifier' tile (added)"),
                                      col([0.5, 0.07, 0.71, 0.81], "'2 dips' tile (added)")]}
# --- 717790: the second (lower) slab was missing; its not-needed cell fragments _2 .. _10 become part of it
R["717790"] = {"objects": ok(1) + [obj({"keep": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
                                       box([0.06, 0.475, 0.8, 0.995], pos=[[0.45, 0.52], [0.25, 0.75], [0.55, 0.95], [0.75, 0.6]], neg=[[0.45, 0.3], [0.9, 0.7]]),
                                       clip=[0.05, 0.47, 0.81, 1.0], close=10, fill_holes=True, fixed=True,
                                       label="second (lower) slab, added (incl. its cell fragments _2 .. _10)")]}
# --- 733186: _2 + _3 (+ the strip between) = the right tile
R["733186"] = {"objects": same(1) + [merge(2, 3, extra=[box([0.47, 0.09, 0.98, 0.79], pos=[[0.7, 0.25], [0.7, 0.42], [0.7, 0.6]], neg=[[0.3, 0.5], [0.7, 0.9]])],
                                           clip=[0.46, 0.08, 0.99, 0.8], label="right tile (_2 + _3)")] + same(4)}
# --- 735165: not the purple plate edge (_4 -> not needed) but the black glazed part of the tile
R["735165"] = {"objects": same(1, 2, 3) + [keep(4, exact=True, set_status="not_needed", label="purple plate edge is not the target (reviewer note)"),
                                           new([0.22, 0.245, 0.77, 0.56], pos=[[0.5, 0.35], [0.35, 0.45], [0.65, 0.45]], neg=[[0.5, 0.7], [0.15, 0.4], [0.85, 0.4]],
                                               label="black glazed part of the tile (added)")]}
# --- 749139: _2 + _7 (left), _3 + _4 (right) = glazed parts
R["749139"] = {"objects": same(1) + [merge(2, 7, label="left tile, glazed part (_2 + _7)"), merge(3, 4, label="right tile, glazed part (_3 + _4)")]
               + same(5, 6)}
# --- 754362: the green tile only, no glove
R["754362"] = {"objects": [obj(box([0.2, 0.06, 0.68, 0.92], pos=[[0.45, 0.3], [0.45, 0.6], [0.45, 0.85], [0.3, 0.5], [0.62, 0.5]],
                                   neg=[[0.1, 0.5], [0.8, 0.5], [0.85, 0.35], [0.08, 0.9]]),
                               within=[1], fill_holes=True, largest=True, fixed=True, label="green tile, glove removed")]}
# --- 757862: the four quarters + the leaf = one tile
R["757862"] = {"objects": [merge(1, 2, 3, 4, 5, close=10, fill_holes=True, label="whole leaf tile (_1 .. _5)")] + same(6)}
# --- 757934: not the stand (_3 -> not needed) but the left tile
R["757934"] = {"objects": same(1, 2) + [keep(3, exact=True, set_status="not_needed", label="stand is not the target (reviewer note)"),
                                        new([0.0, 0.07, 0.36, 0.98], pos=[[0.15, 0.4], [0.2, 0.7]], neg=[[0.5, 0.5]], label="left tile (added)")]}
# --- 758447 / 758448 / 758457: tubes fine; the tubes beside them were missing
R["758447"] = {"objects": same(1) + ok(2, 3) + [new([0.6, 0.17, 0.88, 0.86], pos=[[0.75, 0.4], [0.75, 0.75]], neg=[[0.5, 0.5], [0.75, 0.95]],
                                                   label="right tube (added)")]}
R["758448"] = {"objects": same(1) + ok(2, 3) + [new([0.61, 0.09, 0.91, 0.77], pos=[[0.75, 0.35], [0.75, 0.65]], neg=[[0.5, 0.4], [0.75, 0.85]],
                                                   label="right tube (added)")]}
R["758457"] = {"objects": ok(1) + same(2) + [new([0.0, 0.04, 0.35, 0.97], pos=[[0.25, 0.3], [0.25, 0.6], [0.2, 0.85]], neg=[[0.5, 0.5]], label="left tube (added)"),
                                             new([0.65, 0.03, 0.99, 0.89], pos=[[0.82, 0.3], [0.82, 0.6], [0.82, 0.8]], neg=[[0.5, 0.5]], label="right tube (added)")]}
# --- 759382: _4 + _5
R["759382"] = {"objects": same(1, 2, 3) + [merge(4, 5)]}
# --- 763565 / 770414 / 770425 / 770426 / 773497: the missing tiles of the row
R["763565"] = {"objects": ok(1, 2, 3, 4, 5) + [col([0.0, 0.25, 0.12, 0.64], "tile 1 from the left (added)"),
                                               col([0.22, 0.27, 0.32, 0.63], "tile 3 from the left (added)"),
                                               col([0.31, 0.27, 0.4, 0.62], "tile 4 from the left (added)"),
                                               col([0.4, 0.25, 0.48, 0.63], "tile 5 from the left (added)"),
                                               col([0.48, 0.25, 0.56, 0.63], "tile 6 from the left (added)")]}
R["770414"] = {"objects": ok(1, 2, 3, 4, 5) + [col([0.06, 0.37, 0.17, 0.71], "tile 1 from the left (added)"),
                                               col([0.64, 0.35, 0.73, 0.7], "tile 6 from the left (added)"),
                                               col([0.745, 0.37, 0.84, 0.7], "tile 7 from the left (added)")]}
R["770425"] = {"objects": ok(1, 2, 3, 4, 5, 6) + [col([0.04, 0.35, 0.165, 0.69], "tile 1 from the left (added)"),
                                                  col([0.735, 0.35, 0.835, 0.69], "tile 7 from the left (added)")]}
R["770426"] = {"objects": ok(1, 2, 3, 4, 5, 6, 7) + [col([0.04, 0.35, 0.165, 0.69], "tile 1 from the left (added)")]}
R["773497"] = {"objects": ok(1, 2, 3, 4, 5, 6) + same(7, 8, 9) + [col([0.355, 0.185, 0.465, 0.46], "tile 'Δ1' (added)")]}
# --- 776403: the tile without the thumb
R["776403"] = {"objects": [obj(box([0.06, 0.07, 0.96, 0.89], pos=[[0.5, 0.3], [0.5, 0.6], [0.3, 0.75], [0.15, 0.85]], neg=[[0.6, 0.92], [0.75, 0.85], [0.5, 0.97], [0.25, 0.95]]),
                               within=[1], minus=[box([0.42, 0.76, 0.97, 1.0], pos=[[0.6, 0.92], [0.8, 0.85]])], largest=True, fixed=True,
                               label="tile, thumb removed")]}
# --- 777410: _1 + _2 = one piece
R["777410"] = {"objects": [merge(1, 2, fill_holes=True, label="whole piece (_1 + _2)")]}
# --- 786762: only the bowl
R["786762"] = {"objects": [obj(box([0.07, 0.22, 0.96, 0.56], pos=[[0.5, 0.35], [0.5, 0.5], [0.15, 0.42], [0.85, 0.38]], neg=[[0.5, 0.62], [0.03, 0.4], [0.5, 0.15], [0.98, 0.45]]),
                               fill_holes=True, largest=True, fixed=True, label="bowl only", **{"from": [1]})] + same(2, 3)}
# --- 792674: the left piece was missing
R["792674"] = {"objects": ok(1) + [new([0.06, 0.31, 0.41, 0.67], pos=[[0.25, 0.4], [0.25, 0.58]], neg=[[0.5, 0.5], [0.25, 0.75]], label="left tile (added)")]}
# --- 798234 / 798270: the tile without the PF700 board
R["798234"] = {"objects": [obj(box([0.21, 0.02, 0.87, 0.75], pos=[[0.6, 0.2], [0.55, 0.4], [0.4, 0.62], [0.3, 0.65]], neg=[[0.1, 0.5], [0.4, 0.9], [0.9, 0.5], [0.15, 0.15], [0.75, 0.7]]),
                               within=[1], largest=True, fixed=True, label="tile, PF700 board removed")]}
R["798270"] = {"objects": [obj(box([0.34, 0.05, 0.85, 0.73], pos=[[0.6, 0.2], [0.55, 0.4], [0.45, 0.62]], neg=[[0.15, 0.5], [0.45, 0.85], [0.9, 0.5], [0.2, 0.25], [0.75, 0.68]]),
                               within=[1], largest=True, fixed=True, label="tile, PF700 board removed")]}
# --- 803210 / 803211: brick + the two hooks = one piece
R["803210"] = {"objects": same(1) + [merge(2, 3, 4, close=10, label="brick with both hooks (_2 + _3 + _4)")] + same(5)}
R["803211"] = {"objects": same(1) + [merge(2, 3, 4, close=10, label="brick with both hooks (_2 + _3 + _4)")]}
# --- 803227 / 803231: the hooks + the brick below them ("down part")
R["803227"] = {"objects": same(1, 2, 3) + [merge(4, 5, 6, extra=[box([0.51, 0.55, 0.81, 0.64], pos=[[0.65, 0.6], [0.58, 0.615], [0.75, 0.6]], neg=[[0.65, 0.7]])],
                                                 close=10, label="brick with the hooks (_4 + _5 + _6 + brick)")] + same(7, 8, 9)}
R["803231"] = {"objects": same(1, 2) + [merge(3, 4, extra=[box([0.47, 0.575, 0.91, 0.655], pos=[[0.6, 0.62], [0.75, 0.62], [0.85, 0.62]], neg=[[0.6, 0.72]])],
                                              close=10, label="brick with the hooks (_3 + _4 + brick)")] + same(5, 6, 7)}
# --- 803983: the tile in the hand without the tile lying on the table below it
R["803983"] = {"objects": [obj(box([0.28, 0.3, 0.81, 0.7], pos=[[0.45, 0.5], [0.75, 0.45]], neg=[[0.75, 0.75], [0.65, 0.78]]), within=[1], largest=True,
                               fixed=True, label="tile in the hand (table part removed)")] + same(2, 3, 4, 5)}
# --- 804169 .. 804182: tiles split in bands -> whole tiles as noted
R["804169"] = {"objects": same(1, 2) + [merge(3, 5, 4, label="left tile (_3 + _5 + _4)")] + same(6, 7, 8)}
R["804172"] = {"objects": same(1) + [merge(2, 3, label="left tile (_2 + _3)")] + same(4, 5, 6, 7)}
R["804175"] = {"objects": [merge(1, 3, label="right tile (_1 + _3)"), merge(2, 4, 5, label="left tile (_2 + _5 + _4)")] + same(6)}
R["804180"] = {"objects": [merge(1, 4, 3, label="left tile (_1 + _4 + _3)"), merge(2, 5, 6, label="right tile (_2 + _5 + _6)")] + same(7, 8, 9, 10)}
R["804182"] = {"objects": [merge(1, 4, label="left tile (_1 + _4)"), merge(2, 5, 3, label="right tile (_2 + _5 + _3)")] + same(6, 7)}


# every photo's before/after preview was inspected (manual_corrections_sheets_v2/preview_*.jpg, fixes in fix1/)
VERIFIED = set(R)

if __name__ == "__main__":
    for k in VERIFIED:
        R[k]["verified"] = True
    p = Path(__file__).with_name("manual_corrections_v2.json")
    p.write_text(json.dumps({"_source": "configs/manual_corrections_v2_src.py", **R}, indent=1), encoding="utf-8")
    print(f"{len(R)} photos -> {p}")
