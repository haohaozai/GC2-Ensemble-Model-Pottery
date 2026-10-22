"""Source of configs/manual_corrections.json (run this file to regenerate the JSON).

One entry per photo marked "Needs review" in the review app (review/object_review.csv, 2026-10-15).
Each entry lists the FINAL objects of the photo (see scripts/19_manual_corrections.py for the part syntax).
Coordinates are fractions of the photo width / height, read off the inspection sheets' 0.1 grid.
"""
import json
from pathlib import Path

R = {}


def keep(*ks, **kw):
    return {"parts": [{"keep": list(ks)}], **kw}


def box(b, pos=None, neg=None, pick="best", **kw):
    p = {"box": b, "pick": pick}
    if pos:
        p["pos"] = pos
    if neg:
        p["neg"] = neg
    return p


def obj(*parts, **kw):
    return {"parts": list(parts), **kw}


def same(stem, ks):
    """Unchanged objects of the photo, each kept as it is."""
    return [keep(k, label="unchanged") for k in ks]


# --- 900: only the purple (lavender) cup; the other cups are dropped
# (checked on the untinted photo: the only purple cup is the lavender one at the front right; object 1 had merged
#  it with the teal cup behind it)
R["900"] = {"objects": [obj(box([0.6, 0.5, 0.94, 0.97], pos=[[0.78, 0.75], [0.7, 0.6], [0.88, 0.62]], neg=[[0.68, 0.43], [0.5, 0.7], [0.78, 0.3]]),
                            fill_holes=True, largest=True, label="purple (lavender) cup")]}
# --- 1342: lava tile without the grey background on the right
R["1342"] = {"objects": [obj(box([0.0, 0.0, 0.9, 1.0], pos=[[0.4, 0.5], [0.6, 0.2]], neg=[[0.95, 0.5], [0.96, 0.85]]),
                             fill_holes=True, label="lava tile, grey background removed")]}
# --- 2602: the bowl (dish) mask was broken; other tiles unchanged, 7 not needed
R["2602"] = {"objects": [obj(box([0.6, 0.0, 0.93, 0.41], pos=[[0.77, 0.2], [0.65, 0.3]]), fill_holes=True, label="whole dish")]
             + same("2602", [2, 3, 4, 5, 6])}
# --- 3244: tile with the removed inner pixels restored
R["3244"] = {"objects": [obj({"keep": [1]}, {"rect": [0.05, 0.02, 0.91, 0.97]}, fill_holes=True, label="tile, inner pixels restored")]}
# --- 6051: red vessel without background
R["6051"] = {"objects": [obj(box([0.19, 0.0, 0.87, 1.0], pos=[[0.45, 0.5], [0.4, 0.85], [0.5, 0.2]], neg=[[0.08, 0.5], [0.95, 0.5], [0.12, 0.9], [0.33, 0.96]]),
                             fill_holes=True, label="red vessel, background removed")]}
# --- 6384: mug unchanged; bowl rim at the top with its shadowed part restored
R["6384"] = {"objects": [keep(1, label="unchanged mug"),
                         obj({"keep": [2]}, {"rect": [0.0, 0.12, 0.35, 0.33]}, fill_holes=True, label="bowl rim, inner pixels restored")]}
# --- 18344: the whole tile is one object
R["18344"] = {"objects": [obj(box([0.11, 0.06, 0.89, 0.93], pos=[[0.3, 0.55], [0.7, 0.3], [0.5, 0.8]]), label="whole tile")]}
# --- 19539: plate with inner holes filled
R["19539"] = {"objects": [keep(1, fill_holes=True, label="plate, holes filled")]}
# --- 19734 (no note): lid only -> the whole lidded jar
R["19734"] = {"objects": [obj(box([0.12, 0.07, 0.9, 0.95], pos=[[0.5, 0.3], [0.5, 0.7]]), fill_holes=True, label="whole lidded jar")]}
# --- 19884: the six parts are one sculpture; drop the thin string on the left
R["19884"] = {"objects": [obj({"keep": [1, 2, 3, 4, 5, 6]}, clip=[0.405, 0.0, 0.95, 0.92],
                              fill_holes=True, largest=True, label="whole sculpture (string excluded)")]}
# --- 27852: cup without the whipped cream
R["27852"] = {"objects": [obj({"keep": [1]}, minus=[box([0.3, 0.25, 0.7, 0.42], pos=[[0.5, 0.31], [0.4, 0.35]])], largest=True,
                              label="cup without the cream")]}
# --- 30789: only the yellow pitcher
R["30789"] = {"objects": [obj(box([0.25, 0.2, 0.7, 0.85], pos=[[0.52, 0.3], [0.62, 0.36], [0.33, 0.57]], neg=[[0.62, 0.62], [0.42, 0.32], [0.6, 0.85]]),
                              label="yellow pitcher")]}
# --- 30809: two stacked bowls -> 2 objects
R["30809"] = {"objects": [obj(box([0.06, 0.0, 0.95, 0.51], pos=[[0.5, 0.25]]), within=[1], label="top bowl"),
                          obj(box([0.05, 0.49, 0.96, 1.0], pos=[[0.5, 0.75]]), within=[1], label="bottom bowl")]}
# --- 31640: two vessels -> 2 objects
R["31640"] = {"objects": [obj(box([0.12, 0.1, 0.56, 1.0], pos=[[0.3, 0.5], [0.3, 0.85]], neg=[[0.75, 0.5]]), within=[1], label="black vessel"),
                          obj(box([0.5, 0.0, 0.97, 1.0], pos=[[0.75, 0.5], [0.8, 0.85]], neg=[[0.3, 0.5]]), within=[1], label="rust vessel")]}

# --- 32197: two merged masses of stacked tiles -> every tile separately; tiles 3 and 4 unchanged
R["32197"] = {"objects": [{"auto_split": {"within": [1, 2], "grow": 30, "step": 0.012, "min_area": 0.006, "max_area": 0.2,
                                          "min_iou": 0.7}, "label": "stacked tile"}]
              + same("32197", [3, 4])}
# --- 44327: recipe sheet: only the crawl texture photo and the vase photo, not the text page
R["44327"] = {"objects": [obj(box([0.36, 0.55, 0.52, 0.83], pos=[[0.44, 0.7]]), label="vase photo"),
                          obj({"rect": [0.045, 0.36, 0.395, 0.915]}, label="crawl glaze texture photo")]}
# --- 49972: the left tile was missed (3 tiles, not 2)
R["49972"] = {"objects": [obj(box([0.11, 0.26, 0.38, 0.83], pos=[[0.24, 0.5], [0.24, 0.75]]), label="left tile (was missed)")]
              + same("49972", [1, 2])}
# --- 51909: bowl 4 without the eggs; other dishes unchanged (1, 9, 10 not needed)
R["51909"] = {"objects": [obj({"keep": [4]}, minus=[box([0.42, 0.66, 0.63, 0.8], pos=[[0.47, 0.75], [0.53, 0.72], [0.57, 0.76]])],
                              largest=True, label="bowl without eggs")] + same("51909", [2, 3, 5, 6, 7, 8])}

# --- 53961: hexagonal dish only (the mask had taken the table), basic crop of the dish
R["53961"] = {"objects": [obj(box([0.13, 0.1, 0.8, 0.87], pos=[[0.5, 0.45], [0.3, 0.6], [0.65, 0.4]]), fill_holes=True,
                              label="hexagonal dish")]}
# --- 54689: frame-filling tile with the missing top-right part restored
R["54689"] = {"objects": [obj({"keep": [1]}, {"rect": [0.03, 0.02, 0.97, 0.95]}, fill_holes=True, label="whole tile")]}
# --- 67464: the back-left tile was missed (3 tiles, not 2)
R["67464"] = {"objects": same("67464", [1, 2])
              + [obj(box([0.03, 0.12, 0.44, 0.48], pos=[[0.13, 0.25], [0.08, 0.4]], neg=[[0.4, 0.4]]), label="back-left tile (was missed)")]}
# --- 72382: three stacked bowls -> 3 objects
R["72382"] = {"objects": [obj(box([0.2, 0.2, 0.81, 0.345], pos=[[0.5, 0.27]]), within=[1], label="top yellow bowl"),
                          obj(box([0.19, 0.33, 0.81, 0.43], pos=[[0.5, 0.385]]), within=[1], label="middle grey bowl"),
                          obj(box([0.18, 0.42, 0.82, 0.82], pos=[[0.5, 0.6]]), within=[1], label="bottom pink bowl")]}

# --- 77052: two tiles side by side -> 2 objects
R["77052"] = {"objects": [obj({"keep": [1]}, clip=[0.0, 0.0, 0.487, 1.0], largest=True, label="left tile"),
                          obj({"keep": [1]}, clip=[0.493, 0.0, 1.0, 1.0], largest=True, label="right tile")]}
# --- 81407: planter without the plant; object 1 was the table top (not ceramic) and is dropped
R["81407"] = {"objects": [obj({"keep": [2]}, minus=[box([0.18, 0.08, 0.6, 0.5], pos=[[0.33, 0.25], [0.4, 0.18], [0.27, 0.35]])],
                              clip=[0.0, 0.445, 1.0, 1.0], largest=True, label="planter without the plant")]}
# --- 85496: crystal glaze close-up, basic segmentation = the whole frame
R["85496"] = {"objects": [obj({"keep": [1]}, {"rect": [0.0, 0.012, 1.0, 1.0]}, fill_holes=True, label="whole glaze surface")]}
# --- 88786: figure with 9 samples: each sample re-segmented without the grey paper around it
R["88786"] = {"objects": [obj(box([0.38, 0.55, 0.68, 0.785], pos=[[0.5, 0.66]], neg=[[0.33, 0.62], [0.64, 0.765]]),
                              fill_holes=True, largest=True, label="sample (i) without grey background")]
              + [obj({"refine": k}, fill_holes=True, largest=True, label=f"sample {k} without grey background") for k in range(2, 10)]}

# --- 102766: two test pieces lying on the recipe paper -> 2 objects, paper excluded
R["102766"] = {"objects": [obj(box([0.18, 0.0, 0.93, 0.31], pos=[[0.45, 0.15], [0.82, 0.15]]), label="upper test piece"),
                           obj(box([0.11, 0.3, 0.88, 0.59], pos=[[0.4, 0.45], [0.8, 0.45]]), label="lower test piece")]}
# --- 108815: turquoise bowl and the brown bowl behind it -> 2 objects (2 not needed)
R["108815"] = {"objects": [obj(box([0.05, 0.33, 0.97, 0.87], pos=[[0.5, 0.55], [0.62, 0.8]]), within=[1], fill_holes=True, label="turquoise bowl"),
                           obj(box([0.26, 0.12, 0.76, 0.35], pos=[[0.5, 0.27], [0.4, 0.22]]), within=[1], fill_holes=True, label="brown bowl behind")]}
# --- 111576: only the red-orange tile in the middle (the blue wheel head is not ceramic of interest)
R["111576"] = {"objects": [obj(box([0.28, 0.15, 0.82, 0.84], pos=[[0.52, 0.4], [0.45, 0.75]], neg=[[0.1, 0.7], [0.9, 0.7], [0.45, 0.93]]),
                               clip=[0.0, 0.0, 1.0, 0.84], label="red-orange tile")]}
# --- 114323: tile 1 without the beige grout around it; the other tiles unchanged
R["114323"] = {"objects": [obj(box([0.32, 0.22, 0.67, 0.69], pos=[[0.5, 0.45], [0.4, 0.35], [0.6, 0.6]], neg=[[0.3, 0.72], [0.69, 0.2]]),
                               fill_holes=True, label="dark grey tile without grout")] + same("114323", [2, 3, 4, 5, 6, 7, 8])}

# --- 114928: two test tiles held in a hand -> 2 objects, fingers excluded
R["114928"] = {"objects": [obj(box([0.15, 0.26, 0.5, 0.7], pos=[[0.3, 0.38], [0.33, 0.55]], neg=[[0.55, 0.7]]), label="left tile"),
                           obj(box([0.45, 0.25, 0.8, 0.67], pos=[[0.62, 0.38], [0.7, 0.58]], neg=[[0.58, 0.72]]), label="right tile")]}
# --- 147183: two tiles side by side -> 2 objects
R["147183"] = {"objects": [obj({"keep": [1]}, clip=[0.0, 0.0, 0.534, 1.0], largest=True, label="left tile"),
                           obj({"keep": [1]}, clip=[0.538, 0.0, 1.0, 1.0], largest=True, label="right tile")]}
# --- 171443: 3, 4 and 5 are one test piece; 2 unchanged (1 = background, not needed)
R["171443"] = {"objects": [keep(3, 4, 5, fill_holes=True, label="right test piece (3 parts merged)"), keep(2, label="unchanged")]}
# --- 201167: central bowl only (the mask had taken the white background); other bowls unchanged
R["201167"] = {"objects": [obj(box([0.33, 0.27, 0.74, 0.66], pos=[[0.53, 0.45], [0.4, 0.4]], neg=[[0.3, 0.7], [0.75, 0.25]]),
                               fill_holes=True, label="central bowl")] + same("201167", [2, 3, 4, 5, 6])}

# --- 218616: 1 and 3 are one test tile (+ its foot); 2 was background
R["218616"] = {"objects": [obj({"keep": [1, 3]}, box([0.26, 0.79, 0.8, 0.92], pos=[[0.55, 0.85]], neg=[[0.1, 0.85], [0.9, 0.85]]),
                               fill_holes=True, label="whole test tile (1 + 3 + foot)")]}
# --- 220305: copper red plate with the missing left part restored, hand excluded
R["220305"] = {"objects": [obj({"keep": [1]}, box([0.02, 0.24, 0.99, 0.96], pos=[[0.5, 0.6], [0.12, 0.55]], neg=[[0.03, 0.85], [0.5, 0.12]]),
                               fill_holes=True, largest=True, label="whole plate")]}
# --- 240822: two tiles held together -> 2 objects; orange holder below and the background (1) dropped
R["240822"] = {"objects": [obj({"keep": [2]}, clip=[0.0, 0.0, 0.474, 0.775], largest=True, label="white tile"),
                           obj({"keep": [2]}, clip=[0.478, 0.0, 1.0, 0.775], largest=True, label="dark tile")]}
# --- 266741: the pentagon tile on the wooden table (the mask had taken the whole photo)
R["266741"] = {"objects": [obj(box([0.31, 0.11, 0.72, 0.87], pos=[[0.5, 0.5], [0.48, 0.18], [0.5, 0.8]], neg=[[0.15, 0.5], [0.85, 0.5]]),
                               fill_holes=True, label="pentagon tile")]}

# --- 285462: only the smaller test piece on the right (the big one is not wanted)
R["285462"] = {"objects": [obj(box([0.57, 0.33, 0.79, 0.81], pos=[[0.68, 0.55], [0.66, 0.4]], neg=[[0.5, 0.6], [0.85, 0.6]]),
                               fill_holes=True, label="smaller test piece")]}
# --- 338433: two tiles; the right (white) tile was missed
R["338433"] = {"objects": [keep(1, label="unchanged left tile"),
                           obj(box([0.56, 0.39, 0.92, 0.76], pos=[[0.73, 0.55], [0.7, 0.7]]), label="right tile (was missed)")]}
# --- 366495: one glazed plate with four stripes -> a single object
R["366495"] = {"objects": [obj({"keep": [1, 2, 3, 4]}, box([0.0, 0.0, 1.0, 1.0], pos=[[0.85, 0.5], [0.5, 0.5], [0.15, 0.5]], pick="largest"),
                               fill_holes=True, largest=True, label="whole glazed plate")]}
# --- 371662: cup and the foam inside are one object
R["371662"] = {"objects": [keep(1, 2, fill_holes=True, label="whole cup (1 + 2)")]}

# --- 376617: vase without the leather glove
R["376617"] = {"objects": [obj(box([0.06, 0.16, 0.83, 0.78], pos=[[0.4, 0.35], [0.4, 0.6], [0.2, 0.45]],
                                   neg=[[0.3, 0.85], [0.86, 0.62], [0.04, 0.45], [0.75, 0.8]]), fill_holes=True, label="vase without glove")]}
# --- 388290: coaxial blend board: for the 5 marked tiles take the glazed (blended colour) upper part, not the label part
R["388290"] = {"objects": [obj({"above": k, "factor": 1.25}, largest=True, label=f"glazed upper part above label {k}")
                           for k in (1, 2, 3, 4, 5)]}
# --- 409689: the large LUCA tile was missed
R["409689"] = {"objects": [keep(1, label="unchanged small tile"),
                           obj(box([0.34, 0.03, 0.94, 0.98], pos=[[0.6, 0.5], [0.5, 0.85]], neg=[[0.15, 0.4]]), fill_holes=True,
                               label="LUCA tile (was missed)")]}
# --- 430651: round dish and rectangular tile -> 2 objects
R["430651"] = {"objects": [obj(box([0.04, 0.0, 0.84, 0.65], pos=[[0.5, 0.3]]), within=[1], fill_holes=True, label="round dish"),
                           obj(box([0.03, 0.63, 0.98, 0.99], pos=[[0.2, 0.8], [0.6, 0.8]]), within=[1], label="tile")]}

# --- 467278: 2, 4 and 5 are one glazed test tile (1 = background and 3 = white foot were marked not needed)
R["467278"] = {"objects": [keep(2, 4, 5, fill_holes=True, label="whole glazed tile (2 + 4 + 5)")]}
# --- 472313: phone screenshot with two bowls -> 2 objects, the white wall excluded
R["472313"] = {"objects": [obj(box([0.1, 0.43, 0.52, 0.58], pos=[[0.3, 0.5], [0.3, 0.55]], neg=[[0.55, 0.45]]), label="left bowl"),
                           obj(box([0.56, 0.46, 0.95, 0.58], pos=[[0.75, 0.52]], neg=[[0.55, 0.45]]), label="right bowl")]}
# --- 476537: right tile without the black background; left tile unchanged
R["476537"] = {"objects": [obj(box([0.53, 0.29, 0.82, 0.69], pos=[[0.68, 0.48]], neg=[[0.86, 0.5], [0.6, 0.72]]), fill_holes=True,
                               label="right tile without background"), keep(2, label="unchanged left tile")]}
# --- 477053: the right tile is one object (2 and 3 were parts of it); left tile unchanged
R["477053"] = {"objects": [keep(1, label="unchanged left tile"),
                           obj({"keep": [2, 3]}, box([0.51, 0.37, 0.86, 0.74], pos=[[0.68, 0.6], [0.66, 0.44]], neg=[[0.95, 0.5], [0.45, 0.8]]),
                               fill_holes=True, label="whole right tile")]}

# --- 486265: vase without the bottle standing behind it
R["486265"] = {"objects": [obj(box([0.35, 0.31, 0.67, 0.74], pos=[[0.5, 0.45], [0.5, 0.68]], neg=[[0.6, 0.22], [0.55, 0.1]]),
                               clip=[0.0, 0.31, 1.0, 1.0], fill_holes=True, largest=True, label="vase without the bottle")]}
# --- 490823: 2, 3, 4 were stamps on the BM5 tile -> the whole BM5 tile; C6 tile unchanged
R["490823"] = {"objects": [obj(box([0.1, 0.2, 0.47, 0.85], pos=[[0.25, 0.4], [0.25, 0.65], [0.15, 0.6]]), clip=[0.0, 0.0, 1.0, 0.725],
                               fill_holes=True, label="whole BM5 tile (printed label strip excluded)"), keep(1, label="unchanged C6 tile")]}
# --- 492884: left tile as a whole (2 was only its glazed top); right tile unchanged
R["492884"] = {"objects": [obj(box([0.08, 0.2, 0.44, 0.84], pos=[[0.27, 0.4], [0.27, 0.7], [0.2, 0.62]]), clip=[0.0, 0.0, 1.0, 0.745],
                               fill_holes=True, label="whole left tile (printed label strip excluded)"), keep(1, label="unchanged right tile")]}
# --- 510474: cup without the grey background
R["510474"] = {"objects": [obj(box([0.42, 0.32, 0.69, 0.67], pos=[[0.55, 0.45], [0.55, 0.62]], neg=[[0.37, 0.55], [0.76, 0.55], [0.55, 0.75]]),
                               fill_holes=True, largest=True, label="cup without background")]}

# --- 510475: cup without the background
R["510475"] = {"objects": [obj(box([0.41, 0.27, 0.6, 0.8], pos=[[0.5, 0.45], [0.5, 0.74]], neg=[[0.36, 0.5], [0.64, 0.5], [0.5, 0.85]]),
                               fill_holes=True, largest=True, label="cup without background")]}
# --- 527654: dark tile and the dripping glaze on it are one object
R["527654"] = {"objects": [keep(1, 2, fill_holes=True, label="whole tile (1 + 2)")]}
# --- 529979: the dish and the glaze gloops on it are one object
R["529979"] = {"objects": [keep(1, 2, 3, fill_holes=True, label="whole dish (1 + 2 + 3)")]}
# --- 533349: the full bowl, not only the patch in the middle
R["533349"] = {"objects": [obj({"keep": [1]}, box([0.0, 0.0, 1.0, 1.0], pos=[[0.4, 0.5], [0.6, 0.15], [0.3, 0.85]], neg=[[0.02, 0.97]],
                                                  pick="largest"), fill_holes=True, largest=True, label="full bowl")]}

# --- 536829: two tiles: left = 1 + 4, right = 2 + 3
R["536829"] = {"objects": [keep(1, 4, fill_holes=False, label="left tile (1 + 4)"), keep(2, 3, label="right tile (2 + 3)")]}
# --- 552676: the whole pink plate, background excluded
R["552676"] = {"objects": [obj(box([0.0, 0.17, 1.0, 1.0], pos=[[0.5, 0.65], [0.3, 0.45], [0.8, 0.3], [0.97, 0.5]],
                                   neg=[[0.3, 0.08], [0.9, 0.03], [0.02, 0.2]]), fill_holes=True, label="whole pink plate")]}
# --- 553051: two tiles held side by side -> 2 objects
R["553051"] = {"objects": [obj(box([0.06, 0.0, 0.47, 0.99], pos=[[0.25, 0.5], [0.25, 0.9]], neg=[[0.7, 0.5]]), within=[1], label="left tile"),
                           obj(box([0.43, 0.2, 0.97, 0.98], pos=[[0.7, 0.5], [0.6, 0.9]], neg=[[0.25, 0.5]]), within=[1], label="right tile")]}
# --- 569486: glaze close-up: the whole photo
R["569486"] = {"objects": [obj({"full": True}, label="whole photo (glaze close-up)")]}

# --- 577588: the blue test tile as one object (it had been cut into 2 glazed parts)
R["577588"] = {"objects": [obj({"keep": [1, 2]}, box([0.29, 0.35, 0.66, 0.83], pos=[[0.5, 0.5], [0.47, 0.72]], neg=[[0.2, 0.5], [0.75, 0.6]]),
                               fill_holes=True, label="whole blue test tile")]}
# --- 578763: the whole tile (1 + 2) without the black background
R["578763"] = {"objects": [obj(box([0.32, 0.11, 0.71, 0.98], pos=[[0.52, 0.35], [0.5, 0.8], [0.52, 0.6]], neg=[[0.2, 0.5], [0.85, 0.5], [0.85, 0.15]]),
                               fill_holes=True, label="whole tile without background")]}
# --- 596272: middle tile (3 + 4 + 5 + its lower part) and right tile (6 + 2 + rest) as 2 objects; left tile unchanged
R["596272"] = {"objects": [keep(1, label="unchanged left tile"),
                           obj({"keep": [3, 4, 5]}, box([0.32, 0.03, 0.61, 0.91], pos=[[0.47, 0.3], [0.47, 0.75]], neg=[[0.2, 0.5], [0.77, 0.5]]),
                               fill_holes=True, label="whole middle tile"),
                           obj({"keep": [2, 6]}, box([0.62, 0.04, 0.93, 0.89], pos=[[0.77, 0.4], [0.77, 0.75]], neg=[[0.47, 0.5]]),
                               fill_holes=True, label="whole right tile")]}
# --- 598424: two tiles stacked vertically -> 2 objects
R["598424"] = {"objects": [obj({"keep": [1]}, clip=[0.0, 0.0, 1.0, 0.497], largest=True, label="upper tile"),
                           obj({"keep": [1]}, clip=[0.0, 0.503, 1.0, 1.0], largest=True, label="lower tile")]}

# --- 599893 ("Ash A", reviewer unsure): the glazed test tile resting on the white bisque spatula; the counter top
#     and the spatula are not the glaze sample
R["599893"] = {"objects": [obj(box([0.2, 0.345, 0.75, 0.5], pos=[[0.45, 0.42], [0.3, 0.45], [0.65, 0.41]], neg=[[0.85, 0.37], [0.18, 0.32]]),
                               fill_holes=True, label="ash glaze test tile")]}
# --- 604442: block without the background around it
R["604442"] = {"objects": [obj(box([0.39, 0.3, 0.66, 0.94], pos=[[0.52, 0.5], [0.52, 0.85]], neg=[[0.7, 0.5], [0.52, 0.97], [0.35, 0.5]]),
                               fill_holes=True, largest=True, label="block without background")]}
# --- 606562: full ST, BB and ICF tiles (each had been cut into parts); teal cup (6) not needed
R["606562"] = {"objects": [obj(box([0.12, 0.1, 0.37, 0.78], pos=[[0.24, 0.3], [0.24, 0.7]], neg=[[0.5, 0.5]]), fill_holes=True, label="full ST tile"),
                           obj(box([0.37, 0.04, 0.64, 0.81], pos=[[0.51, 0.3], [0.51, 0.7]], neg=[[0.24, 0.5], [0.75, 0.5]]), fill_holes=True,
                               label="full BB tile"),
                           obj(box([0.63, 0.17, 0.88, 0.84], pos=[[0.75, 0.45], [0.75, 0.78]], neg=[[0.51, 0.5], [0.95, 0.4]]), fill_holes=True,
                               label="full ICF tile")]}
# --- 607115: triangle palette and each rectangular palette separately (1 was the whole shelf); cups unchanged
R["607115"] = {"objects": [obj(box([0.05, 0.39, 0.58, 0.84], pos=[[0.3, 0.65], [0.45, 0.5]], neg=[[0.62, 0.6]]), fill_holes=True,
                               label="triangle palette"),
                           {"auto_split": {"rect": [0.59, 0.31, 1.0, 0.83], "step": 0.02, "min_area": 0.06, "max_area": 0.35, "min_iou": 0.8},
                            "label": "rectangular palette"}] + same("607115", [2, 3, 4])}

# --- 612338: the whole plate (rim included) without the grass
R["612338"] = {"objects": [obj(box([0.16, 0.11, 0.84, 0.84], pos=[[0.5, 0.48], [0.24, 0.5], [0.5, 0.15]], neg=[[0.05, 0.05], [0.95, 0.95], [0.05, 0.95]]),
                               fill_holes=True, largest=True, label="whole plate without grass")]}
# --- 613558: only the blue tile, not the concrete slab
R["613558"] = {"objects": [obj(box([0.22, 0.33, 0.63, 0.66], pos=[[0.43, 0.5]], neg=[[0.15, 0.2], [0.8, 0.8]]), fill_holes=True,
                               label="blue tile")]}
# --- 616520: glaze close-up: the whole photo
R["616520"] = {"objects": [obj({"full": True}, label="whole photo (glaze close-up)")]}
# --- 619901: two tiles -> 2 objects, the left one complete
R["619901"] = {"objects": [obj(box([0.02, 0.15, 0.49, 0.97], pos=[[0.25, 0.45], [0.25, 0.85]], neg=[[0.7, 0.5]]), fill_holes=True, label="left tile"),
                           obj(box([0.45, 0.05, 0.99, 0.96], pos=[[0.72, 0.4], [0.72, 0.85]], neg=[[0.25, 0.5]]), fill_holes=True, label="right tile")]}

# --- 625765: bowl without the brick it stands on
R["625765"] = {"objects": [obj(box([0.06, 0.17, 0.92, 0.79], pos=[[0.5, 0.4], [0.5, 0.7]], neg=[[0.1, 0.78], [0.9, 0.76], [0.5, 0.9]]),
                               fill_holes=True, largest=True, label="bowl without brick")]}
# --- 626947: plate without the dark shadow at the top
R["626947"] = {"objects": [obj(box([0.01, 0.05, 0.98, 0.98], pos=[[0.5, 0.5], [0.3, 0.75], [0.8, 0.3]], neg=[[0.3, 0.02], [0.12, 0.1], [0.97, 0.02]]),
                               fill_holes=True, largest=True, label="plate without shadow")]}
# --- 632586: vessel close-up without the grey background above it
R["632586"] = {"objects": [obj(box([0.0, 0.17, 1.0, 1.0], pos=[[0.5, 0.65], [0.5, 0.3], [0.2, 0.45]], neg=[[0.5, 0.06], [0.1, 0.13], [0.9, 0.13]]),
                               fill_holes=True, label="vessel without background")]}
# --- 635956: the stacked tile as one full square (1 was only its lower part)
R["635956"] = {"objects": [obj({"keep": [1]}, box([0.14, 0.12, 0.91, 0.86], pos=[[0.5, 0.6], [0.5, 0.38], [0.5, 0.2]], neg=[[0.05, 0.5], [0.95, 0.3]],
                                                  pick="largest"), fill_holes=True, largest=True, label="full square tile")]}

# --- 636194: test tile close-up: one object = the whole tile below its top edge
R["636194"] = {"objects": [obj({"rect": [0.0, 0.095, 1.0, 1.0]}, label="whole test tile (close-up)")]}
# --- 638784: the brown test tile standing on the lid of the glaze tub
R["638784"] = {"objects": [obj(box([0.34, 0.19, 0.63, 0.47], pos=[[0.48, 0.3], [0.45, 0.42]], neg=[[0.5, 0.5], [0.3, 0.45]]),
                               fill_holes=True, label="test tile on the lid")]}
# --- 639300: object 1 was the table -> the small (fourth) cup on the right; vases 2-4 unchanged
R["639300"] = {"objects": [obj(box([0.68, 0.6, 0.89, 0.9], pos=[[0.78, 0.74], [0.75, 0.84]], neg=[[0.62, 0.8], [0.95, 0.75]]),
                               fill_holes=True, label="fourth (small) cup")] + same("639300", [2, 3, 4])}
# --- 643417: only the oval tile, not the carpet
R["643417"] = {"objects": [obj(box([0.38, 0.4, 0.71, 0.82], pos=[[0.55, 0.6], [0.5, 0.78]], neg=[[0.2, 0.3], [0.85, 0.85]]),
                               fill_holes=True, label="oval tile")]}

# --- 645408 / 645412 / 645414: the tile only, grey background removed
R["645408"] = {"objects": [obj(box([0.2, 0.19, 0.8, 1.0], pos=[[0.45, 0.6], [0.4, 0.3], [0.6, 0.9]], neg=[[0.1, 0.5], [0.9, 0.5], [0.5, 0.1]]),
                               fill_holes=True, label="tile without grey background")]}
R["645412"] = {"objects": [obj(box([0.17, 0.19, 0.76, 1.0], pos=[[0.5, 0.4], [0.5, 0.8]], neg=[[0.08, 0.5], [0.9, 0.5], [0.5, 0.1]]),
                               fill_holes=True, label="tile without grey background")]}
R["645414"] = {"objects": [obj(box([0.16, 0.15, 0.74, 1.0], pos=[[0.48, 0.4], [0.48, 0.85]], neg=[[0.08, 0.5], [0.85, 0.5], [0.5, 0.08]]),
                               fill_holes=True, label="tile without grey background")]}
# --- 648344: the two tiles held in the hand, without the grey background and fingers, inner area restored
R["648344"] = {"objects": [obj(box([0.0, 0.0, 0.48, 0.85], pos=[[0.2, 0.4], [0.3, 0.6]], neg=[[0.08, 0.95], [0.7, 0.04]]),
                               box([0.46, 0.07, 0.98, 0.93], pos=[[0.7, 0.45], [0.62, 0.75]], neg=[[0.62, 0.98], [0.25, 0.4]]),
                               fill_holes=True, label="both tiles without background")]}

# --- 651519: bowl (1) and its centre (2) are one bowl
R["651519"] = {"objects": [keep(1, 2, fill_holes=True, label="whole bowl (1 + 2)")]}
# --- 653002: 2 and 3 are one cup (seen from below); cup 1 unchanged
R["653002"] = {"objects": [keep(2, 3, fill_holes=True, label="whole cup (2 + 3)"), keep(1, label="unchanged cup")]}
# --- "combine 2 image into 1": bowl series where the bowl's centre was cut out as a second object
for _s in ["655937", "655962", "656403", "658949", "658954", "658960", "658961", "658967", "658969",
           "659271", "659277", "659279", "659287", "659297", "659308", "681601"]:
    R[_s] = {"objects": [keep(1, 2, fill_holes=True, label="whole bowl (1 + 2)")]}

# --- 662430: glaze close-up, no smaller segmentation: the whole photo
R["662430"] = {"objects": [obj({"full": True}, label="whole photo (glaze close-up)")]}
# --- 665477: round tile and the rectangular tile behind it -> 2 objects
R["665477"] = {"objects": [obj(box([0.23, 0.5, 0.63, 0.96], pos=[[0.43, 0.72]], neg=[[0.4, 0.4], [0.7, 0.8]]), fill_holes=False, label="round tile"),
                           obj(box([0.27, 0.07, 0.6, 0.56], pos=[[0.45, 0.3], [0.38, 0.45]], neg=[[0.43, 0.7], [0.15, 0.2]]), label="rectangular tile")]}
# --- 669320 / 669327: two tiles side by side -> 2 objects
R["669320"] = {"objects": [obj(box([0.06, 0.08, 0.555, 1.0], pos=[[0.3, 0.5], [0.3, 0.85]], neg=[[0.75, 0.5]]), within=[1], label="left tile"),
                           obj(box([0.5, 0.0, 0.96, 1.0], pos=[[0.75, 0.4], [0.7, 0.85]], neg=[[0.3, 0.5]]), within=[1], label="right tile")]}

R["669327"] = {"objects": [obj(box([0.05, 0.05, 0.535, 1.0], pos=[[0.3, 0.4], [0.3, 0.85]], neg=[[0.75, 0.4]]), within=[1], label="left tile"),
                           obj(box([0.515, 0.1, 1.0, 1.0], pos=[[0.75, 0.4], [0.8, 0.85]], neg=[[0.3, 0.4]]), within=[1], label="right tile")]}
# --- 673339: 4 and 5 are one glazed tile; tile 1 unchanged (2, 3 not needed)
R["673339"] = {"objects": [keep(4, 5, fill_holes=True, label="left tile (4 + 5)"), keep(1, label="unchanged right tile")]}
# --- 676705: cup without the newspaper under it
R["676705"] = {"objects": [obj(box([0.3, 0.2, 0.93, 0.7], pos=[[0.45, 0.5], [0.4, 0.3], [0.86, 0.36]], neg=[[0.15, 0.5], [0.6, 0.85], [0.95, 0.6]]),
                               label="cup without newspaper")]}
# --- 677525: glaze close-up, no smaller segmentation: the whole photo
R["677525"] = {"objects": [obj({"full": True}, label="whole photo (glaze close-up)")]}

# --- 680726: three mushroom sculptures -> 3 objects
R["680726"] = {"objects": [obj(box([0.0, 0.3, 0.35, 0.84], pos=[[0.17, 0.42], [0.12, 0.72]], neg=[[0.48, 0.6]]), within=[1], label="left mushroom"),
                           obj(box([0.32, 0.33, 0.66, 0.84], pos=[[0.5, 0.42], [0.45, 0.73]], neg=[[0.17, 0.6], [0.8, 0.6]]), within=[1],
                               label="middle mushroom"),
                           obj(box([0.62, 0.22, 1.0, 0.86], pos=[[0.85, 0.35], [0.82, 0.72]], neg=[[0.5, 0.6]]), within=[1], label="right mushroom")]}
# --- 683928: the middle tile on the wall was missed (3 tiles, not 2)
R["683928"] = {"objects": same("683928", [1, 2])
               + [obj(box([0.17, 0.33, 0.77, 0.79], pos=[[0.4, 0.55], [0.55, 0.75]], neg=[[0.05, 0.5], [0.85, 0.5]]), fill_holes=True,
                      label="middle tile (was missed)")]}
# --- 684330: 2 and 3 are one dish; dish 1 unchanged
R["684330"] = {"objects": [keep(2, 3, fill_holes=True, label="upper dish (2 + 3)"), keep(1, label="unchanged lower dish")]}

# --- 684411: 2, 3, 4 were stamps on the left tile -> the whole left tile; right tile unchanged
R["684411"] = {"objects": [obj(box([0.08, 0.18, 0.44, 0.85], pos=[[0.25, 0.35], [0.25, 0.65], [0.25, 0.8]], neg=[[0.55, 0.5]]), fill_holes=True,
                               label="whole left tile"), keep(1, label="unchanged right tile")]}
# --- 686742: the whole wavy tile (the mask had cut off the upper half)
R["686742"] = {"objects": [obj({"keep": [1]}, box([0.03, 0.08, 0.98, 0.92], pos=[[0.5, 0.25], [0.5, 0.45], [0.5, 0.7]], neg=[[0.5, 0.97], [0.05, 0.03]],
                                                  pick="largest"), fill_holes=True, largest=True, label="whole tile")]}
# --- 687000: the left tile as one object (2 + 3 + the yellow middle); right tile unchanged
R["687000"] = {"objects": [obj({"keep": [2, 3]}, box([0.0, 0.0, 0.48, 1.0], pos=[[0.25, 0.4], [0.25, 0.15], [0.25, 0.8]], neg=[[0.75, 0.5]]),
                               fill_holes=True, label="whole left tile"), keep(1, label="unchanged right tile")]}
# --- 687151: 3, 4, 5 are one tile; tiles 1 and 2 unchanged (6 not needed)
R["687151"] = {"objects": [keep(3, 4, 5, fill_holes=True, label="middle tile (3 + 4 + 5)")] + same("687151", [1, 2])}

# --- 687183 / 687209 / 687210 / 687300: white tile without the black stand at the bottom
_brick = [[0.08, 0.9], [0.92, 0.9], [0.5, 0.96]]
R["687183"] = {"objects": [obj(box([0.11, 0.02, 0.89, 0.82], pos=[[0.5, 0.4], [0.5, 0.75]], neg=_brick + [[0.06, 0.66]]), fill_holes=True,
                               label="tile without the black stand")]}
R["687209"] = {"objects": [obj(box([0.17, 0.03, 0.84, 0.83], pos=[[0.5, 0.4], [0.5, 0.75]], neg=_brick + [[0.12, 0.6]]), fill_holes=True,
                               label="tile without the black stand")]}
R["687210"] = {"objects": [obj(box([0.15, 0.02, 0.89, 0.85], pos=[[0.5, 0.4], [0.5, 0.75]], neg=_brick + [[0.08, 0.75]]), fill_holes=True,
                               label="tile without the black stand")]}
R["687300"] = {"objects": [obj(box([0.15, 0.04, 0.85, 0.77], pos=[[0.5, 0.4], [0.5, 0.7]], neg=_brick + [[0.1, 0.6]]), fill_holes=True,
                               label="tile without the black stand")]}

R["687302"] = {"objects": [obj(box([0.13, 0.04, 0.85, 0.85], pos=[[0.5, 0.4], [0.5, 0.75]], neg=_brick + [[0.08, 0.55]]), fill_holes=True,
                               label="tile without the black stand")]}
# --- 687444: 1, 2, 3 are one tile (overall object)
R["687444"] = {"objects": [keep(1, 2, 3, fill_holes=True, label="whole tile (1 + 2 + 3)")]}
# --- 689166: sculpture without the brown table on the left
R["689166"] = {"objects": [obj(box([0.27, 0.0, 1.0, 1.0], pos=[[0.65, 0.3], [0.45, 0.55], [0.72, 0.8]], neg=[[0.13, 0.5], [0.36, 0.12], [0.27, 0.9]]),
                               box([0.3, 0.42, 0.82, 0.7], pos=[[0.5, 0.52], [0.38, 0.6]], neg=[[0.2, 0.5], [0.4, 0.75]]),
                               fill_holes=True, largest=True, label="sculpture with its white arm, without table")]}
# --- 690416: glaze close-up: the whole photo
R["690416"] = {"objects": [obj({"full": True}, label="whole photo (glaze close-up)")]}

# --- 692422: the cup on the left as one object (4, 5, 6 were parts of it); the three tiles unchanged
R["692422"] = {"objects": [obj({"keep": [4, 5, 6]}, box([0.04, 0.05, 0.37, 0.93], pos=[[0.2, 0.4], [0.2, 0.75]], neg=[[0.45, 0.5], [0.6, 0.12]]),
                               fill_holes=True, label="whole cup")] + same("692422", [1, 2, 3])}
# --- 696953: 1 and 2 are one tile, inner pixels restored
R["696953"] = {"objects": [obj({"keep": [1, 2]}, box([0.05, 0.2, 0.92, 0.82], pos=[[0.5, 0.4], [0.5, 0.65], [0.2, 0.55]], neg=[[0.5, 0.9]]),
                               fill_holes=True, label="whole tile (1 + 2)")]}
# --- 697727: only the X-shaped tile (the mask had taken the wooden board)
R["697727"] = {"objects": [obj(box([0.2, 0.27, 0.71, 0.66], pos=[[0.47, 0.42], [0.3, 0.32], [0.62, 0.6]], neg=[[0.1, 0.2], [0.6, 0.72], [0.15, 0.6]]),
                               fill_holes=True, label="X-shaped tile")]}
# --- 698729: the whole tile (1 + 2 + white foot), without the wooden table
R["698729"] = {"objects": [obj({"keep": [1, 2]}, box([0.18, 0.16, 0.93, 0.73], pos=[[0.5, 0.3], [0.4, 0.62], [0.8, 0.6]], neg=[[0.1, 0.8], [0.5, 0.08]]),
                               fill_holes=True, label="whole tile")]}

# --- triaxial tiles standing on glaze buckets: only the tile on top, not the bucket below
def upper_tile(b):
    cx = (b[0] + b[2]) / 2
    return {"objects": [obj(box(b, pos=[[cx, b[1] + 0.35 * (b[3] - b[1])], [cx, b[1] + 0.75 * (b[3] - b[1])]],
                                neg=[[cx, b[3] + 0.12], [b[0] - 0.12, b[3] - 0.02], [b[2] + 0.12, b[3] - 0.02]]),
                            clip=[b[0] - 0.01, b[1] - 0.01, b[2] + 0.01, b[3] + 0.01], fill_holes=True, largest=True,
                            label="test tile on top of the bucket")]}


def holder_tile(b):
    """Tile clamped in a white sample holder: the speckled top band is a different colour, so prompt the top,
    middle and bottom of the tile and take SAM's largest candidate inside the tile box."""
    cx, h = (b[0] + b[2]) / 2, b[3] - b[1]
    return {"objects": [obj(box(b, pos=[[cx, b[1] + 0.12 * h], [cx, b[1] + 0.5 * h], [cx, b[1] + 0.85 * h]],
                                neg=[[cx, b[3] + 0.12]], pick="largest"),
                            clip=[b[0] - 0.01, b[1] - 0.01, b[2] + 0.01, b[3] + 0.01], fill_holes=True, largest=True,
                            label="test tile without the sample holder")]}


R["698730"] = upper_tile([0.335, 0.27, 0.635, 0.555])
R["698733"] = upper_tile([0.35, 0.235, 0.625, 0.5])
R["698735"] = upper_tile([0.335, 0.28, 0.61, 0.535])
R["698736"] = upper_tile([0.36, 0.265, 0.635, 0.525])
R["702315"] = upper_tile([0.28, 0.095, 0.665, 0.665])
R["702319"] = upper_tile([0.31, 0.105, 0.63, 0.795])
# --- 702320: object 2 is exactly the tile; object 1 was the bucket lid
R["702320"] = {"objects": [keep(2, label="test tile (the lid removed)")]}
# --- 702439: two tiles on the bucket -> 2 objects, bucket excluded
R["702439"] = {"objects": [upper_tile([0.07, 0.15, 0.455, 0.415])["objects"][0] | {"label": "left tile"},
                           upper_tile([0.465, 0.145, 0.795, 0.41])["objects"][0] | {"label": "right tile"}]}

R["702585"] = upper_tile([0.31, 0.175, 0.665, 0.43])
# --- 702671: the front test tile without the paper underneath (other tiles not needed)
R["702671"] = {"objects": [obj(box([0.32, 0.14, 0.63, 0.82], pos=[[0.48, 0.25], [0.42, 0.65]], neg=[[0.72, 0.7], [0.18, 0.62], [0.42, 0.9]]),
                               fill_holes=True, label="tile without paper")]}
# --- 703110 / 703114: the front tile only, not the second tile behind it
R["703110"] = {"objects": [obj(box([0.16, 0.2, 0.75, 0.81], pos=[[0.45, 0.5], [0.3, 0.7]], neg=[[0.79, 0.4], [0.4, 0.18], [0.78, 0.55]]),
                               fill_holes=True, label="front tile")]}
R["703114"] = {"objects": [obj(box([0.33, 0.33, 0.85, 0.73], pos=[[0.6, 0.55], [0.75, 0.4]], neg=[[0.3, 0.45], [0.45, 0.315], [0.31, 0.6]]),
                               fill_holes=True, label="front tile")]}

# --- 703677: 1, 2, 3 are one cup
R["703677"] = {"objects": [keep(1, 2, 3, fill_holes=True, label="whole cup (1 + 2 + 3)")]}
# --- 704116 / 704132 ("only get one object"): the photo shows two tiles and only the upper one was found
R["704116"] = {"objects": [keep(1, label="unchanged upper tile"),
                           obj(box([0.18, 0.47, 1.0, 0.86], pos=[[0.4, 0.63], [0.8, 0.63]], neg=[[0.5, 0.3], [0.1, 0.65]]), label="lower tile (was missed)")]}
R["704132"] = {"objects": [keep(1, label="unchanged upper tile"),
                           obj(box([0.06, 0.44, 1.0, 0.83], pos=[[0.3, 0.63], [0.8, 0.63]], neg=[[0.4, 0.3], [0.5, 0.92]]), label="lower tile (was missed)")]}
# --- 704142: upper and lower tile had been merged -> 2 objects
R["704142"] = {"objects": [obj(box([0.18, 0.04, 1.0, 0.49], pos=[[0.4, 0.25], [0.8, 0.25]], neg=[[0.5, 0.7]]), label="upper tile"),
                           obj(box([0.2, 0.47, 1.0, 0.88], pos=[[0.45, 0.68], [0.85, 0.68]], neg=[[0.5, 0.3]]), label="lower tile")]}

# --- 704204: object 1 was the middle tile + the plate under it -> the middle tile only; tiles 2, 3 unchanged
R["704204"] = {"objects": [obj(box([0.32, 0.355, 0.58, 0.68], pos=[[0.47, 0.45], [0.42, 0.62]], neg=[[0.47, 0.8], [0.3, 0.75]]), fill_holes=True,
                               label="middle tile without the plate")] + same("704204", [2, 3])}
# --- test tiles standing on white lids ("upper part, ignore the cup below" / "the ceramic is ignored"): the tile only
R["705250"] = upper_tile([0.37, 0.0, 0.605, 0.815])
R["705270"] = upper_tile([0.33, 0.04, 0.605, 0.92])
R["705274"] = upper_tile([0.395, 0.13, 0.62, 0.935])
R["705275"] = upper_tile([0.375, 0.075, 0.615, 0.845])
R["705276"] = upper_tile([0.395, 0.015, 0.655, 0.8])
R["705277"] = upper_tile([0.34, 0.0, 0.64, 0.905])
# --- 705820: object 1 held two tiles + the wooden board -> the two tiles only; 3 and 4 unchanged
R["705820"] = {"objects": [obj(box([0.34, 0.31, 0.48, 0.61], pos=[[0.4, 0.4], [0.41, 0.56]], neg=[[0.4, 0.7], [0.5, 0.25]]), fill_holes=True,
                               label="small tile"),
                           obj(box([0.66, 0.21, 0.85, 0.61], pos=[[0.74, 0.35], [0.75, 0.55]], neg=[[0.9, 0.35], [0.75, 0.7]]), fill_holes=True,
                               label="right tile")] + same("705820", [3, 4])}

# --- 706221: tile without the grey stand
R["706221"] = {"objects": [obj(box([0.22, 0.04, 0.79, 0.82], pos=[[0.5, 0.4], [0.45, 0.75]], neg=_brick + [[0.12, 0.5]]), fill_holes=True,
                               label="tile without the grey stand")]}
# --- 706225: 2 to 6 are one tile (the stand, 1, not needed)
R["706225"] = {"objects": [keep(2, 3, 4, 5, 6, fill_holes=True, label="whole tile (2 to 6)")]}
# --- sample holder series: the test tile without the white holder clamped below it
R["707671"] = holder_tile([0.3, 0.14, 0.67, 0.6])
R["707673"] = holder_tile([0.28, 0.08, 0.76, 0.68])
R["707674"] = holder_tile([0.28, 0.07, 0.72, 0.6])
R["707675"] = holder_tile([0.29, 0.04, 0.74, 0.67])
R["707676"] = holder_tile([0.27, 0.04, 0.73, 0.61])
R["707678"] = holder_tile([0.28, 0.04, 0.71, 0.61])
R["707679"] = holder_tile([0.28, 0.07, 0.72, 0.6])
R["707680"] = holder_tile([0.29, 0.085, 0.74, 0.63])
R["707681"] = holder_tile([0.26, 0.06, 0.77, 0.62])
R["707682"] = holder_tile([0.29, 0.04, 0.77, 0.63])

# --- 708172: two broken tile pieces -> 2 objects (the 10 small parts were inside them)
R["708172"] = {"objects": [obj(box([0.19, 0.03, 0.76, 0.5], pos=[[0.3, 0.2], [0.6, 0.3]], neg=[[0.5, 0.7], [0.9, 0.2]]), fill_holes=True,
                               label="upper tile piece"),
                           obj(box([0.21, 0.5, 0.74, 0.94], pos=[[0.35, 0.7], [0.65, 0.7]], neg=[[0.5, 0.3], [0.9, 0.7]]), fill_holes=True,
                               label="lower tile piece")]}
# --- glaze discs lying on the recipe sheet: the disc only, not the paper behind
R["708598"] = {"objects": [obj(box([0.07, 0.03, 0.8, 0.94], pos=[[0.45, 0.45], [0.25, 0.3], [0.6, 0.75]], neg=[[0.9, 0.5], [0.03, 0.06], [0.04, 0.9], [0.92, 0.93]]),
                               fill_holes=True, label="glaze disc without paper")]}
R["708663"] = {"objects": [obj(box([0.02, 0.11, 0.76, 0.96], pos=[[0.4, 0.5], [0.2, 0.75]], neg=[[0.9, 0.3], [0.85, 0.85], [0.05, 0.05], [0.4, 0.04]]),
                               fill_holes=True, label="glaze disc without paper")]}

R["708667"] = {"objects": [obj(box([0.07, 0.03, 0.74, 0.98], pos=[[0.45, 0.5], [0.3, 0.3], [0.5, 0.85]], neg=[[0.9, 0.3], [0.03, 0.5], [0.88, 0.88], [0.08, 0.95]]),
                               fill_holes=True, label="glaze disc without paper")]}
R["708683"] = {"objects": [obj(box([0.06, 0.12, 0.8, 1.0], pos=[[0.45, 0.5], [0.3, 0.3], [0.5, 0.85]], neg=[[0.9, 0.3], [0.05, 0.05], [0.9, 0.9], [0.05, 0.9]]),
                               fill_holes=True, label="glaze disc without paper")]}
# --- 709306: 1 and 2 are one vessel
R["709306"] = {"objects": [keep(1, 2, fill_holes=True, label="whole vessel (1 + 2)")]}
# --- 713742: the cup / bowl without the stone plate under it; bowl 2 unchanged
R["713742"] = {"objects": [obj(box([0.24, 0.34, 0.81, 0.87], pos=[[0.5, 0.65], [0.35, 0.55], [0.7, 0.7], [0.5, 0.385]], neg=[[0.15, 0.6], [0.5, 0.95], [0.5, 0.25]]),
                               fill_holes=True, label="cup / bowl without plate"), keep(2, label="unchanged bowl")]}
# --- 715760: 2 and 3 are one tile; tile 1 unchanged
R["715760"] = {"objects": [keep(2, 3, fill_holes=True, label="right tile (2 + 3)"), keep(1, label="unchanged left tile")]}

# --- 718747 / 718752: tile without the fabric background, the pen and the printed text
R["718747"] = {"objects": [obj(box([0.19, 0.2, 0.87, 0.82], pos=[[0.5, 0.4], [0.4, 0.7]], neg=[[0.25, 0.9], [0.93, 0.5], [0.5, 0.1], [0.12, 0.5]]),
                               fill_holes=True, label="tile without background")]}
R["718752"] = {"objects": [obj(box([0.17, 0.2, 0.85, 0.81], pos=[[0.5, 0.4], [0.45, 0.7]], neg=[[0.16, 0.72], [0.92, 0.55], [0.5, 0.05], [0.5, 0.92]]),
                               fill_holes=True, label="tile without background")]}
# --- 719314: right tile without the paper; cup 2 unchanged
R["719314"] = {"objects": [obj(box([0.6, 0.28, 0.92, 0.69], pos=[[0.75, 0.55], [0.75, 0.33]], neg=[[0.52, 0.3], [0.96, 0.5], [0.75, 0.75]]),
                               fill_holes=True, label="right tile without paper"), keep(2, label="unchanged cup")]}
# --- 753848: the raku vessel, background excluded and its surface complete
R["753848"] = {"objects": [obj(box([0.24, 0.02, 1.0, 0.93], pos=[[0.6, 0.5], [0.45, 0.35], [0.8, 0.3], [0.5, 0.8], [0.35, 0.65]],
                                   neg=[[0.08, 0.4], [0.15, 0.85], [0.6, 0.98], [0.05, 0.1]]),
                               box([0.28, 0.0, 1.0, 0.32], pos=[[0.4, 0.2], [0.55, 0.15], [0.85, 0.1]], neg=[[0.1, 0.2]], pick="largest"),
                               clip=[0.27, 0.0, 1.0, 0.95], fill_holes=True, label="whole raku vessel (top included)")]}

# --- 766949: star and coil separated, paper removed; cup 2 unchanged
R["766949"] = {"objects": [obj(box([0.1, 0.14, 0.82, 0.59], pos=[[0.45, 0.4], [0.25, 0.3]], neg=[[0.7, 0.5], [0.5, 0.65]]), fill_holes=False, label="star"),
                           obj(box([0.2, 0.59, 0.88, 0.86], pos=[[0.4, 0.76], [0.7, 0.67]], neg=[[0.5, 0.55], [0.3, 0.68]]), label="coil"),
                           keep(2, label="unchanged cup")]}
# --- 766952: every clay piece on the sheet as its own object, paper removed (2, 3, 4 unchanged)
R["766952"] = {"objects": [obj(box([0.13, 0.09, 0.35, 0.28], pos=[[0.2, 0.2]], neg=[[0.4, 0.35]]), label="small cup (top left)"),
                           obj(box([0.42, 0.14, 0.54, 0.24], pos=[[0.48, 0.19]], neg=[[0.48, 0.3]]), label="small ball"),
                           obj(box([0.29, 0.05, 0.85, 0.24], pos=[[0.4, 0.08], [0.75, 0.17]], neg=[[0.48, 0.19], [0.6, 0.3]]), label="thin coil (top)"),
                           obj(box([0.59, 0.47, 0.8, 0.64], pos=[[0.69, 0.56]], neg=[[0.69, 0.43]]), label="ball"),
                           obj(box([0.4, 0.6, 0.66, 0.81], pos=[[0.45, 0.7], [0.6, 0.72]], neg=[[0.53, 0.7]]), fill_holes=False, label="ring")]
              + same("766952", [2, 3, 4])}
# --- 770522: the bowl without the glove
R["770522"] = {"objects": [obj(box([0.0, 0.3, 0.87, 0.7], pos=[[0.4, 0.42], [0.2, 0.4], [0.5, 0.6]], neg=[[0.75, 0.6], [0.3, 0.85], [0.85, 0.5], [0.15, 0.66]]),
                               fill_holes=True, label="bowl without glove")]}
# --- 777567: the tile without the wooden table
R["777567"] = {"objects": [obj(box([0.41, 0.3, 0.68, 0.61], pos=[[0.55, 0.42], [0.55, 0.56]], neg=[[0.3, 0.3], [0.8, 0.75], [0.55, 0.7]]),
                               fill_holes=True, label="tile without table")]}

# --- 782345: the sculpture, background removed and inner pixels filled
R["782345"] = {"objects": [obj(box([0.11, 0.0, 0.86, 1.0], pos=[[0.45, 0.5], [0.35, 0.2], [0.55, 0.8], [0.25, 0.6]],
                                   neg=[[0.05, 0.3], [0.94, 0.5], [0.86, 0.08], [0.05, 0.95]]), fill_holes=True, label="sculpture")]}
# --- 787364: the whole crawl-glazed arch, including its blurred left leg
R["787364"] = {"objects": [obj({"keep": [1]}, box([0.08, 0.25, 0.64, 1.0], pos=[[0.45, 0.32], [0.15, 0.6], [0.28, 0.85], [0.5, 0.8]],
                                                  neg=[[0.8, 0.5], [0.05, 0.2]], pick="largest"), fill_holes=True, label="whole arch")]}
# --- 790690: the two test tiles as 2 objects (the 7 pieces were parts of them)
R["790690"] = {"objects": [obj(box([0.07, 0.12, 0.38, 0.93], pos=[[0.22, 0.3], [0.2, 0.7]], neg=[[0.45, 0.5], [0.22, 0.97]]), fill_holes=True,
                               label="left tile"),
                           obj(box([0.59, 0.11, 0.97, 0.94], pos=[[0.75, 0.3], [0.78, 0.75]], neg=[[0.5, 0.5], [0.75, 0.98]]), fill_holes=True,
                               label="right tile")]}
# --- 794691: the glazed upper part of the tile (the mask had only the bare foot)
R["794691"] = {"objects": [obj(box([0.03, 0.06, 0.96, 0.63], pos=[[0.5, 0.3], [0.5, 0.5], [0.2, 0.15]], neg=[[0.5, 0.8]]),
                               clip=[0.0, 0.0, 1.0, 0.63], fill_holes=True, label="glazed upper part")]}

# --- 795189: the whole right tile (2 and 3 were only its two glaze patches); left tile unchanged
R["795189"] = {"objects": [obj({"keep": [2, 3]}, box([0.57, 0.17, 0.92, 0.81], pos=[[0.75, 0.3], [0.75, 0.7], [0.62, 0.5]], neg=[[0.5, 0.5], [0.97, 0.5]]),
                               fill_holes=True, label="whole right tile"), keep(1, label="unchanged left tile")]}
# --- 802555 (ambiguous note "ignore the hand ceramic part"): the tile held in the hand is added without the fingers;
#     the six tiles around it stay as they were
R["802555"] = {"objects": [obj(box([0.28, 0.33, 0.77, 0.72], pos=[[0.5, 0.5], [0.72, 0.5], [0.4, 0.65]], neg=[[0.2, 0.5], [0.45, 0.8], [0.25, 0.36]]),
                               fill_holes=True, label="tile held in the hand (fingers excluded)")] + same("802555", [1, 2, 3, 4, 5, 6])}

# --- 27838: a composite of 70 tile photos (10 columns x 7 rows) -> one object per tile ("deeper segmentation")
R["27838"] = {"objects": [obj({"rect": [c / 10 + 0.004, r / 7 + 0.004, (c + 1) / 10 - 0.004, (r + 1) / 7 - 0.004]},
                              label=f"tile row {r + 1} column {c + 1}") for r in range(7) for c in range(10)]}
# --- 27888: object 1 had merged all the tiles at the back -> each back tile separately; tiles 2-12, 14, 15 unchanged
R["27888"] = {"objects": [{"auto_split": {"within": [1], "step": 0.012, "min_area": 0.02, "max_area": 0.3, "min_iou": 0.75},
                           "label": "back tile"}] + same("27888", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15])}


if __name__ == "__main__":
    out = Path(__file__).with_name("manual_corrections.json")
    out.write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(f"{len(R)} photos -> {out}")
