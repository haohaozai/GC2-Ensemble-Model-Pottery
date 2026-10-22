"""Colour words -> colours, and colour PHRASES -> a region-aware colour query (Color Search Lab V3 integration).

Reused from CG2_Ceramic_Color_Search_Lab_V3/index.html (read-only, parsed at start-up): the 126 named colours
(name | hex | family | commonness) and the alias table; the same text normalisation (grey -> gray) and character
2/3/4-gram TF-IDF matching. Added: the 98 reference shades of the segmentation pipeline (glazeprep/appearance.py,
the names used in the review CSV), modifiers (light / dark / pale / deep, "-ish") and position words, so that
  "beige tile with reddish-brown bottom"
becomes regions  [beige (main, top / whole object), reddish brown (bottom, important)].
Colour values are kept as HEX and CIELAB; matching uses CIEDE2000 (mm/balanced.py).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

# the Color Lab V3 page ships with the project (third_party/color_lab_v3, from CG2_Ceramic_Color_Search_Lab_V3.zip)
LAB_HTML = Path(__file__).resolve().parents[2] / "third_party" / "color_lab_v3" / "CG2_Ceramic_Color_Search_Lab_V3" / "index.html"
POSITIONS = {"bottom": (0.5, 0.82), "foot": (0.5, 0.9), "base": (0.5, 0.88), "lower": (0.5, 0.75), "top": (0.5, 0.18),
             "upper": (0.5, 0.25), "rim": (0.5, 0.08), "lip": (0.5, 0.06), "left": (0.18, 0.5), "right": (0.82, 0.5),
             "center": (0.5, 0.5), "centre": (0.5, 0.5), "middle": (0.5, 0.5), "edge": (0.5, 0.5), "edges": (0.5, 0.5)}
SPOT_WORDS = ("spot", "spots", "speckle", "speckles", "speckled", "fleck", "flecks", "dots", "dotted")
MODIFIERS = {"light": (12, 0.7), "pale": (14, 0.6), "dark": (-14, 1.0), "deep": (-10, 1.25), "bright": (4, 1.3),
             "dull": (0, 0.6), "muted": (0, 0.6)}


def text_norm(s: str) -> str:
    s = s.lower().replace("grey", "gray")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def grams(s: str) -> list[str]:
    s = " " + text_norm(s) + " "
    return [s[i:i + n] for n in (2, 3, 4) for i in range(len(s) - n + 1)]


@lru_cache(1)
def vocabulary():
    """[{name, hex, family, source, phrases}] from Color Lab V3 + the pipeline reference shades."""
    items = []
    if LAB_HTML.exists():
        html = LAB_HTML.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"^([A-Za-z' -]+)\|(#[0-9A-Fa-f]{6})\|([a-z]+)\|(\d+)$", html, re.M):
            items.append({"name": m.group(1).strip(), "hex": m.group(2).upper(), "family": m.group(3),
                          "commonness": int(m.group(4)), "source": "Color Lab V3"})
        al = {}
        am = re.search(r"const ALIASES=\{(.*?)\};", html, re.S)
        if am:
            for k, v in re.findall(r"([a-z]+):\[([^\]]*)\]", am.group(1)):
                al[k] = re.findall(r"'([^']*)'", v)
        for it in items:
            it["phrases"] = [it["name"]] + al.get(text_norm(it["name"]).replace(" ", ""), [])
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from glazeprep.appearance import _REFERENCES
    have = {text_norm(i["name"]) for i in items}
    for fam, name, hx in _REFERENCES:
        if text_norm(name) not in have:
            items.append({"name": name, "hex": hx.upper(), "family": fam, "commonness": 5, "source": "pipeline reference",
                          "phrases": [name]})
    return items


@lru_cache(1)
def _index():
    voc = vocabulary()
    docs = [Counter(g for ph in it["phrases"] for g in grams(ph)) for it in voc]
    df = Counter(g for d in docs for g in d)
    N = len(docs)
    idf = {g: math.log((N + 1) / (c + 1)) + 1 for g, c in df.items()}

    def vec(c):
        v = {g: (1 + math.log(n)) * idf.get(g, 0) for g, n in c.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {g: x / nrm for g, x in v.items()}
    return voc, [vec(d) for d in docs], vec


def search_words(q: str, k=8) -> list[dict]:
    """exact name > exact alias > character 2-4-gram TF-IDF cosine (Color Lab V3's ranking)."""
    voc, vecs, vec = _index()
    qn = text_norm(q)
    exact = [i for i, it in enumerate(voc) if text_norm(it["name"]) == qn]
    alias = [i for i, it in enumerate(voc) if i not in exact and any(text_norm(a) == qn for a in it["phrases"][1:])]
    qv = vec(Counter(grams(q)))
    sc = [(sum(qv.get(g, 0) * w for g, w in v.items()), i) for i, v in enumerate(vecs)]
    sc.sort(reverse=True)
    order = [(2.0, i) for i in exact] + [(1.5, i) for i in sorted(alias, key=lambda i: -voc[i].get("commonness", 0))] +             [(s, i) for s, i in sc if i not in exact and i not in alias]
    return [{**{x: voc[i][x] for x in ("name", "hex", "family", "source")}, "score": round(s, 3)} for s, i in order[:k] if s > 0]


def hex2lab(hx: str) -> np.ndarray:
    from skimage.color import rgb2lab
    rgb = np.array([[[int(hx[i:i + 2], 16) for i in (1, 3, 5)]]], np.float64) / 255.0
    return rgb2lab(rgb)[0, 0]


def lab2hex(lab) -> str:
    from skimage.color import lab2rgb
    rgb = np.clip(lab2rgb(np.asarray(lab, np.float64).reshape(1, 1, 3))[0, 0], 0, 1)
    return "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgb)


def _colour_from_words(words: list[str]):
    """'light reddish brown' -> (lab, hex, explanation)"""
    mods = [w for w in words if w in MODIFIERS]
    base = " ".join(w for w in words if w not in MODIFIERS)
    base = re.sub(r"(\w+)ish\b", r"\1", base)          # reddish -> red, bluish -> blu(e)
    base = base.replace("blu ", "blue ").replace("purpl ", "purple ")
    if not base.strip():
        return None
    hits = search_words(base, 3)
    if not hits:
        return None
    lab = hex2lab(hits[0]["hex"]).copy()
    why = f"{' '.join(words)} -> {hits[0]['name']} ({hits[0]['source']})"
    # a two-word colour like "red brown" / "reddish brown": mix the two words' colours 1:2 (the head word dominates)
    parts = base.split()
    if len(parts) == 2 and not any(text_norm(h["name"]) == text_norm(base) for h in hits[:1]):
        a, b = search_words(parts[0], 1), search_words(parts[1], 1)
        if a and b:
            lab = (hex2lab(a[0]["hex"]) + 2 * hex2lab(b[0]["hex"])) / 3
            why = f"{' '.join(words)} -> 1/3 {a[0]['name']} + 2/3 {b[0]['name']} (CIELAB mix)"
    for m in mods:
        dl, cf = MODIFIERS[m]
        lab[0] = float(np.clip(lab[0] + dl, 0, 100))
        lab[1:] = lab[1:] * cf
    return lab, lab2hex(lab), why


def parse_phrase(text: str) -> dict:
    """free text -> {"regions": [{hex, lab, position, centroid, importance, words}], "spots": {...}|None, "notes": [...]}

    Splits on 'with', 'and', ',', '+', ';'. The first colour is the main colour (normal importance); colours given
    a position ("bottom", "rim", ...) or introduced by 'with' are marked important; spot words make a speckle query."""
    if not text_norm(text):
        return {"regions": [], "spots": None, "notes": []}
    chunks = [text_norm(c) for c in re.split(r"\b(?:with|and|plus)\b|[,+;]", text.lower()) if text_norm(c)]
    stop = {"a", "an", "the", "tile", "tiles", "glaze", "glazed", "ceramic", "pottery", "on", "at", "in", "of", "some",
            "small", "large", "big", "part", "section", "area", "band", "stripe", "region", "test", "piece", "colour", "color"}
    regions, notes, spots = [], [], None
    for k, ch in enumerate(chunks):
        words = [w for w in ch.split() if w not in stop]
        pos = [w for w in words if w in POSITIONS]
        is_spot = any(w in SPOT_WORDS for w in words)
        cw = [w for w in words if w not in POSITIONS and w not in SPOT_WORDS]
        c = _colour_from_words(cw)
        if c is None:
            if words:
                notes.append(f"no colour recognised in '{ch}'")
            continue
        lab, hx, why = c
        notes.append(why)
        if is_spot:
            spots = {"hex": hx, "lab": [round(float(v), 2) for v in lab], "words": ch}
            continue
        p = pos[0] if pos else None
        regions.append({"hex": hx, "lab": [round(float(v), 2) for v in lab], "position": p or ("main" if k == 0 else "anywhere"),
                        "centroid": list(POSITIONS[p]) if p else [0.5, 0.5], "use_position": bool(p),
                        "importance": "normal" if (k == 0 and not p) else "high", "words": ch})
    return {"regions": regions, "spots": spots, "notes": notes}


# ---------------------------------------------------------------- related colours and colour families (Color Lab V3)
@lru_cache(1)
def _vocab_lab() -> np.ndarray:
    return np.stack([hex2lab(v["hex"]) for v in vocabulary()]).astype(np.float64)


def nearest_vocab(labs: np.ndarray) -> np.ndarray:
    """index of the nearest vocabulary colour (CIEDE2000) for every Lab row ([N, 3] -> [N])."""
    from .balanced import de2000
    V = _vocab_lab()
    labs = np.asarray(labs, np.float64).reshape(-1, 3)
    out = np.empty(len(labs), int)
    for s in range(0, len(labs), 2048):
        out[s:s + 2048] = de2000(labs[s:s + 2048, None, :], V[None, :, :]).argmin(1)
    return out


def family_of(labs: np.ndarray) -> list[str]:
    voc = vocabulary()
    return [voc[i]["family"] for i in nearest_vocab(labs)]


def related(hx: str, mode: str = "both", common: float = 0.35, n: int = 18) -> dict:
    """Color Lab V3 'related colours': the colour's name (nearest reference, familiar names preferred by the common-name
    bias), then up to n neighbours - nearest only / same family only / both (within dE00 28 or the same family),
    ranked by dE00 - bias x commonness x 1.5 (V3's heuristic, not a measured frequency)."""
    from .balanced import de2000
    voc = vocabulary()
    lab = hex2lab(hx)
    d = de2000(lab[None, :], _vocab_lab())
    pop = np.array([v.get("commonness", 5) for v in voc], np.float64)
    lit = int(d.argmin())
    best = int(np.argmin(d - common * pop * 1.5 * (d <= d[lit] + 6)))      # familiar name only when nearly as close
    base = voc[best]
    keep = []
    for i in np.argsort(d):
        if i == best:
            continue
        same = voc[i]["family"] == base["family"]
        if mode == "near" or (mode == "family" and same) or (mode == "both" and (d[i] <= 28 or same)):
            keep.append(int(i))
    keep.sort(key=lambda i: d[i] - common * pop[i] * 1.5)
    item = lambda i: {"name": voc[i]["name"], "hex": voc[i]["hex"], "family": voc[i]["family"], "dE00": round(float(d[i]), 1),
                      "common": bool(pop[i] >= 9)}
    return {"hex": hx.upper(), "name": item(best), "literal": item(lit), "related": [item(i) for i in keep[:n]]}
