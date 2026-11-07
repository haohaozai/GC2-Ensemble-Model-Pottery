"""Free text -> glaze search values (card G, "Describe it"; POST /api/glaze/understand).

extract(text): the keywords and values a person writes, tolerant to typos, shorthand, word order and filler words:
  colour words (+ light / dark / -ish, Color Lab V3 + the pipeline's reference shades) -> HEX
  Orton cone or range (cone 6, ^6, c6, cone6, 5-6, cone 04, mid fire)  - a leading zero stays: 04 is low fire
  surface / transparency / atmosphere (the 7k vocabularies + spelling variants), glaze effects
  colourant amounts (2% red iron oxide, cobalt 1 %) and an oxide analysis (SiO2 64.2 Al2O3 13.1 ...)
Typos: a word that is no keyword is compared with the keyword lists (difflib ratio >= 0.8; 'cobolt' -> 'cobalt').
TagModel: TF-IDF (words + character 3-5 grams) + logistic regression trained on the recipes' own free-text
descriptions (Glazy users' words) -> probabilities of surface / transparency / atmosphere / colour family for what
the text implies but does not say ('celadon', 'tenmoku', 'shino'). noisy_variants(): the noisy sentences used to train
the joint model's text head and to test this module.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from functools import lru_cache

import numpy as np

from .external import CONE_ORDER

SURFACE = {"Semi-glossy": ["semi gloss", "semi glossy", "semigloss", "semi shiny"], "Satin-matte": ["satin matte", "satin matt"],
           "Semi-matte": ["semi matte", "semi matt", "semimatte"], "Smooth Matte": ["smooth matte", "smooth matt"],
           "Stony Matte": ["stony matte", "stony matt", "stony"], "Dry Matte": ["dry matte", "dry matt"],
           "Glossy": ["glossy", "gloss", "shiny", "shine", "glassy", "high gloss"], "Satin": ["satin", "silky"],
           "Matte": ["matte", "matt", "mat", "flat", "dull"]}
TRANSPARENCY = {"Semi-opaque": ["semi opaque", "semiopaque"], "Transparent": ["transparent", "see through", "clear"],
                "Translucent": ["translucent"], "Opaque": ["opaque"]}
ATMOSPHERE = {"Oxidation": ["oxidation", "oxidised", "oxidized", "oxidising", "oxidizing", "ox", "oxi", "electric kiln", "electric"],
              "Reduction": ["reduction", "reduced", "reducing", "redux", "gas reduction"], "Neutral": ["neutral"],
              "Salt & Soda": ["salt and soda", "salt fired", "soda fired", "salt", "soda"], "Wood": ["wood fired", "woodfired", "woodfire", "wood"],
              "Raku": ["raku"], "Luster": ["luster", "lustre"]}
EFFECTS = {"crackle": ["crackle", "crackled", "crazing", "crazed"], "speckled": ["speckled", "speckles", "speckle", "specks", "flecks", "flecked"],
           "crystalline": ["crystalline", "crystals", "crystal"], "flowing": ["runny", "runs", "running", "flowing", "fluid", "drippy", "drips"],
           "breaking at edges": ["breaks", "breaking"], "mottled": ["mottled", "variegated"], "oil spot": ["oil spot", "oilspot"],
           "metallic / iridescent": ["metallic", "iridescent"], "crawling": ["crawling", "crawl", "crawls"], "pinholes": ["pinholes", "pinholing"]}
COLOURANTS = {"red iron oxide": "Fe2O3", "iron oxide": "Fe2O3", "rio": "Fe2O3", "iron": "Fe2O3", "cobalt carbonate": "CoO",
              "cobalt oxide": "CoO", "cobalt": "CoO", "copper carbonate": "CuO", "copper oxide": "CuO", "copper": "CuO",
              "rutile": "TiO2", "titanium dioxide": "TiO2", "tin oxide": "SnO2", "tin": "SnO2", "zircopax": "ZrO2", "zircon": "ZrO2",
              "chrome oxide": "Cr2O3", "chrome": "Cr2O3", "manganese dioxide": "MnO2", "manganese": "MnO2", "nickel oxide": "NiO", "nickel": "NiO"}
OXIDES = "SiO2 Al2O3 CaO K2O Na2O Fe2O3 MgO TiO2 B2O3 P2O5 ZnO CuO CoO SrO Li2O ZrO2 SnO2 BaO Cr2O3 MnO MnO2 FeO NiO PbO F V2O5".split()
FIRE_WORDS = {"low fire": "06 - 04", "lowfire": "06 - 04", "earthenware": "06 - 04", "mid fire": "5 - 6", "midfire": "5 - 6",
              "mid range": "5 - 6", "midrange": "5 - 6", "high fire": "9 - 10", "highfire": "9 - 10"}
STOP = set("a an the and or with for of to in on at my me i we you want need looking look like something some kind kinda sort sorta "
           "please pls plz glaze glazes glazed recipe recipes color colour colors colours tone shade nice very really bit little "
           "that this is it its be have has want maybe would could fired firing fire kiln temp temperature mug mugs pot pots "
           "bowl bowls vase surface finish type cone cones about around approx liner liners base slip engobe underglaze "
           "overglaze stoneware porcelain clay body tile tiles test thick thin layer layers dip dipped brush brushed spray "
           "sprayed food safe functional durable inside outside interior exterior rim foot edge edges low high mid".split())
CONE_TOKEN = r"0?\d{1,2}(?:\s*1/2|\.5|½)?"
CONE_RE = re.compile(rf"(?:\bcones?|\bcnoe|\bcoen|\bcon|\bc|\^|δ|∆)\s*\.?\s*({CONE_TOKEN})(?:\s*(?:-|–|to|/)\s*(?:cones?|c|\^)?\s*({CONE_TOKEN}))?(?![\d%])")
AMOUNT = r"(\d+(?:[.,]\d+)?)\s*%?"


def norm(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text)).lower().replace("grey", "gray").replace("colour", "color")
    s = re.sub(r"(\w)\1{2,}", r"\1\1", s)                          # shinyyy -> shinyy (fuzzy matching does the rest)
    s = re.sub(r"[^\w\s%^.,/½δ∆+-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(1)
def colour_words() -> set[str]:
    from .colorlab import MODIFIERS, text_norm, vocabulary
    words = {w for it in vocabulary() for ph in it["phrases"] for w in text_norm(ph).split() if len(w) > 2}
    reserved = {k for d in (SURFACE, TRANSPARENCY, ATMOSPHERE, EFFECTS) for v in d.values() for p in v for k in p.split()}
    return (words | set(MODIFIERS)) - reserved - STOP - set(COLOURANTS) | {"cobalt", "copper"}


def _fuzzy(tok: str, choices, cutoff=0.8) -> str | None:
    if len(tok) < 3:
        return None
    hit = difflib.get_close_matches(tok, list(choices), n=1, cutoff=cutoff + (0.05 if len(tok) <= 4 else 0))
    return hit[0] if hit else None


def _orton(tok: str) -> str:
    tok = re.sub(r"\s+", " ", tok.replace("½", " 1/2")).strip()
    return re.sub(r"^(0?\d+)\.5$", r"\1 1/2", tok)


def extract(text: str) -> dict:
    """free text -> {colour, colours, cone, surface, transparency, atmospheres, effects, colourants, oxides, corrections, unused}"""
    raw = str(text or "")[:500]
    s = norm(raw)
    out = {"text": raw, "colour": None, "colours": [], "cone": None, "surface": None, "transparency": None, "atmospheres": [],
           "effects": [], "colourants": [], "oxides": {}, "corrections": [], "unused": []}
    used = [False] * len(s)

    def take(m_start, m_end):
        for i in range(m_start, m_end):
            used[i] = True

    # oxide analysis (formula tokens with a number): needs >= 3 oxides to count as an analysis
    ox = {}
    for m in re.finditer(r"\b(" + "|".join(sorted(OXIDES, key=len, reverse=True)) + r")\s*[:=]?\s*" + AMOUNT,
                         unicodedata.normalize("NFKC", raw), re.I):              # NFKC: SiO₂ -> SiO2, Al₂O₃ -> Al2O3
        name = next(o for o in OXIDES if o.lower() == m.group(1).lower())
        ox[name] = float(m.group(2).replace(",", "."))
    if len(ox) >= 3:
        out["oxides"] = ox
        for m in re.finditer(r"\b(" + "|".join(OXIDES) + r")\s*[:=]?\s*\d+(?:[.,]\d+)?\s*%?", s, re.I):
            take(*m.span())
    # cone first (so "cone 6 cobalt" is never read as 6 % cobalt): explicit cone numbers, then fire words
    for m in CONE_RE.finditer(s):
        if any(used[m.start():m.end()]):
            continue
        a, b = _orton(m.group(1)), _orton(m.group(2)) if m.group(2) else None
        if a in CONE_ORDER and (b is None or b in CONE_ORDER):
            lo, hi = sorted({a, b or a}, key=CONE_ORDER.index) if b else (a, a)
            out["cone"] = {"value": lo if lo == hi else f"{lo} - {hi}", "evidence": m.group(0).strip()}
            take(*m.span())
            break
    for w, v in FIRE_WORDS.items():                      # always consumed; they set the cone only when no number was given
        m = re.search(rf"\b{w}\b", s)
        if m and not any(used[m.start():m.end()]):
            if out["cone"] is None:
                out["cone"] = {"value": v, "evidence": m.group(0), "note": "usual range for this word"}
            take(*m.span())
    # colourant amounts: "2% red iron oxide", "cobalt carbonate 1 %" (a number before the name needs the % sign)
    names = "|".join(sorted(map(re.escape, COLOURANTS), key=len, reverse=True))
    for m in list(re.finditer(rf"(\d+(?:[.,]\d+)?)\s*%\s*(?:of\s+)?\b({names})\b", s)) + list(re.finditer(rf"\b({names})\b\s*:?\s*{AMOUNT}", s)):
        if any(used[m.start():m.end()]):
            continue
        num, nm = (m.group(1), m.group(2)) if m.group(1)[0].isdigit() else (m.group(2), m.group(1))
        out["colourants"].append({"name": nm, "oxide": COLOURANTS[nm], "percent": float(num.replace(",", ".")), "evidence": m.group(0).strip()})
        take(*m.span())
    # phrase vocabularies (longest phrases first, exact words), then single-word typos
    def phrases(vocab: dict, key: str, multi: bool):
        pairs = sorted(((p, lab) for lab, ps in vocab.items() for p in ps), key=lambda x: -len(x[0]))
        for p, lab in pairs:
            for m in re.finditer(r"(?<![\w-])" + r"[ -]?".join(map(re.escape, p.split())) + r"(?![\w-])", s):
                if any(used[m.start():m.end()]):
                    continue
                take(*m.span())
                item = {"value": lab, "evidence": m.group(0)}
                if multi:
                    if lab not in [x["value"] for x in out[key]]:
                        out[key].append(item)
                elif out[key] is None:
                    out[key] = item
    for vocab, key, multi in ((TRANSPARENCY, "transparency", False), (SURFACE, "surface", False), (ATMOSPHERE, "atmospheres", True), (EFFECTS, "effects", True)):
        phrases(vocab, key, multi)
    # tokens left: typos of keywords, colour words, unused words
    single = {p: (vocab, key, lab) for vocab, key in ((SURFACE, "surface"), (TRANSPARENCY, "transparency"), (ATMOSPHERE, "atmospheres"), (EFFECTS, "effects"))
              for lab, ps in vocab.items() for p in ps if " " not in p and len(p) >= 5}
    cw = colour_words()
    tokens, colour_run, runs = [], [], []
    for m in re.finditer(r"[a-z][a-z'-]*", s):
        if any(used[m.start():m.end()]):
            if colour_run:
                runs.append(colour_run)
                colour_run = []
            continue
        tok = m.group(0).strip("'-")
        base = tok[:-3] if tok.endswith("ish") and len(tok) > 5 else tok
        if tok in cw or base in cw:
            colour_run.append(tok)
            continue
        if tok in STOP or len(tok) < 3:
            if colour_run:
                runs.append(colour_run)
                colour_run = []
            continue
        kw = _fuzzy(tok, single, 0.82)
        if kw:
            vocab, key, lab = single[kw]
            out["corrections"].append([tok, kw])
            item = {"value": lab, "evidence": tok}
            if key in ("atmospheres", "effects"):
                if lab not in [x["value"] for x in out[key]]:
                    out[key].append(item)
            elif out[key] is None:
                out[key] = item
            continue
        fixed = _fuzzy(base, cw, 0.8)
        if fixed:
            out["corrections"].append([tok, fixed])
            colour_run.append(fixed)
            continue
        if colour_run:
            runs.append(colour_run)
            colour_run = []
        tokens.append(tok)
    if colour_run:
        runs.append(colour_run)
    # colour phrases -> HEX (Color Lab V3 matching: modifiers, -ish words, two-word mixes)
    from .colorlab import MODIFIERS, _colour_from_words
    for run in runs:
        if all(w in MODIFIERS for w in run):
            continue
        c = _colour_from_words(run)
        if c:
            lab, hx, why = c
            out["colours"].append({"hex": hx, "words": " ".join(run), "why": why})
    out["colour"] = out["colours"][0] if out["colours"] else None
    out["unused"] = tokens
    return out


# ---------------------------------------------------------------- noisy sentences (training augmentation + tests)
SURF_SAY = {"glossy": ["glossy", "gloss", "shiny", "glassy"], "semi-glossy": ["semi gloss", "semi-glossy"], "satin": ["satin", "silky"],
            "satin-matte": ["satin matte"], "semi-matte": ["semi matte", "semi-matt"], "smooth matte": ["smooth matte"],
            "matte": ["matte", "matt", "dull"], "stony matte": ["stony matte"], "dry matte": ["dry matte"]}
ATM_SAY = {"oxidation": ["oxidation", "ox", "electric kiln"], "reduction": ["reduction", "reduced", "gas reduction"], "neutral": ["neutral"],
           "salt & soda": ["salt", "soda fired"], "wood": ["wood fired"], "raku": ["raku"], "luster": ["luster"]}
FILLERS = ["", "", "i want a", "looking for", "something like a", "need a", "pls", "nice", "for my mugs", "kinda", "recipe for"]


def _typo(w: str, rng) -> str:
    if len(w) < 4:
        return w
    i, k = int(rng.integers(1, len(w) - 1)), int(rng.integers(0, 3))
    return w[:i] + w[i + 1:] if k == 0 else w[:i] + w[i + 1] + w[i] + w[i + 2:] if k == 1 else w[:i] + w[i] + w[i:]


def say_cone(lo: str, hi: str, rng) -> str:
    if not lo:
        return ""
    v = lo if not hi or hi == lo else f"{lo}-{hi}"
    if (lo, hi) == ("5", "6") and rng.random() < 0.3:
        return "mid fire"
    return [f"cone {v}", f"^{v}", f"c{v}", f"cone{v}", f"cone {v}"][int(rng.integers(0, 5))]


def noisy_variants(facts: dict, n: int, seed: int, typo_p: float = 0.12) -> list[str]:
    """facts {colour, surface, transparency, cone_lo, cone_hi, atmospheres, subtype} -> n noisy sentences a person might type."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        parts = []
        if facts.get("colour") and rng.random() < 0.95:
            parts.append(facts["colour"])
        if facts.get("surface") and rng.random() < 0.85:
            parts.append(str(rng.choice(SURF_SAY.get(facts["surface"], [facts["surface"]]))))
        if facts.get("transparency") and rng.random() < 0.4:
            parts.append(facts["transparency"])
        if facts.get("cone_lo") and rng.random() < 0.85:
            parts.append(say_cone(facts["cone_lo"], facts.get("cone_hi"), rng))
        for a in facts.get("atmospheres") or []:
            if rng.random() < 0.6:
                parts.append(str(rng.choice(ATM_SAY.get(a, [a]))))
        if facts.get("subtype") and rng.random() < 0.35:
            parts.append(facts["subtype"])
        if not parts:
            parts = [facts.get("colour") or "glaze"]
        head = parts[:1] + list(rng.permutation(parts[1:])) if rng.random() < 0.6 else list(rng.permutation(parts))
        words = (str(rng.choice(FILLERS)) + " " + " ".join(head) + (" glaze" if rng.random() < 0.4 else "")).split()
        words = [_typo(w, rng) if rng.random() < typo_p and not w[0].isdigit() and "^" not in w else w for w in words]
        sep = ", " if rng.random() < 0.3 else " "
        txt = sep.join(words)
        out.append(txt.upper() if rng.random() < 0.05 else txt)
    return out


# ---------------------------------------------------------------- learned tags from free text
class TagModel:
    """free text -> probabilities of the tags it implies (trained on recipe descriptions; held-out scores in eval)."""
    FIELDS = ("surface", "transparency", "atmosphere", "family")

    def __init__(self, models: dict, vec):
        self.models, self.vec = models, vec

    @classmethod
    def fit(cls, texts: list[str], labels: dict[str, list[str]], seed: int = 0) -> "TagModel":
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40000)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)
        X = hstack([wv.fit_transform(texts), cv.fit_transform(texts)]).tocsr()
        models = {}
        for f, y in labels.items():
            y = np.asarray(y, dtype=object)
            ok = np.array([bool(v) for v in y])
            if len(set(y[ok])) >= 2:
                models[f] = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed).fit(X[ok], y[ok])
        return cls(models, (wv, cv))

    def transform(self, texts):
        from scipy.sparse import hstack
        return hstack([v.transform(texts) for v in self.vec]).tocsr()

    def predict(self, text: str, top: int = 3) -> dict:
        X = self.transform([text])
        res = {}
        for f, m in self.models.items():
            p = m.predict_proba(X)[0]
            o = np.argsort(-p)[:top]
            res[f] = [[str(m.classes_[i]), round(float(p[i]), 3)] for i in o]
        return res
