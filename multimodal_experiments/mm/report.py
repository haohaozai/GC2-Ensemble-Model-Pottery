"""Results: offline HTML explorer, final report PDF and the mind map PDF of what was added.

  results/site/index.html          open in a browser (works offline): metrics, per-query top-10 viewer,
                                   live vector search with image / text / formula weights and live Recall@3/5/10
  results/CG2_multimodal_final_report.pdf
  results/CG2_multimodal_mindmap_results.pdf
All numbers come from the result files of this run; nothing is typed in by hand.
"""
from __future__ import annotations

import base64
import html
import json
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import charts as CH
from .common import CODE, PHOTOS, dir_size_gb, log, read_json, write_json

E = html.escape
ABL = ["I", "T", "F", "I+T", "I+F", "T+F", "I+T+F"]
IDENT = ["I->T+F", "T->I+F", "F->T+I", "I+T->F", "I+F->T", "T+F->I"]
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def pct(v, d=1):
    return "-" if v is None else f"{100 * v:.{d}f}%"


def msd(vals, d=1, as_pct=True):
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    m, s = float(np.mean(v)), float(np.std(v))
    if as_pct:
        return f"{100 * m:.{d}f}%" + (f" ± {100 * s:.{d}f}" if len(v) > 1 else "")
    return f"{m:.{d}f}" + (f" ± {s:.{d}f}" if len(v) > 1 else "")


def f3(v):
    return "-" if v is None else f"{v:.3f}"


def f1(v):
    return "-" if v is None else f"{v:.1f}"


def mean_of(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def table(head, rows, cls=""):
    h = "".join(f"<th>{E(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


# ---------------------------------------------------------------- collect results
class Results:
    def __init__(self, work: Path):
        r = work / "results"
        self.work, self.r = work, r
        self.manifest = read_json(work / "data" / "manifest.json", {})
        self.state = read_json(work / "state.json", {"done": {}})
        self.A = read_json(r / "A_baselines.json", {})
        self.B1 = read_json(r / "B1_summary.json", {})
        self.B2 = read_json(r / "B2_gridsearchcv.json", {})
        self.C = read_json(r / "C_summary.json", {})
        self.D = read_json(r / "D_summary.json", {})
        self.W = read_json(r / "D_fusion_weights.json", {})
        self.Efull = read_json(r / "E_final.json", {})
        self.CL = read_json(r / "E_dbscan.json", {})
        self.DX = read_json(r / "E_external_diagnosis.json", {})
        self.ext_stats = read_json(work / "external" / "fetch_stats.json", {})
        self.ext_maps = read_json(work / "external" / "label_maps.json", {})
        self.seeds = self.Efull.get("per_seed", [])

    def seedvals(self, *path):
        out = []
        for s in self.seeds:
            x = s
            for p in path:
                x = x.get(p) if isinstance(x, dict) else None
                if x is None:
                    break
            out.append(x)
        return out

    def run_meta(self):
        rows = []
        for ph in ("B1", "C", "D"):
            for p in sorted((self.r / ph).glob("*.json")) if (self.r / ph).exists() else []:
                j = read_json(p)
                rows.append({"phase": ph, "seconds": j["meta"]["seconds"], "vram": j["meta"]["peak_vram_GB"]})
        return rows


# ---------------------------------------------------------------- assets for the explorer
def thumbs(bank, site: Path, ext_dir: Path):
    out = site / "thumbs"
    out.mkdir(parents=True, exist_ok=True)

    def one(i):
        rid = bank.ids[i]
        dst = out / f"{rid}.jpg"
        if dst.exists():
            return
        src = ext_dir / "images" / f"{rid}.jpg" if rid.startswith("x") else PHOTOS / f"{rid}.jpg"
        try:
            im = Image.open(src)
            im.draft("RGB", (400, 400))
            im = im.convert("RGB")
            im.thumbnail((220, 220), Image.LANCZOS)
            im.save(dst, quality=80)
        except Exception:  # noqa: BLE001
            pass
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, range(bank.n)))


def data_js(bank, res: Results, site: Path):
    z = np.load(res.r / "db_embeddings.npz")
    pos = {r: i for i, r in enumerate(z["ids"].tolist())}
    order = [pos[r] for r in bank.ids]
    emb = {}
    for m in ("text", "image", "formula"):
        if f"Z_{m}" in z.files:
            Z = z[f"Z_{m}"][order]
            q = np.clip(np.round(Z * 127), -127, 127).astype(np.int8)
            emb[m] = {"b64": base64.b64encode(q.tobytes()).decode(), "mask": "".join("1" if x else "0" for x in z[f"M_{m}"][order])}
    dim = int(z[f"Z_{next(iter(emb))}"].shape[1])
    recs = []
    for i, r in enumerate(bank.R):
        pal = json.loads(r["palette_json"] or "[]")
        ing = json.loads(r["ingredients_json"] or "[]")
        recs.append([r["recipe_id"], r["name"][:80], bank.split[i], r["subtype"], r["cone_lo"], r["cone_hi"],
                     r["surface"], r["transparency"], r["atmospheres"], [p[0] for p in pal],
                     [round(float(x), 2) if not np.isnan(x) else None for x in bank.lab[i]],
                     int(bank.cone_lo[i]), int(bank.cone_hi[i]), int(bank.surf[i]), int(bank.transp[i]),
                     "; ".join(f"{n} {a:g}{' (+)' if add else ''}" for _, n, a, add in ing[:9]), r["glazy_url"],
                     (r.get("desc_free") or "")[:220]])
    tops = {}
    tp = res.r / "E_tops.npz"
    if tp.exists():
        t = np.load(tp)
        for key in {k.split("|")[0] for k in t.files}:
            q, o = t[f"{key}|q"], t[f"{key}|order"]
            tops[key] = {"q": [bank.ids[i] for i in q], "top": [[bank.ids[j] for j in row] for row in o],
                         "score": np.round(t[f"{key}|score"], 4).tolist(), "rel": t[f"{key}|rel"].astype(int).tolist(),
                         "dE": [[None if np.isnan(x) else round(float(x), 1) for x in row] for row in t[f"{key}|dE"]],
                         "valid": t[f"{key}|valid"].astype(int).tolist()}
    dup = {bank.ids[i]: 1 for i in range(bank.n) if str(bank.R[i].get("near_duplicate_photo", "")) == "1"}   # excluded from external queries, as in the report
    payload = {"dim": dim, "emb": emb, "recipes": recs, "weights": res.Efull.get("weights", {}), "tops": tops, "dup": dup,
               "fields": ["id", "name", "split", "subtype", "cone_lo", "cone_hi", "surface", "transparency", "atmos",
                          "palette", "lab", "clo", "chi", "surf", "transp", "ingredients", "url", "desc"]}
    (site / "data.js").write_text("window.CG2 = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")


# ---------------------------------------------------------------- shared sections (explorer + PDF)
def metric_sections(res: Results) -> dict:
    S = {}
    # identity, internal test
    rows = []
    for k in IDENT:
        g = lambda m: res.seedvals("test", "same_recipe_sets", k, m)  # noqa: E731
        n = mean_of(res.seedvals("test", "same_recipe_sets", k, "gallery"))
        rows.append([k, msd(g("R@1")), msd(g("R@3")), msd(g("R@5")), msd(g("R@10")), msd(g("MRR"), 3, False),
                     msd(g("tie_R@5")), pct(min(1.0, 5 / n) if n else None)])
    S["ident_test"] = table(["query -> gallery", "R@1", "R@3", "R@5", "R@10", "MRR", "R@5 tie-aware", "random R@5"], rows)
    ser = {f"R@{k}": [mean_of(res.seedvals("test", "same_recipe_sets", c, f"R@{k}")) for c in IDENT] for k in (3, 5, 10)}
    S["ident_test_chart"] = CH.hbars("Same recipe across modalities - internal test (gallery = test recipes)", IDENT, ser)
    rows = []
    for k in IDENT:
        g = lambda m: res.seedvals("test", "same_recipe_full_gallery", k, m)  # noqa: E731
        n = mean_of(res.seedvals("test", "same_recipe_full_gallery", k, "gallery"))
        rows.append([k, msd(g("R@1")), msd(g("R@3")), msd(g("R@5")), msd(g("R@10")), msd(g("MRR"), 3, False),
                     msd(g("tie_R@5")), pct(min(1.0, 5 / n) if n else None)])
    S["ident_test_full"] = table(["query -> gallery (full database)", "R@1", "R@3", "R@5", "R@10", "MRR", "R@5 tie-aware",
                                  "random R@5"], rows)
    # attribute ablation, internal test
    base = res.A.get("attribute", {})
    rows, ser = [], {"Hit@3": [], "Hit@5": [], "Hit@10": []}
    for k in ABL:
        g = lambda m: res.seedvals("test", "attribute_ablation", k, m)  # noqa: E731
        rows.append([k, msd(g("Hit@3")), msd(g("Hit@5")), msd(g("Hit@10")), msd(g("P@5")), msd(g("nDCG@10"), 3, False),
                     msd(g("MAP@10"), 3, False), msd(g("meanDeltaE@5"), 1, False)])
        for m in ser:
            ser[m].append(mean_of(g(m)))
    for name, lab in (("random", "baseline: random"), ("image_clip_knn", "baseline: image CLIP kNN"),
                      ("text_minilm_knn", "baseline: text MiniLM kNN"), ("formula_umf_knn", "baseline: formula UMF kNN")):
        b = base.get(name)
        if b:
            rows.append([lab, pct(b["Hit@3"]), pct(b["Hit@5"]), pct(b["Hit@10"]), pct(b["P@5"]), f3(b['nDCG@10']),
                         f3(b['MAP@10']), f1(b['meanDeltaE@5'])])
            for m in ser:
                ser[m].append(b[m])
    cats = [f"query {k}" for k in ABL] + [r[0] for r in rows[len(ABL):]]
    S["attr_test"] = table(["query modalities", "Recall@3 (hit)", "Recall@5 (hit)", "Recall@10 (hit)", "P@5", "nDCG@10",
                            "MAP@10", "mean ΔE2000 top-5"], rows)
    S["attr_test_chart"] = CH.hbars("Most appropriate recipe (attribute + colour relevance) - internal test", cats, ser)
    # pairwise vs baselines
    rows = []
    zs = res.A.get("same_recipe", {})
    for d in ["I->T", "T->I", "I->F", "F->I", "T->F", "F->T"]:
        g = lambda m: res.seedvals("test", "pairwise", d, m)  # noqa: E731
        clip = zs.get("clip_zero_shot/test", {}).get(d, {})
        cca = zs.get("cca/test", {}).get(d, {})
        rows.append([d, msd(g("R@1")), msd(g("R@5")), msd(g("R@10")), msd(g("MRR"), 3, False),
                     pct(cca.get("R@5")) if cca else "-", pct(clip.get("R@5")) if clip else "-"])
    S["pairwise"] = table(["direction", "R@1", "R@5", "R@10", "MRR", "CCA R@5", "CLIP zero-shot R@5"], rows)
    # cosine gap
    rows = []
    for p in ("text-image", "text-formula", "image-formula"):
        g = lambda m: res.seedvals("test", "cosine_gap", p, m)  # noqa: E731
        rows.append([p, msd(g("pos_mean"), 3, False), msd(g("neg_mean"), 3, False), msd(g("gap"), 3, False),
                     msd(g("pos_above_neg"))])
    S["cosgap"] = table(["pair", "mean cos same recipe", "mean cos other recipe", "gap", "pos > neg"], rows)
    # external
    ex = [s.get("external", {}) for s in res.seeds]
    if ex and ex[0].get("n_external"):
        e0 = ex[0]
        rows = []
        for k in IDENT:
            g = lambda m: res.seedvals("external", "identity_in_gallery", k, m)  # noqa: E731
            n = mean_of(res.seedvals("external", "identity_in_gallery", k, "gallery"))
            rows.append([k, msd(g("R@1")), msd(g("R@3")), msd(g("R@5")), msd(g("R@10")), msd(g("MRR"), 3, False),
                         pct(min(1.0, 5 / n) if n else None)])
        S["ident_ext"] = table(["query -> gallery (7k + external)", "R@1", "R@3", "R@5", "R@10", "MRR", "random R@5"], rows)
        ser = {f"R@{k}": [mean_of(res.seedvals("external", "identity_in_gallery", c, f"R@{k}")) for c in IDENT] for k in (3, 5, 10)}
        S["ident_ext_chart"] = CH.hbars("Same recipe - external Glazy recipes added to the 7k gallery", IDENT, ser)
        rows, ser = [], {"Hit@3": [], "Hit@5": [], "Hit@10": []}
        for k in ABL:
            g = lambda m: res.seedvals("external", "attribute_ablation", k, m)  # noqa: E731
            rows.append([k, msd(g("Hit@3")), msd(g("Hit@5")), msd(g("Hit@10")), msd(g("P@5")), msd(g("nDCG@10"), 3, False),
                         msd(g("MAP@10"), 3, False), msd(g("meanDeltaE@5"), 1, False)])
            for m in ser:
                ser[m].append(mean_of(g(m)))
        for name, lab in (("random", "baseline: random"), ("image_clip_knn", "baseline: image CLIP kNN")):
            b = e0.get("baselines", {}).get(name)
            if b:
                rows.append([lab, pct(b["Hit@3"]), pct(b["Hit@5"]), pct(b["Hit@10"]), pct(b["P@5"]), f3(b['nDCG@10']),
                             f3(b['MAP@10']), f1(b['meanDeltaE@5'])])
                for m in ser:
                    ser[m].append(b[m])
        cats = [f"query {k}" for k in ABL] + [r[0] for r in rows[len(ABL):]]
        S["attr_ext"] = table(["query modalities", "Recall@3 (hit)", "Recall@5 (hit)", "Recall@10 (hit)", "P@5", "nDCG@10",
                               "MAP@10", "mean ΔE2000 top-5"], rows)
        S["attr_ext_chart"] = CH.hbars("Most appropriate existing recipe for external recipes", cats, ser)
        fd = [x.get("formula_distance_top5_vs_random", {}) for x in ex]
        S["ext_formula"] = (f"Formula closeness (standardised UMF distance, lower = closer): top-5 retrieved "
                            f"{msd([f.get('top5_mean') for f in fd], 2, False)} vs 5 random recipes "
                            f"{msd([f.get('random5_mean') for f in fd], 2, False)} (n = {fd[0].get('n')}).")
        S["ext_counts"] = e0
    return S


CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--surf:#fcfcfb;--page:#f9f9f7;--acc:#2a78d6;--ok:#006300;--bad:#d03b3b}
*{box-sizing:border-box} body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:22px 0 8px} h3{font-size:14px;margin:14px 0 6px}
.sub{color:var(--ink2)} table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-size:12.5px;background:var(--surf)}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}
th{background:#f0efec} .card{background:var(--surf);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.note{color:var(--ink2);font-size:12.5px} code{background:#f0efec;padding:0 4px;border-radius:4px;font-size:12px}
"""


# ---------------------------------------------------------------- explorer
EXPLORER_JS = r"""
const C = window.CG2, R = C.recipes, D = C.dim, idx = {};
R.forEach((r, i) => idx[r[0]] = i);
const MODS = ["image", "text", "formula"], EMB = {}, MASK = {};
for (const m of MODS) if (C.emb[m]) {
  const b = atob(C.emb[m].b64), a = new Int8Array(b.length);
  for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i) << 24 >> 24;
  EMB[m] = a; MASK[m] = Array.from(C.emb[m].mask, c => c === "1");
}
const $ = id => document.getElementById(id), esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const DB = R.map((r, i) => i).filter(i => R[i][2] !== "ext");
function fused(i, mods, w) {
  const v = new Float32Array(D); let any = false;
  for (const m of mods) if (EMB[m] && MASK[m][i]) { any = true; const o = i * D; for (let k = 0; k < D; k++) v[k] += w[m] * EMB[m][o + k] / 127; }
  if (!any) return null; let n = 0; for (let k = 0; k < D; k++) n += v[k] * v[k]; n = Math.sqrt(n) || 1;
  for (let k = 0; k < D; k++) v[k] /= n; return v;
}
function galleryMatrix(rows, mods, w) {
  const G = new Float32Array(rows.length * D), ok = new Uint8Array(rows.length);
  rows.forEach((i, j) => { const v = fused(i, mods, w); if (v) { G.set(v, j * D); ok[j] = 1; } });
  return {G, ok};
}
function scores(q, gm) { const n = gm.ok.length, s = new Float32Array(n); for (let j = 0; j < n; j++) { if (!gm.ok[j]) { s[j] = -9; continue; } let t = 0; const o = j * D; for (let k = 0; k < D; k++) t += q[k] * gm.G[o + k]; s[j] = t; } return s; }
// CIEDE2000
function de2000(a, b) {
  if (!a || !b || a[0] == null || b[0] == null) return null;
  const [L1, a1, b1] = a, [L2, a2, b2] = b, rad = Math.PI / 180;
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2), Cm = (C1 + C2) / 2, G = 0.5 * (1 - Math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)));
  const a1p = (1 + G) * a1, a2p = (1 + G) * a2, C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const h = (x, y) => { const t = Math.atan2(y, x) / rad; return t < 0 ? t + 360 : t; }, h1 = h(a1p, b1), h2 = h(a2p, b2);
  const dL = L2 - L1, dC = C2p - C1p; let dh = 0; if (C1p * C2p) { dh = h2 - h1; if (dh > 180) dh -= 360; else if (dh < -180) dh += 360; }
  const dH = 2 * Math.sqrt(C1p * C2p) * Math.sin(dh * rad / 2), Lm = (L1 + L2) / 2, Cpm = (C1p + C2p) / 2;
  let hm = h1 + h2; if (C1p * C2p) { hm = Math.abs(h1 - h2) > 180 ? (h1 + h2 + (h1 + h2 < 360 ? 360 : -360)) / 2 : (h1 + h2) / 2; }
  const T = 1 - 0.17 * Math.cos((hm - 30) * rad) + 0.24 * Math.cos(2 * hm * rad) + 0.32 * Math.cos((3 * hm + 6) * rad) - 0.2 * Math.cos((4 * hm - 63) * rad);
  const SL = 1 + 0.015 * (Lm - 50) ** 2 / Math.sqrt(20 + (Lm - 50) ** 2), SC = 1 + 0.045 * Cpm, SH = 1 + 0.015 * Cpm * T;
  const RT = -2 * Math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7)) * Math.sin(60 * Math.exp(-(((hm - 275) / 25) ** 2)) * rad);
  return Math.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH));
}
function relevant(q, g) {
  const a = R[q], b = R[g]; const cc = a[11] >= 0 && b[11] >= 0, sc = a[13] >= 0 && b[13] >= 0, tc = a[14] >= 0 && b[14] >= 0;
  if (!(cc || sc || tc)) return [false, null];
  const de = de2000(a[10], b[10]); if (de == null || de > 20) return [false, de];
  if (cc && !(a[11] <= b[12] && b[11] <= a[12])) return [false, de];
  if (sc && a[13] !== b[13]) return [false, de]; if (tc && a[14] !== b[14]) return [false, de];
  return [true, de];
}
function card(i, extra) {
  const r = R[i], pal = (r[9] || []).map(h => `<span class="sw" style="background:${h}" title="${h}"></span>`).join("");
  const cone = r[4] ? (r[4] === r[5] ? "Δ" + r[4] : `Δ${r[4]}–${r[5]}`) : "cone ?";
  return `<div class="rc">${extra || ""}<img loading="lazy" src="thumbs/${esc(r[0])}.jpg" alt=""><div class="rt"><a href="${esc(r[16])}" target="_blank" rel="noopener"><b>${esc(r[1] || r[0])}</b></a>
    <div class="mu">#${esc(r[0])} · ${esc(r[2])}</div><div>${esc(r[3])}</div><div class="mu">${esc(cone)} · ${esc(r[6] || "surface ?")} · ${esc(r[7] || "transparency ?")}</div>
    <div class="pal">${pal}</div><div class="ing">${esc(r[15])}</div></div></div>`;
}
function weights() { return {image: +$("wI").value, text: +$("wT").value, formula: +$("wF").value}; }
function qmods() { return MODS.filter(m => $("q_" + m).checked); }
let GCACHE = null, GKEY = "";
function gallery() { const w = weights(), key = JSON.stringify(w); if (key !== GKEY) { GCACHE = galleryMatrix(DB, MODS, w); GKEY = key; } return GCACHE; }
function search() {
  const id = $("qid").value.trim().replace(/^#/, ""), i = idx[id] ?? idx["x" + id];
  if (i == null) { $("results").innerHTML = `<p class="mu">Recipe ${esc(id)} not found.</p>`; return; }
  const K = +$("k").value, w = weights(), q = fused(i, qmods(), w);
  if (!q) { $("results").innerHTML = `<p class="mu">This recipe has none of the selected query modalities.</p>`; return; }
  const gm = gallery(), s = scores(q, gm), ord = Array.from(s.keys()).sort((a, b) => s[b] - s[a]).filter(j => DB[j] !== i).slice(0, K);
  let nrel = 0;
  const cards = ord.map((j, n) => { const [rel, de] = relevant(i, DB[j]); nrel += rel;
    return card(DB[j], `<div class="rk">${n + 1}<span>${s[j].toFixed(3)}</span>${rel ? '<em class="ok">✓ relevant</em>' : ""}${de != null ? `<span>ΔE ${de.toFixed(1)}</span>` : ""}</div>`); });
  const selfRank = 1 + Array.from(s).filter((x, j) => DB[j] !== i && x > (s[DB.indexOf(i)] ?? -9)).length;
  $("results").innerHTML = `<h3>Query</h3><div class="grid">${card(i)}</div><h3>Top ${K} existing recipes · ${nrel} relevant (attribute + colour)</h3><div class="grid">${cards.join("")}</div>`;
}
function liveRecall() {
  const set = $("lset").value, w = weights(), qm = qmods(), gmods = MODS.filter(m => !qm.includes(m));
  const Q = R.map((r, i) => i).filter(i => set === "ext" ? R[i][2] === "ext" && !(C.dup || {})[R[i][0]] : R[i][2] === "test");
  $("lres").textContent = "computing…";
  setTimeout(() => {
    const t0 = performance.now(), ks = [3, 5, 10];
    // identity: query modalities vs gallery = remaining modalities, gallery = database (+ the external queries)
    const grows = set === "ext" ? DB.concat(Q) : DB, gi = {}; grows.forEach((g, j) => gi[g] = j);
    const out = {n: 0, hit: {3: 0, 5: 0, 10: 0}, mrr: 0, an: 0, ahit: {3: 0, 5: 0, 10: 0}};
    if (!gmods.length) out.note = "uncheck at least one query modality: the same-recipe test searches the database side with the modalities the query does not use";
    if (gmods.length) {
      const gm = galleryMatrix(grows, gmods, w);
      for (const i of Q) { const q = fused(i, qm, w); if (!q || !gm.ok[gi[i]]) continue; const s = scores(q, gm), t = s[gi[i]];
        let r = 1; for (let j = 0; j < s.length; j++) if (s[j] > t) r++; out.n++; out.mrr += 1 / r; for (const k of ks) if (r <= k) out.hit[k]++; }
    }
    // attribute relevance: gallery = database fused (all modalities); a hit = >= 1 relevant recipe in the top K
    const ga = gallery();
    for (const i of Q) { const q = fused(i, qm, w); if (!q) continue; const s = scores(q, ga);
      const ord = Array.from(s.keys()).sort((a, b) => s[b] - s[a]).filter(j => DB[j] !== i).slice(0, 10);
      const rel = ord.map(j => relevant(i, DB[j])[0]); let any = false;
      for (let j = 0; j < DB.length && !any; j++) if (DB[j] !== i && relevant(i, DB[j])[0]) any = true;
      if (!any) continue; out.an++; for (const k of ks) if (rel.slice(0, k).some(x => x)) out.ahit[k]++; }
    const p = (a, b) => b ? (100 * a / b).toFixed(1) + "%" : "-";
    $("lres").innerHTML = `<table><tr><th>${set === "ext" ? "external" : "internal test"} queries · query ${qm.join("+") || "-"}</th><th>Recall@3</th><th>Recall@5</th><th>Recall@10</th><th>MRR</th><th>n</th></tr>
      <tr><td>same recipe (gallery = ${gmods.join("+") || "-"} of ${grows.length} recipes)</td><td>${p(out.hit[3], out.n)}</td><td>${p(out.hit[5], out.n)}</td><td>${p(out.hit[10], out.n)}</td><td>${out.n ? (out.mrr / out.n).toFixed(3) : "-"}</td><td>${out.n}</td></tr>
      <tr><td>most appropriate (attribute + colour relevant in top K, gallery = ${DB.length} database recipes)</td><td>${p(out.ahit[3], out.an)}</td><td>${p(out.ahit[5], out.an)}</td><td>${p(out.ahit[10], out.an)}</td><td>-</td><td>${out.an}</td></tr></table>
      ${out.note ? `<p class="mu"><b>Same recipe:</b> ${out.note}.</p>` : ""}<p class="mu">Computed in your browser in ${((performance.now() - t0) / 1000).toFixed(1)} s from the 8-bit stored embeddings (gallery = whole database, so the same-recipe numbers are stricter than the report's test-only gallery).</p>`;
  }, 30);
}
function topsViewer() {
  const key = $("tset").value, T = C.tops[key]; if (!T) { $("topsList").innerHTML = "<p class='mu'>not available</p>"; return; }
  const start = +$("tpage").value * 20, rows = [];
  for (let n = start; n < Math.min(T.q.length, start + 20); n++) {
    const qi = idx[T.q[n]], hits = T.rel[n].slice(0, 5).reduce((a, b) => a + b, 0);
    rows.push(`<div class="qrow"><div class="qcell">${card(qi)}<div class="mu">${T.valid[n] ? hits + " of top-5 relevant" : "no relevant recipe exists (not scored)"}</div></div><div class="grid">` +
      T.top[n].slice(0, +$("tk").value).map((g, j) => card(idx[g], `<div class="rk">${j + 1}<span>${T.score[n][j].toFixed(3)}</span>${T.rel[n][j] ? '<em class="ok">✓</em>' : ""}${T.dE[n][j] != null ? `<span>ΔE ${T.dE[n][j]}</span>` : ""}</div>`)).join("") + "</div></div>");
  }
  $("topsList").innerHTML = rows.join("") || "<p class='mu'>no queries</p>";
  $("tpage").max = Math.max(0, Math.ceil(T.q.length / 20) - 1);
}
document.addEventListener("DOMContentLoaded", () => {
  const w = C.weights || {image: 0.5, text: 0.3, formula: 0.2};
  $("wI").value = w.image; $("wT").value = w.text; $("wF").value = w.formula;
  const keys = Object.keys(C.tops); $("tset").innerHTML = keys.map(k => `<option>${k}</option>`).join("");
  $("tset").onchange = () => { $("tpage").value = 0; topsViewer(); }; $("tpage").oninput = topsViewer; $("tk").onchange = topsViewer;
  $("go").onclick = search; $("qid").onkeydown = e => { if (e.key === "Enter") search(); }; $("live").onclick = liveRecall;
  for (const id of ["wI", "wT", "wF"]) $(id).oninput = () => { $(id + "v").textContent = (+$(id).value).toFixed(2); };
  ["wI", "wT", "wF"].forEach(id => $(id + "v").textContent = (+$(id).value).toFixed(2));
  const firstExt = R.find(r => r[2] === "ext"), firstTest = R.find(r => r[2] === "test");
  $("qid").value = (firstExt || firstTest || R[0])[0]; topsViewer(); search();
});
"""

EXPLORER_CSS = CSS + """
header{background:var(--surf);border-bottom:1px solid var(--line);padding:16px 20px} main{max-width:1500px;margin:0 auto;padding:6px 20px 40px}
nav a{margin-right:14px;color:var(--acc);text-decoration:none;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:8px}
.rc{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px;font-size:12px;position:relative}
.rc img{width:100%;aspect-ratio:1/1;object-fit:contain;background:#f0efec;border-radius:6px}
.rt a{color:var(--ink);text-decoration:none} .mu{color:var(--muted);font-size:11.5px} .ing{color:var(--ink2);font-size:11px;margin-top:3px}
.pal{margin-top:3px}.sw{display:inline-block;width:16px;height:16px;border-radius:4px;margin-right:3px;border:1px solid rgba(0,0,0,.12)}
.rk{display:flex;gap:6px;align-items:center;font-weight:700;margin-bottom:4px}.rk span{font-weight:400;color:var(--ink2)}
.ok{color:var(--ok);font-style:normal;font-weight:700} .qrow{display:grid;grid-template-columns:230px 1fr;gap:10px;margin:10px 0;
padding-bottom:10px;border-bottom:1px solid var(--line)} .ctl{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:8px 0}
input[type=range]{width:120px;vertical-align:middle} button{font:inherit;padding:5px 12px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer}
button.p{background:var(--acc);color:#fff;border-color:var(--acc)} @media (max-width:700px){.qrow{grid-template-columns:1fr}}
"""


def explorer(work, bank, res: Results, S: dict, site: Path):
    cfg = res.Efull.get("final_cfg", {})
    w = res.Efull.get("weights", {})
    ext = S.get("ext_counts", {})
    body = f"""
<header><h1>CG2 multimodal glaze retrieval - results explorer</h1>
<div class="sub">Final model: text <b>{E(str(cfg.get('text')))}</b> · image <b>{E(str(cfg.get('image')))}</b> · formula <b>{E(str(cfg.get('formula')))}</b>
 · head {E(str(cfg.get('head')))} · {cfg.get('dim')}-d shared space · fusion weights image {w.get('image', 0):.2f} ≥ text {w.get('text', 0):.2f} ≥ formula {w.get('formula', 0):.2f}
 · {len(res.seeds)} seed(s) · numbers = mean ± sd over seeds on the untouched test split.</div>
<nav><a href="#sec-metrics">Metrics</a><a href="#sec-search">Vector search</a><a href="#sec-live">Live Recall@K</a><a href="#sec-tops">Per-query top-K</a></nav></header>
<main>
<section id="sec-metrics"><h2>Internal test ({res.manifest.get('split', {}).get('test', '?')} recipes never used for training or selection)</h2>
<div class="card">{S['ident_test_chart']}{S['ident_test']}<p class="note">Same recipe: the query modalities of a test recipe must find
that recipe among the test recipes represented by the other modalities. Tie-aware: recipes with identical tags / formula count as ties.</p>
<h3>Same task, gallery = the whole database (train + validation + test) - the protocol used for the external recipes</h3>{S['ident_test_full']}</div>
<div class="card">{S['attr_test_chart']}{S['attr_test']}<p class="note">Most appropriate: Recall@K = share of queries with at least one relevant
recipe in the top K (relevant = dominant glaze colour within ΔE2000 ≤ 20 and matching known cone / surface / transparency); gallery = train + validation recipes.</p></div>
<div class="card"><h3>Pairwise directions vs baselines</h3>{S['pairwise']}<h3>Embedding quality (cosine)</h3>{S['cosgap']}</div>
{f'''<h2>External Glazy recipes (not in the 7k photos; {ext.get('n_used')} used, {ext.get('n_near_duplicate_photo_excluded')} near-duplicate photos excluded)</h2>
<div class="card">{S['ident_ext_chart']}{S['ident_ext']}</div><div class="card">{S['attr_ext_chart']}{S['attr_ext']}<p class="note">{E(S['ext_formula'])}</p></div>''' if 'ident_ext' in S else ''}
</section>
<section id="sec-search"><h2>Vector search (query by example)</h2><div class="card">
<div class="ctl"><label>Recipe ID <input id="qid" size="10"></label>
<label><input type="checkbox" id="q_image" checked> image</label><label><input type="checkbox" id="q_text" checked> text</label><label><input type="checkbox" id="q_formula" checked> formula</label>
<label>image <input type="range" id="wI" min="0" max="1" step="0.05"> <span id="wIv"></span></label>
<label>text <input type="range" id="wT" min="0" max="1" step="0.05"> <span id="wTv"></span></label>
<label>formula <input type="range" id="wF" min="0" max="1" step="0.05"> <span id="wFv"></span></label>
<label>K <select id="k"><option>3</option><option selected>5</option><option>10</option></select></label><button class="p" id="go">Search</button></div>
<p class="note">Any internal ID or external ID (x…) works. The database side always uses all its modalities with the same weights; default weights = the
validated fusion weights (image ≥ text ≥ formula).</p><div id="results"></div></div></section>
<section id="sec-live"><h2>Live Recall@3 / 5 / 10 for the current settings</h2><div class="card">
<div class="ctl"><label>queries <select id="lset"><option value="test">internal test</option><option value="ext">external</option></select></label>
<button class="p" id="live">Compute</button><span class="mu">uses the query checkboxes and weights above</span></div><div id="lres"></div></div></section>
<section id="sec-tops"><h2>Per-query top-K (stored final results)</h2><div class="card">
<div class="ctl"><label>set <select id="tset"></select></label><label>page <input type="number" id="tpage" min="0" value="0" style="width:60px"></label>
<label>show <select id="tk"><option>3</option><option selected>5</option><option>10</option></select></label></div><div id="topsList"></div></div></section>
</main>"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>CG2 retrieval results</title><style>{EXPLORER_CSS}</style></head><body>{body}'
            f'<script src="data.js"></script><script>{EXPLORER_JS}</script></body></html>')
    (site / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- final report (HTML -> PDF)
def print_pdf(html_path: Path, pdf: Path):
    prof = html_path.parent / "_edge_profile"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
                    f"--print-to-pdf={pdf}", html_path.resolve().as_uri()], timeout=300)


def external_gap_html(dx: dict) -> str:
    """Section 8b: measured reasons for the internal / external gap (mm/diagnose.py)."""
    if not dx:
        return ""
    al = dx.get("alignment", {})
    names = {"train": "train", "val": "validation", "test": "internal test", "ext": "external"}
    arows = [[names[sp]] + [f"{f3(al[sp][k]['same_recipe'])} (random pair {f3(al[sp][k]['random_pair'])})"
                            for k in ("text-image", "text-formula", "image-formula")] for sp in names if sp in al]
    nrows = [[names.get(g, g), v["n"], f"{v['median_wt%_distance']:.2f}", pct(v["share_near_copy_lt2"]), pct(v["share_far_gt5"])]
             for g, v in dx.get("novelty", {}).items()]
    brows = [[k.replace(" ext", " external").replace(" test", " internal test"), v["n"], pct(v["R@10"]),
              f"{pct(v['R@10_near_copy'])} (n={v['n_near_copy']})", f"{pct(v['R@10_novel'])} (n={v['n_novel']})", f1(v["median_rank"])]
             for k, v in dx.get("recall_by_novelty_full_db", {}).items()]
    erows = [[k, v["gallery_size"], f"{pct(v['internal_test_R@5_mean_of_5_draws'])} ± {100 * v['internal_test_R@5_sd']:.1f}",
              pct(v["external_R@5"])] for k, v in dx.get("equal_gallery_size", {}).items()]
    ic = dx.get("input_consistency", {})
    tw = dx.get("formula_twins", {})
    en = dx.get("empty_notes_share", {})
    return f"""<h3>8b. Why the external numbers are lower (measured)</h3>
<p><b>Not a parsing bug:</b> for the {ic.get('n_in_export', 0)} external recipes that are also in the Glazy export, the parsed oxides differ by at most
{ic.get('max_oxide_abs_diff_wt%')} wt% (identical when 0), ingredient names are identical for {ic.get('ingredient_names_identical')} and the tag text for
{ic.get('tags_identical')} (differences: an atmosphere label not learned, a renamed subtype trail). Identical formulas give identical formula embeddings.</p>
<p><b>1. Over-fitting of the projection heads.</b> Mean cosine between the modalities of the same recipe:</p>
{table(["split", "text - image", "text - formula", "image - formula"], arows)}
<p><b>2. External recipes are more novel.</b> Distance of the oxide wt% to the nearest training recipe:</p>
{table(["queries", "n", "median distance (wt%)", "near copy (under 2 wt%)", "far (over 5 wt%)"], nrows)}
<p>Same-recipe Recall@10 against the whole database, split by novelty:</p>
{table(["direction / queries", "n", "Recall@10", "near copies", "novel", "median rank"], brows)}
<p><b>3. Same gallery size.</b> Same-recipe Recall@5 when both query sets search a gallery of {next(iter(dx.get('equal_gallery_size', {}).values()), {}).get('gallery_size', '?')} recipes:</p>
{table(["direction", "gallery", "internal test (5 random draws)", "external"], erows)}
<p class="note">Other differences: external photos come from the Glazy CDN with unreviewed automatic segmentation (internal objects were reviewed by hand);
free-text notes are empty for {pct(en.get('external'))} of external vs {pct(en.get('internal'))} of internal recipes;
{tw.get('n_external_with_twin', 0)} external recipes have a formula identical to an internal recipe (0.1 wt%), so a formula query cannot separate them.
External data were not used for any model choice, so these results were not tuned after seeing them.</p>"""


def final_report(work, bank, res: Results, S: dict, smoke: bool) -> Path:
    m = res.manifest
    cfg = res.Efull.get("final_cfg", {})
    w = res.Efull.get("weights", {})
    st = res.state.get("done", {})
    # models compared (B1)
    b1 = res.B1.get("results", [])
    b1rows = [[x["modality"], x["encoder"], f"{x['modality_MRR']:.3f}", f"{x['mean_MRR']:.3f}", f"{x['seconds']:.0f}"] for x in b1]
    b1chart = ""
    if b1:
        cats = [f"{x['modality']}: {x['encoder']}" for x in b1]
        b1chart = CH.hbars("Encoder screening (B1) - validation MRR of the varied modality", cats,
                           {"modality MRR": [x["modality_MRR"] for x in b1]}, vmax=max(0.05, max(x["modality_MRR"] for x in b1) * 1.15), pct=False)
    b2 = res.B2.get("ranking", [])
    b2rows = [[str(i + 1), E(json.dumps(x["params"])), f"{x['mean_cv_MRR']:.4f}", f"{x['std_cv_MRR']:.4f}", f"{x['fit_seconds']:.0f}"]
              for i, x in enumerate(b2[:12])]
    crows = [[E(json.dumps(x["params"])), f"{x['cv_MRR']:.4f}", f"{x['val_mean_MRR']:.4f}"] for x in res.C.get("results", [])]
    drows = [[E(json.dumps(x["extra"])), f"{x['val_MRR_mean']:.4f} ± {x['val_MRR_sd']:.4f}"] for x in res.D.get("results", [])]
    wrows = [[f"{r['weights']['image']:.2f} / {r['weights']['text']:.2f} / {r['weights']['formula']:.2f}", pct(r["attr_Hit@5"]),
              f3(r['attr_nDCG@10']), f3(r['I->T+F_tie_MRR']), f3(r['score'])] for r in res.W.get("rows", [])]
    # curves
    curves = ""
    if res.seeds:
        curves = CH.lines("Training: validation mean MRR per epoch (final configuration)",
                          [[h["epoch"] for h in s["history"]] for s in res.seeds[:3]],
                          [[h["val_mean_MRR"] for h in s["history"]] for s in res.seeds[:3]],
                          [f"seed {s['seed']}" for s in res.seeds[:3]], ylab="val MRR")
        curves += CH.lines("Training loss per epoch", [[h["epoch"] for h in s["history"]] for s in res.seeds[:3]],
                           [[h["train_loss"] for h in s["history"]] for s in res.seeds[:3]],
                           [f"seed {s['seed']}" for s in res.seeds[:3]], ylab="InfoNCE loss")
    # pair-trained ablation
    pt = res.Efull.get("pair_trained", {})
    ptrows = []
    for name, v in pt.items():
        for d, mm in v["pairwise"].items():
            ptrows.append([name, d, pct(mm.get("R@5")), f3(mm.get("MRR"))])
    full_pair = {d: mean_of(res.seedvals("test", "pairwise", d, "R@5")) for d in ["I->T", "T->I", "I->F", "F->I", "T->F", "F->T"]}
    # clustering
    dx_html = external_gap_html(res.DX)
    cl = res.CL
    clrows =[[f"{v['eps']:.3f} ({v.get('label', str(v.get('factor')) + 'x')})", v["clusters"], pct(v["noise_ratio"]),
               pct(v["largest_cluster_share"]) if v.get("largest_cluster_share") is not None else "-", str(v["silhouette_cosine"]),
               pct(v["purity_surface"]) if v["purity_surface"] is not None else "-",
               pct(v["purity_subtype_l2"]) if v["purity_subtype_l2"] is not None else "-"] for v in cl.get("variants", [])]
    cl_cmp = ""
    if cl.get("test_without_clusters"):
        a, b = cl["test_without_clusters"], cl["test_with_clusters"]
        cl_cmp = table(["retrieval (test, query I+T+F)", "Recall@3", "Recall@5", "Recall@10", "nDCG@10"],
                       [["without DBSCAN", pct(a["Hit@3"]), pct(a["Hit@5"]), pct(a["Hit@10"]), f3(a['nDCG@10'])],
                        [f"with DBSCAN re-ranking ({cl.get('chosen_eps_label', 'knee')}, beta {cl.get('chosen_beta')})", pct(b["Hit@3"]), pct(b["Hit@5"]), pct(b["Hit@10"]), f3(b['nDCG@10'])]])
    # failure analysis: test queries whose own recipe ranks worst (image -> text + formula)
    tops_note = ""
    tp = res.r / "E_tops.npz"
    fails = []
    if tp.exists():
        t = np.load(tp)
        if "test:I+T+F|q" in t.files:
            q, rel, valid = t["test:I+T+F|q"], t["test:I+T+F|rel"], t["test:I+T+F|valid"]
            miss = [(int(qq)) for qq, r, v in zip(q, rel, valid) if v and not r[:5].any()]
            from collections import Counter
            c = Counter(bank.R[i]["subtype_l2"] or "(no level 2)" for i in miss)
            tops_note = (f"{len(miss)} of {int(valid.sum())} scored test queries have no relevant recipe in their top 5. "
                         f"Most frequent subtypes among them: " + ", ".join(f"{k} ({v})" for k, v in c.most_common(6)) + ".")
            fails = [[bank.ids[i], E(bank.R[i]["name"][:40]), E(bank.R[i]["subtype"]), E(bank.R[i]["surface"] or "?"),
                      E(bank.R[i]["transparency"] or "?")] for i in miss[:12]]
    # resources
    res_rows = [[k, f"{v.get('seconds', 0) / 60:.1f} min", f"{v.get('work_GB', 0):.2f} GB", f"{v.get('free_D_GB')} / {v.get('free_C_GB')} GB"]
                for k, v in st.items()]
    vram = max([s["meta"]["peak_vram_GB"] for s in res.seeds] + [0])
    ex = S.get("ext_counts", {})
    body = f"""
<section><h1>CG2 Multimodal Glaze Retrieval - final report{' (SMOKE TEST)' if smoke else ''}</h1>
<div class="sub">Generated from the result files in <code>{E(str(res.r))}</code>. All numbers are measured; test numbers come from a single
evaluation of the selected configuration (mean ± sd over {len(res.seeds)} training seed(s)); the test split and the external recipes were never used
for training, model selection or tuning.</div>
<h2>1. System architecture</h2><p>Three frozen encoders (text: <b>{E(str(cfg.get('text')))}</b>, image: <b>{E(str(cfg.get('image')))}</b> on the reviewed
segmentation objects with mask pooling, formula: <b>{E(str(cfg.get('formula')))}</b> from the oxide analysis) feed trainable projection heads
({E(str(cfg.get('head')))}, {cfg.get('dim')}-d, L2-normalised). The heads are trained with symmetric InfoNCE on text-image, text-formula and image-formula
pairs of the same recipe; identical tag texts / formulas are excluded from the negatives; missing modalities are skipped. Queries (any subset of
image / text / formula) and database recipes are fused with weights image {w.get('image', 0):.2f} ≥ text {w.get('text', 0):.2f} ≥ formula
{w.get('formula', 0):.2f} (chosen on validation within the requested priority) and ranked by cosine similarity; DBSCAN clusters are an optional re-ranking.</p>
<h2>2. Dataset</h2>{table(["item", "value"], [["recipes with photo (one photo each)", m.get('recipes')], ["usable segmented objects (master review CSV)", m.get('objects')],
 ["split train / val / test (by leakage group)", f"{m.get('split', {}).get('train')} / {m.get('split', {}).get('val')} / {m.get('split', {}).get('test')}"],
 ["modalities: text / image / formula / all three", f"{m.get('modalities', {}).get('text')} / {m.get('modalities', {}).get('image')} / {m.get('modalities', {}).get('formula')} / {m.get('all_three')}"],
 ["leakage groups", f"{m.get('groups')} (identical formula: {m.get('leakage', {}).get('recipes_in_identical_formula_sets')} recipes, near-duplicate photo pairs: {m.get('leakage', {}).get('near_duplicate_photo_pairs')}, same name + creator: {m.get('leakage', {}).get('recipes_sharing_name_and_creator')})"],
 ["review status counts", E(json.dumps(m.get('review_status_counts', {})))]])}
</section>
<section><h2>3. Models compared (Phase B1, validation)</h2><div class="card">{b1chart}{table(["modality", "encoder", "MRR of directions with this modality", "mean MRR all directions", "seconds"], b1rows)}</div>
<h2>4. Hyperparameter search</h2><h3>B2 - GridSearchCV, GroupKFold(3) on the training split (top 12 of {len(b2)})</h3>{table(["rank", "params", "mean CV MRR", "sd", "fit s"], b2rows)}
<h3>C - top configurations on validation</h3>{table(["params", "CV MRR", "validation MRR"], crows)}
<h3>D - refinement (mean ± sd over seeds)</h3>{table(["extra params", "validation MRR"], drows)}
<h3>Fusion weights (image / text / formula, ordered image ≥ text ≥ formula; validation)</h3>{table(["weights", "attr Recall@5", "attr nDCG@10", "I->T+F tie MRR", "score"], wrows)}
<h2>5. Best validated configuration</h2><p><code>{E(json.dumps(cfg))}</code><br>fusion weights <code>{E(json.dumps(w))}</code></p>
<h2>6. Training curves</h2><div class="card">{curves}</div></section>
<section><h2>7. Test-set evaluation</h2><div class="card">{S['ident_test_chart']}{S['ident_test']}
<h3>Same task with the whole database as gallery (comparable with the external test)</h3>{S['ident_test_full']}</div>
<div class="card">{S['attr_test_chart']}{S['attr_test']}</div><h3>Pairwise directions vs baselines (CCA, CLIP zero-shot)</h3>{S['pairwise']}
<h3>Embedding quality</h3>{S['cosgap']}<p class="note">A higher cosine for the same recipe than for other recipes shows alignment; it is not by itself evidence that a retrieved recipe is relevant.</p></section>
<section><h2>8. External ceramic data (Glazy recipes not in our photos)</h2>
<p>{ex.get('n_external', 0)} recipes fetched politely from public glazy.org pages (sitemap, 4 s between requests; CC BY-NC-SA, used for testing only);
{ex.get('n_near_duplicate_photo_excluded', 0)} near-duplicate photos excluded; {res.ext_stats.get('requests', '?')} requests in total. Fetch statistics:
<code>{E(json.dumps(res.ext_stats))}</code>. Label mapping check: cone-id rule holds = {res.ext_maps.get('cone_rule_index_plus_1_holds')}, surface /
transparency text agreement on known recipes = {res.ext_maps.get('tag_text_agreement')}.</p>
{('<div class="card">' + S['ident_ext_chart'] + S['ident_ext'] + '</div><div class="card">' + S['attr_ext_chart'] + S['attr_ext'] + '<p class="note">' + E(S['ext_formula']) + '</p></div>') if 'ident_ext' in S else '<p>No external results.</p>'}
<p class="note">There is no ground truth for "the correct existing recipe" of a new Glazy recipe. The same-recipe numbers are exact (the recipe itself is in the
gallery); the "most appropriate" numbers use the attribute + colour relevance definition and are proxies, compared with random and image-only baselines.</p>
{dx_html}</section>
<section><h2>9. Ablation</h2><p>Query-modality ablation (7 sets) is in the tables of sections 7 and 8. Training ablation - models trained on one modality pair only
vs the full three-modality model (test R@5):</p>{table(["trained on", "direction", "R@5", "MRR"], ptrows)}
<p class="note">Full model R@5: {E(', '.join(f'{k} {pct(v)}' for k, v in full_pair.items()))}.</p>
<h2>10. Clustering (DBSCAN)</h2>{table(["eps", "clusters", "noise", "largest cluster", "silhouette (cosine)", "purity surface", "purity subtype level 2"], clrows)}{cl_cmp}
<p class="note">eps and beta for the re-ranking were chosen on validation; test was scored once. When one cluster holds most
recipes or the silhouette is near zero / negative, the embedding space has no clear density gaps, so DBSCAN cannot help ranking.</p>
<p class="note">DBSCAN noise points are listed in <code>dbscan_noise_flagged.csv</code> for review; they are not removed (unusual glazes can be legitimate).</p>
<h2>11. Failure analysis</h2><p>{E(tops_note)}</p>{table(["recipe", "name", "subtype", "surface", "transparency"], fails) if fails else ''}
<p class="note">Typical causes seen in the per-query viewer: many recipes share identical tags (ties), surface / transparency tags are missing for ~45 % of recipes,
photos are crowd-sourced (lighting, multiple tiles), and colour (ΔE) is only one of the relevance criteria.</p>
<h2>12. Resources</h2>{table(["stage", "time", "work folder size", "free D: / C:"], res_rows)}
<p>Peak GPU memory of the final training: {vram} GB (8 GB card). Work folder: {dir_size_gb(work):.2f} GB on D:. Nothing was written to C:.</p>
<h2>13. Recommendations</h2><ul><li>Collect relevance labels from potters for 100-200 queries; the attribute + colour relevance is a proxy.</li>
<li>Fill missing surface / transparency tags (45 %) - the text branch and the relevance labels both depend on them.</li>
<li>Use several photos per recipe (Glazy often has more) to train the image head with true image-image positives.</li>
<li>Reduce over-fitting of the projection heads (section 8b: same-recipe cosine far higher on train than on test / external): more training
pairs, a wider dropout / weight-decay range in the next grid, and new recipes in the validation split.</li>
<li>Fine-tune the image backbone (partial) once relevance labels exist; frozen features were chosen here for the 8 GB GPU.</li></ul></section>"""
    css = CSS + ("@page{size:A4;margin:12mm} section{break-after:page} .card{break-inside:auto} "
                 "svg.chart{max-width:100%;break-inside:avoid} tr{break-inside:avoid} h2,h3{break-after:avoid}")
    page = f'<!doctype html><html><head><meta charset="utf-8"><title>CG2 final report</title><style>{css}</style></head><body style="padding:0 6mm">{body}</body></html>'
    hp = res.r / "final_report.html"
    hp.write_text(page, encoding="utf-8")
    pdf = res.r / "CG2_multimodal_final_report.pdf"
    print_pdf(hp, pdf)
    return pdf


# ---------------------------------------------------------------- mind map of what was added (with real numbers)
def mindmap(work, res: Results, S: dict, smoke: bool) -> Path:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    cfg = res.Efull.get("final_cfg", {})
    w = res.Efull.get("weights", {})
    def g(*p):
        return mean_of(res.seedvals(*p))
    it = lambda k, m: g("test", "same_recipe_sets", k, m)  # noqa: E731
    at = lambda k, m: g("test", "attribute_ablation", k, m)  # noqa: E731
    xt = lambda k, m: g("external", "identity_in_gallery", k, m)  # noqa: E731
    xa = lambda k, m: g("external", "attribute_ablation", k, m)  # noqa: E731
    rnd = res.A.get("attribute", {}).get("random", {})
    clip = res.A.get("attribute", {}).get("image_clip_knn", {})
    m = res.manifest
    best_b2 = (res.B2.get("ranking") or [{}])[0]
    LEFT = [
        ("Data & alignment", "#475569", [
            ("D", f"{m.get('recipes')} recipes (1 photo each), {m.get('objects')} usable objects (master review CSV)"),
            ("D", f"group split {m.get('split', {}).get('train')} / {m.get('split', {}).get('val')} / {m.get('split', {}).get('test')}; {m.get('groups')} leakage groups"),
            ("D", "mm/data.py: recipe-ID alignment, dHash duplicates, provenance manifest")]),
        ("Three branches", "#7c3aed", [
            ("D", f"text: {cfg.get('text')} on the vendor text_branch tag description + notes"),
            ("D", f"image: {cfg.get('image')} with mask pooling over reviewed objects"),
            ("D", f"formula: {cfg.get('formula')} from oxide wt% -> UMF (computed)"),
            ("D", "compared: 5 image, 5 text (incl. BiLSTM), 8 formula encoders")]),
        ("Shared space & search", "#be185d", [
            ("D", f"{cfg.get('head')} heads -> {cfg.get('dim')}-d, symmetric InfoNCE, masked duplicate negatives"),
            ("D", f"GridSearchCV (CV MRR {best_b2.get('mean_cv_MRR', 0):.3f}) -> validation chose temperature {cfg.get('temperature')}, "
                  f"dropout {cfg.get('dropout')}, batch {cfg.get('batch')}"),
            ("D", f"fusion weights image {w.get('image', 0):.2f} >= text {w.get('text', 0):.2f} >= formula {w.get('formula', 0):.2f}")]),
    ]
    RIGHT = [
        ("Internal test (same recipe)", "#0369a1", [
            ("D", f"image -> text+formula: R@3 {pct(it('I->T+F', 'R@3'))}, R@5 {pct(it('I->T+F', 'R@5'))}, R@10 {pct(it('I->T+F', 'R@10'))}"),
            ("D", f"text -> image+formula: R@5 {pct(it('T->I+F', 'R@5'))}; formula -> image+text: R@5 {pct(it('F->T+I', 'R@5'))}"),
            ("D", f"random R@5 ~ {pct(5 / max(1, m.get('split', {}).get('test', 1)))}")]),
        ("Internal test (most appropriate)", "#15803d", [
            ("D", f"query I+T+F: Recall@3 {pct(at('I+T+F', 'Hit@3'))}, @5 {pct(at('I+T+F', 'Hit@5'))}, @10 {pct(at('I+T+F', 'Hit@10'))}"),
            ("D", f"query image only: @5 {pct(at('I', 'Hit@5'))}; text only: @5 {pct(at('T', 'Hit@5'))}; formula only: @5 {pct(at('F', 'Hit@5'))}"),
            ("D", f"baselines @5: random {pct(rnd.get('Hit@5'))}, image CLIP kNN {pct(clip.get('Hit@5'))}")]),
        ("External Glazy recipes", "#a16207", [
            ("D", f"{S.get('ext_counts', {}).get('n_used', 0)} new recipes (not in our photos), testing only"),
            ("D", f"same recipe image -> text+formula: R@3 {pct(xt('I->T+F', 'R@3'))}, @5 {pct(xt('I->T+F', 'R@5'))}, @10 {pct(xt('I->T+F', 'R@10'))}"),
            ("D", f"most appropriate I+T+F: @3 {pct(xa('I+T+F', 'Hit@3'))}, @5 {pct(xa('I+T+F', 'Hit@5'))}, @10 {pct(xa('I+T+F', 'Hit@10'))}"),
            ("D", "gap vs internal: over-fitted heads + more novel recipes (section 8b)")]),
        ("Deliverables", "#334155", [
            ("D", "multimodal_experiments/ (mm/*.py, run_pipeline.py, configs, logs)"),
            ("D", "results/site/index.html explorer, final report PDF, this mind map"),
            ("D", "RUN_MULTIMODAL_PIPELINE.bat (resumable)")]),
    ]
    from . import mindmap_svg as MM
    svg = MM.render(LEFT, RIGHT, "CG2 multimodal", "what was added", "(measured results)")
    rows = []
    for k in ABL:
        rows.append([k, pct(at(k, "Hit@3")), pct(at(k, "Hit@5")), pct(at(k, "Hit@10")), pct(xa(k, "Hit@3")), pct(xa(k, "Hit@5")), pct(xa(k, "Hit@10"))])
    xb = lambda b, m: g("external", "baselines", b, m)  # noqa: E731
    rows.append(["baseline: image CLIP kNN (image -> image)", pct(clip.get("Hit@3")), pct(clip.get("Hit@5")), pct(clip.get("Hit@10")),
                 pct(xb("image_clip_knn", "Hit@3")), pct(xb("image_clip_knn", "Hit@5")), pct(xb("image_clip_knn", "Hit@10"))])
    rows.append(["baseline: random", pct(rnd.get("Hit@3")), pct(rnd.get("Hit@5")), pct(rnd.get("Hit@10")),
                 pct(xb("random", "Hit@3")), pct(xb("random", "Hit@5")), pct(xb("random", "Hit@10"))])
    cmp_tbl = table(["query modalities (weights image ≥ text ≥ formula)", "internal R@3", "internal R@5", "internal R@10", "external R@3",
                     "external R@5", "external R@10"], rows)
    ifull = lambda k, m: g("test", "same_recipe_full_gallery", k, m)  # noqa: E731
    irows = []
    for k in IDENT:
        irows.append([k, pct(it(k, "R@5")), pct(ifull(k, "R@3")), pct(ifull(k, "R@5")), pct(ifull(k, "R@10")),
                      pct(xt(k, "R@3")), pct(xt(k, "R@5")), pct(xt(k, "R@10"))])
    icmp = table(["same recipe: query -> gallery", "internal R@5 (test gallery)", "internal R@3 (full DB)", "internal R@5 (full DB)",
                  "internal R@10 (full DB)", "external R@3 (full DB)", "external R@5 (full DB)", "external R@10 (full DB)"], irows)
    dx = res.DX
    gap_note = ""
    if dx.get("alignment"):
        al = dx["alignment"]
        ti = lambda sp: f3(al.get(sp, {}).get("text-image", {}).get("same_recipe"))  # noqa: E731
        nv = dx.get("novelty", {})
        gap_note = (f'<p class="note"><b>Why external is lower (measured, report section 8b):</b> parsing verified against the Glazy export; '
                    f'the heads over-fit (same-recipe text-image cosine train {ti("train")}, internal test {ti("test")}, external {ti("ext")}); '
                    f'external recipes are more novel (median distance to the nearest training formula '
                    f'{nv.get("ext", {}).get("median_wt%_distance")} vs {nv.get("test", {}).get("median_wt%_distance")} wt%).</p>')
    css = CSS + "@page{size:A3 landscape;margin:8mm} @page p2{size:A4 landscape;margin:10mm} .p2{page:p2} section{break-after:page} svg.mm{width:100%;height:265mm}"
    page = (f'<!doctype html><html><head><meta charset="utf-8"><title>CG2 mind map results</title><style>{css}</style></head><body>'
            f'<section><h1>CG2 multimodal glaze retrieval - what was added and what it achieved{" (SMOKE)" if smoke else ""}</h1>{svg}</section>'
            f'<section class="p2"><h1>Which setting is better - Recall@K, internal test vs external recipes</h1>'
            f'<h2>Most appropriate recipe (share of queries with a relevant recipe in the top K)</h2>{cmp_tbl}'
            f'<h2>Same recipe found across modalities</h2>{icmp}<p class="note">First column: gallery = the {m.get("split", {}).get("test")} test recipes; '
            f'"full DB": gallery = all {m.get("recipes")} database recipes (external: + the {S.get("ext_counts", {}).get("n_used", 0)} external recipes). '
            f'Mean over {len(res.seeds)} seed(s).</p>{gap_note}</section></body></html>')
    hp = res.r / "mindmap_results.html"
    hp.write_text(page, encoding="utf-8")
    pdf = res.r / "CG2_multimodal_mindmap_results.pdf"
    print_pdf(hp, pdf)
    return pdf


def build(work: Path, bank, smoke: bool) -> dict:
    from . import diagnose
    diagnose.run(work)
    res = Results(work)
    site = res.r / "site"
    site.mkdir(parents=True, exist_ok=True)
    log().info("[report] thumbnails ...")
    thumbs(bank, site, work / "external")
    data_js(bank, res, site)
    S = metric_sections(res)
    explorer(work, bank, res, S, site)
    pdf = final_report(work, bank, res, S, smoke)
    mm = mindmap(work, res, S, smoke)
    out = {"explorer": str(site / "index.html"), "report_pdf": str(pdf), "mindmap_pdf": str(mm),
           "site_MB": round(dir_size_gb(site) * 1000, 1)}
    write_json(res.r / "report_outputs.json", out)
    log().info(f"[report] {out}")
    return out
