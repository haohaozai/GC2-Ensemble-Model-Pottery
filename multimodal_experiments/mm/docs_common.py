"""Helpers (HTML, figures, flow diagrams, PDF printing) for make_attribute_report.py, which builds the PDF documents of
the ceramic recognition upgrade into ONE folder (docs/CG2_PDF_reports):

  CG2_Ceramic_Recognition_Technical_Report.pdf   chapters 1-10 (architecture, modifications, models, attributes,
                                                 review, fine-tuning, evaluation, failures, final architecture, running)
  CG2_User_Manual_Review_and_Search.pdf          how to use the review app (incl. review_tile) and Glaze Search
  CG2_Algorithms_and_Metrics.pdf                 every algorithm: what / why / parameters; every metric and how it was
                                                 adapted to the data (for the project report)
Every number is read from the result files of run_attributes.py / run_pipeline.py; missing results are shown as
"not run". Earlier PDFs (plan, multimodal report, mind map) are copied into the same folder.
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
import shutil
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
CODE = EXP.parent
sys.path.insert(0, str(EXP))
from mm.common import SEG, obj_file, read_csv, read_json  # noqa: E402
from mm import charts as CH  # noqa: E402
from mm.report import print_pdf  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ATTR = EXP / "work" / "attributes_v1"
OUT = CODE / "docs" / "CG2_PDF_reports"
SHOTS = OUT / "_screenshots"
E = lambda s: html.escape(str(s))  # noqa: E731


def pct(v, d=1):
    return "–" if v is None else f"{100 * v:.{d}f}%"


def f3(v):
    return "–" if v is None else f"{v:.3f}"


def table(head, rows, cls=""):
    h = "".join(f"<th>{E(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def img_uri(path_or_img, maxw=900, fmt="JPEG"):
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    im = im.convert("RGBA" if fmt == "PNG" else "RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=85)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def badge(kind):
    col = {"existing": "#5f6b7a", "new": "#1f7a45", "tested": "#2a78d6", "experimental": "#b4540a", "not implemented": "#9b2c2c"}[kind]
    return f'<span class="st" style="border-color:{col};color:{col}">{kind}</span>'


CSS = """
@page{size:A4;margin:14mm 13mm}
body{font:10.5pt/1.45 "Segoe UI",Arial,sans-serif;color:#1d2433;margin:0}
h1{font-size:21pt;margin:0 0 4px}h2{font-size:15pt;margin:22px 0 8px;border-bottom:2px solid #1d2433;padding-bottom:3px;break-after:avoid}
h3{font-size:12pt;margin:14px 0 6px;break-after:avoid}h4{font-size:10.5pt;margin:10px 0 4px;break-after:avoid}
section.ch{break-before:page}
p{margin:5px 0}.muted{color:#5f6b7a}.small{font-size:9pt}
table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-size:9pt;break-inside:auto}
th,td{border:1px solid #d5d9e0;padding:4px 6px;text-align:left;vertical-align:top}th{background:#eef1f5}
tr{break-inside:avoid}
code{font:9pt Consolas,monospace;background:#f2f4f7;padding:0 3px;border-radius:3px}
.st{display:inline-block;border:1.5px solid;border-radius:10px;padding:0 7px;font-size:8.5pt;font-weight:600;white-space:nowrap}
.box{border:1px solid #d5d9e0;border-radius:8px;padding:8px 10px;margin:8px 0;break-inside:avoid}
.warn{border-left:4px solid #b4540a;background:#fff7ed}
.ok{border-left:4px solid #1f7a45;background:#f0faf4}
.fig{text-align:center;margin:8px 0;break-inside:avoid}.fig img{max-width:100%;border:1px solid #d5d9e0;border-radius:6px}
.fig .cap{font-size:9pt;color:#5f6b7a;margin-top:3px}
.toc li{margin:2px 0}
svg.chart{max-width:100%}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
"""


def page(title, body):
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{E(title)}</title><style>{CSS}</style></head>'
            f'<body>{body}</body></html>')


def write_pdf(name, title, body):
    OUT.mkdir(parents=True, exist_ok=True)
    hp = OUT / "_html" / (name + ".html")
    hp.parent.mkdir(exist_ok=True)
    hp.write_text(page(title, body), encoding="utf-8")
    pdf = OUT / (name + ".pdf")
    print_pdf(hp, pdf)
    return pdf


# ---------------------------------------------------------------- figures
def fig_2152():
    """2152_1: object | measured regions | old area-ranked palette vs region features."""
    oid = "2152_1"
    a = np.asarray(Image.open(obj_file("smaller_object_rgba_raw", f"{oid}.png")).convert("RGBA"))
    reg = read_json(ATTR / "regions" / f"{oid}.json")
    lbl = np.asarray(Image.open(ATTR / "regions" / f"{oid}_labels.png"))
    H, W = lbl.shape
    im = Image.fromarray(a).resize((W, H), Image.LANCZOS)
    bg = Image.new("RGB", (W, H), (245, 245, 245))
    bg.paste(im, mask=im.split()[3])
    rm = np.full((H, W, 3), 245, np.uint8)
    for r in reg["regions"]:
        rm[lbl == r["id"]] = [int(r["hex"][i:i + 2], 16) for i in (1, 3, 5)]
    edge = np.zeros(lbl.shape, bool)
    edge[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
    edge[1:, :] |= lbl[1:, :] != lbl[:-1, :]
    rm[edge & (lbl > 0)] = 0
    rim = Image.fromarray(rm)
    d = ImageDraw.Draw(rim)
    for r in reg["regions"]:
        ys, xs = np.nonzero(lbl == r["id"])
        d.text((int(xs.mean()) - 6, int(ys.mean()) - 6), f"R{r['id']}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    sheet = Image.new("RGB", (2 * W + 20, H), (255, 255, 255))
    sheet.paste(bg, (0, 0))
    sheet.paste(rim, (W + 20, 0))
    rows = [[f'<span style="display:inline-block;width:14px;height:14px;background:{r["hex"]};border:1px solid #999;vertical-align:-2px"></span> '
             f'R{r["id"]} {E(r["name"])}', pct(r["area_frac"]), E(r["position"]), f'{r["distinctiveness_de00"]:.1f}',
             f'{r["contrast_de00"]:.1f}', f'{r["auto_importance"]:.2f}', E(", ".join(r["flags"]) or "–")] for r in reg["regions"]]
    rev = {r["object_id"]: r for r in read_csv(SEG / "review" / "object_review.csv")}.get(oid, {})
    old = " · ".join(f'{rev.get(f"color{i}_name")} {rev.get(f"color{i}_pct")}%' for i in (1, 2, 3) if rev.get(f"color{i}_hex"))
    return (f'<div class="fig"><img src="{img_uri(sheet, 700)}"><div class="cap">2152_1.jpg - left: object cut-out; right: measured colour '
            f'regions (regions_v1). Recipe 2152 is a SLIP ("Slip Base for Leatherhard Application"): the red part need not be glaze.</div></div>'
            + table(["region", "area", "position", "distinctiveness ΔE00", "contrast ΔE00", "auto importance", "flags"], rows)
            + f'<p class="small">Previous palette (ColorThief, ranked by area only): {E(old)}. Review suggestion: '
              f'{E("; ".join(reg["summary"]["reasons"]))} (priority {reg["summary"]["review_priority"]}).</p>')


def flow_svg(steps, w=760, title=""):
    """simple left-to-right box flow with arrows; steps: [(label, kind)]"""
    n = len(steps)
    bw, gap, h = (w - 20 - (n - 1) * 18) / n, 18, 64
    col = {"existing": "#e8ebf0", "new": "#e3f4ec", "human": "#fdf0dc", "eval": "#e6f0fb"}
    out = [f'<svg viewBox="0 0 {w} {h + 26}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI,Arial" font-size="10">']
    if title:
        out.append(f'<text x="2" y="12" font-weight="700" font-size="11">{E(title)}</text>')
    for i, (lab, kind) in enumerate(steps):
        x = 10 + i * (bw + gap)
        out.append(f'<rect x="{x:.1f}" y="20" width="{bw:.1f}" height="{h}" rx="7" fill="{col[kind]}" stroke="#8a93a3"/>')
        words, lines, cur = lab.split(), [], ""
        for wd in words:
            if len(cur) + len(wd) > max(10, int(bw / 6.2)):
                lines.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        lines.append(cur)
        for k, ln in enumerate(lines[:5]):
            out.append(f'<text x="{x + bw / 2:.1f}" y="{20 + h / 2 - (len(lines[:5]) - 1) * 6.5 + k * 13:.1f}" text-anchor="middle" dominant-baseline="middle">{E(ln)}</text>')
        if i < n - 1:
            ax = x + bw
            out.append(f'<path d="M{ax + 2:.1f} {20 + h / 2} L{ax + gap - 3:.1f} {20 + h / 2}" stroke="#5f6b7a" stroke-width="1.6" marker-end="url(#a)"/>')
    out.insert(1, '<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#5f6b7a"/></marker></defs>')
    out.append("</svg>")
    return "".join(out)


def legend_flow():
    return ('<p class="small"><span style="background:#e8ebf0;padding:1px 6px;border:1px solid #8a93a3">existing</span> '
            '<span style="background:#e3f4ec;padding:1px 6px;border:1px solid #8a93a3">new in this upgrade</span> '
            '<span style="background:#fdf0dc;padding:1px 6px;border:1px solid #8a93a3">human step</span> '
            '<span style="background:#e6f0fb;padding:1px 6px;border:1px solid #8a93a3">evaluation</span></p>')
