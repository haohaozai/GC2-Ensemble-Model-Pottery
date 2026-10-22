"""Mind map as inline SVG: root in the centre, branches left / right, leaves as wrapped boxes."""
from __future__ import annotations

import html
import textwrap

E = html.escape
COL = {"D": "#1f7a45", "R": "#2f5bd3", "N": "#b4540a", "Q": "#b42318"}
TAG = {"D": "done", "R": "reused", "N": "new", "Q": "decide"}
FONT, LINE, WRAP = 11.5, 14.5, 60


def _wrap(t):
    return textwrap.wrap(t, WRAP) or [""]


def render(left, right, *root_lines) -> str:
    W, top = 1560, 40
    cx = W / 2
    leaf_w, bx_off, lx_off = 410, 215, 330

    def side(branches):
        blocks = []
        for name, col, leaves in branches:
            hs = [len(_wrap(t)) * LINE + 9 for _, t in leaves]
            blocks.append((name, col, leaves, hs, sum(hs) + 5 * (len(hs) - 1)))
        return blocks, sum(b[4] for b in blocks) + 22 * (len(blocks) - 1)
    lb, lt = side(left)
    rb, rt = side(right)
    H = max(lt, rt) + top + 30
    cy = top + max(lt, rt) / 2
    out = [f'<svg class="mm" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" '
           f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">']
    for blocks, total, sign in ((lb, lt, -1), (rb, rt, 1)):
        y = top + (max(lt, rt) - total) / 2
        for name, col, leaves, hs, bh in blocks:
            bcy = y + bh / 2
            bx = cx + sign * bx_off
            out.append(f'<path d="M{cx + sign * 95},{cy} C{cx + sign * 150},{cy} {bx - sign * 120},{bcy} {bx - sign * 88},{bcy}" '
                       f'fill="none" stroke="{col}" stroke-width="3" opacity=".75"/>')
            ly = y
            for (st, t), h in zip(leaves, hs):
                lcy = ly + h / 2
                lx = cx + sign * lx_off
                out.append(f'<path d="M{bx + sign * 88},{bcy} C{bx + sign * 105},{bcy} {lx - sign * 25},{lcy} {lx},{lcy}" '
                           f'fill="none" stroke="{col}" stroke-width="1.4" opacity=".6"/>')
                x0 = lx if sign > 0 else lx - leaf_w
                out.append(f'<rect x="{x0}" y="{ly}" width="{leaf_w}" height="{h}" rx="6" fill="#ffffff" stroke="{col}" stroke-opacity=".45"/>'
                           f'<rect x="{x0}" y="{ly}" width="5" height="{h}" rx="2" fill="{COL[st]}"/>')
                for i, line in enumerate(_wrap(t)):
                    out.append(f'<text x="{x0 + 12}" y="{ly + 14 + i * LINE}" font-size="{FONT}" fill="#1d2433">{E(line)}</text>')
                ly += h + 5
            nm = textwrap.wrap(name, 22)
            out.append(f'<rect x="{bx - 88}" y="{bcy - 22}" width="176" height="44" rx="22" fill="{col}"/>')
            for i, line in enumerate(nm):
                out.append(f'<text x="{bx}" y="{bcy + 5 - 7 * (len(nm) - 1) + i * 14}" font-size="12.5" font-weight="700" '
                           f'fill="#fff" text-anchor="middle">{E(line)}</text>')
            y += bh + 22
    out.append(f'<ellipse cx="{cx}" cy="{cy}" rx="98" ry="56" fill="#1d2433"/>')
    for i, line in enumerate(root_lines):
        out.append(f'<text x="{cx}" y="{cy - 16 + i * 17}" font-size="{14 if i < 2 else 11}" font-weight="{700 if i < 2 else 400}" '
                   f'fill="#fff" text-anchor="middle">{E(line)}</text>')
    out.append("</svg>")
    return "".join(out)
