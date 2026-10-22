"""Small SVG chart helpers (inline SVG works in the HTML explorer and prints into the PDF report).

Palette: reference categorical slots 1-3 (validated in headless Edge with the dataviz validator: light PASS
except aqua contrast 2.74:1 -> every chart has a table twin and direct value labels; dark PASS).
Thin bars with 4 px rounded data ends, 2 px gaps, hairline solid grid, legend for >= 2 series, hover = <title>.
"""
from __future__ import annotations

import html

E = html.escape
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED, GRID, AXIS, INK2 = "#898781", "#e1e0d9", "#c3c2b7", "#52514e"


def _bar_path(x, y, w, h):
    """horizontal bar from x (baseline) with a 4 px rounded data end on the right."""
    r = min(4, h / 2, max(0.0, w))
    if w <= 0.5:
        return ""
    return (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h - r:.1f} "
            f"Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")


def hbars(title: str, categories: list[str], series: dict[str, list], vmax: float = 1.0, pct=True, width=760,
          note: str = "") -> str:
    """grouped horizontal bars: categories (rows) x series (<= 3 colours, fixed order)."""
    names = list(series)
    bh, gap, ggap = 9, 2, 12
    longest = max((len(str(c)) for c in categories), default=10)
    left, right, top = int(min(330, max(150, 6.1 * longest + 16))), 70, 34 + (18 if len(names) > 1 else 0)
    gh = len(names) * (bh + gap) - gap
    H = top + len(categories) * (gh + ggap) + 26
    pw = width - left - right
    s = [f'<svg class="chart" viewBox="0 0 {width} {H}" width="100%" role="img" aria-label="{E(title)}" '
         f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         f'<text x="0" y="16" font-size="13" font-weight="600" fill="#0b0b0b">{E(title)}</text>']
    if len(names) > 1:
        lx = 0
        for i, n in enumerate(names):
            s.append(f'<rect x="{lx}" y="25" width="10" height="10" rx="2" fill="{SERIES[i]}"/>'
                     f'<text x="{lx + 14}" y="34" font-size="11" fill="{INK2}">{E(n)}</text>')
            lx += 24 + 6.5 * len(n)
    for t in range(0, 6):
        v = vmax * t / 5
        x = left + pw * t / 5
        s.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{H - 22}" stroke="{GRID}" stroke-width="1"/>'
                 f'<text x="{x:.1f}" y="{H - 8}" font-size="10" fill="{MUTED}" text-anchor="middle">'
                 f'{(f"{v * 100:.0f}%" if pct else f"{v:.2g}")}</text>')
    s.append(f'<line x1="{left}" y1="{top - 4}" x2="{left}" y2="{H - 22}" stroke="{AXIS}" stroke-width="1"/>')
    y = top
    for ci, c in enumerate(categories):
        s.append(f'<text x="{left - 8}" y="{y + gh / 2 + 4:.1f}" font-size="11" fill="#0b0b0b" text-anchor="end">{E(c)}</text>')
        for si, n in enumerate(names):
            v = series[n][ci]
            yy = y + si * (bh + gap)
            if v is None:
                s.append(f'<text x="{left + 4}" y="{yy + bh - 1}" font-size="9" fill="{MUTED}">n/a</text>')
                continue
            w = pw * min(1.0, v / vmax) if vmax else 0
            lab = f"{v * 100:.1f}%" if pct else f"{v:.3g}"
            s.append(f'<path d="{_bar_path(left, yy, w, bh)}" fill="{SERIES[si]}"><title>{E(c)} - {E(n)}: {lab}</title></path>')
            s.append(f'<text x="{left + w + 4:.1f}" y="{yy + bh - 1}" font-size="9.5" fill="{INK2}">{lab}</text>')
        y += gh + ggap
    if note:
        s.append(f'<text x="0" y="{H - 8}" font-size="9.5" fill="{MUTED}">{E(note)}</text>')
    s.append("</svg>")
    return "".join(s)


def lines(title: str, xs_list: list[list], ys_list: list[list], names: list[str], ylab="", width=760, height=240) -> str:
    """line chart (<= 3 series, one y axis)."""
    left, right, top, bottom = 52, 20, 40, 34
    pw, ph = width - left - right, height - top - bottom
    xmax = max(max(x) for x in xs_list)
    ys = [v for y in ys_list for v in y]
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.1 or 0.05
    ymin, ymax = max(0, ymin - pad), ymax + pad
    fx = lambda x: left + pw * (x - 1) / max(1, xmax - 1)  # noqa: E731
    fy = lambda v: top + ph * (1 - (v - ymin) / (ymax - ymin))  # noqa: E731
    s = [f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{E(title)}" '
         f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         f'<text x="0" y="16" font-size="13" font-weight="600" fill="#0b0b0b">{E(title)}</text>']
    lx = 0
    for i, n in enumerate(names):
        s.append(f'<rect x="{lx}" y="23" width="12" height="3" rx="1.5" fill="{SERIES[i]}"/>'
                 f'<text x="{lx + 16}" y="29" font-size="11" fill="{INK2}">{E(n)}</text>')
        lx += 26 + 6.5 * len(n)
    for t in range(5):
        v = ymin + (ymax - ymin) * t / 4
        y = fy(v)
        s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID}"/>'
                 f'<text x="{left - 6}" y="{y + 3:.1f}" font-size="10" fill="{MUTED}" text-anchor="end">{v:.2f}</text>')
    s.append(f'<text x="{left + pw / 2}" y="{height - 4}" font-size="10" fill="{MUTED}" text-anchor="middle">epoch</text>')
    if ylab:
        s.append(f'<text x="{left}" y="{top - 6}" font-size="10" fill="{MUTED}">{E(ylab)}</text>')
    for i, (xs, yv) in enumerate(zip(xs_list, ys_list)):
        pts = " ".join(f"{fx(x):.1f},{fy(v):.1f}" for x, v in zip(xs, yv))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{SERIES[i]}" stroke-width="2" stroke-linejoin="round"/>')
        b = max(range(len(yv)), key=lambda k: yv[k])
        s.append(f'<circle cx="{fx(xs[b]):.1f}" cy="{fy(yv[b]):.1f}" r="4" fill="{SERIES[i]}" stroke="#fcfcfb" stroke-width="2">'
                 f'<title>{E(names[i])}: best {yv[b]:.3f} at epoch {xs[b]}</title></circle>')
    s.append("</svg>")
    return "".join(s)
