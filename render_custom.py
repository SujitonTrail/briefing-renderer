"""Wrapper around render_v6.py (unchanged) that adds two extra paper-cut panel backgrounds:
  "ai"        - neural-network nodes, chat bubbles and sparkles on a deep navy sheet
  "evolution" - karst hills, a cave mouth, ferns and fossil-bearing rock layers on sand paper
Usage is identical to render_v6.py:  python3 render_custom.py story.json out.mp4 [--still SPECS OUTDIR]
"""
import math
import numpy as np
import render_v6  # noqa: F401  (wires up v3/v4/v5/v6 exactly as the original)
import render_v3 as B
import render_v4 as V4
from PIL import Image
from render_v4 import PC, texturize, W
from render_v3 import MAP_H

_orig_panel = V4.make_paper_panel


def _ai_layers(rng, PW, PH):
    layers = []
    def new(): return PC(PW, PH, ss=1, pad=0)
    # far: faint dot grid, like a chip / data field
    c = new()
    dots = [(x, y) for x in range(20, PW, 46) for y in range(20, PH, 46)]
    c.layer(lambda d, P, R: [d.ellipse([P(x - 3, y - 3), P(x + 3, y + 3)], fill=255) for x, y in dots], (44, 58, 100), sh=None, rough=0)
    layers.append((c.sprite().img, 0.15))
    # mid: neural-network graphs on both sides (centre left clear for the number)
    c = new()
    def net(x0, x1, y0, y1, cols, rows_list, seed):
        xs = np.linspace(x0, x1, cols)
        nodes = []
        for xi, rows in zip(xs, rows_list):
            ys = np.linspace(y0, y1, rows + 2)[1:-1]
            nodes.append([(xi, yi) for yi in ys])
        edges = [(a, b) for k in range(cols - 1) for a in nodes[k] for b in nodes[k + 1]]
        c.layer(lambda d, P, R: [d.line([P(*a), P(*b)], fill=255, width=3) for a, b in edges], (74, 112, 186), sh=None, rough=0.3, seed=seed)
        c.layer(lambda d, P, R: [d.ellipse([P(x - 17, y - 17), P(x + 17, y + 17)], fill=255) for col in nodes for x, y in col], (226, 236, 246), sh=(2, 3, 3, .45), rough=0.6, seed=seed + 1)
        c.layer(lambda d, P, R: [d.ellipse([P(x - 8, y - 8), P(x + 8, y + 8)], fill=255) for col in nodes for x, y in col], (96, 206, 220), sh=None, rough=0.3, seed=seed + 2)
    net(PW * .09, PW * .30, PH * .05, PH * .24, 3, (2, 3, 2), 10)
    net(PW * .70, PW * .91, PH * .05, PH * .24, 3, (2, 3, 2), 20)
    layers.append((c.sprite().img, 0.45))
    # near: chat bubbles and sparkles
    c = new()
    for i, (bx, by, w_, h_, flip) in enumerate(((PW * .44, PH * .09, 150, 74, 0), (PW * .58, PH * .17, 124, 64, 1))):
        def bub(d, P, R, bx=bx, by=by, w_=w_, h_=h_, flip=flip):
            d.rounded_rectangle([P(bx - w_ / 2, by - h_ / 2), P(bx + w_ / 2, by + h_ / 2)], radius=30, fill=255)
            tx = bx + (w_ / 2 - 36) * (1 if flip else -1)
            d.polygon([P(tx - 14, by + h_ / 2 - 2), P(tx + 14, by + h_ / 2 - 2), P(tx + (22 if flip else -22), by + h_ / 2 + 26)])
        c.layer(bub, (246, 242, 230) if i == 0 else (250, 204, 84), sh=(3, 4, 4, .5), rough=0.8, seed=30 + i)
        dots = [(bx + k * 30, by) for k in (-1, 0, 1)]
        c.layer(lambda d, P, R, dots=dots: [d.ellipse([P(x - 8, y - 8), P(x + 8, y + 8)], fill=255) for x, y in dots], (40, 52, 90), sh=None, rough=0.2, seed=40 + i)
    def star(d, P, R, x, y, r):
        q = r * 0.28
        d.polygon([P(x, y - r), P(x + q, y - q), P(x + r, y), P(x + q, y + q), P(x, y + r), P(x - q, y + q), P(x - r, y), P(x - q, y - q)])
    for i, (sx, sy, r) in enumerate(((PW * .09, PH * .52, 24), (PW * .93, PH * .50, 20), (PW * .34, PH * .07, 16), (PW * .12, PH * .80, 22), (PW * .90, PH * .80, 26))):
        c.layer(lambda d, P, R, sx=sx, sy=sy, r=r: star(d, P, R, sx, sy, r), (250, 204, 84), sh=(1, 2, 2, .4), rough=0.3, seed=50 + i)
    layers.append((c.sprite().img, 0.8))
    return layers


def _evo_layers(rng, PW, PH):
    layers = []
    def new(): return PC(PW, PH, ss=1, pad=0)
    # far: karst tower hills (southern China landscape), soft
    c = new()
    hz = PH * .58
    towers = []
    x = -40
    while x < PW + 40:
        w_ = rng.uniform(90, 170); h_ = rng.uniform(140, 300); towers.append((x, w_, h_)); x += w_ * rng.uniform(.55, .85)
    def hills(d, P, R):
        for x0, w_, h_ in towers:
            d.rounded_rectangle([P(x0, hz - h_), P(x0 + w_, hz + 40)], radius=int(w_ * .45), fill=255)
        d.rectangle([P(0, hz), P(PW, PH)], fill=255)
    c.layer(hills, (194, 176, 140), sh=(0, -2, 4, .3), rough=1.2, seed=3)
    layers.append((c.sprite().img, 0.2))
    # mid: a cave mouth on the left, ferns on the right
    c = new()
    cx, cy = PW * .10, PH * .50
    c.layer(lambda d, P, R: d.ellipse([P(cx - 260, cy - 300), P(cx + 230, cy + 330)], fill=255), (150, 118, 86), sh=(3, 4, 6, .45), rough=2.0, seed=5)
    c.layer(lambda d, P, R: d.pieslice([P(cx - 110, cy - 90), P(cx + 130, cy + 240)], 180, 360, fill=255), (62, 46, 34), sh=None, rough=1.4, seed=6)
    c.layer(lambda d, P, R: d.rectangle([P(cx - 110, cy + 74), P(cx + 130, cy + 150)], fill=255), (62, 46, 34), sh=None, rough=1.0, seed=7)
    def fern(d, P, R, bx, by, ln, ang, n=9):
        ca, sa = math.cos(ang), math.sin(ang)
        d.line([P(bx, by), P(bx + ln * ca, by + ln * sa)], fill=255, width=7)
        for k in range(1, n):
            u = k / n; px, py = bx + ln * u * ca, by + ln * u * sa; lf = 70 * (1 - u * .7)
            for s in (-1, 1):
                a2 = ang + s * 1.0
                d.polygon([P(px, py), P(px + lf * math.cos(a2) - 9 * math.sin(a2), py + lf * math.sin(a2) + 9 * math.cos(a2)), P(px + lf * math.cos(a2), py + lf * math.sin(a2))])
    for i, (bx, by, ln, ang) in enumerate(((PW * .96, PH * .72, 300, -2.1), (PW * .90, PH * .74, 260, -1.7), (PW * 1.0, PH * .60, 240, -2.6))):
        c.layer(lambda d, P, R, bx=bx, by=by, ln=ln, ang=ang: fern(d, P, R, bx, by, ln, ang), (96, 132, 80) if i != 1 else (122, 156, 98), sh=(2, 3, 3, .4), rough=0.6, seed=10 + i)
    layers.append((c.sprite().img, 0.45))
    # near: rock strata along the bottom, with fossil teeth and a bone in them
    c = new()
    bands = ((PH * .80, (190, 150, 104)), (PH * .86, (168, 128, 88)), (PH * .92, (142, 106, 72)))
    for i, (y0, col) in enumerate(bands):
        c.layer(lambda d, P, R, y0=y0: d.polygon([P(0, y0)] + [P(x, y0 - 10 * math.sin(x / 140 + y0)) for x in range(0, PW + 40, 40)] + [P(PW, PH), P(0, PH)]), col, sh=(0, -3, 4, .35), rough=1.0, seed=20 + i)
    def tooth(d, P, R, x, y, s):
        d.rounded_rectangle([P(x - 26 * s, y - 22 * s), P(x + 26 * s, y + 8 * s)], radius=int(12 * s), fill=255)
        d.polygon([P(x - 22 * s, y), P(x - 6 * s, y), P(x - 12 * s, y + 36 * s)])
        d.polygon([P(x + 6 * s, y), P(x + 22 * s, y), P(x + 12 * s, y + 36 * s)])
    for i, (tx, ty, s, rot) in enumerate(((PW * .30, PH * .89, 1.0, 0), (PW * .62, PH * .95, 0.85, 0), (PW * .80, PH * .87, 0.9, 0))):
        c.layer(lambda d, P, R, tx=tx, ty=ty, s=s: tooth(d, P, R, tx, ty, s), (244, 236, 214), sh=(1, 2, 2, .45), rough=0.4, seed=30 + i)
    bx, by = PW * .47, PH * .885
    def bone(d, P, R):
        d.rounded_rectangle([P(bx - 70, by - 9), P(bx + 70, by + 9)], radius=8, fill=255)
        for ex in (-70, 70):
            d.ellipse([P(bx + ex - 16, by - 24), P(bx + ex + 6, by - 2)], fill=255)
            d.ellipse([P(bx + ex - 16, by + 2), P(bx + ex + 6, by + 24)], fill=255)
    c.layer(bone, (236, 226, 200), sh=(1, 2, 2, .45), rough=0.4, seed=40)
    layers.append((c.sprite().img, 0.9))
    return layers


def make_paper_panel(kind, seed=3):
    if kind not in ("ai", "evolution"):
        return _orig_panel(kind, seed)
    rng = np.random.default_rng(seed); PW, PH = W + 160, MAP_H + 100
    if kind == "ai":
        bgc, dark, layers = (24, 32, 62), True, _ai_layers(rng, PW, PH)
    else:
        bgc, dark, layers = (228, 214, 186), False, _evo_layers(rng, PW, PH)
    base = texturize(Image.new("RGB", (PW, PH), bgc))
    return {"base": base, "layers": [(texturize(l), f) for l, f in layers], "dark": dark}


V4.make_paper_panel = make_paper_panel

if __name__ == "__main__":
    B.main()
