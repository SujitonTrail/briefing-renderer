#!/usr/bin/env python3
"""The Briefing — explainer-style news video renderer.
Usage: python3 render.py story.json out.mp4 [--still SPECS OUTDIR]
"""
import json, math, os, subprocess, sys, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NEWS60_ROOT", os.path.dirname(HERE))
MODEL = os.path.join(ROOT, "models/kokoro-v1.0.onnx")
VOICES = os.path.join(ROOT, "models/voices-v1.0.bin")
SHP = os.path.join(ROOT, "geo/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp")
FD = "/usr/share/fonts/opentype/inter/"

W, H, FPS = 1080, 1920, 30
PAPER = (239, 233, 221); INK = (20, 20, 20); YEL = (255, 210, 63)
RED = (228, 70, 46); LAND = (218, 210, 194); LAND2 = (186, 176, 158); FAINT = (226, 219, 205)
MAP_Y, MAP_H = 250, 960

def F(name, size): return ImageFont.truetype(FD + name, size)
F_HEAD = F("InterDisplay-Black.otf", 96)
F_TITLE = F("InterDisplay-Black.otf", 150)
F_TAG = F("Inter-Bold.otf", 36)
F_NUM = F("InterDisplay-Black.otf", 40)
F_CAP = F("Inter-SemiBold.otf", 50)
F_BIG = F("InterDisplay-Black.otf", 330)
F_STATB = F("InterDisplay-Black.otf", 110)
F_STATS = F("Inter-SemiBold.otf", 36)
F_SMALL = F("Inter-Medium.otf", 34)
F_LABEL = F("Inter-Bold.otf", 34)

def ease(x): x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3
def mix(a, b, t): t = min(max(t, 0.0), 1.0); return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def make_bg():
    rng = np.random.default_rng(7)
    g = rng.normal(0, 7, (H, W, 1))
    base = np.array(PAPER, dtype=np.float32)[None, None, :] + g
    yy, xx = np.mgrid[0:H, 0:W]
    d = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2
    base -= (d * 40)[..., None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
BG = make_bg()

_SHAPES = None
def shapes():
    global _SHAPES
    if _SHAPES is None:
        import shapefile
        r = shapefile.Reader(SHP)
        fields = [f[0] for f in r.fields[1:]]
        _SHAPES = []
        for sr in r.shapeRecords():
            rec = dict(zip(fields, sr.record))
            pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
            rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
            _SHAPES.append((rec.get("iso_a3"), rec.get("name"), rings, sr.shape.bbox))
    return _SHAPES

OCEAN = (78, 146, 200); SHELF = (150, 199, 230); LANDC = (228, 217, 194); F2C = (196, 178, 146)
F_CTRY = ImageFont.truetype("/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf", 30)
try: F_CTRY.set_variation_by_name("Bold")
except Exception: pass
F_PIN = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 32)

def _noise(h, w, octaves, seed):
    rng = np.random.default_rng(seed); out = np.zeros((h, w), np.float32)
    for s, amp in octaves:
        sm = rng.normal(0, 1, (max(2, h // s), max(2, w // s))).astype(np.float32)
        out += np.asarray(Image.fromarray(sm, "F").resize((w, h), Image.BICUBIC)) * amp
    return out

def _blur(a, r):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255

def _shift(a, dx, dy):
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

_FLAGS = {}
def flag_img(code, w, h):
    key = (code, w, h)
    if key not in _FLAGS:
        p = os.path.join(ROOT, "flags", code + ".png")
        _FLAGS[key] = Image.open(p).convert("RGBA").resize((w, h), Image.LANCZOS) if os.path.exists(p) else None
    return _FLAGS[key]

def paste_flag(img, code, x, y, w, h, a=1.0):
    f = flag_img(code, w, h)
    if f is None or a <= 0: return 0
    d = ImageDraw.Draw(img)
    d.rectangle([x + 4, y + 5, x + w + 4, y + h + 5], fill=(55, 45, 35))
    img.paste(f, (int(x), int(y)), f)
    d.rectangle([x - 2, y - 2, x + w + 1, y + h + 1], outline=INK, width=2)
    return w

_LAKES = None
def lakes():
    global _LAKES
    if _LAKES is None:
        import shapefile
        _LAKES = []
        p = os.path.join(ROOT, "geo/lakes/ne_110m_lakes.shp")
        if os.path.exists(p):
            for shp in shapefile.Reader(p).shapes():
                parts = list(shp.parts) + [len(shp.points)]
                _LAKES.append(([shp.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)], shp.bbox))
    return _LAKES

class MapLayer:
    """Documentary-style map: textured relief, shaded ocean, spotlight on focus countries,
    push-in camera from a wide view."""
    OVER = 2.3
    def __init__(self, bbox, focus=(), focus2=(), faint=False, labels=()):
        lon0, lat0, lon1, lat1 = bbox
        self.latc = (lat0 + lat1) / 2; self.lonc = (lon0 + lon1) / 2
        self.k = math.cos(math.radians(self.latc))
        self.BW, self.BH = int(W * self.OVER), int(MAP_H * self.OVER)
        self.scale = min(W / ((lon1 - lon0) * self.k), MAP_H / (lat1 - lat0))
        self.faint = faint
        hw = self.BW / 2 / (self.k * self.scale); hh = self.BH / 2 / self.scale
        win = (self.lonc - hw - 5, self.latc - hh - 5, self.lonc + hw + 5, self.latc + hh + 5)
        ss = 2; SW, SH = self.BW * ss, self.BH * ss
        masks = [Image.new("L", (SW, SH), 0) for _ in range(4)]
        dl, df, df2, db = [ImageDraw.Draw(m) for m in masks]
        for iso, name, rings, sb in shapes():
            if sb[2] < win[0] or sb[0] > win[2] or sb[3] < win[1] or sb[1] > win[3]:
                continue
            isf = iso in focus or name in focus
            isf2 = iso in focus2 or name in focus2
            for ring in rings:
                p = [(x * ss, y * ss) for x, y in (self.geo(lo, la) for lo, la in ring)]
                if len(p) < 3: continue
                dl.polygon(p, fill=255)
                if isf: df.polygon(p, fill=255)
                if isf2: df2.polygon(p, fill=255)
                db.line(p + [p[0]], fill=255, width=2 * ss, joint="curve")
        lake = Image.new("L", (SW, SH), 0); dk = ImageDraw.Draw(lake)
        for rings, sb in lakes():
            if sb[2] < win[0] or sb[0] > win[2] or sb[3] < win[1] or sb[1] > win[3]: continue
            for ring in rings:
                p = [(x * ss, y * ss) for x, y in (self.geo(lo, la) for lo, la in ring)]
                if len(p) > 2: dk.polygon(p, fill=255); db.line(p + [p[0]], fill=160, width=ss)
        masks.append(lake)
        # graticule every 10 degrees
        grat = Image.new("L", (SW, SH), 0); dg = ImageDraw.Draw(grat)
        for lo in range(-180, 181, 10):
            if win[0] - 10 <= lo <= win[2] + 10:
                a = self.geo(lo, max(win[1], -85)); b = self.geo(lo, min(win[3], 85))
                dg.line([(a[0] * ss, a[1] * ss), (b[0] * ss, b[1] * ss)], fill=255, width=ss)
        for la in range(-80, 81, 10):
            if win[1] - 10 <= la <= win[3] + 10:
                a = self.geo(win[0], la); b = self.geo(win[2], la)
                dg.line([(a[0] * ss, a[1] * ss), (b[0] * ss, b[1] * ss)], fill=255, width=ss)
        land, foc, foc2, bord, lake, grat = [np.asarray(m.resize((self.BW, self.BH), Image.LANCZOS), np.float32) / 255
                                       for m in masks + [grat]]
        land = land * (1 - lake); foc = foc * (1 - lake); foc2 = foc2 * (1 - lake)
        h, w = land.shape
        rel = _noise(h, w, [(420, 0.6), (140, 0.3), (40, 0.12), (10, 0.06)], 11)
        rng = np.random.default_rng(5); grain = rng.normal(0, 1, (h, w)).astype(np.float32)
        # ocean: lighter shelf near coasts, deeper further out, soft texture
        near = _blur(land, 45)
        nb = np.clip(near * 1.6, 0, 1)[..., None]
        ocean = np.array(OCEAN, np.float32)[None, None, :] * (1 - nb) + np.array(SHELF, np.float32)[None, None, :] * nb
        ocean += (rel * 5 + grain * 2.5)[..., None]
        ocean *= (1 - 0.06 * grat)[..., None]
        # land: parchment with fake hillshade (gradient of relief noise)
        gy, gx = np.gradient(_noise(h, w, [(90, 1.0), (30, 0.45), (9, 0.2)], 23))
        shade = np.clip(-(gx + gy) * 22, -1, 1)
        landc = np.array(LANDC, np.float32)[None, None, :] + (rel * 6 + shade * 6 + grain * 4)[..., None]
        # land casts a soft shadow onto the sea
        lsh = _blur(_shift(land, 6, 9), 7)
        ocean *= (1 - 0.30 * lsh * (1 - land))[..., None]
        base = ocean * (1 - land[..., None]) + landc * land[..., None]
        base *= (1 - 0.32 * bord)[..., None]
        if faint:
            paper = np.array(PAPER, np.float32)[None, None, :]
            base = paper + (base - paper) * 0.35
            hl = base
        else:
            # spotlight: dim the other land strongly, the water only a little so it stays blue
            hl = base * (0.92 - 0.24 * land)[..., None]
            f2fill = np.array(F2C, np.float32)[None, None, :] + (rel * 5 + shade * 5 + grain * 4)[..., None]
            hl = hl * (1 - foc2[..., None]) + f2fill * foc2[..., None] * 0.92
            fsh = _blur(_shift(foc, 12, 16), 14)
            hl *= (1 - 0.55 * fsh * (1 - foc))[..., None]
            glow = _blur(foc, 22)
            hl += (np.array(YEL, np.float32) - hl) * (0.45 * glow * (1 - foc))[..., None]
            ffill = np.array(YEL, np.float32)[None, None, :] + (rel * 6 + shade * 7 + grain * 4)[..., None]
            hl = hl * (1 - foc[..., None]) + ffill * foc[..., None]
            rim = np.clip(_blur(foc, 2.2) * 2.2 - foc * 2.2, 0, 1) + np.clip(foc - (1 - np.clip((1 - _blur(foc, 2.0)) * 2.2, 0, 1)), 0, 1) * 0
            hl = hl * (1 - rim[..., None]) + np.array(INK, np.float32) * rim[..., None]
        imgs = [Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB").convert("RGBA") for a in (base, hl)]
        for im in imgs:
            self._labels(im, labels)
        self.base, self.hl = imgs
        m = Image.new("L", (W, MAP_H), 255); md = ImageDraw.Draw(m)
        for i in range(22):
            a = int(255 * (i / 22) ** 1.2)
            md.line([(0, i), (W, i)], fill=a); md.line([(0, MAP_H - 1 - i), (W, MAP_H - 1 - i)], fill=a)
        self.edge = m
        yy, xx = np.mgrid[0:MAP_H, 0:W]
        v = 1 - 0.38 * (((xx - W / 2) / (W / 2)) ** 2 * 0.6 + ((yy - MAP_H / 2) / (MAP_H / 2)) ** 2 * 0.25) ** 1.3
        self.vig = Image.fromarray((np.clip(v, 0, 1)[..., None].repeat(3, 2) * 255).astype(np.uint8), "RGB")

    def _labels(self, im, labels):
        if not labels: return
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        for lb in labels:
            x, y = self.geo(lb["lon"], lb["lat"]); txt = lb["text"].upper()
            sp = 9; tw = sum(F_CTRY.getlength(c) + sp for c in txt) - sp
            fw = 52 if lb.get("flag") else 0
            cx = x - (tw + fw + (16 if fw else 0)) / 2
            if fw:
                im.alpha_composite(ov); ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
                paste_flag(im, lb["flag"], cx, y - 19, 52, 39); cx += fw + 16
            for c in txt:
                d.text((cx, y), c, font=F_CTRY, fill=(40, 34, 28, 165), anchor="lm"); cx += F_CTRY.getlength(c) + sp
        im.alpha_composite(ov)

    def geo(self, lon, lat):
        x = self.BW / 2 + (lon - self.lonc) * self.k * self.scale
        y = self.BH / 2 - (lat - self.latc) * self.scale
        return x, y

    def frame(self, zoom, hl_amt):
        im = Image.blend(self.base, self.hl, hl_amt) if 0 < hl_amt < 1 else (self.hl if hl_amt >= 1 else self.base)
        cw, ch = W / zoom, MAP_H / zoom
        x0, y0 = (self.BW - cw) / 2, (self.BH - ch) / 2
        self._crop = (x0, y0, zoom)
        fr = im.resize((W, MAP_H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch)).convert("RGB")
        if not self.faint:
            fr = ImageChops.multiply(fr, self.vig)
            fd = ImageDraw.Draw(fr); fd.line([(0, 0), (W, 0)], fill=INK, width=3); fd.line([(0, MAP_H - 2), (W, MAP_H - 2)], fill=INK, width=3)
            return fr.convert("RGBA")
        fr = fr.convert("RGBA"); fr.putalpha(self.edge)
        return fr

    def to_screen(self, lon, lat):
        x, y = self.geo(lon, lat); x0, y0, z = self._crop
        return (x - x0) * z, MAP_Y + (y - y0) * z

def cam_zoom(t, dur, over):
    z0 = 1.0 / over * 1.25
    p = min(max(t / 1.5, 0), 1); p = p * p * (3 - 2 * p)
    z = z0 + (1.0 - z0) * p
    return z * (1 + 0.06 * max(t - 1.5, 0) / max(dur - 1.5, 0.1))

def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if font.getlength(t) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = w_
    if cur: lines.append(cur)
    return lines

def draw_headline(img, d, text, highlight, y, t, font=F_HEAD, maxw=960, lead=1.08, start=0.15):
    tokens = []
    if highlight and highlight.lower() in text.lower():
        i = text.lower().index(highlight.lower())
        pre, ph, post = text[:i], text[i:i + len(highlight)], text[i + len(highlight):]
        tokens = [(w_, False) for w_ in pre.split()] + [(ph, True)] + [(w_, False) for w_ in post.split()]
    else:
        tokens = [(w_, False) for w_ in text.split()]
    lines, cur = [], []
    for tok in tokens:
        cand = " ".join(x[0] for x in cur + [tok])
        if font.getlength(cand) <= maxw or not cur: cur.append(tok)
        else: lines.append(cur); cur = [tok]
    if cur: lines.append(cur)
    lh = int(font.size * lead)
    sw = ease((t - start - 0.45) / 0.5)
    asc = font.getbbox("Hg")
    for li, toks in enumerate(lines):
        a = ease((t - start - li * 0.12) / 0.35)
        yy = y + li * lh + int((1 - a) * 40)
        line = " ".join(x[0] for x in toks); x = (W - font.getlength(line)) / 2
        for wi, (w_, hl) in enumerate(toks):
            if hl and sw > 0:
                wx = x + font.getlength(" ".join(z[0] for z in toks[:wi]) + (" " if wi else ""))
                x0, x1 = wx - 14, wx + font.getlength(w_) + 14
                d.rectangle([x0, yy + asc[1] + 4, x0 + (x1 - x0) * sw, yy + asc[3] - 2], fill=YEL)
        d.text((x, yy), line, font=font, fill=mix(PAPER, INK, a))
    return y + len(lines) * lh

def draw_caption(d, lines, t):
    cur = None
    for ln in lines:
        if ln["start"] <= t < ln["end"] + 0.12: cur = ln
    if not cur: return
    a = ease((t - cur["start"]) / 0.15)
    rows = wrap(cur["text"], F_CAP, 920)
    lh = 64; y = 1640 - (len(rows) - 1) * lh // 2
    for i, r in enumerate(rows):
        d.text(((W - F_CAP.getlength(r)) / 2, y + i * lh), r, font=F_CAP, fill=mix(PAPER, (55, 55, 55), a))

def draw_tag(d, num, label, t):
    a = ease(t / 0.35)
    x = 70 - int((1 - a) * 60); y = 150
    d.rectangle([x, y, x + 96, y + 70], fill=INK)
    d.text((x + 48, y + 35), num, font=F_NUM, fill=YEL, anchor="mm")
    lx = x + 120
    for ch in label.upper():
        d.text((lx, y + 35), ch, font=F_TAG, fill=mix(PAPER, INK, a), anchor="lm")
        lx += F_TAG.getlength(ch) + 4
    d.line([(lx + 20, y + 35), (lx + 20 + (W - 70 - lx - 20) * a, y + 35)], fill=INK, width=3)

def draw_marker(d, x, y, label, t, dx=40, dy=-90, flag=None):
    a = ease(t / 0.4)
    if a <= 0: return
    pr = (t * 0.9) % 1.0
    r = 14 + 46 * pr
    d.ellipse([x - r, y - r, x + r, y + r], outline=mix((150, 150, 150), RED, (1 - pr) * a), width=4)
    rr = 15 * a
    d.ellipse([x - rr + 3, y - rr + 5, x + rr + 3, y + rr + 5], fill=(60, 50, 40))
    d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=RED, outline=(250, 245, 235), width=4)
    if label:
        lab = label.upper()
        lx, ly = x + dx, y + dy
        la = ease((t - 0.15) / 0.35)
        if la <= 0: return
        ex, ey = x + (lx - x) * la, y + (ly - y) * la
        d.line([(x, y), (ex, ey)], fill=INK, width=3)
        fx = 50 if flag else 0
        tw = F_PIN.getlength(lab) + fx
        bx0 = lx if dx >= 0 else lx - tw - 40
        if la > 0.7:
            d.rectangle([bx0 + 7, ly - 30 + 8, bx0 + tw + 40 + 7, ly + 30 + 8], fill=(55, 45, 35))
            d.rectangle([bx0, ly - 30, bx0 + tw + 40, ly + 30], fill=(247, 242, 230), outline=INK, width=3)
            if flag: paste_flag(d._image, flag, bx0 + 16, ly - 15, 40, 30)
            d.text((bx0 + 20 + fx, ly + 1), lab, font=F_PIN, fill=INK, anchor="lm")

def draw_stat(d, big, small, t):
    a = ease(t / 0.45)
    if a <= 0: return
    bw = max(F_STATB.getlength(big), F_STATS.getlength(small)) + 70
    x1 = W - 60 + int((1 - a) * (bw + 80)); x0 = x1 - bw
    y0, y1 = 1000, 1000 + 210
    d.rectangle([x0 + 10, y0 + 10, x1 + 10, y1 + 10], fill=YEL)
    d.rectangle([x0, y0, x1, y1], fill=INK)
    d.text((x0 + 35, y0 + 18), big, font=F_STATB, fill=YEL)
    d.text((x0 + 35, y0 + 150), small, font=F_STATS, fill=PAPER)

def draw_progress(d, gt, total):
    d.rectangle([0, 0, W * min(gt / total, 1), 12], fill=YEL)

def _icon(d, kind, cx, cy, s, col):
    w = max(3, int(s * 0.16))
    if kind == "chip":
        d.rectangle([cx - s * .55, cy - s * .55, cx + s * .55, cy + s * .55], outline=col, width=w)
        d.rectangle([cx - s * .22, cy - s * .22, cx + s * .22, cy + s * .22], fill=col)
        for k in (-0.3, 0, 0.3):
            for a, b in (((cx + k * s, cy - s * .55), (cx + k * s, cy - s * .85)), ((cx + k * s, cy + s * .55), (cx + k * s, cy + s * .85)),
                         ((cx - s * .55, cy + k * s), (cx - s * .85, cy + k * s)), ((cx + s * .55, cy + k * s), (cx + s * .85, cy + k * s))):
                d.line([a, b], fill=col, width=w)
    elif kind == "cloud":
        for ox, oy, r in ((-.35, .1, .35), (0, -.15, .45), (.38, .1, .33)):
            d.ellipse([cx + (ox - r) * s, cy + (oy - r) * s, cx + (ox + r) * s, cy + (oy + r) * s], fill=col)
        d.rectangle([cx - .35 * s, cy + .1 * s, cx + .38 * s, cy + .43 * s], fill=col)
    elif kind == "coin":
        d.ellipse([cx - .8 * s, cy - .8 * s, cx + .8 * s, cy + .8 * s], outline=col, width=w)
        d.ellipse([cx - .45 * s, cy - .45 * s, cx + .45 * s, cy + .45 * s], outline=col, width=w)
    elif kind == "shield":
        d.polygon([(cx - .7 * s, cy - .75 * s), (cx + .7 * s, cy - .75 * s), (cx + .7 * s, cy), (cx, cy + .85 * s), (cx - .7 * s, cy)], fill=col)
    elif kind == "dome":
        d.pieslice([cx - .55 * s, cy - .75 * s, cx + .55 * s, cy + .35 * s], 180, 360, fill=col)
        d.rectangle([cx - .8 * s, cy - .2 * s, cx + .8 * s, cy - .08 * s], fill=col)
        for k in (-.6, -.3, 0, .3, .6):
            d.rectangle([cx + k * s - w / 2, cy - .08 * s, cx + k * s + w / 2, cy + .55 * s], fill=col)
        d.rectangle([cx - .85 * s, cy + .55 * s, cx + .85 * s, cy + .72 * s], fill=col)
    elif kind == "health":
        pts = [(-.85, .1), (-.4, .1), (-.2, -.6), (.1, .7), (.3, .1), (.85, .1)]
        d.line([(cx + x * s, cy + y * s) for x, y in pts], fill=col, width=w, joint="curve")
    elif kind == "star":
        pts = []
        for i in range(10):
            r = .85 if i % 2 == 0 else .38; a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * s * math.cos(a), cy + r * s * math.sin(a)))
        d.polygon(pts, fill=col)
    else:  # institution
        d.polygon([(cx - .85 * s, cy - .35 * s), (cx, cy - .85 * s), (cx + .85 * s, cy - .35 * s)], fill=col)
        for k in (-.55, -.18, .18, .55):
            d.rectangle([cx + k * s - w / 2, cy - .3 * s, cx + k * s + w / 2, cy + .5 * s], fill=col)
        d.rectangle([cx - .85 * s, cy + .55 * s, cx + .85 * s, cy + .72 * s], fill=col)

def _badge_w(name): return 72 + 22 + F_LABEL.getlength(name.upper()) + 26

def draw_badge(d, x, y, name, icon, a):
    if a <= 0: return
    bw = _badge_w(name); y = y + (1 - a) * 30
    d.rectangle([x + 7, y + 8, x + bw + 7, y + 72 + 8], fill=mix(PAPER, (55, 45, 35), a))
    d.rectangle([x, y, x + bw, y + 72], fill=mix(PAPER, (247, 242, 230), a), outline=mix(PAPER, INK, a), width=3)
    d.rectangle([x, y, x + 72, y + 72], fill=mix(PAPER, INK, a))
    _icon(d, icon, x + 36, y + 36, 24, mix(INK, YEL, a))
    d.text((x + 72 + 22, y + 37), name.upper(), font=F_LABEL, fill=mix(PAPER, INK, a), anchor="lm")

F_ROLE = F("Inter-Medium.otf", 27)

def _person_w(p): return 84 + 22 + max(F_LABEL.getlength(p["name"].upper()), F_ROLE.getlength(p.get("role", ""))) + 26

def draw_person(d, x, y, p, a):
    if a <= 0: return
    bw = _person_w(p); y = y + (1 - a) * 30; h = 84
    d.rectangle([x + 7, y + 8, x + bw + 7, y + h + 8], fill=mix(PAPER, (55, 45, 35), a))
    d.rectangle([x, y, x + bw, y + h], fill=mix(PAPER, (247, 242, 230), a), outline=mix(PAPER, INK, a), width=3)
    d.rectangle([x, y, x + 84, y + h], fill=mix(PAPER, INK, a))
    cx, cy = x + 42, y + 42; col = mix(INK, YEL, a)
    d.ellipse([cx - 13, cy - 26, cx + 13, cy], fill=col)                        # generic head
    d.pieslice([cx - 27, cy + 2, cx + 27, cy + 56], 180, 360, fill=col)        # generic shoulders
    d.text((x + 84 + 22, y + 30), p["name"].upper(), font=F_LABEL, fill=mix(PAPER, INK, a), anchor="lm")
    if p.get("role"):
        d.text((x + 84 + 22, y + 62), p["role"], font=F_ROLE, fill=mix(PAPER, (90, 80, 70), a), anchor="lm")

def draw_orgs(d, sc, t):
    items = [("org", o) for o in (sc.get("orgs") or [])] + [("person", p) for p in (sc.get("people") or [])]
    if not items: return
    def at(o):
        ln = sc["lines"][min(o.get("at_line", 0), len(sc["lines"]) - 1)]
        return ease((t - ln["start"] - 0.1) / 0.4)
    def wh(kind, o): return (_badge_w(o["name"]), 72) if kind == "org" else (_person_w(o), 84)
    def draw(kind, o, x, y):
        if kind == "org": draw_badge(d, x, y, o["name"], o.get("icon", "institution"), at(o))
        else: draw_person(d, x, y, o, at(o))
    if sc["type"] == "map":
        y = MAP_Y + 36
        for kind, o in items:
            w_, h_ = wh(kind, o); draw(kind, o, 40, y); y += h_ + 22
    else:
        rows, cur, cw = [], [], 0
        for it in items:
            w_ = wh(*it)[0]
            if cur and cw + 28 + w_ > 1000: rows.append(cur); cur, cw = [], 0
            cur.append(it); cw += (28 if len(cur) > 1 else 0) + w_
        if cur: rows.append(cur)
        y = 955 if len(rows) == 1 else 925
        for row in rows:
            tot = sum(wh(*it)[0] for it in row) + 28 * (len(row) - 1); x = (W - tot) / 2
            for it in row:
                draw(*it, x, y); x += wh(*it)[0] + 28
            y += max(wh(*it)[1] for it in row) + 22

def _glow(im, r=6, k=1.0):
    g = im.filter(ImageFilter.GaussianBlur(r))
    return ImageChops.add(im, ImageChops.multiply(g, Image.new("RGB", im.size, tuple(int(255 * k) for _ in range(3)))))

def make_bg_panel(kind, seed=3):
    """Procedurally drawn, generic illustration for a topic (no photos, no logos)."""
    BWp, BHp = int(W * 1.15), int(MAP_H * 1.15)
    rng = np.random.default_rng(seed)
    if kind == "circuit":
        base = (14, 32, 26); im = Image.new("RGB", (BWp, BHp), base); d = ImageDraw.Draw(im)
        g = 36
        for _ in range(260):
            x = int(rng.integers(0, BWp // g)) * g; y = int(rng.integers(0, BHp // g)) * g
            pts = [(x, y)]
            for _ in range(int(rng.integers(2, 6))):
                dirn = rng.integers(0, 4); L = int(rng.integers(2, 7)) * g
                dx, dy = [(L, 0), (-L, 0), (0, L), (0, -L)][dirn]
                if rng.random() < 0.4: dx, dy = (dx if dx else (L if rng.random() < .5 else -L)), (dy if dy else (L if rng.random() < .5 else -L))
                x, y = x + dx, y + dy; pts.append((x, y))
            hot = rng.random() < 0.12
            d.line(pts, fill=(200, 170, 60) if hot else (52, 110, 84), width=4 if hot else 3, joint="curve")
            for px, py in (pts[0], pts[-1]):
                d.ellipse([px - 7, py - 7, px + 7, py + 7], outline=(200, 170, 60) if hot else (70, 140, 106), width=3)
        cx, cy, s = int(BWp * 0.84), int(BHp * 0.2), 150
        for k in range(-5, 6):
            for a, b in (((cx + k * 26, cy - s), (cx + k * 26, cy - s - 50)), ((cx + k * 26, cy + s), (cx + k * 26, cy + s + 50)),
                         ((cx - s, cy + k * 26), (cx - s - 50, cy + k * 26)), ((cx + s, cy + k * 26), (cx + s + 50, cy + k * 26))):
                d.line([a, b], fill=(190, 190, 180), width=8)
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=(30, 34, 36), outline=(120, 125, 120), width=6)
        d.rectangle([cx - s + 30, cy - s + 30, cx + s - 30, cy + s - 30], outline=(70, 75, 72), width=3)
        im = _glow(im, 8, 0.9)
    elif kind == "servers":
        im = Image.new("RGB", (BWp, BHp), (12, 18, 30)); d = ImageDraw.Draw(im)
        rw, gap = 190, 40; x0 = (BWp - (5 * rw + 4 * gap)) // 2
        for r in range(5):
            x = x0 + r * (rw + gap)
            d.rectangle([x, 40, x + rw, BHp - 40], fill=(22, 30, 46), outline=(60, 72, 96), width=3)
            for u in range(int((BHp - 100) / 46)):
                y = 60 + u * 46
                d.rectangle([x + 14, y, x + rw - 14, y + 36], fill=(30, 40, 60), outline=(48, 60, 84), width=2)
                for v in range(5): d.line([(x + 26 + v * 10, y + 8), (x + 26 + v * 10, y + 28)], fill=(44, 56, 80), width=3)
                state = rng.random()
                col = (230, 70, 50) if state < 0.35 else ((255, 200, 60) if state < 0.5 else (70, 200, 120))
                for k in range(3):
                    if rng.random() < 0.7: d.ellipse([x + rw - 34 - k * 18, y + 14, x + rw - 26 - k * 18, y + 22], fill=col)
        im = _glow(im, 6, 0.8)
    elif kind == "chart":
        im = Image.new("RGB", (BWp, BHp), (16, 18, 24)); d = ImageDraw.Draw(im)
        for gx in range(0, BWp, 80): d.line([(gx, 0), (gx, BHp)], fill=(30, 34, 44), width=2)
        for gy in range(0, BHp, 80): d.line([(0, gy), (BWp, gy)], fill=(30, 34, 44), width=2)
        n = 46; xs = np.linspace(60, BWp - 60, n); v = np.cumsum(rng.normal(0, 1, n)) - np.linspace(0, 9, n)
        ys = BHp * 0.25 + (v - v.min()) / (np.ptp(v) + 1e-6) * BHp * 0.5
        ys = BHp - ys
        for i in range(n - 1):
            o, c = ys[i], ys[i + 1]; up = c < o
            col = (70, 190, 120) if up else (230, 80, 60)
            d.line([(xs[i], min(o, c) - rng.uniform(8, 30)), (xs[i], max(o, c) + rng.uniform(8, 30))], fill=col, width=3)
            d.rectangle([xs[i] - 9, min(o, c), xs[i] + 9, max(o, c) + 2], fill=col)
        im = _glow(im, 5, 0.6)
    elif kind == "neurons":
        im = Image.new("RGB", (BWp, BHp), (18, 14, 32)); d = ImageDraw.Draw(im)
        pts = [(rng.uniform(0, BWp), rng.uniform(0, BHp)) for _ in range(34)]
        for i, (x, y) in enumerate(pts):
            for _ in range(int(rng.integers(3, 7))):
                ang = rng.uniform(0, 2 * np.pi); px, py = x, y
                for s_ in range(int(rng.integers(4, 9))):
                    ang += rng.normal(0, 0.45); L = rng.uniform(14, 34)
                    nx, ny = px + L * np.cos(ang), py + L * np.sin(ang)
                    d.line([(px, py), (nx, ny)], fill=(110, 90, 170), width=max(1, 4 - s_ // 2)); px, py = nx, ny
            r = rng.uniform(8, 16)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(160, 130, 220))
        x, y = BWp * 0.55, BHp * 0.45
        for _ in range(8):
            ang = rng.uniform(0, 2 * np.pi); px, py = x, y
            for s_ in range(9):
                ang += rng.normal(0, 0.4); nx, ny = px + 30 * np.cos(ang), py + 30 * np.sin(ang)
                d.line([(px, py), (nx, ny)], fill=(255, 210, 63), width=max(2, 6 - s_ // 2)); px, py = nx, ny
        d.ellipse([x - 22, y - 22, x + 22, y + 22], fill=(255, 220, 90))
        im = _glow(im, 7, 1.0)
    else:
        return None
    # darken so foreground text stays readable
    k = {"circuit": 0.45, "chart": 0.32, "servers": 0.5, "neurons": 0.6}.get(kind, 0.5)
    im = Image.blend(Image.new("RGB", im.size, (8, 8, 10)), im, k)
    return im

def bg_frame(sc, t):
    im = sc["_bg"]; dur = sc["dur"]; z = 1.0 + 0.08 * t / dur
    cw, ch = im.width / 1.15 / z, im.height / 1.15 / z
    x0 = (im.width - cw) / 2 + 20 * math.sin(t * 0.3); y0 = (im.height - ch) / 2
    fr = im.resize((W, MAP_H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
    fd = ImageDraw.Draw(fr); fd.line([(0, 0), (W, 0)], fill=INK, width=3); fd.line([(0, MAP_H - 2), (W, MAP_H - 2)], fill=INK, width=3)
    return fr

def scene_frame(sc, idx, t, gt, total):
    img = BG.copy(); d = ImageDraw.Draw(img)
    typ = sc["type"]; dur = sc["dur"]
    if typ in ("map", "intro"):
        ml = sc["_map"]
        zoom = cam_zoom(t, dur, ml.OVER) if typ == "map" else 1.0 + 0.06 * (t / dur)
        fr = ml.frame(zoom, ease((t - 1.15) / 0.6) if typ == "map" else 0)
        img.paste(fr, (0, MAP_Y), fr); d = ImageDraw.Draw(img)
    FG = INK
    if sc.get("_bg") is not None:
        img.paste(bg_frame(sc, t), (0, MAP_Y)); d = ImageDraw.Draw(img); FG = PAPER
    if typ == "intro":
        a = ease(t / 0.4)
        o = 150
        d.text((W / 2, 520 + o), sc.get("title", "THE WORLD"), font=F_TITLE, fill=mix(PAPER, INK, a), anchor="mm")
        sw = ease((t - 0.35) / 0.5)
        txt = sc.get("subtitle", "IN 60 SECONDS"); f2 = F("InterDisplay-Black.otf", 112)
        tw = f2.getlength(txt)
        d.rectangle([(W - tw) / 2 - 20, 625 + o, (W - tw) / 2 - 20 + (tw + 40) * sw, 755 + o], fill=YEL)
        d.text((W / 2, 690 + o), txt, font=f2, fill=mix(PAPER, INK, a), anchor="mm")
        d.text((W / 2, 860 + o), sc.get("date", ""), font=F_LABEL, fill=mix(PAPER, INK, ease((t - .5) / .4)), anchor="mm")
        d.text((W / 2, 915 + o), sc.get("edition", ""), font=F_SMALL, fill=mix(PAPER, (90, 90, 90), ease((t - .6) / .4)), anchor="mm")
    elif typ == "section":
        a = ease(t / 0.3)
        lab = sc["label"].upper(); fs = F("InterDisplay-Black.otf", 170 if len(lab) <= 8 else 130)
        tw = fs.getlength(lab); sw = ease((t - 0.15) / 0.35)
        d.rectangle([(W - tw) / 2 - 30, 830, (W - tw) / 2 - 30 + (tw + 60) * sw, 1010], fill=YEL)
        d.text((W / 2, 920), lab, font=fs, fill=mix(PAPER, INK, a), anchor="mm")
        if sc.get("count"):
            d.text((W / 2, 1090), sc["count"], font=F_TAG, fill=mix(PAPER, INK, ease((t - 0.2) / 0.3)), anchor="mm")
    elif typ == "outro":
        draw_headline(img, d, sc["headline"], sc.get("highlight"), 560, t, font=F("InterDisplay-Black.otf", 110))
        y = 1060
        a = ease((t - 0.6) / 0.4)
        d.text((W / 2, y), "SOURCES", font=F_TAG, fill=mix(PAPER, INK, a), anchor="mm")
        rows = wrap(" · ".join(sc.get("sources", [])), F_SMALL, 900)
        for i, r in enumerate(rows):
            d.text((W / 2, y + 60 + i * 48), r, font=F_SMALL, fill=mix(PAPER, (80, 80, 80), a), anchor="mm")
    else:
        draw_tag(d, f"{idx:02d}", sc["tag"], t)
        if typ == "map":
            ml = sc["_map"]
            for j, mk in enumerate(sc.get("markers", [])):
                x, y = ml.to_screen(mk["lon"], mk["lat"])
                draw_marker(d, x, y, mk.get("label"), t - 1.6 - 0.35 * j, mk.get("dx", 40), mk.get("dy", -90), mk.get("flag"))
        elif typ == "bars":
            items = sc["items"][:5]; vmax = max(abs(it["value"]) for it in items) or 1
            y0 = 420; rowh = min(150, 700 // max(len(items), 1)); cx = W / 2 + 60
            d.line([(cx, y0 - 30), (cx, y0 + rowh * len(items) - 20)], fill=FG, width=3)
            for j, it in enumerate(items):
                p = ease((t - 0.3 - 0.15 * j) / 0.6); yy = y0 + j * rowh
                bl = (W / 2 - 160) * abs(it["value"]) / vmax * p
                col = YEL if it["value"] >= 0 else RED
                if it["value"] >= 0: d.rectangle([cx, yy, cx + bl, yy + rowh * 0.55], fill=col)
                else: d.rectangle([cx - bl, yy, cx, yy + rowh * 0.55], fill=col)
                la = ease((t - 0.2 - 0.15 * j) / 0.3)
                tx = cx - 20 if it["value"] >= 0 else cx + 20
                d.text((tx, yy + rowh * 0.27), it["label"], font=F_LABEL, fill=FG if FG == PAPER else mix(PAPER, INK, la), anchor="rm" if it["value"] >= 0 else "lm")
                vt = it.get("text", f'{it["value"]:+g}')
                vx = cx + bl + 18 if it["value"] >= 0 else cx - bl - 18
                if p > 0.3: d.text((vx, yy + rowh * 0.27), vt, font=F_LABEL, fill=FG, anchor="lm" if it["value"] >= 0 else "rm")
            if sc.get("note"):
                d.text((W / 2, y0 + rowh * len(items) + 30), sc["note"], font=F_SMALL, fill=(200, 196, 186) if FG == PAPER else mix(PAPER, (90, 90, 90), ease((t - .8) / .4)), anchor="mm")
        elif typ == "number":
            n = sc["number"]; p = ease((t - 0.3) / 1.4)
            v = n * p
            val = sc.get("prefix", "") + (f"{v:.{sc.get('decimals', 0)}f}") + sc.get("suffix", "")
            fb = F_BIG
            if fb.getlength(val) > 960: fb = F("InterDisplay-Black.otf", int(330 * 960 / fb.getlength(val)))
            d.text((W / 2, 640), val, font=fb, fill=YEL if FG == PAPER else INK, anchor="mm")
            ua = ease((t - 0.6) / 0.4)
            d.text((W / 2, 860), sc.get("unit", "").upper(), font=F_TAG, fill=PAPER if FG == PAPER else mix(PAPER, INK, ua), anchor="mm")
            if sc.get("from") and sc.get("to"):
                x0, x1, yb = 150, 930, 1010
                d.line([(x0, yb), (x1, yb)], fill=LAND2, width=10)
                d.line([(x0, yb), (x0 + (x1 - x0) * p, yb)], fill=YEL, width=10)
                for xx, lab in ((x0, sc["from"]), (x1, sc["to"])):
                    d.ellipse([xx - 14, yb - 14, xx + 14, yb + 14], fill=INK)
                    d.text((xx, yb + 50), lab, font=F_LABEL, fill=mix(PAPER, INK, ua), anchor="mm")
        draw_orgs(d, sc, t)
        st = sc.get("stat")
        if st:
            ln = sc["lines"][min(st.get("at_line", 1), len(sc["lines"]) - 1)]
            draw_stat(d, st["big"], st["small"], t - ln["start"])
        draw_headline(img, d, sc["headline"], sc.get("highlight"), 1240, t)
    draw_caption(d, sc["lines"], t)
    draw_progress(d, gt, total)
    if idx is not None and sc.get("_wipe_out"):
        w = ease((t - (dur - 0.25)) / 0.25)
        if w > 0: d.rectangle([0, 0, W * w, H], fill=YEL)
    if sc.get("_wipe_in"):
        w = ease(t / 0.25)
        if w < 1: d.rectangle([W * w, 0, W, H], fill=YEL)
    return img

import re as _re
_LETTER = {"A": "eɪ", "B": "biː", "C": "siː", "D": "diː", "E": "iː", "F": "ɛf", "G": "dʒiː", "H": "eɪtʃ", "I": "aɪ",
           "J": "dʒeɪ", "K": "keɪ", "L": "ɛl", "M": "ɛm", "N": "ɛn", "O": "oʊ", "P": "piː", "Q": "kjuː", "R": "ɑːɹ",
           "S": "ɛs", "T": "tiː", "U": "juː", "V": "viː", "W": "dʌbəljuː", "X": "ɛks", "Y": "waɪ", "Z": "ziː"}
# abbreviations that are read as words, not letter by letter
WORD_ACRONYMS = {"NASA", "NATO", "PAHO", "OPEC", "UNESCO", "UNICEF", "COVID", "FIFA", "NAFTA", "ASEAN", "SWAT",
                 "NOAA", "CERN", "LIDAR", "RADAR", "LASER", "SCOTUS", "FEMA", "OSHA", "DARPA", "ICE"}
_ACR = _re.compile(r"(?<![A-Za-z])(?:(?:[A-Za-z]\.){2,}|[A-Z]&[A-Z]|[A-Z]{2,6})(?:'s|s)?(?![A-Za-z])")

def _spell(tok):
    """Letter-by-letter phonemes joined with no gaps (one smooth word)."""
    suf = ""
    if tok.endswith("'s"): tok, suf = tok[:-2], "z"
    elif len(tok) > 2 and tok.endswith("s") and tok[:-1].replace(".", "").isupper(): tok, suf = tok[:-1], "z"
    if "&" in tok:
        a, b = tok.split("&"); return "ˌ" + _LETTER[a.upper()] + "ənd" + "ˈ" + _LETTER[b.upper()] + suf
    letters = [c.upper() for c in tok if c.isalpha()]
    return "".join(("ˈ" if i == len(letters) - 1 else "ˌ") + _LETTER[c] for i, c in enumerate(letters)) + suf

def speak_phonemes(k, text, lang="en-us"):
    """Phonemize text, spelling abbreviations as one continuous word (no pauses between letters)."""
    out, pos = [], 0
    for m in _ACR.finditer(text):
        tok = m.group(0); core = tok.replace("'s", "")
        core = core[:-1] if (len(core) > 2 and core.endswith("s") and core[:-1].isupper()) else core
        if core in WORD_ACRONYMS or (core.isalpha() and core.islower()): continue
        if core.isalpha() and not core.isupper(): continue
        chunk = text[pos:m.start()]
        if chunk.strip(): out.append(k.tokenizer.phonemize(chunk, lang).strip())
        out.append(_spell(tok))
        pos = m.end()
        if tok.endswith(".") and pos >= len(text.rstrip()): out.append(".")
    rest = text[pos:]
    if rest.strip(): out.append(k.tokenizer.phonemize(rest, lang).strip())
    return " ".join(x for x in out if x)

def tts_all(story, speed):
    from kokoro_onnx import Kokoro
    k = Kokoro(MODEL, VOICES)
    voice = story.get("voice", "af_heart"); lang = "en-gb" if voice.startswith("b") else "en-us"
    out = []
    for sc in story["scenes"]:
        segs = []
        for ln in sc["lines"]:
            ph = speak_phonemes(k, ln.get("say", ln["text"]), lang)
            s, sr = k.create(ph, voice=voice, speed=speed, lang=lang, is_phonemes=True)
            segs.append(np.asarray(s, dtype=np.float32))
        out.append(segs)
    return out, 24000

def build_timing(story, segs, sr, pre=0.35, gap=0.18, post=0.45):
    t0 = 0.0; audio = []
    for sc, ss in zip(story["scenes"], segs):
        t = pre; parts = [np.zeros(int(pre * sr), np.float32)]
        for ln, s in zip(sc["lines"], ss):
            ln["start"] = t; ln["end"] = t + len(s) / sr
            parts += [s, np.zeros(int(gap * sr), np.float32)]
            t = ln["end"] + gap
        dur = max(t - gap + post, sc.get("hold", 0)) if sc["lines"] else sc.get("hold", 1.3)
        dur = math.ceil(dur * FPS) / FPS
        total_len = int(round(dur * sr))
        a = np.concatenate(parts)[:total_len]
        a = np.pad(a, (0, total_len - len(a)))
        sc["dur"] = dur; sc["t0"] = t0; t0 += dur
        audio.append(a)
    return np.concatenate(audio), t0

def music_bed(n, sr, scene_starts):
    t = np.arange(n) / sr
    chords = [[110.0, 164.81, 220.0, 277.18], [98.0, 146.83, 196.0, 246.94]]
    bed = np.zeros(n, np.float32)
    seg = 8.0
    for i, ch in enumerate(chords * 30):
        s0 = int(i * seg * sr); s1 = min(n, int((i + 1) * seg * sr + sr))
        if s0 >= n: break
        tt = t[s0:s1] - i * seg
        env = np.clip(tt / 1.5, 0, 1) * np.clip((seg + 1 - tt) / 1.5, 0, 1)
        for f in ch:
            bed[s0:s1] += (np.sin(2 * np.pi * f * t[s0:s1]) + 0.3 * np.sin(2 * np.pi * f * 2.003 * t[s0:s1])) * env * 0.12
    puls = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t) ** 8
    bed *= 0.7 + 0.3 * puls
    rng = np.random.default_rng(1)
    for st in scene_starts[1:]:
        s0 = int((st - 0.3) * sr); L = int(0.55 * sr)
        if s0 < 0 or s0 + L > n: continue
        nz = rng.normal(0, 1, L).astype(np.float32)
        nz = np.convolve(nz, np.ones(40) / 40, mode="same")
        env = np.sin(np.linspace(0, np.pi, L)) ** 2
        bed[s0:s0 + L] += nz * env * 1.2
    fade = np.minimum(1, np.minimum(t / 1.0, (t[-1] - t) / 1.5))
    return bed * fade

def music_bed2(n, sr, scene_starts, voice):
    """Original procedural underscore: soft pad, pulsing bass, plucked arpeggio, light percussion.
    Ducked under the narration so it never competes with the voice."""
    t = np.arange(n) / sr; bpm = 96; beat = 60 / bpm; bar = 4 * beat
    out = np.zeros(n, np.float32)
    A2 = 110.0
    def hz(semi): return A2 * 2 ** (semi / 12)
    prog = [(0, [0, 3, 7]), (-4, [0, 4, 7]), (3, [0, 4, 7]), (-2, [0, 4, 7])]   # Am - F - C - G
    nb = int(n / (bar * sr)) + 2
    rng = np.random.default_rng(4)
    for b in range(nb):
        root, tri = prog[b % 4]
        s0 = int(b * bar * sr); s1 = min(n, int((b + 1) * bar * sr + 0.3 * sr))
        if s0 >= n: break
        tt = t[s0:s1] - b * bar
        env = np.clip(tt / 0.6, 0, 1) * np.clip((bar + 0.3 - tt) / 0.6, 0, 1)
        for iv in tri:   # pad (detuned, two octaves)
            f = hz(root + iv + 12)
            out[s0:s1] += (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.004 * tt + 1.3)
                           + 0.25 * np.sin(2 * np.pi * 2 * f * tt)) * env * 0.035
        for k in range(8):   # bass on eighths, root
            bs = s0 + int(k * beat / 2 * sr); L = int(beat / 2 * sr * 0.9)
            if bs + L > n: break
            tb = np.arange(L) / sr; eb = np.exp(-tb * 6)
            f = hz(root - 12)
            out[bs:bs + L] += np.tanh(2.2 * np.sin(2 * np.pi * f * tb)) * eb * (0.10 if k % 2 == 0 else 0.06)
        notes = [root + 24 + iv for iv in (tri + [12, tri[1] + 12])]
        for k in range(16):   # plucked arpeggio on sixteenths
            ps = s0 + int(k * beat / 4 * sr); L = int(0.28 * sr)
            if ps + L > n: break
            f = hz(notes[(k * 3) % len(notes)]); tp = np.arange(L) / sr
            out[ps:ps + L] += (np.sin(2 * np.pi * f * tp) + 0.3 * np.sin(4 * np.pi * f * tp)) * np.exp(-tp * 14) * (0.05 if k % 4 == 0 else 0.03)
        for k in range(4):    # soft kick on each beat, hat on off-beats
            ks = s0 + int(k * beat * sr); L = int(0.25 * sr)
            if ks + L > n: break
            tk = np.arange(L) / sr; ph = 2 * np.pi * np.cumsum(50 + 70 * np.exp(-tk * 30)) / sr
            out[ks:ks + L] += np.sin(ph) * np.exp(-tk * 12) * 0.22
            hs = ks + int(beat / 2 * sr); Lh = int(0.05 * sr)
            if hs + Lh < n:
                nz = rng.normal(0, 1, Lh).astype(np.float32); nz = nz - np.convolve(nz, np.ones(6) / 6, mode="same")
                out[hs:hs + Lh] += nz * np.exp(-np.arange(Lh) / sr * 70) * 0.05
    for st in scene_starts[1:]:   # soft whoosh on transitions
        s0 = int((st - 0.3) * sr); L = int(0.55 * sr)
        if s0 < 0 or s0 + L > n: continue
        nz = np.convolve(rng.normal(0, 1, L).astype(np.float32), np.ones(30) / 30, mode="same")
        out[s0:s0 + L] += nz * np.sin(np.linspace(0, np.pi, L)) ** 2 * 0.5
    out = out / (np.sqrt(np.mean(out ** 2)) + 1e-9)          # unit RMS
    # ducking envelope from the voice
    k = int(0.05 * sr); e = np.sqrt(np.convolve(voice ** 2, np.ones(k) / k, mode="same"))
    e = np.convolve((e > 0.02).astype(np.float32), np.ones(int(0.35 * sr)) / int(0.35 * sr), mode="same")
    duck = 1.0 - 0.45 * np.clip(e, 0, 1)
    fade = np.minimum(1, np.minimum(t / 1.5, (t[-1] - t) / 2.0))
    return out * duck * fade

def encode_scene(sc, idx, total, path):
    tmp = path + ".part.mp4"
    nf = int(round(sc["dur"] * FPS))
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                          "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), tmp], stdin=subprocess.PIPE)
    for f in range(nf):
        t = f / FPS
        p.stdin.write(scene_frame(sc, idx, t, sc["t0"] + t, total).tobytes())
    p.stdin.close()
    if p.wait() != 0: raise SystemExit("ffmpeg failed on scene %s" % idx)
    os.replace(tmp, path)

def prepare(story):
    for sc in story["scenes"]:
        if sc["type"] == "map":
            sc["_map"] = MapLayer(sc["bbox"], sc.get("focus", []), sc.get("focus2", []), labels=sc.get("labels", []))
        elif sc["type"] == "intro":
            sc["_map"] = MapLayer([-170, -50, 190, 75], faint=True)
    for i, sc in enumerate(story["scenes"]):
        sc["_bg"] = make_bg_panel(sc["bg"], seed=i) if sc.get("bg") else None
    for i, sc in enumerate(story["scenes"]):
        sc["_wipe_out"] = i < len(story["scenes"]) - 1
        sc["_wipe_in"] = i > 0

def main():
    story = json.load(open(sys.argv[1])); out = sys.argv[2]
    speed = float(story.get("speed", 1.08)); target = float(story.get("max_seconds", 60))
    segs, sr = tts_all(story, speed)
    audio, total = build_timing(story, segs, sr)
    if total > target:
        speech = sum(len(x) for ss in segs for x in ss) / sr
        speed2 = min(1.3, speed * speech / max(speech - (total - target) - 1.0, 1.0))
        print(f"total {total:.1f}s > {target}s, re-voicing at speed {speed2:.2f}")
        segs, sr = tts_all(story, speed2)
        audio, total = build_timing(story, segs, sr)
    print(f"duration {total:.2f}s")
    prepare(story)
    numbered = 0
    for sc in story["scenes"]:
        if sc["type"] in ("map", "number", "bars", "diagram"): numbered += 1; sc["_num"] = numbered
    if "--still" in sys.argv:
        i = sys.argv.index("--still"); outdir = sys.argv[i + 2]; os.makedirs(outdir, exist_ok=True)
        for spec in sys.argv[i + 1].split(","):
            si, ts = spec.split(":"); sc = story["scenes"][int(si)]; ts = float(ts) if ts != "end" else sc["dur"] - 0.3
            scene_frame(sc, sc.get("_num"), ts, sc["t0"] + ts, total).save(os.path.join(outdir, f"still_{si}_{ts:.1f}.png"))
        return
    work = out + ".work"; os.makedirs(work, exist_ok=True)
    files = []
    for i, sc in enumerate(story["scenes"]):
        fp = os.path.join(work, f"s{i:02d}.mp4"); encode_scene(sc, sc.get("_num"), total, fp); files.append(fp)
        print("scene", i, "done", flush=True)
    with open(os.path.join(work, "list.txt"), "w") as f:
        for fp in files: f.write(f"file '{os.path.abspath(fp)}'\n")
    starts = [sc["t0"] for sc in story["scenes"]]
    peak = np.max(np.abs(audio)) or 1
    voice = audio / peak * 0.9
    vr = np.sqrt(np.mean(voice[np.abs(voice) > 0.01] ** 2))
    bed = music_bed2(len(audio), sr, starts, voice)
    mixd = voice + bed * vr * float(story.get("music_level", 0.22))
    mixd = np.clip(mixd, -1, 1)
    import soundfile as sf
    wav = os.path.join(work, "audio.wav"); sf.write(wav, mixd, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(work, "list.txt"),
                    "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-shortest", "-movflags", "+faststart", out + ".part.mp4"], check=True)
    os.replace(out + ".part.mp4", out)
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", out)

if __name__ == "__main__":
    main()
