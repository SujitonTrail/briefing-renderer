#!/usr/bin/env python3
"""The Briefing — v4: paper-cut motion graphics on top of the v3 renderer.
Usage: python3 render_v4.py story.json out.mp4 [--still SPECS OUTDIR]
Everything drawn here is generic, procedurally made paper-cut art (no photos, no logos)."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_v3 as B
from render_v3 import (W, H, FPS, PAPER, INK, YEL, RED, MAP_H, F, F_HEAD, F_TAG, F_LABEL, F_SMALL, F_BIG,
                       F_STATB, F_STATS, F_CTRY, ease, mix, shapes, lakes, _noise, _blur, _shift, wrap)
MAP_Y = B.MAP_Y
SHADOW = (38, 30, 22)

# ------------------------------------------------------------------ helpers
def spring(x, k=6.5, w=10.5):
    if x <= 0: return 0.0
    if x >= 1.5: return 1.0
    return 1 - math.exp(-k * x) * math.cos(w * x)

def q12(t):  # stop-motion time (12 fps) for wobbles
    return math.floor(t * 12) / 12

_FIB = None
def fiber(h, w):
    """Multiplicative paper-fibre texture, tiled from one 1024 tile."""
    global _FIB
    if _FIB is None:
        n = 1024; rng = np.random.default_rng(9)
        low = _noise(n, n, [(256, 1.0), (64, 0.5)], 31)
        fine = rng.normal(0, 1, (n, n)).astype(np.float32)
        st = np.asarray(Image.fromarray(rng.normal(0, 1, (n // 2, n // 24)).astype(np.float32), "F").resize((n, n), Image.BICUBIC))
        _FIB = (1 + 0.025 * low + 0.018 * fine + 0.015 * st).astype(np.float32)
    reps = (h // 1024 + 1, w // 1024 + 1)
    return np.tile(_FIB, reps)[:h, :w]

def texturize(im):
    a = np.asarray(im).astype(np.float32)
    a[..., :3] *= fiber(a.shape[0], a.shape[1])[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), im.mode)

def roughen(m, amt, seed=1):
    """Give a mask a hand-cut, slightly torn edge."""
    a = np.asarray(m.filter(ImageFilter.GaussianBlur(1.6)), np.float32) / 255
    nz = _noise(a.shape[0], a.shape[1], [(7, 1.0), (3, 0.6)], seed)
    a = np.clip((a - 0.5 + nz * 0.06 * amt) * 7 + 0.5, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))

class Sprite:
    def __init__(self, img):
        self.img = img
        self.sh = img.getchannel("A").filter(ImageFilter.GaussianBlur(5))

class _FillDraw:
    """ImageDraw proxy whose shapes are filled by default (cut paper is solid)."""
    def __init__(self, d): self._d = d
    def __getattr__(self, k):
        f = getattr(self._d, k)
        if k in ("polygon", "rectangle", "ellipse", "pieslice", "rounded_rectangle", "chord"):
            def g(*a, **kw):
                if "fill" not in kw and "outline" not in kw: kw["fill"] = 255
                return f(*a, **kw)
            return g
        return f

class PC:
    """Paper canvas: each layer is a cut piece of coloured paper casting a small shadow."""
    def __init__(self, w, h, ss=2, pad=26):
        self.ss, self.pad, self.w, self.h = ss, pad, w, h
        self.Wc, self.Hc = (w + 2 * pad) * ss, (h + 2 * pad) * ss
        self.img = Image.new("RGBA", (self.Wc, self.Hc), (0, 0, 0, 0))
    def P(self, x, y): return ((x + self.pad) * self.ss, (y + self.pad) * self.ss)
    def R(self, v): return v * self.ss
    def layer(self, fn, color, sh=(2, 3, 3, 0.38), rough=1.0, seed=1):
        m = Image.new("L", (self.Wc, self.Hc), 0); dd = _FillDraw(ImageDraw.Draw(m)); fn(dd, self.P, self.R)
        if rough: m = roughen(m, rough, seed)
        if sh:
            dx, dy, r, a = sh
            s = ImageChops.offset(m.filter(ImageFilter.GaussianBlur(r * self.ss)), int(dx * self.ss), int(dy * self.ss))
            lay = Image.new("RGBA", m.size, SHADOW + (0,)); lay.putalpha(s.point(lambda v: int(v * a))); self.img.alpha_composite(lay)
        f = Image.new("RGBA", m.size, tuple(color) + (255,)); f.putalpha(m); self.img.alpha_composite(f)
    def text(self, xy, txt, font, size, color, anchor="mm"):
        ImageDraw.Draw(self.img).text(self.P(*xy), txt, font=F(font, int(size * self.ss)), fill=tuple(color) + (255,), anchor=anchor)
    def sprite(self):
        im = self.img.resize((self.Wc // self.ss, self.Hc // self.ss), Image.LANCZOS)
        return Sprite(texturize(im))

def put(img, sp, cx, cy, scale=1.0, rot=0.0, alpha=1.0, lift=1.0):
    if sp is None or alpha <= 0.01 or scale <= 0.02: return
    im, sh = sp.img, sp.sh
    if abs(scale - 1) > 0.01:
        sz = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
        im = im.resize(sz, Image.BILINEAR); sh = sh.resize(sz, Image.BILINEAR)
    if abs(rot) > 0.3:
        im = im.rotate(rot, Image.BICUBIC, expand=True); sh = sh.rotate(rot, Image.BICUBIC, expand=True)
    x, y = int(cx - im.width / 2), int(cy - im.height / 2)
    alpha = min(alpha, 1.0)
    if lift > 0:
        k = 0.42 * alpha * min(lift, 1.2)
        s = sh.point(lambda v: int(v * k))
        ox, oy = int(6 * lift), int(9 * lift)
        img.paste(SHADOW + ((255,) if img.mode == "RGBA" else ()), (x + ox, y + oy, x + ox + im.width, y + oy + im.height), s)
    if alpha < 0.99:
        im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    img.paste(im, (x, y), im)

# ------------------------------------------------------------------ sprite library
_SP = {}
def SP(kind, size=80, **kw):
    key = (kind, size, tuple(sorted((k, str(v)) for k, v in kw.items())))
    if key in _SP: return _SP[key]
    s = size; sp = None
    if kind == "drone":
        c = PC(s, s)
        c.layer(lambda d, P, R: d.polygon([P(s * .95, s * .5), P(s * .2, s * .08), P(s * .38, s * .5), P(s * .2, s * .92)]), (66, 68, 74))
        c.layer(lambda d, P, R: d.polygon([P(s * .8, s * .5), P(s * .42, s * .36), P(s * .48, s * .5), P(s * .42, s * .64)]), (120, 122, 128), sh=(1, 1, 1, .3))
        sp = c.sprite()
    elif kind == "plane":
        c = PC(s, s)
        c.layer(lambda d, P, R: d.polygon([P(s * .95, s * .5), P(s * .85, s * .44), P(s * .1, s * .44), P(s * .05, s * .5), P(s * .1, s * .56), P(s * .85, s * .56)]), tuple(kw.get("color", (245, 242, 235))))
        c.layer(lambda d, P, R: d.polygon([P(s * .62, s * .47), P(s * .38, s * .05), P(s * .3, s * .05), P(s * .42, s * .47), P(s * .42, s * .53), P(s * .3, s * .95), P(s * .38, s * .95), P(s * .62, s * .53)]), tuple(kw.get("color2", (230, 226, 216))), sh=(1, 2, 2, .3))
        sp = c.sprite()
    elif kind == "burst":
        c = PC(s, s)
        for r, col in ((.5, (228, 70, 46)), (.36, (245, 150, 50)), (.22, (255, 214, 80))):
            pts = []
            for i in range(24):
                rr = r if i % 2 == 0 else r * .55; a = i * math.pi / 12
                pts.append((s / 2 + rr * s * math.cos(a), s / 2 + rr * s * math.sin(a)))
            c.layer(lambda d, P, R, pts=pts: d.polygon([P(*p) for p in pts]), col, sh=(1, 2, 2, .3))
        sp = c.sprite()
    elif kind == "flame":
        c = PC(s, int(s * 1.3))
        def tear(cx, cy, r, h):
            pts = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in range(-25, 206, 10)]
            return pts + [(cx - r * .3, cy - h * .5), (cx, cy - h), (cx + r * .35, cy - h * .45)]
        for (cy, r, h, col) in ((s * .95, s * .42, s * 1.05, (215, 55, 40)), (s * 1.0, s * .30, s * .78, (245, 140, 45)), (s * 1.05, s * .17, s * .48, (255, 214, 80))):
            pts = tear(s / 2, cy, r, h)
            c.layer(lambda d, P, R, pts=pts: d.polygon([P(*p) for p in pts]), col, sh=(1, 2, 2, .3))
        sp = c.sprite()
    elif kind == "virus":
        c = PC(s, s); cx = cy = s / 2
        def spikes(d, P, R):
            for i in range(10):
                a = i * math.pi / 5
                d.line([P(cx, cy), P(cx + .45 * s * math.cos(a), cy + .45 * s * math.sin(a))], fill=255, width=int(R(s * .06)))
                x, y = cx + .45 * s * math.cos(a), cy + .45 * s * math.sin(a)
                d.ellipse([P(x - s * .06, y - s * .06), P(x + s * .06, y + s * .06)], fill=255)
        c.layer(spikes, (176, 46, 40))
        c.layer(lambda d, P, R: d.ellipse([P(cx - s * .3, cy - s * .3), P(cx + s * .3, cy + s * .3)], fill=255), (222, 74, 56))
        c.layer(lambda d, P, R: [d.ellipse([P(cx + ox * s - s * .06, cy + oy * s - s * .06), P(cx + ox * s + s * .06, cy + oy * s + s * .06)], fill=255) for ox, oy in ((-.1, -.08), (.1, .05), (-.02, .14))], (245, 140, 120), sh=None)
        sp = c.sprite()
    elif kind == "bust":
        col = tuple(kw.get("color", (70, 96, 150)))
        c = PC(s, s)
        c.layer(lambda d, P, R: d.pieslice([P(s * .08, s * .55), P(s * .92, s * 1.4)], 180, 360, fill=255), col)
        c.layer(lambda d, P, R: d.ellipse([P(s * .32, s * .12), P(s * .68, s * .5)], fill=255), tuple(min(255, v + 25) for v in col))
        sp = c.sprite()
    elif kind == "coin":
        c = PC(s, s)
        c.layer(lambda d, P, R: d.ellipse([P(0, 0), P(s, s)], fill=255), (214, 160, 40))
        c.layer(lambda d, P, R: d.ellipse([P(s * .14, s * .14), P(s * .86, s * .86)], fill=255), (246, 196, 70), sh=(1, 1, 2, .3))
        c.layer(lambda d, P, R: d.ellipse([P(s * .32, s * .32), P(s * .68, s * .68)], outline=255, width=int(R(s * .07))), (214, 160, 40), sh=None, rough=0)
        sp = c.sprite()
    elif kind == "lock":
        c = PC(s, int(s * 1.25))
        c.layer(lambda d, P, R: d.arc([P(s * .2, s * .02), P(s * .8, s * .7)], 180, 360, fill=255, width=int(R(s * .12))), (170, 172, 176))
        c.layer(lambda d, P, R: d.line([P(s * .26, s * .36), P(s * .26, s * .6)], fill=255, width=int(R(s * .12))), (170, 172, 176), sh=None)
        c.layer(lambda d, P, R: d.line([P(s * .74, s * .36), P(s * .74, s * .6)], fill=255, width=int(R(s * .12))), (170, 172, 176), sh=None)
        c.layer(lambda d, P, R: d.rounded_rectangle([P(s * .05, s * .55), P(s * .95, s * 1.2)], radius=int(R(s * .08)), fill=255), (40, 40, 46))
        c.layer(lambda d, P, R: (d.ellipse([P(s * .43, s * .74), P(s * .57, s * .88)], fill=255), d.rectangle([P(s * .47, s * .82), P(s * .53, s * 1.02)], fill=255)), YEL, sh=None)
        sp = c.sprite()
    elif kind == "cloud":
        c = PC(int(s * 1.4), s)
        c.layer(lambda d, P, R: [d.ellipse([P(x - r, y - r), P(x + r, y + r)], fill=255) for x, y, r in ((s * .4, s * .6, s * .28), (s * .72, s * .45, s * .36), (s * 1.02, s * .62, s * .26))] + [d.rectangle([P(s * .4, s * .6), P(s * 1.02, s * .88)], fill=255)], (238, 242, 246))
        c.layer(lambda d, P, R: d.polygon([P(s * .78, s * .25), P(s * .6, s * .62), P(s * .74, s * .62), P(s * .62, s * 1.0), P(s * .92, s * .52), P(s * .78, s * .52), P(s * .9, s * .25)]), (228, 70, 46), sh=(1, 2, 2, .35))
        sp = c.sprite()
    elif kind in ("mail_x", "mail_ok"):
        c = PC(s, int(s * .72))
        c.layer(lambda d, P, R: d.rectangle([P(0, 0), P(s, s * .7)], fill=255), (246, 240, 226))
        c.layer(lambda d, P, R: d.polygon([P(0, 0), P(s / 2, s * .4), P(s, 0)]), (226, 216, 196), sh=(0, 1, 1, .3))
        ok = kind == "mail_ok"; col = (70, 160, 95) if ok else (220, 64, 46)
        c.layer(lambda d, P, R: d.ellipse([P(s * .66, s * .36), P(s * 1.0, s * .70)], fill=255), col, sh=(1, 1, 1, .3))
        if ok:
            c.layer(lambda d, P, R: d.line([P(s * .74, s * .53), P(s * .81, s * .61), P(s * .93, s * .45)], fill=255, width=int(R(s * .05))), (255, 255, 255), sh=None, rough=0)
        else:
            c.layer(lambda d, P, R: (d.line([P(s * .76, s * .46), P(s * .9, s * .6)], fill=255, width=int(R(s * .05))), d.line([P(s * .9, s * .46), P(s * .76, s * .6)], fill=255, width=int(R(s * .05)))), (255, 255, 255), sh=None, rough=0)
        sp = c.sprite()
    elif kind == "chip":
        c = PC(s, s)
        def pins(d, P, R):
            for k in range(6):
                o = s * (.22 + k * .112)
                d.rectangle([P(o - s * .025, 0), P(o + s * .025, s)], fill=255); d.rectangle([P(0, o - s * .025), P(s, o + s * .025)], fill=255)
        c.layer(pins, (196, 196, 188))
        c.layer(lambda d, P, R: d.rectangle([P(s * .12, s * .12), P(s * .88, s * .88)], fill=255), (36, 38, 42))
        c.layer(lambda d, P, R: d.rectangle([P(s * .3, s * .3), P(s * .7, s * .7)], fill=255), (70, 74, 80), sh=(1, 1, 1, .3))
        sp = c.sprite()
    elif kind in ("arrow_down", "arrow_up"):
        col = tuple(kw.get("color", (228, 70, 46) if kind == "arrow_down" else (70, 160, 95)))
        c = PC(int(s * .7), s)
        pts = [(s * .22, 0), (s * .48, 0), (s * .48, s * .55), (s * .68, s * .55), (s * .35, s), (s * .02, s * .55), (s * .22, s * .55)]
        if kind == "arrow_up": pts = [(x, s - y) for x, y in pts]
        c.layer(lambda d, P, R: d.polygon([P(*p) for p in pts]), col)
        sp = c.sprite()
    elif kind == "dome":
        c = PC(s, s); col = (240, 232, 214)
        c.layer(lambda d, P, R: d.rectangle([P(s * .05, s * .78), P(s * .95, s * .95)], fill=255), (214, 204, 182))
        c.layer(lambda d, P, R: [d.rectangle([P(s * (.14 + k * .1), s * .52), P(s * (.18 + k * .1), s * .78)], fill=255) for k in range(8)], col, sh=(1, 1, 1, .3))
        c.layer(lambda d, P, R: d.rectangle([P(s * .1, s * .45), P(s * .9, s * .53)], fill=255), (226, 216, 194))
        c.layer(lambda d, P, R: d.pieslice([P(s * .25, s * .12), P(s * .75, s * .78)], 180, 360, fill=255), col)
        c.layer(lambda d, P, R: d.rectangle([P(s * .46, s * .02), P(s * .54, s * .14)], fill=255), (226, 216, 194), sh=(1, 1, 1, .3))
        sp = c.sprite()
    elif kind == "vs":
        c = PC(s, s)
        c.layer(lambda d, P, R: d.ellipse([P(0, 0), P(s, s)], fill=255), YEL, rough=2.5)
        c.text((s / 2, s / 2 + 2), "VS", "InterDisplay-Black.otf", s * .42, INK)
        sp = c.sprite()
    elif kind == "sheet":
        c = PC(int(s * .75), s)
        c.layer(lambda d, P, R: d.rectangle([P(0, 0), P(s * .75, s)], fill=255), (248, 244, 234))
        c.layer(lambda d, P, R: [d.rectangle([P(s * .1, s * (.12 + k * .09)), P(s * (.65 if k % 3 else .45), s * (.12 + k * .09) + s * .025)], fill=255) for k in range(8)], (190, 184, 170), sh=None, rough=0)
        sp = c.sprite()
    elif kind == "neuron":
        c = PC(s, s); rng = np.random.default_rng(3)
        branches = []
        for i in range(7):
            a = i * 2 * math.pi / 7 + rng.normal(0, .2); x, y = s / 2, s / 2; seg = []
            for k in range(5):
                a += rng.normal(0, .35); x2, y2 = x + s * .085 * math.cos(a), y + s * .085 * math.sin(a); seg.append(((x, y), (x2, y2), max(2, 9 - 2 * k))); x, y = x2, y2
            branches.append(seg)
        c.layer(lambda d, P, R: [d.line([P(*a_), P(*b_)], fill=255, width=int(R(w_ * s / 160))) for seg in branches for a_, b_, w_ in seg], (240, 190, 50))
        c.layer(lambda d, P, R: d.ellipse([P(s * .38, s * .38), P(s * .62, s * .62)], fill=255), YEL)
        sp = c.sprite()
    elif kind == "pin":
        c = PC(int(s * .7), s)
        c.layer(lambda d, P, R: (d.ellipse([P(0, 0), P(s * .7, s * .7)], fill=255), d.polygon([P(s * .08, s * .5), P(s * .35, s), P(s * .62, s * .5)])), (228, 70, 46))
        c.layer(lambda d, P, R: d.ellipse([P(s * .2, s * .2), P(s * .5, s * .5)], fill=255), (250, 245, 235), sh=None)
        sp = c.sprite()
    elif kind == "tag":
        lines = kw.get("lines", []); f1 = F("Inter-Bold.otf", 30); f2 = F("InterDisplay-Black.otf", 46)
        tw = int(max([f1.getlength(lines[0])] + [f2.getlength(x) for x in lines[1:]]) + 90); th = 40 + 46 * len(lines)
        c = PC(tw, th)
        c.layer(lambda d, P, R: d.polygon([P(0, th * .5), P(30, 0), P(tw, 0), P(tw, th), P(30, th)]), (248, 242, 228), rough=1.2)
        c.layer(lambda d, P, R: d.ellipse([P(22, th * .5 - 8), P(38, th * .5 + 8)], fill=255), (60, 50, 40), sh=None, rough=0)
        c.text((58, 34), lines[0], "Inter-Bold.otf", 30, (90, 80, 70), "lm")
        for i, ln in enumerate(lines[1:]):
            c.text((58, 34 + 48 * (i + 1)), ln, "InterDisplay-Black.otf", 46, INK, "lm")
        sp = c.sprite()
    elif kind == "calendar":
        txt = kw.get("text", "NEXT WEEK"); c = PC(s, s)
        c.layer(lambda d, P, R: d.rectangle([P(0, s * .12), P(s, s)], fill=255), (248, 244, 234))
        c.layer(lambda d, P, R: d.rectangle([P(0, s * .12), P(s, s * .36)], fill=255), (228, 70, 46), sh=(0, 1, 1, .3))
        c.layer(lambda d, P, R: [d.rectangle([P(s * x - s * .03, 0), P(s * x + s * .03, s * .2)], fill=255) for x in (.28, .72)], (60, 60, 64), sh=None)
        c.text((s / 2, s * .66), txt, "InterDisplay-Black.otf", s * .135, INK)
        sp = c.sprite()
    elif kind in ("tower_ok", "tower_x"):
        c = PC(s, s); ok = kind == "tower_ok"
        c.layer(lambda d, P, R: d.polygon([P(s * .5, s * .22), P(s * .66, s * .95), P(s * .34, s * .95)]), (70, 72, 78))
        c.layer(lambda d, P, R: [d.line([P(s * (.5 - k * .045), s * (.4 + k * .14)), P(s * (.5 + k * .045), s * (.4 + k * .14))], fill=255, width=int(R(s * .03))) for k in range(1, 5)], (150, 152, 158), sh=None, rough=0)
        arc = (80, 190, 120) if ok else (170, 168, 160)
        c.layer(lambda d, P, R: [d.arc([P(s * (.5 - r), s * (.24 - r)), P(s * (.5 + r), s * (.24 + r))], a0, a1, fill=255, width=int(R(s * .05))) for r in (.16, .28) for a0, a1 in ((200, 250), (290, 340))], arc, sh=(1, 1, 1, .3), rough=0.4)
        c.layer(lambda d, P, R: d.ellipse([P(s * .43, s * .17), P(s * .57, s * .31)], fill=255), arc, sh=None)
        if not ok:
            c.layer(lambda d, P, R: d.ellipse([P(s * .56, s * .5), P(s * .98, s * .92)], fill=255), (220, 64, 46), sh=(1, 2, 2, .35))
            c.layer(lambda d, P, R: (d.line([P(s * .67, s * .61), P(s * .87, s * .81)], fill=255, width=int(R(s * .06))), d.line([P(s * .87, s * .61), P(s * .67, s * .81)], fill=255, width=int(R(s * .06)))), (255, 255, 255), sh=None, rough=0)
        sp = c.sprite()
    elif kind == "riverboat":   # generic paddlewheel river cruiser, side view (w = s, h = .55 s)
        h_ = int(s * .55); c = PC(s, h_)
        c.layer(lambda d, P, R: d.polygon([P(s * .04, h_ * .62), P(s * .98, h_ * .62), P(s * .92, h_ * .98), P(s * .1, h_ * .98)]), (246, 243, 236))
        c.layer(lambda d, P, R: d.rectangle([P(s * .06, h_ * .84), P(s * .95, h_ * .9)], fill=255), (200, 60, 50), sh=None, rough=.4)
        for k, (x0, x1, y0, y1) in enumerate(((.1, .9, .4, .62), (.16, .84, .22, .4), (.26, .72, .08, .22))):
            c.layer(lambda d, P, R, x0=x0, x1=x1, y0=y0, y1=y1: d.rectangle([P(s * x0, h_ * y0), P(s * x1, h_ * y1)], fill=255), (240, 236, 226), seed=k + 2)
            c.layer(lambda d, P, R, x0=x0, x1=x1, y0=y0, y1=y1: [d.rectangle([P(s * (x0 + .03 + i * .05), h_ * (y0 + .05)), P(s * (x0 + .055 + i * .05), h_ * (y1 - .05))], fill=255) for i in range(int((x1 - x0 - .04) / .05))], (60, 74, 96), sh=None, rough=0)
        c.layer(lambda d, P, R: (d.rectangle([P(s * .6, -h_ * .12), P(s * .64, h_ * .1)], fill=255), d.rectangle([P(s * .66, -h_ * .12), P(s * .7, h_ * .1)], fill=255)), (40, 40, 44))
        c.layer(lambda d, P, R: d.pieslice([P(-s * .06, h_ * .42), P(s * .2, h_ * 1.0)], 90, 270, fill=255), (196, 54, 44))
        c.layer(lambda d, P, R: [d.line([P(s * .07, h_ * .71), P(s * .07 - s * .12 * math.sin(a), h_ * .71 + h_ * .27 * math.cos(a))], fill=255, width=int(R(3))) for a in (0.4, 1.2, 2.0, 2.8)], (250, 230, 220), sh=None, rough=0)
        sp = c.sprite()
    elif kind == "tanker":     # generic tanker, side view, bow on the right (w = s, hull h = .3 s)
        h_ = int(s * .3); o = h_ * .9; c = PC(s, int(h_ * 1.9))
        c.layer(lambda d, P, R: d.polygon([P(0, o + h_ * .35), P(s * .9, o + h_ * .35), P(s, o + h_ * .2), P(s * .96, o + h_ * .98), P(s * .03, o + h_ * .98)]), (44, 56, 80))
        c.layer(lambda d, P, R: d.polygon([P(s * .02, o + h_ * .78), P(s * .975, o + h_ * .78), P(s * .96, o + h_ * .98), P(s * .03, o + h_ * .98)]), (170, 52, 42), sh=None)
        c.layer(lambda d, P, R: d.rectangle([P(s * .04, o - h_ * .55), P(s * .2, o + h_ * .36)], fill=255), (240, 236, 226))
        c.layer(lambda d, P, R: [d.rectangle([P(s * (.06 + i * .035), o - h_ * .4), P(s * (.08 + i * .035), o - h_ * .3)], fill=255) for i in range(4)], (60, 74, 96), sh=None, rough=0)
        c.layer(lambda d, P, R: d.rectangle([P(s * .12, o - h_ * .85), P(s * .16, o - h_ * .55)], fill=255), (196, 54, 44))
        c.layer(lambda d, P, R: [d.rectangle([P(s * (.26 + i * .11), o + h_ * .22), P(s * (.32 + i * .11), o + h_ * .35)], fill=255) for i in range(6)], (90, 104, 130), sh=(1, 1, 1, .3))
        sp = c.sprite()
    elif kind == "lifering":
        c = PC(s, s)
        c.layer(lambda d, P, R: d.ellipse([P(0, 0), P(s, s)], outline=255, width=int(R(s * .22))), (232, 70, 46))
        c.layer(lambda d, P, R: [d.pieslice([P(0, 0), P(s, s)], a, a + 22, fill=255) for a in (35, 125, 215, 305)], (250, 246, 238), sh=None, rough=0)
        c.layer(lambda d, P, R: d.ellipse([P(s * .22, s * .22), P(s * .78, s * .78)], fill=255), (0, 0, 0), sh=None, rough=0)
        im = c.img; a_ = np.asarray(im).copy(); blk = (a_[..., 0] < 8) & (a_[..., 1] < 8) & (a_[..., 2] < 8); a_[blk, 3] = 0
        c.img = Image.fromarray(a_); sp = c.sprite()
    elif kind == "rescueboat":
        h_ = int(s * .45); c = PC(s, h_)
        c.layer(lambda d, P, R: d.polygon([P(0, h_ * .5), P(s, h_ * .4), P(s * .88, h_ * .95), P(s * .08, h_ * .95)]), (236, 120, 40))
        c.layer(lambda d, P, R: d.rectangle([P(s * .3, h_ * .05), P(s * .62, h_ * .45)], fill=255), (246, 243, 236))
        c.layer(lambda d, P, R: d.rectangle([P(s * .36, h_ * .14), P(s * .56, h_ * .3)], fill=255), (60, 74, 96), sh=None, rough=0)
        sp = c.sprite()
    elif kind == "car":
        col = tuple(kw.get("color", (196, 60, 48))); h_ = int(s * .42); c = PC(s, h_)
        c.layer(lambda d, P, R: d.polygon([P(s * .02, h_ * .45), P(s * .25, h_ * .42), P(s * .36, h_ * .08), P(s * .72, h_ * .08), P(s * .86, h_ * .42), P(s * .98, h_ * .5), P(s * .98, h_ * .78), P(s * .02, h_ * .78)]), col)
        c.layer(lambda d, P, R: (d.polygon([P(s * .4, h_ * .16), P(s * .53, h_ * .16), P(s * .53, h_ * .42), P(s * .31, h_ * .42)]), d.polygon([P(s * .57, h_ * .16), P(s * .69, h_ * .16), P(s * .8, h_ * .42), P(s * .57, h_ * .42)])), (190, 210, 222), sh=None, rough=0)
        c.layer(lambda d, P, R: [d.ellipse([P(x - s * .1, h_ * .58), P(x + s * .1, h_ * .98)], fill=255) for x in (s * .24, s * .76)], (36, 36, 40))
        sp = c.sprite()
    elif kind == "puff":
        c = PC(s, s); rng = np.random.default_rng(int(kw.get("seed", 1)))
        circ = [(s * rng.uniform(.3, .7), s * rng.uniform(.3, .7), s * rng.uniform(.18, .3)) for _ in range(5)]
        c.layer(lambda d, P, R: [d.ellipse([P(x - r, y - r), P(x + r, y + r)], fill=255) for x, y, r in circ], tuple(kw.get("color", (150, 146, 140))), sh=(1, 2, 3, .25), rough=1.5)
        sp = c.sprite()
    elif kind == "crane":
        c = PC(s, s)
        c.layer(lambda d, P, R: (d.rectangle([P(s * .45, s * .1), P(s * .52, s)], fill=255), d.rectangle([P(s * .05, s * .1), P(s * .98, s * .16)], fill=255), d.line([P(s * .85, s * .16), P(s * .85, s * .5)], fill=255, width=int(R(3)))), (226, 170, 40))
        sp = c.sprite()
    elif kind == "strip":   # torn yellow highlighter strip
        w_, h_ = kw["w"], kw["h"]; c = PC(w_, h_, pad=14)
        c.layer(lambda d, P, R: d.polygon([P(0, h_ * .04), P(w_, 0), P(w_ * .995, h_), P(w_ * .005, h_ * .97)]), YEL, sh=(2, 3, 3, .3), rough=2.2, seed=w_)
        sp = c.sprite()
    elif kind == "statcard":
        big, small = kw["big"], kw["small"]
        bw = int(max(F_STATB.getlength(big), F_STATS.getlength(small)) + 70)
        c = PC(bw + 14, 224)
        c.layer(lambda d, P, R: d.rectangle([P(12, 14), P(bw + 12, 224)], fill=255), YEL, rough=1.6, seed=5)
        c.layer(lambda d, P, R: d.rectangle([P(0, 0), P(bw, 210)], fill=255), (24, 24, 26), rough=1.6, seed=6)
        c.text((35, 18), big, "InterDisplay-Black.otf", 110, YEL, "la")
        c.text((35, 150), small, "Inter-SemiBold.otf", 36, PAPER, "la")
        sp = c.sprite()
    elif kind == "crowdrow":
        wr = kw.get("w", 1200); n = kw.get("n", 14); rng = np.random.default_rng(kw.get("seed", 1))
        pal = [tuple(p) for p in kw.get("pal", [(70, 96, 150), (180, 70, 60), (90, 130, 100), (200, 160, 70), (120, 100, 150)])]
        c = PC(wr, s, pad=10)
        for i in range(n):
            x = i * wr / n + rng.uniform(-10, 10); col = pal[int(rng.integers(0, len(pal)))]; ss_ = s * rng.uniform(.85, 1.0)
            c.layer(lambda d, P, R, x=x, ss_=ss_: (d.pieslice([P(x, s - ss_ * .45), P(x + ss_ * .8, s + ss_ * .3)], 180, 360, fill=255),
                                                   d.ellipse([P(x + ss_ * .22, s - ss_ * .85), P(x + ss_ * .58, s - ss_ * .49)], fill=255)), col, sh=(2, 2, 2, .3), seed=i + 1)
        sp = c.sprite()
    _SP[key] = sp
    return sp

def paper_sheet(edge):  # transition sheet with one torn edge
    key = ("sheet_tr", edge)
    if key in _SP: return _SP[key]
    wv = W + 90; c = PC(wv, H, ss=1, pad=40)
    if edge == "right":
        pts = [(0, 0), (W, 0)] + [(W + 40 + 30 * math.sin(i * 1.7) + (i % 3) * 9, i * H / 40) for i in range(41)] + [(0, H)]
    else:
        pts = [(90 - 40 - 30 * math.sin(i * 1.3) - (i % 3) * 9, i * H / 40) for i in range(41)] + [(wv, H), (wv, 0)]
    c.layer(lambda d, P, R: d.polygon([P(x, y) for x, y in pts]), YEL, sh=(10, 0, 10, .45), rough=1.5)
    sp = c.sprite(); _SP[key] = sp
    return sp

# ------------------------------------------------------------------ paper-cut map
class PaperMap(B.MapLayer):
    OVER = 2.3
    def __init__(self, bbox, focus=(), focus2=(), faint=False):
        lon0, lat0, lon1, lat1 = bbox
        self.latc = (lat0 + lat1) / 2; self.lonc = (lon0 + lon1) / 2
        self.k = math.cos(math.radians(self.latc))
        self.BW, self.BH = int(W * self.OVER), int(MAP_H * self.OVER)
        self.scale = min(W / ((lon1 - lon0) * self.k), MAP_H / (lat1 - lat0))
        self.faint = faint
        hw = self.BW / 2 / (self.k * self.scale); hh = self.BH / 2 / self.scale
        win = (self.lonc - hw - 5, self.latc - hh - 5, self.lonc + hw + 5, self.latc + hh + 5)
        ss = 2; SW, SH = self.BW * ss, self.BH * ss
        masks = [Image.new("L", (SW, SH), 0) for _ in range(5)]
        dl, df, df2, db, dk = [ImageDraw.Draw(m) for m in masks]
        for iso, name, rings, sb in shapes():
            if sb[2] < win[0] or sb[0] > win[2] or sb[3] < win[1] or sb[1] > win[3]: continue
            isf = iso in focus or name in focus; isf2 = iso in focus2 or name in focus2
            for ring in rings:
                p = [(x * ss, y * ss) for x, y in (self.geo(lo, la) for lo, la in ring)]
                if len(p) < 3: continue
                dl.polygon(p, fill=255)
                if isf: df.polygon(p, fill=255)
                if isf2: df2.polygon(p, fill=255)
                db.line(p + [p[0]], fill=255, width=2 * ss, joint="curve")
        for rings, sb in lakes():
            if sb[2] < win[0] or sb[0] > win[2] or sb[3] < win[1] or sb[1] > win[3]: continue
            for ring in rings:
                p = [(x * ss, y * ss) for x, y in (self.geo(lo, la) for lo, la in ring)]
                if len(p) > 2: dk.polygon(p, fill=255)
        land, foc, foc2, bord, lake = [np.asarray(m.resize((self.BW, self.BH), Image.LANCZOS), np.float32) / 255 for m in masks]
        land = land * (1 - lake); foc *= (1 - lake); foc2 *= (1 - lake)
        h, w = land.shape; tex = fiber(h, w)
        # paper-cut sea: stacked bands of blue paper stepping down from the coast
        deep = np.array((50, 104, 162), np.float32); cols = [(70, 128, 184), (98, 158, 208), (138, 190, 228)]
        water = np.ones((h, w, 3), np.float32) * deep
        nz = _noise(h, w, [(9, 1.0), (4, .5)], 77)
        for (r, th), col in zip(((80, .05), (40, .09), (18, .15)), cols):
            m = np.clip((_blur(land, r) - th) * 40 + nz * .5 + .5, 0, 1)
            sh = _blur(_shift(m, 4, 6), 4) * (1 - m)
            water *= (1 - 0.28 * sh)[..., None]
            water = water * (1 - m[..., None]) + np.array(col, np.float32) * m[..., None]
        lw = (np.array(cols[1], np.float32) * 1.0)
        landc = np.array((236, 226, 204), np.float32)[None, None, :] + (_noise(h, w, [(300, 1.0)], 5) * 5)[..., None]
        lsh = _blur(_shift(land, 7, 10), 6)
        water *= (1 - 0.45 * lsh * (1 - land))[..., None]
        base = water * (1 - land[..., None]) + landc * land[..., None]
        ish = _blur(_shift(1 - lake, 3, 4), 3) * lake       # lakes cut into the land
        base = base * (1 - lake[..., None]) + (lw * (1 - 0.3 * ish[..., None])) * lake[..., None]
        base *= (1 - 0.22 * bord)[..., None]
        base *= tex[..., None]
        if faint:
            paper = np.array(PAPER, np.float32)[None, None, :]
            base = paper + (base - paper) * 0.4
            dim = base
        else:
            dim = base * (0.93 - 0.25 * land)[..., None]
            if foc2.max() > 0:
                f2 = np.array((200, 182, 150), np.float32)[None, None, :] * tex[..., None]
                s2 = _blur(_shift(foc2, 4, 6), 4) * (1 - foc2)
                dim *= (1 - 0.35 * s2)[..., None]
                dim = dim * (1 - foc2[..., None]) + f2 * foc2[..., None]
        self.base = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        self.dim = Image.fromarray(np.clip(dim, 0, 255).astype(np.uint8), "RGB")
        self.fsp = None
        if not faint and foc.max() > 0.05:
            ys, xs = np.where(foc > 0.02); pad = 30
            x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad); x1, y1 = min(w, xs.max() + pad), min(h, ys.max() + pad)
            fm = foc[y0:y1, x0:x1]; tx = tex[y0:y1, x0:x1]
            edge = np.clip(fm * (1 - _blur(fm, 2.5)) * 3.0, 0, 1)
            col = np.array(YEL, np.float32)[None, None, :] * tx[..., None]
            col = col * (1 - 0.25 * edge[..., None])
            rgba = np.dstack([np.clip(col, 0, 255), fm * 255]).astype(np.uint8)
            self.fsp = Image.fromarray(rgba, "RGBA"); self.fsh = self.fsp.getchannel("A").filter(ImageFilter.GaussianBlur(9))
            self.fpos = (x0, y0)
        m = Image.new("L", (W, MAP_H), 255); md = ImageDraw.Draw(m)
        for i in range(22):
            a = int(255 * (i / 22) ** 1.2); md.line([(0, i), (W, i)], fill=a); md.line([(0, MAP_H - 1 - i), (W, MAP_H - 1 - i)], fill=a)
        self.edge = m
        yy, xx = np.mgrid[0:MAP_H, 0:W]
        v = 1 - 0.34 * (((xx - W / 2) / (W / 2)) ** 2 * 0.6 + ((yy - MAP_H / 2) / (MAP_H / 2)) ** 2 * 0.25) ** 1.3
        self.vig = Image.fromarray((np.clip(v, 0, 1)[..., None].repeat(3, 2) * 255).astype(np.uint8), "RGB")
        self.lift = 0.0

    def frame(self, zoom, hl, lift=None):
        lift = hl if lift is None else lift; self.lift = lift
        im = Image.blend(self.base, self.dim, min(hl, 1)) if 0 < hl < 1 else (self.dim if hl >= 1 else self.base)
        cw, ch = W / zoom, MAP_H / zoom
        x0, y0 = (self.BW - cw) / 2, (self.BH - ch) / 2
        self._crop = (x0, y0, zoom)
        fr = im.resize((W, MAP_H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        if self.fsp is not None and hl > 0:
            bx0, by0 = self.fpos; sw_, sh_ = self.fsp.size
            ix0, iy0 = max(bx0, x0 - 40), max(by0, y0 - 40); ix1, iy1 = min(bx0 + sw_, x0 + cw + 40), min(by0 + sh_, y0 + ch + 40)
            if ix1 > ix0 + 2 and iy1 > iy0 + 2:
                box = (int(ix0 - bx0), int(iy0 - by0), int(ix1 - bx0), int(iy1 - by0))
                size = (max(1, int((box[2] - box[0]) * zoom)), max(1, int((box[3] - box[1]) * zoom)))
                sub = self.fsp.crop(box).resize(size, Image.BILINEAR); ssub = self.fsh.crop(box).resize(size, Image.BILINEAR)
                px, py = int((box[0] + bx0 - x0) * zoom), int((box[1] + by0 - y0) * zoom)
                a = min(hl, 1.0); L = max(lift, 0)
                fr.paste(SHADOW, (px + int(12 * L), py + int(17 * L), px + int(12 * L) + size[0], py + int(17 * L) + size[1]), ssub.point(lambda v: int(v * 0.5 * a)))
                if a < 0.99: sub.putalpha(sub.getchannel("A").point(lambda v: int(v * a)))
                fr.paste(sub, (px - int(3 * L), py - int(6 * L)), sub)
        if not self.faint:
            fr = ImageChops.multiply(fr, self.vig)
            fd = ImageDraw.Draw(fr); fd.line([(0, 0), (W, 0)], fill=INK, width=3); fd.line([(0, MAP_H - 2), (W, MAP_H - 2)], fill=INK, width=3)
            return fr.convert("RGBA")
        fr = fr.convert("RGBA"); fr.putalpha(self.edge)
        return fr

def draw_map_labels(img, ml, labels, a):
    if not labels or a <= 0: return
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    flags = []
    for lb in labels:
        x, y = ml.to_screen(lb["lon"], lb["lat"]); txt = lb["text"].upper()
        sp = 8; tw = sum(F_CTRY.getlength(c) + sp for c in txt) - sp
        fw = 50 if lb.get("flag") else 0
        cx = x - (tw + fw + (14 if fw else 0)) / 2
        if fw: flags.append((lb["flag"], cx, y - 19)); cx += fw + 14
        for c in txt:
            d.text((cx, y), c, font=F_CTRY, fill=(40, 34, 28, int(185 * a)), anchor="lm"); cx += F_CTRY.getlength(c) + sp
    img.alpha_composite(ov)
    for code, x, y in flags:
        f = B.flag_img(code, 50, 37)
        if f is None: continue
        yo = (1 - a) * 20
        img.paste(SHADOW + (255,), (int(x) + 4, int(y + yo) + 5, int(x) + 54, int(y + yo) + 42), Image.new("L", (50, 37), int(150 * a)))
        ff = f.copy(); ff.putalpha(int(255 * a)); img.paste(ff, (int(x), int(y + yo)), ff)

# ------------------------------------------------------------------ paper-cut panels (number / bars scenes)
def make_paper_panel(kind, seed=3):
    rng = np.random.default_rng(seed); PW, PH = W + 160, MAP_H + 100
    spec = {"city": ((214, 206, 190), False), "river": ((226, 220, 204), False), "chart": ((224, 214, 192), False), "circuit": ((40, 86, 66), True), "servers": ((28, 40, 64), True), "neurons": ((48, 36, 82), True)}
    if kind not in spec: return None
    bgc, dark = spec[kind]
    layers = []
    if kind == "chart":
        for li, (col, top, drop) in enumerate((((206, 194, 168), .25, .35), ((188, 172, 142), .42, .32), ((168, 150, 120), .6, .28))):
            c = PC(PW, PH, ss=1, pad=0)
            xs = np.linspace(0, PW, 26); ys = PH * top + np.linspace(0, drop * PH, 26) + np.cumsum(rng.normal(0, 18, 26))
            pts = [(x, y) for x, y in zip(xs, ys)] + [(PW, PH), (0, PH)]
            c.layer(lambda d, P, R, pts=pts: d.polygon([P(*p) for p in pts]), col, sh=(0, -4, 5, .35), rough=1.2, seed=li + 3)
            layers.append((c.sprite().img, 0.4 + li * 0.5))
    elif kind == "city":
        hz = PH * .62
        for li, (col, hmin, hmax, f) in enumerate((((194, 184, 166), 120, 330, .2), ((168, 158, 140), 60, 220, .45))):
            c = PC(PW, PH, ss=1, pad=0); bl = []; x = -20
            while x < PW:
                w_ = rng.uniform(60, 140); h_ = rng.uniform(hmin, hmax); bl.append((x, hz - h_, x + w_, hz + 30)); x += w_ + rng.uniform(0, 18)
            dome = (PW * .58, hz - 150)
            def sk(d, P, R, bl=bl, li=li):
                for a, b_, cc, dd_ in bl: d.rectangle([P(a, b_), P(cc, dd_)], fill=255)
                if li == 0:
                    d.rectangle([P(PW * .3 - 14, hz - 420), P(PW * .3 + 14, hz)], fill=255)
                    d.pieslice([P(PW * .3 - 30, hz - 450), P(PW * .3 + 30, hz - 390)], 180, 360, fill=255)
            c.layer(sk, col, sh=(2, 2, 3, .3), rough=.8, seed=li + 4)
            layers.append((c.sprite().img, f))
        c = PC(PW, PH, ss=1, pad=0)
        c.layer(lambda d, P, R: d.rectangle([P(0, hz + 10), P(PW, PH)], fill=255), (120, 116, 110), sh=(0, -3, 4, .35), rough=.5)
        c.layer(lambda d, P, R: [d.rectangle([P(x_, hz + 120), P(x_ + 60, hz + 130)], fill=255) for x_ in range(0, PW, 120)], (230, 222, 200), sh=None, rough=0)
        layers.append((c.sprite().img, 0.0))
    elif kind == "river":
        c = PC(PW, PH, ss=1, pad=0); hz = PH * .5; bl = []
        x = 0
        while x < PW:
            w_ = rng.uniform(50, 110); h_ = rng.uniform(60, 230); bl.append((x, hz - h_, x + w_, hz + 20)); x += w_ + rng.uniform(4, 20)
        c.layer(lambda d, P, R: [d.rectangle([P(a, b_), P(cc, dd_)], fill=255) for a, b_, cc, dd_ in bl], (196, 186, 166), sh=(2, 2, 3, .3), rough=.8)
        layers.append((c.sprite().img, 0.3))
        c = PC(PW, PH, ss=1, pad=0)
        def bridge(d, P, R):
            d.rectangle([P(0, hz - 40), P(PW, hz - 26)], fill=255)
            for i in range(14):
                x0 = i * PW / 14; d.line([P(x0, hz - 26), P(x0 + PW / 28, hz - 70 - 30 * math.sin(i / 13 * math.pi)), P(x0 + PW / 14, hz - 26)], fill=255, width=6)
            for px_ in (PW * .3, PW * .62): d.rectangle([P(px_ - 10, hz - 120), P(px_ + 10, hz + 30)], fill=255)
        c.layer(bridge, (150, 140, 124), sh=(2, 2, 2, .3), rough=.4)
        c.layer(lambda d, P, R: d.rectangle([P(PW * .74, hz + 30), P(PW, hz + 70)], fill=255), (150, 116, 84), sh=(2, 4, 4, .4))
        c.layer(lambda d, P, R: d.polygon([P(0, hz + 40)] + [P(x_, hz + 34 + 6 * math.sin(x_ / 40)) for x_ in range(0, PW + 40, 40)] + [P(PW, PH), P(0, PH)]), (104, 164, 210), sh=(0, -3, 4, .3), rough=.6)
        layers.append((c.sprite().img, 0.12))
        cf = PC(PW, PH, ss=1, pad=0)
        cf.layer(lambda d, P, R: d.polygon([P(0, PH * .76)] + [P(x_, PH * .74 + 8 * math.sin(x_ / 55 + 1)) for x_ in range(0, PW + 55, 55)] + [P(PW, PH), P(0, PH)]), (74, 136, 192), sh=(0, -4, 5, .35), rough=.6, seed=9)
        front = texturize(cf.sprite().img)
        base = texturize(Image.new("RGB", (PW, PH), bgc))
        return {"base": base, "layers": [(texturize(l), f) for l, f in layers], "dark": False, "front": front}
    elif kind == "circuit":
        c = PC(PW, PH, ss=1, pad=0); g = 40; segs = []
        for _ in range(120):
            x = int(rng.integers(0, PW // g)) * g; y = int(rng.integers(0, PH // g)) * g; pts = [(x, y)]
            for _ in range(int(rng.integers(2, 5))):
                dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1)][int(rng.integers(0, 6))]; L = int(rng.integers(2, 6)) * g
                x, y = x + dx * L, y + dy * L; pts.append((x, y))
            segs.append(pts)
        c.layer(lambda d, P, R: [(d.line([P(*p) for p in pts], fill=255, width=8, joint="curve"), d.ellipse([P(pts[-1][0] - 10, pts[-1][1] - 10), P(pts[-1][0] + 10, pts[-1][1] + 10)], fill=255)) for pts in segs], (74, 132, 100), sh=(2, 3, 3, .45), rough=0.6)
        c.layer(lambda d, P, R: [d.line([P(*p) for p in pts], fill=255, width=8, joint="curve") for pts in segs[:14]], (206, 172, 70), sh=(2, 3, 3, .45), rough=0.6)
        layers.append((c.sprite().img, 0.5))
        cp = PC(PW, PH, ss=1, pad=0)
        chip = SP("chip", 240)
        cp.img.alpha_composite(chip.img, (int(PW * .9) - chip.img.width // 2, int(PH * .1) - chip.img.height // 2))
        layers.append((cp.img, 1.2))
    elif kind == "servers":
        c = PC(PW, PH, ss=1, pad=0); rw, gap = 200, 44; x0 = (PW - (5 * rw + 4 * gap)) // 2
        c.layer(lambda d, P, R: [d.rectangle([P(x0 + r * (rw + gap), 30), P(x0 + r * (rw + gap) + rw, PH - 30)], fill=255) for r in range(5)], (46, 62, 96), sh=(3, 4, 4, .45))
        c.layer(lambda d, P, R: [d.rectangle([P(x0 + r * (rw + gap) + 16, 50 + u * 50), P(x0 + r * (rw + gap) + rw - 16, 50 + u * 50 + 38)], fill=255)
                                 for r in range(5) for u in range(int((PH - 90) / 50))], (36, 50, 80), sh=(1, 2, 2, .4), rough=0.5)
        leds = [(x0 + r * (rw + gap) + rw - 40 - k * 20, 50 + u * 50 + 19, rng.random()) for r in range(5) for u in range(int((PH - 90) / 50)) for k in range(2)]
        c.layer(lambda d, P, R: [d.ellipse([P(x - 6, y - 6), P(x + 6, y + 6)], fill=255) for x, y, s_ in leds if s_ < .45], (228, 70, 46), sh=None, rough=0)
        c.layer(lambda d, P, R: [d.ellipse([P(x - 6, y - 6), P(x + 6, y + 6)], fill=255) for x, y, s_ in leds if s_ >= .75], (80, 200, 120), sh=None, rough=0)
        layers.append((c.sprite().img, 0.6))
    elif kind == "neurons":
        for li, (col, n) in enumerate((((80, 64, 132), 16), ((112, 92, 172), 9))):
            c = PC(PW, PH, ss=1, pad=0); cells = []
            for _ in range(n):
                x, y = rng.uniform(0, PW), rng.uniform(0, PH); br = []
                for _ in range(int(rng.integers(4, 7))):
                    a = rng.uniform(0, 2 * np.pi); px, py = x, y
                    for k in range(6):
                        a += rng.normal(0, .4); nx, ny = px + 34 * math.cos(a), py + 34 * math.sin(a); br.append(((px, py), (nx, ny), max(3, 12 - 2 * k))); px, py = nx, ny
                cells.append((x, y, br))
            c.layer(lambda d, P, R, cells=cells: [(d.ellipse([P(x - 18, y - 18), P(x + 18, y + 18)], fill=255), [d.line([P(*a_), P(*b_)], fill=255, width=w_) for a_, b_, w_ in br]) for x, y, br in cells], col, sh=(3, 4, 4, .45), rough=0.6, seed=li)
            layers.append((c.sprite().img, 0.5 + li * 0.7))
    base = Image.new("RGB", (PW, PH), bgc); base = texturize(base)
    return {"base": base, "layers": [(texturize(l), f) for l, f in layers], "dark": dark}

def panel_front(img, pn, t):
    if pn is None or pn.get("front") is None: return
    ox = 80 + 6 * math.sin(q12(t) * 1.3); oy = 50 + 3 * math.sin(q12(t) * 2.1)
    fr = pn["front"].crop((int(ox), int(oy), int(ox) + W, int(oy) + MAP_H))
    img.alpha_composite(fr, (0, MAP_Y))

def panel_frame(pn, t, dur):
    PW, PH = pn["base"].size
    fr = pn["base"].crop((80, 50, 80 + W, 50 + MAP_H)).convert("RGBA")
    for l, f in pn["layers"]:
        ox = 80 + 50 * f * math.sin(t * 0.35) - 30 * f * t / max(dur, 1); oy = 50 + 18 * f * math.sin(t * 0.27)
        fr.alpha_composite(l.crop((int(ox), int(oy), int(ox) + W, int(oy) + MAP_H)))
    d = ImageDraw.Draw(fr); d.line([(0, 0), (W, 0)], fill=INK, width=3); d.line([(0, MAP_H - 2), (W, MAP_H - 2)], fill=INK, width=3)
    return fr

# ------------------------------------------------------------------ props (visual storytelling)
def _bez(a, b, lift, u):
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy) + 1e-6
    nx, ny = -dy / L, dx / L
    if ny > 0: nx, ny = -nx, -ny
    cx, cy = mx + nx * lift * L, my + ny * lift * L
    x = (1 - u) ** 2 * a[0] + 2 * (1 - u) * u * cx + u * u * b[0]; y = (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * cy + u * u * b[1]
    tx = 2 * (1 - u) * (cx - a[0]) + 2 * u * (b[0] - cx); ty = 2 * (1 - u) * (cy - a[1]) + 2 * u * (b[1] - cy)
    return x, y, math.degrees(math.atan2(-ty, tx))

def draw_props(img, sc, t):
    props = sc.get("props") or []
    if not props: return
    ml = sc.get("_map"); d = ImageDraw.Draw(img)
    def XY(p, k="pos"):
        v = p.get(k)
        if v is None: return None
        if ml is not None and sc["type"] in ("map", "intro") and not p.get("screen"): return ml.to_screen(v[0], v[1])
        return (v[0], v[1])
    for pi, p in enumerate(props):
        ln = sc["lines"][min(p.get("at_line", 0), len(sc["lines"]) - 1)] if sc["lines"] else {"start": 0}
        t0 = ln["start"] + p.get("delay", 0)
        if sc["type"] == "map": t0 = max(t0, 1.75)
        tr = t - t0
        if tr < 0: continue
        ex = None
        if "exit_line" in p:
            el = sc["lines"][min(p["exit_line"], len(sc["lines"]) - 1)]
            ex = t - (el["start"] + p.get("exit_delay", 0))
        k = p["kind"]
        if k == "sprite":
            sp = SP(p["sprite"], p.get("size", 80), **p.get("args", {}))
            x, y = XY(p); s = 1.0; rot = p.get("rot", 0); a = 1.0
            an = p.get("anim", "pop")
            if an == "pop": s = spring(tr / .55); rot += (1 - min(tr / .5, 1)) * -14
            elif an == "slide_l": x -= (1 - spring(tr / .65)) * 800
            elif an == "slide_r": x += (1 - spring(tr / .65)) * 800
            elif an == "rise": y += (1 - spring(tr / .7)) * 500
            elif an == "drop": y -= (1 - spring(tr / .6)) * 600; rot += (1 - min(tr / .6, 1)) * 20
            if an == "drive":
                u = ease(min(tr / p.get("dur", 1.2), 1.0)); x = p["from_x"] + (x - p["from_x"]) * u
            if "capsize_line" in p:
                cl = sc["lines"][min(p["capsize_line"], len(sc["lines"]) - 1)]["start"] + p.get("capsize_delay", 0)
                cu = t - cl
                if cu > 0:
                    k_ = ease(min(cu / p.get("capsize_dur", 2.2), 1.0)); jolt = 6 * math.exp(-cu * 5) * math.sin(cu * 30)
                    rot += p.get("capsize_rot", -70) * k_ + jolt; y += p.get("capsize_drop", 70) * k_; x += p.get("capsize_dx", 30) * k_
            tq = q12(tr) + pi
            if p.get("float"): y += 7 * math.sin(tq * 2.3)
            if p.get("flicker"): s *= 1 + 0.07 * math.sin(tq * 9.0); rot += 3 * math.sin(tq * 7)
            if p.get("bob"): rot += 4 * math.sin(tq * 3)
            if ex is not None and ex > 0:
                e = ease(ex / .6); y += 700 * e * e; rot += 35 * e; a = 1 - e
            put(img, sp, x, y, s, rot, a)
        elif k == "scatter":
            sp = SP(p["sprite"], p.get("size", 44), **p.get("args", {}))
            for j, pt in enumerate(p["points"]):
                tj = tr - j * p.get("stagger", .12)
                if tj < 0: continue
                x, y = ml.to_screen(*pt) if (ml is not None and sc["type"] in ("map", "intro")) else pt
                tq = q12(tj) + j
                put(img, sp, x, y + (6 * math.sin(tq * 2.5) if p.get("float") else 0), spring(tj / .5), 10 * math.sin(tq * 1.7))
        elif k == "route":
            a_, b_ = XY(p, "from"), XY(p, "to"); dur = p.get("dur", 1.2); u1 = min(tr / dur, 1.0); lift = p.get("lift", .25)
            N = 70; pts = [_bez(a_, b_, lift, u1 * i / N) for i in range(N + 1)]
            for i in range(0, N, 2):
                d.line([pts[i][:2], pts[i + 1][:2]], fill=(250, 246, 236), width=10)
            for i in range(0, N, 2):
                d.line([pts[i][:2], pts[i + 1][:2]], fill=tuple(p.get("color", INK)), width=5)
            trav = p.get("traveler")
            if trav and u1 < 1.0 or (trav and p.get("keep")):
                x, y, ang = pts[-1]; put(img, SP(trav, p.get("tsize", 60)), x, y, 1.0, ang)
            if u1 >= 1.0 and p.get("arrow", True):
                x, y, ang = pts[-1]; r = math.radians(ang)
                tip = (x + 18 * math.cos(r), y - 18 * math.sin(r)); l1 = (x + 18 * math.cos(r + 2.5), y - 18 * math.sin(r + 2.5)); l2 = (x + 18 * math.cos(r - 2.5), y - 18 * math.sin(r - 2.5))
                d.polygon([tip, l1, l2], fill=INK)
            if p.get("label") and tr > dur * .6:
                mx, my, _ = _bez(a_, b_, lift, .5)
                put(img, SP("tag", 0, lines=tuple([p["label"]])), mx + p.get("ldx", 0), my + p.get("ldy", -60), spring((tr - dur * .6) / .5), -3)
        elif k == "intercept":
            for j, pt in enumerate(p["points"]):
                tj = tr - j * .35
                if tj < 0: continue
                x, y = XY({"pos": pt}) if ml is not None else pt
                if tj < .9:
                    put(img, SP("drone", 58), x - 120 + 120 * ease(tj / .9), y - 30 + 30 * ease(tj / .9), 1.0, -12 + 12 * ease(tj / .9))
                else:
                    tb = tj - .9; put(img, SP("burst", 110), x, y, spring(tb / .4) * (1 - 0.25 * min(max(tb - 1.2, 0), 1)), tb * 30, 1 - min(max(tb - 1.6, 0) / .5, 1))
        elif k == "swarm":
            rng = np.random.default_rng(p.get("seed", 7)); n = p.get("n", 16); dur = p.get("dur", 1.6)
            src = p["from"]; tgts = p["targets"]
            for j in range(n):
                tj = tr - j * p.get("stagger", .09)
                sjit = (src[0] + rng.uniform(-1.5, 1.5), src[1] + rng.uniform(-1.5, 1.5)); tg = tgts[j % len(tgts)]
                f_end = rng.uniform(.5, .92) if rng.random() > p.get("reach", .15) else 1.0
                if tj < 0: continue
                a_ = ml.to_screen(*sjit); b_ = ml.to_screen(*tg); u = min(tj / dur, f_end)
                x, y, ang = _bez(a_, b_, .18, u)
                if tj / dur < f_end:
                    put(img, SP("drone", 40), x, y, 1.0, ang, lift=.6)
                else:
                    tb = tj - f_end * dur
                    if tb < .8: put(img, SP("burst", 54), x, y, spring(tb / .3) * (1 - tb / .8 * .5), tb * 40, 1 - max(tb - .45, 0) / .35, lift=.4)
        elif k == "stream":
            a_, b_ = XY(p, "from"), XY(p, "to"); n = p.get("n", 8); dur = p.get("dur", .9); hits = 0
            tgt = p.get("target")
            for j in range(n):
                tj = tr - j * p.get("stagger", .14)
                if tj < 0: continue
                u = tj / dur
                if u < 1:
                    x, y, ang = _bez(a_, b_, .3, ease(u)); put(img, SP(p.get("sprite", "coin"), p.get("size", 54)), x, y, 1 - .3 * u, tj * 200, lift=.7)
                else: hits = max(hits, tj - dur)
            if tgt:
                bump = 1 + 0.12 * math.exp(-hits * 6) if hits > 0 else 1
                put(img, SP(tgt, p.get("tsize", 120)), b_[0], b_[1], spring(tr / .5) * bump, 0)
        elif k == "fan":
            sp = SP(p.get("sprite", "sheet"), p.get("size", 150)); n = p.get("n", 8); spr = p.get("spread", 80)
            x, y = XY(p)
            for j in range(n):
                tj = tr - j * .08
                if tj < 0: continue
                ang = (-spr / 2 + j * spr / max(n - 1, 1)) * spring(tj / .6)
                r = math.radians(ang); hh = sp.img.height / 2 - 26
                put(img, sp, x - math.sin(r) * hh, y - math.cos(r) * hh, 1.0, ang, lift=.8)
        elif k == "mailrow":
            n = p.get("n", 6); y = p["pos"][1]; x0 = p["pos"][0]; gap = p.get("gap", 130)
            fl = sc["lines"][min(p.get("fix_line", 1), len(sc["lines"]) - 1)]["start"] + p.get("fix_delay", .3)
            for j in range(n):
                tj = tr - j * .1
                if tj < 0: continue
                fixed = t - fl - j * .3
                kind = "mail_ok" if fixed > 0 else "mail_x"
                s = spring(tj / .5) if fixed <= 0 else 1 + 0.15 * math.exp(-fixed * 7) * math.sin(fixed * 20)
                put(img, SP(kind, p.get("size", 96)), x0 + j * gap, y + 4 * math.sin(q12(tj) * 3 + j), s, 4 * math.sin(j * 1.3))
        elif k == "switch":
            sl = sc["lines"][min(p.get("switch_line", 0), len(sc["lines"]) - 1)]["start"] + p.get("switch_delay", .8)
            order = p.get("order") or list(range(len(p["points"])))
            for j, pt in enumerate(p["points"]):
                tj = tr - j * p.get("stagger", .06)
                if tj < 0: continue
                x, y = ml.to_screen(*pt) if (ml is not None and sc["type"] in ("map", "intro")) else pt
                fs = t - sl - order.index(j) * p.get("switch_stagger", .08)
                kind_ = p["b"] if fs > 0 else p["a"]
                s = spring(tj / .5) if fs <= 0 else 1 + 0.18 * math.exp(-fs * 7) * math.sin(fs * 22)
                put(img, SP(kind_, p.get("size", 56)), x, y, s, 0)
        elif k == "traffic":
            y = p["pos"][1]; n = p.get("n", 5); sp_ = p.get("speed", 160); span = W + 400
            cols = [(196, 60, 48), (60, 90, 150), (230, 180, 60), (90, 130, 100), (240, 236, 226)]
            for j in range(n):
                x = (j * span / n + sp_ * tr * (1 if j % 2 == 0 else 1.3)) % span - 200
                put(img, SP("car", p.get("size", 150), color=cols[j % len(cols)]), x, y + (j % 2) * 26 + 2 * math.sin(q12(tr) * 8 + j), 1.0, 0, lift=.6)
        elif k == "smoke":
            for e, (ex, ey) in enumerate(p["points"]):
                for j in range(p.get("n", 5)):
                    u = ((tr * p.get("rate", .35) + j / p.get("n", 5) + e * .13) % 1.0)
                    if tr * p.get("rate", .35) + j / p.get("n", 5) < 0: continue
                    put(img, SP("puff", p.get("size", 90), seed=j % 3 + 1, color=p.get("color", [150, 146, 140])), ex + 40 * u + 10 * math.sin(u * 6 + j), ey - 260 * u, .5 + 1.3 * u, 0, alpha=min(1, u * 5) * (1 - u) * .9, lift=.3)
        elif k == "haze":
            a = ease(tr / p.get("dur", 3.0)) * p.get("max", .7)
            if a > 0:
                key = "_haze_tex"
                if key not in _SP:
                    nz = _noise(MAP_H, W, [(180, 1.0), (60, .5)], 3)
                    g = np.linspace(.55, 1.0, MAP_H)[:, None]
                    al = np.clip((.75 + .25 * nz) * g, 0, 1)
                    _SP[key] = (Image.new("RGB", (W, MAP_H), (196, 186, 166)), al)
                col, al = _SP[key]
                m = Image.fromarray((al * 255 * a).astype(np.uint8))
                img.paste(col, (0, MAP_Y), m)
        elif k == "crowd":
            y = p["pos"][1]
            for j, (dy, sz, sd, pal) in enumerate(((-60, 70, 3, [(150, 140, 120), (170, 160, 140)]), (-30, 82, 5, None), (0, 96, 9, None))):
                tj = tr - j * .18
                if tj < 0: continue
                args = {"w": 1200, "n": 13 + j, "seed": sd}
                if pal: args["pal"] = pal
                yy = y + dy + (1 - spring(tj / .7)) * 260 + 3 * math.sin(q12(tj) * 4 + j)
                put(img, SP("crowdrow", sz, **args), W / 2, yy, 1.0, 0, lift=.6)

# ------------------------------------------------------------------ text pieces in paper style
def draw_headline(img, d, text, highlight, y, t, font=F_HEAD, maxw=960, lead=1.08, start=0.15):
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
    lh = int(font.size * lead); asc = font.getbbox("Hg")
    sw = spring((t - start - 0.4) / 0.55)
    for li, toks in enumerate(lines):
        a = ease((t - start - li * 0.12) / 0.35)
        yy = y + li * lh + int((1 - a) * 40)
        line = " ".join(x[0] for x in toks); x = (W - font.getlength(line)) / 2
        for wi, (w_, hl) in enumerate(toks):
            if hl and sw > 0:
                wx = x + font.getlength(" ".join(z[0] for z in toks[:wi]) + (" " if wi else ""))
                sw_ = int(font.getlength(w_) + 34); sh_ = int(asc[3] - asc[1] - 2)
                sp = SP("strip", 0, w=sw_, h=sh_)
                cw = int(sp.img.width * min(sw, 1.0))
                if cw > 4:
                    sub = Sprite(sp.img.crop((0, 0, cw, sp.img.height))); sub.sh = sp.sh.crop((0, 0, cw, sp.img.height))
                    put(img, sub, wx - 17 - 14 + cw / 2, yy + asc[1] + 4 + sh_ / 2, 1.0, -0.8, lift=.8)
        d = ImageDraw.Draw(img)
        d.text((x, yy), line, font=font, fill=mix(PAPER, INK, a))
    return y + len(lines) * lh

def draw_stat(img, st, t):
    if t <= 0: return
    sp = SP("statcard", 0, big=st["big"], small=st["small"])
    s = spring(t / .55); x1 = W - 60
    cx = x1 - sp.img.width / 2 + 26 + (1 - s) * (sp.img.width + 80)
    put(img, sp, cx, 1000 + 112, 1.0, -2.0 * (1 - min(t / .6, 1)) - 1.0, lift=1.0)

def draw_orgs(d, sc, t):
    items = [("org", o) for o in (sc.get("orgs") or [])] + [("person", p) for p in (sc.get("people") or [])]
    if not items: return
    def at(o):
        ln = sc["lines"][min(o.get("at_line", 0), len(sc["lines"]) - 1)]
        return min(spring((t - ln["start"] - 0.1) / 0.5), 1.08)
    def wh(kind, o): return (B._badge_w(o["name"]), 72) if kind == "org" else (B._person_w(o), 84)
    def draw(kind, o, x, y):
        a = at(o)
        if a <= 0: return
        if kind == "org": B.draw_badge(d, x, y, o["name"], o.get("icon", "institution"), a)
        else: B.draw_person(d, x, y, o, a)
    if sc["type"] == "map":
        y = MAP_Y + 36
        for kind, o in items:
            w_, h_ = wh(kind, o); x = 40 - (1 - min(at(o), 1)) * 500; draw(kind, o, x, y); y += h_ + 22
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
            for it in row: draw(*it, x, y); x += wh(*it)[0] + 28
            y += max(wh(*it)[1] for it in row) + 22

# ------------------------------------------------------------------ frame
def scene_frame(sc, idx, t, gt, total):
    img = B.BG.copy().convert("RGBA"); d = ImageDraw.Draw(img)
    typ = sc["type"]; dur = sc["dur"]; FG = INK
    if typ in ("map", "intro"):
        ml = sc["_map"]
        zoom = B.cam_zoom(t, dur, ml.OVER) if typ == "map" else 1.0 + 0.06 * (t / dur)
        hl = ease((t - 1.15) / 0.6) if typ == "map" else 0
        lift = spring((t - 1.15) / 0.7) if typ == "map" else 0
        fr = ml.frame(zoom, hl, lift)
        img.paste(fr, (0, MAP_Y), fr); d = ImageDraw.Draw(img)
        if typ == "map": draw_map_labels(img, ml, sc.get("labels", []), ease((t - 1.3) / 0.5))
    pn = sc.get("_panel")
    if pn is not None:
        img.paste(panel_frame(pn, t, dur), (0, MAP_Y)); d = ImageDraw.Draw(img)
        if pn["dark"]: FG = PAPER
    if typ == "intro":
        draw_props(img, sc, t); d = ImageDraw.Draw(img)
        a = ease(t / 0.4); o = 150
        d.text((W / 2, 520 + o), sc.get("title", "THE BRIEFING"), font=B.F_TITLE, fill=mix(PAPER, INK, a), anchor="mm")
        txt = sc.get("subtitle", ""); f2 = F("InterDisplay-Black.otf", 112); tw = f2.getlength(txt)
        sw = spring((t - 0.3) / 0.55)
        if sw > 0:
            sp = SP("strip", 0, w=int(tw + 40), h=124); cw = int(sp.img.width * min(sw, 1))
            if cw > 4:
                sub = Sprite(sp.img.crop((0, 0, cw, sp.img.height))); sub.sh = sp.sh.crop((0, 0, cw, sp.img.height))
                put(img, sub, (W - tw) / 2 - 20 - 14 + cw / 2, 690 + o, 1.0, -1.0)
        d = ImageDraw.Draw(img)
        d.text((W / 2, 690 + o), txt, font=f2, fill=mix(PAPER, INK, a), anchor="mm")
        d.text((W / 2, 860 + o), sc.get("date", ""), font=F_LABEL, fill=mix(PAPER, INK, ease((t - .5) / .4)), anchor="mm")
        d.text((W / 2, 915 + o), sc.get("edition", ""), font=F_SMALL, fill=mix(PAPER, (90, 90, 90), ease((t - .6) / .4)), anchor="mm")
    elif typ == "section":
        lab = sc["label"].upper(); fs = F("InterDisplay-Black.otf", 170 if len(lab) <= 8 else 130); tw = fs.getlength(lab)
        s = spring(t / 0.55)
        sp = SP("strip", 0, w=int(tw + 80), h=180)
        put(img, sp, W / 2 - (1 - s) * 1200, 920, 1.0, -2.0, lift=1.0)
        d = ImageDraw.Draw(img); a = ease((t - .1) / 0.3)
        d.text((W / 2, 920 - (1 - a) * 30), lab, font=fs, fill=mix(PAPER, INK, a), anchor="mm")
        if sc.get("count"):
            d.text((W / 2, 1090), sc["count"], font=F_TAG, fill=mix(PAPER, INK, ease((t - 0.25) / 0.3)), anchor="mm")
    elif typ == "outro":
        draw_headline(img, d, sc["headline"], sc.get("highlight"), 560, t, font=F("InterDisplay-Black.otf", 110))
        d = ImageDraw.Draw(img); y = 1060; a = ease((t - 0.6) / 0.4)
        if sc.get("sources"): d.text((W / 2, y), "SOURCES", font=F_TAG, fill=mix(PAPER, INK, a), anchor="mm")
        for i, r in enumerate(wrap(" · ".join(sc.get("sources", [])), F_SMALL, 900)):
            d.text((W / 2, y + 60 + i * 48), r, font=F_SMALL, fill=mix(PAPER, (80, 80, 80), a), anchor="mm")
    else:
        B.draw_tag(d, f"{idx:02d}", sc["tag"], t)
        if typ == "map":
            ml = sc["_map"]
            for j, mk in enumerate(sc.get("markers", [])):
                x, y = ml.to_screen(mk["lon"], mk["lat"])
                B.draw_marker(d, x, y, mk.get("label"), t - 1.6 - 0.35 * j, mk.get("dx", 40), mk.get("dy", -90), mk.get("flag"))
        elif typ == "bars":
            items = sc["items"][:5]; vmax = max(abs(it["value"]) for it in items) or 1
            y0 = 420; rowh = min(150, 700 // max(len(items), 1)); cx = W / 2 + 60
            d.line([(cx, y0 - 30), (cx, y0 + rowh * len(items) - 20)], fill=FG, width=4)
            for j, it in enumerate(items):
                p = spring((t - 0.3 - 0.15 * j) / 0.6); yy = y0 + j * rowh
                bl = (W / 2 - 160) * abs(it["value"]) / vmax * p; col = YEL if it["value"] >= 0 else RED
                x0_, x1_ = (cx, cx + bl) if it["value"] >= 0 else (cx - bl, cx)
                if bl > 2:
                    d.rectangle([x0_ + 6, yy + 8, x1_ + 6, yy + rowh * 0.55 + 8], fill=(120, 105, 85))
                    d.rectangle([x0_, yy, x1_, yy + rowh * 0.55], fill=col)
                la = ease((t - 0.2 - 0.15 * j) / 0.3)
                d.text((cx - 20 if it["value"] >= 0 else cx + 20, yy + rowh * 0.27), it["label"], font=F_LABEL, fill=mix(PAPER, FG, la), anchor="rm" if it["value"] >= 0 else "lm")
                vt = it.get("text", f'{it["value"]:+g}')
                if p > 0.3: d.text((cx + bl + 18 if it["value"] >= 0 else cx - bl - 18, yy + rowh * 0.27), vt, font=F_LABEL, fill=FG, anchor="lm" if it["value"] >= 0 else "rm")
            if sc.get("note"):
                d.text((W / 2, y0 + rowh * len(items) + 30), sc["note"], font=F_SMALL, fill=mix(PAPER, (90, 90, 90), ease((t - .8) / .4)), anchor="mm")
        elif typ == "number":
            n = sc["number"]; p = ease((t - 0.3) / 1.4); v = n * p
            val = sc.get("prefix", "") + f"{v:.{sc.get('decimals', 0)}f}" + sc.get("suffix", "")
            fb = F_BIG
            if fb.getlength(val) > 960: fb = F("InterDisplay-Black.otf", int(330 * 960 / fb.getlength(val)))
            pop = spring((t - 0.25) / 0.6)
            d.text((W / 2 + 8, 640 + 10 + (1 - min(pop, 1)) * 60), val, font=fb, fill=(20, 20, 20) if FG == PAPER else (190, 178, 156), anchor="mm")
            d.text((W / 2, 640 + (1 - min(pop, 1)) * 60), val, font=fb, fill=YEL if FG == PAPER else INK, anchor="mm")
            ua = ease((t - 0.6) / 0.4)
            d.text((W / 2, 860), sc.get("unit", "").upper(), font=F_TAG, fill=mix((40, 40, 40) if FG == PAPER else PAPER, FG, ua), anchor="mm")
        draw_props(img, sc, t)
        if typ == "diagram": panel_front(img, pn, t)
        d = ImageDraw.Draw(img)
        draw_orgs(d, sc, t)
        st = sc.get("stat")
        if st:
            ln = sc["lines"][min(st.get("at_line", 1), len(sc["lines"]) - 1)]
            draw_stat(img, st, t - ln["start"])
        d = ImageDraw.Draw(img)
        draw_headline(img, d, sc["headline"], sc.get("highlight"), 1240, t)
        d = ImageDraw.Draw(img)
    B.draw_caption(d, sc["lines"], t)
    B.draw_progress(d, gt, total)
    if idx is not None and sc.get("_wipe_out"):
        w = ease((t - (dur - 0.32)) / 0.32)
        if w > 0:
            sp = paper_sheet("right"); put(img, sp, -sp.img.width / 2 + (sp.img.width - 20) * w, H / 2, 1.0, 0, lift=1.4)
    if sc.get("_wipe_in"):
        w = ease(t / 0.3)
        if w < 1:
            sp = paper_sheet("left"); put(img, sp, sp.img.width / 2 - 70 + (W + 120) * w, H / 2, 1.0, 0, lift=1.4)
    return img.convert("RGB")

def prepare(story):
    for i, sc in enumerate(story["scenes"]):
        if sc["type"] == "map":
            sc["_map"] = PaperMap(sc["bbox"], sc.get("focus", []), sc.get("focus2", []))
        elif sc["type"] == "intro":
            sc["_map"] = PaperMap([-170, -50, 190, 75], faint=True)
        sc["_panel"] = make_paper_panel(sc["bg"], seed=i) if sc.get("bg") else None
    for i, sc in enumerate(story["scenes"]):
        sc["_wipe_out"] = i < len(story["scenes"]) - 1
        sc["_wipe_in"] = i > 0

B.scene_frame = scene_frame
B.prepare = prepare

if __name__ == "__main__":
    B.main()
