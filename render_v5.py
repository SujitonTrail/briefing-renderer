#!/usr/bin/env python3
"""The Briefing — v5: adds three things on top of the v4 paper-cut renderer.

  1. "hook" scene type  — an animated opening card (kicker, headline with torn yellow strip, props).
  2. Word-by-word captions — the word being spoken gets a yellow marker, spoken words turn dark.
  3. "photo" prop       — a real photo in a torn white paper frame with a slow push-in and a credit line.

Usage: python3 render_v5.py story.json out.mp4 [--still SPECS OUTDIR]
render_v5.py imports render_v4.py, which imports render_v3.py. All three files must sit in one folder.

hook scene:
  {"type":"hook","kicker":"TONIGHT","headline":"...","highlight":"...","hold":5,
   "lines":[{"text":"..."}],"props":[...]}          (prop positions are screen pixels, 1080x1920)

photo prop (any story scene; position is always in screen pixels):
  {"kind":"photo","file":"photos/rocket.jpg","pos":[540,640],"w":560,"aspect":1.33,
   "credit":"NASA","at_line":0,"rot":-2,"exit_line":2}
  - "file" is relative to NEWS60_ROOT (the work folder). A missing or unreadable file is skipped.
  - "credit" is drawn under the picture inside the frame. Always give one.
Photos are supplied by the caller; this file never downloads anything."""
import math, os, sys
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_v3 as B
import render_v4 as V4
from render_v3 import W, H, PAPER, INK, YEL, F, F_CAP, F_TAG, ease, mix, wrap
from render_v4 import spring, q12, put, roughen, texturize, Sprite, SHADOW

ROOT = B.ROOT
F_CREDIT = F("Inter-Medium.otf", 24)
F_HOOK = F("InterDisplay-Black.otf", 118)
F_KICK = F("Inter-Bold.otf", 40)

# ------------------------------------------------------------------ 2. word-by-word captions
def _word_index(ln, words, t):
    """Which word is being spoken at time t (words share the line's time by length)."""
    span = max(ln["end"] - ln["start"], 0.05)
    u = min(max((t - ln["start"]) / span, 0.0), 0.9999)
    wts = [len(w) + 1.5 for w in words]; tot = sum(wts); acc = 0.0
    for i, wt in enumerate(wts):
        acc += wt / tot
        if u < acc: return i
    return len(words) - 1

def draw_caption(d, lines, t):
    cur = None
    for ln in lines:
        if ln["start"] <= t < ln["end"] + 0.12: cur = ln
    if not cur: return
    a = ease((t - cur["start"]) / 0.15)
    rows = wrap(cur["text"], F_CAP, 920)
    words = [w for r in rows for w in r.split()]
    if not words: return
    done = t >= cur["end"]
    wi = len(words) if done else _word_index(cur, words, t)
    lh = 64; y = 1640 - (len(rows) - 1) * lh // 2
    k = 0; space = F_CAP.getlength(" ")
    for i, r in enumerate(rows):
        x = (W - F_CAP.getlength(r)) / 2; yy = y + i * lh
        for w in r.split():
            wl = F_CAP.getlength(w)
            if k == wi:
                d.rounded_rectangle([x - 8, yy + 4, x + wl + 8, yy + 60], radius=8, fill=mix(PAPER, YEL, a))
                col = INK
            elif k < wi: col = (45, 45, 45)
            else: col = (150, 144, 132)
            d.text((x, yy), w, font=F_CAP, fill=mix(PAPER, col, a))
            x += wl + space; k += 1

# ------------------------------------------------------------------ 3. photo prop
_PH = {}
def _photo(p):
    """Frame sprite + oversized inner picture for the slow push-in. None if the file can't be used."""
    w = int(p.get("w", 520)); asp = float(p.get("aspect", 4 / 3)); credit = str(p.get("credit", ""))[:60]
    key = (p.get("file"), w, round(asp, 3), credit)
    if key in _PH: return _PH[key]
    out = None
    try:
        path = p["file"] if os.path.isabs(p["file"]) else os.path.join(ROOT, p["file"])
        im = Image.open(path); im.load(); im = im.convert("RGB")
        if min(im.size) < 200: raise ValueError("image too small")
        h = int(w / asp)
        # centre-crop to the frame's aspect
        if im.width / im.height > asp:
            nw = int(im.height * asp); x0 = (im.width - nw) // 2; im = im.crop((x0, 0, x0 + nw, im.height))
        else:
            nh = int(im.width / asp); y0 = (im.height - nh) // 2; im = im.crop((0, y0, im.width, y0 + nh))
        inner = im.resize((int(w * 1.14), int(h * 1.14)), Image.LANCZOS)
        bd = 22; foot = 58 if credit else bd; pad = 26
        fw, fh = w + 2 * bd, h + bd + foot
        m = Image.new("L", (fw + 2 * pad, fh + 2 * pad), 0)
        ImageDraw.Draw(m).rectangle([pad, pad, pad + fw, pad + fh], fill=255)
        m = roughen(m, 1.3, seed=len(credit) + w)
        fr = Image.new("RGBA", m.size, (250, 247, 240, 255)); fr.putalpha(m); fr = texturize(fr)
        if credit:
            ImageDraw.Draw(fr).text((pad + bd, pad + bd + h + foot / 2 - 2), "Photo: " + credit, font=F_CREDIT,
                                    fill=(95, 90, 82, 255), anchor="lm")
        out = {"frame": fr, "sh": m.filter(ImageFilter.GaussianBlur(5)), "inner": inner, "w": w, "h": h,
               "ox": pad + bd, "oy": pad + bd}
    except Exception as e:
        print(f"photo skipped ({p.get('file')}): {e}", flush=True)
    _PH[key] = out
    return out

def _draw_photo(img, sc, p, pi, t):
    ph = _photo(p)
    if ph is None: return
    ln = sc["lines"][min(p.get("at_line", 0), len(sc["lines"]) - 1)] if sc["lines"] else {"start": 0}
    t0 = ln["start"] + p.get("delay", 0)
    if sc["type"] == "map": t0 = max(t0, 1.75)
    tr = t - t0
    if tr < 0: return
    x, y = p["pos"]; rot = p.get("rot", -2); a = 1.0
    s = spring(tr / .6); y += (1 - min(s, 1)) * 500; rot += (1 - min(tr / .6, 1)) * 10
    y += 5 * math.sin((q12(tr) + pi) * 1.9)
    if "exit_line" in p:
        el = sc["lines"][min(p["exit_line"], len(sc["lines"]) - 1)]
        ex = t - (el["start"] + p.get("exit_delay", 0))
        if ex > 0:
            e = ease(ex / .6); y += 700 * e * e; rot += 30 * e; a = 1 - e
    # slow push-in inside the frame
    u = min(tr / max(sc["dur"] - t0, 1.0), 1.0); z = 1.14 - 0.14 * (1 - u)   # crop window grows smaller -> zooms in
    cw, ch = int(ph["w"] * (1.14 / z) ), int(ph["h"] * (1.14 / z))
    cw = min(cw, ph["inner"].width); ch = min(ch, ph["inner"].height)
    cx0 = (ph["inner"].width - cw) // 2; cy0 = (ph["inner"].height - ch) // 2
    pic = ph["inner"].crop((cx0, cy0, cx0 + cw, cy0 + ch)).resize((ph["w"], ph["h"]), Image.BILINEAR)
    fr = ph["frame"].copy(); fr.paste(pic, (ph["ox"], ph["oy"]))
    sp = Sprite.__new__(Sprite); sp.img = fr; sp.sh = ph["sh"]
    put(img, sp, x, y, 1.0, rot, a, lift=1.2)

_v4_draw_props = V4.draw_props
def draw_props(img, sc, t):
    props = sc.get("props") or []
    if not props: return
    photos = [(i, p) for i, p in enumerate(props) if p.get("kind") == "photo"]
    if len(photos) < len(props):
        keep = dict(sc); keep["props"] = [p for p in props if p.get("kind") != "photo"]
        _v4_draw_props(img, keep, t)
    for i, p in photos: _draw_photo(img, sc, p, i, t)

# ------------------------------------------------------------------ 1. hook scene
def _hook_frame(sc, t, gt, total):
    img = B.BG.copy().convert("RGBA"); dur = sc["dur"]
    draw_props(img, sc, t)
    d = ImageDraw.Draw(img)
    kick = sc.get("kicker")
    if kick:
        s = spring(t / .5); tw = sum(F_KICK.getlength(ch) + 5 for ch in kick.upper())
        x = (W - tw) / 2 - (1 - min(s, 1)) * 700; y = 470
        d.rectangle([x - 26, y - 38, x + tw + 22, y + 38], fill=INK)
        for ch in kick.upper():
            d.text((x, y), ch, font=F_KICK, fill=YEL, anchor="lm"); x += F_KICK.getlength(ch) + 5
    V4.draw_headline(img, d, sc["headline"], sc.get("highlight"), 600, t, font=F_HOOK, maxw=980, start=0.2)
    d = ImageDraw.Draw(img)
    # a thin underline that draws itself across the card for the length of the scene
    yb = 1500; u = ease(t / max(dur - 0.4, 0.5))
    d.line([(120, yb), (120 + (W - 240) * u, yb)], fill=INK, width=6)
    B.draw_caption(d, sc["lines"], t)
    B.draw_progress(d, gt, total)
    if sc.get("_wipe_out"):
        w = ease((t - (dur - 0.32)) / 0.32)
        if w > 0:
            sp = V4.paper_sheet("right"); put(img, sp, -sp.img.width / 2 + (sp.img.width - 20) * w, H / 2, 1.0, 0, lift=1.4)
    if sc.get("_wipe_in"):
        w = ease(t / 0.3)
        if w < 1:
            sp = V4.paper_sheet("left"); put(img, sp, sp.img.width / 2 - 70 + (W + 120) * w, H / 2, 1.0, 0, lift=1.4)
    return img.convert("RGB")

_v4_scene_frame = V4.scene_frame
def scene_frame(sc, idx, t, gt, total):
    if sc["type"] == "hook": return _hook_frame(sc, t, gt, total)
    return _v4_scene_frame(sc, idx, t, gt, total)

# ------------------------------------------------------------------ wire up
B.draw_caption = draw_caption
V4.draw_props = draw_props
B.scene_frame = scene_frame

if __name__ == "__main__":
    B.main()
