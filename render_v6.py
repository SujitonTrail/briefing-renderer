#!/usr/bin/env python3
"""The Briefing — v6: a pack of extra paper-cut shapes and ready-made actions on top of v5.

Usage: python3 render_v6.py story.json out.mp4 [--still SPECS OUTDIR]
render_v6.py imports render_v5.py -> render_v4.py -> render_v3.py. All four files must sit in one folder.
Everything here is generic, procedurally drawn paper-cut art (no photos, no logos).

NEW SPRITES (use anywhere a sprite name is accepted: "sprite", "scatter", "stream", "switch", "fan"):
  space/science : rocket, satellite, telescope, microscope
  law/politics  : gavel, block, ballot_box, ballot, bank
  industry      : factory, oil_rig, barrel, pylon, bolt, turbine_mast, turbine_blades
  weather       : storm, wave, tree
  transport     : cargo_ship, truck, train, helicopter, tank
  health        : hospital, pill_bottle, syringe
  tech          : phone, laptop, robot
  other         : house
  Vehicles face right. "size" is the width of a 1.0-wide shape; wider shapes scale with it.

NEW ACTIONS (prop kinds; "pos" is lon/lat on map scenes unless "screen": true, pixels elsewhere):
  {"kind":"launch","pos":[x,y],"size":150,"rise":620,"hold":0.7,"dur":2.6,"at_line":0}
      a rocket sits on smoke, shakes, then lifts off upward by "rise" pixels with flame and a smoke trail.
  {"kind":"strike","pos":[x,y],"size":190,"hits":2,"at_line":0}
      a gavel swings down onto a block ("pos" is the block) "hits" times.
  {"kind":"spin","sprite":"storm","pos":[x,y],"to":[x2,y2],"size":170,"speed":120,"dur":3.0,"at_line":0}
      a sprite that turns on the spot ("speed" in degrees per second); with "to" it also travels there.
  {"kind":"turbine","pos":[x,y],"size":150,"speed":140,"at_line":0}
      a wind turbine with turning blades ("pos" is the hub).
All four accept "delay", "exit_line" and "exit_delay" like other props."""
import math, os, sys
from PIL import ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_v3 as B
import render_v4 as V4
import render_v5 as V5
from render_v3 import YEL, ease
from render_v4 import PC, put, spring, q12

WHT = (245, 242, 235); GRY = (150, 152, 158); LGR = (200, 200, 192); DRK = (42, 42, 48); RD = (228, 70, 46)
BLU = (70, 96, 150); LBL = (150, 199, 230); GRN = (70, 160, 95); DGN = (74, 104, 66); ORG = (245, 140, 45)
BRN = (150, 108, 66); DBR = (104, 70, 44); GLD = (214, 160, 40)
NS = None  # no shadow

def _blade(a, r=0.46, w=0.07):
    """One turbine blade from the hub (0.5,0.5) at angle a (degrees)."""
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a)); px, py = -sa, ca
    return ("p", [(0.5 + px * w * .5, 0.5 + py * w * .5), (0.5 + ca * r * .35 + px * w, 0.5 + sa * r * .35 + py * w),
                  (0.5 + ca * r, 0.5 + sa * r), (0.5 - px * w * .5, 0.5 - py * w * .5)])

def _wave(y0, amp, ph):
    pts = [(x / 20 * 1.6, y0 + amp * math.sin(x / 20 * 2 * math.pi * 2 + ph)) for x in range(21)]
    return ("p", pts + [(1.6, 0.7), (0, 0.7)])

# name: (width, height, [(colour, shadow, [shapes])...])   — units are multiples of "size"
SPEC = {
    "rocket": (1.0, 1.3, [
        (RD, 1, [("p", [(0.31, 0.82), (0.08, 1.2), (0.31, 1.1)]), ("p", [(0.69, 0.82), (0.92, 1.2), (0.69, 1.1)])]),
        (DRK, 1, [("p", [(0.38, 1.12), (0.62, 1.12), (0.68, 1.27), (0.32, 1.27)])]),
        (WHT, 1, [("p", [(0.5, 0.02), (0.67, 0.3), (0.7, 1.14), (0.3, 1.14), (0.33, 0.3)])]),
        (RD, NS, [("p", [(0.5, 0.02), (0.655, 0.27), (0.345, 0.27)])]),
        (BLU, NS, [("e", 0.41, 0.45, 0.59, 0.63)]),
        (RD, NS, [("r", 0.31, 0.92, 0.69, 0.97)])]),
    "satellite": (1.6, 0.8, [
        (GRY, 1, [("l", [(0.45, 0.4), (1.15, 0.4)], 0.05)]),
        (BLU, 1, [("r", 0.02, 0.24, 0.5, 0.56), ("r", 1.1, 0.24, 1.58, 0.56)]),
        (LBL, NS, [("l", [(0.18, 0.24), (0.18, 0.56)], 0.02), ("l", [(0.34, 0.24), (0.34, 0.56)], 0.02),
                   ("l", [(1.26, 0.24), (1.26, 0.56)], 0.02), ("l", [(1.42, 0.24), (1.42, 0.56)], 0.02)]),
        (GLD, 1, [("r", 0.62, 0.18, 0.98, 0.66)]),
        (WHT, 1, [("p", [(0.8, 0.2), (0.66, 0.02), (0.94, 0.02)])])]),
    "telescope": (1.2, 1.2, [
        (DRK, 1, [("l", [(0.6, 0.58), (0.34, 1.16)], 0.05), ("l", [(0.6, 0.58), (0.86, 1.16)], 0.05), ("l", [(0.6, 0.58), (0.6, 1.16)], 0.05)]),
        (BLU, 1, [("p", [(0.1, 0.62), (0.2, 0.82), (1.02, 0.34), (0.9, 0.12)])]),
        (YEL, 1, [("p", [(0.9, 0.12), (1.02, 0.34), (1.1, 0.29), (0.98, 0.07)])]),
        (DRK, NS, [("p", [(0.1, 0.62), (0.2, 0.82), (0.26, 0.78), (0.16, 0.59)])])]),
    "microscope": (1.0, 1.3, [
        (DRK, 1, [("rr", 0.1, 1.14, 0.9, 1.28, 0.04)]),
        (DRK, 1, [("p", [(0.6, 1.16), (0.76, 1.16), (0.82, 0.42), (0.58, 0.2), (0.48, 0.3), (0.68, 0.5)])]),
        (LGR, 1, [("r", 0.18, 0.76, 0.72, 0.83)]),
        (GRY, 1, [("p", [(0.28, 0.12), (0.44, 0.03), (0.63, 0.44), (0.47, 0.53)])]),
        (YEL, NS, [("p", [(0.47, 0.53), (0.63, 0.44), (0.66, 0.52), (0.52, 0.6)])])]),
    "gavel": (1.2, 1.0, [
        (BRN, 1, [("p", [(0.5, 0.46), (0.58, 0.38), (1.12, 0.9), (1.04, 0.98)])]),
        (DBR, 1, [("p", [(0.1, 0.4), (0.45, 0.05), (0.7, 0.3), (0.35, 0.65)])]),
        (YEL, NS, [("p", [(0.16, 0.34), (0.39, 0.11), (0.44, 0.16), (0.21, 0.39)]), ("p", [(0.41, 0.59), (0.64, 0.36), (0.69, 0.41), (0.46, 0.64)])])]),
    "block": (1.0, 0.36, [
        (DBR, 1, [("rr", 0.02, 0.16, 0.98, 0.34, 0.04)]),
        (BRN, 1, [("rr", 0.12, 0.02, 0.88, 0.2, 0.04)])]),
    "ballot_box": (1.0, 1.0, [
        (BLU, 1, [("r", 0.08, 0.36, 0.92, 0.98)]),
        ((52, 74, 120), 1, [("r", 0.02, 0.24, 0.98, 0.4)]),
        (DRK, NS, [("r", 0.28, 0.295, 0.72, 0.345)]),
        (WHT, NS, [("l", [(0.34, 0.68), (0.46, 0.8), (0.68, 0.55)], 0.07)])]),
    "ballot": (0.6, 0.8, [
        (WHT, 1, [("r", 0.02, 0.02, 0.58, 0.78)]),
        (LGR, NS, [("l", [(0.12, 0.22), (0.48, 0.22)], 0.03), ("l", [(0.12, 0.36), (0.48, 0.36)], 0.03)]),
        (RD, NS, [("l", [(0.2, 0.5), (0.4, 0.68)], 0.05), ("l", [(0.4, 0.5), (0.2, 0.68)], 0.05)])]),
    "bank": (1.2, 1.0, [
        (LGR, 1, [("r", 0.02, 0.86, 1.18, 0.98)]),
        (WHT, 1, [("r", 0.12, 0.44, 0.24, 0.87), ("r", 0.39, 0.44, 0.51, 0.87), ("r", 0.69, 0.44, 0.81, 0.87), ("r", 0.96, 0.44, 1.08, 0.87)]),
        (LGR, 1, [("r", 0.06, 0.34, 1.14, 0.45)]),
        (WHT, 1, [("p", [(0.02, 0.35), (0.6, 0.02), (1.18, 0.35)])]),
        (GLD, NS, [("e", 0.52, 0.14, 0.68, 0.3)])]),
    "factory": (1.4, 1.0, [
        ((176, 84, 60), 1, [("r", 0.14, 0.08, 0.3, 0.6), ("r", 0.4, 0.2, 0.54, 0.6)]),
        (GRY, 1, [("p", [(0.05, 0.98), (0.05, 0.56), (0.6, 0.56), (0.6, 0.4), (0.85, 0.56), (0.85, 0.4), (1.1, 0.56), (1.1, 0.4), (1.35, 0.56), (1.35, 0.98)])]),
        (YEL, NS, [("r", 0.15, 0.7, 0.3, 0.83), ("r", 0.45, 0.7, 0.6, 0.83), ("r", 0.75, 0.7, 0.9, 0.83), ("r", 1.05, 0.7, 1.2, 0.83)])]),
    "oil_rig": (1.0, 1.4, [
        (GRY, 1, [("r", 0.18, 1.1, 0.26, 1.38), ("r", 0.74, 1.1, 0.82, 1.38)]),
        (DRK, 1, [("l", [(0.3, 1.0), (0.5, 0.05)], 0.045), ("l", [(0.7, 1.0), (0.5, 0.05)], 0.045), ("l", [(0.36, 0.72), (0.64, 0.72)], 0.035),
                  ("l", [(0.42, 0.42), (0.58, 0.42)], 0.035), ("l", [(0.36, 0.72), (0.58, 0.42)], 0.03), ("l", [(0.64, 0.72), (0.42, 0.42)], 0.03),
                  ("l", [(0.3, 1.0), (0.64, 0.72)], 0.03), ("l", [(0.7, 1.0), (0.36, 0.72)], 0.03)]),
        (ORG, 1, [("r", 0.08, 0.98, 0.92, 1.12)])]),
    "barrel": (0.8, 1.0, [
        ((52, 74, 120), 1, [("rr", 0.06, 0.04, 0.74, 0.96, 0.08)]),
        (BLU, NS, [("r", 0.06, 0.16, 0.74, 0.34), ("r", 0.06, 0.66, 0.74, 0.84)]),
        (YEL, NS, [("p", [(0.4, 0.38), (0.5, 0.52), (0.4, 0.62), (0.3, 0.52)])])]),
    "pylon": (1.0, 1.5, [
        (DRK, 1, [("l", [(0.5, 0.04), (0.26, 1.46)], 0.04), ("l", [(0.5, 0.04), (0.74, 1.46)], 0.04), ("l", [(0.06, 0.32), (0.94, 0.32)], 0.04),
                  ("l", [(0.16, 0.58), (0.84, 0.58)], 0.04), ("l", [(0.4, 0.58), (0.67, 1.0)], 0.03), ("l", [(0.6, 0.58), (0.33, 1.0)], 0.03),
                  ("l", [(0.33, 1.0), (0.67, 1.0)], 0.03), ("l", [(0.33, 1.0), (0.74, 1.46)], 0.03), ("l", [(0.67, 1.0), (0.26, 1.46)], 0.03),
                  ("l", [(0.06, 0.32), (0.06, 0.4)], 0.03), ("l", [(0.94, 0.32), (0.94, 0.4)], 0.03), ("l", [(0.16, 0.58), (0.16, 0.66)], 0.03), ("l", [(0.84, 0.58), (0.84, 0.66)], 0.03)])]),
    "bolt": (0.7, 1.0, [
        (YEL, 1, [("p", [(0.42, 0.02), (0.08, 0.56), (0.32, 0.56), (0.22, 0.98), (0.64, 0.4), (0.4, 0.4), (0.56, 0.02)])])]),
    "turbine_mast": (1.0, 1.5, [
        (WHT, 1, [("p", [(0.475, 0.5), (0.525, 0.5), (0.56, 1.48), (0.44, 1.48)])]),
        (LGR, 1, [("e", 0.44, 0.44, 0.56, 0.56)])]),
    "turbine_blades": (1.0, 1.0, [
        (WHT, 1, [_blade(-90), _blade(30), _blade(150)]),
        (GRY, NS, [("e", 0.455, 0.455, 0.545, 0.545)])]),
    "storm": (1.0, 1.0, [
        (WHT, 1, [("a", 0.06, 0.06, 0.94, 0.94, 195, 300, 0.17), ("a", 0.06, 0.06, 0.94, 0.94, 15, 120, 0.17)]),
        (WHT, 1, [("e", 0.24, 0.24, 0.76, 0.76)]),
        (LBL, NS, [("a", 0.3, 0.3, 0.7, 0.7, 200, 330, 0.05), ("a", 0.3, 0.3, 0.7, 0.7, 20, 150, 0.05)]),
        (BLU, NS, [("e", 0.43, 0.43, 0.57, 0.57)])]),
    "wave": (1.6, 0.7, [
        (BLU, 1, [_wave(0.2, 0.1, 0.0)]),
        ((78, 146, 200), 1, [_wave(0.36, 0.08, 1.6)]),
        (LBL, 1, [_wave(0.52, 0.06, 3.0)])]),
    "tree": (0.8, 1.2, [
        (BRN, 1, [("r", 0.34, 0.72, 0.46, 1.18)]),
        (DGN, 1, [("e", 0.06, 0.3, 0.74, 0.86)]),
        (GRN, 1, [("e", 0.16, 0.04, 0.64, 0.56)])]),
    "cargo_ship": (1.8, 0.9, [
        (WHT, 1, [("r", 1.3, 0.14, 1.56, 0.56)]),
        (LBL, NS, [("r", 1.36, 0.2, 1.5, 0.28)]),
        (RD, 1, [("r", 0.3, 0.34, 0.55, 0.55)]), (BLU, 1, [("r", 0.57, 0.34, 0.82, 0.55)]), (YEL, 1, [("r", 0.84, 0.34, 1.09, 0.55)]),
        (GRN, 1, [("r", 0.44, 0.13, 0.69, 0.34)]), (ORG, 1, [("r", 0.71, 0.13, 0.96, 0.34)]),
        (DRK, 1, [("p", [(0.02, 0.55), (1.78, 0.55), (1.6, 0.88), (0.18, 0.88)])]),
        ((176, 60, 44), NS, [("p", [(0.1, 0.74), (1.68, 0.74), (1.6, 0.88), (0.18, 0.88)])])]),
    "truck": (1.6, 0.9, [
        (WHT, 1, [("r", 0.02, 0.1, 1.05, 0.68)]),
        (RD, 1, [("p", [(1.08, 0.3), (1.38, 0.3), (1.56, 0.5), (1.56, 0.68), (1.08, 0.68)])]),
        (LBL, NS, [("p", [(1.14, 0.35), (1.35, 0.35), (1.46, 0.49), (1.14, 0.49)])]),
        (LGR, NS, [("l", [(0.36, 0.1), (0.36, 0.68)], 0.015), ("l", [(0.7, 0.1), (0.7, 0.68)], 0.015)]),
        (DRK, 1, [("e", 0.15, 0.6, 0.39, 0.84), ("e", 0.7, 0.6, 0.94, 0.84), ("e", 1.2, 0.6, 1.44, 0.84)]),
        (LGR, NS, [("e", 0.23, 0.68, 0.31, 0.76), ("e", 0.78, 0.68, 0.86, 0.76), ("e", 1.28, 0.68, 1.36, 0.76)])]),
    "train": (1.9, 0.8, [
        (DRK, 1, [("e", 0.2, 0.56, 0.38, 0.74), ("e", 0.44, 0.56, 0.62, 0.74), ("e", 1.2, 0.56, 1.38, 0.74), ("e", 1.44, 0.56, 1.62, 0.74)]),
        (BLU, 1, [("p", [(0.04, 0.14), (1.62, 0.14), (1.86, 0.44), (1.86, 0.62), (0.04, 0.62)])]),
        (YEL, NS, [("r", 0.04, 0.48, 1.86, 0.54)]),
        (WHT, NS, [("r", 0.14, 0.22, 0.36, 0.4), ("r", 0.44, 0.22, 0.66, 0.4), ("r", 0.74, 0.22, 0.96, 0.4), ("r", 1.04, 0.22, 1.26, 0.4),
                   ("p", [(1.36, 0.22), (1.58, 0.22), (1.72, 0.4), (1.36, 0.4)])])]),
    "helicopter": (1.7, 0.95, [
        (DRK, 1, [("l", [(0.52, 0.86), (1.2, 0.86)], 0.035), ("l", [(0.66, 0.7), (0.62, 0.86)], 0.03), ("l", [(1.0, 0.7), (1.04, 0.86)], 0.03),
                  ("r", 0.78, 0.1, 0.83, 0.26)]),
        (RD, 1, [("p", [(0.42, 0.36), (0.42, 0.5), (0.04, 0.44), (0.04, 0.38)])]),
        (RD, 1, [("p", [(0.46, 0.26), (1.04, 0.26), (1.26, 0.46), (1.26, 0.6), (1.06, 0.72), (0.52, 0.72), (0.38, 0.5)])]),
        (LBL, NS, [("p", [(0.92, 0.31), (1.03, 0.31), (1.2, 0.47), (0.92, 0.47)])]),
        (DRK, 1, [("l", [(0.22, 0.1), (1.4, 0.1)], 0.04), ("e", 0.0, 0.3, 0.14, 0.5)])]),
    "tank": (1.6, 0.9, [
        (DRK, 1, [("l", [(0.98, 0.29), (1.58, 0.2)], 0.06)]),
        ((92, 118, 78), 1, [("rr", 0.5, 0.17, 1.02, 0.42, 0.05)]),
        (DGN, 1, [("p", [(0.1, 0.56), (0.24, 0.38), (1.26, 0.38), (1.42, 0.56)])]),
        (DRK, 1, [("rr", 0.04, 0.54, 1.48, 0.86, 0.15)]),
        (GRY, NS, [("e", 0.14, 0.6, 0.34, 0.8), ("e", 0.42, 0.6, 0.62, 0.8), ("e", 0.7, 0.6, 0.9, 0.8), ("e", 0.98, 0.6, 1.18, 0.8), ("e", 1.22, 0.62, 1.4, 0.8)])]),
    "hospital": (1.1, 1.1, [
        (WHT, 1, [("r", 0.08, 0.18, 1.02, 1.08)]),
        (LGR, 1, [("r", 0.04, 0.12, 1.06, 0.22)]),
        (RD, NS, [("r", 0.47, 0.3, 0.63, 0.66), ("r", 0.37, 0.4, 0.73, 0.56)]),
        (BLU, NS, [("r", 0.45, 0.8, 0.65, 1.08)]),
        (LBL, NS, [("r", 0.16, 0.74, 0.32, 0.9), ("r", 0.78, 0.74, 0.94, 0.9), ("r", 0.16, 0.36, 0.3, 0.5), ("r", 0.8, 0.36, 0.94, 0.5)])]),
    "pill_bottle": (0.8, 1.2, [
        (ORG, 1, [("rr", 0.08, 0.22, 0.72, 1.18, 0.07)]),
        (WHT, 1, [("rr", 0.16, 0.02, 0.64, 0.25, 0.04)]),
        (WHT, NS, [("r", 0.08, 0.5, 0.72, 0.92)]),
        (RD, NS, [("r", 0.36, 0.58, 0.44, 0.84), ("r", 0.27, 0.67, 0.53, 0.75)])]),
    "syringe": (1.4, 0.5, [
        (GRY, 1, [("l", [(1.0, 0.25), (1.38, 0.25)], 0.025), ("r", 0.06, 0.22, 0.32, 0.28), ("r", 0.02, 0.08, 0.08, 0.42)]),
        (WHT, 1, [("r", 0.3, 0.12, 1.02, 0.38)]),
        (LBL, NS, [("r", 0.58, 0.15, 0.99, 0.35)]),
        (GRY, NS, [("r", 0.27, 0.04, 0.33, 0.46), ("l", [(0.7, 0.12), (0.7, 0.2)], 0.015), ("l", [(0.84, 0.12), (0.84, 0.2)], 0.015)])]),
    "phone": (0.65, 1.2, [
        (DRK, 1, [("rr", 0.02, 0.02, 0.63, 1.18, 0.09)]),
        (LBL, NS, [("r", 0.08, 0.14, 0.57, 1.02)]),
        (WHT, NS, [("r", 0.14, 0.24, 0.51, 0.32), ("r", 0.14, 0.4, 0.4, 0.46), ("r", 0.14, 0.54, 0.46, 0.6)]),
        (GRY, NS, [("e", 0.28, 1.06, 0.37, 1.15)])]),
    "laptop": (1.5, 1.0, [
        (DRK, 1, [("rr", 0.2, 0.05, 1.3, 0.78, 0.05)]),
        (LBL, NS, [("r", 0.27, 0.12, 1.23, 0.68)]),
        (WHT, NS, [("r", 0.36, 0.22, 0.86, 0.29), ("r", 0.36, 0.37, 1.1, 0.43), ("r", 0.36, 0.51, 0.72, 0.57)]),
        (LGR, 1, [("p", [(0.04, 0.94), (0.2, 0.76), (1.3, 0.76), (1.46, 0.94)])])]),
    "robot": (1.0, 1.2, [
        (GRY, 1, [("l", [(0.5, 0.05), (0.5, 0.2)], 0.035)]),
        (RD, 1, [("e", 0.44, 0.0, 0.56, 0.12)]),
        (BLU, 1, [("rr", 0.14, 0.68, 0.86, 1.18, 0.07)]),
        (LGR, 1, [("rr", 0.2, 0.18, 0.8, 0.63, 0.07)]),
        (YEL, NS, [("e", 0.3, 0.29, 0.44, 0.43), ("e", 0.56, 0.29, 0.7, 0.43)]),
        (DRK, NS, [("r", 0.36, 0.5, 0.64, 0.545)]),
        (LBL, NS, [("r", 0.36, 0.8, 0.64, 1.0)])]),
    "house": (1.1, 1.0, [
        (WHT, 1, [("r", 0.16, 0.44, 0.94, 0.98)]),
        ((176, 84, 60), 1, [("p", [(0.02, 0.5), (0.55, 0.03), (1.08, 0.5)])]),
        (BRN, NS, [("r", 0.46, 0.66, 0.64, 0.98)]),
        (LBL, NS, [("r", 0.24, 0.58, 0.4, 0.74), ("r", 0.7, 0.58, 0.86, 0.74)])]),
}
NEW_SPRITES = sorted(SPEC)

def _build(kind, s):
    aw, ah, layers = SPEC[kind]
    c = PC(int(aw * s), int(ah * s))
    for li, (col, sh, shapes) in enumerate(layers):
        def fn(d, P, R, shapes=shapes):
            for sp in shapes:
                k = sp[0]
                if k == "p": d.polygon([P(x * s, y * s) for x, y in sp[1]])
                elif k == "r": d.rectangle([P(sp[1] * s, sp[2] * s), P(sp[3] * s, sp[4] * s)])
                elif k == "e": d.ellipse([P(sp[1] * s, sp[2] * s), P(sp[3] * s, sp[4] * s)])
                elif k == "rr": d.rounded_rectangle([P(sp[1] * s, sp[2] * s), P(sp[3] * s, sp[4] * s)], radius=int(R(sp[5] * s)))
                elif k == "l": d.line([P(x * s, y * s) for x, y in sp[1]], fill=255, width=max(2, int(R(sp[2] * s))), joint="curve")
                elif k == "a": d.arc([P(sp[1] * s, sp[2] * s), P(sp[3] * s, sp[4] * s)], sp[5], sp[6], fill=255, width=max(2, int(R(sp[7] * s))))
        c.layer(fn, col, sh=(2, 3, 3, 0.38) if sh else None, rough=0.9, seed=li + 1)
    return c.sprite()

_v4_SP = V4.SP
def SP(kind, size=80, **kw):
    if kind in SPEC:
        key = ("v6", kind, size)
        if key not in V4._SP: V4._SP[key] = _build(kind, size)
        return V4._SP[key]
    return _v4_SP(kind, size, **kw)

# ------------------------------------------------------------------ actions
ACTIONS = ("launch", "strike", "spin", "turbine")

def _put_pivot(img, sp, px, py, ox, oy, rot, alpha=1.0):
    """Draw sp turned by rot (degrees, counter-clockwise) around a pivot; (ox, oy) = sprite centre minus pivot at rot 0."""
    r = math.radians(rot); c, s_ = math.cos(r), math.sin(r)
    put(img, sp, px + ox * c + oy * s_, py - ox * s_ + oy * c, 1.0, rot, alpha)

def _draw_action(img, sc, p, pi, t):
    ml = sc.get("_map")
    def XY(k="pos"):
        v = p.get(k)
        if v is None: return None
        if ml is not None and sc["type"] in ("map", "intro") and not p.get("screen"): return ml.to_screen(v[0], v[1])
        return (v[0], v[1])
    ln = sc["lines"][min(p.get("at_line", 0), len(sc["lines"]) - 1)] if sc["lines"] else {"start": 0}
    t0 = ln["start"] + p.get("delay", 0)
    if sc["type"] == "map": t0 = max(t0, 1.75)
    tr = t - t0
    if tr < 0: return
    a = 1.0; drop = 0.0
    if "exit_line" in p:
        el = sc["lines"][min(p["exit_line"], len(sc["lines"]) - 1)]
        ex = t - (el["start"] + p.get("exit_delay", 0))
        if ex > 0:
            e = ease(ex / .6); a = 1 - e; drop = 700 * e * e
            if a <= 0.01: return
    k = p["kind"]; x, y = XY(); y += drop
    tq = q12(tr) + pi
    if k == "launch":
        s = p.get("size", 150); hold = p.get("hold", 0.7); dur = p.get("dur", 2.6); rise = p.get("rise", 620)
        pop = spring(tr / .5)
        u = min(max((tr - hold) / dur, 0.0), 1.0); dy = rise * u * u
        shake = (3.5 * math.sin(tq * 31) if tr < hold + dur * .25 else 0) * (1 if tr > 0.25 else 0)
        base_y = y + s * 0.62
        puff = SP("puff", int(s * .55))
        n = 7
        for j in range(n):   # ground smoke, spreading sideways
            tj = tr - 0.15 - j * 0.09
            if tj < 0: continue
            side = (-1 if j % 2 else 1) * (0.25 + 0.22 * (j // 2)) * s * min(tj / 1.2, 1.0)
            put(img, puff, x + side, base_y + 6 * math.sin(tq * 2 + j), min(0.5 + tj * .6, 1.25), 12 * math.sin(tq + j), a * max(0.0, 0.85 - 0.16 * max(tj - 2.2, 0)), lift=.5)
        if u > 0:            # smoke trail under the rocket
            for j in range(5):
                yy = base_y - dy * (j + 1) / 6
                if yy > base_y: continue
                put(img, puff, x + 7 * math.sin(tq * 1.7 + j * 2), yy + s * .25, 0.75 - 0.08 * j, 10 * math.sin(tq + j), a * (0.55 - 0.07 * j), lift=.4)
        if tr > 0.3:
            fl = SP("flame", int(s * .42)); fs = (0.75 if u == 0 else 1.15) * (1 + 0.1 * math.sin(tq * 23))
            put(img, fl, x + shake, y - dy + s * 0.8, fs, 180 + 4 * math.sin(tq * 9), a, lift=.4)
        put(img, SP("rocket", s), x + shake, y - dy, min(pop, 1.0) if tr < .5 else 1.0, 0, a)
    elif k == "strike":
        s = p.get("size", 190); hits = max(1, int(p.get("hits", 2))); per = 0.75
        put(img, SP("block", int(s * .8)), x, y, spring(tr / .5), 0, a)
        i = min(int(tr / per), hits - 1); ph = (tr - i * per) / per
        up, down = -12.0, 40.0
        if tr >= hits * per: rot = down - 6 * (1 - math.exp(-(tr - hits * per) * 4)); hit_t = tr - ((hits - 1) * per + per * .45)
        elif ph < 0.25: rot = down + (up - down) * ease(ph / .25) if i > 0 else up; hit_t = -1
        elif ph < 0.45: rot = up + (down - up) * ((ph - .25) / .2) ** 2; hit_t = -1
        else: rot = down - 5 * math.exp(-(ph - .45) * 9) * abs(math.sin((ph - .45) * 24)); hit_t = (ph - .45) * per
        px, py = x + 0.86 * s, y - 0.12 * s
        _put_pivot(img, SP("gavel", s), px, py, -0.46 * s, -0.44 * s, rot, a * min(tr / .2, 1.0))
        if 0 <= hit_t < 0.22:
            d = ImageDraw.Draw(img); r0 = s * (.32 + hit_t * 1.3); r1 = r0 + s * .16
            for ang in (200, 235, 270, 305, 340):
                ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                d.line([(x + ca * r0, y - s * .1 + sa * r0), (x + ca * r1, y - s * .1 + sa * r1)], fill=YEL, width=7)
    elif k == "spin":
        s = p.get("size", 160); sp = SP(p["sprite"], s, **p.get("args", {}))
        to = XY("to")
        if to is not None:
            u = ease(min(tr / p.get("dur", 3.0), 1.0)); x = x + (to[0] - x) * u; y = y + (to[1] + drop - y) * u
        put(img, sp, x, y, spring(tr / .5), -(tr * p.get("speed", 120)) % 360, a)
    elif k == "turbine":
        s = p.get("size", 150)
        put(img, SP("turbine_mast", s), x, y + s * .25, spring(tr / .6), 0, a)
        put(img, SP("turbine_blades", s), x, y, spring(tr / .6), -(q12(tr) * p.get("speed", 140)) % 360, a, lift=.6)

_v5_draw_props = V4.draw_props          # v5's version (it is installed on the v4 module)
def draw_props(img, sc, t):
    props = sc.get("props") or []
    if not props: return
    acts = [(i, p) for i, p in enumerate(props) if p.get("kind") in ACTIONS]
    if len(acts) < len(props):
        keep = dict(sc); keep["props"] = [p for p in props if p.get("kind") not in ACTIONS]
        _v5_draw_props(img, keep, t)
    for i, p in acts: _draw_action(img, sc, p, i, t)

# ------------------------------------------------------------------ wire up
V4.SP = SP
V4.draw_props = draw_props
V5.draw_props = draw_props      # the hook card in v5 calls its own draw_props

if __name__ == "__main__":
    B.main()
