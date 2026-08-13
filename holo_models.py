#!/usr/bin/env python3
"""holo_models.py - the hard-light 3D models: web-shooter, gauntlet, suit.

Model spaces
------------
web-shooter   +x wrist -> fingers (aim), +y out of the back of the wrist,
              +z across toward the thumb.  Wrist radius ~0.78.
gauntlet      +x wrist -> elbow, +y back of the hand, +z toward the thumb.
suit          +x body right, +y down, +z out of the chest.  Chest half-height 1.

Each model is authored once and cached; the renderer only re-projects it.
"""

from __future__ import annotations

import math

import numpy as np

from holo3d import Mesh

# -- web-shooter part ids (explode directions live with the geometry) -------- #
(WS_BAND, WS_HOUSING, WS_CART, WS_BARREL, WS_NOZZLE, WS_TRIGGER, WS_POD,
 WS_WRAP) = range(8)

_CACHE: dict = {}


def _cached(name: str, builder):
    mesh = _CACHE.get(name)
    if mesh is None:
        mesh = builder()
        _CACHE[name] = mesh
    return mesh


# --------------------------------------------------------------------------- #
# The Mk-V web-shooter
# --------------------------------------------------------------------------- #
def _build_shooter(detail: int = 1) -> Mesh:
    """The Mk-V bracer: a forearm shell running from mid-forearm to the wrist.

    Origin sits AT THE WRIST, so the whole rig extends back up the arm along
    -x and only the barrel and spinneret cross forward past the wrist crease -
    the way the real thing is worn.  The cross-section is wider across the arm
    (z) than it is deep (y), because a forearm is.

    detail 1 = the blueprint hero model, 0 = the worn LOD.
    """
    m = Mesh()
    seg = 12 if detail else 9
    maj = 14 if detail else 10
    mino = 5 if detail else 4
    ribs = 3 if detail else 2
    SQ = 1.35                            # forearm is wider than it is deep

    # 1. wrist collar: where the bracer clamps shut, right on the wrist bone
    m.part(WS_BAND, (0.35, 0.0, 0.0))
    m.torus((-0.06, 0.0, 0.0), (1, 0, 0), 0.50, 0.085, maj, mino, 1.0, squash=SQ)
    if detail:
        m.torus((-0.26, 0.0, 0.0), (1, 0, 0), 0.46, 0.04, maj, 4, 0.75, squash=SQ)
    for sz in (-1, 1):                   # locking clamps on the sides
        m.box((-0.10, 0.10, sz * 0.66), (0.15, 0.07, 0.055), 0.9)

    # 2. the bracer shell: a slim tapered sleeve, flattened onto the forearm so
    #    the hardware bolted to it stays proud of the silhouette
    m.part(WS_HOUSING, (0.0, 1.30, 0.10))
    m.tube((-2.80, 0.0, 0.0), (-0.10, 0.0, 0.0), 0.34, 0.48, seg, 0.95,
           squash=SQ, cap0=False, cap1=False)
    for i, x in enumerate((-2.45, -1.80, -1.15, -0.50)):  # armour bands
        m.torus((x, 0.0, 0.0), (1, 0, 0), 0.36 + 0.032 * i, 0.05, maj, 4, 0.9,
                squash=SQ)
    # raised dorsal deck along the back of the forearm
    m.box((-1.30, 0.50, 0.0), (1.05, 0.10, 0.26), 1.0)
    m.box((-1.30, 0.62, 0.0), (0.80, 0.045, 0.18), 0.95)
    for i in range(4):                                   # heat vents
        m.box((-1.95 + i * 0.42, 0.67, 0.0), (0.055, 0.028, 0.13), 0.7)
    m.disc((-0.40, 0.62, 0.0), (0, 1, 0), 0.10, 10, 1.0)  # status lens
    m.torus((-0.40, 0.63, 0.0), (0, 1, 0), 0.125, 0.024, 10, 4, 0.9)

    # 3. twin web-fluid cartridges, slung either side of the deck
    m.part(WS_CART, (0.0, -1.30, 0.40))
    for sz in (-1, 1):
        m.tube((-2.20, 0.26, sz * 0.44), (-0.40, 0.26, sz * 0.44), 0.105, 0.105,
               seg, 0.95)
        if detail:
            m.torus((-0.52, 0.26, sz * 0.44), (1, 0, 0), 0.125, 0.026, 9, 4, 0.8)
            m.torus((-2.08, 0.26, sz * 0.44), (1, 0, 0), 0.125, 0.026, 9, 4, 0.8)

    # 4. barrel: crosses the wrist toward the hand, with cooling ribs
    m.part(WS_BARREL, (0.60, 0.55, 0.45))
    m.tube((-0.22, 0.40, 0.0), (0.34, 0.44, 0.0), 0.115, 0.085, seg, 1.0)
    for i in range(ribs):
        m.torus((-0.10 + i * 0.17, 0.41 + i * 0.012, 0.0), (1, 0.06, 0), 0.128,
                0.024, 9, 4, 0.85)

    # 5. spinneret: sits just past the wrist crease, aimed down the fingers
    m.part(WS_NOZZLE, (1.35, 0.15, 0.0))
    m.tube((0.34, 0.44, 0.0), (0.58, 0.45, 0.0), 0.085, 0.052, seg, 1.0)
    if detail:
        m.torus((0.54, 0.45, 0.0), (1, 0, 0), 0.072, 0.022, 9, 4, 1.0)
    m.disc((0.59, 0.45, 0.0), (1, 0, 0), 0.045, 8, 1.0)

    # 6. trigger pad, on the palm side under the wrist
    m.part(WS_TRIGGER, (0.0, -1.20, -0.60))
    m.box((-0.02, -0.40, 0.0), (0.20, 0.075, 0.13), 1.0)
    m.box((0.14, -0.47, 0.0), (0.09, 0.045, 0.06), 0.8)

    # 7. side pods: stabiliser plates along the flanks + status nodes
    m.part(WS_POD, (-0.35, 0.10, -1.45))
    for sz in (-1, 1):
        m.box((-1.45, 0.10, sz * 0.58), (0.80, 0.12, 0.045), 0.85)
        if detail:
            m.ellipsoid((-0.68, 0.14, sz * 0.60), (0.07, 0.07, 0.04), 7, 3, 1.0)
    # 8. the hand wrap: straps that carry on over the wrist and close around
    #    the back of the hand, so the rig is worn rather than parked on the arm
    m.part(WS_WRAP, (0.9, 0.30, 0.0))
    for x, r in ((0.42, 0.50), (0.95, 0.46)):            # two closed straps
        m.torus((x, 0.10, 0.0), (1, 0.10, 0), r, 0.06, maj, 4, 0.95, squash=SQ)
    m.box((0.70, 0.40, 0.0), (0.42, 0.06, 0.26), 1.0)    # knuckle plate
    m.box((1.18, 0.34, 0.0), (0.14, 0.05, 0.20), 0.85)
    for sz in (-1, 1):                                   # side rails to the plate
        m.box((0.70, 0.16, sz * 0.44), (0.46, 0.05, 0.05), 0.8)
    if detail:
        m.torus((0.70, 0.44, 0.0), (0, 1, 0), 0.20, 0.03, 10, 4, 0.9)

    # a bracer is long and slim: authored full-width, then narrowed across so
    # it hugs the forearm instead of swallowing it
    return m.slim(0.68).compile()


def shooter_mesh(detail: int = 1) -> Mesh:
    return _cached(f"ws{detail}", lambda: _build_shooter(detail))


# --------------------------------------------------------------------------- #
# The forearm gauntlet (worn on the wrist with the body gear)
# --------------------------------------------------------------------------- #
def _build_gauntlet() -> Mesh:
    m = Mesh()
    m.part(0, (0.0, 0.0, 0.0))
    # tapered forearm cuff, flattened in z so it reads as an armour shell
    m.tube((0.02, 0.0, 0.0), (1.45, 0.0, 0.0), 0.60, 0.86, 12, 0.95,
           squash=0.72, cap0=False, cap1=False)
    # overlapping armour bands
    for i, x in enumerate((0.18, 0.62, 1.06)):
        r = 0.63 + 0.09 * i
        m.torus((x, 0.0, 0.0), (1, 0, 0), r, 0.075, 12, 4, 1.0, squash=0.72)
    # knuckle guard over the back of the hand
    m.part(1, (0.0, 0.0, 0.0))
    m.box((-0.42, 0.40, 0.0), (0.30, 0.09, 0.46), 1.0)
    m.box((-0.72, 0.34, 0.0), (0.16, 0.06, 0.36), 0.85)
    # wrist reactor on the back of the cuff
    m.part(2, (0.0, 0.0, 0.0))
    m.disc((0.30, 0.62, 0.0), (0, 1, 0), 0.24, 12, 1.0)
    m.torus((0.30, 0.63, 0.0), (0, 1, 0), 0.30, 0.05, 12, 4, 1.0)
    # vents + thumb plate
    for i in range(3):
        m.box((0.55 + i * 0.26, 0.52, 0.30), (0.09, 0.05, 0.10), 0.7)
    m.box((0.20, 0.22, 0.62), (0.34, 0.16, 0.07), 0.8)
    return m.compile()


def gauntlet_mesh() -> Mesh:
    return _cached("gauntlet", _build_gauntlet)


# --------------------------------------------------------------------------- #
# The suit: torso shell, shoulders, belt (no mask - never a head)
# --------------------------------------------------------------------------- #
def _build_suit() -> Mesh:
    m = Mesh()
    m.part(0, (0.0, 0.0, 0.0))
    # torso: chest -> waist, flattened front-to-back
    m.tube((0.0, -1.00, 0.0), (0.0, 0.30, 0.0), 1.02, 0.74, 14, 0.95,
           squash=0.44, cap0=False, cap1=False)
    m.tube((0.0, 0.30, 0.0), (0.0, 0.98, 0.0), 0.74, 0.80, 14, 0.9,
           squash=0.44, cap0=False, cap1=False)
    # pectoral plates
    m.part(1, (0.0, 0.0, 0.0))
    for sx in (-1, 1):
        m.box((sx * 0.44, -0.62, 0.34), (0.34, 0.24, 0.07), 1.0)
    m.box((0.0, -0.18, 0.40), (0.30, 0.22, 0.06), 0.9)
    # abdominal segments
    for i in range(3):
        m.torus((0.0, 0.18 + i * 0.24, 0.0), (0, 1, 0), 0.78 - 0.02 * i, 0.045,
                12, 4, 0.8, squash=0.46)
    # shoulders
    m.part(2, (0.0, 0.0, 0.0))
    for sx in (-1, 1):
        m.ellipsoid((sx * 1.06, -0.86, 0.0), (0.36, 0.30, 0.26), 8, 4, 1.0)
        m.torus((sx * 1.06, -0.70, 0.0), (0, 1, 0), 0.34, 0.05, 10, 4, 0.9,
                squash=0.85)
    # belt
    m.part(3, (0.0, 0.0, 0.0))
    m.torus((0.0, 1.02, 0.0), (0, 1, 0), 0.84, 0.09, 14, 4, 1.0, squash=0.46)
    m.box((0.0, 1.02, 0.38), (0.18, 0.11, 0.06), 1.0)
    return m.compile()


def suit_mesh() -> Mesh:
    return _cached("suit", _build_suit)


# --------------------------------------------------------------------------- #
# Chest decals: the spider emblem and the webbing, as 3D point paths
# --------------------------------------------------------------------------- #
def _spider_paths() -> list:
    """Emblem strokes in suit model space, sitting on the chest surface."""
    body = []
    for i in range(14):
        th = 2.0 * math.pi * i / 14
        body.append((0.16 * math.sin(th), -0.44 + 0.30 * math.cos(th), 0.44))
    head = []
    for i in range(10):
        th = 2.0 * math.pi * i / 10
        head.append((0.09 * math.sin(th), -0.78 + 0.10 * math.cos(th), 0.44))
    paths = [("fill", body), ("fill", head)]
    for sx in (-1, 1):                       # eight legs, two joints each
        for j, (y0, ky, kx) in enumerate(((-0.70, -0.34, 1.00), (-0.58, -0.16, 1.16),
                                          (-0.44, 0.10, 1.16), (-0.32, 0.32, 1.00))):
            p0 = (sx * 0.14, y0, 0.44)
            p1 = (sx * (0.52 * kx), y0 + ky * 0.55, 0.42)
            p2 = (sx * (0.86 * kx), y0 + ky * 0.20, 0.34)
            paths.append(("line", [p0, p1, p2]))
    return paths


def _web_paths() -> list:
    """Classic web pattern over the chest: radials + sagging cross-arcs."""
    paths = []
    cx, cy = 0.0, -0.30
    rays = []
    for i in range(12):
        th = math.pi * (i / 11.0) - math.pi / 2.0
        rays.append(th)
    for th in rays:
        pts = []
        for r in (0.25, 0.65, 1.05, 1.45):
            x = cx + math.sin(th) * r * 0.72
            y = cy + math.cos(th) * r * 0.60 * (1.0 if math.cos(th) > 0 else 0.85)
            z = 0.42 - 0.10 * (r / 1.45) ** 2
            pts.append((x, y, z))
        paths.append(("line", pts))
    for r in (0.45, 0.85, 1.25):
        for i in range(len(rays) - 1):
            a, b = rays[i], rays[i + 1]
            mid = (a + b) * 0.5
            sag = r * 1.12
            pts = []
            for th, rr in ((a, r), (mid, sag), (b, r)):
                x = cx + math.sin(th) * rr * 0.72
                y = cy + math.cos(th) * rr * 0.60
                z = 0.42 - 0.10 * (rr / 1.45) ** 2
                pts.append((x, y, z))
            paths.append(("line", pts))
    return paths


def spider_decals() -> list:
    key = "decals"
    if key not in _CACHE:
        out = []
        for kind, pts in _spider_paths():
            out.append((kind, np.asarray(pts, float), 1.0))
        for kind, pts in _web_paths():
            out.append((kind, np.asarray(pts, float), 0.45))
        _CACHE[key] = out
    return _CACHE[key]
