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
 WS_WRAP, WS_REACTOR) = range(9)

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
    """The Mk-V web-shooter: a compact wrist-mounted launcher.

    Origin sits AT THE WRIST.  The housing only climbs a short way up the
    forearm - about one wrist-width - so it reads as a SHOOTER strapped to
    the wrist, not a gauntlet swallowing the whole arm.  The barrel and
    spinneret cross forward past the wrist crease, the way the real prop is
    worn, and the wrap carries on over the back of the hand so it looks worn
    rather than parked.  Cross-section is wider across the arm (z) than deep
    (y).

    detail 1 = the hero model (more segments), 0 = the worn LOD.
    """
    m = Mesh()
    seg = 12 if detail else 9
    maj = 14 if detail else 10
    mino = 5 if detail else 4
    ribs = 2
    SQ = 1.30

    # 1. wrist collar: where the shooter clamps shut, right on the wrist bone
    m.part(WS_BAND, (0.35, 0.0, 0.0))
    m.torus((-0.06, 0.0, 0.0), (1, 0, 0), 0.48, 0.09, maj, mino, 1.0, squash=SQ)
    if detail:
        m.torus((-0.22, 0.0, 0.0), (1, 0, 0), 0.44, 0.045, maj, 4, 0.75, squash=SQ)
    for sz in (-1, 1):                   # locking clamps on the sides
        m.box((-0.10, 0.10, sz * 0.62), (0.15, 0.07, 0.055), 0.9)

    # 2. the housing: a short, chunky launcher block bolted to the forearm -
    #    a shooter, not a sleeve running the length of the arm
    m.part(WS_HOUSING, (0.0, 1.30, 0.10))
    m.tube((-1.30, 0.0, 0.0), (-0.10, 0.0, 0.0), 0.40, 0.52, seg, 0.95,
           squash=SQ, cap0=False, cap1=False)
    for i, x in enumerate((-1.05, -0.65, -0.28)):        # armour bands
        m.torus((x, 0.0, 0.0), (1, 0, 0), 0.42 + 0.035 * i, 0.055, maj, 4, 0.9,
                squash=SQ)
    # raised dorsal deck along the back of the forearm
    m.box((-0.65, 0.52, 0.0), (0.55, 0.11, 0.28), 1.0)
    m.box((-0.65, 0.65, 0.0), (0.40, 0.045, 0.19), 0.95)
    for i in range(3):                                   # heat vents
        m.box((-1.00 + i * 0.32, 0.70, 0.0), (0.06, 0.03, 0.13), 0.7)
    # the arc reactor: a Stark-pattern ring sunk into the dorsal deck, with a
    # raised core lens.  Its white-hot pulse is drawn by the object layer.
    m.part(WS_REACTOR, (0.25, 0.85, 0.0))
    m.torus((-0.40, 0.66, 0.0), (0, 1, 0), 0.185, 0.030, 12, 4, 1.0)
    m.torus((-0.40, 0.67, 0.0), (0, 1, 0), 0.135, 0.020, 10, 4, 0.9)
    m.disc((-0.40, 0.66, 0.0), (0, 1, 0), 0.115, 10, 1.0)
    m.ellipsoid((-0.40, 0.69, 0.0), (0.075, 0.045, 0.075), 8, 4, 1.15)

    # 3. twin web-fluid cartridges, slung either side of the housing
    m.part(WS_CART, (0.0, -1.30, 0.40))
    for sz in (-1, 1):
        m.tube((-1.10, 0.28, sz * 0.42), (-0.32, 0.28, sz * 0.42), 0.10, 0.10,
               seg, 0.95)
        if detail:
            m.torus((-0.42, 0.28, sz * 0.42), (1, 0, 0), 0.12, 0.026, 9, 4, 0.8)
            m.torus((-1.00, 0.28, sz * 0.42), (1, 0, 0), 0.12, 0.026, 9, 4, 0.8)

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
        m.torus((0.50, 0.45, 0.0), (1, 0, 0), 0.030, 0.016, 8, 4, 1.0)
        m.disc((0.62, 0.455, 0.0), (1, 0, 0), 0.030, 8, 1.0)
    else:
        m.disc((0.59, 0.45, 0.0), (1, 0, 0), 0.045, 8, 1.0)

    # 6. trigger pad, on the palm side under the wrist
    m.part(WS_TRIGGER, (0.0, -1.20, -0.60))
    m.box((-0.02, -0.40, 0.0), (0.20, 0.075, 0.13), 1.0)
    m.box((0.14, -0.47, 0.0), (0.09, 0.045, 0.06), 0.8)

    # 7. side pods: stabiliser plates along the flanks + status nodes
    m.part(WS_POD, (-0.35, 0.10, -1.45))
    for sz in (-1, 1):
        m.box((-0.72, 0.10, sz * 0.56), (0.42, 0.12, 0.045), 0.85)
        if detail:
            m.ellipsoid((-0.72, 0.14, sz * 0.58), (0.07, 0.07, 0.04), 7, 3, 1.0)
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

    # authored full-width, then narrowed across so it hugs the wrist instead
    # of swallowing it
    return m.slim(0.80).compile()


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
# The chest arc reactor: the Stark signature, worn over the sternum
# --------------------------------------------------------------------------- #
def _build_reactor() -> Mesh:
    m = Mesh()
    m.part(0, (0.0, 0.0, 0.0))
    # outer glow ring, inner focusing ring, then the white-hot core
    m.torus((0.0, 0.0, 0.0), (0, 0, 1), 1.00, 0.16, 16, 5, 1.0)
    m.torus((0.0, 0.0, 0.0), (0, 0, 1), 0.62, 0.11, 14, 5, 0.95)
    m.disc((0.0, 0.0, 0.0), (0, 0, 1), 0.40, 12, 1.0)
    m.ellipsoid((0.0, 0.0, 0.16), (0.30, 0.30, 0.14), 8, 4, 1.2)
    for i in range(3):                        # conductor pips on the ring
        th = math.pi * 2.0 * i / 3.0
        m.ellipsoid((math.sin(th) * 0.80, math.cos(th) * 0.80, 0.0),
                    (0.10, 0.10, 0.07), 6, 3, 1.0)
    return m.compile()


def reactor_mesh() -> Mesh:
    return _cached("reactor", _build_reactor)


# --------------------------------------------------------------------------- #
# The palm-screen: the thin holographic tablet a palm projection floats on
# --------------------------------------------------------------------------- #
def _build_palm_screen() -> Mesh:
    m = Mesh()
    m.part(0, (0.0, 0.0, 0.0))
    # a slim slab facing the viewer, slightly tilted, with a bezel ring
    m.box((0.035, 0.0, 0.0), (0.045, 0.42, 0.30), 0.95)
    m.torus((0.06, 0.0, 0.0), (1, 0, 0), 0.255, 0.030, 12, 4, 1.0)
    m.torus((0.07, 0.0, 0.0), (1, 0, 0), 0.255, 0.014, 12, 4, 0.9)
    for i in range(4):                        # corner data nodes
        th = math.pi * 2.0 * i / 4.0
        m.ellipsoid((0.09 + math.sin(th) * 0.19, math.cos(th) * 0.19, 0.0),
                    (0.035, 0.035, 0.02), 6, 3, 1.0)
    return m.compile()


def palm_screen_mesh() -> Mesh:
    return _cached("palm_screen", _build_palm_screen)


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
