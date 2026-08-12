#!/usr/bin/env python3
"""holo_objects.py - the hard-light objects: web-shooters, blueprint, suit.

Every object draws into ONE shared half-resolution overlay per frame (the
upscale doubles as the bloom), and may add crisp full-resolution annotations
afterwards.  All objects that can be picked up implement the drag protocol:

    drag_point()  -> (x, y) px anchor, or None when not grabbable
    drag_radius() -> px capture radius
    begin_drag(px), drag_to(px), end_drag()
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import holo_models as MODELS
from holo3d import (HOLO_BLUE, HOLO_CYAN, HOLO_DEEP, HOLO_DIM, HOLO_WHITE,
                    FLIP_X, clamp01, dim, ease_out_back, project, render_glow,
                    render_mesh, rot_matrix, smoothstep)


def hand_frame(feat, w: int, h: int):
    """Screen-space frame of one hand: wrist, aim axis, thumb side, size."""
    if feat is None:
        return None
    wr = feat["wrist"] * np.array([w, h])
    mc = feat["mcp9"] * np.array([w, h])
    th = feat["thumb_tip"] * np.array([w, h])
    length = float(np.linalg.norm(mc - wr))
    if length < 10.0:
        return None
    ax = ((mc[0] - wr[0]) / length, (mc[1] - wr[1]) / length)
    side = (-ax[1], ax[0])
    if side[0] * (th[0] - wr[0]) + side[1] * (th[1] - wr[1]) < 0.0:
        side = (-side[0], -side[1])
    return {"W": wr, "M": mc, "L": length, "ax": ax, "side": side,
            "theta": math.atan2(ax[1], ax[0])}


def _mount_rot(fr, wobble: float = 0.0):
    """Rotation putting the model's +x along the arm and +y on the thumb side."""
    th = fr["theta"]
    rot = rot_matrix(0.55 + wobble, -0.34, th)
    # does model +y currently point toward the thumb?  if not, mirror it
    up = (math.sin(th), math.cos(th))
    if up[0] * fr["side"][0] + up[1] * fr["side"][1] < 0.0:
        rot = rot @ FLIP_X
    return rot


def _ring(dst, c, r, colour, thick=1, squash=0.42, ang=0.0, a0=0, a1=360):
    cv2.ellipse(dst, (int(c[0]), int(c[1])), (int(max(1, r)), int(max(1, r * squash))),
                math.degrees(ang), a0, a1, colour, thick, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Web-shooter
# --------------------------------------------------------------------------- #
class WebShooter:
    """One wrist's web-shooter, summoned by a pinch-and-pull of the OTHER hand.

    States
    ------  off       not summoned - nothing is drawn
            arrive    flying from the summoning pinch to this wrist
            seek      the wrist is not visible: parked, waiting to snap on
            on        mounted on the wrist, tracking the hand
            held      grabbed (4 s lock) and following a pinch
            float     parked in mid-air where it was let go
            dismiss   spinning down and dissolving
    """

    ARRIVE = 0.70
    DISMISS = 0.45

    def __init__(self, side: str):
        self.side = side
        self.state = "off"
        self.t0 = 0.0
        self.pos = None
        self.scale = 34.0
        self.rot = rot_matrix(0.5, -0.3, 0.0)
        self.spin = 0.0
        self._from = None
        self._park = None
        self._flash = -9.0
        self._lock = -9.0
        self._drag_off = (0.0, 0.0)
        self._fr = None

    # -- lifecycle ---------------------------------------------------------- #
    def active(self) -> bool:
        return self.state != "off"

    def summon(self, t: float, from_px, park_px) -> None:
        self.state = "arrive"
        self.t0 = t
        self._from = (float(from_px[0]), float(from_px[1]))
        self._park = (float(park_px[0]), float(park_px[1]))
        self.pos = self._from
        self._flash = t

    def dismiss(self, t: float) -> None:
        if self.state in ("off", "dismiss"):
            return
        self.state = "dismiss"
        self.t0 = t

    def toggle(self, t: float, from_px, park_px) -> str:
        if self.active():
            self.dismiss(t)
            return "dismissed"
        self.summon(t, from_px, park_px)
        return "summoned"

    # -- drag protocol ------------------------------------------------------ #
    def drag_point(self):
        return self.pos if self.state in ("on", "float", "held") else None

    def drag_radius(self) -> float:
        return max(46.0, self.scale * 1.7)

    def begin_drag(self, px, t: float) -> None:
        self.state = "held"
        self._lock = t
        if self.pos is not None:
            self._drag_off = (self.pos[0] - px[0], self.pos[1] - px[1])
        else:
            self._drag_off = (0.0, 0.0)

    def drag_to(self, px) -> None:
        if self.state == "held":
            self.pos = (px[0] + self._drag_off[0], px[1] + self._drag_off[1])

    def end_drag(self, t: float) -> None:
        if self.state == "held":
            self.state = "float"
            self.t0 = t
            self._park = self.pos

    # -- per-frame ---------------------------------------------------------- #
    def update(self, dt: float, t: float, feat, w: int, h: int) -> None:
        fr = hand_frame(feat, w, h)
        self._fr = fr
        self.spin += dt
        if fr is not None:
            # band just behind the wrist, spinneret just past the knuckles
            anchor = (fr["W"][0] + fr["ax"][0] * 0.26 * fr["L"],
                      fr["W"][1] + fr["ax"][1] * 0.26 * fr["L"])
            live_scale = 0.56 * fr["L"]
        else:
            anchor, live_scale = None, None

        if self.state == "off":
            self.pos = None
        elif self.state == "arrive":
            f = clamp01((t - self.t0) / self.ARRIVE)
            e = smoothstep(f)
            tgt = anchor if anchor is not None else self._park
            self.pos = (self._from[0] + (tgt[0] - self._from[0]) * e,
                        self._from[1] + (tgt[1] - self._from[1]) * e)
            base = live_scale if live_scale else 42.0
            self.scale = base * max(0.18, ease_out_back(max(0.05, f)))
            self.rot = _mount_rot(fr, wobble=(1.0 - f) * 5.0) if fr is not None \
                else rot_matrix(0.5 + self.spin * 3.0 * (1.0 - f), -0.3, 0.0)
            if f >= 1.0:
                self.state = "on" if fr is not None else "seek"
                self._flash = t
                self._park = self.pos
        elif self.state == "seek":
            self.pos = (self._park[0], self._park[1] + 3.0 * math.sin(t * 1.7))
            self.rot = rot_matrix(0.45 + self.spin * 0.55, -0.28, 0.15 * math.sin(t * 0.7))
            if fr is not None:                    # the wrist showed up: snap on
                self.state = "arrive"
                self.t0 = t
                self._from = self.pos
        elif self.state == "on":
            if fr is None:
                self.pos = None                   # hand gone: hide, stay mounted
            else:
                self.pos = anchor
                self.scale = live_scale
                self.rot = _mount_rot(fr, wobble=0.05 * math.sin(t * 1.6))
        elif self.state == "held":
            self.rot = rot_matrix(0.45 + self.spin * 0.5, -0.30,
                                  0.20 * math.sin(t * 1.1))
            self.scale = max(self.scale, 44.0)
        elif self.state == "float":
            if self._park is not None:
                self.pos = (self._park[0], self._park[1] + 3.5 * math.sin(t * 1.6))
            self.rot = rot_matrix(0.45 + self.spin * 0.42, -0.28,
                                  0.16 * math.sin(t * 0.8))
        elif self.state == "dismiss":
            f = clamp01((t - self.t0) / self.DISMISS)
            self.scale *= (1.0 - 0.06 * (dt * 60.0))
            self.rot = rot_matrix(0.45 + self.spin * (1.0 + 6.0 * f), -0.3, 0.0)
            if f >= 1.0:
                self.state = "off"
                self.pos = None

    # -- render ------------------------------------------------------------- #
    def _shading(self, t: float):
        """(alpha, wire, hot) for the current state."""
        alpha, wire = 1.0, 0.0
        if self.state == "arrive":
            f = clamp01((t - self.t0) / self.ARRIVE)
            wire = 1.0 - smoothstep(f)          # wireframe -> solid as it lands
            alpha = 0.45 + 0.55 * f
        elif self.state == "dismiss":
            f = clamp01((t - self.t0) / self.DISMISS)
            alpha = 1.0 - f
            wire = f
        alpha *= 0.93 + 0.07 * math.sin(t * 19.0) * math.sin(t * 6.1)
        hot = 0.0
        if t - self._flash < 0.45:
            hot = 0.9 * (1.0 - (t - self._flash) / 0.45)
        if t - self._lock < 0.5:
            hot = max(hot, 1.1 * (1.0 - (t - self._lock) / 0.5))
        return alpha, wire, hot

    def draw(self, ov, k: float, t: float) -> None:
        """Glow, rings and motes into the shared half-res overlay."""
        if self.pos is None or self.state == "off" or self.scale < 4.0:
            return
        self._fx(ov, k, t, (self.pos[0] * k, self.pos[1] * k), self.scale * k)

    def draw_sharp(self, frame, t: float) -> None:
        """The shooter itself, at full resolution, after the overlay composite.

        A wrist-worn model is only ~55 px wide in the half-res overlay, where
        its panel lines collapse into a blue smudge.  Rendering it into a small
        full-res ROI keeps every rib, vent and cartridge readable.
        """
        if self.pos is None or self.state == "off" or self.scale < 4.0:
            return
        alpha, wire, hot = self._shading(t)
        render_glow(frame, MODELS.shooter_mesh(1), self.pos, self.scale, self.rot,
                    alpha=alpha, wire=wire, hot=hot, cull=True,
                    gain=0.92, glow=0.5)

    def _fx(self, ov, k: float, t: float, c, s: float) -> None:
        if t - self._flash < 0.55:               # materialise ripple
            f = (t - self._flash) / 0.55
            _ring(ov, c, s * (0.8 + 2.0 * f), dim(HOLO_CYAN, 0.85 * (1.0 - f)), 2)
        if self.state in ("held", "float", "seek"):
            _ring(ov, c, s * 1.55, dim(HOLO_CYAN, 0.35 + 0.25 * math.sin(t * 5.0)), 1)
            for i in range(6):                   # orbiting motes
                a = t * 1.1 + i * math.pi / 3.0
                mx = c[0] + math.cos(a) * s * 1.55
                my = c[1] + math.sin(a) * s * 0.65
                cv2.circle(ov, (int(mx), int(my)), 2,
                           dim(HOLO_CYAN, 0.4 + 0.5 * math.sin(t * 4.0 + i)), -1)
        if self.state == "on" and self._fr is not None:
            # strap glow around the wrist band, so it reads as WORN rather
            # than as an object floating near the arm
            band = project(np.array([[-0.58, 0.0, 0.0]]), c, s, self.rot)[0]
            _ring(ov, band, s * 0.92, dim(HOLO_BLUE, 0.45), 1,
                  squash=0.80, ang=self._fr["theta"] + math.pi / 2.0)
            # muzzle bloom at the spinneret
            nozzle = project(np.array([[1.36, 0.99, 0.0]]), c, s, self.rot)[0]
            pulse = 0.45 + 0.30 * (0.5 + 0.5 * math.sin(t * 3.4))
            cv2.circle(ov, (int(nozzle[0]), int(nozzle[1])), max(2, int(0.13 * s)),
                       dim(HOLO_CYAN, pulse * 0.7), 1, cv2.LINE_AA)
            cv2.circle(ov, (int(nozzle[0]), int(nozzle[1])), max(1, int(0.05 * s)),
                       dim(HOLO_WHITE, pulse), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Blueprint: the exploded engineering view of the web-shooter
# --------------------------------------------------------------------------- #
PART_LABELS = {
    MODELS.WS_BAND: "01 WRIST COLLAR",
    MODELS.WS_HOUSING: "02 CHASSIS / LOGIC DECK",
    MODELS.WS_CART: "03 FLUID CARTRIDGE x2",
    MODELS.WS_BARREL: "04 PRESSURE BARREL",
    MODELS.WS_NOZZLE: "05 SPINNERET NOZZLE",
    MODELS.WS_TRIGGER: "06 TRIGGER PAD",
    MODELS.WS_POD: "07 STABILISER POD x2",
}
_SPECS = ("SHEAR      120 MPa", "FLUID      2 x 40 ml", "PRESSURE   310 bar",
          "RANGE      18.0 m", "CYCLE      0.42 s", "MASS       0.31 kg")


class Blueprint:
    """A rotating, exploding, dimensioned 3D breakdown of the web-shooter.

    Spawned by a single fist; stays until the fist gesture dismisses it, so it
    can be grabbed (4 s lock) and moved anywhere on screen.
    """

    CYCLE = 6.0

    def __init__(self, px, radius: float, t: float):
        self.pos = (float(px[0]), float(px[1]))
        self.scale = max(48.0, float(radius))
        self.t = 0.0
        self.born = t
        self.dying = None
        self._drag_off = (0.0, 0.0)
        self._lock = -9.0
        self._focus = 0
        self._focus_t = 0.0

    # -- lifecycle ---------------------------------------------------------- #
    def alive(self) -> bool:
        return self.dying is None or (self.t - self.dying) < 0.45

    def dismiss(self) -> None:
        if self.dying is None:
            self.dying = self.t

    def update(self, dt: float) -> None:
        self.t += dt
        self._focus_t += dt
        if self._focus_t > 2.2:
            self._focus_t = 0.0
            self._focus = (self._focus + 1) % len(MODELS.shooter_mesh().parts)

    # -- drag protocol ------------------------------------------------------ #
    def drag_point(self):
        return self.pos

    def drag_radius(self) -> float:
        return max(70.0, self.scale * 1.9)

    def begin_drag(self, px, t: float) -> None:
        self._lock = self.t
        self._drag_off = (self.pos[0] - px[0], self.pos[1] - px[1])

    def drag_to(self, px) -> None:
        self.pos = (px[0] + self._drag_off[0], px[1] + self._drag_off[1])

    def end_drag(self, t: float) -> None:
        pass

    # -- phase -------------------------------------------------------------- #
    def _phase(self):
        cyc = (self.t % self.CYCLE) / self.CYCLE
        if cyc < 0.28:
            return 1.0 - smoothstep(cyc / 0.28)      # assembling
        if cyc < 0.55:
            return 0.0                               # held together
        if cyc < 0.80:
            return smoothstep((cyc - 0.55) / 0.25)   # exploding
        return 1.0                                   # inspection hold

    def _alpha(self):
        a = smoothstep(clamp01(self.t / 0.35))
        if self.dying is not None:
            a *= 1.0 - clamp01((self.t - self.dying) / 0.45)
        return a

    # -- render ------------------------------------------------------------- #
    def draw(self, ov, k: float, t: float) -> None:
        alpha = self._alpha()
        if alpha <= 0.02:
            return
        spread = self._phase()
        c = (self.pos[0] * k, self.pos[1] * k)
        s = self.scale * k * (0.55 + 0.45 * smoothstep(clamp01(self.t / 0.35)))
        rot = rot_matrix(0.55 + self.t * 0.34, -0.30 + 0.08 * math.sin(self.t * 0.5),
                         0.06 * math.sin(self.t * 0.7))
        self._rot, self._c, self._s = rot, c, s
        hot = 1.0 * (1.0 - clamp01((self.t - self._lock) / 0.5)) \
            if self.t - self._lock < 0.5 else 0.0
        render_mesh(ov, MODELS.shooter_mesh(), c, s, rot,
                    explode=0.95 * spread, alpha=alpha,
                    wire=0.30 * spread, hot=hot, cull=True)
        # containment ring + orbiting motes
        pr = s * 2.0 * (0.96 + 0.04 * math.sin(self.t * 2.4))
        _ring(ov, c, pr, dim(HOLO_CYAN, alpha * (0.3 + 0.2 * math.sin(self.t * 2.0))), 1)
        for i in range(9):
            a = self.t * 0.8 + i * 2.0 * math.pi / 9.0
            cv2.circle(ov, (int(c[0] + math.cos(a) * pr),
                            int(c[1] + math.sin(a) * pr * 0.42)), 2,
                       dim(HOLO_CYAN, alpha * (0.35 + 0.5 *
                                               (0.5 + 0.5 * math.sin(self.t * 5 + i)))), -1)
        if self.t - self.born < 0.5:              # converging focus ring
            f = (self.t - self.born) / 0.5
            _ring(ov, c, s * (3.2 - 2.0 * f), dim(HOLO_CYAN, 0.8 * (1.0 - f)), 2)

    def annotate(self, frame, t: float) -> None:
        """Crisp full-resolution callouts, dimension lines and the spec block."""
        alpha = self._alpha()
        if alpha <= 0.05:
            return
        hud = frame                    # drawn straight on: thin, crisp geometry
        h, w = frame.shape[:2]
        c, s, rot = self.pos, self.scale, self._rot
        spread = self._phase()
        mesh = MODELS.shooter_mesh()
        col = dim(HOLO_CYAN, 0.9 * alpha)
        # frame + title block
        x0, y0 = int(c[0] - s * 2.3), int(c[1] - s * 2.0)
        x1, y1 = int(c[0] + s * 2.3), int(c[1] + s * 2.0)
        for (ax, ay, bx, by) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                 (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(hud, (ax, ay), (ax + bx * 26, ay), col, 1, cv2.LINE_AA)
            cv2.line(hud, (ax, ay), (ax, ay + by * 26), col, 1, cv2.LINE_AA)
        cv2.line(hud, (x0, y0 - 14), (x0 + 190, y0 - 14), dim(HOLO_BLUE, 0.8 * alpha), 1)
        _text(hud, "WEB-SHOOTER MK-V  //  EXPLODED", (x0, y0 - 20), 0.42,
              dim(HOLO_CYAN, alpha))
        _text(hud, "ARACHNID OS  ::  SCHEMATIC 04-B", (x0, y1 + 22), 0.36,
              dim(HOLO_BLUE, alpha))
        # spec block down the near side
        sx = x1 + 14 if x1 + 190 < w else x0 - 186
        for i, line in enumerate(_SPECS):
            bar = 0.35 + 0.5 * (0.5 + 0.5 * math.sin(t * 2.0 + i))
            _text(hud, line, (sx, y0 + 26 + i * 19), 0.36, dim(HOLO_BLUE, 0.95 * alpha))
            cv2.line(hud, (sx, y0 + 30 + i * 19), (int(sx + 60 * bar), y0 + 30 + i * 19),
                     dim(HOLO_CYAN, 0.7 * alpha), 1)
        # part callouts: leader line from each exploded part to a label
        if spread > 0.15:
            for n, pid in enumerate(mesh.parts):
                centre3 = mesh.part_centre(pid, 0.95 * spread)
                p = project(centre3[None, :], c, s, rot)[0]
                out = 1.0 if p[0] >= c[0] else -1.0
                lx = p[0] + out * (s * 0.55 + 26.0)
                ly = p[1] + (n - 3) * 6.0
                strong = (n == self._focus)
                lc = dim(HOLO_WHITE if strong else HOLO_CYAN,
                         (0.95 if strong else 0.55) * alpha)
                cv2.line(hud, (int(p[0]), int(p[1])), (int(lx), int(ly)), lc, 1, cv2.LINE_AA)
                cv2.line(hud, (int(lx), int(ly)), (int(lx + out * 30), int(ly)), lc, 1,
                         cv2.LINE_AA)
                cv2.circle(hud, (int(p[0]), int(p[1])), 3, lc, 1, cv2.LINE_AA)
                label = PART_LABELS.get(pid, "")
                tx = lx + out * 34 if out > 0 else lx + out * 34 - 8.2 * len(label)
                _text(hud, label, (int(tx), int(ly) - 4), 0.36 if not strong else 0.40,
                      lc)
            # dimension line across the whole assembly
            a3 = mesh.part_centre(MODELS.WS_BAND, 0.95 * spread)
            b3 = mesh.part_centre(MODELS.WS_NOZZLE, 0.95 * spread)
            pa, pb = project(np.stack([a3, b3]), c, s, rot)
            _dimline(hud, pa, pb, dim(HOLO_BLUE, 0.85 * alpha))


def _text(img, s: str, org, scale=0.4, colour=HOLO_CYAN, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thick, cv2.LINE_AA)


def _dimline(img, p1, p2, colour, dash=8, gap=6):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    d = math.hypot(x2 - x1, y2 - y1)
    if d < 2.0:
        return
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    pos = 0.0
    while pos < d:
        e = min(d, pos + dash)
        cv2.line(img, (int(x1 + ux * pos), int(y1 + uy * pos)),
                 (int(x1 + ux * e), int(y1 + uy * e)), colour, 1, cv2.LINE_AA)
        pos = e + gap
    for (px, py) in ((x1, y1), (x2, y2)):
        cv2.line(img, (int(px - uy * 6), int(py + ux * 6)),
                 (int(px + uy * 6), int(py - ux * 6)), colour, 1, cv2.LINE_AA)
