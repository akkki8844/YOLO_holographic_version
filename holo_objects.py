#!/usr/bin/env python3
"""holo_objects.py - the hard-light objects: web-shooters, body gear, palm HUD.

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
from holo3d import (HOLO_BLUE, HOLO_CYAN, HOLO_DEEP, HOLO_WHITE, FLIP_X,
                    clamp01, dim, ease_out_back, project, render_glow,
                    render_mesh, rot_matrix, smoothstep)


def hand_frame(feat, w: int, h: int):
    """Screen-space frame of one hand: wrist, aim axis, thumb side, size.

    'dz' is how far the knuckles sit IN FRONT of the wrist, in hand-lengths,
    read off the landmarks' depth channel: positive when the hand is pointing
    at the camera.  Mounted hardware uses it to tilt in depth, so it turns with
    the arm in 3D instead of sliding around on a flat plane.
    """
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
    dz = 0.0
    lms = feat.get("landmarks")
    if lms is not None:
        try:                                  # 0 = wrist, 9 = middle-finger MCP
            span = math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y)
            dz = (lms[0].z - lms[9].z) / (span + 1e-6)
        except Exception:  # noqa: BLE001 - depth is a bonus, never a requirement
            dz = 0.0
        dz = max(-1.1, min(1.1, dz))
    return {"W": wr, "M": mc, "L": length, "ax": ax, "side": side, "dz": dz,
            "theta": math.atan2(ax[1], ax[0])}


def _mount_rot(fr, wobble: float = 0.0):
    """Rotation putting the model's +x along the arm and +y on the thumb side.

    The yaw also carries the hand's depth tilt: point your hand at the camera
    and the barrel foreshortens toward you, exactly as a real one bolted to
    that wrist would.
    """
    th = fr["theta"]
    rot = rot_matrix(0.42 + wobble - 0.85 * fr.get("dz", 0.0), -0.30, th)
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
            peel      being physically pulled off the wrist right now - a
                      short unclip-and-swing-free animation, not a teleport
            held      grabbed (4 s lock) or just peeled off, following a pinch
            float     detached: its own object, parked where it was let go
            dismiss   spinning down and dissolving
    """

    ARRIVE = 0.70
    PEEL = 0.22
    DISMISS = 0.45

    # The strap physics.  A real shooter is STRAPPED to the arm, not welded to
    # it: whip your wrist and the housing lags a few millimetres, then settles.
    # Critical damping is the whole point - underdamped would wobble like jelly,
    # which is a different and equally wrong kind of unreal.
    MOUNT_W = 30.0                       # spring angular frequency, rad/s
    # Peak kick works out at RECOIL/(MOUNT_W*e) px, so this is ~10 px of travel
    # back along the forearm - readable against a ~70 px housing without
    # looking like the thing came loose.
    RECOIL = 850.0                       # px/s impulse when a web fires

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
        self._webpose = False
        self._webpose_since = None
        self._peel_from = None
        self._peel_to = None
        self._vel = (0.0, 0.0)       # mount-spring velocity, px/s
        self._prev = None            # last frame's position, for swing
        self._swing = 0.0            # trailing roll of a dangling shooter

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

    def reattach(self, t: float, from_px=None) -> bool:
        """Put it back on: it flies home to the wrist from wherever it is."""
        if self.state in ("on", "arrive"):
            return False
        src = self.pos if self.pos is not None else from_px
        self._from = (float(src[0]), float(src[1])) if src is not None else (0.0, 0.0)
        if self._park is None:
            self._park = self._from
        self.state = "arrive"
        self.t0 = t
        self._flash = t
        return True

    def detach(self, t: float, px) -> bool:
        """Peel the shooter off the wrist - the real 'take it off' gesture.

        Reaching over, pinching the shooter on the other wrist and pulling is
        exactly how you take one off in real life, so it starts coming off
        immediately: no 4-second move lock, no dismissal.  It doesn't teleport
        into your grip though - it unclips and swings free over PEEL seconds
        (a short pop away from the wrist along the forearm, then a swing into
        the pinch), which is what actually reads as being taken off rather
        than being replaced by a different object.  Once the animation lands
        it becomes its own free object, follows the pinch, and stays wherever
        it is let go.
        """
        if self.state not in ("on", "seek"):
            return False
        self.state = "peel"
        self.t0 = t
        self._flash = t
        self.scale = max(self.scale, 40.0)
        self._peel_from = self.pos if self.pos is not None else (float(px[0]), float(px[1]))
        self._peel_to = (float(px[0]), float(px[1]))
        return True

    # -- drag protocol ------------------------------------------------------ #
    def drag_point(self):
        return self.pos if self.state in ("on", "float", "held") else None

    def drag_radius(self) -> float:
        return max(46.0, self.scale * 1.05)

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
    def _settle(self, target, dt: float):
        """Chase the wrist anchor through a stiff critically-damped spring.

        Snapping the model onto the anchor every frame is what made it read as
        a decal painted onto the video: nothing in the real world tracks your
        arm with zero lag.  A stiff spring costs about a sixth of a second of
        settle and buys the single strongest cue that the thing is a physical
        object riding your arm rather than a sprite pinned to a landmark.
        """
        if self.pos is None:
            self._vel = (0.0, 0.0)
            return (float(target[0]), float(target[1]))
        # The EXACT solution of a critically-damped spring over dt, not a
        # stepped approximation.  A strap stiff enough to feel like a strap has
        # C*dt > 1 at 30 fps, and explicit integration of that flips the
        # velocity's sign every frame - the model shakes itself apart instead of
        # settling.  The closed form is unconditionally stable at any framerate
        # and never overshoots, so the same feel survives a frame-rate drop.
        dt = max(1e-4, min(dt, 0.1))
        w = self.MOUNT_W
        e = math.exp(-w * dt)
        pos, vel = [0.0, 0.0], [0.0, 0.0]
        for i in (0, 1):
            x = self.pos[i] - target[i]          # displacement from the anchor
            k = (self._vel[i] + w * x) * dt
            pos[i] = target[i] + (x + k) * e
            vel[i] = (self._vel[i] - w * k) * e
        self._vel = (vel[0], vel[1])
        return (pos[0], pos[1])

    def update(self, dt: float, t: float, feat, w: int, h: int) -> None:
        fr = hand_frame(feat, w, h)
        self._fr = fr
        wp = bool(feat is not None and feat.get("webpose", False))
        if wp and not self._webpose:
            self._webpose_since = t
            # A web firing kicks the housing back along the forearm.  The
            # impulse goes straight INTO the mount spring rather than playing a
            # canned bounce animation, so the kick and the settle afterwards
            # are the same physics that already holds it on the arm - which is
            # why it lands like recoil instead of like a wobble effect.
            if self.state == "on" and fr is not None:
                self._vel = (self._vel[0] - fr["ax"][0] * self.RECOIL,
                             self._vel[1] - fr["ax"][1] * self.RECOIL)
        elif not wp:
            self._webpose_since = None
        self._webpose = wp
        self.spin += dt
        if fr is not None:
            # the model origin is the wrist itself: the shooter housing runs back up the
            # forearm from here and only the spinneret crosses onto the hand
            anchor = (fr["W"][0] + fr["ax"][0] * 0.04 * fr["L"],
                      fr["W"][1] + fr["ax"][1] * 0.04 * fr["L"])
            live_scale = 1.05 * fr["L"]
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
                self._vel = (0.0, 0.0)
            else:
                # strapped on, not welded: it trails the wrist by a hair and
                # settles, and a fired web kicks it back through the same spring
                self.pos = self._settle(anchor, dt)
                self.scale = live_scale
                self.rot = _mount_rot(fr, wobble=0.05 * math.sin(t * 1.6))
        elif self.state == "peel":
            f = clamp01((t - self.t0) / self.PEEL)
            e = smoothstep(f)
            # the axis to pop away along: whatever forearm direction we last
            # saw, so the shooter unclips straight back off the wrist instead
            # of sliding sideways through it
            ax = fr["ax"] if fr is not None else (self._fr["ax"] if self._fr
                                                   is not None else (0.0, -1.0))
            base_scale = live_scale if live_scale else max(self.scale, 42.0)
            bulge = math.sin(math.pi * f) * 0.60 * base_scale
            bx = self._peel_from[0] + (self._peel_to[0] - self._peel_from[0]) * e
            by = self._peel_from[1] + (self._peel_to[1] - self._peel_from[1]) * e
            self.pos = (bx - ax[0] * bulge, by - ax[1] * bulge)
            self.rot = rot_matrix(0.45 + self.spin * 0.8, -0.30, (1.0 - e) * 1.15)
            self.scale = base_scale * (1.0 + 0.18 * math.sin(math.pi * f))
            if f >= 1.0:
                self.state = "held"
                self._lock = t
                self.pos = self._peel_to
                self._drag_off = (0.0, 0.0)
        elif self.state == "held":
            # A held object SWINGS.  Carry it left and it trails right, the way
            # anything dangling from your fingers does, and it settles when you
            # stop.  The roll comes from the pinch's REAL lateral speed rather
            # than a loop, so the motion is the one you actually made.
            vx = 0.0
            if self.pos is not None and self._prev is not None and dt > 1e-4:
                vx = (self.pos[0] - self._prev[0]) / dt
            want = math.copysign(0.85, vx) * clamp01(abs(vx) / 900.0)
            self._swing += (want - self._swing) * min(1.0, dt * 7.0)
            self.rot = rot_matrix(0.45 + self.spin * 0.5, -0.30,
                                  0.20 * math.sin(t * 1.1) + self._swing)
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
        self._prev = self.pos

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
        # cull=False and a part-wireframe shell: you see the FAR wall of the
        # cuff through the near one, which is what makes it read as wrapped
        # around the arm instead of pasted in front of it.  The LOD mesh keeps
        # drawing both sides affordable.
        render_glow(frame, MODELS.shooter_mesh(0), self.pos, self.scale, self.rot,
                    alpha=alpha, wire=max(wire, 0.34), hot=hot, cull=False,
                    scan_phase=t * 21.0, gain=1.05, glow=0.68)

    def _fx(self, ov, k: float, t: float, c, s: float) -> None:
        if t - self._flash < 0.55:               # materialise ripple
            f = (t - self._flash) / 0.55
            _ring(ov, c, s * (0.5 + 1.3 * f), dim(HOLO_CYAN, 0.85 * (1.0 - f)), 2)
        if self.state == "dismiss":
            # DE-REZ.  The housing comes apart into horizontal bands that peel
            # upward and fade, so it dissolves like a projection losing signal
            # instead of just shrinking - a shrink reads as an object moving
            # away from you, which is the wrong idea entirely.
            f = clamp01((t - self.t0) / self.DISMISS)
            for i in range(5):
                fi = i / 5.0
                yb = c[1] + (fi - 0.5) * s * 0.9 - f * (1.0 + fi * 1.6) * s * 1.4
                band = (1.0 - f) * (1.0 - 0.45 * fi)
                cv2.line(ov, (int(c[0] - s * 0.85), int(yb)),
                         (int(c[0] + s * 0.85), int(yb)),
                         dim(HOLO_CYAN, 0.75 * band), 1, cv2.LINE_AA)
        if self.state == "arrive":
            # a horizontal scan-line sweeps down through the housing as it
            # builds itself out of hard light, top to bottom, once
            f = clamp01((t - self.t0) / self.ARRIVE)
            sy = c[1] - s * 0.9 + s * 1.8 * f
            cv2.line(ov, (int(c[0] - s * 1.15), int(sy)), (int(c[0] + s * 1.15), int(sy)),
                     dim(HOLO_WHITE, 0.85 * (1.0 - abs(f - 0.5) * 1.2)), 2, cv2.LINE_AA)
            cv2.line(ov, (int(c[0] - s * 1.15), int(sy)), (int(c[0] + s * 1.15), int(sy)),
                     dim(HOLO_CYAN, 0.5), 1, cv2.LINE_AA)
        if self.state in ("held", "float", "seek"):
            _ring(ov, c, s * 1.05, dim(HOLO_CYAN, 0.35 + 0.25 * math.sin(t * 5.0)), 1)
            for i in range(6):                   # orbiting motes
                a = t * 1.1 + i * math.pi / 3.0
                mx = c[0] + math.cos(a) * s * 1.05
                my = c[1] + math.sin(a) * s * 0.45
                cv2.circle(ov, (int(mx), int(my)), 2,
                           dim(HOLO_CYAN, 0.4 + 0.5 * math.sin(t * 4.0 + i)), -1)
        if self.state == "on" and self._fr is not None:
            # strap glow around the wrist band, so it reads as WORN rather
            # than as an object floating near the arm
            band = project(np.array([[-0.06, 0.0, 0.0]]), c, s, self.rot)[0]
            _ring(ov, band, s * 0.58, dim(HOLO_BLUE, 0.45), 1,
                  squash=0.80, ang=self._fr["theta"] + math.pi / 2.0)
            # muzzle bloom at the spinneret
            nozzle = project(np.array([[0.57, 0.55, 0.0]]), c, s, self.rot)[0]
            pulse = 0.45 + 0.30 * (0.5 + 0.5 * math.sin(t * 3.4))
            cv2.circle(ov, (int(nozzle[0]), int(nozzle[1])), max(2, int(0.13 * s)),
                       dim(HOLO_CYAN, pulse * 0.7), 1, cv2.LINE_AA)
            cv2.circle(ov, (int(nozzle[0]), int(nozzle[1])), max(1, int(0.05 * s)),
                       dim(HOLO_WHITE, pulse), -1, cv2.LINE_AA)
            # web-fluid cartridge readout on the underside: a small bank of
            # ticks reporting the canister's charge, the way the real gadget
            # would tell you it's about to run dry
            cart = project(np.array([[-0.55, -0.62, 0.0]]), c, s, self.rot)[0]
            level = 0.8 + 0.06 * math.sin(t * 0.5)
            for i in range(4):
                lit = (i / 4.0) < level
                tx = cart[0] + i * 0.10 * s
                cv2.line(ov, (int(tx), int(cart[1])), (int(tx), int(cart[1] - 0.16 * s)),
                         dim(HOLO_CYAN if lit else HOLO_DEEP, 0.75 if lit else 0.35),
                         max(1, int(0.045 * s)), cv2.LINE_AA)
            # the web-shoot pose snaps a targeting line out of the nozzle:
            # dotted filament along the aim axis with a reticle at the end
            if getattr(self, "_webpose", False):
                self._webline(ov, c, s, t)

    MULTI_LOCK_HOLD = 0.90       # seconds the pose must hold before it fans out

    def _webline(self, ov, c, s, t) -> None:
        """A thin holographic web-line from the spinneret toward the aim.

        Holding the pose past MULTI_LOCK_HOLD fans the single reticle out into
        a three-point lock, the way the suit's multi-web targeting reads a
        held aim as "cover more than one point" instead of just one shot.
        """
        if self._fr is None:
            return
        p0 = project(np.array([[0.60, 0.45, 0.0]]), c, s, self.rot)[0]
        p1 = project(np.array([[2.6, 0.48, 0.0]]), c, s, self.rot)[0]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        d = math.hypot(dx, dy)
        if d < 4.0:
            return
        ux, uy = dx / d, dy / d
        ln = min(0.55 * ov.shape[1], d)
        col = dim(HOLO_CYAN, 0.42 + 0.25 * math.sin(t * 5.0))
        pos = 0.0
        while pos < ln:                     # dotted filament
            e = min(ln, pos + 0.09 * s)
            cv2.line(ov, (int(p0[0] + ux * pos), int(p0[1] + uy * pos)),
                     (int(p0[0] + ux * e), int(p0[1] + uy * e)), col, 1, cv2.LINE_AA)
            pos = e + 0.05 * s
        ex, ey = p0[0] + ux * ln, p0[1] + uy * ln     # target reticle
        rr = max(3, int(0.14 * s))
        held = (t - self._webpose_since) if self._webpose_since is not None else 0.0
        if held < self.MULTI_LOCK_HOLD:
            self._reticle(ov, (ex, ey), rr, col)
            return
        # multi-lock: the primary reticle plus two more fanned off the aim
        # perpendicular, all tied back to it with thin bracket lines
        px, py = -uy, ux
        spread = rr * 3.0 * smoothstep(clamp01((held - self.MULTI_LOCK_HOLD) / 0.35))
        for i, off in enumerate((0.0, spread, -spread)):
            tx, ty = ex + px * off, ey + py * off
            self._reticle(ov, (tx, ty), int(rr * (1.0 if i == 0 else 0.72)), col)
            if off != 0.0:
                cv2.line(ov, (int(ex), int(ey)), (int(tx), int(ty)),
                         dim(HOLO_CYAN, 0.30), 1, cv2.LINE_AA)

    @staticmethod
    def _reticle(ov, centre, rr, col) -> None:
        ex, ey = centre
        cv2.circle(ov, (int(ex), int(ey)), rr, col, 1, cv2.LINE_AA)
        cv2.circle(ov, (int(ex), int(ey)), max(1, rr // 3), col, -1, cv2.LINE_AA)
        for gx, gy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(ov, (int(ex + gx * rr * 1.6), int(ey + gy * rr * 1.6)),
                     (int(ex + gx * rr * 2.3), int(ey + gy * rr * 2.3)),
                     col, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# PalmProjector: the hologram that materialises above an open palm
# --------------------------------------------------------------------------- #
class PalmProjector:
    """A Stark palm-projection: a holographic screen hovering over the hand.

    Continuous rather than toggled - it fades in while the palm stays open and
    spread (the resting hand is open, so the projector is the ambient furniture
    of the AR desk), tracks the palm in 3D, and fades out the moment the hand
    closes.  A small rotating web-shooter holo plays on the screen, so the
    projection reads as a live HUD rather than a static decal.
    """

    ARM = 0.45                  # seconds an open palm must hold to project
    FADE_OUT = 0.30             # seconds to dissolve once the palm closes

    def __init__(self, side: str):
        self.side = side
        self.pos = None             # screen centre, full-resolution px
        self._fr = None
        self._held = 0.0
        self._last_hold = None
        self._released_at = None

    # -- desk-driven lifecycle --------------------------------------------- #
    def hold(self, t: float) -> None:
        if self._last_hold is not None:
            self._held += max(0.0, t - self._last_hold)
        self._last_hold = t
        self._released_at = None

    def release(self, t: float) -> None:
        if self._last_hold is not None:
            self._released_at = t
        self._last_hold = None

    def dead(self, t: float) -> bool:
        if self._last_hold is not None:
            return False
        return (self._released_at is not None
                and t - self._released_at > self.FADE_OUT + 0.15)

    def track(self, feat, w: int, h: int) -> None:
        fr = hand_frame(feat, w, h)
        self._fr = fr
        if fr is None:
            return
        # the screen hovers just past the fingertips, tilted with the hand
        self.pos = (fr["M"][0] + fr["ax"][0] * 0.92 * fr["L"],
                    fr["M"][1] + fr["ax"][1] * 0.92 * fr["L"])

    def _alpha(self, t: float) -> float:
        arm = smoothstep(clamp01(self._held / self.ARM))
        if self._last_hold is not None:
            return arm
        if self._released_at is None:
            return 0.0
        return arm * (1.0 - clamp01((t - self._released_at) / self.FADE_OUT))

    # -- render ------------------------------------------------------------- #
    def draw(self, ov, k: float, t: float) -> None:
        if self.pos is None or self._fr is None:
            return
        alpha = self._alpha(t)
        if alpha <= 0.03:
            return
        c = (self.pos[0] * k, self.pos[1] * k)
        fr = self._fr
        s = 1.05 * fr["L"] * k
        th = fr["theta"]
        # the light beam from the palm up into the screen
        base = (fr["M"][0] * k, fr["M"][1] * k)
        for i in range(7):
            f = i / 7.0
            bx = base[0] + (c[0] - base[0]) * f + 2.0 * math.sin(t * 6.0 + i) * f
            by = base[1] + (c[1] - base[1]) * f
            cv2.circle(ov, (int(bx), int(by)), max(1, int(1.4 * s * (1.0 - f))),
                       dim(HOLO_CYAN, alpha * (0.5 - 0.35 * f)), -1, cv2.LINE_AA)
        # the screen itself: a tilted holographic ring with ticks
        cv2.ellipse(ov, (int(c[0]), int(c[1])),
                    (int(s), int(s * 0.42)), math.degrees(th),
                    0, 360, dim(HOLO_CYAN, alpha * 0.8), 2, cv2.LINE_AA)
        cv2.ellipse(ov, (int(c[0]), int(c[1])),
                    (int(s * 1.28), int(s * 0.54)), math.degrees(th),
                    0, 360, dim(HOLO_DEEP, alpha * 0.5), 1, cv2.LINE_AA)
        for i in range(8):
            a = t * 0.9 + i * math.pi / 4.0
            r0, r1 = s * 1.04, s * 1.04 + (4.0 if i % 2 == 0 else 8.0)
            cv2.line(ov,
                     (int(c[0] + math.cos(a) * r0 * 0.42), int(c[1] + math.sin(a) * r0)),
                     (int(c[0] + math.cos(a) * r1 * 0.42), int(c[1] + math.sin(a) * r1)),
                     dim(HOLO_WHITE, alpha * (0.5 + 0.3 * (i % 2))), 1, cv2.LINE_AA)
        # charge ring when it is still arming
        if self._held < self.ARM:
            p = clamp01(self._held / self.ARM)
            cv2.ellipse(ov, (int(c[0]), int(c[1])),
                        (int(s * 1.6), int(s * 0.67)), math.degrees(th),
                        -90, -90 + 360 * p, dim(HOLO_WHITE, alpha * 0.9), 2,
                        cv2.LINE_AA)

    def draw_sharp(self, frame, t: float) -> None:
        """The holographic tablet + rotating mini-shooter, at full resolution."""
        if self.pos is None or self._fr is None:
            return
        alpha = self._alpha(t)
        if alpha <= 0.03:
            return
        fr = self._fr
        s = 1.05 * fr["L"]
        rot = rot_matrix(0.42 + 0.10 * math.sin(t * 0.8), -0.34,
                         fr["theta"])
        render_glow(frame, MODELS.palm_screen_mesh(), self.pos, s, rot,
                    alpha=alpha, wire=0.30, cull=False,
                    scan_phase=t * 30.0, gain=0.92, glow=0.56)
        # the rotating mini web-shooter playing on the screen
        mini = rot_matrix(t * 1.4, 0.4, 0.15)
        mc = (self.pos[0], self.pos[1] - 0.06 * s)
        render_glow(frame, MODELS.shooter_mesh(0), mc, 0.40 * s, mini,
                    alpha=alpha, wire=0.45, cull=False,
                    scan_phase=t * 30.0, gain=0.98, glow=0.60)
        # corner tag
        tag = f"{self.side.upper()} PALM  //  AR LINK"
        cv2.putText(frame, tag, (int(self.pos[0] - s * 0.9),
                                 int(self.pos[1] + s * 0.72)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    dim(HOLO_CYAN, 0.75 * alpha), 1, cv2.LINE_AA)
