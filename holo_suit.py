#!/usr/bin/env python3
"""holo_suit.py - the full holographic body gear (no mask, ever).

Both fists held ~0.7 s and opened materialise the whole rig: the torso shell
with the spider emblem and webbing, shoulder plates, belt, an armoured
gauntlet with its own arc reactor on every visible wrist, and the chest
reactor burning over the sternum.  The head is deliberately never drawn.

The torso is anchored from the REAL BODY whenever holo_pose.PoseTracker can
see one: the shoulder/hip midpoint places it, shoulder width and torso length
size it, and the shoulder line's tilt plus its foreshortening roll and turn
it, so the armour leans and twists with the wearer instead of floating.  When
no body is available (no model, offline, out of frame) it falls back to the
old hand-derived anchor - hand size for scale, hand midpoint for position -
so the gear keeps working exactly as it did before.  Everything renders
through the shared half-res overlay (bloomed by the upscale), except the chest
reactor and the wrist reactors, which get a crisp full-resolution pass of
their own.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import holo_models as MODELS
from holo3d import (HOLO_BLUE, HOLO_CYAN, HOLO_DEEP, HOLO_WHITE, clamp01,
                    dim, project, render_glow, render_mesh, rot_matrix,
                    smoothstep)
from holo_objects import hand_frame, _mount_rot


# --------------------------------------------------------------------------- #
# Where the torso mesh actually sits in its own model space (holo_models
# _build_suit).  The suit is authored around an origin that is NOT the middle
# of the chest, so mapping a real body onto it means knowing these numbers:
#   shoulder balls at x = +-1.06, y = -0.86
#   belt ring at y = +1.02
# Everything below is derived from those three constants, so if the mesh is
# ever re-authored this is the single place that has to change.
# --------------------------------------------------------------------------- #
MODEL_SH_HALF = 1.06                  # half the shoulder span, model units
MODEL_TORSO = 1.88                    # shoulder line -> belt, model units
MODEL_SH_Y = -0.86                    # shoulder line's height above the origin


class SuitGear:
    """Torso + emblem + webbing + shoulders + belt + gauntlets + reactors."""

    BUILD = 1.30                      # materialise sweep, seconds

    def __init__(self):
        self.t = 0.0
        self.born = 0.0
        self.dying = None
        # smoothed [cx, cy, scale, yaw, pitch, roll] in FULL-FRAME pixels
        self._anchor = None
        self._posed = 0.0             # smoothed 0..1 "a real body is driving this"
        self._rot = None              # last rotation used, for draw_sharp
        self._c = None                # last full-res centre, for draw_sharp
        self._s = 0.0                 # last full-res scale, for draw_sharp

    # -- lifecycle ---------------------------------------------------------- #
    def alive(self) -> bool:
        return self.dying is None or (self.t - self.dying) < 0.55

    def dismiss(self) -> None:
        if self.dying is None:
            self.dying = self.t

    def update(self, dt: float) -> None:
        self.t += dt

    # -- anchoring ----------------------------------------------------------- #
    @staticmethod
    def _from_hands(frames, w: int, h: int):
        """The legacy fallback: guess a torso from where the hands are.

        It cannot know the wearer's build, so it fakes one - hand length as a
        proxy for body size, and the hand midpoint pulled a third of the way
        toward centre because arms wander far further than a chest does.  Only
        used when there is no pose.
        """
        lens = float(np.mean([fr["L"] for fr in frames]))
        mx = float(np.mean([fr["W"][0] for fr in frames]))
        my = float(np.mean([fr["W"][1] for fr in frames]))
        cx = 0.5 * w + (mx - 0.5 * w) * 0.35
        cy = min(0.92 * h, max(0.42 * h, my + 2.1 * lens))
        sc = max(60.0, min(0.30 * h, 3.1 * lens))
        return [cx, cy, sc, 0.0, 0.0, 0.0]

    def _from_pose(self, pose, w: int, h: int):
        """The real thing: a torso placed, sized and oriented by the body."""
        sh = np.asarray(pose["shoulder"], float)
        hip = np.asarray(pose["hip"], float)
        yaw = float(pose.get("yaw", 0.0))
        roll = float(pose.get("roll", 0.0))
        pitch = float(pose.get("pitch", 0.0))

        # SCALE.  Two independent measurements, because each is blind in a
        # different direction: shoulder span collapses when you turn (undone
        # here by dividing out cos(yaw), floored so a near-profile view cannot
        # blow the suit up), and torso length collapses when you lean toward
        # the camera.  They are NOT weighted equally.  The mesh is authored
        # broad and short - span/length is 1.13 where a real torso is nearer
        # 0.65 - so matching the length would leave the shoulder plates hanging
        # a hand's width off the body, which is the single most obvious way for
        # armour to look pasted on.  Width wins; length only moderates it, and
        # the clamps stop a mis-measured torso from producing a comedy result.
        sh_w = float(pose.get("sh_w", 0.0)) / max(0.45, math.cos(min(1.2, abs(yaw))))
        torso = float(pose.get("torso_h", 0.0))
        sc_w = sh_w / (2.0 * MODEL_SH_HALF)
        sc_h = torso / MODEL_TORSO
        sc = 0.70 * sc_w + 0.30 * sc_h
        sc = max(0.55 * sc_h, min(1.10 * sc_h, sc))
        sc = max(40.0, min(0.42 * h, sc))

        # POSITION.  Pin the mesh's SHOULDER LINE to the real shoulder line
        # rather than centring it on the torso: shoulders are the best-tracked
        # landmarks and the ones the eye checks, and since the shell is a chest
        # piece it should hang DOWN from them and stop where it stops.  The
        # model origin sits MODEL_SH_Y above its own shoulders, so step that
        # far down the real shoulder->hip direction.  This is what stops the
        # suit sliding about: it is pinned to the body, not to the arms.
        span = hip - sh
        n = float(np.linalg.norm(span))
        down = span / n if n > 1e-6 else np.array([0.0, 1.0])
        c = sh + down * (-MODEL_SH_Y * sc)
        return [float(c[0]), float(c[1]), float(sc), yaw, pitch, roll]

    def _resolve(self, feats, w: int, h: int, pose=None):
        frames = [fr for fr in (hand_frame(f, w, h) for f in feats) if fr is not None]
        conf = float(pose.get("conf", 0.0)) if pose else 0.0
        tgt = None
        if conf > 0.01:
            tgt = self._from_pose(pose, w, h)
            if frames and conf < 0.999:
                # crossfade rather than switch: when a body is fading out of
                # view the suit should drift back to the hand anchor, not jump
                hnd = self._from_hands(frames, w, h)
                tgt = [t * conf + g * (1.0 - conf) for t, g in zip(tgt, hnd)]
        elif frames:
            tgt = self._from_hands(frames, w, h)
        if tgt is None:
            return None, frames

        # how "posed" we are decides the responsiveness below, and it is itself
        # smoothed so the alpha does not jump on the frame pose appears
        self._posed += (conf - self._posed) * 0.15
        if self._anchor is None:
            self._anchor = list(tgt)
        else:
            # the hand anchor was noisy garbage, hence the old a=0.10 crawl.  A
            # pose signal is already smoothed by the tracker and is genuinely
            # where the body is, so it can be followed hard - at a=0.45 the
            # armour keeps up with a real lean or turn instead of oozing after
            # it.  Scale stays lazier than position either way: a torso that
            # breathes in size is far more distracting than one that lags a
            # pixel or two.
            a = 0.10 + 0.35 * self._posed
            asz = 0.08 + 0.17 * self._posed
            for i in (0, 1):
                self._anchor[i] += (tgt[i] - self._anchor[i]) * a
            self._anchor[2] += (tgt[2] - self._anchor[2]) * asz
            for i in (3, 4, 5):
                self._anchor[i] += (tgt[i] - self._anchor[i]) * a
        return self._anchor, frames

    def _fade(self) -> float:
        if self.dying is None:
            return 1.0
        return 1.0 - clamp01((self.t - self.dying) / 0.55)

    # -- render -------------------------------------------------------------- #
    def draw(self, ov, k: float, t: float, feats, pose=None) -> None:
        """`pose` is the optional torso frame from holo_pose.PoseTracker.detect.

        It stays optional on purpose: every existing caller passes four
        arguments and must keep working, and a missing pose has to degrade to
        the old behaviour rather than to a blank screen.
        """
        w, h = int(ov.shape[1] / k), int(ov.shape[0] / k)
        anchor, frames = self._resolve(feats, w, h, pose)
        if anchor is None:
            return
        fade = self._fade()
        if fade <= 0.02:
            return
        build = clamp01(self.t / self.BUILD)
        if self.dying is not None:
            build = 1.0
        cx, cy, sc, byaw, bpitch, broll = anchor
        c = (cx * k, cy * k)
        s = sc * k
        # With a body the orientation IS the body's - turning and leaning drive
        # the armour directly.  Without one, fall back to the old idle wobble
        # plus a screen-position lean, which at least stops it looking frozen.
        # The wobble is faded out by self._posed so the two blend cleanly.
        idle = 1.0 - self._posed
        yaw = byaw + idle * (0.10 * math.sin(self.t * 0.6)
                             + clamp01((cx / w) - 0.5) * 0.3)
        pitch = bpitch + idle * (-0.06 + 0.03 * math.sin(self.t * 0.8))
        rot = rot_matrix(yaw, pitch, broll)
        # draw_sharp runs after the overlay composite and needs the same frame
        self._rot, self._c, self._s = rot, (cx, cy), sc
        alpha = fade * (0.92 + 0.08 * math.sin(self.t * 17.0) * math.sin(self.t * 5.3))
        wire = (1.0 - smoothstep(build)) * 0.85
        render_mesh(ov, MODELS.suit_mesh(), c, s, rot, alpha=alpha * (0.35 + 0.65 * build),
                    wire=wire, cull=True, scan_phase=t * 14.0)
        if build > 0.35:
            self._decals(ov, c, s, rot, alpha * smoothstep((build - 0.35) / 0.65))
        self._build_sweep(ov, c, s, build, alpha)
        for fr in frames:
            self._gauntlet(ov, fr, k, build, alpha, t)

    def draw_sharp(self, frame, t: float) -> None:
        """The chest reactor, crisp and glowing, after the overlay composite."""
        if self._anchor is None:
            return
        fade = self._fade()
        if fade <= 0.02 or self.dying is not None:
            return
        cx, cy, sc = self._anchor[0], self._anchor[1], self._anchor[2]
        s = sc * 0.42
        # the reactor lives on the sternum, so project its model-space seat
        # through the SAME rotation the torso used: lean or turn and it rides
        # the chest instead of sliding off to one side
        if self._rot is not None:
            seat = project(np.array([[0.0, -0.42, 0.42]]), (cx, cy), sc, self._rot)[0]
            c = (float(seat[0]), float(seat[1]))
        else:
            c = (cx, cy - sc * 0.40)
        # the reactor reads as a solid disc, so give it the same wire-over-fill
        # treatment as the shooters: you see the far ring through the core
        hot = 0.85 + 0.55 * math.sin(t * 3.6) * math.sin(t * 7.3)
        pulse = (0.55 + 0.45 * hot)
        # the reactor face turns with the chest it is bolted to, but only
        # partly in yaw: a disc seen edge-on vanishes, and the glow is the one
        # element that must stay readable at every angle
        byaw, broll = self._anchor[3], self._anchor[5]
        rot = rot_matrix(0.55 * byaw + 0.10 * math.sin(t * 0.6) * (1.0 - self._posed),
                         -0.05, broll)
        render_glow(frame, MODELS.reactor_mesh(), c, s, rot,
                    alpha=fade * pulse, wire=0.30, hot=0.55 * fade,
                    cull=False, scan_phase=t * 26.0, gain=0.95, glow=0.60)
        # A soft light halo so it reads as a light source, not a sticker.  It
        # has to be ADDED, not painted: a filled circle replaces pixels, which
        # punched a dark disc straight through the wearer's chest.  Only the
        # halo's own bounding box is touched, so this stays cheap.
        r = int(s * 1.6)
        x0, y0 = max(0, int(c[0]) - r), max(0, int(c[1]) - r)
        x1, y1 = (min(frame.shape[1], int(c[0]) + r + 1),
                  min(frame.shape[0], int(c[1]) + r + 1))
        if x1 > x0 and y1 > y0:
            halo = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
            cv2.circle(halo, (int(c[0]) - x0, int(c[1]) - y0), r,
                       dim(HOLO_DEEP, 0.10 * fade * pulse), -1, cv2.LINE_AA)
            roi = frame[y0:y1, x0:x1]
            cv2.add(roi, halo, roi)

    def _decals(self, ov, c, s, rot, alpha: float) -> None:
        for kind, pts, weight in MODELS.spider_decals():
            p = project(pts, c, s, rot).astype(np.int32)
            colour = dim(HOLO_WHITE if weight > 0.9 else HOLO_CYAN,
                         alpha * (0.95 if weight > 0.9 else 0.42))
            if kind == "fill":
                cv2.fillPoly(ov, [p], dim(HOLO_CYAN, alpha * 0.55))
                cv2.polylines(ov, [p], True, colour, 1, cv2.LINE_AA)
            else:
                cv2.polylines(ov, [p], False, colour, 1, cv2.LINE_AA)

    def _build_sweep(self, ov, c, s, build: float, alpha: float) -> None:
        if build >= 1.0:
            return
        y = c[1] + s * (1.25 - 2.6 * build)
        x0, x1 = int(c[0] - s * 1.5), int(c[0] + s * 1.5)
        cv2.line(ov, (x0, int(y)), (x1, int(y)), dim(HOLO_WHITE, 0.9 * alpha), 2,
                 cv2.LINE_AA)
        cv2.line(ov, (x0, int(y) + 3), (x1, int(y) + 3), dim(HOLO_CYAN, 0.4 * alpha), 1)

    def _gauntlet(self, ov, fr, k: float, build: float, alpha: float, t: float) -> None:
        arm = {"theta": math.atan2(-fr["ax"][1], -fr["ax"][0]),
               "side": fr["side"], "L": fr["L"], "dz": fr.get("dz", 0.0)}
        rot = _mount_rot(arm, wobble=0.04 * math.sin(self.t * 1.4))
        c = ((fr["W"][0] - fr["ax"][0] * 0.12 * fr["L"]) * k,
             (fr["W"][1] - fr["ax"][1] * 0.12 * fr["L"]) * k)
        s = 0.52 * fr["L"] * k
        render_mesh(ov, MODELS.gauntlet_mesh(), c, s, rot,
                    alpha=alpha * (0.4 + 0.6 * build),
                    wire=(1.0 - smoothstep(build)) * 0.85, cull=True,
                    scan_phase=t * 21.0)
        if build < 1.0:                       # materialise ring at the wrist
            rr = (0.7 + 2.0 * build) * s
            cv2.ellipse(ov, (int(c[0]), int(c[1])), (int(rr), int(rr * 0.42)),
                        math.degrees(arm["theta"]), 0, 360,
                        dim(HOLO_CYAN, alpha * (1.0 - build)), 2, cv2.LINE_AA)
        # wrist arc reactor: a pulsing core where the cuff's reactor ring sits
        core = project(np.array([[0.30, 0.62, 0.0]]), c, s, rot)[0]
        pulse = 0.5 + 0.5 * math.sin(t * 3.1 + (core[0] + core[1]) * 0.01)
        cv2.circle(ov, (int(core[0]), int(core[1])), max(2, int(0.16 * s)),
                   dim(HOLO_CYAN, alpha * (0.45 + 0.5 * pulse)), 1, cv2.LINE_AA)
        cv2.circle(ov, (int(core[0]), int(core[1])), max(1, int(0.07 * s)),
                   dim(HOLO_WHITE, alpha * (0.6 + 0.4 * pulse)), -1, cv2.LINE_AA)
