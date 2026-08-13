#!/usr/bin/env python3
"""holo_suit.py - the full holographic body gear (no mask, ever).

Both fists held ~0.7 s and opened materialise the whole rig: the torso shell
with the spider emblem and webbing, shoulder plates, belt, an armoured
gauntlet with its own arc reactor on every visible wrist, and the chest
reactor burning over the sternum.  The head is deliberately never drawn.

The torso is anchored from the visible hands: hand size gives the scale and
the hand midpoint nudges the body left/right, with heavy smoothing so the
suit sits still while the hands move.  Everything renders through the shared
half-res overlay (bloomed by the upscale), except the chest reactor and the
wrist reactors, which get a crisp full-resolution pass of their own.
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


class SuitGear:
    """Torso + emblem + webbing + shoulders + belt + gauntlets + reactors."""

    BUILD = 1.30                      # materialise sweep, seconds

    def __init__(self):
        self.t = 0.0
        self.born = 0.0
        self.dying = None
        self._anchor = None           # smoothed (cx, cy, scale)

    # -- lifecycle ---------------------------------------------------------- #
    def alive(self) -> bool:
        return self.dying is None or (self.t - self.dying) < 0.55

    def dismiss(self) -> None:
        if self.dying is None:
            self.dying = self.t

    def update(self, dt: float) -> None:
        self.t += dt

    # -- anchoring ----------------------------------------------------------- #
    def _resolve(self, feats, w: int, h: int):
        frames = [fr for fr in (hand_frame(f, w, h) for f in feats) if fr is not None]
        if not frames:
            return None, []
        lens = float(np.mean([fr["L"] for fr in frames]))
        mx = float(np.mean([fr["W"][0] for fr in frames]))
        my = float(np.mean([fr["W"][1] for fr in frames]))
        cx = 0.5 * w + (mx - 0.5 * w) * 0.35
        cy = min(0.92 * h, max(0.42 * h, my + 2.1 * lens))
        sc = max(60.0, min(0.30 * h, 3.1 * lens))
        if self._anchor is None:
            self._anchor = [cx, cy, sc]
        else:
            a = 0.10
            self._anchor[0] += (cx - self._anchor[0]) * a
            self._anchor[1] += (cy - self._anchor[1]) * a
            self._anchor[2] += (sc - self._anchor[2]) * a
        return self._anchor, frames

    def _fade(self) -> float:
        if self.dying is None:
            return 1.0
        return 1.0 - clamp01((self.t - self.dying) / 0.55)

    # -- render -------------------------------------------------------------- #
    def draw(self, ov, k: float, t: float, feats) -> None:
        anchor, frames = self._resolve(feats, int(ov.shape[1] / k), int(ov.shape[0] / k))
        if anchor is None:
            return
        fade = self._fade()
        if fade <= 0.02:
            return
        build = clamp01(self.t / self.BUILD)
        if self.dying is not None:
            build = 1.0
        cx, cy, sc = anchor
        c = (cx * k, cy * k)
        s = sc * k
        yaw = 0.10 * math.sin(self.t * 0.6) + clamp01((cx / (ov.shape[1] / k)) - 0.5) * 0.3
        rot = rot_matrix(yaw, -0.06 + 0.03 * math.sin(self.t * 0.8), 0.0)
        alpha = fade * (0.92 + 0.08 * math.sin(self.t * 17.0) * math.sin(self.t * 5.3))
        wire = (1.0 - smoothstep(build)) * 0.85
        render_mesh(ov, MODELS.suit_mesh(), c, s, rot, alpha=alpha * (0.35 + 0.65 * build),
                    wire=wire, cull=True)
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
        cx, cy, sc = self._anchor
        s = sc * 0.42
        c = (cx, cy - sc * 0.40)
        # the reactor reads as a solid disc, so give it the same wire-over-fill
        # treatment as the shooters: you see the far ring through the core
        hot = 0.85 + 0.55 * math.sin(t * 3.6) * math.sin(t * 7.3)
        pulse = (0.55 + 0.45 * hot)
        rot = rot_matrix(0.10 * math.sin(t * 0.6), -0.05, 0.0)
        render_glow(frame, MODELS.reactor_mesh(), c, s, rot,
                    alpha=fade * pulse, wire=0.30, hot=0.55 * fade,
                    cull=False, gain=0.85, glow=0.50)
        # a soft light halo so it reads as a light source, not a sticker
        cv2.circle(frame, (int(c[0]), int(c[1])), int(s * 1.6),
                   dim(HOLO_DEEP, 0.10 * fade * pulse), -1, cv2.LINE_AA)

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
                    wire=(1.0 - smoothstep(build)) * 0.85, cull=True)
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
