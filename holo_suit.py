#!/usr/bin/env python3
"""holo_suit.py - the full holographic Spider-Man kit (no mask, ever).

Both fists held for one second materialise the whole rig: torso shell with the
spider emblem and webbing, shoulder plates, belt, and an armoured gauntlet on
every visible wrist.  The head is deliberately never drawn.

The torso is anchored from the visible hands: hand size gives the scale and the
hand midpoint nudges the body left/right, with heavy smoothing so the suit sits
still while the hands move.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import holo_models as MODELS
from holo3d import (HOLO_BLUE, HOLO_CYAN, HOLO_WHITE, clamp01, dim, project,
                    render_glow, render_mesh, rot_matrix, smoothstep)
from holo_objects import hand_frame, _mount_rot


class SuitGear:
    """The whole suit: torso + emblem + webbing + shoulders + belt + gauntlets."""

    BUILD = 1.30                      # materialise sweep, seconds

    def __init__(self, t: float = 0.0):
        self.t = 0.0
        self.born = 0.0
        self.dying = None
        self._anchor = None           # smoothed (cx, cy, scale)

    def alive(self) -> bool:
        return self.dying is None or (self.t - self.dying) < 0.55

    def dismiss(self) -> None:
        if self.dying is None:
            self.dying = self.t

    def update(self, dt: float) -> None:
        self.t += dt

    # -- anchoring ----------------------------------------------------------- #
    def _resolve(self, feats, w: int, h: int):
        """Where the torso sits.  Deliberately sluggish: the suit is worn, so
        it must sit still on the body while the hands move freely.

        Hands only *suggest* the anchor - the body is assumed to be centred in
        frame with the chest low, and the hand midpoint is allowed to lean it
        a little.  Once resolved, the anchor is kept even when both hands leave
        the frame, so the suit never blinks out mid-pose.
        """
        frames = [fr for fr in (hand_frame(f, w, h) for f in feats) if fr is not None]
        if not frames:
            return self._anchor, []          # hands gone: keep wearing the suit
        lens = float(np.mean([fr["L"] for fr in frames]))
        mx = float(np.mean([fr["W"][0] for fr in frames]))
        my = float(np.mean([fr["W"][1] for fr in frames]))
        cx = 0.5 * w + (mx - 0.5 * w) * 0.20
        cy = min(0.95 * h, max(0.55 * h, my + 2.3 * lens))
        sc = max(70.0, min(0.34 * h, 2.9 * lens))
        if self._anchor is None:
            self._anchor = [cx, cy, sc]
        else:
            a = 0.06
            for i, tgt in enumerate((cx, cy, sc)):
                d = tgt - self._anchor[i]
                if abs(d) > 3.0:             # deadband kills landmark jitter
                    self._anchor[i] += d * a
        return self._anchor, frames

    def _fade(self) -> float:
        if self.dying is None:
            return 1.0
        return 1.0 - clamp01((self.t - self.dying) / 0.55)

    # -- render -------------------------------------------------------------- #
    def _build(self) -> float:
        return 1.0 if self.dying is not None else clamp01(self.t / self.BUILD)

    def draw_sharp(self, frame, t: float, feats) -> None:
        """Wrist gauntlets at full resolution, after the overlay composite.

        Same reason as the web-shooters: at half resolution a forearm-sized
        model is only a few dozen pixels wide and loses all of its plating.
        """
        fade = self._fade()
        if fade <= 0.02:
            return
        h, w = frame.shape[:2]
        alpha = fade * (0.92 + 0.08 * math.sin(self.t * 17.0) * math.sin(self.t * 5.3))
        build = self._build()
        for f in feats:
            fr = hand_frame(f, w, h)
            if fr is None:
                continue
            arm = {"theta": math.atan2(-fr["ax"][1], -fr["ax"][0]),
                   "side": fr["side"], "L": fr["L"]}
            rot = _mount_rot(arm, wobble=0.04 * math.sin(self.t * 1.4))
            c = (fr["W"][0] - fr["ax"][0] * 0.12 * fr["L"],
                 fr["W"][1] - fr["ax"][1] * 0.12 * fr["L"])
            render_glow(frame, MODELS.gauntlet_mesh(), c, 0.52 * fr["L"], rot,
                        alpha=alpha * (0.4 + 0.6 * build),
                        wire=(1.0 - smoothstep(build)) * 0.85, cull=True,
                        gain=0.90, glow=0.5)

    def draw(self, ov, k: float, t: float, feats) -> None:
        anchor, frames = self._resolve(feats, int(ov.shape[1] / k), int(ov.shape[0] / k))
        if anchor is None:
            return
        fade = self._fade()
        if fade <= 0.02:
            return
        build = self._build()
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
            self._gauntlet(ov, fr, k, build, alpha)

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

    def _gauntlet(self, ov, fr, k: float, build: float, alpha: float) -> None:
        """The wrist's materialise ring (the armour itself is drawn sharp)."""
        if build >= 1.0:
            return
        c = ((fr["W"][0] - fr["ax"][0] * 0.12 * fr["L"]) * k,
             (fr["W"][1] - fr["ax"][1] * 0.12 * fr["L"]) * k)
        s = 0.52 * fr["L"] * k
        rr = (0.7 + 2.0 * build) * s
        cv2.ellipse(ov, (int(c[0]), int(c[1])), (int(rr), int(rr * 0.42)),
                    math.degrees(math.atan2(-fr["ax"][1], -fr["ax"][0])), 0, 360,
                    dim(HOLO_CYAN, alpha * (1.0 - build)), 2, cv2.LINE_AA)
