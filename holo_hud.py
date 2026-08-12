#!/usr/bin/env python3
"""holo_hud.py - the screen furniture: a Stark-pattern blue HUD.

Everything is drawn full-resolution into one scratch overlay and blended once,
so the readouts stay crisp while the holograms stay bloomed.  Blue only.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from holo3d import HOLO_BLUE, HOLO_CYAN, HOLO_DEEP, HOLO_DIM, HOLO_WHITE, clamp01, dim

FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.4, colour=HOLO_CYAN, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, colour, thick,
                cv2.LINE_AA)


def arc(img, c, r, a0, a1, colour, thick=2, squash=1.0):
    cv2.ellipse(img, (int(c[0]), int(c[1])), (int(r), int(r * squash)), 0.0,
                a0, a1, colour, thick, cv2.LINE_AA)


def brackets(img, c, r, colour, thick=1, gap=0.30):
    """Four corner brackets around a point - the 'locked on' look."""
    x, y = int(c[0]), int(c[1])
    r = int(r)
    d = int(r * gap)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cv2.line(img, (x + sx * r, y + sy * r), (x + sx * (r - d), y + sy * r),
                     colour, thick, cv2.LINE_AA)
            cv2.line(img, (x + sx * r, y + sy * r), (x + sx * r, y + sy * (r - d)),
                     colour, thick, cv2.LINE_AA)


class Hud:
    """Persistent HUD state (rolling telemetry traces)."""

    def __init__(self):
        self._wave = np.zeros(96)
        self._rng = np.random.default_rng(5)
        self._meters = self._rng.uniform(0.0, 2 * math.pi, 4)

    # -- public -------------------------------------------------------------- #
    def draw(self, frame, ctx: dict) -> None:
        # drawn straight onto the frame: the HUD is thin geometry, so a
        # full-resolution scratch buffer + blend would cost more than it adds
        h, w = frame.shape[:2]
        ov = frame
        t = float(ctx.get("t", 0.0))
        self._frame_furniture(ov, w, h, t)
        self._top_bar(ov, w, h, t, ctx)
        self._left_rail(ov, w, h, t, ctx)
        self._radar(ov, w, h, t, ctx)
        self._bottom(ov, w, h, t, ctx)
        self._hands(ov, w, h, t, ctx)
        self._drags(ov, t, ctx)
        if ctx.get("debug"):
            self._debug(ov, w, h, ctx)
        self._scanline(ov, w, h, t)

    # -- pieces -------------------------------------------------------------- #
    def _frame_furniture(self, ov, w, h, t):
        col = dim(HOLO_BLUE, 0.85)
        m = 14
        L = int(70 + 10 * math.sin(t * 1.4))
        for (x, y, sx, sy) in ((m, m, 1, 1), (w - m, m, -1, 1),
                               (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            cv2.line(ov, (x, y), (x + sx * L, y), col, 2, cv2.LINE_AA)
            cv2.line(ov, (x, y), (x, y + sy * L), col, 2, cv2.LINE_AA)
            cv2.line(ov, (x + sx * 8, y + sy * 8), (x + sx * 34, y + sy * 8),
                     dim(HOLO_CYAN, 0.5), 1, cv2.LINE_AA)
        # nothing is drawn across the middle of the frame: that is where the
        # user's face and the holograms live, and furniture there just reads
        # as clutter

    def _top_bar(self, ov, w, h, t, ctx):
        col = dim(HOLO_CYAN, 0.95)
        text(ov, "ARACHNID OS  //  HARD-LIGHT SUIT MK-V", (28, 40), 0.46, col)
        text(ov, "STARK-PATTERN HOLOGRAPHICS   BLUE CHANNEL ONLY", (28, 58), 0.34,
             dim(HOLO_BLUE, 0.9))
        cv2.line(ov, (28, 66), (28 + 420, 66), dim(HOLO_BLUE, 0.6), 1)
        run = int(t)
        stamp = f"T+{run // 60:02d}:{run % 60:02d}"
        fps = float(ctx.get("fps", 0.0))
        hands = int(ctx.get("hand_count", 0))
        right = f"{stamp}   {fps:4.1f} FPS   HANDS {hands}/2"
        text(ov, right, (w - 28 - 8 * len(right) * 0.62, 40), 0.42, col)
        # sweeping progress line under the title
        p = (t * 0.35) % 1.0
        x0 = w - 28 - 300
        cv2.line(ov, (x0, 52), (w - 28, 52), dim(HOLO_DIM, 0.9), 1)
        cv2.line(ov, (int(x0 + 300 * p) - 26, 52), (int(x0 + 300 * p), 52),
                 dim(HOLO_WHITE, 0.8), 2)

    def _left_rail(self, ov, w, h, t, ctx):
        names = ("PWR", "WEB", "SYNC", "TRK")
        vals = (0.72 + 0.2 * math.sin(t * 1.1 + self._meters[0]),
                0.55 + 0.3 * math.sin(t * 0.7 + self._meters[1]),
                0.80 + 0.15 * math.sin(t * 1.9 + self._meters[2]),
                clamp01(ctx.get("hand_count", 0) / 2.0) * 0.95 + 0.05)
        y0 = int(h * 0.34)
        for i, (n, v) in enumerate(zip(names, vals)):
            y = y0 + i * 30
            text(ov, n, (30, y + 5), 0.36, dim(HOLO_BLUE, 0.95))
            cv2.rectangle(ov, (72, y - 7), (72 + 120, y + 3), dim(HOLO_DIM, 0.9), 1)
            cv2.rectangle(ov, (73, y - 6), (int(73 + 118 * clamp01(v)), y + 2),
                          dim(HOLO_CYAN, 0.55), -1)
            for s in range(1, 4):                     # tick marks
                x = 72 + int(120 * s / 4)
                cv2.line(ov, (x, y - 7), (x, y - 4), dim(HOLO_BLUE, 0.9), 1)
        # vertical ladder with a travelling marker
        lx = 22
        cv2.line(ov, (lx, y0 - 40), (lx, y0 + 190), dim(HOLO_DIM, 0.9), 1)
        for i in range(12):
            y = y0 - 40 + i * 21
            cv2.line(ov, (lx, y), (lx + (7 if i % 3 else 13), y), dim(HOLO_BLUE, 0.8), 1)
        my = y0 - 40 + int(230 * (0.5 + 0.5 * math.sin(t * 0.8)))
        cv2.line(ov, (lx - 4, my), (lx + 16, my), dim(HOLO_WHITE, 0.9), 2, cv2.LINE_AA)

    def _radar(self, ov, w, h, t, ctx):
        cx, cy, r = w - 96, int(h * 0.40), 60
        for k in (1.0, 0.66, 0.33):
            cv2.circle(ov, (cx, cy), int(r * k), dim(HOLO_BLUE, 0.55), 1, cv2.LINE_AA)
        cv2.line(ov, (cx - r, cy), (cx + r, cy), dim(HOLO_DIM, 0.9), 1)
        cv2.line(ov, (cx, cy - r), (cx, cy + r), dim(HOLO_DIM, 0.9), 1)
        a = (t * 1.6) % (2 * math.pi)
        for i in range(14):                            # fading sweep tail
            aa = a - i * 0.05
            k = 1.0 - i / 14.0
            cv2.line(ov, (cx, cy), (int(cx + math.cos(aa) * r), int(cy + math.sin(aa) * r)),
                     dim(HOLO_CYAN, 0.55 * k), 1, cv2.LINE_AA)
        for hnd in ctx.get("hands", []):
            px, py = hnd["palm"]
            bx = cx + (px - 0.5) * 2.0 * r * 0.85
            by = cy + (py - 0.5) * 2.0 * r * 0.85
            cv2.circle(ov, (int(bx), int(by)), 4, dim(HOLO_WHITE, 0.9), -1, cv2.LINE_AA)
            cv2.circle(ov, (int(bx), int(by)), 8, dim(HOLO_CYAN, 0.5), 1, cv2.LINE_AA)
        text(ov, "TRACK", (cx - 22, cy + r + 18), 0.34, dim(HOLO_BLUE, 0.9))

    def _bottom(self, ov, w, h, t, ctx):
        # rolling telemetry trace
        self._wave = np.roll(self._wave, -1)
        self._wave[-1] = (math.sin(t * 6.0) * 0.5 + math.sin(t * 2.3) * 0.3
                          + (0.6 if ctx.get("hand_count") else 0.0)
                          * math.sin(t * 17.0) * 0.25)
        bx, by, bw, bh = 28, h - 56, 260, 34
        cv2.rectangle(ov, (bx, by - bh // 2), (bx + bw, by + bh // 2),
                      dim(HOLO_DIM, 0.8), 1)
        pts = [(bx + int(i * bw / (len(self._wave) - 1)), int(by - v * bh * 0.45))
               for i, v in enumerate(self._wave)]
        cv2.polylines(ov, [np.array(pts, np.int32)], False, dim(HOLO_CYAN, 0.85), 1,
                      cv2.LINE_AA)
        text(ov, "GESTURE BUS", (bx, by - bh // 2 - 7), 0.32, dim(HOLO_BLUE, 0.9))
        # status chips
        chips = ctx.get("chips", [])
        x = bx + bw + 24
        for label, lit, prog in chips:
            wlab = int(9.2 * len(label)) + 18
            c = dim(HOLO_CYAN, 0.95) if lit else dim(HOLO_DEEP, 0.75)
            cv2.rectangle(ov, (x, by - 14), (x + wlab, by + 10), c, 1, cv2.LINE_AA)
            text(ov, label, (x + 9, by + 4), 0.36, c)
            if lit:
                cv2.line(ov, (x + 2, by + 10), (x + wlab - 2, by + 10),
                         dim(HOLO_WHITE, 0.55), 1)
            if prog > 0.001:
                cv2.rectangle(ov, (x, by + 13), (x + int(wlab * clamp01(prog)), by + 16),
                              dim(HOLO_WHITE, 0.85), -1)
            x += wlab + 12
        legend = ("PINCH+PULL = SHOOTER ON OTHER WRIST   |   FIST = BLUEPRINT   |   "
                  "BOTH FISTS 1s = SUIT   |   HOLD GRAB 4s = MOVE")
        text(ov, legend, (28, h - 18), 0.34, dim(HOLO_BLUE, 0.85))

    def _hands(self, ov, w, h, t, ctx):
        for hnd in ctx.get("hands", []):
            px = hnd["palm"][0] * w
            py = hnd["palm"][1] * h
            r = max(38.0, hnd.get("size", 0.1) * h * 0.75)
            spin = math.degrees(t * 1.1) * (1 if hnd["side"] == "R" else -1)
            for off in (0, 120, 240):
                cv2.ellipse(ov, (int(px), int(py)), (int(r), int(r)), spin + off,
                            0, 74, dim(HOLO_BLUE, 0.7), 1, cv2.LINE_AA)
            brackets(ov, (px, py), r * 1.35, dim(HOLO_CYAN, 0.6))
            text(ov, hnd["side"], (px - r * 1.35, py - r * 1.35 - 6), 0.4,
                 dim(HOLO_CYAN, 0.95))
            # gesture charge arcs: fist (inner) and pinch-pull (outer)
            if hnd.get("fist_p", 0.0) > 0.02:
                arc(ov, (px, py), r * 1.05, -90, -90 + 360 * clamp01(hnd["fist_p"]),
                    dim(HOLO_WHITE, 0.9), 2)
            if hnd.get("pull_p", 0.0) > 0.02:
                arc(ov, (px, py), r * 1.5, 90, 90 + 360 * clamp01(hnd["pull_p"]),
                    dim(HOLO_CYAN, 0.9), 2)
            if hnd.get("pinch"):
                cv2.circle(ov, (int(px), int(py)), 5, dim(HOLO_WHITE, 0.95), -1,
                           cv2.LINE_AA)
        gear_p = float(ctx.get("gear_p", 0.0))
        if gear_p > 0.02:
            cx, cy = w // 2, int(h * 0.16)
            arc(ov, (cx, cy), 34, -90, -90 + 360 * clamp01(gear_p),
                dim(HOLO_WHITE, 0.95), 3)
            cv2.circle(ov, (cx, cy), 34, dim(HOLO_BLUE, 0.6), 1, cv2.LINE_AA)
            text(ov, "SUIT", (cx - 19, cy + 5), 0.4, dim(HOLO_CYAN, 0.95))

    def _drags(self, ov, t, ctx):
        for d in ctx.get("drags", []):
            px, py = d["pos"]
            r = d["radius"]
            p = clamp01(d["progress"])
            if not d["armed"]:
                cv2.circle(ov, (int(px), int(py)), int(r), dim(HOLO_BLUE, 0.45), 1,
                           cv2.LINE_AA)
                arc(ov, (px, py), r, -90, -90 + 360 * p, dim(HOLO_CYAN, 0.95), 3)
                for i in range(12):               # countdown ticks
                    a = -math.pi / 2 + i * math.pi / 6
                    lit = (i / 12.0) < p
                    r0, r1 = r * 1.06, r * (1.20 if lit else 1.12)
                    cv2.line(ov, (int(px + math.cos(a) * r0), int(py + math.sin(a) * r0)),
                             (int(px + math.cos(a) * r1), int(py + math.sin(a) * r1)),
                             dim(HOLO_WHITE if lit else HOLO_DIM, 0.9), 1, cv2.LINE_AA)
                text(ov, f"LOCK {p * 100:3.0f}%", (px - r, py - r - 10), 0.36,
                     dim(HOLO_CYAN, 0.9))
            else:
                age = t - d["armed_t"]
                if age < 0.5:                     # READY: rings snap inward
                    f = age / 0.5
                    cv2.circle(ov, (int(px), int(py)), int(r * (2.0 - 1.0 * f)),
                               dim(HOLO_WHITE, 0.9 * (1.0 - f)), 2, cv2.LINE_AA)
                    brackets(ov, (px, py), r * (1.9 - 0.6 * f), dim(HOLO_WHITE, 0.95), 2)
                else:
                    brackets(ov, (px, py), r * 1.3, dim(HOLO_WHITE, 0.9), 2)
                pulse = 0.6 + 0.4 * math.sin(t * 9.0)
                cv2.circle(ov, (int(px), int(py)), int(r), dim(HOLO_WHITE, pulse), 1,
                           cv2.LINE_AA)
                text(ov, "MOVE LOCK", (px - r, py - r - 10), 0.38,
                     dim(HOLO_WHITE, 0.95))

    def _debug(self, ov, w, h, ctx):
        """Raw gesture numbers, so a gesture that will not fire can be seen.

        Green is never used (blue only): a satisfied test is drawn hot white,
        an unsatisfied one stays dim blue.
        """
        th = ctx.get("thresholds", {})
        grab_on = float(th.get("grab_on", 0.30))
        fist_curl = float(th.get("fist_curl", 0.62))
        x, y = 28, int(h * 0.60)
        text(ov, "GESTURE DEBUG  [D]", (x, y), 0.38, dim(HOLO_WHITE, 0.95))
        text(ov, f"fist: finger curl < {fist_curl:.2f}   pinch: dist < {grab_on:.2f}",
             (x, y + 16), 0.32, dim(HOLO_BLUE, 0.9))
        y += 38
        if not ctx.get("hands"):
            text(ov, "no hands tracked", (x, y), 0.34, dim(HOLO_DEEP, 0.95))
            return
        for hnd in ctx.get("hands", []):
            curls = hnd.get("curls", [])
            text(ov, f"{hnd['side']}  I M R P T", (x, y), 0.34, dim(HOLO_CYAN, 0.95))
            for i, v in enumerate(curls):
                lit = v < fist_curl and i < 4
                text(ov, f"{v:.2f}", (x + 54 + i * 40, y), 0.34,
                     dim(HOLO_WHITE if lit else HOLO_BLUE, 0.95))
            pd = hnd.get("pinch3", 1.0)
            text(ov, f"pinch {pd:.2f}", (x + 224, y), 0.34,
                 dim(HOLO_WHITE if pd < grab_on else HOLO_BLUE, 0.95))
            flags = []
            if hnd.get("fist"):
                flags.append("FIST")
            if hnd.get("pinch"):
                flags.append("PINCH")
            text(ov, " ".join(flags) or "-", (x + 320, y), 0.34,
                 dim(HOLO_WHITE if flags else HOLO_DEEP, 0.95))
            y += 22

    def _scanline(self, ov, w, h, t):
        """A soft blue band sweeping down the frame - blended, never opaque."""
        y = int((t * 0.22 % 1.0) * (h - 4))
        band = ov[y:y + 3]
        if band.size:
            tint = np.empty_like(band)
            tint[:] = dim(HOLO_BLUE, 0.55)
            cv2.addWeighted(band, 0.82, tint, 0.35, 0, band)
