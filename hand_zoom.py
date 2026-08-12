#!/usr/bin/env python3
"""hand_zoom.py - real-time hand tracking for a hologram studio.

This version is the TRACKING CORE: a MediaPipe hand-landmarker pipeline with
an aggressive camera setup (720p MJPEG + fallback ladder) and per-hand
geometry features (finger curls, pinch, fist, spread) that later commits turn
into holograms and gestures.

Recognition is tuned to be as forgiving as possible:
  * MediaPipe confidence thresholds at the practical minimum
    (0.10 / 0.10 / 0.20) so hands are picked up in weak light, at the frame
    edges and at distance
  * the camera is pushed to 1280x720 (MJPEG + low-latency buffers when the
    driver supports them) and falls back to lower resolutions if refused
  * both hands are tracked and labelled right/left via MediaPipe handedness
    (mirrored/selfie input), with a mirrored-frame position fallback

Usage
-----
    python hand_zoom.py             # show your hands to the camera
    python hand_zoom.py --camera 1  # force a specific webcam
    python hand_zoom.py --record    # save the preview to recordings/
    python hand_zoom.py --selftest  # headless self-test (no camera needed)

Keys while running
------------------
    Esc / Q   quit
    V         start / stop recording the preview
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp  # heavy import; handled later with clear errors
except ImportError:  # pragma: no cover - exercised when mediapipe is missing
    mp = None

# --------------------------------------------------------------------------- #
# Paths, model, palette
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
SAMPLE_IMAGE_URL = (
    "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg"
)

# 21-point MediaPipe hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # wrist to pinky base
]

# hologram palette (BGR) - Stark cyan/blue with Spidey orange accents
HOLO_CYAN = (235, 255, 150)      # bright cyan-white
HOLO_BLUE = (255, 170, 60)       # deep blue
HOLO_ORANGE = (55, 150, 255)     # web-fluid orange
HOLO_DIM = (120, 160, 80)        # faded cyan for depth layers

WRIST, MIDDLE_MCP = 0, 9
INDEX_TIP, MIDDLE_TIP = 8, 12
THUMB_TIP = 4

# finger joint index groups (mcp, pip, dip, tip)
FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# --------------------------------------------------------------------------- #
# Model handling
# --------------------------------------------------------------------------- #
def ensure_model() -> Path:
    """Download the MediaPipe hand-landmarker model on first use."""
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".tmp")
    print("[model] first run: downloading hand_landmarker.task ...")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        os.replace(tmp, MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - surface any download problem
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download the hand-tracking model:\n  {exc}\n\n"
            f"Please download it manually from\n  {MODEL_URL}\n"
            f"and save it to\n  {MODEL_PATH}"
        ) from exc
    print(f"[model] saved to {MODEL_PATH}")
    return MODEL_PATH


def build_landmarker(model_path: Path):
    """Create a MediaPipe Tasks HandLandmarker in VIDEO running mode."""
    if mp is None:
        raise SystemExit(
            "mediapipe is not installed.\n"
            "Run:  .venv\\Scripts\\python -m pip install -r requirements.txt\n"
            "(or just double-click run.bat)"
        )
    try:
        tasks = mp.tasks
    except AttributeError:  # pragma: no cover - mediapipe 1.x layout safety
        from mediapipe import tasks  # noqa: PLC0415
    base = tasks.BaseOptions(model_asset_path=str(model_path))
    options = tasks.vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        # minimum practical confidence: hands are recognised in weak light,
        # at the frame edges and at distance.
        min_hand_detection_confidence=0.10,
        min_hand_presence_confidence=0.10,
        min_tracking_confidence=0.20,
    )
    return tasks.vision.HandLandmarker.create_from_options(options)


def detect_frame(landmarker, frame_bgr: np.ndarray, timestamp_ms: int):
    """Run the landmarker on one BGR frame; returns raw result object."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


# --------------------------------------------------------------------------- #
# Hand geometry + per-hand features
# --------------------------------------------------------------------------- #
def _pt(landmarks, i: int) -> np.ndarray:
    return np.array([landmarks[i].x, landmarks[i].y], dtype=np.float64)


def hand_scale(landmarks) -> float:
    """Hand size in NORMALISED units: wrist <-> middle-finger MCP distance."""
    return float(np.linalg.norm(_pt(landmarks, WRIST) - _pt(landmarks, MIDDLE_MCP)))


def palm_center(landmarks) -> np.ndarray:
    return 0.5 * (_pt(landmarks, WRIST) + _pt(landmarks, MIDDLE_MCP))


def curl_features(landmarks) -> dict:
    """Straightness in [0..1] for each finger (1 = fully straight).

    Fingers: ratio of the actual mcp->tip distance to the fully-extended
    chain length.  Thumb: same chain metric over its mcp->pip->tip joints.
    """
    curls: dict = {}
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        d_ext = (np.linalg.norm(_pt(landmarks, mcp) - _pt(landmarks, pip))
                 + np.linalg.norm(_pt(landmarks, pip) - _pt(landmarks, dip))
                 + np.linalg.norm(_pt(landmarks, dip) - _pt(landmarks, tip)))
        d_act = float(np.linalg.norm(_pt(landmarks, mcp) - _pt(landmarks, tip)))
        curls[name] = clamp01(d_act / (d_ext + 1e-6))
    t_ext = (np.linalg.norm(_pt(landmarks, 2) - _pt(landmarks, 3))
             + np.linalg.norm(_pt(landmarks, 3) - _pt(landmarks, 4)))
    t_act = float(np.linalg.norm(_pt(landmarks, 2) - _pt(landmarks, 4)))
    curls["thumb"] = clamp01(t_act / (t_ext + 1e-6))  # same chain metric as fingers
    return curls


def hand_features(landmarks, is_right: bool) -> dict:
    """Everything the gesture engine + renderers need for one hand."""
    scale = hand_scale(landmarks) + 1e-6
    curls = curl_features(landmarks)
    tt = _pt(landmarks, THUMB_TIP)
    it = _pt(landmarks, INDEX_TIP)
    mt = _pt(landmarks, MIDDLE_TIP)
    pinches = (float(np.linalg.norm(tt - it)) + float(np.linalg.norm(tt - mt))) / 2.0
    tips = [_pt(landmarks, i) for i in (INDEX_TIP, MIDDLE_TIP, 12, 16, 20)]
    spread = float(np.mean([np.linalg.norm(tips[i] - tips[i + 1]) for i in range(4)]))
    return {
        "landmarks": landmarks,
        "is_right": bool(is_right),
        "curls": curls,
        "fist": (curls["index"] < 0.55 and curls["middle"] < 0.55
                 and curls["ring"] < 0.55 and curls["pinky"] < 0.55
                 and curls["thumb"] < 0.75),
        "open": (curls["index"] > 0.72 and curls["middle"] > 0.60
                 and curls["ring"] > 0.60),
        "pinch3": pinches / scale,          # thumb<->index/middle distance ratio
        "spread": spread / scale,
        "palm": palm_center(landmarks),
        "wrist": _pt(landmarks, WRIST),
        "mcp9": _pt(landmarks, MIDDLE_MCP),
        "index_tip": it,
        "middle_tip": mt,
        "thumb_tip": tt,
    }


def classify_hands(result) -> list:
    """Turn a detection result into a list of per-hand feature dicts.

    Every hand gets an 'is_right' flag: MediaPipe's handedness when present
    (mirrored/selfie input, which is exactly what the webcam pipeline feeds
    it); otherwise the mirrored-frame position fallback (rightmost = the
    user's right hand).
    """
    hands = list(result.hand_landmarks or []) if result is not None else []
    if not hands:
        return []
    try:
        hd = list(result.handedness or [])
    except Exception:  # noqa: BLE001 - varies across mediapipe versions
        hd = []
    labels = []
    for i in range(len(hands)):
        name = ""
        if i < len(hd) and hd[i]:
            cat = hd[i][0]
            name = str(getattr(cat, "category_name", "") or "")
        labels.append(name)
    if not any(labels):
        order = sorted(range(len(hands)), key=lambda i: -hands[i][0].x)
        labels[order[0]] = "Right"
        labels[order[-1]] = "Left"
    return [hand_features(h, labels[i] == "Right") for i, h in enumerate(hands)]


def _dim(color, k):
    return (int(color[0] * k), int(color[1] * k), int(color[2] * k))


def _rot2(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def draw_hand_holo(frame, hand, alpha: float = 0.55) -> None:
    """Draw the hand as a translucent cyan hologram skeleton."""
    h, w = frame.shape[:2]
    pts = [(lm.x * w, lm.y * h) for lm in hand]
    overlay = np.zeros_like(frame)
    for a, b in HAND_CONNECTIONS:
        pa = (int(pts[a][0]), int(pts[a][1]))
        pb = (int(pts[b][0]), int(pts[b][1]))
        cv2.line(overlay, pa, pb, _dim(HOLO_CYAN, 0.30), 3, cv2.LINE_AA)
        cv2.line(overlay, pa, pb, _dim(HOLO_CYAN, 0.75), 1, cv2.LINE_AA)
    for i, p in enumerate(pts):
        c = (int(p[0]), int(p[1]))
        if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, 16, 20):  # fingertips glow
            cv2.circle(overlay, c, 5, _dim(HOLO_CYAN, 0.9), -1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, c, 3, _dim(HOLO_CYAN, 0.6), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)


# --------------------------------------------------------------------------- #
# Gesture tracker: raw hand features -> discrete gesture events
# --------------------------------------------------------------------------- #
class GestureTracker:
    """Turns smoothed per-hand features into gesture events.

    Events (tuples):
      ('spawn', feat)    a single fist held ~0.45 s then opened
                         -> show a holographic gadget blueprint
      ('grab', feat)     the RIGHT hand begins pinching
                         -> detach the web-shooter / call it back
      ('release', feat)  the RIGHT hand stops pinching
                         -> let the web-shooter float where it is
      ('gear', None)     BOTH fists held ~0.6 s then opened
                         -> toggle the holographic body gear

    Per-hand features are EMA-smoothed (with matching by palm proximity), and
    hands that drop out for a few frames keep their state ("ghosting") so a
    tracking hiccup never cancels a gesture mid-way.
    """

    FIST_HOLD = 0.45        # seconds a single fist must be held before open
    GEAR_HOLD = 0.60        # seconds both fists must be held before open
    GRAB_ON = 0.20          # pinch3 below this -> pinch begins
    GRAB_OFF = 0.30         # pinch3 above this -> pinch ends
    PINCH_CONFIRM = 2       # frames of confirmation before a pinch edge
    GHOST_TTL = 0.18        # seconds a dropped hand keeps its state

    def __init__(self):
        self._slots: list = []
        self._both_fist_since: float | None = None

    # -- public ------------------------------------------------------------- #
    def feed(self, hands: list, t: float) -> list:
        events: list = []
        for s in self._slots:
            s["matched"] = False
            s["suppress"] = False
        for h in hands:
            s = self._match(h)
            if s is None:
                s = self._new_slot(h, t)
                self._slots.append(s)
            else:
                s["matched"] = True
                s["last_seen"] = t
                self._apply(s, h)
        # prune hands that have been gone too long
        for s in [x for x in self._slots
                  if not x["matched"] and t - x["last_seen"] > self.GHOST_TTL]:
            self._slots.remove(s)
        # -- both-fists -> gear (ghosts keep a held fist "present" briefly) --
        fists = [s for s in self._slots
                 if s["fist"] and t - s["last_seen"] <= self.GHOST_TTL]
        if len(fists) >= 2:
            if self._both_fist_since is None:
                self._both_fist_since = t
        elif self._both_fist_since is not None:
            held = t - self._both_fist_since
            self._both_fist_since = None
            if held >= self.GEAR_HOLD:
                events.append(("gear", None))
                for s in self._slots:      # consume the whole fist cycle so
                    s["spawn_fired"] = True  # the opener doesn't also spawn
                    s["suppress"] = True
        # -- per-slot gesture edges --
        for s in self._slots:
            if s["matched"]:
                self._pinch_edges(s, t, events)
                self._fist_events(s, t, events)
        return events

    # -- internals ----------------------------------------------------------- #
    def _match(self, h):
        best, bd = None, 0.5        # max normalised match distance
        for s in self._slots:
            if s["matched"]:
                continue
            d = float(np.linalg.norm(s["palm"] - h["palm"]))
            if d < bd:
                bd, best = d, s
        return best

    def _new_slot(self, h, t):
        s = {
            "palm": h["palm"], "wrist": h["wrist"], "mcp9": h["mcp9"],
            "index_tip": h["index_tip"], "middle_tip": h["middle_tip"],
            "thumb_tip": h["thumb_tip"], "is_right": h["is_right"],
            "landmarks": h["landmarks"], "hand": h,
            "curls": None, "pinch3": 0.5, "spread": 0.5, "fist": False,
            "matched": True, "last_seen": t,
            "fist_since": None, "spawn_fired": False,
            "pinch": False, "pinch_frames": 0, "suppress": False,
        }
        self._apply(s, h)
        return s

    def _apply(self, s, h):
        if s["curls"] is None:
            s["curls"] = dict(h["curls"])
            s["pinch3"] = h["pinch3"]
            s["spread"] = h["spread"]
        else:
            a = 0.45
            for k in ("thumb", "index", "middle", "ring", "pinky"):
                s["curls"][k] = a * h["curls"][k] + (1.0 - a) * s["curls"][k]
            s["pinch3"] = 0.40 * h["pinch3"] + 0.60 * s["pinch3"]
            s["spread"] = 0.40 * h["spread"] + 0.60 * s["spread"]
        s["palm"] = h["palm"]
        s["wrist"] = h["wrist"]
        s["mcp9"] = h["mcp9"]
        s["index_tip"] = h["index_tip"]
        s["middle_tip"] = h["middle_tip"]
        s["thumb_tip"] = h["thumb_tip"]
        s["is_right"] = h["is_right"]
        s["landmarks"] = h["landmarks"]
        s["hand"] = h
        c = s["curls"]
        s["fist"] = (c["index"] < 0.55 and c["middle"] < 0.55
                     and c["ring"] < 0.55 and c["pinky"] < 0.55
                     and c["thumb"] < 0.75)

    def _pinch_edges(self, s, t, events):
        p = s["pinch3"] < self.GRAB_ON
        s["pinch_frames"] = min(s["pinch_frames"] + 1, 6) if p \
            else max(s["pinch_frames"] - 1, 0)
        if p and s["pinch_frames"] >= self.PINCH_CONFIRM and not s["pinch"]:
            s["pinch"] = True
            if s["is_right"]:
                events.append(("grab", s["hand"]))
        elif not p and s["pinch_frames"] == 0 and s["pinch"]:
            s["pinch"] = False
            if s["is_right"]:
                events.append(("release", s["hand"]))

    def _fist_events(self, s, t, events):
        if not s["matched"]:
            return
        if s["fist"]:
            if s["fist_since"] is None:
                s["fist_since"] = t
                s["spawn_fired"] = False      # new fist cycle
        elif s["fist_since"] is not None:
            held = t - s["fist_since"]
            s["fist_since"] = None
            if (held >= self.FIST_HOLD and not s["spawn_fired"]
                    and not s["suppress"] and s["curls"]["index"] > 0.55):
                events.append(("spawn", s["hand"]))
                s["spawn_fired"] = True


# --------------------------------------------------------------------------- #
# The holographic web-shooter (detachable!)
# --------------------------------------------------------------------------- #
class WebShooterHologram:
    """Tony-Stark-style holographic web-shooter with a grab/detach lifecycle.

    States
    ------  on        bolted to the wrist, follows the hand
            detach    animation lifting it off the wrist into a pinch
            held      floating above the pinching hand, following it
            float     parked in mid-air (gentle bob + slow drift)
            reattach  streak animation flying it back onto the wrist

    The whole assembly is drawn from an "orientation" (anchor, aim direction,
    hand axes) that is refreshed from the live hand while 'on'/'held' and
    frozen while floating, so the hologram stays coherent when airborne.
    """

    DETACH_DUR = 0.50
    REATTACH_DUR = 0.42

    def __init__(self):
        self._rng = np.random.default_rng(7)
        self._orbits = self._rng.uniform(0, 2 * math.pi, 9)
        self._bars = self._rng.uniform(0, 2 * math.pi, 5)
        self.state = "on"
        self._state_t = 0.0
        self._geo = None            # frozen orientation bundle
        self._pos = None            # current pixel anchor
        self._ang = 0.0
        self._scale = 1.0
        self._R = 60.0
        self._ax = (0.0, -1.0)
        self._side = (1.0, 0.0)
        self._aim_u = (1.0, 0.0)
        self._aim_len = 60.0
        self._from_px = None
        self._hold_target = None
        self._pinch_px = None
        self._hold_off = (0.0, 0.0)
        self._float_px = None
        self._burst = None          # (t0, pos, R) shockwave flare
        self._trail = []            # reattach streak ghosts

    # -- geometry ------------------------------------------------------------ #
    @staticmethod
    def _ell_pt(cx, cy, a, b, ang, phi):
        """Point on a rotated ellipse (ang, phi in radians)."""
        c, s = math.cos(ang), math.sin(ang)
        x = a * math.cos(phi) * c - b * math.sin(phi) * s
        y = a * math.cos(phi) * s + b * math.sin(phi) * c
        return (cx + x, cy + y)

    def _orient_from_feat(self, feat, w, h):
        W = feat["wrist"] * np.array([w, h])
        M = feat["mcp9"] * np.array([w, h])
        T = feat["thumb_tip"] * np.array([w, h])
        L = float(np.linalg.norm(M - W))
        if L < 12:
            return None
        ax = ((M[0] - W[0]) / L, (M[1] - W[1]) / L)      # wrist -> fingers
        side = (-ax[1], ax[0])
        if side[0] * (T[0] - W[0]) + side[1] * (T[1] - W[1]) < 0:
            side = (-side[0], -side[1])                  # toward the thumb
        C = (W[0] + ax[0] * 0.34 * L, W[1] + ax[1] * 0.34 * L)
        I = feat["index_tip"] * np.array([w, h])
        aim_len = float(np.linalg.norm(I - C)) or 1.0
        aim_u = ((I[0] - C[0]) / aim_len, (I[1] - C[1]) / aim_len)
        return {"W": W, "M": M, "L": L, "ax": ax, "side": side,
                "C": C, "R": 0.52 * L, "ang": math.atan2(side[1], side[0]),
                "I": I, "aim_u": aim_u, "aim_len": aim_len}

    def _refresh_orient(self, feat, w, h):
        g = self._orient_from_feat(feat, w, h)
        if g is not None:
            self._geo = g
            self._R = g["R"]
            self._ax = g["ax"]
            self._side = g["side"]
            self._aim_u = g["aim_u"]
            self._aim_len = g["aim_len"]
            self._ang = g["ang"]

    # -- gesture entry points ------------------------------------------------ #
    def grab(self, feat, w, h, t):
        """Pinch: take the shooter off the wrist, or call a floating one back."""
        if self.state == "on":
            if self._geo is None:
                return
            self.state = "detach"
            self._state_t = 0.0
            self._burst = None
            self._from_px = self._geo["C"]
            pinch = feat["palm"] * np.array([w, h])
            self._pinch_px = (float(pinch[0]), float(pinch[1]))
            ax, side = self._ax, self._side
            self._hold_off = (ax[0] * 0.55 + side[0] * 0.25,
                              ax[1] * 0.55 + side[1] * 0.25)
            self._hold_off = (self._hold_off[0] * self._R,
                              self._hold_off[1] * self._R)
            self._hold_target = (self._pinch_px[0] + self._hold_off[0],
                                 self._pinch_px[1] + self._hold_off[1])
        elif self.state == "float":
            self.state = "reattach"
            self._state_t = 0.0
            self._float_px = (self._pos[0], self._pos[1]) if self._pos \
                else self._hold_target
            self._trail.clear()

    def release(self, feat, w, h, t):
        """Pinch released: the shooter stays floating where it is."""
        if self.state in ("held", "detach"):
            self.state = "float"
            self._state_t = 0.0
            self._float_px = (self._pos[0], self._pos[1]) if self._pos \
                else self._hold_target

    # -- per-frame update ---------------------------------------------------- #
    def update(self, dt, t, feat, w, h):
        self._state_t += dt
        if feat is not None:
            self._refresh_orient(feat, w, h)
        if self.state == "on":
            if self._geo is not None:
                self._pos = self._geo["C"]
                self._ang = self._geo["ang"]
                self._scale = 1.0
            else:
                self._pos = None          # hand gone -> shooter gone
        elif self.state == "detach":
            f = min(1.0, self._state_t / self.DETACH_DUR)
            e = 1.0 - (1.0 - f) ** 3      # ease-out cubic
            fx, fy = self._from_px
            tx, ty = self._hold_target
            self._pos = (fx + (tx - fx) * e, fy + (ty - fy) * e)
            self._scale = 1.0 + 0.12 * math.sin(f * math.pi)
            if f >= 0.35 and self._burst is None:
                self._burst = (t, self._hold_target, self._R)
            if f >= 1.0:
                self.state = "held"
                self._state_t = 0.0
                self._pos = self._hold_target
        elif self.state == "held":
            if feat is not None:
                self._refresh_orient(feat, w, h)
                pinch = feat["palm"] * np.array([w, h])
                self._pinch_px = (float(pinch[0]), float(pinch[1]))
            self._pos = (self._pinch_px[0] + self._hold_off[0],
                         self._pinch_px[1] + self._hold_off[1]
                         + 0.035 * self._R * math.sin(t * 2.2))
            self._scale = 1.05
        elif self.state == "float":
            self._pos = (self._float_px[0],
                         self._float_px[1] + 0.035 * self._R * math.sin(t * 1.7))
            self._ang = self._geo["ang"] + 0.12 * math.sin(t * 0.6) \
                if self._geo else self._ang
            self._scale = 1.0
        elif self.state == "reattach":
            f = min(1.0, self._state_t / self.REATTACH_DUR)
            e = f * f * (3.0 - 2.0 * f)   # smoothstep
            target = self._geo["C"] if self._geo is not None else self._float_px
            self._pos = (self._float_px[0] + (target[0] - self._float_px[0]) * e,
                         self._float_px[1] + (target[1] - self._float_px[1]) * e)
            self._scale = 1.0 - 0.10 * e
            self._trail.append((self._pos[0], self._pos[1], t))
            if len(self._trail) > 6:
                self._trail.pop(0)
            if f >= 1.0:
                self.state = "on"
                self._state_t = 0.0
                self._trail.clear()

    # -- rendering ----------------------------------------------------------- #
    def draw(self, frame, t):
        if self._pos is None:
            return
        w, h = frame.shape[1], frame.shape[0]
        C = self._pos
        R = self._R * self._scale
        if R < 8:
            return
        ang = self._ang + 0.06 * math.sin(t * 1.4)
        overlay = np.zeros_like(frame)
        self._draw_core(overlay, w, h, C, R, ang, t)
        self._draw_rings(overlay, C, R, ang, t)
        self._draw_threads(overlay, C, R, t)
        self._draw_cartridge(overlay, C, R, t)
        self._draw_panels(overlay, C, R, t)
        self._draw_particles(overlay, C, R, ang, t)
        self._draw_sweep(overlay, C, R, t)
        self._draw_state_fx(overlay, C, R, t)
        # hologram flicker: constant shimmer, never fully off
        alpha = 0.78 * (0.92 + 0.08 * math.sin(t * 21.3) * math.sin(t * 7.7))
        # hard-light glow via a HALF-RES blurred pass (cheap at 720p)
        sw, sh = max(1, w // 2), max(1, h // 2)
        glow = cv2.GaussianBlur(cv2.resize(overlay, (sw, sh)), (0, 0), 1.4)
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(glow, 0.9, frame, 1.0, 0, frame)
        cv2.addWeighted(overlay, max(alpha, 0.25), frame, 1.0, 0, frame)

    # -- layers -------------------------------------------------------------- #
    def _draw_core(self, overlay, w, h, C, R, ang, t):
        x0 = max(0, int(C[0] - 1.5 * R)); x1 = min(w, int(C[0] + 1.5 * R))
        y0 = max(0, int(C[1] - 1.5 * R)); y1 = min(h, int(C[1] + 1.5 * R))
        for yy in range(y0, y1, 6):
            cv2.line(overlay, (x0, yy), (x1, yy), _dim(HOLO_DIM, 0.22), 1)
        cv2.circle(overlay, (int(C[0]), int(C[1])), int(0.30 * R),
                   _dim(HOLO_BLUE, 0.30), -1, cv2.LINE_AA)
        # arc-reactor core: inner ring + rotating white sweep
        cv2.ellipse(overlay, (int(C[0]), int(C[1])), (int(0.20 * R), int(0.085 * R)),
                    math.degrees(ang), 0, 360, _dim(HOLO_CYAN, 0.9), 2, cv2.LINE_AA)
        phi = t * 2.2
        p1 = self._ell_pt(*C, 0.20 * R, 0.085 * R, ang, phi)
        p2 = self._ell_pt(*C, 0.20 * R, 0.085 * R, ang, phi + 0.9)
        cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 (255, 255, 255), 3, cv2.LINE_AA)
        for i in range(12):
            ph = t * 0.25 + i * 2 * math.pi / 12
            q1 = self._ell_pt(*C, 0.88 * R, 0.35 * R, ang, ph)
            q2 = self._ell_pt(*C, 1.04 * R, 0.42 * R, ang, ph)
            cv2.line(overlay, (int(q1[0]), int(q1[1])), (int(q2[0]), int(q2[1])),
                     _dim(HOLO_CYAN, 0.5), 1, cv2.LINE_AA)
        if (t * 0.5) % 1.0 < 0.12:      # occasional "data refresh" flash
            cv2.ellipse(overlay, (int(C[0]), int(C[1])), (int(0.20 * R), int(0.085 * R)),
                        math.degrees(ang), 0, 360, (255, 255, 255), 3, cv2.LINE_AA)

    def _draw_rings(self, overlay, C, R, ang, t):
        cx, cy = int(C[0]), int(C[1])
        for k, (scale, alpha_k) in enumerate(((0.72, 0.30), (1.0, 0.55), (1.15, 0.75))):
            cv2.ellipse(overlay, (cx, cy), (int(R * scale), int(0.40 * R * scale)),
                        math.degrees(ang), 0, 360, _dim(HOLO_CYAN, alpha_k),
                        2 if k < 2 else 1, cv2.LINE_AA)
        base = math.degrees(ang) + (t * 26.0) % 360
        for off in (0, 180):            # rotating dashed outer ring
            cv2.ellipse(overlay, (cx, cy), (int(R * 1.15), int(0.40 * R * 1.15)),
                        base + off, 0, 150, _dim(HOLO_CYAN, 0.85), 2, cv2.LINE_AA)
            cv2.ellipse(overlay, (cx, cy), (int(R * 1.15), int(0.40 * R * 1.15)),
                        base + off + 165, 0, 30, _dim(HOLO_BLUE, 0.4), 2, cv2.LINE_AA)
        for i in range(2):              # bright energy arcs sweeping the ring
            ph0 = t * 1.6 + i * math.pi
            pts = [self._ell_pt(*C, R, 0.40 * R, ang, ph0 + j * 0.05) for j in range(14)]
            cv2.polylines(overlay, [np.array(pts, np.int32)], False,
                          (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_threads(self, overlay, C, R, t):
        N = (C[0] + self._aim_u[0] * 0.55 * self._aim_len,
             C[1] + self._aim_u[1] * 0.55 * self._aim_len)
        for i, off in ((0, 0.0), (1, 0.55)):    # toward index + middle tips
            ux, uy = _rot2(self._aim_u, off)
            tip = (C[0] + ux * self._aim_len * 0.95,
                   C[1] + uy * self._aim_len * 0.95)
            k = 0.35 + 0.3 * math.sin(t * 3.0 + i * 2.1)
            cv2.line(overlay, (int(N[0]), int(N[1])), (int(tip[0]), int(tip[1])),
                     _dim(HOLO_CYAN, k), 2, cv2.LINE_AA)

    def _draw_cartridge(self, overlay, C, R, t):
        u = self._aim_u
        d = self._aim_len
        p = (-u[1], u[0])
        b0 = (C[0] + u[0] * 0.30 * R, C[1] + u[1] * 0.30 * R)
        b1 = (C[0] + u[0] * 0.85 * R, C[1] + u[1] * 0.85 * R)
        half = 0.085 * R
        quad = [(b0[0] + p[0] * half, b0[1] + p[1] * half),
                (b0[0] - p[0] * half, b0[1] - p[1] * half),
                (b1[0] - p[0] * half * 1.25, b1[1] - p[1] * half * 1.25),
                (b1[0] + p[0] * half * 1.25, b1[1] + p[1] * half * 1.25)]
        cv2.fillPoly(overlay, [np.array(quad, np.int32)], _dim(HOLO_BLUE, 0.30))
        cv2.polylines(overlay, [np.array(quad, np.int32)], True,
                      _dim(HOLO_CYAN, 0.85), 1, cv2.LINE_AA)
        lit = 1 + int(3 * (0.5 + 0.5 * math.sin(t * 1.2)))   # fluid gauge
        for i in range(4):
            f = 0.30 + (i + 0.5) * 0.14
            gx = b0[0] + u[0] * f * R + p[0] * half * 1.6
            gy = b0[1] + u[1] * f * R + p[1] * half * 1.6
            col = HOLO_ORANGE if i < lit else _dim(HOLO_ORANGE, 0.18)
            cv2.circle(overlay, (int(gx), int(gy)), max(2, int(0.022 * R)),
                       col, -1, cv2.LINE_AA)
        nz = (int(b1[0]), int(b1[1]))                      # nozzle
        cv2.circle(overlay, nz, int(0.075 * R), _dim(HOLO_BLUE, 0.5), 2, cv2.LINE_AA)
        cv2.circle(overlay, nz, int(0.045 * R), HOLO_ORANGE, -1, cv2.LINE_AA)
        cv2.circle(overlay, nz, max(2, int(0.018 * R)), (255, 255, 255), -1)
        tip = (C[0] + u[0] * d, C[1] + u[1] * d)           # projection cone
        cone = np.array([nz,
                         (int(tip[0] - p[0] * 0.09 * R), int(tip[1] - p[1] * 0.09 * R)),
                         (int(tip[0] + p[0] * 0.09 * R), int(tip[1] + p[1] * 0.09 * R))],
                        np.int32)
        cv2.fillPoly(overlay, [cone], _dim(HOLO_BLUE, 0.10))

    def _draw_panels(self, overlay, C, R, t):
        u = self._side
        v = (-self._ax[0], -self._ax[1])
        pa = (C[0] + u[0] * 1.32 * R, C[1] + u[1] * 1.32 * R)
        self._panel_a(overlay, pa, u, v, 0.62 * R, 0.34 * R, t)
        pb = (C[0] + u[0] * 1.36 * R, C[1] + u[1] * 1.36 * R - 0.5 * R)
        self._panel_b(overlay, pb, u, v, 0.42 * R, 0.30 * R, t)

    def _panel_a(self, overlay, c, u, v, wd, ht, t):
        corners = [(c[0] + u[0] * wd - v[0] * ht, c[1] + u[1] * wd - v[1] * ht),
                   (c[0] - u[0] * wd - v[0] * ht, c[1] - u[1] * wd - v[1] * ht),
                   (c[0] - u[0] * wd + v[0] * ht, c[1] - u[1] * wd + v[1] * ht),
                   (c[0] + u[0] * wd + v[0] * ht, c[1] + u[1] * wd + v[1] * ht)]
        quad = np.array(corners, np.int32)
        cv2.fillPoly(overlay, [quad], _dim(HOLO_BLUE, 0.16))
        cv2.polylines(overlay, [quad], True, _dim(HOLO_CYAN, 0.7), 1, cv2.LINE_AA)
        for i in range(5):                                  # animated data bars
            f = 0.20 + 0.60 * (0.5 + 0.5 * math.sin(t * 2.4 + self._bars[i]))
            bx = c[0] + u[0] * (wd * (i / 4.0 - 0.5) + 0.05 * wd)
            by = c[1] + u[1] * (wd * (i / 4.0 - 0.5) + 0.05 * wd)
            bh = 0.35 * ht * f
            cv2.line(overlay, (int(bx), int(by)), (int(bx + u[0] * 0.05 * wd),
                     int(by + u[1] * 0.05 * wd)), _dim(HOLO_CYAN, 0.85), 2)
            cv2.line(overlay, (int(bx + u[0] * 0.05 * wd), int(by + u[1] * 0.05 * wd)),
                     (int(bx + u[0] * 0.05 * wd + v[0] * bh),
                      int(by + u[1] * 0.05 * wd + v[1] * bh)),
                     _dim(HOLO_CYAN, 0.85), 1)
        pts = []                                            # waveform
        for i in range(24):
            f = i / 23.0
            sx = c[0] + u[0] * wd * (f - 0.5) - v[0] * ht * 0.55
            sy = c[1] + u[1] * wd * (f - 0.5) - v[1] * ht * 0.55
            sy += v[1] * 0.10 * ht * math.sin(f * 9.0 + t * 5.0)
            sx += u[0] * 0.10 * wd * math.sin(f * 9.0 + t * 5.0)
            pts.append((sx, sy))
        cv2.polylines(overlay, [np.array(pts, np.int32)], False,
                      _dim(HOLO_CYAN, 0.65), 1, cv2.LINE_AA)

    def _panel_b(self, overlay, c, u, v, wd, ht, t):
        corners = [(c[0] + u[0] * wd - v[0] * ht, c[1] + u[1] * wd - v[1] * ht),
                   (c[0] - u[0] * wd - v[0] * ht, c[1] - u[1] * wd - v[1] * ht),
                   (c[0] - u[0] * wd + v[0] * ht, c[1] - u[1] * wd + v[1] * ht),
                   (c[0] + u[0] * wd + v[0] * ht, c[1] + u[1] * wd + v[1] * ht)]
        quad = np.array(corners, np.int32)
        cv2.fillPoly(overlay, [quad], _dim(HOLO_BLUE, 0.14))
        cv2.polylines(overlay, [quad], True, _dim(HOLO_CYAN, 0.6), 1, cv2.LINE_AA)
        cc = (int(c[0]), int(c[1]))                         # gauge + needle
        cv2.ellipse(overlay, cc, (int(wd * 0.62), int(ht * 0.62)),
                    90, 200, 340, _dim(HOLO_CYAN, 0.7), 2, cv2.LINE_AA)
        th = t * 1.8
        nx = c[0] + u[0] * (0.5 * wd) * math.cos(th) - v[0] * (0.5 * ht) * math.sin(th)
        ny = c[1] + u[1] * (0.5 * wd) * math.cos(th) - v[1] * (0.5 * ht) * math.sin(th)
        cv2.line(overlay, cc, (int(nx), int(ny)), _dim(HOLO_ORANGE, 0.9), 2, cv2.LINE_AA)

    def _draw_particles(self, overlay, C, R, ang, t):
        for i, ph0 in enumerate(self._orbits):
            ph = ph0 + t * (0.5 + 0.06 * i)
            p = self._ell_pt(*C, R * 1.08, 0.40 * R * 1.08, ang, ph)
            tw = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 5.0 + ph0 * 3.0))
            cv2.circle(overlay, (int(p[0]), int(p[1])), 2, _dim(HOLO_CYAN, tw), -1)

    def _draw_sweep(self, overlay, C, R, t):
        period = 1.4
        f = (t % period) / period
        sy = C[1] + (f - 0.5) * 2.6 * R
        x0 = int(C[0] - 1.8 * R); x1 = int(C[0] + 1.8 * R)
        cv2.line(overlay, (x0, int(sy) - 1), (x1, int(sy) - 1), _dim(HOLO_CYAN, 0.3), 1)
        cv2.line(overlay, (x0, int(sy)), (x1, int(sy)), _dim(HOLO_CYAN, 0.75), 1)

    def _draw_state_fx(self, overlay, C, R, t):
        if self.state in ("held", "detach"):   # grip ring: reads as "active"
            k = 0.5 + 0.4 * math.sin(t * 6.0)
            cv2.ellipse(overlay, (int(C[0]), int(C[1])),
                        (int(1.25 * R), int(0.5 * R)), math.degrees(self._ang),
                        0, 360, _dim(HOLO_CYAN, k), 1, cv2.LINE_AA)
        if self.state == "detach":              # snap threads lifting it off
            f = min(1.0, self._state_t / self.DETACH_DUR)
            if f < 0.35 and self._from_px is not None:
                al = 1.0 - f / 0.35
                for i in range(3):
                    wd = 0.30 * R * (0.4 + 0.6 * i / 2.0)
                    sx = self._from_px[0] + (i - 1) * wd
                    cv2.line(overlay, (int(sx), int(self._from_px[1])),
                             (int(C[0]), int(C[1])),
                             (255, 255, 255) if i == 1 else _dim(HOLO_CYAN, 0.7),
                             max(1, int(2 * al)), cv2.LINE_AA)
        if self._burst is not None:              # detachment shockwave flare
            b_t, b_pos, b_R = self._burst
            age = t - b_t
            if 0.0 <= age < 0.55:
                fr = age / 0.55
                rr = (0.3 + 1.4 * fr) * b_R
                al = 0.85 * (1.0 - fr)
                cv2.ellipse(overlay, (int(b_pos[0]), int(b_pos[1])),
                            (int(rr), int(0.4 * rr)), math.degrees(self._ang),
                            0, 360, _dim(HOLO_CYAN, al), 2, cv2.LINE_AA)
                cv2.circle(overlay, (int(b_pos[0]), int(b_pos[1])),
                           int(rr * 0.5), _dim(HOLO_ORANGE, al * 0.6), 1, cv2.LINE_AA)
                for i in range(10):              # radial particle burst
                    ph = i * 2 * math.pi / 10 + age * 3.0
                    px = b_pos[0] + math.cos(ph) * rr
                    py = b_pos[1] + math.sin(ph) * rr * 0.4
                    cv2.circle(overlay, (int(px), int(py)), 2,
                               _dim(HOLO_CYAN, al), -1)
            elif age >= 0.55:
                self._burst = None
        for (tx, ty, tt) in self._trail:         # reattach streak ghosts
            age = t - tt
            if age < 0.35:
                al = 0.5 * (1.0 - age / 0.35)
                cv2.ellipse(overlay, (int(tx), int(ty)),
                            (int(R * 0.9), int(0.36 * R)),
                            math.degrees(self._ang), 0, 360,
                            _dim(HOLO_CYAN, al), 1, cv2.LINE_AA)



# --------------------------------------------------------------------------- #
# Camera / sources
# --------------------------------------------------------------------------- #
def open_camera(index: int | None):
    candidates = [index] if index is not None else [0, 1, 2, 3]
    dshow = getattr(cv2, "CAP_DSHOW", 700)
    for i in candidates:
        try:
            cap = cv2.VideoCapture(i, dshow)
        except Exception:  # noqa: BLE001
            cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        print(f"[camera] using device #{i}")
        try:  # fastest capture + lowest latency when the driver allows it
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:  # noqa: BLE001 - optional tweaks
            pass
        for cw, ch in ((1280, 720), (960, 720), (640, 480)):  # resolution ladder
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
            except Exception:  # noqa: BLE001
                pass
            for _ in range(4):
                cap.read()
            rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            if rw >= cw - 40:               # driver honoured this size
                break
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        print(f"[camera] running at {actual_w}x{actual_h}")
        if actual_w and actual_w < 800:
            print("[camera] note: low resolution limits hand recognition - try "
                  "a different camera, more light, or --camera 1/2/3")
        for _ in range(8):                  # let auto-exposure settle
            cap.read()
        return cap
    raise SystemExit(
        "No webcam was found. Check that a camera is connected/not in use by "
        "another app, then re-run.\n"
        "  - force a specific camera:  python hand_zoom.py --camera 1"
    )


def open_source(path: str):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[source] file not found: {path}")
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit(f"[source] could not read image: {path}")
        return ("image", img)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise SystemExit(f"[source] could not open video: {path}")
    return ("video", cap)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    ensure_model()
    landmarker = build_landmarker(MODEL_PATH)
    print("[ready] hand tracker initialised - show your hands to the camera.")

    if args.source:
        kind, source = open_source(args.source)
    else:
        kind, source = "camera", open_camera(args.camera)

    show_window = not args.no_window
    win = "HOLOGRAM STUDIO"
    if show_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        except Exception:  # noqa: BLE001 - cosmetic only
            pass
        cv2.resizeWindow(win, 960, 720)

    tracker = GestureTracker()
    holo = WebShooterHologram()
    frame_idx = 0
    last_ts = 0
    fps = 30.0
    last_frame_t = time.monotonic()

    recorder = None
    rec_path = None
    rec_start_t = 0.0

    def toggle_record():
        nonlocal recorder, rec_path, rec_start_t
        if recorder is not None:
            recorder.release()
            print(f"[record] saved {rec_path} ({time.monotonic() - rec_start_t:.1f} s)")
            recorder = None
            rec_path = None
            return
        (BASE_DIR / "recordings").mkdir(parents=True, exist_ok=True)
        rec_path = BASE_DIR / "recordings" / \
            (time.strftime("holo_%Y%m%d_%H%M%S") + ".mp4")
        fw = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(10.0, fps), (frame.shape[1], frame.shape[0]))
        if not fw.isOpened():
            rec_path = rec_path.with_suffix(".avi")
            fw = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 max(10.0, fps), (frame.shape[1], frame.shape[0]))
        recorder = fw
        rec_start_t = time.monotonic()
        print(f"[record] started -> {rec_path}")

    def release():
        nonlocal recorder
        if recorder is not None:
            recorder.release()
            print(f"[record] saved {rec_path}")
            recorder = None
        if show_window:
            cv2.destroyAllWindows()
        try:
            if kind in ("camera", "video"):
                source.release()
        except Exception:  # noqa: BLE001
            pass

    try:
        while True:
            t0 = time.monotonic()
            if kind == "image":
                frame = source
                ok = True
            else:
                ok, frame = source.read()
            if not ok:
                print("[camera] stream ended." if kind == "video"
                      else "[camera] failed to read a frame.")
                break

            frame = cv2.flip(frame, 1)      # mirrored view feels natural
            frame_idx += 1
            now_ms = max(int(t0 * 1000), last_ts + 1)
            last_ts = now_ms

            try:
                result = detect_frame(landmarker, frame, now_ms)
            except Exception as exc:  # noqa: BLE001
                print(f"[tracking] error: {exc}")
                result = None

            feats = classify_hands(result)
            right_feat = next((f for f in feats if f["is_right"]), None)
            if args.verbose and frame_idx % 30 == 0:
                print(f"[track] {len(feats)} hand(s) | "
                      f"shooter-hand={'present' if right_feat is not None else 'absent'}")

            # -- gesture events ------------------------------------------------- #
            now = t0
            events = tracker.feed(feats, now)
            for evt, payload in events:
                if evt == "grab":
                    holo.grab(payload, frame.shape[1], frame.shape[0], now)
                    print("[gesture] grab ->", holo.state)
                elif evt == "release":
                    holo.release(payload, frame.shape[1], frame.shape[0], now)
                    print("[gesture] release ->", holo.state)

            # if the right hand vanishes while holding the shooter, let it float
            if right_feat is None and holo.state in ("held", "detach"):
                holo.release(None, frame.shape[1], frame.shape[0], now)

            # -- render ---------------------------------------------------------- #
            dt = max(0.0, t0 - last_frame_t)
            last_frame_t = t0
            if right_feat is not None:
                draw_hand_holo(frame, right_feat["landmarks"])
            holo.update(dt, now, right_feat, frame.shape[1], frame.shape[0])
            holo.draw(frame, now)

            if recorder is not None:        # recording: red dot only, no text
                cv2.circle(frame, (frame.shape[1] - 42, 30), 8, (0, 0, 255), -1)
                recorder.write(frame)

            if show_window:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                elif key in (ord("v"), ord("V")):
                    toggle_record()

            if args.record and recorder is None and kind != "image":
                toggle_record()     # auto-start recording from launch

            if kind == "image":
                if show_window:
                    cv2.imshow(win, frame)
                    cv2.waitKey(3000)
                break

            if args.frames and frame_idx >= args.frames:
                break

            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

    except KeyboardInterrupt:
        pass
    finally:
        release()

    print("[done] bye")
    return 0


# --------------------------------------------------------------------------- #
# Self-test (headless; used to validate installs and regressions)
# --------------------------------------------------------------------------- #
def selftest(args: argparse.Namespace) -> int:
    print("== hologram studio selftest ==")
    ensure_model()
    print("[ok] model ready")
    landmarker = build_landmarker(MODEL_PATH)
    print("[ok] HandLandmarker created (VIDEO mode, 0.10/0.10/0.20 confidence)")

    # 1. real detection on a sample image with hands
    sample = Path(tempfile.gettempdir()) / "hand_zoom_sample.jpg"
    try:
        if not sample.is_file():
            urllib.request.urlretrieve(SAMPLE_IMAGE_URL, sample)
        img = cv2.imread(str(sample))
        assert img is not None, "could not decode sample image"
        res = detect_frame(landmarker, img, 1)
        hands = list(res.hand_landmarks or [])
        assert len(hands) >= 1, "expected at least one hand in the sample image"
        print(f"[ok] detected {len(hands)} hand(s) in sample image")
    except Exception as exc:  # noqa: BLE001 - offline runs must still pass
        print(f"[warn] image-detection check skipped ({exc})")
        return 1

    # 2. classify_hands labels every hand and picks the right one
    feats = classify_hands(res)
    assert feats, "classify_hands must return per-hand features"
    assert any(f["is_right"] for f in feats), "exactly the RIGHT hand must be flagged"
    print(f"[ok] classified {len(feats)} hand(s), right-hand picked")

    # 3. skeleton overlay draws without crashing
    f = np.zeros((480, 640, 3), np.uint8)
    right = next(f2 for f2 in feats if f2["is_right"])
    draw_hand_holo(f, right["landmarks"])
    assert f.any(), "skeleton overlay must draw pixels"
    print("[ok] hand skeleton overlay renders")

    # 4. preview video recorder
    rec = Path(tempfile.gettempdir()) / "holo_selftest.mp4"
    rec.unlink(missing_ok=True)
    vw = cv2.VideoWriter(str(rec), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    if vw.isOpened():
        for _ in range(12):
            vw.write(np.zeros((240, 320, 3), np.uint8))
        vw.release()
        assert rec.is_file() and rec.stat().st_size > 500, "recorded file too small"
        print(f"[ok] video recorder: wrote {rec.stat().st_size} bytes to mp4")
    else:
        print("[warn] mp4v writer unavailable; skipped recorder check")

    print("== selftest PASSED ==")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hand_zoom",
        description="Holographic Spider-Man gadget studio - hand-tracking core.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=None, help="webcam device index")
    p.add_argument("--source", type=str, default=None,
                   help="path to an image/video file instead of the camera (testing)")
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--no-window", action="store_true",
                   help="run without the preview window (headless)")
    p.add_argument("--verbose", action="store_true",
                   help="log tracking status to the console")
    p.add_argument("--record", action="store_true",
                   help="start recording the preview to recordings/ from launch")
    p.add_argument("--selftest", action="store_true",
                   help="run headless self-test and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
