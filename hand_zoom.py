#!/usr/bin/env python3
"""hand_zoom.py - a holographic Spider-Man web-shooter / gadget studio.

Tony-Stark-style hard-light holograms, controlled entirely by hand gestures and
rendered as real shaded 3D geometry (see holo3d.py / holo_models.py).  Electric
blue only - no other hue is ever produced.

Gestures
--------
  PINCH (thumb+index+middle) and PULL ......... summon a 3D web-shooter onto
                                               the OPPOSITE wrist: pinch-pull
                                               with the RIGHT hand and it flies
                                               onto the LEFT wrist, and vice
                                               versa.  Pinch-pull again to send
                                               that shooter away.
  PINCH A WORN SHOOTER and PULL ............... take it off.  Reach over,
                                               pinch the shooter sitting on
                                               your wrist and pull, exactly as
                                               you would strip a real one off:
                                               it detaches on the spot, becomes
                                               its own object in your hand and
                                               stays wherever you drop it.
  ONE FIST, hold ~0.45 s ...................... a detailed exploded 3D
                                               BLUEPRINT of the web-shooter in
                                               black and white, docked opposite
                                               the fist, with callouts and
                                               dimensions.  It holds still so
                                               it can be read.  Same gesture
                                               again dismisses it.
  BOTH FISTS, hold 1.0 s ...................... the full holographic suit -
                                               torso, spider emblem, webbing,
                                               shoulders, belt and an armoured
                                               gauntlet on each wrist.  Never a
                                               mask.  Hold again to take it off.
  GRAB an object and HOLD 4 s ................. a lock ring charges around it;
                                               when it snaps READY the object
                                               follows your hand.  Let go and it
                                               simply stays where you left it.
  D ........................................... live gesture readout (shows the
                                               raw finger-curl / pinch numbers
                                               so a stubborn gesture can be
                                               seen, not guessed at)
  V ........................................... start / stop recording
  Esc / Q ..................................... quit

Video calls
-----------
  With  --virtualcam  the rendered hologram is broadcast to a Windows virtual
  camera (Unity Capture or OBS), so WhatsApp / Teams / Zoom / Meet see it
  while the script runs.  See README.md and setup_virtualcam.bat.

Recognition is tuned to be as forgiving as possible:
  * MediaPipe confidence thresholds at the practical minimum
    (0.10 / 0.10 / 0.20)
  * the camera is pushed to 1280x720 (MJPEG + low-latency buffers) with an
    automatic fallback ladder
  * gesture features are EMA-smoothed per hand with hysteresis and frame
    confirmation; dropped hands are ghosted so hiccups never cancel a gesture

Usage
-----
    python hand_zoom.py               # show your hands to the camera
    python hand_zoom.py --camera 1    # force a specific webcam
    python hand_zoom.py --virtualcam  # broadcast to video calls (see README)
    python hand_zoom.py --record      # save the preview to recordings/
    python hand_zoom.py --selftest    # headless self-test (no camera needed)
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from holo3d import (HOLO_BLUE, HOLO_CYAN, HOLO_DEEP, HOLO_DIM, HOLO_WHITE,
                    clamp01, dim as _dim)
from holo_hud import Hud
from holo_objects import Blueprint, WebShooter, hand_frame
from holo_suit import SuitGear

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

# the palette lives in holo3d.py - electric BLUE only, imported above

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


FIST_CURL = 0.62          # a finger below this is folded
FIST_MEAN = 0.60          # ...and the whole hand has to be closed on average


def is_fist(curls: dict) -> bool:
    """True when the four fingers are folded into the palm.

    The THUMB is deliberately ignored.  In a natural fist it lies flat across
    the folded fingers and stays almost straight (measured ~0.95 by the same
    chain metric that reads 1.00 for a fully open hand), so testing it meant no
    real fist was ever recognised.  One finger is allowed to disagree, since a
    single mis-placed landmark should not cancel the gesture.
    """
    vals = [curls["index"], curls["middle"], curls["ring"], curls["pinky"]]
    folded = sum(1 for v in vals if v < FIST_CURL)
    return folded >= 3 and (sum(vals) / 4.0) < FIST_MEAN


def hand_features(landmarks, is_right: bool) -> dict:
    """Everything the gesture engine + renderers need for one hand."""
    scale = hand_scale(landmarks) + 1e-6
    curls = curl_features(landmarks)
    tt = _pt(landmarks, THUMB_TIP)
    it = _pt(landmarks, INDEX_TIP)
    mt = _pt(landmarks, MIDDLE_TIP)
    # closest of thumb->index / thumb->middle: pinching with two fingers is the
    # natural gesture, and averaging the two (as this used to) meant a real
    # pinch never got close to the threshold.
    pinches = min(float(np.linalg.norm(tt - it)), float(np.linalg.norm(tt - mt)))
    tips = [_pt(landmarks, i) for i in (INDEX_TIP, MIDDLE_TIP, 12, 16, 20)]
    spread = float(np.mean([np.linalg.norm(tips[i] - tips[i + 1]) for i in range(4)]))
    return {
        "landmarks": landmarks,
        "is_right": bool(is_right),
        "curls": curls,
        "fist": is_fist(curls),
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


MIRROR_HANDEDNESS = True    # see classify_hands()


def classify_hands(result) -> list:
    """Turn a detection result into a list of per-hand feature dicts.

    Every hand gets an 'is_right' flag.

    MediaPipe's own handedness label is reported for the image as given to it,
    and this pipeline hands it a horizontally FLIPPED frame (the mirrored
    selfie view, which is the only view that feels natural to gesture into).
    In that frame the user's right hand is on the right of the picture, which
    the model reads as a left hand - so its labels arrive inverted and are
    swapped back here.  Getting this wrong is very visible: the shooter lands
    on the wrist that summoned it instead of the opposite one.

    Without handedness, the position fallback applies (in the mirrored frame
    the rightmost hand is the user's right hand).
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
            if MIRROR_HANDEDNESS and name:      # un-mirror: see the docstring
                name = "Left" if name.lower().startswith("r") else "Right"
        labels.append(name)
    if not any(labels):
        order = sorted(range(len(hands)), key=lambda i: -hands[i][0].x)
        labels[order[0]] = "Right"
        if len(order) > 1:
            labels[order[-1]] = "Left"   # never overwrite a lone hand
    return [hand_features(h, labels[i] == "Right") for i, h in enumerate(hands)]


def draw_hand_holo(overlay, hand, k: float = 1.0, t: float = 0.0) -> None:
    """Trace the hand skeleton into the shared half-res hologram overlay.

    This is the user's recognition feedback, so it stays clearly visible and
    pulses gently instead of sitting static - but it is deliberately dimmer
    than the hardware so it never fights the holograms for attention.

    The landmarks carry a depth channel, so the skeleton is drawn as a real 3D
    wire model rather than a flat tracing: joints reaching toward the camera
    are fatter and brighter, joints behind the palm recede.  Tilt your hand and
    you can see it turn in space.
    """
    h, w = overlay.shape[0] / k, overlay.shape[1] / k
    pts = [(lm.x * w * k, lm.y * h * k) for lm in hand]
    # depth relative to the wrist, in hand-lengths; +ve = toward the camera
    span = math.hypot(hand[MIDDLE_MCP].x - hand[WRIST].x,
                      hand[MIDDLE_MCP].y - hand[WRIST].y) + 1e-6
    zs = [clamp01(0.5 + (hand[WRIST].z - lm.z) / (2.4 * span)) for lm in hand]
    pulse = 0.85 + 0.15 * math.sin(t * 3.1)
    for a, b in HAND_CONNECTIONS:
        pa = (int(pts[a][0]), int(pts[a][1]))
        pb = (int(pts[b][0]), int(pts[b][1]))
        z = 0.5 * (zs[a] + zs[b])
        cv2.line(overlay, pa, pb, _dim(HOLO_DEEP, (0.30 + 0.50 * z) * pulse),
                 2 + int(round(2 * z)), cv2.LINE_AA)
        cv2.line(overlay, pa, pb, _dim(HOLO_CYAN, 0.28 + 0.55 * z), 1, cv2.LINE_AA)
    for i, p in enumerate(pts):
        c = (int(p[0]), int(p[1]))
        z = zs[i]
        if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, 16, 20):  # fingertips glow
            cv2.circle(overlay, c, 2 + int(round(3 * z)),
                       _dim(HOLO_CYAN, (0.5 + 0.5 * z) * pulse), -1, cv2.LINE_AA)
            cv2.circle(overlay, c, 4 + int(round(4 * z)),
                       _dim(HOLO_BLUE, (0.18 + 0.3 * z) * pulse), 1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, c, 1 + int(round(2 * z)),
                       _dim(HOLO_CYAN, (0.3 + 0.4 * z) * pulse), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Gesture tracker: raw hand features -> discrete gesture events
# --------------------------------------------------------------------------- #
class GestureTracker:
    """Turns smoothed per-hand features into gesture events.

    Events (tuples):
      ('pinch', feat)    a hand closed thumb+index+middle
      ('pull', feat)     a pinching hand travelled ~one hand-length while
                         still pinched -> the summon gesture
      ('unpinch', feat)  the pinch opened again
      ('fist', feat)     ONE hand held a fist for FIST_HOLD (the other hand
                         must not also be a fist) -> blueprint
      ('gear', None)     BOTH fists held for GEAR_HOLD (1 s) -> the suit

    Per-hand features are EMA-smoothed (matched by palm proximity), and hands
    that drop out for a few frames keep their state ("ghosting") so a tracking
    hiccup never cancels a gesture mid-way.
    """

    FIST_HOLD = 0.40        # seconds one fist must be held
    GEAR_HOLD = 1.00        # seconds BOTH fists must be held
    GRAB_ON = 0.30          # pinch3 below this -> pinch begins
    GRAB_OFF = 0.46         # pinch3 above this -> pinch ends
    PINCH_OPEN = 0.50       # index must be at least this straight to pinch
    PINCH_CONFIRM = 2       # frames of confirmation before a pinch edge
    GHOST_TTL = 0.35        # seconds a dropped hand keeps its state
    PULL_DIST = 0.75        # pull distance, in hand-lengths
    PULL_WINDOW = 2.5       # seconds a pinch stays eligible to become a pull
    BOTH_GRACE = 0.30       # seconds a both-fist hold survives a tracking blip

    def __init__(self):
        self._slots: list = []
        self._both_fist_since: float | None = None
        self._both_fist_last = -9.0
        self._gear_fired = False

    # -- public ------------------------------------------------------------- #
    def feed(self, hands: list, t: float) -> list:
        events: list = []
        for s in self._slots:
            s["matched"] = False
        for h in hands:
            s = self._match(h)
            if s is None:
                s = self._new_slot(h, t)
                self._slots.append(s)
            else:
                s["matched"] = True
                s["last_seen"] = t
                self._apply(s, h)
        # prune hands gone for too long (identity compare: dicts hold arrays)
        self._slots = [s for s in self._slots
                       if s["matched"] or t - s["last_seen"] <= self.GHOST_TTL]

        live = [s for s in self._slots if t - s["last_seen"] <= self.GHOST_TTL]
        fists = [s for s in live if s["fist"]]
        # -- both fists -> the suit (fires while still held, no opening needed).
        # A blip that loses a hand for a frame or two must not restart the
        # count, so the hold survives BOTH_GRACE seconds of missing fists.
        if len(fists) >= 2:
            if self._both_fist_since is None or t - self._both_fist_last > self.BOTH_GRACE:
                self._both_fist_since = t
                self._gear_fired = False
            self._both_fist_last = t
            if not self._gear_fired and t - self._both_fist_since >= self.GEAR_HOLD:
                self._gear_fired = True
                events.append(("gear", None))
                for s in self._slots:        # consume the whole fist cycle
                    s["fist_fired"] = True
        elif t - self._both_fist_last > self.BOTH_GRACE:
            self._both_fist_since = None
            self._gear_fired = False

        for s in self._slots:
            if s["matched"]:
                self._pinch_edges(s, t, events)
        # -- one fist -> the blueprint --------------------------------------- #
        # ...but never while a both-fist hold is still inside its grace window,
        # or losing one hand for a frame would spawn a stray blueprint.
        if len(fists) == 1 and t - self._both_fist_last > self.BOTH_GRACE:
            s = fists[0]
            if s["fist_since"] is not None and not s["fist_fired"] \
                    and t - s["fist_since"] >= self.FIST_HOLD:
                s["fist_fired"] = True
                events.append(("fist", s["hand"]))
        return events

    def hud_state(self, t: float) -> dict:
        """Everything the HUD needs to visualise gestures in progress."""
        hands = []
        for s in self._slots:
            if t - s["last_seen"] > self.GHOST_TTL:
                continue
            fist_p = 0.0
            if s["fist_since"] is not None and not s["fist_fired"]:
                fist_p = clamp01((t - s["fist_since"]) / self.FIST_HOLD)
            pull_p = 0.0
            if s["pinch"] and not s["pulled"] and s["pinch_p0"] is not None:
                d = float(np.linalg.norm(s["pinch_px"] - s["pinch_p0"]))
                pull_p = clamp01(d / (self.PULL_DIST * max(s["scale"], 1e-3)))
            c = s["curls"] or {}
            hands.append({
                "side": "R" if s["is_right"] else "L",
                "palm": (float(s["palm"][0]), float(s["palm"][1])),
                "size": float(s["scale"]),
                "pinch": bool(s["pinch"]),
                "fist": bool(s["fist"]),
                "fist_p": fist_p,
                "pull_p": pull_p,
                # raw numbers for the debug readout (D key)
                "pinch3": float(s["pinch3"]),
                "curls": [float(c.get(k, 0.0)) for k in
                          ("index", "middle", "ring", "pinky", "thumb")],
            })
        gear_p = 0.0
        if self._both_fist_since is not None and not self._gear_fired:
            gear_p = clamp01((t - self._both_fist_since) / self.GEAR_HOLD)
        return {"hands": hands, "gear_p": gear_p}

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
            "scale": 0.1, "matched": True, "last_seen": t,
            "fist_since": None, "fist_fired": False, "pinch_ok": False,
            "pinch": False, "pinch_frames": 0,
            "pinch_px": h["palm"], "pinch_p0": None, "pinch_t0": 0.0,
            "pulled": False,
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
            s["pinch3"] = 0.50 * h["pinch3"] + 0.50 * s["pinch3"]
            s["spread"] = 0.50 * h["spread"] + 0.50 * s["spread"]
        for k in ("palm", "wrist", "mcp9", "index_tip", "middle_tip",
                  "thumb_tip", "is_right", "landmarks"):
            s[k] = h[k]
        s["hand"] = h
        s["scale"] = float(np.linalg.norm(h["mcp9"] - h["wrist"])) or 0.1
        s["pinch_px"] = 0.5 * (h["thumb_tip"] + h["index_tip"])
        c = s["curls"]
        fist = is_fist(c)
        s["fist"] = fist
        # a closed fist also puts the thumb next to the fingertips, so without
        # this gate every fist read as a pinch and pull-summoned a shooter
        s["pinch_ok"] = (not fist) and c["index"] > self.PINCH_OPEN
        if not fist:                    # opening the hand re-arms the gesture
            s["fist_since"] = None
            s["fist_fired"] = False

    def _pinch_edges(self, s, t, events):
        if s["fist"] and s["fist_since"] is None and not s["fist_fired"]:
            s["fist_since"] = t
        p = s["pinch_ok"] and \
            s["pinch3"] < (self.GRAB_OFF if s["pinch"] else self.GRAB_ON)
        # confirm quickly, let go quickly: the counter saturates low and drains
        # twice as fast, so releasing a grab never feels sticky
        s["pinch_frames"] = min(s["pinch_frames"] + 1, 4) if p \
            else max(s["pinch_frames"] - 2, 0)
        if p and s["pinch_frames"] >= self.PINCH_CONFIRM and not s["pinch"]:
            s["pinch"] = True
            s["pinch_p0"] = s["pinch_px"].copy()
            s["pinch_t0"] = t
            s["pulled"] = False
            events.append(("pinch", s["hand"]))
        elif not p and s["pinch_frames"] == 0 and s["pinch"]:
            s["pinch"] = False
            events.append(("unpinch", s["hand"]))
        elif s["pinch"] and not s["pulled"] and s["pinch_p0"] is not None \
                and t - s["pinch_t0"] <= self.PULL_WINDOW:
            d = float(np.linalg.norm(s["pinch_px"] - s["pinch_p0"]))
            if d >= self.PULL_DIST * max(s["scale"], 1e-3):
                s["pulled"] = True
                events.append(("pull", s["hand"]))


# --------------------------------------------------------------------------- #
# The holo desk: every object on screen, plus the grab-to-move rig
# --------------------------------------------------------------------------- #
def pinch_px(feat, w: int, h: int):
    p = 0.5 * (feat["thumb_tip"] + feat["index_tip"]) * np.array([w, h])
    return (float(p[0]), float(p[1]))


class HoloDesk:
    """Owns the shooters, the blueprint, the suit and the 4-second move lock."""

    DRAG_ARM = 4.0            # seconds a grab must be held before it moves

    def __init__(self, log=print):
        self.shooters = {"Right": WebShooter("Right"), "Left": WebShooter("Left")}
        self.blueprint = None
        self.suit = None
        self.drags: dict = {"Right": None, "Left": None}
        self.hud = Hud()
        self.debug = False
        self._log = log

    # -- helpers ------------------------------------------------------------- #
    @staticmethod
    def _side(feat) -> str:
        return "Right" if feat["is_right"] else "Left"

    def _draggables(self):
        objs = []
        for sh in self.shooters.values():
            if sh.drag_point() is not None:
                objs.append(sh)
        if self.blueprint is not None and self.blueprint.dying is None:
            objs.append(self.blueprint)
        return objs

    def _pick(self, px):
        best, bd = None, None
        for o in self._draggables():
            p = o.drag_point()
            d = math.hypot(p[0] - px[0], p[1] - px[1])
            if d <= o.drag_radius() and (bd is None or d < bd):
                best, bd = o, d
        return best

    # -- gesture entry points ------------------------------------------------ #
    def on_pinch(self, feat, w, h, t):
        side = self._side(feat)
        px = pinch_px(feat, w, h)
        obj = self._pick(px)
        self.drags[side] = {"obj": obj, "t0": t, "px": px,
                            "armed": False, "armed_t": 0.0}
        if obj is not None:
            self._log(f"[grab] {side} hand locked on - hold {self.DRAG_ARM:.0f}s to move")

    def on_pull(self, feat, w, h, t):
        side = self._side(feat)
        d = self.drags.get(side)
        px = pinch_px(feat, w, h)
        if d is not None and d["obj"] is not None:
            obj = d["obj"]
            # Pinching a shooter that is WORN and pulling takes it off, exactly
            # the way you would strip one off your wrist.  It detaches straight
            # away - no move-lock countdown - and from here on it is its own
            # object: it follows the pinch and stays where it is dropped.
            if isinstance(obj, WebShooter) and obj.detach(t, px):
                d["armed"] = True
                d["armed_t"] = t
                self._log(f"[gesture] pulled the {obj.side} shooter off - detached")
            return                                  # this pinch is a grab, not a pull
        target = "Left" if feat["is_right"] else "Right"
        # If that wrist is off-camera the shooter waits at the edge of the
        # frame on its own side - never floating in the middle of the view.
        park = (0.11 * w if target == "Left" else 0.89 * w, 0.34 * h)
        sh = self.shooters[target]
        what = sh.toggle(t, px, park)
        self._log(f"[gesture] pinch-pull {side} -> {target} wrist shooter {what}")

    def on_unpinch(self, feat, w, h, t):
        self._end_drag(self._side(feat), t)

    def on_fist(self, feat, w, h, t):
        if self.blueprint is not None and self.blueprint.dying is None:
            self.blueprint.dismiss()
            self._log("[gesture] fist -> blueprint dismissed")
            return
        hand_x = float(feat["palm"][0])
        dock = (0.76 * w, 0.46 * h) if hand_x < 0.5 else (0.24 * w, 0.46 * h)
        length = float(np.linalg.norm((feat["mcp9"] - feat["wrist"]) * np.array([w, h])))
        self.blueprint = Blueprint(dock, min(0.15 * h, max(0.085 * h, 1.2 * length)), 0.0)
        self._log("[gesture] fist -> web-shooter blueprint")

    def on_gear(self, t):
        if self.suit is not None and self.suit.dying is None:
            self.suit.dismiss()
            self._log("[gesture] both fists -> suit OFF")
        else:
            self.suit = SuitGear(t)
            self._log("[gesture] both fists -> SUIT ON")

    def handle(self, events, w, h, t):
        for evt, payload in events:
            if evt == "pinch":
                self.on_pinch(payload, w, h, t)
            elif evt == "pull":
                self.on_pull(payload, w, h, t)
            elif evt == "unpinch":
                self.on_unpinch(payload, w, h, t)
            elif evt == "fist":
                self.on_fist(payload, w, h, t)
            elif evt == "gear":
                self.on_gear(t)

    def _end_drag(self, side, t):
        d = self.drags.get(side)
        if d is None:
            return
        if d["armed"] and d["obj"] is not None:
            d["obj"].end_drag(t)
            self._log(f"[grab] {side} released - object parked")
        self.drags[side] = None

    # -- per-frame ----------------------------------------------------------- #
    def update(self, dt, t, feats, w, h):
        by_side = {"Right": None, "Left": None}
        for f in feats:
            by_side[self._side(f)] = f
        for side, d in list(self.drags.items()):
            feat = by_side[side]
            if d is None:
                continue
            if feat is None:                        # hand vanished: let go
                self._end_drag(side, t)
                continue
            d["px"] = pinch_px(feat, w, h)
            obj = d["obj"]
            if obj is None:
                continue
            if not d["armed"] and t - d["t0"] >= self.DRAG_ARM:
                d["armed"] = True
                d["armed_t"] = t
                obj.begin_drag(d["px"], t)
                self._log(f"[grab] {side} MOVE LOCK armed")
            elif d["armed"]:
                obj.drag_to(d["px"])
        for side in ("Right", "Left"):
            self.shooters[side].update(dt, t, by_side[side], w, h)
        if self.blueprint is not None:
            self.blueprint.update(dt)
            if not self.blueprint.alive():
                self.blueprint = None
        if self.suit is not None:
            self.suit.update(dt)
            if not self.suit.alive():
                self.suit = None

    def drag_hud(self, t):
        out = []
        for d in self.drags.values():
            if d is None or d["obj"] is None:
                continue
            p = d["obj"].drag_point() or d["px"]
            out.append({"pos": p, "radius": d["obj"].drag_radius(),
                        "progress": clamp01((t - d["t0"]) / self.DRAG_ARM),
                        "armed": d["armed"], "armed_t": d["armed_t"]})
        return out

    def chips(self, t):
        lock = 0.0
        for d in self.drags.values():
            if d is not None and d["obj"] is not None and not d["armed"]:
                lock = max(lock, clamp01((t - d["t0"]) / self.DRAG_ARM))
        armed = any(d is not None and d["armed"] for d in self.drags.values())
        return [("L WRIST", self.shooters["Left"].active(), 0.0),
                ("R WRIST", self.shooters["Right"].active(), 0.0),
                ("SUIT", self.suit is not None and self.suit.dying is None, 0.0),
                ("BLUEPRINT", self.blueprint is not None, 0.0),
                ("MOVE LOCK", armed, lock)]

    # -- render -------------------------------------------------------------- #
    def draw(self, frame, t, feats, tracker, fps):
        h, w = frame.shape[:2]
        k = 0.5
        ov = np.zeros((max(2, int(h * k)), max(2, int(w * k)), 3), np.uint8)
        if self.suit is not None:
            self.suit.draw(ov, k, t, feats)
        for side in ("Right", "Left"):
            self.shooters[side].draw(ov, k, t)
        if self.blueprint is not None:
            self.blueprint.draw(ov, k, t)
        for f in feats:                       # skeleton tracing on every hand
            if f["landmarks"] is not None:
                draw_hand_holo(ov, f["landmarks"], k, t)
        # the bilinear upscale from half-res IS the bloom - no blur pass needed
        bloom = cv2.resize(ov, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(bloom, 0.72, frame, 1.0, 0, frame)
        # the wrist shooters are small, so they get a crisp full-resolution
        # pass of their own once the bloom is down
        for side in ("Right", "Left"):
            self.shooters[side].draw_sharp(frame, t)
        if self.suit is not None:
            self.suit.draw_sharp(frame, t, feats)
        if self.blueprint is not None:
            self.blueprint.annotate(frame, t)
        state = tracker.hud_state(t)
        self.hud.draw(frame, {"t": t, "fps": fps, "hand_count": len(feats),
                              "hands": state["hands"], "gear_p": state["gear_p"],
                              "drags": self.drag_hud(t), "chips": self.chips(t),
                              "debug": self.debug,
                              "thresholds": {"grab_on": GestureTracker.GRAB_ON,
                                             "fist_curl": FIST_CURL}})


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
# Windows virtual camera (show the hologram on video calls)
# --------------------------------------------------------------------------- #
class VirtualCamError(RuntimeError):
    """Raised when no usable virtual-camera backend could be initialised."""


class VirtualCam:
    """Push the rendered frames to a Windows virtual camera via pyvirtualcam.

    Backends are tried in order: 'unitycapture' (the Unity Capture DirectShow
    filter, a one-time admin install) then 'obs' (OBS Studio's built-in
    virtual camera).  The device only produces video while this app is
    running, so WhatsApp / Teams / Zoom / Meet see the hologram exactly when
    the script runs and nothing when it stops.

    Frames are pushed through a tiny bounded queue to a background thread, so
    a slow consumer never stalls the tracking/render loop.
    """

    def __init__(self, width: int, height: int, fps: float, backend: str = "auto"):
        try:
            import pyvirtualcam  # optional dependency, imported lazily
        except ImportError as exc:
            raise VirtualCamError(
                "pyvirtualcam is not installed - run setup_virtualcam.bat"
            ) from exc
        candidates = ("unitycapture", "obs") if backend == "auto" else (backend,)
        self._cam = None
        self._backend = None
        errs = []
        for b in candidates:
            try:
                self._cam = pyvirtualcam.Camera(
                    width=width, height=height, fps=int(max(15.0, min(30.0, fps))),
                    backend=b, fmt=pyvirtualcam.PixelFormat.BGR,
                )
                self._backend = b
                break
            except Exception as exc:  # noqa: BLE001 - try the next backend
                errs.append(f"{b}: {" ".join(str(exc).split())}")
        if self._cam is None:
            raise VirtualCamError(" | ".join(errs))
        self._stop = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._cam.send(frame)
                self._cam.sleep_until_next_frame()
            except Exception:  # noqa: BLE001 - consumer vanished, stop quietly
                break

    def send(self, frame) -> None:
        if self._cam is None:
            return
        try:
            if self._queue.full():
                try:
                    self._queue.get_nowait()   # drop the stale frame
                except queue.Empty:
                    pass
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def close(self) -> None:
        self._stop.set()
        if getattr(self, "_thread", None) is not None:
            self._thread.join(timeout=1.0)
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:  # noqa: BLE001
                pass
            self._cam = None

    @staticmethod
    def try_create(width: int, height: int, fps: float, backend: str = "auto"):
        """Create a VirtualCam or return None with setup guidance on failure."""
        try:
            return VirtualCam(width, height, fps, backend)
        except VirtualCamError as exc:
            print(f"[virtualcam] unavailable ({exc})")
            print(VIRTUALCAM_SETUP_HINT)
            return None


VIRTUALCAM_SETUP_HINT = """\
[virtualcam] To show the hologram on video calls (WhatsApp, Teams, Zoom...):
  1) run  setup_virtualcam.bat   (installs pyvirtualcam + the Unity Capture
     driver; one admin prompt, done once)
  2) or install OBS Studio and start its built-in Virtual Camera
  3) relaunch with:  python hand_zoom.py --virtualcam
  4) in the call app pick the camera: "Unity Video Capture" (or "OBS Virtual
     Camera"). The hologram appears only while this script runs.
"""


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
    desk = HoloDesk()
    desk.debug = bool(args.debug)
    t_start = time.monotonic()
    vcam = None
    vcam_enabled = bool(args.virtualcam)
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
        nonlocal recorder, vcam
        if recorder is not None:
            recorder.release()
            print(f"[record] saved {rec_path}")
            recorder = None
        if vcam is not None:
            vcam.close()
            print(f"[virtualcam] stopped (backend: {vcam._backend})")
            vcam = None
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
            left_feat = next((f for f in feats if not f["is_right"]), None)
            if args.verbose and frame_idx % 30 == 0:
                print(f"[track] {len(feats)} hand(s) | "
                      f"R={'present' if right_feat is not None else 'absent'} "
                      f"L={'present' if left_feat is not None else 'absent'}")

            # -- gesture events ------------------------------------------------- #
            now = t0 - t_start
            w, h = frame.shape[1], frame.shape[0]
            events = tracker.feed(feats, now)
            desk.handle(events, w, h, now)

            # -- render ---------------------------------------------------------- #
            dt = max(0.0, t0 - last_frame_t)
            last_frame_t = t0
            desk.update(dt, now, feats, w, h)
            desk.draw(frame, now, feats, tracker, fps)

            if recorder is not None:        # recording: red dot only, no text
                if not vcam_enabled or vcam is None:  # keep the dot out of calls
                    cv2.circle(frame, (frame.shape[1] - 42, 30), 8, (0, 0, 255), -1)
                recorder.write(frame)

            # -- virtual camera (video calls see the hologram) ----------------- #
            if vcam_enabled:
                if vcam is None:
                    vcam = VirtualCam.try_create(frame.shape[1], frame.shape[0],
                                                 30.0, args.vc_backend)
                    if vcam is None:
                        vcam_enabled = False
                if vcam is not None:
                    # flip back: other callers should see a normal (unmirrored)
                    # camera view, not your mirrored selfie preview
                    vcam.send(cv2.flip(frame, 1))

            if show_window:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                elif key in (ord("v"), ord("V")):
                    toggle_record()
                elif key in (ord("d"), ord("D")):
                    desk.debug = not desk.debug

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
def _fake_feat(is_right=True, fist=False, pinch3=None, hand_id=0.0, dx=0.0, dy=0.0):
    """Synthetic feature dict for deterministic gesture-engine tests.

    The fist is modelled the way a real one measures: fingers folded but the
    THUMB nearly straight across them (~0.95), and the thumb tip therefore
    sitting right next to the fingertips (a low pinch distance).
    """
    if fist:
        curls = dict(thumb=0.95, index=0.2, middle=0.2, ring=0.2, pinky=0.2)
        spread = 0.30
    else:
        curls = dict(thumb=0.9, index=0.95, middle=0.9, ring=0.9, pinky=0.9)
        spread = 0.75
    if pinch3 is None:
        pinch3 = 0.15 if fist else 0.60
    x = 0.3 + hand_id + dx
    y = dy
    return {
        "landmarks": None, "is_right": bool(is_right), "curls": curls,
        "fist": fist, "open": not fist, "pinch3": pinch3, "spread": spread,
        "palm": np.array([x, 0.50 + y]), "wrist": np.array([x, 0.75 + y]),
        "mcp9": np.array([x, 0.62 + y]), "index_tip": np.array([x, 0.44 + y]),
        "middle_tip": np.array([x + 0.05, 0.44 + y]),
        "thumb_tip": np.array([x, 0.46 + y]),
    }


def _feat_from_landmarks(landmarks, is_right):
    return hand_features(landmarks, is_right)


def _pinch_run(tr, side, n, t0, dx=0.0, step=1 / 30.0):
    """Feed n pinching frames, drifting the hand by dx per frame."""
    evs = []
    for i in range(n):
        evs += tr.feed([_fake_feat(side, pinch3=0.10, dx=dx * i)], t0 + i * step)
    return evs


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
    rights = [f for f in feats if f["is_right"]]
    assert rights, "exactly the RIGHT hand must be flagged"
    right = rights[0]
    print(f"[ok] classified {len(feats)} hand(s), right-hand picked")

    # 3. the 3D models compile and render real shaded geometry
    import holo3d as H3
    import holo_models as HM
    for name, mesh in (("shooter", HM.shooter_mesh()), ("gauntlet", HM.gauntlet_mesh()),
                       ("suit", HM.suit_mesh())):
        assert mesh.V is not None and len(mesh.FI) > 100, f"{name} mesh too coarse"
        canvas = np.zeros((240, 320, 3), np.uint8)
        H3.render_mesh(canvas, mesh, (160, 120), 60, H3.rot_matrix(0.6, -0.3, 0.2))
        lit = int((canvas.sum(axis=2) > 120).sum())
        assert lit > 400, f"{name} mesh drew almost nothing ({lit} px)"
        assert canvas[:, :, 0].sum() > canvas[:, :, 2].sum(), \
            f"{name} must render BLUE-dominant"
    print(f"[ok] 3D models compile and render blue "
          f"({len(HM.shooter_mesh().FI)} shooter faces)")

    # 4. shooter lifecycle: summon -> arrive -> on the wrist -> dismissed
    sh = WebShooter("Left")
    assert not sh.active() and sh.drag_point() is None, "shooters start OFF"
    sh.summon(0.0, (100.0, 100.0), (160.0, 240.0))
    canvas = np.zeros((480, 640, 3), np.uint8)
    for i in range(30):
        t = i / 30.0
        sh.update(1 / 30.0, t, right, 640, 480)
        half = np.zeros((240, 320, 3), np.uint8)
        sh.draw(half, 0.5, t)
    assert sh.state == "on", f"shooter must land on the wrist (got {sh.state})"
    half = np.zeros((240, 320, 3), np.uint8)
    sh.draw(half, 0.5, 1.0)
    assert half.any(), "mounted shooter must draw its glow into the overlay"
    full = np.zeros((480, 640, 3), np.uint8)
    sh.draw_sharp(full, 1.0)          # the crisp full-resolution pass
    wx, wy = right["wrist"] * np.array([640, 480])
    near = full[max(0, int(wy - 120)):int(wy + 120), max(0, int(wx - 120)):int(wx + 120)]
    assert near.size and int((near.sum(axis=2) > 120).sum()) > 200, \
        "the shooter must sit on the wrist, at full resolution"
    assert full[:, :, 0].sum() > full[:, :, 2].sum(), "the shooter must stay blue"
    sh.dismiss(1.0)
    for i in range(20):
        sh.update(1 / 30.0, 1.0 + i / 30.0, right, 640, 480)
    assert sh.state == "off" and sh.drag_point() is None, "dismiss must clear it"
    print("[ok] shooter: summon -> fly-on -> mounted at the wrist -> dismissed")

    # 5. shooter drag protocol: grab moves it, release parks it
    sh.summon(2.0, (100.0, 100.0), (160.0, 240.0))
    for i in range(30):
        sh.update(1 / 30.0, 2.0 + i / 30.0, right, 640, 480)
    sh.begin_drag((300.0, 300.0), 3.0)
    sh.drag_to((420.0, 260.0))
    sh.update(1 / 30.0, 3.1, right, 640, 480)
    assert sh.state == "held" and abs(sh.pos[0] - 420.0) < 200.0, "grab must carry it"
    sh.end_drag(3.2)
    parked = sh.pos
    sh.update(1 / 30.0, 3.3, None, 640, 480)
    assert sh.state == "float" and abs(sh.pos[0] - parked[0]) < 1.0, \
        "released objects must stay where they were left"
    print("[ok] shooter: 4 s grab carries it, release parks it in place")

    # 6. blueprint: renders, annotates and fades out on dismiss
    bp = Blueprint((420.0, 240.0), 110.0, 0.0)
    for i in range(40):
        bp.update(1 / 30.0)
        half = np.zeros((240, 320, 3), np.uint8)
        bp.draw(half, 0.5, i / 30.0)
        assert half.any(), "blueprint must draw pixels"
    full = np.zeros((480, 640, 3), np.uint8)
    mono = np.zeros((240, 320, 3), np.uint8)
    bp.draw(mono, 0.5, 1.4)
    b, r = float(mono[:, :, 0].sum()), float(mono[:, :, 2].sum())
    assert abs(b - r) < 0.04 * max(b, 1.0), \
        "the blueprint is a BLACK AND WHITE hologram - no colour cast"
    rot_a = bp._rot.copy()                     # ...and it must not spin
    for _ in range(90):
        bp.update(1 / 30.0)
    bp.draw(np.zeros((240, 320, 3), np.uint8), 0.5, 4.4)
    assert np.abs(bp._rot - rot_a).max() < 0.15, \
        "the blueprint must hold its view, not rotate"
    bp.annotate(full, 4.4)
    assert full.any(), "blueprint must draw its callouts at full resolution"
    bp.dismiss()
    for _ in range(20):
        bp.update(1 / 30.0)
    assert not bp.alive(), "a dismissed blueprint must expire"
    print("[ok] blueprint: black-and-white exploded view, holds still, callouts")

    # 7. suit: draws torso + gauntlets from the visible hands, never a head
    suit = SuitGear(0.0)
    half = np.zeros((240, 320, 3), np.uint8)
    for i in range(45):
        suit.update(1 / 30.0)
    suit.draw(half, 0.5, 1.5, feats)
    assert half.any(), "the suit must draw with hands visible"
    ys = np.nonzero(half.sum(axis=2) > 100)[0]
    assert ys.size and ys.min() > 8, "the suit must never reach the top of frame"
    print("[ok] suit: torso, emblem, webbing and gauntlets materialise")

    # 8. tracker: pinch + pull fires exactly one summon per pinch
    tr = GestureTracker()
    evs = _pinch_run(tr, True, 3, 0.0)
    assert [e for e in evs if e[0] == "pinch"], "pinch edge must fire"
    evs = _pinch_run(tr, True, 8, 0.2, dx=0.03)
    pulls = [e for e in evs if e[0] == "pull"]
    assert len(pulls) == 1, f"pull must fire once per pinch (got {len(pulls)})"
    evs = []
    for i in range(4):
        evs += tr.feed([_fake_feat(True, pinch3=0.9, dx=0.24)], 0.6 + i / 30.0)
    assert [e for e in evs if e[0] == "unpinch"], "unpinch edge must fire"
    tr2 = GestureTracker()
    evs = _pinch_run(tr2, True, 12, 0.0)      # pinch, but never moved
    assert not [e for e in evs if e[0] == "pull"], "a still pinch must not summon"
    print("[ok] tracker: pinch+pull summons once, a still pinch does not")

    # 9. tracker: one fist -> blueprint, both fists 1 s -> suit (no blueprint)
    assert is_fist(dict(thumb=0.95, index=0.3, middle=0.3, ring=0.35, pinky=0.4)), \
        "a real fist keeps the thumb straight across the fingers - it must count"
    assert not is_fist(dict(thumb=1.0, index=0.99, middle=0.99, ring=0.99, pinky=0.99)), \
        "an open hand is not a fist"
    tr3 = GestureTracker()
    evs = []
    for i in range(24):
        evs += tr3.feed([_fake_feat(True, fist=True)], i / 30.0)
    fists = [e for e in evs if e[0] == "fist"]
    assert len(fists) == 1, f"one fist must fire once per cycle (got {len(fists)})"
    # a fist puts the thumb on the fingertips: it must never read as a pinch,
    # and a fist swung across the frame must never summon a shooter
    assert not [e for e in evs if e[0] in ("pinch", "pull")], \
        "a fist must not register as a pinch"
    tr3b = GestureTracker()
    evs = []
    for i in range(24):
        evs += tr3b.feed([_fake_feat(True, fist=True, dx=0.03 * i)], i / 30.0)
    assert not [e for e in evs if e[0] == "pull"], "a moving fist must not summon"
    tr4 = GestureTracker()
    evs = []
    for i in range(40):
        evs += tr4.feed([_fake_feat(True, fist=True),
                         _fake_feat(False, fist=True, hand_id=0.35)], i / 30.0)
    assert [e for e in evs if e[0] == "gear"], "both fists (1 s) must fire the suit"
    assert not [e for e in evs if e[0] == "fist"], \
        "both fists must never also spawn a blueprint"
    early = []
    for i in range(20):                        # 0.63 s: too short for the suit
        early += tr4.feed([_fake_feat(True, fist=True, dy=0.2),
                           _fake_feat(False, fist=True, hand_id=0.35, dy=0.2)],
                          10.0 + i / 30.0)
    assert not [e for e in early if e[0] == "gear"], "a short both-fist must not fire"
    print("[ok] tracker: one fist -> blueprint, both fists 1 s -> suit")

    # 10. desk: pinch-pull with the RIGHT hand arms the LEFT wrist, and vice versa
    desk = HoloDesk(log=lambda *a: None)
    rf = _fake_feat(True, pinch3=0.10)
    desk.handle([("pinch", rf), ("pull", rf)], 640, 480, 0.0)
    assert desk.shooters["Left"].active() and not desk.shooters["Right"].active(), \
        "a right-hand pull must equip the LEFT wrist"
    lf = _fake_feat(False, pinch3=0.10, hand_id=0.4)
    desk.handle([("pinch", lf), ("pull", lf)], 640, 480, 0.1)
    assert desk.shooters["Right"].active(), "a left-hand pull must equip the RIGHT wrist"
    desk.handle([("pull", rf)], 640, 480, 0.2)
    desk.update(1 / 30.0, 0.25, [], 640, 480)
    assert desk.shooters["Left"].state == "dismiss", "pulling again sends it away"
    print("[ok] desk: pinch-pull equips the OPPOSITE wrist, again to dismiss")

    # 10b. pinching a WORN shooter and pulling takes it off, then it stays put
    desk3 = HoloDesk(log=lambda *a: None)
    lf2 = _fake_feat(False, pinch3=0.10, hand_id=0.4)
    desk3.handle([("pinch", lf2), ("pull", lf2)], 640, 480, 0.0)
    sh_r = desk3.shooters["Right"]
    for i in range(30):                        # let it fly on and mount
        desk3.update(1 / 30.0, i / 30.0, [right], 640, 480)
    assert sh_r.state == "on", f"the shooter must be worn first (got {sh_r.state})"
    worn = sh_r.pos
    grab2 = _fake_feat(True, pinch3=0.10)      # reach over and pinch it
    grab2["thumb_tip"] = np.array([worn[0] / 640.0, worn[1] / 480.0])
    grab2["index_tip"] = grab2["thumb_tip"].copy()
    desk3.handle([("pinch", grab2)], 640, 480, 1.0)
    assert desk3.drags["Right"]["obj"] is sh_r, "the pinch must catch the worn shooter"
    desk3.handle([("pull", grab2)], 640, 480, 1.02)
    assert sh_r.state == "held", "pulling a worn shooter must take it off at once"
    off = _fake_feat(True, pinch3=0.10)
    off["thumb_tip"] = np.array([0.25, 0.25])
    off["index_tip"] = off["thumb_tip"].copy()
    desk3.update(1 / 30.0, 1.1, [off], 640, 480)
    assert abs(sh_r.pos[0] - worn[0]) > 40.0, "a detached shooter follows the hand"
    desk3.handle([("unpinch", off)], 640, 480, 1.2)
    dropped = sh_r.pos
    desk3.update(1 / 30.0, 1.3, [right], 640, 480)
    assert sh_r.state == "float", "a detached shooter must not jump back on"
    assert abs(sh_r.pos[0] - dropped[0]) < 1.0, "it stays where it was dropped"
    print("[ok] desk: pinch a worn shooter + pull -> it comes off as its own object")

    # 11. desk: a 4 s grab arms the move lock, then the object follows the hand
    desk2 = HoloDesk(log=lambda *a: None)
    desk2.handle([("fist", _fake_feat(True, fist=True))], 640, 480, 0.0)
    assert desk2.blueprint is not None, "a fist must spawn the blueprint"
    bp_pos = desk2.blueprint.pos
    grab = _fake_feat(True, pinch3=0.10)
    grab["thumb_tip"] = np.array([bp_pos[0] / 640.0, bp_pos[1] / 480.0])
    grab["index_tip"] = grab["thumb_tip"].copy()
    desk2.handle([("pinch", grab)], 640, 480, 0.0)
    assert desk2.drags["Right"]["obj"] is desk2.blueprint, "the grab must capture it"
    desk2.update(1 / 30.0, 2.0, [grab], 640, 480)
    assert not desk2.drags["Right"]["armed"], "2 s is not enough to move it"
    assert desk2.blueprint.pos == bp_pos, "an unarmed grab must not move anything"
    hud = desk2.drag_hud(2.0)
    assert hud and 0.4 < hud[0]["progress"] < 0.6, "the lock ring must show progress"
    desk2.update(1 / 30.0, 4.2, [grab], 640, 480)
    assert desk2.drags["Right"]["armed"], "4 s must arm the move lock"
    moved = _fake_feat(True, pinch3=0.10)
    moved["thumb_tip"] = np.array([0.2, 0.2])
    moved["index_tip"] = np.array([0.2, 0.2])
    desk2.update(1 / 30.0, 4.4, [moved], 640, 480)
    assert abs(desk2.blueprint.pos[0] - bp_pos[0]) > 40.0, "an armed grab must move it"
    here = desk2.blueprint.pos
    desk2.handle([("unpinch", moved)], 640, 480, 4.5)
    desk2.update(1 / 30.0, 4.6, [moved], 640, 480)
    assert desk2.blueprint.pos == here and desk2.drags["Right"] is None, \
        "releasing must simply leave the object where it is"
    print("[ok] desk: grab 4 s -> move lock -> object follows -> release parks it")

    # 12. the HUD renders over a live frame without touching the geometry
    frame = np.zeros((480, 640, 3), np.uint8)
    tr5 = GestureTracker()
    tr5.feed([_fake_feat(True, pinch3=0.10)], 0.0)
    tr5.feed([_fake_feat(True, pinch3=0.10)], 1 / 30.0)
    desk2.draw(frame, 3.0, feats, tr5, 30.0)
    assert frame.any(), "the HUD must draw"
    assert frame[:, :, 0].sum() > frame[:, :, 2].sum(), "the HUD must stay blue"
    print("[ok] HUD renders (brackets, gesture charge, equipment chips), stays blue")

    # 13. virtual camera: works when a backend exists, degrades gracefully
    vc = VirtualCam.try_create(320, 240, 30.0, backend="unitycapture")
    if vc is not None:
        vc.close()
        print("[ok] virtual camera backend present and usable")
    else:
        print("[ok] virtual camera degrades gracefully when the driver is missing")

    # 14. preview video recorder
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
    p.add_argument("--debug", action="store_true",
                   help="show the live gesture readout (also toggled with D)")
    p.add_argument("--virtualcam", action="store_true",
                   help="broadcast the preview to a Windows virtual camera "
                        "so video calls (WhatsApp, Teams, Zoom) see the hologram")
    p.add_argument("--vc-backend", choices=("auto", "unitycapture", "obs"),
                   default="auto", help="virtual-camera backend (default: auto)")
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
