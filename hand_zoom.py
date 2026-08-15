#!/usr/bin/env python3
"""hand_zoom.py - a holographic Spider-Man web-shooter / gadget studio.

Tony-Stark-style hard-light holograms, controlled entirely by hand gestures and
rendered as real shaded 3D geometry (see holo3d.py / holo_models.py).  Electric
blue only - no other hue is ever produced.

Gestures (final set - each one is gated so it can never fire another by
accident; see GestureTracker / HoloDesk for the exact collision guards)
--------------------------------------------------------------------------
  (automatic) SHOW A WRIST .................... a web-shooter equips onto it
                                               the instant it's seen - no
                                               gesture needed.  Hard cap of
                                               TWO on screen at once, one per
                                               wrist.
  WEB-SHOOT SIGN (middle + ring pressed
    into the palm - the trigger - with
    index and pinky out) then PULL ........... sends that wrist's shooter
                                               away (pinch-pull again brings
                                               it back).  A tight thumb+index
                                               grab also works.
  PINCH A WORN SHOOTER and PULL ............... take it off by hand, the way
                                               you'd strip a real one off: it
                                               unclips and swings free over a
                                               short peel animation - it does
                                               not teleport into your grip -
                                               then follows the pinch and
                                               stays wherever you let it go.
  ONE FIST, hold ~0.45 s, then OPEN ........... if that wrist's shooter is
                                               off, this wears it back on -
                                               closing and opening the hand
                                               like pulling a glove back on.
                                               Fires on release, so a fist
                                               held for another gesture's sake
                                               never triggers this one early.
  BOTH FISTS, hold ~0.55 s, then OPEN ......... toggle the full BODY GEAR:
                                               torso shell + spider emblem +
                                               webbing, shoulder plates, belt,
                                               an armoured gauntlet with its
                                               own arc reactor on every wrist,
                                               and the chest reactor.  Same
                                               gesture (or R) stands it down.
  OPEN PALM, hold it ........................... a Stark hologram materialises
                                               above the palm and tracks it;
                                               it fades the moment the hand
                                               closes.
  WEB-SHOOT POSE held (shooter worn) ........... a targeting web-line snaps out
                                               of the spinneret to a reticle.
  GRAB an object and HOLD 1.5 s ................. a lock ring charges around
                                               it; when it snaps READY the
                                               object follows your hand.
                                               Release and it stays exactly
                                               where you left it.
  (automatic) MOVE A HAND SHARPLY ............. SPIDER-SENSE: a broken radial
                                               warning ripples out of the
                                               palm, pings once and settles.
  G ............................................ toggle the on-screen gesture
                                               guide panel
  D ............................................ live gesture readout (raw
                                               finger-curl / pinch numbers, so
                                               a stubborn gesture can be seen,
                                               not guessed at)
  V ............................................ start / stop recording
  R ............................................ toggle body gear (keyboard)
  Esc / Q ....................................... quit

Video calls
-----------
  With  --virtualcam  the rendered hologram is broadcast to a Windows virtual
  camera (Unity Capture or OBS), so WhatsApp / Teams / Zoom / Meet see it
  while the script runs.  See README.md and setup_virtualcam.bat.

Recognition is tuned to be as forgiving as possible:
  * MediaPipe confidence at 0.20 / 0.20 / 0.40 - low enough to keep hands in
    weak light and at the frame edges, high enough to stop the detector
    re-firing on noise and making hands pop in and out
  * the camera is pushed to 1280x720 (MJPEG + low-latency buffers) with an
    automatic fallback ladder
  * fingers are judged by their JOINT ANGLES in 3D (invariant to hand size
    and distance), pinches by 3D thumb-to-finger distance, and handedness by
    MediaPipe's own label (correct on the mirrored feed) with a position
    fallback - so crossing your hands to reach the other wrist no longer
    flips left/right mid-gesture
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
from holo_objects import PalmProjector, WebShooter, hand_frame
from holo_pose import PoseTracker
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

# thumb joint indexes (cmc, mcp, ip, tip)
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP2 = 1, 2, 3, 4
INDEX_MCP = 5


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
        # low-but-not-minimal confidence: at 0.10 the detector re-fires on
        # noise, so hands pop in and out and every gesture looks broken.  The
        # VIDEO-mode tracker re-detects a lost hand within a frame or two at
        # 0.20, while 0.40 keeps an established track from being dropped by
        # single-frame flickers.
        min_hand_detection_confidence=0.20,
        min_hand_presence_confidence=0.20,
        min_tracking_confidence=0.40,
    )
    return tasks.vision.HandLandmarker.create_from_options(options)


DETECT_WIDTH = 640      # the landmarker's own input size; see detect_frame


def detect_frame(landmarker, frame_bgr: np.ndarray, timestamp_ms: int):
    """Run the landmarker on one BGR frame; returns raw result object.

    The frame is scaled down to DETECT_WIDTH first.  The model works at a small
    fixed input size internally anyway, so nothing is lost by not feeding it
    720p - but the colour convert, the copy and the pre-processing all scale
    with the pixels, and doing them on a quarter of the data roughly halves the
    per-frame tracking cost.  Landmarks come back NORMALISED, so they map onto
    the full-size frame with no correction at all.  Faster frames are what
    tracking actually feels like: a gesture that is sampled twice as often
    follows the hand instead of lagging behind it.
    """
    src = frame_bgr
    if src.shape[1] > DETECT_WIDTH:
        k = DETECT_WIDTH / float(src.shape[1])
        src = cv2.resize(src, (DETECT_WIDTH, max(2, int(round(src.shape[0] * k)))),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
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


def _p3(landmarks, i: int) -> np.ndarray:
    """Landmark as a 3D point; MediaPipe's z is normalised like x/y."""
    lm = landmarks[i]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _angle3(a, b, c) -> float:
    """Angle (radians) at b formed by the vectors b->a and b->c, in 3D."""
    v1 = a - b
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    return math.acos(max(-1.0, min(1.0, cosv)))


def curl_features(landmarks) -> dict:
    """Curl in [0..1] for each finger (1 = fully folded into the palm).

    Every finger is judged by the BEND at its PIP and DIP joints, measured in
    3D: a straight joint reads ~180 degrees and a folded one collapses toward
    90 or less.  Angles are invariant to hand size, distance from the camera
    and perspective foreshortening - the old mcp->tip distance ratio depended
    on all three, which is why a real fist read as open in one pose and a
    resting hand read as a fist in another.  The thumb (which has no PIP/DIP
    in the MediaPipe skeleton) blends its MCP/IP joint angles with the classic
    tip-vs-index-base distance test.
    """
    pts = {i: _p3(landmarks, i) for i in range(21)}
    curls: dict = {}
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        a_pip = _angle3(pts[mcp], pts[pip], pts[dip])
        a_dip = _angle3(pts[pip], pts[dip], pts[tip])
        # joints that have folded past ~112 degrees are the folded part of the
        # range; anything straighter than that contributes nothing
        c_pip = clamp01((math.pi - a_pip) / (0.62 * math.pi))
        c_dip = clamp01((math.pi - a_dip) / (0.62 * math.pi))
        curls[name] = 0.55 * c_pip + 0.45 * c_dip
    a_mcp = _angle3(pts[THUMB_CMC], pts[THUMB_MCP], pts[THUMB_IP])
    a_ip = _angle3(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP2])
    c_ang = clamp01((math.pi - max(a_mcp, a_ip)) / (0.55 * math.pi))
    d_tip = float(np.linalg.norm(pts[THUMB_TIP2] - pts[INDEX_MCP]))
    d_arm = float(np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])) or 1e-6
    c_dst = clamp01(1.15 - d_tip / d_arm)      # tip tucked against the hand -> 1
    curls["thumb"] = 0.45 * c_ang + 0.55 * c_dst
    return curls


FIST_CURL = 0.55          # a finger more curled than this counts as folded
FIST_MEAN = 0.62          # ...and the whole hand has to be closed on average
FIST_CURL_OFF = 0.46      # exit thresholds are looser: closing takes a firm
FIST_MEAN_OFF = 0.50      # squeeze, opening only needs a clear relax - no
                          # single-frame flicker right on the boundary


def is_fist(curls: dict, was_fist: bool = False) -> bool:
    """True when the four fingers are folded into the palm.

    The THUMB is deliberately ignored.  In a natural fist it lies flat across
    the folded fingers and reads moderately curled, but a fully open hand has
    the thumb out at ~0.1, so testing it would either reject every real fist
    or accept every relaxed hand.  One finger is allowed to disagree, since a
    single mis-placed landmark should not cancel the gesture.

    Hysteresis: once a fist is registered, it takes a clear relax (not just a
    wobble back across the same line) to release it - the same on/off split
    already used for pinches, so a fist held near the boundary doesn't chatter
    open/closed and re-fire the hold gesture.
    """
    curl_t = FIST_CURL_OFF if was_fist else FIST_CURL
    mean_t = FIST_MEAN_OFF if was_fist else FIST_MEAN
    vals = [curls["index"], curls["middle"], curls["ring"], curls["pinky"]]
    folded = sum(1 for v in vals if v > curl_t)
    return folded >= 3 and (sum(vals) / 4.0) > mean_t


def hand_features(landmarks, is_right: bool) -> dict:
    """Everything the gesture engine + renderers need for one hand."""
    pts = {i: _p3(landmarks, i) for i in range(21)}
    scale2 = hand_scale(landmarks) + 1e-6
    scale3 = float(np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])) or 1e-6
    curls = curl_features(landmarks)
    # 3D pinch distances (the z channel makes a real pinch read far tighter
    # than a flat 2D distance, which is what a grab needs)
    d_ti = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]))
    d_tm = float(np.linalg.norm(pts[THUMB_TIP] - pts[MIDDLE_TIP]))
    pinch2 = d_ti / scale3                     # thumb<->index only (grabbing)
    pinch3 = min(d_ti, d_tm) / scale3          # thumb to either of the two
    it = _pt(landmarks, INDEX_TIP)
    mt = _pt(landmarks, MIDDLE_TIP)
    tt = _pt(landmarks, THUMB_TIP)
    tips = [_pt(landmarks, i) for i in (INDEX_TIP, MIDDLE_TIP, 12, 16, 20)]
    spread = float(np.mean([np.linalg.norm(tips[i] - tips[i + 1]) for i in range(4)]))
    spread_n = spread / scale2
    # The real SPIDER-MAN web-shoot sign: MIDDLE + RING pressed down into the
    # palm - that is the hand actually hitting the trigger - with INDEX and
    # PINKY left extended.  That is the gesture people make, so it is the one
    # that has to be detected; the old index+middle version was a pinch that
    # merely looked vaguely similar and felt nothing like doing it for real.
    # Still unambiguous against the rest of the set: a fist folds all four, an
    # open palm folds none, and this folds exactly the middle two.
    webpose = (curls["index"] < 0.45 and curls["pinky"] < 0.55
               and curls["middle"] > 0.55 and curls["ring"] > 0.55
               and pinch3 < 0.45)
    return {
        "landmarks": landmarks,
        "is_right": bool(is_right),
        "curls": curls,
        "fist": is_fist(curls),
        "open": (curls["index"] < 0.45 and curls["middle"] < 0.50
                 and curls["ring"] < 0.50 and curls["pinky"] < 0.60
                 and spread_n > 0.25),
        "webpose": webpose,
        "pinch3": pinch3,          # thumb<->index/middle 3D distance ratio
        "pinch2": pinch2,          # thumb<->index 3D distance ratio (grab)
        "spread": spread_n,
        "palm": palm_center(landmarks),
        "wrist": _pt(landmarks, WRIST),
        "mcp9": _pt(landmarks, MIDDLE_MCP),
        "index_tip": it,
        "middle_tip": mt,
        "thumb_tip": tt,
    }


def classify_hands(result, mirrored: bool = True) -> list:
    """Turn a detection result into a list of per-hand feature dicts.

    Every hand gets an 'is_right' flag, decided from MediaPipe's OWN
    handedness label whenever it is confident, with screen position as the
    fallback.  Both conventions agree with each other and with the user here:

    * The main loop flips the frame before detection, so the model sees a
      MIRRORED (selfie) image - which is exactly the convention MediaPipe's
      handedness is defined for.  "Right" really is your right hand.
    * Position: in a mirrored frame, whatever is right of screen centre is
      your right hand.

    Position alone was the old scheme, and it silently breaks the moment the
    hands CROSS - which is precisely what the pinch-pull summon does.  The
    model's label comes from the hand's actual anatomy and survives crossing,
    so a confident label wins; when the label is weak the position keeps the
    answer sane instead of flipping frame to frame.
    """
    hands = list(result.hand_landmarks or []) if result is not None else []
    if not hands:
        return []
    handness = list(result.handedness or []) if result is not None else []
    xs = [float(np.mean([h[WRIST].x, h[MIDDLE_MCP].x])) for h in hands]
    out = []
    for i, h in enumerate(hands):
        cat = handness[i][0] if i < len(handness) and handness[i] else None
        mp_right = (cat.category_name == "Right") if cat is not None else None
        score = float(getattr(cat, "score", 0.0)) if cat is not None else 0.0
        if mp_right is not None and score >= 0.60:
            is_right = mp_right
        else:
            # mirrored: right of screen is your right hand (flip for raw input)
            is_right = (xs[i] > 0.5) if mirrored else (xs[i] < 0.5)
        out.append(hand_features(h, is_right))
    return out


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
      ('pinch', feat)    a hand closed thumb+index (or the web-shoot pose)
      ('pull', feat)     a pinching hand travelled ~one hand-length while
                         still pinched -> the summon gesture
      ('unpinch', feat)  the pinch opened again
      ('fist', feat)     ONE hand held a fist for FIST_HOLD and then OPENED,
                         with no other hand pinching.  Firing on the release
                         rather than on the hold is what makes it feel like
                         pulling a glove back on.
      ('gear', None)     BOTH hands held fists for GEAR_HOLD and then opened
                         -> toggle the holographic body gear.  Consumes the
                         whole fist cycle so the openers can't also fire the
                         single-fist re-wear gesture.

    Per-hand features are EMA-smoothed (matched by palm proximity), and hands
    that drop out for a few frames keep their state ("ghosting") so a tracking
    hiccup never cancels a gesture mid-way.
    """

    FIST_HOLD = 0.45        # seconds one fist must be held before releasing
    GEAR_HOLD = 0.55        # seconds BOTH fists must be held before releasing
    GRAB_ON = 0.42          # pinch3 below this -> pinch begins (3D, forgiving)
    GRAB_OFF = 0.58         # pinch3 above this -> pinch ends
    PINCH_OPEN = 0.62       # index must be no more curled than this to pinch
    PINCH_CONFIRM = 2       # frames of confirmation before a pinch edge
    GHOST_TTL = 0.60        # seconds a dropped hand keeps its state
    PULL_DIST = 0.60        # pull distance, in hand-lengths
    PULL_WINDOW = 2.5       # seconds a pinch stays eligible to become a pull
    PINCH_QUIET = 0.60      # seconds after a pinch that no fist may fire
    SENSE_VEL = 0.55        # hand-lengths/frame that sets the spider-sense off
    SENSE_QUIET = 0.70      # seconds before the sense can ping again

    def __init__(self):
        self._slots: list = []
        self._pinch_last = -9.0
        self._both_fist_since = None

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
                # SPIDER-SENSE: a sudden movement of the hand sets the sense
                # off.  Re-arming only after SENSE_QUIET keeps a fast sweep
                # from strobing - it pings once and settles, the way it reads
                # on screen: a warning, not a status light.
                if s["vel"] > self.SENSE_VEL and t - s["sense"] > self.SENSE_QUIET:
                    s["sense"] = t
        # prune hands gone for too long (identity compare: dicts hold arrays)
        self._slots = [s for s in self._slots
                       if s["matched"] or t - s["last_seen"] <= self.GHOST_TTL]

        live = [s for s in self._slots if t - s["last_seen"] <= self.GHOST_TTL]
        fists = [s for s in live if s["fist"]]

        # -- both-fists held then RELEASED -> toggle body gear --------------- #
        # (ghosted hands keep a held fist 'present' through a tracking blip)
        if len(fists) >= 2:
            if self._both_fist_since is None:
                self._both_fist_since = t
        elif self._both_fist_since is not None:
            held = t - self._both_fist_since
            self._both_fist_since = None
            if held >= self.GEAR_HOLD:
                events.append(("gear", None))
                # consume the whole fist cycle so the openers don't ALSO fire
                # the single-fist re-wear gesture
                for s in self._slots:
                    s["fist_fired"] = True
                    s.pop("fist_ready", None)

        for s in self._slots:
            if s["matched"]:
                self._pinch_edges(s, t, events)
        if any(s["pinch"] for s in live):
            self._pinch_last = t
        # -- single fist, held then RELEASED --------------------------------- #
        # Never while ANY hand is pinching, or just has been: taking a shooter
        # off means reaching across with one hand while the other holds still,
        # and a hand held still is usually half-closed.  Without this, trying
        # to detach a shooter re-wore it instead.
        if len(fists) <= 1 and t - self._pinch_last > self.PINCH_QUIET:
            for s in live:
                if s["fist"] or s.get("fist_fired") \
                        or not s.pop("fist_ready", False):
                    continue
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
                "open": bool(s.get("open", False)),
                "webpose": bool(s.get("webpose", 0.0) > 0.5),
                "gear_p": self._gear_progress(t),
                "fist_p": fist_p,
                "pull_p": pull_p,
                "sense": clamp01(1.0 - (t - s["sense"]) / self.SENSE_QUIET),
                # raw numbers for the debug readout (D key)
                "pinch3": float(s["pinch3"]),
                "curls": [float(c.get(k, 0.0)) for k in
                          ("index", "middle", "ring", "pinky", "thumb")],
            })
        return {"hands": hands}

    def _gear_progress(self, t: float) -> float:
        """0..1 charge of the both-fists -> gear gesture, for the HUD."""
        if self._both_fist_since is None:
            return 0.0
        return clamp01((t - self._both_fist_since) / self.GEAR_HOLD)

    # -- internals ----------------------------------------------------------- #
    def _match(self, h):
        best, bd = None, 0.5        # max normalised match distance
        for s in self._slots:
            if s["matched"]:
                continue
            d = float(np.linalg.norm(s["palm"] - h["palm"]))
            # A slot that believes it is the OTHER hand has to be much closer
            # to win.  Raw proximity alone happily swaps the two slots the
            # moment your hands cross - which is exactly what reaching over to
            # take a shooter off does - and every held gesture jumps hands
            # with them.  Anatomy is the tie-breaker proximity can't be.
            if s["is_right"] != h["is_right"]:
                d += 0.22
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
            "webpose": 0.0, "open": False, "scale": 0.1,
            "matched": True, "last_seen": t,
            "fist_since": None, "fist_fired": False, "pinch_ok": False,
            "pinch": False, "pinch_frames": 0,
            "pinch_px": h["palm"], "pinch_p0": None, "pinch_t0": 0.0,
            "pulled": False, "pull_frames": 0,
            "vel": 0.0, "sense": -9.0,
        }
        self._apply(s, h)
        return s

    def _apply(self, s, h):
        if s["curls"] is None:
            s["curls"] = dict(h["curls"])
            s["pinch3"] = h["pinch3"]
            s["pinch2"] = h["pinch2"]
            s["spread"] = h["spread"]
            s["webpose"] = 1.0 if h["webpose"] else 0.0
            s["open"] = bool(h["open"])
        else:
            # ADAPTIVE smoothing.  A hand holding still gets heavy averaging,
            # so landmark jitter can never trip a gesture on its own; a hand
            # that is MOVING gets almost none, so the gesture lands the frame
            # you make it instead of trailing the hand.  A single fixed alpha
            # has to trade one of those away for the other - this keeps both.
            vel = float(np.linalg.norm(h["palm"] - s["palm"])) \
                / max(s["scale"], 1e-3)
            s["vel"] = 0.6 * vel + 0.4 * s["vel"]
            a = 0.40 + 0.50 * clamp01(vel / 0.35)
            for k in ("thumb", "index", "middle", "ring", "pinky"):
                s["curls"][k] = a * h["curls"][k] + (1.0 - a) * s["curls"][k]
            s["pinch3"] = a * h["pinch3"] + (1.0 - a) * s["pinch3"]
            s["pinch2"] = a * h["pinch2"] + (1.0 - a) * s["pinch2"]
            s["spread"] = a * h["spread"] + (1.0 - a) * s["spread"]
            s["webpose"] = 0.55 * (1.0 if h["webpose"] else 0.0) \
                + 0.45 * s["webpose"]
            s["open"] = bool(h["open"])
        for k in ("palm", "wrist", "mcp9", "index_tip", "middle_tip",
                  "thumb_tip", "is_right", "landmarks"):
            s[k] = h[k]
        s["hand"] = h
        s["scale"] = float(np.linalg.norm(h["mcp9"] - h["wrist"])) or 0.1
        s["pinch_px"] = 0.5 * (h["thumb_tip"] + h["index_tip"])
        c = s["curls"]
        fist = is_fist(c, was_fist=s["fist"])
        s["fist"] = fist
        # a closed fist also puts the thumb next to the fingertips, so without
        # this gate every fist read as a pinch and pull-summoned a shooter
        s["pinch_ok"] = (not fist) and c["index"] < self.PINCH_OPEN
        if not fist:                    # opening the hand re-arms the gesture
            s["fist_since"] = None
            s["fist_fired"] = False

    def _pinch_edges(self, s, t, events):
        if s["fist"]:
            if s["fist_since"] is None and not s["fist_fired"]:
                s["fist_since"] = t
            # held long enough: armed, and it fires the moment the hand opens
            if s["fist_since"] is not None and t - s["fist_since"] >= self.FIST_HOLD:
                s["fist_ready"] = True
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
            ready = d >= self.PULL_DIST * max(s["scale"], 1e-3)
            # two consecutive frames past the distance, not one: a single
            # noisy landmark spike must not fire the summon on its own
            s["pull_frames"] = min(s["pull_frames"] + 1, 3) if ready else 0
            if s["pull_frames"] >= 2:
                s["pulled"] = True
                events.append(("pull", s["hand"]))


# --------------------------------------------------------------------------- #
# The holo desk: every object on screen, plus the grab-to-move rig
# --------------------------------------------------------------------------- #
def pinch_px(feat, w: int, h: int):
    p = 0.5 * (feat["thumb_tip"] + feat["index_tip"]) * np.array([w, h])
    return (float(p[0]), float(p[1]))


class HoloDesk:
    """Owns the shooters, the suit and the 4-second move lock."""

    DRAG_ARM = 1.5            # seconds a grab must be held before it moves
    MAX_SHOOTERS = 2          # one per wrist, and never more than that

    def __init__(self, log=print):
        # exactly two shooters exist, ever - one per wrist.  They are re-used,
        # never re-created, so no gesture can ever put a third on screen.
        self.shooters = {"Right": WebShooter("Right"), "Left": WebShooter("Left")}
        self._auto = {"Right": False, "Left": False}
        self.gear: SuitGear | None = None
        self.projectors: dict = {"Right": None, "Left": None}
        self.drags: dict = {"Right": None, "Left": None}
        self.hud = Hud()
        self.debug = False
        self.guide = False
        self._log = log

    # -- helpers ------------------------------------------------------------- #
    @staticmethod
    def _side(feat) -> str:
        return "Right" if feat["is_right"] else "Left"

    def _draggables(self):
        return [sh for sh in self.shooters.values() if sh.drag_point() is not None]

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
            # the way you would strip one off your wrist.  It plays a short
            # peel-away animation - popping off the forearm, then swinging
            # into your grip - no move-lock countdown, and from here on it is
            # its own object: it follows the pinch and stays where it's dropped.
            if isinstance(obj, WebShooter) and obj.detach(t, px):
                d["armed"] = True
                d["armed_t"] = t
                self._log(f"[gesture] pulled the {obj.side} shooter off - peeling away")
            return                                  # this pinch is a grab, not a pull
        # summoning wants the WEB-SHOOT sign (middle+ring pressed onto the palm
        # trigger, index+pinky out) - the one gesture that means "fire a shooter".
        # A tight thumb+index grab still works for people who can't hold it,
        # but a lazy thumb brush no longer summons one by accident.
        if not (feat.get("webpose") or feat.get("pinch2", 1.0) < 0.30):
            return
        target = "Left" if feat["is_right"] else "Right"
        # If that wrist is off-camera the shooter waits at the edge of the
        # frame on its own side - never floating in the middle of the view.
        park = (0.11 * w if target == "Left" else 0.89 * w, 0.34 * h)
        sh = self.shooters[target]
        if not sh.active() and self.live_shooters() >= self.MAX_SHOOTERS:
            return                                  # two on screen is the limit
        what = sh.toggle(t, px, park)
        self._log(f"[gesture] pinch-pull {side} -> {target} wrist shooter {what}")

    def on_unpinch(self, feat, w, h, t):
        self._end_drag(self._side(feat), t)

    def on_fist(self, feat, w, h, t):
        """Fist, held and released: puts that hand's shooter back on.

        Closing your hand and opening it again is how you'd pull a glove back
        over the wrist.  If the shooter is already on, the gesture simply has
        nothing to do - it is not overloaded with a second meaning.
        """
        side = self._side(feat)
        sh = self.shooters[side]
        if sh.state == "on":
            return
        wr = feat["wrist"] * np.array([w, h])
        if sh.reattach(t, (float(wr[0]), float(wr[1] + 0.25 * h))):
            self._auto[side] = True
            self._log(f"[gesture] fist + release -> {side} shooter back on")

    def on_gear(self, t):
        """Both fists, held and released: suit up / stand down."""
        if self.gear is None:
            self.gear = SuitGear()
            self._log("[gesture] both fists -> BODY GEAR ONLINE")
        else:
            self.gear.dismiss()
            self._log("[gesture] both fists -> body gear off")

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
    def live_shooters(self) -> int:
        return sum(1 for s in self.shooters.values() if s.active())

    def _autospawn(self, t, by_side, w, h):
        """Kit up the moment a wrist comes into frame.

        The shooter for a wrist arrives on its own the first time that hand is
        seen - no gesture needed.  It happens once per wrist: if you then send
        one away with pinch-pull, it stays away until you summon it back.
        """
        for side in ("Right", "Left"):
            feat = by_side[side]
            sh = self.shooters[side]
            if feat is None or self._auto[side] or sh.active():
                continue
            if self.live_shooters() >= self.MAX_SHOOTERS:
                continue
            wr = feat["wrist"] * np.array([w, h])
            sh.summon(t, (float(wr[0]), float(wr[1] + 0.22 * h)),
                      (float(wr[0]), float(wr[1])))
            self._auto[side] = True
            self._log(f"[auto] {side} wrist in frame -> shooter equipped")

    def _update_projectors(self, t, by_side, w, h):
        """Open, spread palm -> a hologram materialises above it.

        Continuous, not a toggle: the projector fades in while the palm stays
        open and fades out when it closes, so it never steals the gesture
        vocabulary - an open hand is the resting state, and the projection is
        the ambient Stark furniture.
        """
        for side in ("Right", "Left"):
            feat = by_side[side]
            p = self.projectors[side]
            if feat is not None and feat["open"] and not feat["fist"] \
                    and not feat["webpose"]:
                if p is None:
                    p = PalmProjector(side)
                    self.projectors[side] = p
                p.hold(t)
                p.track(feat, w, h)
            elif p is not None:
                p.release(t)
                if p.dead(t):
                    self.projectors[side] = None

    def update(self, dt, t, feats, w, h):
        by_side = {"Right": None, "Left": None}
        for f in feats:
            by_side[self._side(f)] = f
        self._autospawn(t, by_side, w, h)
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
        if self.gear is not None:
            self.gear.update(dt)
            if not self.gear.alive():
                self.gear = None
        self._update_projectors(t, by_side, w, h)

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
        gear_on = self.gear is not None and self.gear.dying is None
        palm = any(p is not None and not p.dead(t)
                   for p in self.projectors.values())
        return [("L WRIST", self.shooters["Left"].active(), 0.0),
                ("R WRIST", self.shooters["Right"].active(), 0.0),
                ("BODY GEAR", gear_on, 0.0),
                ("PALM AR", palm, 0.0),
                ("MOVE LOCK", armed, lock)]

    # -- render -------------------------------------------------------------- #
    @staticmethod
    def _spider_sense(ov, k, state):
        """The spider-sense ping: a sharp move sets off a radial warning.

        Three offset arcs snapping outward from the palm rather than one clean
        ring - the sense is a jolt, not a readout, and a broken ring reads that
        way where a solid circle would just look like another UI element.
        """
        oh, ow = ov.shape[:2]
        for hnd in state["hands"]:
            f = hnd.get("sense", 0.0)
            if f <= 0.02:
                continue
            # palm/size arrive NORMALISED, so they map straight onto the
            # half-res overlay by its own dimensions - no k correction needed
            cx, cy = int(hnd["palm"][0] * ow), int(hnd["palm"][1] * oh)
            base = max(8.0, hnd["size"] * ow)
            grow = 1.0 - f                      # 0 at the ping, 1 as it dies
            for i in range(3):
                r = base * (0.55 + 1.7 * grow + 0.22 * i)
                a0 = -60 + i * 120 + int(126 * grow)
                cv2.ellipse(ov, (cx, cy), (int(r), int(r * 0.82)), 0,
                            a0, a0 + 78, _dim(HOLO_CYAN, 0.85 * f), 2, cv2.LINE_AA)

    def _status(self, t, state):
        """A one-line JARVIS-style status readout built from REAL state only."""
        if self.gear is not None and self.gear.dying is None:
            return "BODY GEAR ONLINE  //  ALL SYSTEMS NOMINAL"
        for hnd in state["hands"]:
            if hnd.get("webpose"):
                return "WEB-SHOOT POSE LOCKED  //  PINCH + PULL TO FIRE"
            if hnd.get("open") and hnd.get("pinch"):
                return "GRAB LOCKED  //  HOLD 4s TO ARM MOVE"
        for hnd in state["hands"]:
            if hnd.get("open"):
                return "PALM HOLOGRAM READY  //  OPEN PALM PROJECTS"
        return "STANDBY  //  SHOW YOUR HANDS TO THE CAMERA"

    def draw(self, frame, t, feats, tracker, fps, pose=None):
        h, w = frame.shape[:2]
        k = 0.5
        state = tracker.hud_state(t)
        ov = np.zeros((max(2, int(h * k)), max(2, int(w * k)), 3), np.uint8)
        if self.gear is not None:           # the body gear sits behind everything
            self.gear.draw(ov, k, t, feats, pose)
        for side in ("Right", "Left"):
            self.shooters[side].draw(ov, k, t)
        for p in self.projectors.values():  # palm holograms sit on top
            if p is not None:
                p.draw(ov, k, t)
        for f in feats:                       # skeleton tracing on every hand
            if f["landmarks"] is not None:
                draw_hand_holo(ov, f["landmarks"], k, t)
        self._spider_sense(ov, k, state)
        # the bilinear upscale from half-res IS the bloom - no blur pass needed
        bloom = cv2.resize(ov, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(bloom, 0.72, frame, 1.0, 0, frame)
        # the wrist shooters and the palm holograms are small, so they get a
        # crisp full-resolution pass of their own once the bloom is down
        for side in ("Right", "Left"):
            self.shooters[side].draw_sharp(frame, t)
        for p in self.projectors.values():
            if p is not None:
                p.draw_sharp(frame, t)
        if self.gear is not None:           # chest reactor glows at full res
            self.gear.draw_sharp(frame, t)
        self.hud.draw(frame, {"t": t, "fps": fps, "hand_count": len(feats),
                              "hands": state["hands"],
                              "drags": self.drag_hud(t), "chips": self.chips(t),
                              "debug": self.debug, "guide": self.guide,
                              "status": self._status(t, state),
                              "gear_p": max((hnd.get("gear_p", 0.0)
                                              for hnd in state["hands"]),
                                             default=0.0),
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
    # Body tracking for the suit.  Non-blocking by design: the model downloads
    # on a background thread, so a fresh install starts on the hand-anchored
    # fallback and silently upgrades to a real torso fit a second or two later
    # rather than making everyone wait at a black screen.
    pose_tracker = PoseTracker()

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
        pose_tracker.close()            # join the pose worker before we exit
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

            # Body pose, for fitting the suit to the real torso.  This has to
            # run BEFORE anything is drawn onto the frame - the tracker hands
            # the frame to a worker thread, and a frame with holograms already
            # painted on it would have the app tracking its own output.
            try:
                pose = pose_tracker.detect(frame, now_ms)
            except Exception as exc:  # noqa: BLE001 - pose is never load-bearing
                print(f"[pose] error: {exc}")
                pose = None

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
            desk.draw(frame, now, feats, tracker, fps, pose)

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
                elif key in (ord("g"), ord("G")):
                    desk.guide = not desk.guide
                elif key in (ord("r"), ord("R")):
                    desk.on_gear(now)

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
def _fake_feat(is_right=True, fist=False, pinch3=None, hand_id=0.0, dx=0.0, dy=0.0,
               open_palm=False):
    """Synthetic feature dict for deterministic gesture-engine tests.

    Curl values are the NEW metric: 1 = fully folded (a real fist reads
    ~0.85 on all four fingers with the thumb barely curled, and the thumb tip
    sits right next to the fingertips - a low pinch distance).
    """
    if fist:
        curls = dict(thumb=0.55, index=0.85, middle=0.85, ring=0.85, pinky=0.85)
        spread = 0.30
        webpose = False
        open_ = False
    elif open_palm:
        curls = dict(thumb=0.15, index=0.15, middle=0.15, ring=0.15, pinky=0.15)
        spread = 0.75
        webpose = False
        open_ = True
    else:                                   # the web-shoot trigger pose
        curls = dict(thumb=0.55, index=0.15, middle=0.80, ring=0.80, pinky=0.20)
        spread = 0.45
        webpose = True
        open_ = False
    if pinch3 is None:
        pinch3 = 0.15 if fist else (0.90 if open_palm else 0.12)
    pinch2 = 0.10 if not fist and not open_palm else 0.90
    x = 0.3 + hand_id + dx
    y = dy
    return {
        "landmarks": None, "is_right": bool(is_right), "curls": curls,
        "fist": fist, "open": open_, "webpose": webpose,
        "pinch3": pinch3, "pinch2": pinch2, "spread": spread,
        "palm": np.array([x, 0.50 + y]), "wrist": np.array([x, 0.75 + y]),
        "mcp9": np.array([x, 0.62 + y]), "index_tip": np.array([x, 0.44 + y]),
        "middle_tip": np.array([x + 0.05, 0.44 + y]),
        "thumb_tip": np.array([x, 0.46 + y]),
    }


def _synthetic_hand(mode="open", scale=0.15):
    """Build 21 landmark objects for a hand in one of three clean poses.

    Used to validate the angle-based curl metric on real geometry.  Landmarks
    are (x, y) in normalised space; z stays 0 for all of them.
    """
    class _LM:
        __slots__ = ("x", "y", "z")

        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    lm = [
        _LM(0.50, 0.85),                        # 0 wrist
        _LM(0.46, 0.80), _LM(0.45, 0.78),       # 1-4 thumb (cmc, mcp, ip, tip)
        _LM(0.44, 0.76), _LM(0.43, 0.74),
    ]

    def finger(mcp, straight):
        """Four joints (mcp, pip, dip, tip); straight = 180-degree chain."""
        x, y = mcp
        if straight:
            return [(x, y), (x, y - 0.07), (x, y - 0.13), (x, y - 0.19)]
        # folded: dip + tip collapse back toward the palm
        return [(x, y), (x, y - 0.04), (x - 0.03, y - 0.02), (x - 0.05, y + 0.02)]

    # index, middle, ring, pinky - the web-shoot sign folds the MIDDLE TWO
    # onto the palm trigger and leaves index and pinky out
    fold = {"open": [False, False, False, False],
            "webpose": [False, True, True, False],
            "fist": [True, True, True, True]}[mode]
    for i, mcp in enumerate([(0.44, 0.68), (0.50, 0.68), (0.56, 0.68),
                             (0.62, 0.70)]):
        for (x, y) in finger(mcp, not fold[i]):
            lm.append(_LM(x, y))                # 5-20 in MediaPipe order
    if mode == "open":                          # thumb out wide, clear of index
        lm[THUMB_TIP] = _LM(0.40, 0.60)
    elif mode == "webpose":                     # thumb lies over the folded middle
        lm[THUMB_IP] = _LM(0.47, 0.72)
        lm[THUMB_TIP2] = _LM(0.47, 0.66)
    else:                                       # fist: thumb across the fingers
        lm[THUMB_IP] = _LM(0.50, 0.63)
        lm[THUMB_TIP2] = _LM(0.53, 0.58)
    return lm


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
    print("[ok] HandLandmarker created (VIDEO mode, 0.20/0.20/0.40 confidence)")

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
    for name, mesh in (("shooter", HM.shooter_mesh()),
                       ("worn LOD", HM.shooter_mesh(0)),
                       ("gauntlet", HM.gauntlet_mesh()),
                       ("suit", HM.suit_mesh()),
                       ("reactor", HM.reactor_mesh()),
                       ("palm screen", HM.palm_screen_mesh())):
        assert mesh.V is not None and len(mesh.FI) > 40, f"{name} mesh too coarse"
        canvas = np.zeros((240, 320, 3), np.uint8)
        H3.render_mesh(canvas, mesh, (160, 120), 60, H3.rot_matrix(0.6, -0.3, 0.2))
        lit = int((canvas.sum(axis=2) > 120).sum())
        assert lit > 200, f"{name} mesh drew almost nothing ({lit} px)"
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
    print("[ok] shooter: an armed grab carries it, release parks it in place")

    # 5a. the mount spring: a worn shooter TRAILS the wrist and settles onto it
    #     without bouncing, and does the same at any framerate
    def _wrist_at(px):
        class _L:
            __slots__ = ("x", "y", "z")

            def __init__(s, x, y):
                s.x, s.y, s.z = x, y, 0.0
        lms = [_L(px, 0.75)] + [_L(0.0, 0.0)] * 8 + [_L(px, 0.62)] \
            + [_L(0.0, 0.0)] * 11
        return {"wrist": np.array([px, 0.75]), "mcp9": np.array([px, 0.62]),
                "thumb_tip": np.array([px + 0.05, 0.70]), "landmarks": lms,
                "webpose": False}

    def _settle_run(step, secs, first=0.5, then=0.8):
        s = WebShooter("Right")
        s.summon(0.0, (300.0, 300.0), (300.0, 300.0))
        tt = 0.0
        for _ in range(int(1.4 / step)):        # ride the arrive out, then rest
            tt += step
            s.update(step, tt, _wrist_at(first), 640, 480)
        trail = []
        for _ in range(int(secs / step)):
            tt += step
            s.update(step, tt, _wrist_at(then), 640, 480)
            trail.append(s.pos[0])
        return s, trail

    sh_s, trail = _settle_run(1 / 30.0, 0.4)
    tgt = 0.8 * 640.0
    assert sh_s.state == "on", "the spring test needs a worn shooter"
    assert abs(trail[0] - tgt) > 20.0, \
        "a worn shooter must TRAIL a jerked wrist, not teleport with it"
    assert abs(trail[-1] - tgt) < 1.0, "...and settle onto the wrist"
    assert max(t_ - tgt for t_ in trail) < 1.0, \
        "critical damping means it must never overshoot and bounce"
    _, fast = _settle_run(1 / 60.0, 0.4)
    assert abs(trail[-1] - fast[-1]) < 1.0, \
        "the strap must settle the same at 30 and 60 fps, not stiffen with the framerate"
    print("[ok] shooter: strapped-on spring trails the wrist, settles, no bounce")

    # 5b. firing a web kicks the housing back along the arm, then it recovers
    sh_r2 = WebShooter("Right")
    sh_r2.summon(0.0, (300.0, 300.0), (300.0, 300.0))
    tr2 = 0.0
    for _ in range(45):                        # settle it onto the wrist first
        tr2 += 1 / 30.0
        sh_r2.update(1 / 30.0, tr2, _wrist_at(0.5), 640, 480)
    rest_y = sh_r2.pos[1]
    kick = []
    for i in range(14):                        # the web-shoot pose fires once
        tr2 += 1 / 30.0
        f = _wrist_at(0.5)
        f["webpose"] = (i == 0)
        sh_r2.update(1 / 30.0, tr2, f, 640, 480)
        kick.append(sh_r2.pos[1])
    assert max(abs(y - rest_y) for y in kick) > 5.0, \
        "firing a web must kick the housing back along the arm"
    assert abs(kick[-1] - rest_y) < 1.0, "...and the strap must pull it back to rest"
    print("[ok] shooter: firing a web recoils the housing, the strap recovers it")

    # 6. the shooter is a compact wrist launcher: it must NOT run far back up
    #    the forearm like a full gauntlet
    mesh = HM.shooter_mesh()
    xs = mesh.V[:, 0]
    assert -1.5 < xs.min() < -1.0, "the housing must stay a short way up the wrist"
    assert 0.9 < xs.max() < 1.8, "the wrap must carry over onto the hand"
    assert abs(xs.min()) < 1.3 * abs(xs.max()), \
        "the shooter must read as compact, not a forearm-length gauntlet"
    print(f"[ok] shooter is a compact wrist launcher (x {xs.min():.2f}..{xs.max():.2f}, "
          f"origin at the wrist)")

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

    # 9. tracker: one fist -> re-wear a detached shooter, but never while a
    #    hand is pinching
    #    (curl values are the new metric: 1 = folded)
    assert is_fist(dict(thumb=0.55, index=0.80, middle=0.80, ring=0.85, pinky=0.85)), \
        "a real fist folds all four fingers - it must count"
    assert not is_fist(dict(thumb=0.15, index=0.15, middle=0.15, ring=0.15, pinky=0.15)), \
        "an open hand is not a fist"
    assert not is_fist(dict(thumb=0.55, index=0.20, middle=0.80, ring=0.80, pinky=0.20)), \
        "the web-shoot sign (middle+ring on the trigger) is not a fist"
    tr3 = GestureTracker()
    evs = []
    for i in range(24):                        # 0.8 s closed: armed, not fired
        evs += tr3.feed([_fake_feat(True, fist=True)], i / 30.0)
    assert not [e for e in evs if e[0] == "fist"], \
        "a fist that is still closed must not fire - it fires on the release"
    for i in range(4):                         # ...and now open the hand
        evs += tr3.feed([_fake_feat(True, fist=False, pinch3=0.9)], 0.8 + i / 30.0)
    fists = [e for e in evs if e[0] == "fist"]
    assert len(fists) == 1, f"one fist must fire once per cycle (got {len(fists)})"
    short = GestureTracker()
    evs2 = []
    for i in range(6):                         # 0.2 s: too short
        evs2 += short.feed([_fake_feat(True, fist=True)], i / 30.0)
    for i in range(4):
        evs2 += short.feed([_fake_feat(True, fist=False, pinch3=0.9)], 0.2 + i / 30.0)
    assert not [e for e in evs2 if e[0] == "fist"], "a quick fist must not fire"
    # a fist puts the thumb on the fingertips: it must never read as a pinch,
    # and a fist swung across the frame must never summon a shooter
    assert not [e for e in evs if e[0] in ("pinch", "pull")], \
        "a fist must not register as a pinch"
    tr3b = GestureTracker()
    evs = []
    for i in range(24):
        evs += tr3b.feed([_fake_feat(True, fist=True, dx=0.03 * i)], i / 30.0)
    assert not [e for e in evs if e[0] == "pull"], "a moving fist must not summon"
    # reaching across to take a shooter off must NOT also fire the re-wear
    # gesture: one hand pinches while the other rests half-closed
    tr4 = GestureTracker()
    evs = []
    for i in range(40):
        evs += tr4.feed([_fake_feat(True, pinch3=0.10, dx=0.02 * i),
                         _fake_feat(False, fist=True, hand_id=0.45)], i / 30.0)
    assert [e for e in evs if e[0] == "pull"], "the reaching hand must still pull"
    assert not [e for e in evs if e[0] == "fist"], \
        "a fist must never fire the re-wear gesture while the other hand is pinching"
    print("[ok] tracker: one fist -> re-wear, never while a hand is pinching")

    # 9b. spider-sense: a sharp move pings, a slow drift never does
    trs = GestureTracker()
    for i in range(6):                         # settle, barely moving
        trs.feed([_fake_feat(True, open_palm=True, dx=0.002 * i)], i / 30.0)
    assert trs.hud_state(0.2)["hands"][0]["sense"] <= 0.02, \
        "a hand drifting slowly must never set the spider-sense off"
    trs.feed([_fake_feat(True, open_palm=True, dx=0.20)], 0.24)
    lit = trs.hud_state(0.24)["hands"][0]["sense"]
    assert lit > 0.9, f"a sudden jump of the hand must ping the sense (got {lit})"
    # ...and it fades out again rather than latching on
    later = trs.hud_state(0.24 + 0.5)["hands"][0]["sense"]
    assert later < 0.4, f"the ping must decay, not latch (got {later})"
    print("[ok] spider-sense: sharp moves ping, slow drift does not")

    # 9c. crossing hands must not swap tracker SLOTS.  Slots carry the held
    #     gesture state, so a swap mid-cross hands your fist timer to the other
    #     hand and the gesture silently dies - which is what reaching across to
    #     take a shooter off looks like to the matcher.
    trx = GestureTracker()
    for i in range(12):                        # LEFT holds a fist, RIGHT open
        trx.feed([_fake_feat(False, fist=True, hand_id=0.0),
                  _fake_feat(True, open_palm=True, hand_id=0.35)], i / 30.0)
    for i in range(12):                        # ...and now they cross over
        f = i / 11.0
        trx.feed([_fake_feat(False, fist=True, hand_id=0.35 * f),
                  _fake_feat(True, open_palm=True, hand_id=0.35 - 0.35 * f)],
                 0.4 + i / 30.0)
    evs = []
    for i in range(4):                         # the LEFT hand opens again
        evs += trx.feed([_fake_feat(False, open_palm=True, hand_id=0.35),
                         _fake_feat(True, open_palm=True, hand_id=0.0)],
                        0.8 + i / 30.0)
    fires = [e for e in evs if e[0] == "fist"]
    assert len(fires) == 1 and not fires[0][1]["is_right"], \
        "a held fist must survive the hands crossing and fire on its OWN hand"
    print("[ok] tracker: held gestures survive the hands crossing over")

    # 9a. the angle-based curl metric on real joint geometry: straight fingers
    #     read open, folded fingers read closed, and the poses are distinct
    hm = hand_features(_synthetic_hand("open"), True)
    assert all(hm["curls"][k] < 0.25 for k in
               ("index", "middle", "ring", "pinky")), \
        f"straight fingers must read as open (got {hm['curls']})"
    assert hm["open"] and not hm["fist"] and not hm["webpose"], \
        "an open hand must be open-only"
    hf = hand_features(_synthetic_hand("fist"), True)
    assert hf["fist"] and not hf["open"] and not hf["webpose"], \
        "a folded hand must read as a fist only"
    assert all(hf["curls"][k] > 0.55 for k in
               ("index", "middle", "ring", "pinky")), \
        f"folded fingers must read as curled (got {hf['curls']})"
    hw = hand_features(_synthetic_hand("webpose"), True)
    assert hw["webpose"] and not hw["fist"] and not hw["open"], \
        "the web-shoot pose must be its own thing"
    assert hw["pinch3"] < 0.45, "the web-shoot pose tucks the thumb onto the tips"
    print("[ok] angle metric: straight vs folded vs web-shoot pose all distinct")

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

    # 10a. a wrist coming into frame equips itself, and only ever two exist
    desk_a = HoloDesk(log=lambda *a: None)
    assert desk_a.live_shooters() == 0, "nothing is equipped before a hand is seen"
    desk_a.update(1 / 30.0, 0.0, feats, 640, 480)
    assert desk_a.live_shooters() == 2, "both wrists must equip on sight"
    for i in range(30):
        desk_a.update(1 / 30.0, i / 30.0, feats, 640, 480)
    pf = _fake_feat(True, pinch3=0.10)
    for extra in range(3):                     # keep pulling: still only two
        desk_a.handle([("pull", pf)], 640, 480, 2.0 + extra)
        desk_a.update(1 / 30.0, 2.0 + extra, feats, 640, 480)
        assert desk_a.live_shooters() <= HoloDesk.MAX_SHOOTERS, \
            "never more than two shooters on screen"
    print("[ok] desk: wrists equip on sight, and only ever two shooters exist")

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
    assert sh_r.state == "peel", "pulling a worn shooter must start the peel-away"
    off = _fake_feat(True, pinch3=0.10)
    off["thumb_tip"] = np.array([0.25, 0.25])
    off["index_tip"] = off["thumb_tip"].copy()
    tp = 1.02
    while sh_r.state == "peel":                # ride out the peel animation
        tp += 1 / 30.0
        desk3.update(1 / 30.0, tp, [off], 640, 480)
    assert sh_r.state == "held", "the peel must settle into a held shooter"
    desk3.update(1 / 30.0, tp + 1 / 30.0, [off], 640, 480)
    assert abs(sh_r.pos[0] - worn[0]) > 40.0, "a detached shooter follows the hand"
    desk3.handle([("unpinch", off)], 640, 480, 1.2)
    dropped = sh_r.pos
    desk3.update(1 / 30.0, 1.3, [right], 640, 480)
    assert sh_r.state == "float", "a detached shooter must not jump back on"
    assert abs(sh_r.pos[0] - dropped[0]) < 1.0, "it stays where it was dropped"
    print("[ok] desk: pinch a worn shooter + pull -> it comes off as its own object")

    # 11. desk: a 1.5 s grab arms the move lock, then the object follows the hand
    desk2 = HoloDesk(log=lambda *a: None)
    # the fist+release puts the missing shooter back on...
    desk2.handle([("fist", _fake_feat(True, fist=True))], 640, 480, 0.0)
    assert desk2.shooters["Right"].state == "arrive", \
        "fist + release must put the shooter back on that wrist"
    for i in range(30):                        # let it fly on and mount
        desk2.update(1 / 30.0, i / 30.0, [right], 640, 480)
    sh2 = desk2.shooters["Right"]
    assert sh2.state == "on", f"the shooter must be worn first (got {sh2.state})"
    worn2 = sh2.pos
    # pinch it and pull it off so it becomes its own free-floating object
    grab = _fake_feat(True, pinch3=0.10)
    grab["thumb_tip"] = np.array([worn2[0] / 640.0, worn2[1] / 480.0])
    grab["index_tip"] = grab["thumb_tip"].copy()
    desk2.handle([("pinch", grab)], 640, 480, 1.0)
    assert desk2.drags["Right"]["obj"] is sh2, "the pinch must catch the worn shooter"
    desk2.handle([("pull", grab)], 640, 480, 1.02)
    tp2 = 1.02
    while sh2.state == "peel":                 # ride out the peel animation
        tp2 += 1 / 30.0
        desk2.update(1 / 30.0, tp2, [grab], 640, 480)
    desk2.handle([("unpinch", grab)], 640, 480, tp2 + 0.02)
    desk2.update(1 / 30.0, tp2 + 0.04, [], 640, 480)
    assert sh2.state == "float", "a released detached shooter must sit and wait"
    bp_pos = sh2.pos
    grab2 = _fake_feat(True, pinch3=0.10)
    grab2["thumb_tip"] = np.array([bp_pos[0] / 640.0, bp_pos[1] / 480.0])
    grab2["index_tip"] = grab2["thumb_tip"].copy()
    t0 = tp2 + 0.1
    desk2.handle([("pinch", grab2)], 640, 480, t0)
    assert desk2.drags["Right"]["obj"] is sh2, "the grab must capture it"
    desk2.update(1 / 30.0, t0 + 0.8, [grab2], 640, 480)
    assert not desk2.drags["Right"]["armed"], "0.8 s is not enough to move it"
    # a parked shooter bobs gently in place (holographic idle motion) - only
    # its x must stay put while unarmed, the y bob is expected
    assert abs(sh2.pos[0] - bp_pos[0]) < 1.0, "an unarmed grab must not move anything"
    hud = desk2.drag_hud(t0 + 0.8)
    assert hud and 0.4 < hud[0]["progress"] < 0.6, "the lock ring must show progress"
    desk2.update(1 / 30.0, t0 + 1.7, [grab2], 640, 480)
    assert desk2.drags["Right"]["armed"], "1.5 s must arm the move lock"
    moved = _fake_feat(True, pinch3=0.10)
    moved["thumb_tip"] = np.array([0.2, 0.2])
    moved["index_tip"] = np.array([0.2, 0.2])
    desk2.update(1 / 30.0, t0 + 1.9, [moved], 640, 480)
    assert abs(sh2.pos[0] - bp_pos[0]) > 40.0, "an armed grab must move it"
    here = sh2.pos
    desk2.handle([("unpinch", moved)], 640, 480, t0 + 2.0)
    desk2.update(1 / 30.0, t0 + 2.1, [moved], 640, 480)
    assert abs(sh2.pos[0] - here[0]) < 1.0 and desk2.drags["Right"] is None, \
        "releasing must simply leave the object where it is"
    print("[ok] desk: grab 1.5 s -> move lock -> object follows -> release parks it")

    # 11a. BOTH fists held then opened -> exactly one 'gear' event, and the
    #      openers are consumed so they never ALSO fire the single-fist re-wear
    trg = GestureTracker()
    evs = []
    for i in range(24):                        # 0.8 s of both fists
        evs += trg.feed([_fake_feat(True, fist=True),
                         _fake_feat(False, fist=True, hand_id=0.45)], i / 30.0)
    assert not [e for e in evs if e[0] == "gear"], \
        "gear fires on the release, not while held"
    assert not [e for e in evs if e[0] == "fist"], \
        "both fists must never fire the single-fist gesture"
    for i in range(4):                         # ...and both open
        evs += trg.feed([_fake_feat(True, open_palm=True),
                         _fake_feat(False, open_palm=True, hand_id=0.45)],
                        0.8 + i / 30.0)
    gears = [e for e in evs if e[0] == "gear"]
    assert len(gears) == 1, f"both fists must fire gear exactly once (got {len(gears)})"
    assert not [e for e in evs if e[0] == "fist"], \
        "the gear release must consume the fist cycle"
    # a short both-fist flash must not fire gear
    trgs = GestureTracker()
    evs = []
    for i in range(6):                         # 0.2 s: too short
        evs += trgs.feed([_fake_feat(True, fist=True),
                          _fake_feat(False, fist=True, hand_id=0.45)], i / 30.0)
    for i in range(4):
        evs += trgs.feed([_fake_feat(True, open_palm=True),
                          _fake_feat(False, open_palm=True, hand_id=0.45)],
                         0.2 + i / 30.0)
    assert not [e for e in evs if e[0] == "gear"], "a quick both-fist must not fire"
    # and the desk toggles the full body gear on and off
    dg = HoloDesk(log=lambda *a: None)
    dg.handle([("gear", None)], 640, 480, 0.0)
    assert dg.gear is not None and dg.gear.dying is None, "gear must come online"
    dg.update(1 / 30.0, 0.2, feats, 640, 480)
    ov = np.zeros((240, 320, 3), np.uint8)
    dg.gear.draw(ov, 0.5, 0.2, feats)
    assert ov.any(), "the body gear must draw into the overlay"
    full = np.zeros((480, 640, 3), np.uint8)
    dg.gear.draw_sharp(full, 0.2)
    assert full.any(), "the chest reactor must draw at full resolution"
    dg.handle([("gear", None)], 640, 480, 1.0)
    assert dg.gear.dying is not None, "gear must stand down"
    for i in range(22):                        # let the fade run out
        dg.update(1 / 30.0, 1.0 + i / 30.0, feats, 640, 480)
    assert dg.gear is None, "a dismissed gear must expire"
    print("[ok] gear: both-fists -> body gear on/off, one event, no re-wear")

    # 11b. body pose is an ENHANCEMENT, never a dependency.  Offline, still
    #      downloading, or no body in frame, the suit must fall back to hand
    #      anchoring rather than the app falling over.
    import holo_pose as HP
    pt_off = HP.PoseTracker(auto_download=False, threaded=True)
    blank = np.zeros((240, 320, 3), np.uint8)
    assert all(pt_off.detect(blank, i * 33) is None for i in range(8)), \
        "with no model, pose must report None rather than raising"
    pt_off.close()
    gear_np = SuitGear()
    for _ in range(40):
        gear_np.update(1 / 30.0)
    ov_np = np.zeros((240, 320, 3), np.uint8)
    gear_np.draw(ov_np, 0.5, 1.5, [right], None)     # explicit "no pose"
    gear_np.draw(ov_np, 0.5, 1.5, [right])           # and the old 4-arg call
    assert ov_np.any(), "the suit must still render on the hand-anchored fallback"
    assert ov_np[:, :, 0].sum() > ov_np[:, :, 2].sum(), "...and stay blue"
    print("[ok] pose: absent/offline degrades to hand anchoring, never crashes")

    # 11b. an open palm starts a projection, it tracks the hand, and closing
    #      the hand retires it
    dp = HoloDesk(log=lambda *a: None)
    op = _fake_feat(True, open_palm=True)
    for i in range(20):                        # hold the palm open 0.67 s
        dp.update(1 / 30.0, i / 30.0, [op], 640, 480)
    proj = dp.projectors["Right"]
    assert proj is not None, "an open palm must start a projection"
    assert proj._alpha(0.7) > 0.9, "a held-open palm must fully project"
    ov = np.zeros((240, 320, 3), np.uint8)
    proj.draw(ov, 0.5, 0.7)
    assert ov.any(), "the projection must draw its beam + ring"
    full = np.zeros((480, 640, 3), np.uint8)
    proj.draw_sharp(full, 0.7)
    assert full.any(), "the projection must render full-res"
    for i in range(15):                        # close the hand
        dp.update(1 / 30.0, 0.7 + i / 30.0, [_fake_feat(True, fist=True)], 640, 480)
    assert dp.projectors["Right"] is None, "a closed palm must retire the projection"
    print("[ok] palm projection: open palm -> hologram -> fades on close")

    # 11c. summoning requires the web-shoot pose (or a tight thumb+index); a
    #      lazy pinch that never forms the pose must not summon anything
    ds = HoloDesk(log=lambda *a: None)
    lazy = _fake_feat(True, pinch3=0.35)      # pinchy but NOT the web-shoot pose
    lazy["webpose"] = False
    lazy["pinch2"] = 0.50
    ds.handle([("pinch", lazy), ("pull", lazy)], 640, 480, 0.0)
    ds.update(1 / 30.0, 0.1, [], 640, 480)
    assert not ds.shooters["Left"].active(), \
        "a non-web-shoot pinch must not summon a shooter"
    tight = _fake_feat(True, pinch3=0.10)     # the proper pose
    ds.handle([("pinch", tight), ("pull", tight)], 640, 480, 0.2)
    ds.update(1 / 30.0, 0.3, [], 640, 480)
    assert ds.shooters["Left"].active(), "the web-shoot pose must summon"
    print("[ok] summon gate: web-shoot pose summons, a lazy pinch does not")

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
