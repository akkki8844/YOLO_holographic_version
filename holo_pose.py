#!/usr/bin/env python3
"""holo_pose.py - real body tracking, so the suit fits the person wearing it.

The gear used to be anchored off the HANDS: mean wrist, shoved down by a
couple of hand-lengths, scaled by hand size.  That guess drifts the moment you
move your arms, and it can never know how wide your shoulders actually are or
which way your chest is facing.  This module runs MediaPipe Tasks
PoseLandmarker alongside the hand landmarker and hands SuitGear a real torso
frame: shoulder midpoint, hip midpoint, shoulder width, torso length, plus a
roll and a yaw read off the shoulder line.

Three things matter more than accuracy here:

* It must be CHEAP.  A pose-lite detection measures ~55 ms here - nearly two
  whole frames - so running it inline would stutter the camera every time it
  fired, no matter how rarely.  Two defences: it runs at a reduced cadence
  (POSE_EVERY), and it runs on a WORKER THREAD, so the main loop only ever
  pays for one small resize and reads whatever the worker last produced.  The
  torso frame is then smoothed continuously on every call, so the suit keeps
  easing toward the newest measurement on the frames in between and reads as
  smooth motion rather than a 10 Hz staircase.  A tenth of a second of extra
  latency on a chest is invisible; the same latency on a gesture would not be,
  which is exactly why the HANDS stay synchronous and only pose goes async.
* It must never be load-bearing.  No model, no network, no body in shot: every
  entry point returns None and the caller falls back to the old hand-derived
  anchor.  Pose is an enhancement, never a dependency.
* It must not stall startup.  The model is 5.8 MB and downloads on first run,
  so the download happens on a background thread and pose simply switches on a
  second or two into the session instead of freezing the camera preview.
"""

from __future__ import annotations

import math
import os
import shutil
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - the app already handles this for hands
    mp = None

# same models/ directory and same download-on-first-use pattern as hand_zoom.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# BlazePose 33-point topology: the six joints that define a torso
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

# Post a frame to the worker every 3rd main-loop frame.  At ~30 fps that asks
# for a 10 Hz pose update, which the ~55 ms worker can comfortably keep up with
# (one job every ~100 ms) so jobs never queue and the extra core is left mostly
# idle for the hand tracker.  10 Hz on a torso is plenty: shoulders do not move
# like fingertips, and the smoothing below fills the gaps.  Every 2nd frame
# would saturate the worker for no visible gain; every 5th starts to feel like
# the suit is being dragged behind you when you turn quickly.
POSE_EVERY = 3

# The pose model resizes to 256x256 internally, so feeding it 720p only pays
# for a bigger colour convert.  480 wide keeps a whole standing body legible
# while cutting the pre-processing cost - same trick as detect_frame() uses.
POSE_WIDTH = 480

# A body that has not been seen for this long stops driving the suit, and the
# confidence ramps down over the fade so the suit eases back to the hand anchor
# instead of snapping.
STALE_HOLD = 0.45     # seconds of full-confidence coasting after a lost body
STALE_FADE = 0.55     # seconds to fade confidence to zero after that

# smoothing time constants, seconds (alpha = 1 - exp(-dt / tau)).  Position is
# the most visible channel so it is the most responsive; scale is the least
# trustworthy measurement (it jumps whenever a shoulder flickers) so it is the
# laziest.
TAU_POS = 0.07
TAU_SIZE = 0.20
TAU_ANG = 0.11


def _visible(lm) -> float:
    """Visibility of one landmark, tolerating builds that omit the field."""
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else float(v)


class PoseTracker:
    """Reduced-cadence body tracker returning a smoothed torso frame.

    Usage from the main loop (all of it optional, all of it failure-tolerant):

        pose_tracker = PoseTracker()            # non-blocking, downloads async
        ...
        pose = pose_tracker.detect(frame, now_ms)   # every frame; may be None
        ...
        pose_tracker.close()
    """

    def __init__(self, every: int = POSE_EVERY, width: int = POSE_WIDTH,
                 auto_download: bool = True, threaded: bool = True):
        self.every = max(1, int(every))
        self.width = int(width)
        self.threaded = bool(threaded)
        self._landmarker = None
        self._frames = 0
        self._last_ms = None
        self._ts = 0                  # our own monotonic clock for VIDEO mode
        self._failed = False          # hard-disable after an unrecoverable error
        self._downloading = False
        self._raw = None              # last real measurement (unsmoothed)
        self._raw_age = 0.0           # seconds since that measurement
        self._s = None                # smoothed state dict
        self._ref_w = 0.0             # widest shoulders seen: the "facing me" ref
        self._ref_h = 0.0             # longest torso seen: the "upright" ref
        # worker plumbing: a ONE-SLOT mailbox, never a queue.  If the worker is
        # still busy when the next frame is due we simply drop that frame -
        # stale pose is worthless, and a backlog would make the suit follow the
        # body several seconds late.
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._job = None              # newest frame waiting to be measured
        self._out = None              # newest measurement waiting to be read
        self._stop = False
        self._worker = None
        if auto_download:
            self._ensure_model_async()

    # -- model plumbing ------------------------------------------------------ #
    def _ensure_model_async(self) -> None:
        """Fetch the model on a worker thread; the camera must not wait on it."""
        if POSE_MODEL_PATH.is_file() and POSE_MODEL_PATH.stat().st_size > 1_000_000:
            return
        if self._downloading:
            return
        self._downloading = True

        def work() -> None:
            tmp = POSE_MODEL_PATH.with_suffix(".tmp")
            try:
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                print("[pose] downloading pose_landmarker_lite.task ...")
                with urllib.request.urlopen(POSE_MODEL_URL, timeout=120) as resp, \
                        open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
                os.replace(tmp, POSE_MODEL_PATH)
                print(f"[pose] saved to {POSE_MODEL_PATH}")
            except Exception as exc:  # noqa: BLE001 - offline is a normal state
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                print(f"[pose] body tracking unavailable ({exc}); "
                      "the suit will fall back to hand anchoring.")
            finally:
                self._downloading = False

        threading.Thread(target=work, daemon=True).start()

    def _build(self) -> bool:
        """Create the landmarker once the file is on disk.  False = not yet."""
        if self._landmarker is not None:
            return True
        if self._failed or mp is None:
            return False
        if not (POSE_MODEL_PATH.is_file()
                and POSE_MODEL_PATH.stat().st_size > 1_000_000):
            return False
        try:
            try:
                tasks = mp.tasks
            except AttributeError:  # pragma: no cover - mediapipe 1.x layout
                from mediapipe import tasks  # noqa: PLC0415
            base = tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
            options = tasks.vision.PoseLandmarkerOptions(
                base_options=base,
                running_mode=tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                # the wearer is the only body that matters and they fill the
                # frame, so demand a confident detection: a half-seen bystander
                # in the background would yank the suit off the user's chest
                min_pose_detection_confidence=0.50,
                min_pose_presence_confidence=0.50,
                min_tracking_confidence=0.50,
                output_segmentation_masks=False,
            )
            self._landmarker = tasks.vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - never take the app down
            print(f"[pose] could not start body tracking: {exc}")
            self._failed = True
            return False
        return True

    def ready(self) -> bool:
        """True once a landmarker exists (i.e. pose can actually contribute)."""
        return self._landmarker is not None

    # -- worker -------------------------------------------------------------- #
    def _spawn(self) -> None:
        if self._worker is not None or self._failed or mp is None:
            return
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        """Worker loop: build the landmarker here, then chew on posted frames.

        The landmarker is created and used on this thread only - MediaPipe
        graphs dislike being driven from two threads, and keeping ownership in
        one place means the main loop can never block on model construction
        (which itself takes a noticeable moment).
        """
        while not self._stop:
            with self._wake:
                while self._job is None and not self._stop:
                    self._wake.wait(0.25)
                job, self._job = self._job, None
            if self._stop or job is None:
                continue
            if not self._build():     # model still downloading: drop the frame
                continue
            try:
                m = self._measure_small(*job)
            except Exception as exc:  # noqa: BLE001 - one bad frame is not fatal
                print(f"[pose] detection error: {exc}")
                m = None
            with self._lock:
                self._out = (m,)      # tuple marks "a fresh result exists"

    def close(self) -> None:
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None

    # -- measurement --------------------------------------------------------- #
    def _shrink(self, frame_bgr: np.ndarray):
        """Main-thread half of a measurement: the cheap part, and the copy.

        The resize doubles as the copy the worker needs - the main loop keeps
        drawing all over `frame`, so handing the worker the original array
        would have it measuring a body with a HUD painted on top.
        """
        h, w = frame_bgr.shape[:2]
        if w > self.width:
            k = self.width / float(w)
            small = cv2.resize(frame_bgr, (self.width, max(2, int(round(h * k)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr.copy()
        return small, w, h

    def _measure(self, frame_bgr: np.ndarray):
        """Synchronous convenience: shrink + measure on the calling thread."""
        if not self._build():
            return None
        return self._measure_small(*self._shrink(frame_bgr))

    def _measure_small(self, src: np.ndarray, w: int, h: int):
        """Worker half: one real detection -> torso numbers in FULL-FRAME px."""
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += 1
        res = self._landmarker.detect_for_video(image, self._ts)
        packs = getattr(res, "pose_landmarks", None)
        if not packs:
            return None
        lm = packs[0]
        if len(lm) <= R_HIP:
            return None
        core = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        if float(np.mean([_visible(lm[i]) for i in core])) < 0.55:
            return None                       # a torso we cannot see is a guess

        # landmarks are NORMALISED, so the downscale above is invisible here -
        # they map onto the full-size frame with no correction (same property
        # detect_frame() relies on for the hands)
        def px(i):
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float64)

        ls, rs = px(L_SHOULDER), px(R_SHOULDER)
        lh, rh = px(L_HIP), px(R_HIP)
        sh_mid = 0.5 * (ls + rs)
        hip_mid = 0.5 * (lh + rh)
        sh_vec = rs - ls
        sh_w = float(np.linalg.norm(sh_vec))
        torso = hip_mid - sh_mid
        torso_h = float(np.linalg.norm(torso))
        if sh_w < 20.0 or torso_h < 20.0:
            return None                       # too small to be a real torso

        # ROLL: the shoulder line's tilt on screen IS the body's roll.  Lean
        # left and the line tilts; the armour has to tilt with it or it looks
        # like a sticker pasted on the video.
        roll = math.atan2(sh_vec[1], sh_vec[0])
        if roll > math.pi / 2:                # keep it near zero regardless of
            roll -= math.pi                   # which shoulder the model calls
        elif roll < -math.pi / 2:             # "left" after the mirror flip
            roll += math.pi

        # YAW from FORESHORTENING: shoulders are a fixed-width bar in the real
        # world, so the only way they can get narrower on screen is if you
        # turned.  Track the widest span seen as the "square to camera"
        # reference and read the angle back out of the ratio.
        self._ref_w = max(sh_w, self._ref_w * 0.9995 + sh_w * 0.0005)
        self._ref_h = max(torso_h, self._ref_h * 0.9995 + torso_h * 0.0005)
        ratio = min(1.0, sh_w / max(1e-6, self._ref_w))
        yaw_mag = math.acos(max(0.0, min(1.0, ratio)))
        # direction: whichever shoulder is nearer the camera (smaller z) is the
        # one swinging forward.  z is unaffected by the main loop's mirror
        # flip, and comparing by SCREEN x keeps the sign right either way.
        near_left = (ls[0] < rs[0])
        dz = float(getattr(lm[L_SHOULDER], "z", 0.0) - getattr(lm[R_SHOULDER], "z", 0.0))
        sign = 1.0 if (dz < 0.0) == near_left else -1.0
        yaw = sign * min(yaw_mag, 1.05)       # ~60 deg; past that the fit is junk

        # PITCH is a bonus: a torso that shortens without the shoulders
        # narrowing means the body leaned toward or away from the camera.
        t_ratio = min(1.0, torso_h / max(1e-6, self._ref_h))
        pitch = -0.55 * math.acos(max(0.0, min(1.0, t_ratio)))

        return {"sh": sh_mid, "hip": hip_mid, "sh_w": sh_w, "torso_h": torso_h,
                "roll": roll, "yaw": yaw, "pitch": pitch,
                "elbows": [px(i) for i in (L_ELBOW, R_ELBOW)
                           if _visible(lm[i]) > 0.5]}

    # -- public API ---------------------------------------------------------- #
    def detect(self, frame_bgr, timestamp_ms: int):
        """Call EVERY frame.  Returns a smoothed torso frame dict, or None.

        Only every Nth call posts a frame to the worker, and no call ever waits
        for it; the rest just advance the smoothing toward the last measurement
        that came back, which is what keeps the suit moving continuously
        between detections.  Cost on the calling thread is one downscale on
        posting frames and a few floating-point lerps on the rest.  The
        returned dict is in
        FULL-FRAME pixel coordinates of the frame handed in:

            {"conf":     0..1 how much to trust this (fades out when lost),
             "shoulder": (x, y) shoulder midpoint,
             "hip":      (x, y) hip midpoint,
             "center":   (x, y) torso centre, midway between the two,
             "sh_w":     shoulder span in px,
             "torso_h":  shoulder-to-hip length in px,
             "roll":     radians, shoulder-line tilt,
             "yaw":      radians, turn away from camera (signed),
             "pitch":    radians, lean toward/away from camera}
        """
        if frame_bgr is None or mp is None or self._failed:
            return None
        # wall-clock delta drives the smoothing, so the suit follows at the same
        # rate whether the app is running at 15 fps or 60
        dt = 1.0 / 30.0
        if self._last_ms is not None:
            dt = max(0.0, min(0.25, (timestamp_ms - self._last_ms) / 1000.0))
        self._last_ms = timestamp_ms

        self._frames += 1
        due = (self._frames % self.every == 0) or self._frames == 1
        if self.threaded:
            self._spawn()
            if due:
                # post the newest frame; if one is already waiting it gets
                # overwritten, because the fresher frame is always the better
                # one to measure
                job = self._shrink(frame_bgr)
                with self._wake:
                    self._job = job
                    self._wake.notify()
            with self._lock:
                out, self._out = self._out, None
            m = out[0] if out is not None else None
        else:
            m = self._measure(frame_bgr) if due else None

        self._raw_age += dt           # a measurement only gets older with time
        if m is not None:
            self._raw = m
            self._raw_age = 0.0

        if self._raw is None:
            return None

        # confidence: full while the body is fresh, then a soft ramp down so a
        # brief occlusion does not make the suit jump back to the hand anchor
        over = self._raw_age - STALE_HOLD
        conf = 1.0 if over <= 0.0 else max(0.0, 1.0 - over / STALE_FADE)
        if conf <= 0.01:
            self._raw = None
            self._s = None
            return None

        r = self._raw
        if self._s is None:
            self._s = {"sh": r["sh"].copy(), "hip": r["hip"].copy(),
                       "sh_w": r["sh_w"], "torso_h": r["torso_h"],
                       "roll": r["roll"], "yaw": r["yaw"], "pitch": r["pitch"]}
        else:
            ap = 1.0 - math.exp(-dt / TAU_POS)
            asz = 1.0 - math.exp(-dt / TAU_SIZE)
            aa = 1.0 - math.exp(-dt / TAU_ANG)
            s = self._s
            s["sh"] += (r["sh"] - s["sh"]) * ap
            s["hip"] += (r["hip"] - s["hip"]) * ap
            s["sh_w"] += (r["sh_w"] - s["sh_w"]) * asz
            s["torso_h"] += (r["torso_h"] - s["torso_h"]) * asz
            for key in ("roll", "yaw", "pitch"):
                s[key] += (r[key] - s[key]) * aa

        s = self._s
        return {"conf": conf,
                "shoulder": s["sh"].copy(), "hip": s["hip"].copy(),
                "center": 0.5 * (s["sh"] + s["hip"]),
                "sh_w": s["sh_w"], "torso_h": s["torso_h"],
                "roll": s["roll"], "yaw": s["yaw"], "pitch": s["pitch"]}
