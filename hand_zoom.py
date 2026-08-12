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

            # -- render ---------------------------------------------------------- #
            dt = max(0.0, t0 - last_frame_t)
            last_frame_t = t0
            if right_feat is not None:
                draw_hand_holo(frame, right_feat["landmarks"])

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
