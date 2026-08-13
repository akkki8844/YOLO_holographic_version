# 🕷️ Hologram Studio — Tony Stark tech, driven by your hands

Point your webcam at your hands and a futuristic, Tony-Stark-style **hologram
suite** comes to life — every hologram rendered in electric-blue hard-light
(you'll also see the blue **skeleton tracing** on every tracked hand, so you
know exactly when you're recognised). No mouse, no keyboard, no input
injection. Every feature is triggered by a hand gesture.

## The gestures

| Gesture | What happens |
|---|---|
| **WEB-SHOOT POSE + PULL** (thumb pressed on index+middle, ring+pinky down) | The pinch grabs hard-light out of the air and a **3D web-shooter flies onto the OPPOSITE wrist** — with your **right** hand it lands on your **left** wrist, and vice versa. It arrives as a wireframe, solidifies on landing, then tracks that wrist in full shaded 3D. Pinch-pull again to send it away. (A tight thumb+index grab also works; a lazy thumb brush no longer summons one by accident.) |
| **PINCH A WORN SHOOTER and PULL** | **Take it off.** Reach over, pinch the shooter sitting on your wrist and pull — the way you'd strip a real one off. It detaches immediately (no move-lock countdown), becomes its own object following your hand, and stays wherever you drop it. |
| **ONE FIST, hold ~0.55 s, then open** | If that wrist isn't wearing its shooter, this **puts it back on** — like pulling a glove on. Once you ARE wearing it, the same gesture raises a **detailed exploded blueprint** in black-and-white drafting style: real 3D geometry holding a fixed three-quarter view (it does not spin — a spinning schematic can't be read), cycling assemble → hold → explode against a ghosted outline, with leader-line callouts for all nine parts, a dimension line and a live spec block. The same gesture dismisses it. |
| **BOTH FISTS, hold ~0.7 s, then open** | Toggle the full **holographic body gear**: the torso shell with the spider emblem and webbing, shoulder plates, belt, an armoured gauntlet with its own arc reactor on every visible wrist, and the **chest reactor** burning over the sternum. The `R` key does the same. |
| **OPEN PALM, hold it** | A **Stark palm-projection** materialises above the hand — a holographic screen with a rotating mini web-shooter playing on it — and tracks the palm in 3D. It fades out the moment the hand closes. |
| **WEB-SHOOT POSE held** (wearing a shooter) | A dotted **targeting web-line** snaps out of the spinneret along the aim axis, ending in a reticle. |
| **GRAB an object and HOLD 4 s** | A lock ring charges around whatever you grabbed (a shooter, the blueprint) with a ticking countdown. When it snaps **READY**, the object follows your hand. Release and it simply **stays where you left it** — nothing else happens. |
| `G` | Toggle the on-screen gesture guide panel. |
| `D` | Live gesture readout — the raw finger-curl / pinch numbers, so a stubborn gesture can be seen, not guessed at. |
| `V` | Start / stop recording the preview to `recordings/`. |
| `R` | Toggle body gear (keyboard). |
| `Esc` / `Q` | Quit. |

Open `preview_*.png` in this folder (generated on a sample photo) to see the
floating shooter, the blueprints, the body gear, and a full combined scene.

## Recognition is pushed to the limit

- **Confidence tuned for stability, not just sensitivity** — the detector
  runs at `0.20 / 0.20 / 0.40` (detection / presence / tracking). The
  absolute minimums (0.10) make the detector re-fire on noise, so hands pop
  in and out and every gesture looks broken; at 0.20 the VIDEO-mode tracker
  re-acquires a lost hand within a frame or two while 0.40 keeps an
  established track from being dropped by single-frame flickers.
- **Fingers are judged by joint angles in 3D** — each finger's curl is
  measured at the PIP and DIP joints, which is invariant to hand size,
  camera distance and perspective foreshortening. The old mcp→tip distance
  ratio depended on all three, which is why a real fist read as open in one
  pose and a resting hand read as a fist in another.
- **Pinches are 3D** — thumb-to-finger distance uses the landmark depth
  channel, so a real pinch reads far tighter than a flat 2D distance, with
  forgiving hysteresis (0.42 on / 0.58 off).
- **Handedness comes from MediaPipe's own label** — the feed is mirrored, so
  "Right" really is your right hand. Position (right of screen = right hand)
  is only the fallback when the label is unsure. The old position-only
  scheme silently flipped left/right the moment your hands **crossed** — which
  is exactly what the pinch-pull summon does.
- **Gesture robustness** — finger curls, pinch, fist and the web-shoot pose
  are **EMA-smoothed per hand** (hands are matched frame-to-frame by palm
  position), with hysteresis and frame-confirmation so jitter never trips a
  gesture. If a hand drops out for a few frames, its state is **ghosted** so a
  tracking hiccup can't cancel a held gesture.
- **720p camera** — the app forces 1280×720 (MJPEG + low-latency buffering)
  and falls back down a resolution ladder if the driver refuses. It prints
  the actual resolution on startup.
- **Self-explanatory logs** — `--verbose` shows how many hands are tracked
  every second; the gesture guide is printed to the console on startup.

## Performance

All holograms render into a **half-resolution overlay** that is upscaled once
per frame — the upscale doubles as the glow bloom. Measured at 1280×720:
web-shooter ~6 ms, gauntlets ~6 ms, blueprint ~4 ms per frame. Detection is
the only bottleneck, so the app runs at full webcam speed even with every
system active at once.

## Use it on video calls (WhatsApp, Teams, Zoom, Meet) 🎥

The hologram can be broadcast as a **virtual camera**, so anyone on a call
sees it live on your hands — while the script is running. One-time setup:

1. **Double-click `setup_virtualcam.bat`** — it installs `pyvirtualcam` into
   the virtual environment, downloads the **Unity Capture** driver, and
   registers it (click **Yes** on the one-time administrator prompt).
   *(Alternative: install [OBS Studio](https://obsproject.com/) and start its
   built-in Virtual Camera instead — no driver prompt, but OBS must be open.)*
2. **Run the app with the broadcast flag:**

   ```bash
   run.bat --virtualcam
   # or: python hand_zoom.py --virtualcam
   ```

3. **In WhatsApp** → Settings → Video (or the camera picker in any call) →
   select **“Unity Video Capture”** (or “OBS Virtual Camera”).

The virtual camera only produces video while the script runs — close it and
the device goes dark. Frames are sent unmirrored so the other person sees a
normal camera view. `--vc-backend obs` forces the OBS backend if both are
installed.

## Running

```bash
python hand_zoom.py             # show your hands to the camera
python hand_zoom.py --camera 1  # force a specific webcam
python hand_zoom.py --virtualcam  # broadcast to video calls (see above)
python hand_zoom.py --record    # save the preview to recordings/
python hand_zoom.py --selftest  # headless self-test (no camera needed)
```

Double-click `run.bat` on Windows — it sets up a virtual environment,
installs dependencies, downloads the hand-tracking model on first run, and
launches the app.

## How it works

- **Tracking** — MediaPipe Tasks Hand Landmarker in VIDEO mode
  (`min_hand_detection_confidence 0.20`, `min_hand_presence_confidence 0.20`,
  `min_tracking_confidence 0.40`), 2 hands, handedness from the model itself.
- **Gestures** — `GestureTracker` turns EMA-smoothed, angle-based per-hand
  features into discrete events: `pinch`/`pull`/`unpinch` (web-shoot pose or
  thumb+index), `fist` (single fist → open, on release), `gear` (both fists
  → open).
- **Rendering** — `WebShooter` (on-wrist / detaching / held / floating /
  reattaching states), `Blueprint` (exploded-view breakdown), `SuitGear`
  (torso shell + emblem + webbing + gauntlets + chest reactor) and
  `PalmProjector` (open-palm hologram). All drawn as translucent 3D quad
  meshes into a shared half-resolution overlay whose upscale doubles as the
  bloom, with crisp full-resolution passes for the small objects.

## Troubleshooting

- **"No webcam was found"** — try `python hand_zoom.py --camera 1` (or 2/3).
- **Hands not recognised** — more light, move closer, keep the hand fully in
  frame; `--verbose` shows the tracked hand count every second.
- **Slow** — the resolution ladder should find your camera's best mode; if it
  still chugs, close other camera apps.
- **First run downloads a model** (~7 MB) — needs internet the first time.
