# 🕷️ Hologram Studio — Tony Stark tech, driven by your hands

Point your webcam at your hands and a futuristic, Tony-Stark-style **hologram
suite** comes to life — all drawn as see-through "hard-light" projections.
No mouse, no keyboard, no input injection. Every feature is triggered by a
hand gesture.

## The gestures (final list)

| Gesture | What happens |
|---|---|
| **Show your right hand** | A holographic **web-shooter** materialises on your wrist — three concentric rings, arc-reactor core, web-fluid cartridge with a draining gauge, floating HUD panels, orbiting particles, scanlines and a soft flicker. It follows your hand in real time. |
| **Right-hand PINCH** (thumb + index + middle together) | **Take the web-shooter OFF your wrist.** Snap-threads stretch and break, a shockwave flare bursts, and the shooter floats in your pinch, following your hand. Let go and it **stays floating in mid-air**, gently bobbing. |
| **PINCH again while it's floating** | The shooter **flies back onto your wrist** with a glowing streak trail. |
| **FIST, hold ~0.5 s, then OPEN** | A holographic **gadget blueprint** appears — an animated exploded-view breakdown of one of Spider-Man's gadgets (web-shooter Mark V, web-fluid cartridge, or spider drone). Parts assemble, explode, and reassemble in a loop; dimension lines and part badges label the pieces. It fades out after a few seconds. |
| **BOTH FISTS, hold ~0.6 s, then OPEN** | Toggles holographic **body gear**: Iron-Man-style gauntlets on both wrists (segmented armour plates, palm arc reactor, knuckle nodes, energy veins, orbiting motes) plus a chest arc reactor between the hands whenever both are visible. |
| `V` | Start / stop recording the preview to `recordings/`. |
| `R` | Toggle body gear from the keyboard (handy for testing). |
| `Esc` / `Q` | Quit. |

Open `preview_*.png` in this folder (generated on a sample photo) to see the
floating shooter, all three blueprints, the body gear, and a full combined
scene.

## Recognition is pushed to the limit

- **Minimum confidence thresholds** — `min_hand_detection_confidence 0.10`,
  `min_hand_presence_confidence 0.10`, `min_tracking_confidence 0.20`. Hands
  are recognised in weak light, at frame edges, and at distance.
- **720p camera** — the app forces 1280×720 (MJPEG + low-latency buffering)
  and falls back down a resolution ladder if the driver refuses. It prints
  the actual resolution on startup.
- **Gesture robustness** — finger curls, pinch and fist state are
  **EMA-smoothed per hand** (hands are matched frame-to-frame by palm
  position), with hysteresis and frame-confirmation so jitter never trips a
  gesture. If a hand drops out for a few frames, its state is **ghosted** so a
  tracking hiccup can't cancel a held gesture.
- **Both hands tracked** — right/left is decided by MediaPipe's handedness
  (mirrored/selfie input), with a mirrored-frame position fallback.
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
  (`min_hand_detection_confidence 0.10`, `min_hand_presence_confidence 0.10`,
  `min_tracking_confidence 0.20`), 2 hands.
- **Gestures** — `GestureTracker` turns smoothed per-hand features into
  discrete events: `spawn` (fist → open), `grab`/`release` (right-hand
  pinch), `gear` (both fists → open).
- **Rendering** — `WebShooterHologram` (on-wrist / detaching / held /
  floating / reattaching states), `GadgetBlueprint` (exploded-view
  breakdowns), and `BodyGear` (gauntlets + chest reactor). All pure 2D
  vector drawing blended additively for the see-through hologram look.

## Troubleshooting

- **"No webcam was found"** — try `python hand_zoom.py --camera 1` (or 2/3).
- **Hands not recognised** — more light, move closer, keep the hand fully in
  frame; `--verbose` shows the tracked hand count every second.
- **Slow** — the resolution ladder should find your camera's best mode; if it
  still chugs, close other camera apps.
- **First run downloads a model** (~7 MB) — needs internet the first time.
