# 🕷️ Hologram Studio — Tony Stark tech, driven by your hands

Point your webcam at your hands and a futuristic, Tony-Stark-style **hologram
suite** comes to life — every hologram rendered in electric-blue hard-light
(you'll also see the blue **skeleton tracing** on every tracked hand, so you
know exactly when you're recognised). No mouse, no keyboard, no input
injection. Every feature is triggered by a hand gesture.

## The gestures

Every gesture is gated so it can never misfire into another one — see
`GestureTracker` / `HoloDesk` in `hand_zoom.py` for the exact collision
guards (pinch-quiet windows, fist-fired flags, hysteresis).

| Gesture | What happens |
|---|---|
| **(automatic) show a wrist** | A **compact web-shooter** — a wrist-mounted launcher that only climbs a short way up the forearm, the way a real one straps on — equips onto it the instant it's seen, no gesture needed. Hard cap of **two** on screen at once (one per wrist). |
| **WEB-SHOOT SIGN + PULL** (middle + ring pressed into the palm — the trigger — index and pinky out) | The real thwip sign, so making it for real is what fires it. Sends that wrist's shooter away; do it again to bring it back. (A tight thumb+index grab also works; a lazy thumb brush no longer triggers it by accident.) |
| **PINCH A WORN SHOOTER and PULL** | **Take it off by hand.** Reach over, pinch the shooter sitting on your wrist and pull — the way you'd strip a real one off. It plays a short **peel-away animation** — popping off the forearm, then swinging into your grip, exactly like the real motion — instead of a jump-cut swap. No move-lock countdown once it settles: it's its own object, following your hand and staying wherever you drop it. |
| **ONE FIST, hold ~0.45 s, then open** | If that wrist's shooter is off, this **wears it back on** — closing and opening the hand like pulling a glove on. It fires on **release**, so a fist held for any other purpose can never trigger it early. |
| **BOTH FISTS, hold ~0.55 s, then open** | Toggle the full **holographic body gear**: the torso shell with the spider emblem and webbing, shoulder plates, belt, an armoured gauntlet with its own arc reactor on every visible wrist, and the **chest reactor** burning over the sternum. The `R` key does the same. |
| **OPEN PALM, hold it** | A **Stark palm-projection** materialises above the hand — a holographic screen with a rotating mini web-shooter playing on it — and tracks the palm in 3D. It fades out the moment the hand closes. |
| **WEB-SHOOT POSE held** (shooter worn) | A dotted **targeting web-line** snaps out of the spinneret to a reticle. Hold the pose past ~0.9 s and it fans out into a **three-point multi-lock**, the way the suit's multi-web targeting reads a sustained aim as more than one shot. |
| **GRAB an object and HOLD 1.5 s** | A lock ring charges around whatever you grabbed (a shooter) with a ticking countdown. When it snaps **READY**, the object follows your hand. Release and it simply **stays where you left it** — nothing else happens. |
| **(automatic) move a hand sharply** | **Spider-sense.** A sudden jolt of the hand fires a broken radial warning ripple out of the palm — three offset arcs snapping outward, because the sense is a jolt rather than a readout. It pings once and settles instead of strobing while you keep moving. |
| `G` | Toggle the on-screen gesture guide panel. |
| `D` | Live gesture readout — the raw finger-curl / pinch numbers, so a stubborn gesture can be seen, not guessed at. |
| `V` | Start / stop recording the preview to `recordings/`. |
| `R` | Toggle body gear (keyboard). |
| `Esc` / `Q` | Quit. |

Open `preview_*.png` in this folder (generated on a sample photo) to see the
floating shooter, the body gear, and a full combined scene.

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
- **Adaptive smoothing** — the per-hand EMA changes strength with how fast the
  hand is moving. A hand holding still gets heavy averaging, so landmark jitter
  can never trip a gesture on its own; a hand that is *moving* gets almost
  none, so the gesture lands the frame you make it instead of trailing behind.
  A single fixed smoothing constant has to trade one of those away for the
  other — this keeps both.
- **Hands survive crossing over** — frame-to-frame slot matching adds a penalty
  for swapping handedness, so proximity alone can't hand your held fist timer
  to the other hand when your wrists cross. That matters because reaching
  across to take a shooter off is *exactly* a hand-crossing move; without it
  the held gesture silently died mid-reach.
- **Gesture robustness** — hysteresis and frame-confirmation on every edge, so
  jitter never trips a gesture. If a hand drops out for a few frames, its state
  is **ghosted** so a tracking hiccup can't cancel a held gesture.
- **720p camera** — the app forces 1280×720 (MJPEG + low-latency buffering)
  and falls back down a resolution ladder if the driver refuses. It prints
  the actual resolution on startup.
- **Self-explanatory logs** — `--verbose` shows how many hands are tracked
  every second; the gesture guide is printed to the console on startup.

## The suit fits your actual body

The body gear is anchored to a **real MediaPipe pose**, not guessed from your
hands. It sits on your true shoulder line, scales to your real shoulder width
and torso length, and turns and leans with you: yaw comes from the shoulder
span foreshortening as you rotate, roll straight from the shoulder-line tilt.
The chest reactor rides the sternum through the same rotation.

It is deliberately cheap and never load-bearing:

- Pose runs on a **background thread every 3rd frame** (~10 Hz), costing the
  main loop ~1.4 ms/frame against ~24 ms for a raw detection. Inline, even
  1-in-3 would hitch the loop by a full frame every third frame.
- Between detections the torso keeps easing toward the last measurement, so
  the output is continuous rather than a 10 Hz staircase.
- **If pose is unavailable — offline, still downloading, no body in frame, or
  hips hidden behind a desk — the suit silently falls back** to the old
  hand-derived anchor. A fresh install starts on the fallback and upgrades a
  second or two later once the 5.8 MB model lands. Nothing blocks on it.

## It moves like the real thing

The gear is simulated, not keyframed — the goal is that doing something for
real produces the motion you'd actually get.

- **The shooter is strapped on, not welded on.** A worn shooter chases the
  wrist through a critically-damped spring, so whipping your arm makes the
  housing trail by a few pixels and settle. Snapping it exactly onto the
  landmark every frame is what made it read as a decal painted on the video:
  nothing real tracks your arm with zero lag.
  - It uses the *closed-form* solution of the spring, not a stepped one. A
    strap stiff enough to feel like a strap has `C·dt > 1` at 30 fps, and
    explicitly integrating that flips the velocity's sign every frame — the
    model shakes itself apart instead of settling. The analytic form is
    unconditionally stable, never overshoots, and **feels identical at 30 and
    60 fps** (verified in the selftest).
- **Firing recoils it.** The kick is an impulse injected straight into that
  same mount spring rather than a canned bounce animation, so the recoil and
  the recovery are the physics that already holds it on your arm.
- **A held shooter swings.** Carry it left and it trails right, then settles —
  driven by the pinch's real lateral speed, so the swing is the motion you
  actually made rather than a loop.
- **Taking it off peels it off.** Pinch a worn shooter and pull and it pops off
  along the forearm, then swings into your grip — the real motion, not a
  jump-cut swap.

## The hard-light look

Every hologram is real shaded 3D geometry, not a sprite. The renderer
(`holo3d.py`) is a quad-mesh painter's-algorithm engine with four cues doing
the heavy lifting:

- **Fresnel on the fill, not just the outline** — real volumetric light is
  brightest where you look *along* its surface rather than into it, so faces
  turning toward the silhouette carry more glow. Without this the interior
  reads flat and only the outline says "3D"; with it the whole body has a
  shell to it.
- **White-hot speculars** — a highlight that keeps the body hue reads as a
  glowing decal, so glints blow out toward white instead, and the same
  geometry reads as a hard surface catching a light.
- **Depth-graded edges** — near faces carry a heavier outline than far ones.
  It's the cheapest depth cue there is, and it's suppressed below ~24 px scale
  where a 2 px outline would swallow the panel lines it's meant to separate.
- **Rolling scan-lines** — the raster creeps down each model instead of
  sitting on it like a printed texture, so it reads as a live projection.

## Performance

All holograms render into a **half-resolution overlay** that is upscaled once
per frame — the upscale doubles as the glow bloom, and the small objects get a
crisp full-resolution pass into their own bounding box. Measured at 1280×720:
web-shooter ~6.8 ms, body gear ~6.2 ms, gauntlets ~3.4 ms per frame. Detection
is the only bottleneck, so the app runs at full webcam speed even with every
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
  Body pose comes from `holo_pose.PoseTracker` (Pose Landmarker lite, one
  person, threaded, every 3rd frame) and only ever feeds the suit's fit.
- **Gestures** — `GestureTracker` turns EMA-smoothed, angle-based per-hand
  features into discrete events: `pinch`/`pull`/`unpinch` (web-shoot pose or
  thumb+index), `fist` (single fist → open, on release), `gear` (both fists
  → open).
- **Rendering** — `WebShooter` (on-wrist / peeling off / held / floating /
  reattaching states, a scan-line sweep as it materialises, and a web-fluid
  cartridge readout while worn), `SuitGear` (torso shell + emblem + webbing +
  gauntlets + chest reactor) and `PalmProjector` (open-palm hologram). All
  drawn as translucent 3D quad meshes into a shared half-resolution overlay
  whose upscale doubles as the bloom, with crisp full-resolution passes for
  the small objects.

## Troubleshooting

- **"No webcam was found"** — try `python hand_zoom.py --camera 1` (or 2/3).
- **Hands not recognised** — more light, move closer, keep the hand fully in
  frame; `--verbose` shows the tracked hand count every second.
- **Slow** — the resolution ladder should find your camera's best mode; if it
  still chugs, close other camera apps.
- **First run downloads a model** (~7 MB) — needs internet the first time.
