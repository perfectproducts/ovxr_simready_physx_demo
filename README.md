# VR SimReady Physics

Stand in VR (Meta Quest 3), **say "add a mug"**, and a SimReady asset from Hugging Face drops
in front of you. Then **grab it and throw it**.

| Piece | Used for |
|---|---|
| [ovxr](https://github.com/NVIDIA-Omniverse/ovxr) + ovrtx + ovstage | RTX rendering to the headset over OpenXR, plus the shared scene |
| [Newton](https://github.com/newton-physics/newton) (XPBD, GPU) | Rigid-body physics: colliders and mass come from each asset's USD physics |
| SimReady repos on Hugging Face | The asset catalog, downloaded on demand |
| [Wit.ai](https://wit.ai) | Speech-to-intent. This is the service Meta's Voice SDK wraps (see "Why not the Unity Voice SDK") |

## Why not the Unity Voice SDK?

ovxr and Newton are a C/Python stack that runs on the PC and streams to the Quest through the
OpenXR runtime (Quest Link or Air Link). Meta's Voice SDK only ships for Unity and Unreal, and
while Link is active the Quest can't run its own app alongside. So this app calls **Wit.ai**
directly, the same backend and intent model the Voice SDK uses. If you train intents in your
Wit app, they carry over unchanged to a Unity Voice SDK client.

## Setup (Windows)

1. **Quest Link.** Install the Meta Quest Link app and set it as the active OpenXR runtime
   (Settings → General → OpenXR Runtime). Connect the Quest 3 with a cable or Air Link and enter
   Link mode.
2. **Python environment** (uv, Python 3.11–3.13):
   ```powershell
   uv sync
   ```
   This pulls ovxr/ovrtx/ovstage from `pypi.nvidia.com`, plus Newton, Warp, usd-core and pyopenxr.
3. **Hugging Face.** The catalog crawls the public repos listed in `SIMREADY_PARTNER_REPOS`. They
   default to NVIDIA's own `nvidia/simready-assets` (about 2,700 packages) plus
   `ALLSIDES/simready-scanned-objects` and `Vertiv/simready-assets`, about 2,800 assets in all.
   A token is optional but gives higher rate limits. Put it in `huggingface_api.txt` in the
   project root (gitignored), or set `$env:HF_TOKEN`. To change the repos, set them as a
   comma-separated list, then run `uv run python -m vrsim.catalog refresh`.
   Searches go first to [SimReady Central](https://nvidia-simready-central.hf.space/), NVIDIA's
   USD Search service for SimReady assets on Hugging Face. It needs no token, understands
   descriptions ("something to drink from"), and covers about 20 OEM datasets beyond the crawl.
   If it is unreachable, the app uses the local index. Set `SIMREADY_CENTRAL_URL=none` to turn it
   off.
4. **Wit.ai.** Create an app at https://wit.ai, open *Management → Settings*, and put the
   tokens in `wit_ai_api.txt` in the project root. The file is gitignored, and the app uses
   `CLIENT_ACCESS_TOKEN`:
   ```
   APP_ID=...
   SERVER_ACCESS_TOKEN=...
   CLIENT_ACCESS_TOKEN=...
   ```
   Alternatively, set `$env:WIT_AI_TOKEN`, which takes precedence.
   An untrained app is enough, because the transcription is parsed locally ("add / spawn / give me
   … <thing>"). For better accuracy, train these intents:
   - `spawn_object` with entities `object` (a free-text/keyword role) and `wit/number`
   - `clear_objects`, e.g. "clear everything"
   - `undo`, e.g. "remove the last one"
5. **Microphone.** Over Link the headset mic shows up as a Windows input device. If it isn't the
   default, list devices and pick one:
   ```powershell
   uv run python -m vrsim --list-mics
   uv run python -m vrsim --mic "Headset Microphone"
   ```

## Run

The C++ app (ovphysx) and its Python sidecar start with one batch file:

```powershell
.\start_vr.bat              # Minimal rendering, 72 fps
.\start_vr.bat rt2          # full RT2 rendering with the real HDRI sky, about 37 fps
.\start_vr_rt2.bat          # the same, for double-clicking
```

It opens the sidecar (voice, catalog, downloads) in its own window, waits for it to listen on
port 47800, then starts `cpp\build\vrsim_cpp.exe`. Extra arguments go to the exe, and
`VRSIM_MIC` selects another microphone. Build the exe first with `cpp\scripts\build.bat`.

The all-Python version (Newton physics) runs with:

```powershell
uv run python -m vrsim
```

The first launch compiles RTX shaders and Warp kernels, which takes a few minutes. Later launches
start in seconds. You can also type commands into the console (`add a mug`, `spawn 3 footballs`,
`undo`, `clear all`), which works even without a headset.

### Controls (Quest Touch)

| Input | Action |
|---|---|
| **A / X** (hold) | Push-to-talk: "add a mug", "spawn three footballs", "clear all", "undo", or ask: "what balls are available?", "do you have any boxes?", "what can I add?" (results appear on the panel) |
| **Grip** | Grab the nearest object. If nothing is within reach, force-pull the object you're pointing at |
| Release grip | Throw. The object leaves with your hand's velocity × `--throw-scale` (default 1.3) |
| **Trigger** (half) | Show the pointer ray |
| New asset (C++ app) | A red marker follows the pointer on the surface you point at (floor, desk); its arrow is the asset's facing. The thumbstick turns it, the **trigger** spawns the asset there, fixed. **B** (or "undo") cancels |
| **Trigger** while holding | Fix the prop where it is, or free a fixed one. Voice: "fix it" / "free it". A fixed prop's main body stays put while its jointed parts (lid, pedal, glasses arms) still move with either hand |
| **Thumbstick** while holding a fixed prop | Turn it about its centre's vertical axis, so it keeps standing on the floor |
| **B** | Undo the last spawn |
| **Y** | Clear all props |
| "cancel" / × on the panel | Stop a running download (C++ app) |

Haptics: a short tick when grabbing or spawning, and a long buzz when no asset matched what you said.

### Voice feedback panel

A text panel floats about 1 m in front of you, slightly below eye level, and drifts after your
head as you turn. A colored bar on its left shows the state:

| Bar | When | Example |
|---|---|---|
| Blue | Listening, recognizing, searching or downloading | `Listening…` → `“add three mugs”` / `Searching SimReady for 'mug'…` |
| Green | Done | `Added 3 × mug` / `3 objects in the scene · grip to grab, release to throw` |
| Red | Problem | `No SimReady asset for 'rubber duck'` / `Closest: dumbbell, beer stein, …` |

Results stay up for 5–7 s, then the panel hides. When the headset connects, it shows a 12 s hint
with example commands.

The panel is ordinary scene geometry, not an OpenXR compositor layer. `vrsim/hud.py` rasterizes
text with Pillow into merged pixel rectangles and rewrites only the `points` of a fixed-topology
mesh (6000 quads) at the frame's stage ordinal. Nothing is written while the message stays the same.

## How it works

```
frame:  ovxr.begin_frame ─► OpenXR actions synced at the predicted display time
        ─► hands: grab / pull / throw ─► Newton step (velocity-driven grab targets)
        ─► update_prim_transforms(body prims + hand visuals) at ordinal N
        ─► bind_frame(N) seals it ─► submit_frame
between frames: spawn/remove ─► ovstage add_usd_reference / remove_usd ─► advance_write_floor
worker thread:  voice text ─► fuzzy catalog match ─► snapshot_download ─► wrapper USD + Newton sub-builder
```

- **One USD feeds both systems.** Each asset gets a generated wrapper layer that normalizes its
  up-axis and metersPerUnit to the app's Z-up, metre-based stage. The renderer references the
  wrapper and Newton's `add_usd` parses the same file, so the rendered mesh and the simulated
  colliders can't drift apart.
- **Spawning rebuilds the Newton model.** Newton models are immutable once finalized, so the app
  rebuilds and carries every body's pose and velocity across. After warm-up this takes about
  100 ms, so the headset drops a few frames on each spawn.
- **Grabs drive velocity, not position.** Grabbed bodies get the velocity that reaches the hand
  target within one frame, capped at 12 m/s. Contacts stay physical, force-pulled objects fly to
  your hand, and releasing hands the object your hand's velocity.
- **ovxr runs in `StageOrdinalMode.APPLICATION`** because the app co-writes the stage.
  `scene.OrdinalClock` holds one ordinal until it is sealed, either by `bind_frame` or by
  `advance_write_floor`, so ovstage never sees a gap.

## Layout

```
vrsim/
  app.py        frame loop, hand interaction, spawn orchestration
  scene.py      ovrtx renderer + ovstage, ordinal clock, generated environment USD
  physics.py    Newton world, rebuild-on-spawn, grab drive kernel, picking
  assets.py     SimReady package -> wrapper USD + Newton sub-builder
  catalog.py    HF index (official catalog + partner crawl), fuzzy search, download
  xr_input.py   OpenXR actions on ovxr's borrowed session (pyopenxr ctypes)
  voice.py      push-to-talk capture, Wit.ai /speech, command parser, console input
tests/smoke_headless.py   catalog -> physics -> grab -> throw, no headset needed
```

## Verification status

- ✅ Headless: `uv run python tests/smoke_headless.py mug football`. Assets settle on the table,
  a grabbed body tracks the hand to within 5 mm, and a throw lands about 3 m away.
- ✅ The app starts the renderer, loads the environment, creates the ovxr context on the Oculus
  OpenXR runtime, and handles spawn, undo and clear from console commands while waiting for a
  headset.
- ✅ HUD text: `uv run python tests/render_hud.py` renders two messages offscreen through
  ovstage `points` writes into ovrtx and checks that the text appears and changes.
- ✅ Wit.ai with the real token: synthesized English speech ("Add three mugs", "Clear all
  objects", "Spawn a football", "Undo") is transcribed and parsed correctly with an untrained app.
- ⏳ **Not yet run with a connected Quest:** the stereo render, controller actions
  (`xrAttachSessionActionSets` on ovxr's session), and the panel's head-follow placement in the
  headset (that math is unit-checked). If the action set fails to attach, the log names the call.

## Known limitations

- Only assets with `PhysicsRigidBodyAPI` can be spawned. Static assets such as server racks are
  rejected with a message.
- Picking uses bounding spheres, so large or elongated objects are grabbable a little outside
  their surface.
- The voice panel is world geometry, so it is lit by the scene and can be hidden behind objects
  between you and it. An OpenXR quad layer (`submit_frame(additional_layers=...)`) would always
  draw on top and look sharper, but needs a Vulkan swapchain and uploads.
- ovrtx applies no auto-exposure, so light intensities are low (`scene.SKY_INTENSITY`,
  `SUN_INTENSITY`). The Kit-typical 1000+ washes everything out to flat grey.
- With Link active, pass `--mic` with the headset microphone. Otherwise the Windows default
  input is used (on this machine, a HyperX Quadcast).
