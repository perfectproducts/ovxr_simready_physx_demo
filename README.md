# VR SimReady Physics

Stand in VR (Meta Quest 3), **say "add a mug"**, and a SimReady asset from Hugging Face drops
in front of you. Then **grab it and throw it**.

| Piece | Used for |
|---|---|
| [ovxr](https://github.com/NVIDIA-Omniverse/ovxr) + ovrtx + ovstage | RTX rendering to the headset over OpenXR, plus the shared scene |
| [Newton](https://github.com/newton-physics/newton) (XPBD, GPU) + [ovnewton](https://github.com/NVIDIA-Omniverse/ovnewton) | Rigid-body physics: colliders and mass come from each asset's USD physics; ovnewton builds the model straight from the ovstage |
| [ovphysx](https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx) (PhysX) | The C++ app's physics, simulating the same ovstage |
| SimReady repos on Hugging Face | The asset catalog, downloaded on demand |
| [Wit.ai](https://wit.ai) | Speech-to-intent. This is the service Meta's Voice SDK wraps (see "Why not the Unity Voice SDK") |

Three apps share the content (catalog, voice, browser, HUD, the smart toaster and its FMUs):

| App | Physics | Where | Status |
|---|---|---|---|
| **C++ app** | ovphysx (PhysX) | `cpp/` + Python sidecar `vrsim/sidecar.py` | headset-tested, 72 fps Minimal |
| **Newton edition** | Newton through ovnewton, one ovstage for everything | [`newton_app/`](newton_app/README.md) | headset-tested, 30–36 fps |
| all-Python app (the first one) | Newton, parsing USD itself | `vrsim/app.py` | superseded |

These libraries are pre-release, so the versions in `pyproject.toml` (and `newton_app/pyproject.toml`)
are pinned on purpose.

## Documentation

- [How the parts interact](docs/architecture.html): technical description of the running system.
  Download it and open it in a browser.
- [Article draft](docs/medium-article.md): the libraries, SimReady, and the pitfalls along the way.
- [Native app](cpp/README.md): build steps and the per-frame loop.
- [Newton edition](newton_app/README.md): ovxr + ovrtx + ovstage + ovnewton in one Python process,
  what ovnewton needed changed, and its [validation report](newton_app/VALIDATION.md).

## Agent skills

Two Claude Code skills load automatically when you work in this repo:

- [`build-ovxr-simready-demo`](.claude/skills/build-ovxr-simready-demo/SKILL.md) sets up this demo
  or rebuilds it from scratch, milestone by milestone.
- [`ovxr-vr-physics`](.claude/skills/ovxr-vr-physics/SKILL.md) holds the debugging rules: why prims
  don't move, frame-rate limits, grabbing, voice and panels.

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

The **Newton edition** has its own environment and launcher (see [newton_app/README.md](newton_app/README.md)):

```powershell
cd newton_app
uv sync
.\start_newton_vr.bat       # Minimal rendering; ".\start_newton_vr.bat rt2" for RT2
```

Its sidecar listens on port 47801, so it can run next to the C++ app's. Start Quest Link before the
app: an app started first does not see a headset that appears later.

The first all-Python version (Newton physics) runs with:

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
| "realign" / "recenter the room" | Move the scene so your head is at the seat, facing the desk |
| **A + X** together (Newton edition) | Recenter: the desk in front of where you look |
| **Thumbstick click** | Open or close the SimReady catalog browser; point and pull the trigger on a tile to add it, flick the stick to page |

Haptics: a short tick when grabbing or spawning, and a long buzz when no asset matched what you said.

### Smart toaster

A toaster stands at the front left of the Studierstube desk, fixed in place, with a toast in slot 1.
It is the dbeyond demonstrator's smart toaster (`assets/dbeyond`, see its README): the NVIDIA SimReady
toaster plus three FMUs, an OpenModelica ECU, a thermal and browning model, and the toast's flight.
They are wired to the toaster through the ovfmi USD-FMI binding in its USD layers.

| Input | Action |
|---|---|
| **Grip** on the lever, push down | Start toasting. The ECU latches the lever and the heater glows in the slots |
| **Grip** on the dial, twist | Browning level 1 to 7 (45° per level, 60 to 200 s) |
| **Trigger** with the controller at the CANCEL button | Stop the cycle. The lever pops up |
| "new toast" / "reset the toaster" | A fresh, untoasted slice and a cold toaster |
| **Menu** (left controller) | Show or hide the FMU monitor beside the toaster |

The FMU monitor stands on the desk next to the toaster (the Newton edition: on the book stack behind
it) and turns toward you. It shows the live values
and states of all three FMUs (ECU state, level, time left, heater, latch, eject; chamber temperature,
browning, toast colour; toast mode, carriage, height, tumble) and two 60 s plots. It is quad meshes
like the info panel (`vrsim/toaster_monitor.py`), redrawn about three times a second while shown; the
text is antialiased with a second, blended quad layer per colour for partly covered pixels.
Offscreen check: `uv run python tests/render_toaster_monitor.py`.

When the timer runs out, the latch lets go, the toast shoots out, tumbles, and lands flat on the desk
in front of the laptop. Its colour comes from the thermal model's browning. Push the lever again
to put it back in the slot. Undo and "clear all" leave the toaster alone. `VRSIM_TOASTER=none`
turns it off, and `vrsim.config.ToasterConfig` sets where it stands.

`ToasterConfig` also holds settings that default to the behaviour above (real time, authored flight)
and that the Newton edition's sidecar turns on for VR (environment variables in brackets):

| Setting | Newton edition | Effect |
|---|---|---|
| `toast_seconds` (`VRSIM_TOAST_SECONDS`) | 10 | a cycle takes this many real seconds at any dial level; the FMUs run faster while the heater is on |
| `eject_height` (`VRSIM_EJECT_HEIGHT`) | 0.9 | the toast rises this high above the slot [m] (authored 0.46 m) and lands on the same spot |
| `new_toast_after_eject` (`VRSIM_NEW_TOAST`) | on | the landed toast stays on the desk as a grabbable copy, a fresh slice goes into the slot |
| `monitor_scale` (`VRSIM_MONITOR_SCALE`) | 2 | size of the FMU monitor |
| `monitor_offset` (`VRSIM_MONITOR_OFFSET`) | on the books | monitor position `x,y[,h]` in the toaster frame; `h` = height of the surface it stands on |

The sidecar (`vrsim/toaster.py`) reads the binding, steps the FMUs with FMPy at 100 Hz in real
time, and routes their values. The app measures the lever, dial and button joints from the body
poses (PhysX or Newton), then applies the outputs it gets back: the latch, the toast pose and colour,
and the glow.
Offline: `uv run python -m vrsim.toaster` runs the dbeyond 600 s scripted user in a few seconds.

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
  toaster.py    dbeyond smart toaster: ovfmi binding -> FMPy co-simulation, app protocol
  toaster_monitor.py  the FMU monitor panel: Pillow -> merged quads per colour (+ antialiasing layers)
  sidecar.py    voice, catalog, downloads, browser pages, HUD text and toaster FMUs for the apps
cpp/                      the C++ app (ovxr + ovrtx + ovstage + ovphysx), see cpp/README.md
newton_app/               the Newton edition (vrnx package, own uv project), see newton_app/README.md
tests/smoke_headless.py   catalog -> physics -> grab -> throw, no headset needed
tests/toaster_physics.py  toaster on the desk in ovphysx: lever push, spring, latch, dial, FMU loop
tests/render_toaster.py   toaster offscreen: toast pose, colour and slot glow writes reach ovrtx
assets/dbeyond/           smart toaster product, toast, FMUs, VR station layer (copied from dbeyond)
```

## Verification status

- ✅ **In the headset (Quest 3 over Link, C++ app):** stereo rendering, controllers, voice spawning,
  smooth grabbing, the catalog panels, the Studierstube environment and the 72 fps Minimal mode.
- ✅ Headless: `uv run python tests/smoke_headless.py mug football`. Assets settle on the table,
  a grabbed body tracks the hand to within 5 mm, and a throw lands about 3 m away.
- ✅ HUD text: `uv run python tests/render_hud.py` renders two messages offscreen through
  ovstage `points` writes into ovrtx and checks that the text appears and changes.
- ✅ Wit.ai with the real token: synthesized English speech ("Add three mugs", "Clear all
  objects", "Spawn a football", "Undo") is transcribed and parsed correctly with an untrained app.
- ✅ Toaster, headless: `tests/toaster_physics.py` (ovphysx on the desk: lever pushed to the bottom,
  spring return, latch hold, dial turn and hold, FMUs latching and heating from measured joints),
  `tests/render_toaster.py` (runtime toast pose, colour and glow writes in RT2), and a sidecar +
  `vrsim_cpp` run without a headset, where the toaster spawns and the co-simulation starts.
  ❌ **Not yet tried in the headset:** grabbing the lever and dial, the cancel poke, the Minimal-mode
  glow brightness (`kGlowEmission`).
- ✅ **Newton edition in the headset:** controllers, A + X recentering, voice, the catalog, and the
  toaster (lever latch and spring return, 10 s cycles, higher eject, fresh toast, monitor on the
  books). It runs at half rate, ~36 fps Minimal and ~30 fps RT2. Headless and offscreen tests and the
  validation report: [newton_app/VALIDATION.md](newton_app/VALIDATION.md).
- Developed and measured on one machine only (RTX 6000 Ada, Windows 11).

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
- Downloads use an IPv4 Hugging Face client with a 10 s connect timeout (`vrsim/catalog.py`):
  huggingface.co publishes IPv6 addresses, and where IPv6 is unreachable the stock client stalled
  without a timeout and blocked every later spawn.
- The Newton edition reaches only half the Quest's 72 Hz (rendering 14–16 ms plus physics and Python
  per frame), and a spawn freezes frames for 0.5–1 s while Newton rebuilds.

## License

[MIT](LICENSE). This covers this repository's own code and documents. Third-party items keep their
own terms: the logos in `assets/logos/` are trademarks of their owners (see `SOURCES.txt`), the HDRI
in `assets/hdri/` is CC0 from Poly Haven, and the NVIDIA libraries and SimReady assets the app
downloads are under their own licenses.
