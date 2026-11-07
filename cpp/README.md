# VR SimReady Physics: native C++ edition

The same experience as the Python app (say "add a mug", grab it, throw it), with the per-frame
work in C++, and **PhysX via [ovphysx](https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx)**
instead of Newton.

```
            ┌──────────────── one ovstage (GPU hierarchy model) ────────────────┐
 Quest 3 ◄─ ovxr ◄─ ovrtx (RTX)      ovphysx (PhysX, CPU scene)      vrsim_cpp.exe
            └─ renders the stage      simulates the stage's USD physics  writes body/controller/HUD transforms
                                                ▲
 Python sidecar (python -m vrsim.sidecar) ──────┘ voice (Wit.ai), SimReady catalog/downloads, HUD text
```

**Why ovphysx:** it simulates the USD physics in the same ovstage that ovrtx renders. SimReady
assets already carry rigid bodies, mass and convex colliders in USD, so a spawned asset is just
a USD reference. Physics picks it up through `update_from_ovstage`, with no collider conversion.

**Why a sidecar:** voice, fuzzy catalog search, Hugging Face downloads and text rasterization
aren't real-time. They reuse the tested Python code and talk to the app over
`127.0.0.1:47800` (JSON lines, protocol in `vrsim/sidecar.py`). Everything on the frame path is
native.

## Per-frame loop

1. `ovxr_begin_frame`, then controllers located at the predicted display time
2. grab targets become `linearVelocity` / `angularVelocity` writes into ovphysx
3. `ovphysx_step` at a fixed 1/72 s (the scene substeps at 144 Hz)
4. one `ovphysx_read` of every body's position and orientation
5. one ovstage write of `localMatrix` + `worldMatrix` + `resetXformStack` for all bodies, in
   the read's prim-list order, plus controllers, rays and the HUD, all at ordinal N
6. `ovxr_bind_frame(N)` seals N and renders it, then `ovxr_submit_frame`

Spawns and removals run between frames at their own sealed ordinals. ovphysx drains exactly
those (`update_from_ovstage(M, M)`) and never the frame ordinals it produced.

Lessons from testing that the code depends on:
- **The GPU hierarchy model is required.** With ovstage's default CPU-incremental model, ovrtx
  applies a prim's transform only on its first change.
- **Bodies need local + world + reset.** Writing only `omni:fabric:worldMatrix` (what
  ovphysx's `writeWorldTransformsToOvstage` helper does) doesn't move child meshes, and
  SimReady visuals are children of the rigid-body prim.

## Build

Prerequisites: Visual Studio 2022 Build Tools, and the project's Python environment
(`uv sync` at the repo root, which also installs `physx-usd-schemas`).

```powershell
# 1. ovphysx C SDK (prebuilt, NVIDIA Omniverse License)
gh release download ovphysx-0.7.0 -R NVIDIA-Omniverse/PhysX -p "ovphysx-windows-x86_64-*.zip" -D cpp\_deps
Expand-Archive cpp\_deps\ovphysx-windows-x86_64-0.7.0.74035321.zip cpp\_deps\ovphysx-sdk

# 2. Configure + build (fetches ovrtx/ovstage/ovxr SDKs, OpenXR headers, nlohmann/json)
cpp\scripts\build.bat
```

## Run

```powershell
uv run python -m vrsim.sidecar --mic "Headset Microphone (Oculus"   # terminal 1
cpp\build\vrsim_cpp.exe                                              # terminal 2
```

The sidecar writes `.cache/generated/environment_physics.usda` (CPU PhysicsScene plus floor and
table colliders) and the app loads it. By default it references `assets/Studierstube.usdz`. The
scene is shifted so its deck is at floor level and its desk is in front of you, and its meshes get
static convex colliders. Set `VRSIM_ENVIRONMENT` to another Z-up, metre-scale .usd/.usdz file, or
to `none` for the generated floor and table. The first launch of the native renderer compiles RTX
shaders, which takes about 2 minutes, mostly while Quest Link is active.

## Status

- ✅ Builds clean (MSVC /W4). Runs: environment, PhysX schema registration, ovphysx attach,
  ovxr context and sidecar link.
- ✅ The pipeline was validated offscreen with the same stack through the ovphysx Python
  bindings. A spawned SimReady football is picked up by `update_from_ovstage` and falls onto
  the table, and with the local+world+reset write it renders resting there.
- ✅ Runs in the headset: controllers, grab and throw, voice spawning and panels. Minimal mode holds
  71–72 fps with several scanned assets, and RT2 about 37 fps.

`scripts/build_physx.bat` builds the raw PhysX 5 SDK from source. It isn't needed for this
app, which uses ovphysx's bundled PhysX.
