---
name: ovxr-vr-physics
description: Build or debug PC-VR apps on NVIDIA's ovxr/ovrtx/ovstage stack streaming to a Meta Quest over Link, with Newton or ovphysx (PhysX) physics, SimReady USD assets from Hugging Face, Wit.ai voice and controller grab/throw. Use when prims don't move or render in the headset, frame rate is low, OpenXR won't start, the Quest mic is silent, or when adding spawn/grab/UI-panel features to this project (vrsim/ Python app, cpp/ native app + Python sidecar).
---

# ovxr + physics VR apps (Quest 3 over Link)

Hard-won rules from building `vrsim/` (Python: ovxr + Newton) and `cpp/` (C++: ovxr + ovphysx,
with `vrsim/sidecar.py` for voice, catalog and HUD text). Versions: ovrtx 0.6, ovstage 0.3,
ovxr 0.1, ovphysx 0.7, Newton/Warp current, Meta Horizon Link runtime.
**Check the "nothing moves" list first.** Each item there cost hours, and none is documented upstream.

## 1. Nothing moves / nothing appears

1. **Hierarchy model must be `GPU_INCREMENTAL`** (Python `StageConfig`, C `ovstage_initialize`).
   The default CPU_INCREMENTAL makes ovrtx apply a prim's transform **only on its first change**.
   Later writes, including `ovxr.update_prim_transforms`, silently never render.
2. **Write three attributes per moving prim, in one batch per frame:**
   `omni:fabric:localMatrix` + `omni:fabric:worldMatrix` (same matrix) +
   `omni:fabric:resetXformStack = true`. A worldMatrix-only write (which is what ovphysx's
   `writeWorldTransformsToOvstage` does) does not move child meshes, and SimReady visuals are
   children of the rigid-body prim. `xformOp:transform` / `omni:xform` alias localMatrix, so
   never put them in the same batch.
3. **Ordinals seal.** In ovxr APPLICATION mode every call that takes an ordinal seals it.
   `update_prim_transforms(N)` followed by `bind_frame` needs ordinal N+1. Spawns and removals
   go between frames at their own ordinals. ovphysx drains exactly those with
   `update_from_ovstage(M, M)`, never the frame ordinals it produced itself.
4. After add/remove of references call `compute_hierarchy`, otherwise new prims have no world xform.
5. Verify offscreen before blaming the headset: render with ovrtx to PNG
   (`tests/render_hud.py` pattern). It is far faster than a Link round-trip.

## 2. Runtime / device setup

- **`XR_ERROR_RUNTIME_UNAVAILABLE`**: the Oculus → "Meta Horizon" app update leaves HKLM
  `OpenXR\1\ActiveRuntime` pointing at a deleted json. Set `XR_RUNTIME_JSON` process-locally to
  the Meta Horizon runtime json (see `_ensure_openxr_runtime` in `vrsim/app.py`).
- **"Creating renderer" hangs for ~2 min**: ovrtx recompiles RT shaders when Link is active.
  This is normal on first run per configuration. The native C SDK keeps its own cache.
- **Quest mic over Link**: use the shared WASAPI device "Headset Microphone (Oculus Virtual
  Audio Device)". The WDM-KS OCULUSVAD endpoint is exclusive and records silence. Capture at
  the native rate and resample (`resample_poly`) to 16 kHz for Wit.ai.
- pyopenxr can drive actions on ovxr's borrowed instance/session handles. Locate controllers at
  the predicted display time from `begin_frame`.

## 3. Performance (RTX 6000 Ada, Quest 3 at 72 Hz)

| Setup | Result |
|---|---|
| RT2 renderer, 1 frame in flight | render ≈ 9 ms + submit ≈ 4.6 ms, **36 fps** |
| RT2, `--frames-in-flight 2` | 72 fps with an empty scene; scanned assets drop it to ~37 fps |
| **Minimal render mode** | **71–72 fps** with several scanned assets (bind ≈ 6.5–7 ms) |

- CPU work (physics, input, writes) is under 1 ms. The renderer is the budget.
- **Minimal mode lighting limits:**
  - It uses only the first DistantLight. Dome lights, other lights and the
    `omni:rtx:sceneDb:ambientLight*` settings do nothing, and there is no fill light, so
    shadows are pure black.
  - The sky is therefore faked: a 300 m sphere of faces bound to constant-emissive palette
    materials, with `doNotCastShadows`. Emission shows only in `omni:rtx:minimal:mode = 3`.
  - The environment carries a `/World/Render/MinimalSource` RenderProduct with that mode, and
    the C++ app passes it as ovxr's `source_render_product`.
  - **That source product needs exactly one `camera` target** (a placeholder Camera is fine).
    Without one, every `ovxr_bind_frame` fails with "camera relationship ... has an invalid
    tensor shape" and the headset is black. Offscreen ovrtx renders don't catch this.
  - Large occluders over the work area (the pergola beams) get
    `primvars:doNotCastShadows = 1` via `EnvironmentConfig.shadowless_prims`, applied only in
    the Minimal environment. `omni:rtx:minimal:castShadows = 0` would also drop the props'
    contact shadows.
- Next levers: lower per-eye resolution, foveation (on by default, `--no-foveation` to compare),
  lod01 meshes. Scanned SimReady packages are huge (football 304 MB, pumpkin 353 MB).
- Newton CUDA graphs: capture on a dedicated `wp.Stream` inside `ScopedDevice` + `ScopedStream`,
  and fall back to eager stepping if capture fails ("Warp error: unknown stream" otherwise).

## 4. Physics and grabbing

- SimReady assets already carry `RigidBodyAPI`, mass and convex colliders, so a spawn is just a
  USD reference. ovphysx picks it up from the shared stage. Newton needs a wrapper USD with unit
  and up-axis correction (`vrsim/assets.py`) and a model rebuild on spawn.
- Register the PhysX USD schemas (`physx-usd-schemas` wheel, `PHYSX_USD_SCHEMAS_ROOT`) before
  opening stages in the native app, or the physics APIs are ignored.
- **Grab = velocity drive, not teleport.** Compute the target once per frame over the frame
  horizon (velocity = (target − current) / frame_dt) and set linear and angular velocity before
  stepping. Driving per substep or from stale poses jitters. Throw = hand velocity × scale on release.
- **Jointed assets: whole first, then parts.** The first hand on a prop holds all its bodies
  rigidly: each body gets its own velocity toward grip × offset × (pose relative to the touched
  body), one `Physics::setBodyVelocities` write over an explicit prim-handle selection. Same
  velocity for all bodies would fight the joints when rotating. Only while that hand holds it can
  the other hand take a movable part (sidecar `body_movable`: body1 of a non-fixed joint), which
  it then drives alone, and the first hand leaves it out. Releasing either hand re-captures the
  relative poses for the remaining hand, with no throw. A throw gives each body v + w × r.
  Joints in nvidia/simready-assets are mostly standalone PxJoints (eyeglasses, waste bin, blue box,
  ratchet). Articulated ones (compressor) aren't covered: RIGID_BODY queries skip articulation links.
  Python ovphysx: `write_sync` prefetches its groups into `session.groups` (`fetch_next` then returns None).
- Step on a fixed 1/72 s with substeps (144 Hz scene) and use wall-clock dt only for UI animation.

## 5. Voice and catalog

- Meta's Voice SDK is Unity/Unreal only. Call **Wit.ai `/speech`** directly with the CLIENT
  token (`wit_ai_api.txt`, gitignored). Pad with 0.4 s silence and keep 0.5 s pre-roll, or the
  first word gets clipped. Parse the transcript locally too, so an untrained Wit app still works.
- NVIDIA's library is **`nvidia/simready-assets`** (public, about 2,700 packages, small:
  4–30 MB each). `nvidia/simready-catalog` doesn't exist (404 even when signed in). Partner repos:
  `ALLSIDES/simready-scanned-objects` (scans, around 300 MB each) and `Vertiv/simready-assets`.
  The HF token comes from `huggingface_api.txt`, exported as `HF_TOKEN` by `vrsim/config.py`.
- **nvidia/simready-assets colliders sit inside instanceable references, and ovphysx 0.7
  ignores colliders in instance prototypes.** The body falls through everything. The spawn
  wrapper adds `instanceable = false` overs on the outermost instances
  (`vrsim/assets.py:_asset_overs`), and the C++ app references `@wrapper@</Prop/Asset>`.
- **Some packages ship their own `PhysicsScene`** (ForgeCAD). Referenced under the app's stage it
  becomes a second PhysX scene, and PhysX logs "environment ID is not supported in CPU
  broadphases" for every shape. `_asset_overs` also writes `active = false` on those scenes.
- **Search: SimReady Central first.** `catalog.central_search()` POSTs to
  `https://nvidia-simready-central.hf.space/search_hybrid` (NVIDIA USD Search API, no token, about
  0.5 s, 3.6k assets across about 20 OEM repos). Keep only hits with a `hybrid` or `name`
  explanation, because the vector leg always returns something ("football" gave ping-pong balls).
  Hit paths are `hf://huggingface.co/datasets/<repo>/resolve/<sha>/<pkg>/simready_usd/<file>`, so
  download with that `revision`. Opaque package names (PalatialSim `D00364`) take the VLM
  `object_type`. Central doesn't index ALLSIDES, so exact local names win. Timeout or
  `SIMREADY_CENTRAL_URL=none` falls back to the local index.
  `UsdStage.Traverse()` skips instance proxies, so inspect with `Usd.TraverseInstanceProxies()`.
- Their `PhysX` variant (default `Disabled`) only switches the colliders from convex
  decomposition to SDF. The default variant is the cheaper choice for a CPU scene.
- Crawling 180k file paths: group the files by package before choosing root USDs, otherwise
  the crawl is quadratic.
- Fuzzy-match spoken names with a sound-alike fallback (ASR hears "mark" for "mug").
- Download files concurrently in a worker pool and report progress. Never block the frame
  loop. "Latest request wins" for catalog page builds.

## 6. In-VR UI panels

- World-anchor panels; re-place one only when it drifts more than ~70° out of view. Head-locked
  panels can't be pointed at.
- **One hit test across all panels: the nearest surface owns the trigger.** Handling panels in
  sequence lets a panel behind steal the press (bug: the info panel's × fired the catalog tile
  behind it). See `hudInFront()` in `cpp/src/main.cpp`.
- Use hysteresis on the trigger (press above 0.7, release below 0.5) and a short haptic on every hit.
- Grip on a move handle must be checked **before** prop grabbing, and should mark the squeeze as consumed.
- On Windows, import `scipy.spatial` eagerly in Python processes that read stdin, to avoid a
  DLL-load deadlock.

## Run

```powershell
uv run python -m vrsim                                              # Python app
uv run python -m vrsim.sidecar --mic "Headset Microphone (Oculus"   # C++ app, terminal 1
cpp\build\vrsim_cpp.exe --render minimal                            # C++ app, terminal 2
```
Build: `cpp\scripts\build.bat`. Stop `vrsim_cpp.exe` first, because the exe is locked while it runs.
