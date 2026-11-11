# VR SimReady, Newton edition

The same Quest 3 experience as the C++ app (say "add a mug", place it with the red marker, grab
it, throw it, push the smart toaster's lever, browse the SimReady catalog in VR), rebuilt on the
library set the Omniverse Realtime Viewer skill prescribes, with
**[ovnewton](https://github.com/NVIDIA-Omniverse/ovnewton) and Newton instead of ovphysx/PhysX**.

```
                  ┌───────────── one ovstage (GPU hierarchy model), owned by vrnx.stage ─────────────┐
 Quest 3 ◄─ ovxr ◄─ ovrtx (RTX)          ovnewton ──► Newton (XPBD, CUDA graph)        vrnx.app
                  └─ renders the sealed    builds the model from the populated stage   hands, toaster, panels:
                     publication           (Newton never parses USD)                   one transform write per frame
                                                     ▲
 sidecar: python -m vrnx.sidecar ────────────────────┘ voice (Wit.ai), SimReady catalog + downloads, browser pages,
 (vrsim.sidecar + ovnewton fixups, port 47801)        HUD text, toaster FMUs (FMPy), asset fixups for ovnewton
```

| Library | Role here |
|---|---|
| `ovstage` | The one runtime scene. Environment, props, browser pages and panels are populated into it (`population.open_usd_with_desc` / `add_usd_reference_from_string`), and every per-frame change is an OVStage data-plane write at an application-owned ordinal. |
| `ovrtx` | Renders the stage (`Renderer.attach_ovstage`). RT2 or Minimal mode. |
| `ovxr` | OpenXR session, eye cameras, `bind_frame(N)` seals ordinal N and renders it (`StageOrdinalMode.APPLICATION`). |
| `ovnewton` | Builds the Newton model from the populated stage (`add_ovstage`), binds it (`attach_ovstage`), restores authored drive targets (`update_from_ovstage`), optionally publishes state (`update_to_ovstage`). |
| `newton` / `warp` | XPBD rigid bodies, joints and drives; all substeps of a frame replay as one CUDA graph. |

The realtime-viewer rules it follows: `ovrtx` renders everything; one owner (`vrnx.stage.RuntimeStage`)
creates the stage, populates it, hands out ordinals and seals them; user USD files stay untouched
(asset fixes are override layers, viewer state lives in OVStage runtime writes); the sidecar
exchanges DTOs (JSON lines), never stage handles.

## Run

```powershell
cd newton_app
uv sync
.\start_newton_vr.bat          # Minimal rendering; ".\start_newton_vr.bat rt2" for full RT2
```

Or by hand, in two terminals: `uv run python -m vrnx.sidecar` and `uv run python -m vrnx --render minimal`.
The app waits for the headset (Quest Link / Air Link as the active OpenXR runtime) and for the sidecar.
Controls are those of the C++ app (see the repository README): grip grabs and throws, trigger points,
places and fixes, A/X push-to-talk, B undo, Y clear, thumbstick click opens the catalog, left menu
toggles the toaster's FMU monitor. **A + X together recenters the room** (the desk in front of where
you look; voice: "recenter the room"). The C++ app's sidecar (port 47800) can run alongside.
If Quest Link starts after the app, restart the app: its OpenXR instance does not see a headset
that appears later.

Toaster settings (sidecar options; environment variables in brackets; the shared `vrsim` defaults
keep the C++ app's behaviour):

| Option | Newton default | Effect |
|---|---|---|
| `--toast-seconds` (`VRSIM_TOAST_SECONDS`) | 10 | a toasting cycle takes this many real seconds, whatever the dial level; 0 = the FMUs' 60–200 s |
| `--eject-height` (`VRSIM_EJECT_HEIGHT`) | 0.9 | how high the toast rises above the slot [m], landing on the same spot; 0 = authored 0.46 m |
| `--no-new-toast` (`VRSIM_NEW_TOAST`) | on | after the toast lands it stays on the desk as a grabbable copy and a fresh slice goes into the slot |
| `--monitor-scale` (`VRSIM_MONITOR_SCALE`) | 2 | size of the FMU monitor panel |

Prerequisites, Wit.ai and Hugging Face tokens and the microphone are set up as for the C++ app
(repository README). This app needs no build step.

## What ovnewton changed

ovnewton validates authored physics strictly and does not fall back to guesses, which surfaced
three things PhysX had absorbed silently. Each is handled without touching the source assets:

1. **The asset's Newton physics variant.** SimReady packages carry `PhysX`, `MuJoCo` and `Newton`
   variant sets (35 of the packages downloaded so far). `vrnx.usd_fixups` writes a small override
   layer per wrapper that selects `Newton = Enabled` and turns the others off.
2. **Joint targets below a rigid body.** The toaster's joints name a mesh under each body.
   ovnewton raises `InvalidPhysicsError`; the override layer retargets `physics:body0/1` to the
   rigid-body ancestor and re-expresses `localPos/localRot` in its frame, so the joint is unchanged.
3. **Drive targets outside the joint limits.** The lever's spring targets 0.04 m, above its upper
   stop, which is what holds it up. Newton replaced out-of-range targets with the limits' midpoint,
   and the lever sagged to -35 mm, past the toaster's latch threshold. After each rebuild the app
   restores the authored targets with `binding.update_from_ovstage(..., attributes=("jointPositionTarget",))`.

Two performance decisions:

- **Triangle meshes on moving bodies become convex hulls.** Even the toaster's Newton variant
  collides with triangle meshes (17k vertices on the housing). One collision pass cost 3.5–4 ms of
  GPU time. With convex hulls a whole physics frame (2 substeps, contacts every substep) takes
  about 1.2 ms on an RTX 6000 Ada.
- **The environment is imported once.** ovnewton imports one stage into an empty builder, and a
  spawn means a rebuild. The render stage therefore has no environment colliders. They come from
  `environment_physics.usda` through a separate physics-only ovstage, imported once and appended
  to every rebuilt model. Import and finalize took 2–2.7 s per spawn before; a whole rebuild
  (including the collision pipeline and the CUDA graph) now takes 0.5–1.1 s.
- **Body poses go out in the app's transform batch.** `binding.update_to_ovstage` costs about 3 ms
  per frame (GPU tensors, velocities and joint state as well). The app writes the poses in its own
  per-frame OVStage write instead (`local + world + resetXformStack`, about 0.1 ms extra).
  `--publish ovnewton` switches back to ovnewton's publication.

## Layout

```
vrnx/
  app.py          frame loop: hands, placement, toaster joints, panels, browser, spawn/remove (port of cpp/src/main.cpp)
  stage.py        runtime owner: ovrtx + ovstage, ordinal clock, population, data-plane writes
  physics.py      ovnewton model build/rebuild, environment colliders, drive kernel, CUDA graph
  usd_fixups.py   override layers: Newton physics variant, joint retargeting
  xr_input.py     controller input on ovxr's OpenXR session (thumbstick, menu, haptics)
  link.py         sidecar connection (JSON lines over TCP)
  sidecar.py      vrsim.sidecar + environments for this app + fixups on every spawn
tests/
  physics_headless.py   env + toaster + mug + football: settle, grab within 1 mm, throw; rebuild and frame times
  render_physics.py     offscreen ovrtx: a dropped mug and a moved controller change the image
  app_headless.py       the real App.run loop with ovxr stubbed and scripted hands (needs the sidecar):
                        lever push latched by the ECU FMU, mug lift and throw, final render
  bench_physics.py      GPU time per physics frame for a few solver/collision settings
```

The content modules (catalog, voice, browser pages, HUD text, toaster FMUs and monitor, environment
generation) are the repository's `vrsim` package, shared with the C++ app's sidecar.

## Status and limits

See [VALIDATION.md](VALIDATION.md). Verified headless and offscreen, and in the headset (Quest 3 over
Link, RTX 6000 Ada, Windows 11): controllers, A + X recentering, toaster cycles with eject and fresh
toast, voice, catalog. About 30 fps in RT2 and 36 fps in Minimal (half rate: rendering takes 13–16 ms,
physics 3–7 ms per frame).

- A spawn or removal rebuilds the Newton model: about 0.6–1.1 s during which no frame is
  submitted (the Quest's compositor holds the last frame). Most of it is ovnewton scanning the
  whole stage, including the render-only prims.
- ovnewton's CI covers Linux; it works here on Windows (pure Python), but that is not a supported
  platform upstream.
- Dynamic props collide with convex hulls of their triangle-mesh colliders, so a hollow asset
  (a bowl, the toaster's slots) is solid to other props. Bodies of one jointed asset do not collide
  with each other, and its jointed parts not with the static environment (their hulls overlap).
- Newton's XPBD skips a joint's drive while the joint rests on a limit; the app lifts the released
  toaster lever off its bottom stop so its spring can return it.
- As in the C++ app, picking uses bounding spheres, and only assets with rigid bodies can be spawned.
