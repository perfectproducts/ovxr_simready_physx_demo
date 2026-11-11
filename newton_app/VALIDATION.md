# Viewer Validation Report: VR SimReady, Newton edition

## Metadata

| Field | Value |
|---|---|
| App or repo | `newton_app/` (package `vrnx`) |
| Date | 2026-11-11 |
| Delivery path | Native local PC-VR: ovrtx → ovxr → OpenXR (Quest Link), Python process |
| Runtime environment | Windows 11, RTX 6000 Ada, driver 591.59, Python 3.12, uv |
| Library versions | ovnewton 0.2.1, ovstage 0.3.0.1791986420, ovrtx 0.6.0.1791947691, ovxr 0.1.0.5842, newton 1.7.0, warp-lang 1.19.0 |
| Scene inputs | Generated Studierstube environment, dbeyond smart toaster, NVIDIA mug, SimReady football |

## Commands Run

| Step | Command | Result | Artifact |
|---|---|---|---|
| Setup | `uv lock && uv sync` (in `newton_app`) | Pass | `uv.lock` |
| Build | none (pure Python) | Skipped | |
| Environments | `uv run python -m vrnx.sidecar --prepare-only` | Pass | `.cache/generated/environment_newton*.usda` |
| Runtime launch | `python -m vrnx.sidecar` + `python -m vrnx` (no headset) | Pass to "Waiting for headset": renderer, environment, 346 colliders, sidecar link, mug + toaster spawned, FMUs running | console log |
| Physics | `python tests/physics_headless.py` | Pass | console |
| Render | `python tests/render_physics.py` | Pass | `.cache/newton_test/newton_before.png`, `newton_after.png` |
| App loop | `python tests/app_headless.py` (sidecar running), and with `ovnewton` publication | Pass / Pass | `.cache/newton_test/app_headless.png` |
| GPU timing | `python tests/bench_physics.py` | ~1.2 ms per physics frame | console |

## Evidence Checklist

| Evidence | Status | Notes |
|---|---|---|
| Startup and dependency output captured | Pass | All libraries resolve from the pinned set; ovnewton installs on Windows |
| First nonblank rendered frame captured | Pass | Offscreen ovrtx frames of the desk (RT2) |
| Camera orbit, pan, zoom, fit-to-stage | Skipped | Not applicable: the camera is the tracked headset (ovxr eye cameras) |
| Object selection | Pass (headless) | Grip picks the nearest body / jointed part; the scripted hand took the toaster lever as a part |
| Scene switch | Skipped | Not applicable: one environment; spawns and removals rebuild Newton from the stage |
| Render settings | Partial | `--render rt2|minimal` selects the renderer config; Minimal not rendered in this run |
| Focused physics proof | Pass | Mug settles on the desk (z 0.783), tracks a hand target to 0.9 mm, is thrown 2.1–2.6 m; dropped mug and moved controller change the rendered image (48k pixels); toaster lever pushed to -59.5 mm, the ECU FMU latches it (`lever_target -0.0595`) and it stays down after release |
| Shutdown / reconnect | Pass (headless) | App shutdown detaches cleanly; the sidecar accepts the next app connection |

Physics proof details (the realtime-viewer physics checklist, adapted to ovnewton):

- Process model: in-process. ovnewton builds the model from the app's own populated ovstage
  (`add_ovstage` / `attach_ovstage`); no competing USD population, Newton parses no USD.
- Pose publication: app transform batch (`omni:fabric:localMatrix` + `worldMatrix` +
  `resetXformStack`) at the frame ordinal, sealed by `bind_frame` (stub: `advance_write_floor`);
  `--publish ovnewton` uses `binding.update_to_ovstage` (`omni:xform`) instead. Both pass.
- Direct ovrtx transform writes: 0.

Measured per frame in the app loop (headless, 72 Hz pacing, toaster + mug): physics step about
2.3 ms median (drive upload 0.2, graph launch 0.5, readback 1.6 while waiting on 1.9 ms of GPU work),
all transform writes 1.6 ms. A rebuild after a spawn takes 0.5–1.1 s.

## Issues And Waivers

| ID | Severity | Summary | Resolution |
|---|---|---|---|
| 1 | Medium | Headset (2026-11-11): controllers, A + X recentering (ovxr managed anchor), toaster cycles with lever latch/return, higher eject and fresh toast, voice and catalog verified by the user. Frame rate is half rate: ~36 fps Minimal, ~30 fps RT2 | Fixed on the way: double-applied tracking anchor (controllers, realign), lever stuck on its stop, jointed-part contacts |
| 2 | Medium | Spawn/remove rebuild stalls frames for 0.5–1.1 s | Documented; most of it is ovnewton scanning the whole stage |
| 3 | Low | ovnewton is CI-tested on Linux only; used on Windows here | Works in all tests above |
| 4 | Low | Dynamic triangle-mesh colliders are hulled (hollow props are solid) | Documented trade for 3–4x lower GPU physics time |

## Result

Overall status: Pass headless, offscreen and in the headset. Open: 72 fps (both modes run at half rate).
