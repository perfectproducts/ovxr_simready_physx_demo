# Milestones for building the demo

Each milestone adds one risk. Don't start the next until its check passes. "Who checks" matters:
only the user can confirm anything marked **headset**. File names refer to the reference repo.

| # | Build | Check | Who checks |
|---|---|---|---|
| 1 | Project skeleton: `uv` project with pinned versions, gitignored key files | `uv sync` succeeds, `git status` shows no key files | Agent |
| 2 | Toolchain proof: build NVIDIA's `minimal-xr` C example from ovxr `v0.1.0`, unchanged | It compiles and links | Agent |
| 3 | Shared stage: ovstage with `GPU_INCREMENTAL`, ovrtx attached, environment loaded | Offscreen PNG shows the scene | Agent |
| 4 | Physics: ovphysx on the same stage, one SimReady asset as a USD reference | The asset falls and rests on the table in a sequence of offscreen frames, not one frame | Agent |
| 5 | Headset: ovxr session and stereo frames | The user sees the scene, the log shows 0 `bind_frame` errors | **Headset** |
| 6 | Controllers: OpenXR actions on ovxr's session, spheres or models at the hands | Log shows valid poses, the user confirms they follow the hands | **Headset** |
| 7 | Sidecar and voice: JSON lines link, Wit.ai `/speech`, local command parser, typed console input | Typed "add a mug" spawns. Synthesized speech transcribes. Then spoken in the headset | Agent, then **headset** |
| 8 | Catalog: SimReady Central search with text-match filter, local index fallback, on-demand download | "add a mug" and "find fruits" return sensible hits, and "rubber duck" is rejected cleanly | Agent |
| 9 | Grab and throw: velocity drive computed once per frame, hand velocity on release | Headless: hand tracked to within 5 mm, throw lands metres away. Then smooth in hand | Agent, then **headset** |
| 10 | Panels: status panel, asset browser with thumbnails, one nearest-surface hit test | Offscreen shows text changing, the user can page, pick and close | Agent, then **headset** |
| 11 | Environment: `assets/Studierstube.usdz` or the user's scene, static colliders, deck at floor level | Spawned props rest on the desk and floor | Agent, then **headset** |
| 12 | Frame rate: per-stage timing, then Minimal against RT2 | Steady 71 to 72 fps in Minimal with several scanned assets | **Headset** |
| 13 | Lighting: HDRI dome for RT2, faked sky for Minimal | Checked offscreen in the mode being shipped, then in VR | Agent, then **headset** |
| 14 | Jointed assets: first hand holds the whole prop, second hand moves a part | Headless two-hand test: hinge turns, joint distance holds. Then by hand | Agent, then **headset** |
| 15 | Placement: red marker on the aimed surface, thumbstick turns, trigger spawns fixed, B cancels | Marker follows the ray, asset appears fixed and its joints still move | **Headset** |
| 16 | One-command start: `start_vr.bat` with CRLF line endings | Double-click start works from a clean state, errors stay readable | Agent, then **headset** |

## Failures to expect at specific milestones

- **3 and 4:** Objects render once and then freeze if the hierarchy model isn't `GPU_INCREMENTAL`.
  Test a sequence of frames, because a single-frame test passes with this bug present.
- **4:** Write `localMatrix`, `worldMatrix` and `resetXformStack` together, or the visible child
  meshes stay behind the rigid body. Register the PhysX USD schemas before opening stages.
- **4 and 8:** `nvidia/simready-assets` colliders sit inside instanceable references, which ovphysx
  ignores, so the asset falls through everything. Add `instanceable = false` overs in the spawn layer.
  Deactivate any `PhysicsScene` a package brings with it.
- **5:** If OpenXR reports `XR_ERROR_RUNTIME_UNAVAILABLE`, set `XR_RUNTIME_JSON` for the process. The
  registry can still point at the old Oculus path.
- **5 and 13:** The Minimal source render product needs exactly one camera target, or the headset is
  black while offscreen renders look fine.
- **7:** Use the shared WASAPI "Headset Microphone (Oculus Virtual Audio Device)". The WDM-KS endpoint
  records silence. Keep 0.5 s pre-roll and 0.4 s silence padding, or the first word is clipped.
- **8:** Drop hits that only the vector search matched. It always returns something.
- **9:** Driving velocity per substep from stale poses makes held objects shake.
- **12:** The renderer is the budget. Physics, input and transform writes took under 1 ms per frame,
  so measure before optimizing anything else. Scanned assets are heavy (a football is 304 MB).
- **13:** Minimal uses only the first DistantLight and has no fill light, so shadows are pure black.
- **16:** Don't parse `netstat` state names, they are localized. Windows batch files need CRLF.
