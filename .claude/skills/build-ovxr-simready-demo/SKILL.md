---
name: build-ovxr-simready-demo
description: Set up, run or rebuild from scratch the Quest 3 VR demo where you say "add a mug", a SimReady asset from Hugging Face appears, and you grab and throw it. Stack is NVIDIA ovxr + ovrtx + ovstage + ovphysx (C++), a Python sidecar and Wit.ai voice. Use this whenever someone wants to reproduce, install, clone, port or create this demo, or start a similar voice-driven or physics-driven PC-VR app on the ov libraries, even if they only say "get the VR demo running" or "build something like the SimReady VR sandbox". For debugging one symptom in an existing build, use ovxr-vr-physics instead.
---

# Build the ovxr SimReady VR demo

The finished demo: stand in a small pavilion in a Quest 3, say "add a mug", place the SimReady
asset that arrives from Hugging Face, then grab it, throw it, or pull on its jointed parts.
Reference implementation: https://github.com/perfectproducts/ovxr_simready_physx_demo

Pick the path first:

- **Reproduce it:** the user wants this demo running. Follow "Path A".
- **Build it or a variant:** the user wants the agent to write the app. Follow "Path B", using the
  repo as reference code.

Read the sibling skill `ovxr-vr-physics` before writing or debugging any frame-loop code. It holds
the verified rules for moving prims, lighting, grabbing, voice and panels. If it isn't installed,
get it from `.claude/skills/ovxr-vr-physics/SKILL.md` in the repo above. This skill says what to
build and in what order. That one says why things fail.

## Requirements

Check these before starting, and tell the user what is missing.

| Need | Detail |
|---|---|
| OS | Windows 11. The C++ build scripts are Windows-only |
| GPU | NVIDIA RTX. Developed and measured on one RTX 6000 Ada (48 GB) only, so treat frame rates elsewhere as unknown |
| Headset | Meta Quest 3 over Quest Link (cable or Air Link), with Meta Horizon Link as the OpenXR runtime |
| Toolchain | Visual Studio 2022 Build Tools, `uv`, `gh`, Python 3.11 to 3.13 |
| Wit.ai | An app and its Client Access Token. Required for voice, free, no training needed |
| Hugging Face | Read token. Optional for public repos, but it raises rate limits |

Keys go in `wit_ai_api.txt` and `huggingface_api.txt` at the project root, or in `WIT_AI_TOKEN` and
`HF_TOKEN`. Ask the user to create these files themselves, and never print their contents. In a new
project, add both names to `.gitignore` before the first commit.

The library versions are pinned for a reason: ovphysx 0.7 pins the same ovstage build as ovxr 0.1,
which is what lets all of them share one stage in one process. Keep them together:
`ovrtx==0.6.0.1791947691`, `ovstage==0.3.0.1791986420`, `ovxr==0.1.0.5842`,
`ovphysx==0.7.0.74035321`, `physx-usd-schemas==25.11.2.72696984`. These libraries are pre-release.
If you move any version, re-run the offscreen checks in `references/milestones.md`.

## Path A: reproduce the demo

```powershell
git clone https://github.com/perfectproducts/ovxr_simready_physx_demo.git
cd ovxr_simready_physx_demo
uv sync

# ovphysx C SDK, prebuilt
gh release download ovphysx-0.7.0 -R NVIDIA-Omniverse/PhysX -p "ovphysx-windows-x86_64-*.zip" -D cpp\_deps
Expand-Archive cpp\_deps\ovphysx-windows-x86_64-0.7.0.74035321.zip cpp\_deps\ovphysx-sdk

cpp\scripts\build.bat            # fetches the ovrtx, ovstage and ovxr SDKs and OpenXR headers
uv run python tests/smoke_headless.py mug football   # physics check, no headset needed
.\start_vr.bat                   # Minimal rendering, 72 fps. Use "rt2" for full lighting, about 37 fps
```

Expect these, because they look like failures and aren't:

- The first launch compiles RTX shaders, and with Link active that takes minutes. The log sits at
  "Creating renderer". Check for a compile before treating it as a hang.
- `vrsim_cpp.exe` is locked while it runs, so stop it before rebuilding.
- `start_vr.bat` refuses to start if a sidecar already holds port 47800.

Then hand over to the user with a short list: hold A or X and say "add a mug", aim the red marker,
pull the trigger, grip to grab. Controls are in `README.md`.

## Path B: build it or a variant

Work through `references/milestones.md` in order. Each milestone has a check, and the order is
chosen so a failure can only come from what was just added.

Use this architecture unless the user chooses otherwise, and explain any change:

```
            one ovstage, GPU hierarchy model
 Quest 3 <- ovxr <- ovrtx        ovphysx          C++ frame loop
                                    ^
 Python sidecar (JSON lines on localhost): voice, catalog search, downloads, panel text
```

- **One stage, many readers.** Renderer and physics share the ovstage, so a spawned asset is one USD
  reference and needs no conversion step. SimReady assets carry their own mass and colliders.
- **Split by latency.** Everything on the frame path is native. Anything that waits on a network or
  a person lives in the sidecar, so it can never stall a frame.

Ask the user about these up front, because they change what gets built. Recommend the first option
in each case:

1. **Voice:** Wit.ai called directly from the PC. Meta's Voice SDK exists only for Unity and Unreal,
   and this app has neither. The alternatives are a CloudXR + Unity client, or local speech-to-text.
2. **Physics:** ovphysx in C++. The all-Python version with Newton is quicker to start, but it keeps
   a second copy of the scene and rebuilds its model on every spawn. It reached the headset first
   and the demo then moved to C++.
3. **Asset search:** SimReady Central first (`POST /search_hybrid`, no token), with a local index of
   the Hugging Face repos as fallback. Central doesn't index every repo.

## Working with a headset you can't see

The agent never sees the headset, and most of the hard bugs in this project lived in that gap. These
habits are what made the work converge:

- **Prove it offscreen first.** Render a frame to PNG with ovrtx (`tests/render_hud.py` is the
  pattern) and look at it. It answers "is this the transform or the headset?" in seconds.
- **Offscreen is not proof of VR.** A render product that passed every offscreen check made every
  `ovxr_bind_frame` fail, and the headset stayed black, because offscreen rendering skips ovxr. Tuned
  RT2 lighting also meant nothing in Minimal mode. Test in the render mode the headset uses, and
  report anything unseen in VR as untested.
- **Log what you can't see.** Print head and controller poses, body heights, frame time and a
  per-stage time breakdown. Then "the ball doesn't fall" can be separated into "physics is wrong"
  and "physics is right, the image isn't following it".
- **Ask for observations, not verdicts.** Ask where the ball is and whether the spheres follow the
  hands. Make each request specific and quick to answer with the headset on.
- **Don't disrupt a live session.** Build and test, then wait for the user to say restart.
- **Change the design when tuning stalls.** Placing assets by dragging them with the pointer jittered
  through two fixes, and the user replaced it with a marker that places the asset on trigger.
  Removing the moving object removed the bug.

## Before calling it done

Report status per item as headset-tested, offscreen-tested, headless-tested or untested. Then capture
anything new you learned in `ovxr-vr-physics`, so the next run starts ahead of this one.
