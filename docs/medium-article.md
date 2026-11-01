# "Add a Mug": Voice-Spawning Physics-Ready 3D Assets in VR with NVIDIA ovxr, PhysX and SimReady

*Speak a word, catch the object, throw it across the room. What it took to wire RTX rendering,
real physics and Hugging Face assets into a Meta Quest 3, and the pitfalls nobody documents.*

---

## The idea

I wanted a VR sandbox that feels like a magic trick. You put on a Quest 3 and say *"add a
football"*. A few seconds later a photoscanned football drops onto the table in front of you.
You grab it and throw it, and it bounces like a football should.

No hand-authored game objects and no collider tweaking. Every object is a **SimReady** asset
from Hugging Face, and its physics comes with the file. The goal was to find out whether
"physically correct by default" content can make spawning anything as simple as asking for it.

<!-- IMAGE: headset POV, a football mid-throw over the table -->

## The building blocks: NVIDIA's "ov" libraries

Omniverse used to mean a big application platform (Kit, extensions, Nucleus). The new **ov
libraries** take the opposite approach. They are small, standalone C libraries with Python
bindings that you embed in *your own* application. You install them with `pip`, or download a
C SDK archive, and no Omniverse install runs in the background. All of them are pre-release,
so APIs still change between versions. I used four of them.

### ovstage: the shared scene

[ovstage](https://github.com/NVIDIA-Omniverse/ovstage) is the foundation. It holds an
[OpenUSD](https://openusd.org) scene in memory as fast, vectorized, GPU-capable data:
transforms, velocities, materials, hierarchy and metadata. You can read and write it in bulk
from C or Python, with zero-copy paths and DLPack tensor interchange.

Two concepts matter in practice:

- **Ordinals.** Every write is tagged with an increasing number, and consumers ask "what changed
  since ordinal N?". That is how several engines share one scene without stepping on each
  other's updates.
- **Hierarchy computation.** ovstage derives world transforms from local ones. Which
  computation model you choose turns out to matter a lot (see the pitfalls below).

The point of ovstage is that **renderer, physics and sensors all read and write the same scene**
instead of each keeping its own copy and syncing it.

### ovrtx: RTX rendering as a library

[ovrtx](https://github.com/NVIDIA-Omniverse/ovrtx) embeds Omniverse's RTX renderer and sensor
simulation (cameras, lidar, radar) into an application. It is aimed at robotics, synthetic data
and industrial visualization. Since version 0.4 it *attaches* to an ovstage and renders whatever
is in it, picking up changes by ordinal. It offers several render modes, from full real-time
ray tracing to a lighter "minimal" mode, which will matter once we look at frame rates.

### ovxr: ovrtx in a headset

[ovxr](https://github.com/NVIDIA-Omniverse/ovxr) bridges ovrtx to **OpenXR**. It borrows your
ovrtx renderer and ovstage, creates the OpenXR session, renders head-tracked stereo (or
four-view foveated inset) frames and submits them to the runtime. Any OpenXR runtime works. For
a Quest that runtime is Meta's Link software on the PC, which streams the frames to the headset
over a cable or Wi-Fi. ovxr 0.1 ships for Windows and Linux x86_64.

### ovphysx: PhysX on the shared stage

[ovphysx](https://github.com/NVIDIA-Omniverse/PhysX) is NVIDIA PhysX packaged the same way. You
point it at an ovstage, and it reads the scene's **USD physics schemas** (rigid bodies, colliders,
mass, physics materials), simulates them and hands the results back as tensors. It ships as a
Python wheel (with Warp array output) and a C/C++ SDK. I used the C SDK.

## What is SimReady?

A typical 3D model is made to be *looked at*: a mesh and some textures. A simulator needs much
more. How heavy is it, where is its center of mass, what shape does it collide with, how
slippery and bouncy is it, what is it called, and is it a prop or part of the building?

**SimReady** is NVIDIA's specification for USD assets that answer those questions in the file
itself, so the asset behaves correctly the moment it is dropped into a simulation. Here is what
I found when I opened one of the assets I used, a 3D-scanned balance board:

- **Physics.** `PhysicsRigidBodyAPI` and `PhysicsMassAPI` on the root, with a mass of 0.305 kg,
  center of mass and inertia. There are 72 collision meshes with `PhysicsCollisionAPI` and an
  approximation setting, plus a physics material with friction and restitution. PhysX settings
  sit in a switchable `PhysX` variant.
- **Correct units and orientation.** Meters, Z-up, declared on the stage.
- **Semantics.** A Wikidata label (`Q804619`, "balance board") on the asset and material labels
  on its materials. That is useful for synthetic data and, in my case, for search.
- **Metadata and validation.** Name, category, description, license (CC-BY-4.0), author,
  extents and mass. It also records which validation profile it passed:
  `Robotics-Prop`, version 4.0.0. Profiles come from **SimReady Foundation**, which publishes
  the specifications, profiles and validation tools. A profile groups the features a given use
  case needs: collision and physics for manipulation, joints for articulated equipment,
  semantic labels for synthetic data, and materials for sensor simulation ([NVIDIA's
  announcement][central-blog] has a good overview).
- **Structure.** A small root file that pulls in payloads for the base, the meshes and the
  levels of detail (LOD) per body.

For an app like mine, this changes the job completely. Spawning a new object is not an import
pipeline. It's **one USD reference**.

## SimReady on Hugging Face

SimReady assets are distributed as **Hugging Face datasets**, so you can browse them on the web
and download them with the standard `huggingface_hub` tools. In October 2026 NVIDIA announced
**[SimReady Central][central-blog]**: an open, free, ungated library of more than 2,000
SimReady OpenUSD assets on Hugging Face. Each manufacturer publishes and maintains its own
dataset:

- **`nvidia/simready-assets`**: NVIDIA's own library and the largest, with about 2,700 packages.
  They range from mugs, boxes and golf balls to office chairs and hospital equipment.
- **Partner and OEM repositories**, also public:
  - Robot arms and grippers from Comau, FANUC, Franka Robotics, Robotiq, Standard Bots and
    Universal Robots.
  - `Vertiv/simready-assets`: data-center racks and cooling equipment for AI factory digital
    twins.
  - `PalatialSim/palatial-nv-asset-library`: Palatial's household and food objects, including
    the NVIDIA Robotics 1K dataset.
  - LG Electronics appliances such as refrigerators and dishwashers, and ForgeCAD's articulated
    props.
  - `ALLSIDES/simready-scanned-objects`: photoscanned everyday objects such as balls, mugs,
    fruit and sports gear.

To find assets across all these datasets, SimReady Central runs a **search service** built on
NVIDIA USD Search, at [nvidia-simready-central.hf.space][central-space]. It combines keyword
search with a SigLIP2 image embedding and captions generated from each thumbnail. Behind the web
UI is a plain HTTP API: you POST a query to `/search_hybrid` and get JSON back. No token is
needed. When I tried it, it had indexed 3,660 assets and answered in about half a second.
"fruits" returned citrus, passion fruit and canned fruit, with no category list on my side.
Each hit includes the dataset, a revision pinned to a commit, the path to the USD file, a
generated caption and the asset's dimensions. According to the announcement, Hugging Face has
also built USD Search into its own platform search.

My app sends every voice query to this API first. "Find fruits" now opens the catalog on
lemons, lychees, guavas and passion fruit, including Palatial assets whose files are only called
`D00364.usd`. The panel labels them with the generated `object_type`, such as "citrus fruit".
Two details matter:

- **Keep only hits with a text match.** The vector part of the search always returns
  *something*. Asking for "football" returned ping-pong balls and a tire rack, all with nearly
  the same score. Each hit includes an explanation of what it matched, so I drop hits that only
  the embedding matched.
- **Keep a local fallback.** The search runs as a Hugging Face Space, and it doesn't index every
  repo. For example, the ALLSIDES scans are missing, and that's where the football is.

Each asset is a folder with a `simready_usd/` directory (root file plus payloads), thumbnails
and metadata. For the fallback, my app also crawls the NVIDIA, ALLSIDES and Vertiv repos into a
local index of about 2,800 entries (the listing alone is 180,000 files). An exact name in that
index wins, and it takes over if Central times out. Either way, the app downloads an asset's
files only when someone asks for it, from the commit the search result names.
Sizes vary a lot. NVIDIA's modeled props are 4–30 MB, while photoscanned assets with all their
LODs and textures are large: the football is 304 MB and the pumpkin 353 MB.

One surprise: NVIDIA's packages put each collision mesh inside an *instanceable* USD reference,
and ovphysx ignores colliders inside instances. The first mug I spawned had mass but no shape,
and it fell straight through the desk. The fix is one `instanceable = false` override in the
layer that spawns the asset. ForgeCAD's packages had the opposite problem: each brings its own
`PhysicsScene`, which becomes a second PhysX scene next to the app's. An `active = false`
override in the same layer switches it off.

<!-- IMAGE: Hugging Face dataset page of a SimReady asset -->

## Our setup

### Hardware

- **Meta Quest 3**, connected over **Quest Link** (USB-C cable; Air Link also works)
- **Windows 11 workstation** with an **NVIDIA RTX 6000 Ada** (48 GB)
- The Quest's built-in microphone, which shows up in Windows as an input device while Link is active

### Software

| Piece | Version | Role |
|---|---|---|
| ovstage | 0.3 | Shared USD scene |
| ovrtx | 0.6 | RTX rendering |
| ovxr | 0.1 | OpenXR session, stereo rendering to the headset |
| ovphysx | 0.7 | PhysX on the shared stage |
| OpenXR SDK headers | 1.1 | Controller actions on ovxr's session |
| Meta Horizon Link | — | OpenXR runtime, streams to the Quest |
| Wit.ai | — | Speech-to-text and intents |
| Python sidecar: huggingface_hub, usd-core, Pillow | — | Catalog downloads, asset inspection, panel text |

The app is C++ (MSVC, CMake + Ninja). CMake fetches the ovrtx, ovstage and ovxr SDKs, and the
prebuilt ovphysx SDK comes from its GitHub release. Because ovphysx pins the same ovstage
version as ovxr, all three libraries share one stage in one process.

### The world: a garden study under an HDRI sky

The room is *Studierstube*, a small garden pavilion with a desk, built in Blender and exported
as a single USDZ. It is static architecture with no physics, so the app adds a static convex
collider to every solid mesh when it loads the scene. It also shifts the scene so the deck sits
at your real floor level and the desk is in front of you.

Lighting comes from an **HDRI sky dome**: a 4K equirectangular image of a partly cloudy sky
(Poly Haven, CC0). It works both as the visible backdrop and as image-based light, plus a
distant light aimed at the HDRI's sun for crisp shadows. ovrtx applies no auto-exposure, so the
intensities had to be tuned by rendering offscreen and measuring pixel brightness. Below the
horizon a sky-only HDRI just mirrors its clouds, so a wide ground plane covers it.

<!-- IMAGE: the pavilion under the HDRI sky, seen from the standing spot -->

### Accounts and API tokens you need

Only two services need a key, and both have free tiers:

| Service | What to create | Where | Used for |
|---|---|---|---|
| **Wit.ai** (Meta) | An app, then copy its **Client Access Token** | wit.ai → your app → *Management → Settings* | Speech-to-text and intents. Required for voice. |
| **Hugging Face** | A **Read** access token | huggingface.co → *Settings → Access Tokens* | Optional for the public SimReady repos, but it raises the rate limits. Required for gated repos. |

- **Wit.ai** shows two tokens. The *Client Access Token* is enough for the `/speech` endpoint,
  so that's the one the app uses. The *Server Access Token* is only needed if you train intents
  through the API. You don't have to train anything: the app parses the transcript locally as a
  fallback.
- **Hugging Face**: `nvidia/simready-assets` and the partner repos are public, so downloads work
  anonymously. Without a token, `huggingface_hub` warns about rate limits, which matters when a
  single asset is dozens of files. Some SimReady repos (several kitchen and home sets, for
  example) are gated: accept their terms on the website first, and the token gives access.
- Keep both tokens in small text files next to the code (`wit_ai_api.txt`,
  `huggingface_api.txt`) or in environment variables (`WIT_AI_TOKEN`, `HF_TOKEN`), and keep them
  out of version control.

Things that need **no** key: the ov libraries (from PyPI, `pypi.nvidia.com` and GitHub
releases), the OpenXR SDK, and Poly Haven's HDRIs. The Quest needs only a Meta account and the
Link app on the PC. Developer mode isn't needed for Link.

### Architecture

```
            ┌──────────────── one ovstage (GPU hierarchy model) ────────────────┐
 Quest 3 ◄─ ovxr ◄─ ovrtx (RTX)      ovphysx (PhysX)                 vrsim_cpp.exe
            └─ renders the stage      simulates the stage's USD physics  writes transforms
                                                ▲
 Python sidecar ────────────────────────────────┘ voice (Wit.ai), catalog search, downloads, HUD text
```

The key idea is **one stage, many readers**. The renderer and the physics engine both work on
the same ovstage. A spawned asset is just a USD reference added to that stage. PhysX discovers
its rigid body and colliders from the asset's own schemas, so there is no conversion step.

Everything that isn't real-time lives in the Python sidecar: speech, catalog search (SimReady
Central, with a local fuzzy index as fallback),
multi-hundred-megabyte downloads and text rasterization for the in-VR panels. The sidecar talks
to the app in JSON lines over localhost. Everything on the frame path is native.

Each frame does the following:

1. Ask OpenXR for the predicted head and controller poses.
2. Turn grabs into velocity targets for the held objects.
3. Step PhysX by a fixed 1/72 s, with substeps at 144 Hz.
4. Read all body poses in one batch.
5. Write all transforms (bodies, controllers, pointer rays, panels) to ovstage in one batch.
6. Let ovxr render and submit the frame.

### Controls

| Input | Action |
|---|---|
| A / X (hold) | Push-to-talk: "add a football", "spawn three mugs", "what balls do you have?", "undo", "clear all" |
| Grip | Grab the nearest object, or pull the one you're pointing at |
| Release grip | Throw |
| Trigger | Pointer ray, and selecting things on panels |
| Thumbstick click / flick | Open the asset browser / page through it |
| "Cancel", or × on the info panel | Stop a running download |

## Why not Meta's Voice SDK?

The Voice SDK ships for Unity and Unreal. This app runs on the PC and streams to the headset, so
there is no Unity in the loop. Wit.ai *is* the service behind the Voice SDK, though, so calling
its `/speech` endpoint directly gives the same recognition. Any intents you train carry over if
you later build a standalone Unity client.

Small things that made voice feel reliable:

- **0.5 s pre-roll and 0.4 s silence padding.** Without them the first word ("add") gets clipped.
- **Local parsing as a fallback.** "add / spawn / give me … \<thing\>" works even with an
  untrained Wit app.
- **Sound-alike matching.** The recognizer hears "mark" when you say "mug", so catalog search
  falls back to phonetic keys.
- **Use the right microphone.** Over Link, the headset mic exists twice in Windows. Only the
  shared WASAPI endpoint works; the WDM-KS one records silence.

<!-- IMAGE: asset browser panel with thumbnails -->

## The five pitfalls that cost me the most time

### 1. Objects rendered once and then froze

Physics ran and the logged poses changed, but in the headset nothing moved. The cause:
ovstage's default **CPU-incremental hierarchy model**. With it, ovrtx applies a prim's transform
only on its *first* change. Switching to `GPU_INCREMENTAL` fixed it instantly.

### 2. The body moved but the mesh didn't

SimReady assets put their visual meshes *under* the rigid-body prim. Writing only
`omni:fabric:worldMatrix` (the obvious attribute, and what the ovphysx helper does) leaves the
children behind. The write that works is **localMatrix + worldMatrix + resetXformStack**,
all three in one batch, every frame.

### 3. Ordinals seal

ovstage versions every change by ordinal. In ovxr's application mode, *any* call that takes an
ordinal seals it. Updating transforms at N and then binding the frame at N silently drops the
update. Number every step explicitly.

### 4. The OpenXR runtime vanished

After Meta renamed the Oculus app to "Meta Horizon", the registry still pointed at the old
runtime path, and every launch failed with `XR_ERROR_RUNTIME_UNAVAILABLE`. Setting
`XR_RUNTIME_JSON` in the process fixed it without touching system settings.

### 5. Grabbing jittered

The first grab implementation wrote a target velocity on every physics substep from slightly
stale controller poses, and the held object shivered. Computing the drive **once per frame over
the whole frame horizon** (velocity = distance to target ÷ frame time) made it rock solid.
Throwing is then simply the hand's velocity at release.

## Performance: the renderer is the budget

On an RTX 6000 Ada, everything I control (physics, input, transform writes) takes under **1 ms**
per frame. The rest goes to rendering:

| Mode | Frame rate |
|---|---|
| Full RT, 1 frame in flight | 36 fps |
| Full RT, 2 frames in flight | 72 fps empty, about 37 fps with scanned assets |
| Minimal render mode | **71–72 fps** with several scanned assets |

Photoscanned SimReady assets are heavy (a football package is 304 MB, a pumpkin 353 MB). For
VR at 72 Hz, the lighter render mode is the pragmatic choice. Next on the list are LOD meshes
and lower per-eye resolution.

## UI in VR: small details matter

- **World-anchored panels**: the info panel sits 35° to your left and stays put while you
  turn to read it, re-placing itself only when it's far out of view.
- **The nearest surface wins**: when the info panel floated in front of the catalog, its close
  button fired the catalog tile *behind* it, because each panel handled the trigger in turn.
  The fix is a single hit test across all panels.
- **Haptics on every hit**, with hysteresis on the trigger, makes the pointer feel physical.
- **Never block the frame**: downloads run in a worker pool, a spinner shows progress, and
  catalog paging uses "latest request wins".

<!-- GIF: open catalog, page with thumbstick, pick an item, it drops in -->

## Takeaways

1. **SimReady content changes the economics.** When physics is part of the asset, "spawn
   anything" is a USD reference, not a pipeline.
2. **A shared stage is the right abstraction.** Renderer and physics never exchange meshes;
   they read the same scene.
3. **Test offscreen first.** Rendering a frame to PNG answers "is it the transform or the
   headset?" in seconds instead of minutes.
4. **Profile before optimizing.** I expected physics or the sidecar to be the bottleneck. It
   was the renderer every time.
5. **Split by latency, not by language.** Keep the frame loop native, and put everything that
   waits on the network or a human in a separate process.

---

## Appendix: other things I tried

The C++ app is where I ended up, not where I started. These are the detours, and why I left them.

### A. An all-Python version with Newton

The first version was pure Python: ovxr, ovstage and ovrtx through their Python bindings,
**[Newton](https://github.com/newton-physics/newton)** (NVIDIA's open-source GPU physics
engine built on Warp) for physics, and pyopenxr for controllers. It was the fastest way to get
something on screen, and most of its code (voice, catalog, panels) lives on as the C++ app's
sidecar.

Why I moved on:

- **Two copies of the scene.** Newton builds its own model from USD, so every spawn meant
  rebuilding the physics model and copying poses back into ovstage each frame. ovphysx
  simulates the stage that ovrtx renders, so there is nothing to copy.
- **Asset preparation.** Newton needed a wrapper USD per asset to fix units and up axis
  before import. With ovphysx, a SimReady asset is referenced as it is.
- **Python on the frame path.** Even with CUDA graphs, a 72 Hz loop in Python leaves little
  headroom for the extra work of input, grabbing and panels.

Python-specific traps worth knowing: capture Warp CUDA graphs on a dedicated stream (or you
get "Warp error: unknown stream"), and on Windows import `scipy.spatial` at startup, because
loading its DLLs later, while another thread reads stdin, deadlocked.

### B. Meta's Voice SDK in Unity, or CloudXR with a Unity client

The "official" voice route is Meta's Voice SDK, which only exists for Unity and Unreal. Running
it would have meant a Unity app on the Quest streaming from the PC via CloudXR, which is a whole
second application just to get a microphone. Calling Wit.ai directly from the PC gives the same
recognition with none of that.

### C. Building PhysX from source

Before finding ovphysx's prebuilt C SDK, I set up a build of the raw PhysX 5 SDK from source.
It wasn't needed: ovphysx bundles PhysX and, more importantly, already understands USD physics
and ovstage. With raw PhysX I would have had to write that part myself.

### D. A curated asset shortlist

The simplest option was a hand-picked list of a dozen assets with known names. I chose to crawl
the Hugging Face repos into a searchable index instead. That way "what balls do you have?" and
"add a pumpkin" work for anything in the catalog, and new assets show up without code changes.
Then I added a hand-written category table, so "find fruits" could expand to apples and
bananas. Once the [SimReady Central search API][central-space] became available, it replaced
that table: it covers more datasets and understands descriptions like "something to drink
from". The crawl and the table remain as the fallback.

### E. Full ray tracing in the headset

The full real-time ray-tracing renderer looks great, but with photoscanned assets it stayed
around 37 fps, even after moving to two frames in flight. For VR, the minimal render mode at
72 fps was the better trade (see the performance table above).

---

## References

- NVIDIA, [*Announcing SimReady Central: An Open Library of Simulation-Ready 3D Assets for
  Robotics and AI Factories*][central-blog], Hugging Face blog, October 21, 2026.
- [SimReady Central search][central-space] (USD Search API on Hugging Face Spaces).
- [`nvidia/simready-assets`](https://huggingface.co/datasets/nvidia/simready-assets) on Hugging
  Face.

[central-blog]: https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories
[central-space]: https://nvidia-simready-central.hf.space/

---

*Built with NVIDIA ovxr, ovrtx, ovstage and ovphysx, SimReady assets on Hugging Face, Wit.ai
and a Meta Quest 3.*

<!-- TODO before publishing: screenshots/GIFs, repo link, author bio, check version numbers -->
