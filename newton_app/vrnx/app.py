"""VR SimReady, Newton edition: the per-frame app (a port of the C++ app, cpp/src/main.cpp).

One process owns one ovstage. ovrtx renders it, ovxr streams it to the headset, and Newton
simulates it through ovnewton. The Python sidecar (python -m vrnx.sidecar) does voice, the
SimReady catalog, downloads, the asset browser pages, HUD text and the toaster's FMUs, over the
same JSON-lines protocol the C++ app uses.

Per frame (ovxr split loop, application-owned ordinals):
    begin_frame -> controllers at the predicted display time -> hand/anchor/toaster drive targets
    -> Newton frame (one CUDA graph: drives + substeps) -> ovnewton update_to_ovstage at ordinal N
    -> controller/HUD/toaster/browser writes at N -> bind_frame(N) seals N and renders it -> submit
Between frames: sidecar messages; spawns and removals change the population at their own sealed
ordinal, and Newton is rebuilt from the stage (vrnx.physics).
"""

from __future__ import annotations

import argparse
import base64
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import vrnx
import ovxr

from vrsim.mathutil import Pose, quat_from_axis_angle, quat_mul, quat_conj, quat_normalize  # noqa: E402

from .link import SidecarLink  # noqa: E402
from .physics import Drive, NewtonPhysics  # noqa: E402
from .stage import RuntimeStage  # noqa: E402
from .xr_input import HandInput, InputFrame, XrInput  # noqa: E402

log = logging.getLogger("vrnx")

SIDECAR_PORT = 47801
# Seated eye height [m] in the world: the 0.78 m desk plus ~0.44 m. Set at start and on every recenter
# (the tracking space's own floor or origin can leave the view at the desk's height). 0 = tracking's height.
EYE_HEIGHT = float(os.environ.get("VRSIM_EYE_HEIGHT", "1.22") or 0)
PHYSICS_DT = 1.0 / 72.0  # Quest 3 refresh
MAX_PROPS = 40
GRAB_ON, GRAB_OFF, GRAB_REACH, RAY_MAX = 0.55, 0.35, 0.12, 8.0
MAX_LINEAR, MAX_ANGULAR, THROW_SCALE = 12.0, 30.0, 1.3
TURN_RATE = 1.6  # [rad/s] thumbstick turning of a placement or a held fixed prop
PART_REACH = 0.025  # [m] grip within this of a jointed part takes the part, not its prop
POKE_REACH = 0.06  # [m] controller to the toaster's cancel button for a trigger press
GLOW_EMISSION = 6.0  # slot strip emissiveColor at full heater glow
DESK_PROP_MAX = 0.35  # [m] largest side of a prop that spawns on the desk
HUD_FLOATS = 6000 * 4 * 3  # vrsim.hud.MAX_QUADS quads x 4 corners x xyz
# Info panel placement: (yaw left of the view [deg], drop below the eyes [m]) in order of preference,
# the first one that doesn't cut through the toaster or its monitor (vrsim.hud FOLLOW_* is the first).
HUD_SPOTS = ((35, 0.25), (35, 0.05), (55, 0.25), (55, 0.05), (-35, 0.25), (-35, 0.05), (20, -0.1))
HUD_DISTANCE = 0.85  # [m] from the head
HUD_HALF = (0.36, 0.125)  # [m] vrsim.hud PANEL_W / 2, PANEL_H / 2
HUD_NAV = ((0.36 - 0.165, -0.125 + 0.024), (0.36 - 0.035, -0.125 + 0.024))  # vrsim.hud NAV_PREV, NAV_NEXT
HUD_NAV_HALF = 0.018  # vrsim.hud NAV_HALF
HUD_CLEARANCE = 0.03  # [m] gap kept to obstacles
HUD_REPLACE_OVERLAP = 0.02  # [m] an anchored panel moves when something reaches this far into it
MONITOR_SIZE = (0.40, 0.30)  # [m] vrsim.toaster_monitor PANEL_W, PANEL_H at scale 1

HANDS = ("/World/Hands/Left", "/World/Hands/Right")
RAYS = ("/World/Hands/LeftRay", "/World/Hands/RightRay")
PANEL, TEXT = "/World/HUD/Panel", "/World/HUD/Panel/Text"
ACCENTS = ("/World/HUD/Panel/AccentBusy", "/World/HUD/Panel/AccentOk", "/World/HUD/Panel/AccentError")
BROWSER, HIGHLIGHT = "/World/Browser", "/World/Browser/Highlight"
MARKER = "/World/Hands/PlaceMarker"
LOADING, SPINNER = "/World/BrowserLoading", "/World/BrowserLoading/Spinner"
RIG = "/World/Rig"  # ovxr tracking anchor: world = rig * tracking pose
MONITOR = "/World/ToasterMonitor"
HIDDEN = Pose((0.0, 0.0, -100.0)).to_row_major(1e-4)
Z = np.array([0.0, 0.0, 1.0])


def axis_angle(axis, angle: float) -> np.ndarray:
    return quat_from_axis_angle(axis, angle)


def rotation_vector(q) -> np.ndarray:
    """Axis * angle of a unit quaternion (shortest arc)."""
    q = quat_normalize(q)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-9:
        return np.zeros(3)
    return q[:3] / s * 2.0 * math.atan2(s, q[3])


def clamp_length(v: np.ndarray, limit: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * (limit / n) if n > limit else v


def facing(pos, head: Pose | None, fallback=(0.0, -1.0, 0.0)) -> Pose:
    """Upright pose at ``pos`` whose +Z faces the head horizontally."""
    pos = np.asarray(pos, dtype=float)
    z = (head.p - pos) if head is not None else np.asarray(fallback, dtype=float)
    z = np.array([z[0], z[1], 0.0])
    z = z / np.linalg.norm(z) if np.linalg.norm(z) > 1e-3 else np.asarray(fallback, dtype=float)
    x = np.cross(Z, z)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, Z, z, pos
    return Pose.from_row_major(m.reshape(-1))


def decode_floats(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32) if text else np.zeros(0, dtype=np.float32)


def flat_forward(head: Pose | None) -> np.ndarray:
    if head is None:
        return np.array([0.0, 1.0, 0.0])
    f = head.axis((0, 0, -1))
    f[2] = 0
    return f / np.linalg.norm(f) if np.linalg.norm(f) > 1e-3 else np.array([0.0, 1.0, 0.0])


# --- scene objects ------------------------------------------------------------------------------
@dataclass
class Prop:
    root: str
    name: str
    handle: int
    body_paths: list[str]
    spheres: list[list[float]]  # per body: centre in the body frame, radius [m]
    movable: list[bool]  # per body: a jointed part (lid, lever) the second hand may take
    anchor: list[Pose] = field(default_factory=list)  # fixed: poses its non-movable bodies are held at
    fix_on_spawn: bool = False
    radius: float = 0.1
    device: str = ""  # "toaster": driven by FMUs in the sidecar; kept by undo / clear


@dataclass
class DeviceJoint:
    """One operated joint of the toaster, measured from body poses (vrsim.toaster.joint_frames)."""

    body: int = -1  # index into the prop's bodies
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    linear: bool = True
    offset: float = 0.0
    lower: float = -1e9
    upper: float = 1e9
    turns: float = 0.0
    last_angle: float = 0.0
    seen: bool = False
    value: float = 0.0  # joint units: m or deg
    hold: float = 0.0  # angular: held here while no hand turns it (the dial's friction)


@dataclass
class Toaster:
    prop: Prop
    base: int
    lever: DeviceJoint
    dial: DeviceJoint
    cancel: DeviceJoint
    toast: str
    toast_material: str
    glow_light: str
    glow_shader: str
    glow_max: float = 30000.0
    latched_below: float = -0.03
    lever_target: float = -0.0005
    toast_local: tuple = (0.0, -0.026, 0.17, 0.0)  # x, y, z [m], tumble about X [deg]
    toast_color: tuple = (0.93, 0.85, 0.68)
    glow: float = 0.0
    color_dirty: bool = True
    glow_dirty: bool = True
    cancel_press_until: float = -1.0
    last_sent: float = -1.0


@dataclass
class MonitorLayer:
    prim: str
    budget: int
    z: float = 0.002
    points: np.ndarray | None = None
    pending: bool = False


@dataclass
class Monitor:
    handle: int = 0
    on: bool = False  # hidden until the menu button or "toggle signal monitor"
    layers: list[MonitorLayer] = field(default_factory=list)
    offset: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.34, 0.19]))
    scale: float = 1.0
    swatch: str = ""


@dataclass
class Hand:
    prop: Prop | None = None
    body: int = -1  # index into the prop's bodies: the one this hand touched
    offset: Pose = field(default_factory=Pose)  # that body's pose in the grip frame
    whole: bool = True  # holds all of the prop rigidly (rel); else only a movable part
    rel: list[Pose] = field(default_factory=list)
    pulling: bool = False
    squeezing: bool = False
    trigger_down: bool = False
    trigger_clicked: bool = False
    grip: Pose = field(default_factory=Pose)
    history: list = field(default_factory=list)  # (time, grip position)
    ray_length: float = 0.0
    last_angular: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def velocity(self) -> np.ndarray:
        if len(self.history) < 2:
            return np.zeros(3)
        (t0, p0), (t1, p1) = self.history[0], self.history[-1]
        return (p1 - p0) / max(t1 - t0, 1e-3)

    def drop(self) -> None:
        self.prop, self.body, self.pulling = None, -1, False


@dataclass
class Browser:
    open: bool = False
    handle: int = 0
    cols: int = 4
    rows: int = 3
    count: int = 0
    tile_w: float = 0.26
    tile_h: float = 0.3
    gap: float = 0.03
    ox: float = 0.0
    oy: float = 0.0
    pad: float = 0.012
    close_rect: tuple = (0.0, 0.0, 0.0, 0.0)
    handle_rect: tuple = (0.0, 0.0, 0.0, 0.0)
    pose: Pose = field(default_factory=Pose)
    placed: bool = False
    hover: int = -1
    reported_hover: int = -1
    hover_widget: int = 0  # 1 = close button, 2 = move handle
    moving_hand: int = -1
    move_offset: Pose = field(default_factory=Pose)
    loading: bool = False
    trigger_down: list = field(default_factory=lambda: [False, False])
    stick_armed: list = field(default_factory=lambda: [True, True])


@dataclass
class Hud:
    points: np.ndarray | None = None
    pending: bool = False
    kind: str = "info"
    hold: float = 0.0
    posted: float = 0.0
    pos: np.ndarray | None = None
    anchor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pose: Pose = field(default_factory=Pose)
    visible: bool = False
    trigger_down: list = field(default_factory=lambda: [False, False])
    nav: bool = False  # the text has page buttons (help): ◀ at HUD_NAV[0], ▶ at HUD_NAV[1]


def in_rect(local, r) -> bool:
    return r[0] <= local[0] <= r[2] and r[1] <= local[1] <= r[3]


# --- the app ------------------------------------------------------------------------------------
class App:
    def __init__(self, args):
        gen = vrnx.REPO_ROOT / ".cache" / "generated"
        minimal = args.render == "minimal"
        env = Path(args.env) if args.env else gen / ("environment_newton_minimal.usda" if minimal else "environment_newton.usda")
        physics_env = gen / "environment_physics.usda"
        self.link = SidecarLink(args.port)
        if not env.exists() or not physics_env.exists():
            log.info("Waiting for the sidecar to write %s (python -m vrnx.sidecar)...", env.name)
            while not (env.exists() and physics_env.exists()):
                time.sleep(0.5)
        self.rt = RuntimeStage(env, minimal_rendering=minimal)
        self.physics = NewtonPhysics(self.rt.stage, str(physics_env), device=args.device, physics_dt=PHYSICS_DT,
                                     substeps=args.substeps)
        self.physics.rebuild(self.rt.population_ordinal)
        self.hulls = self._load_hulls(gen / "placement_hulls.bin")

        source = ""
        if minimal and 'def RenderProduct "MinimalSource"' in env.read_text(encoding="utf-8"):
            source = "/World/Render/MinimalSource"  # cloned per eye: its mode 3 evaluates emission
        self.ctx = ovxr.Context(ovxr.ContextDesc(
            ovrtx_renderer=self.rt.renderer,
            ovstage=self.rt.stage,
            application_name="vrnx.newton",
            display_mode=ovxr.DisplayMode.STEREO,
            session_mode=ovxr.SessionMode.VR,
            environment_blend_mode=ovxr.EnvironmentBlendMode.AUTO,
            local_foveation_mode=ovxr.LocalFoveationMode.WARPED if args.foveation else ovxr.LocalFoveationMode.DISABLED,
            runtime_foveated_submission_policy=ovxr.FeaturePolicy.AUTO if args.foveation else ovxr.FeaturePolicy.DISABLED,
            foveation_unwarp=args.foveation,
            graphics_api=ovxr.GraphicsApi.VULKAN,
            # 2 lets the CPU prepare the next frame while the GPU renders this one.
            max_frames_in_flight=args.frames_in_flight,
            create_render_products=True,
            source_render_product=source,
            tracking_anchor_path=RIG,
            up_axis=ovxr.StageUpAxis.Z,
            meters_per_unit=1.0,
            scene_info_policy=ovxr.SceneInfoPolicy.AUTO,
            ordinal_mode=ovxr.StageOrdinalMode.APPLICATION,
        ))
        probe = self.ctx.probe_openxr()
        if probe.result == ovxr.Result.SUCCESS:
            log.info("OpenXR: runtime=%r system=%r %dx%d", probe.runtime_name, probe.system_name,
                     probe.recommended_swapchain_width, probe.recommended_swapchain_height)
        else:
            log.warning("OpenXR probe: %s (%s). Is Quest Link / Air Link active?",
                        ovxr.result_name(probe.result), self.ctx.last_error)

        self.publish_ovnewton = args.publish == "ovnewton"
        self.running = True
        self.xr: XrInput | None = None
        self.head: Pose | None = None
        self.raw_head: Pose | None = None
        self.rig = Pose()
        self.eyes_levelled = False  # the rig's height is set from the first head pose (EYE_HEIGHT)
        self.pending_floats: list[tuple[str, str, tuple]] = []  # one-off material writes (toast copies)
        self.chord = False  # A + X held for a recenter: no push-to-talk until both are released
        self.props: list[Prop] = []
        self.body_prop: dict[str, tuple[Prop, int]] = {}  # body path -> (prop, index in prop)
        self.hands = [Hand(), Hand()]
        self.hud = Hud()
        self.browser = Browser()
        self.monitor = Monitor()
        self.toaster: Toaster | None = None
        self.start = time.perf_counter()
        self.hud_last = self.start
        self.next_prop = 0
        self.next_spawn_right = True
        self.last_prop: Prop | None = None
        self.pending_spawn: dict | None = None
        self.place_hand = 1
        self.place_valid = False
        self.place_at = np.zeros(3)
        self.place_side = np.array([1.0, 0.0, 0.0])
        self.place_yaw = 0.0
        self.dt = PHYSICS_DT
        self.prof = dict.fromkeys(("drive", "step", "write", "bind", "submit"), 0.0)

    def now(self) -> float:
        return time.perf_counter() - self.start

    def haptic(self, hand: int, amplitude: float, seconds: float) -> None:
        if self.xr:
            self.xr.haptic(hand, amplitude, seconds)

    # --- frame loop ------------------------------------------------------------------------------
    def run(self) -> None:
        last = last_status = time.perf_counter()
        waiting_log = 0.0
        accumulator, frames, steps_total = 0.0, 0, 0
        while self.running:
            result, plan = self.ctx.begin_frame()
            now = time.perf_counter()
            if result != ovxr.Result.SUCCESS:
                if now - waiting_log > 10.0:
                    log.info("Waiting for headset (%s: %s)", ovxr.result_name(result), self.ctx.last_error)
                    waiting_log = now
                self.handle_sidecar()
                time.sleep(0.05)
                last = now
                continue
            try:
                dt = min(now - last, 0.1)
                self.dt, last = dt, now
                if self.xr is None:
                    handles = self.ctx.get_openxr_handles()
                    if handles is not None:
                        self.xr = XrInput(self.ctx, handles)
                frame = self.xr.sync(plan.timing.predicted_display_time_ns) if self.xr else InputFrame()
                if frame.head is not None:
                    self.head = frame.head
                    self.raw_head = self.tracking_pose(frame.head)
                    if not self.eyes_levelled:  # the first head pose: seated eye height, same spot and heading
                        self.eyes_levelled = True
                        self.publish_rig("start")
                if self.xr:
                    self.buttons(frame)
                    self.browser_grab(frame)  # before interact(): grip on the handle must not grab props
                    self.interact(frame)
                    self.place(frame)
                    self.turn_fixed(frame)
                    self.browser_interact(frame)
                    self.hud_interact(frame)

                # Fixed-step physics against real time (at most 3 frames per display frame).
                mark = time.perf_counter()
                accumulator = min(accumulator + dt, 3 * PHYSICS_DT)
                steps = int(accumulator / PHYSICS_DT + 1e-9)
                accumulator -= steps * PHYSICS_DT
                if steps:
                    drives = self.drives(frame)
                    t = time.perf_counter()
                    self.prof["drive"] += t - mark
                    self.physics.step(steps, drives)
                    self.prof["step"] += time.perf_counter() - t
                    steps_total += steps
                self.measure_toaster()

                mark = time.perf_counter()
                ordinal = self.rt.current()
                if self.publish_ovnewton:
                    self.physics.publish(ordinal)
                self.write_visuals(frame, ordinal)
                t = time.perf_counter()
                self.prof["write"] += t - mark
                bound, sealed = self.ctx.bind_frame(plan, dt, ordinal)
                self.prof["bind"] += time.perf_counter() - t
                if sealed:
                    self.rt.resolve()
                if bound != ovxr.Result.SUCCESS:
                    self.ctx.cancel_frame(plan)
                    plan = None
                    if bound != ovxr.Result.NOT_READY:
                        log.warning("bind_frame: %s (%s)", ovxr.result_name(bound), self.ctx.last_error)
                    continue
                t = time.perf_counter()
                submitted, _ = self.ctx.submit_frame(plan)
                plan = None
                self.prof["submit"] += time.perf_counter() - t
                if submitted != ovxr.Result.SUCCESS:
                    log.warning("submit_frame: %s (%s)", ovxr.result_name(submitted), self.ctx.last_error)
            finally:
                if plan is not None:  # an exception left the plan live
                    self.ctx.cancel_frame(plan)

            self.handle_sidecar()  # population changes between frames, at their own ordinals
            frames += 1
            if now - last_status > 2.0:
                n = max(frames, 1)
                log.info("status | %.1f fps | dt %.1f ms | bodies %d | props %d | sidecar %s | per frame ms: "
                         "drive %.2f step %.2f (%.1f steps) write %.2f bind %.2f submit %.2f",
                         frames / (now - last_status), dt * 1000, len(self.physics.labels), len(self.props),
                         "up" if self.link.connected else "down", *(self.prof[k] * 1000 / n for k in ("drive", "step")),
                         steps_total / n, *(self.prof[k] * 1000 / n for k in ("write", "bind", "submit")))
                last_status, frames, steps_total = now, 0, 0
                self.prof = dict.fromkeys(self.prof, 0.0)

    # --- input -----------------------------------------------------------------------------------
    def tracking_pose(self, world: Pose) -> Pose:
        """A world pose back in tracking space. ``transform_openxr_pose`` already maps controller and
        head poses through ovxr's frame-latched anchor (/World/Rig), exactly like the eye cameras,
        so input poses are world poses; only realign needs the tracking-space head. The latched
        anchor can trail ``self.rig`` by a frame after a realign, so ask ovxr which one it used."""
        try:
            anchor = Pose.from_row_major(self.ctx.get_anchor_transform())
        except Exception:
            anchor = self.rig
        return anchor.inverse() * world

    def publish_rig(self, why: str) -> None:
        """Lift the rig so the eyes are at EYE_HEIGHT (it only turns about Z, so world eye z = rig z +
        tracking eye z), then make it ovxr's anchor."""
        if EYE_HEIGHT > 0 and self.raw_head is not None:
            self.rig = Pose((self.rig.p[0], self.rig.p[1], EYE_HEIGHT - float(self.raw_head.p[2])), self.rig.q)
        # ovxr places the eye cameras (and maps controller poses) through its anchor, latched at
        # each frame boundary. Writing /World/Rig alone does not move it: publish the rig as ovxr's
        # managed anchor. That call writes and seals its own ordinal, so the clock moves past it.
        try:
            self.ctx.set_managed_anchor_transform(self.rig.to_row_major(), stage_ordinal=self.rt.current())
        except Exception as exc:
            log.warning("%s: set_managed_anchor_transform failed: %s", why, exc)
        finally:
            self.rt.resolve()
        log.info("Rig (%s): eyes at %.2f m (tracking %.2f m)", why, EYE_HEIGHT if EYE_HEIGHT > 0 else
                 float(self.raw_head.p[2]) if self.raw_head is not None else 0.0,
                 float(self.raw_head.p[2]) if self.raw_head is not None else 0.0)

    def realign(self) -> None:
        """Turn and shift the rig so the head stands over the seat (world origin), facing +Y."""
        if self.raw_head is None:
            return
        f = self.raw_head.axis((0, 0, -1))
        f[2] = 0
        if np.linalg.norm(f) < 1e-3:
            return
        f /= np.linalg.norm(f)
        q = axis_angle(Z, math.pi / 2 - math.atan2(f[1], f[0]))
        head = np.array([self.raw_head.p[0], self.raw_head.p[1], 0.0])
        self.rig = Pose(-(Pose((0, 0, 0), q).axis(head)), q)
        self.publish_rig("realign")
        for h in self.hands:
            h.history.clear()  # no throw from the jump
        self.hud.pos = None
        for h in range(2):
            self.haptic(h, 0.4, 0.05)
        try:
            active = Pose.from_row_major(self.ctx.get_anchor_transform())
        except Exception as exc:
            active = f"unavailable ({exc})"
        log.info("Realigned: rig at (%.2f, %.2f) yaw %.0f deg | head world %s tracking %s | ovxr anchor before %s",
                 self.rig.p[0], self.rig.p[1], math.degrees(2 * math.atan2(self.rig.q[2], self.rig.q[3])),
                 np.round(self.head.p, 2).tolist() if self.head else None, np.round(self.raw_head.p, 2).tolist(), active)

    def buttons(self, frame: InputFrame) -> None:
        h = frame.hands
        # A + X together: recenter the room. The first button already started push-to-talk, so
        # that recording is dropped, and talk stays off until both buttons are up again.
        if h[0].primary and h[1].primary and (h[0].primary_pressed or h[1].primary_pressed):
            self.chord = True
            self.link.send({"t": "talk_cancel"})
            self.realign()
            self.link.send({"t": "realigned"})
        elif self.chord:
            self.chord = h[0].primary or h[1].primary
        elif h[0].primary_pressed or h[1].primary_pressed:
            self.link.send({"t": "talk", "on": True})
            self.haptic(1 if h[1].primary_pressed else 0, 0.25, 0.03)
        elif (h[0].primary_released or h[1].primary_released) and not h[0].primary and not h[1].primary:
            self.link.send({"t": "talk", "on": False})
        if h[1].secondary_pressed:
            self.cancel_placement() if self.pending_spawn else self.undo()
        if h[0].secondary_pressed:
            self.clear()
        if (h[0].menu_pressed or h[1].menu_pressed) and self.monitor.handle:
            self.set_monitor(not self.monitor.on)
            self.haptic(0, 0.3, 0.03)

    def set_monitor(self, on: bool) -> None:
        """Show / hide the toaster's FMU monitor (menu button, or "toggle signal monitor" via the sidecar)."""
        if not self.monitor.handle:
            return
        self.monitor.on = on
        self.link.send({"t": "toaster_monitor", "on": self.monitor.on})  # the sidecar starts / stops frames
        if self.toaster:
            self.toaster.color_dirty = True  # repaint the swatch with the panel

    # --- bodies ----------------------------------------------------------------------------------
    def body_index(self, prop: Prop, b: int) -> int | None:
        if b < 0 or b >= len(prop.body_paths):
            return None
        return self.physics.index.get(prop.body_paths[b])

    def body_pose(self, prop: Prop, b: int) -> Pose | None:
        i = self.body_index(prop, b)
        return self.physics.pose(i) if i is not None else None

    def body_sphere(self, prop: Prop, b: int, pose: Pose) -> tuple[np.ndarray, float]:
        if b >= len(prop.spheres):
            return pose.p, prop.radius
        s = prop.spheres[b]
        return pose.p + pose.axis(s[:3]), s[3]

    def _bodies(self):
        for prop in self.props:
            for b in range(len(prop.body_paths)):
                pose = self.body_pose(prop, b)
                if pose is not None:
                    yield prop, b, pose

    def pick_ray(self, origin, direction):
        best = None
        for prop, b, pose in self._bodies():
            center, radius = self.body_sphere(prop, b, pose)
            rel = center - origin
            along = float(rel @ direction)
            perp = np.linalg.norm(rel - direction * along)
            if 0 < along < RAY_MAX and perp <= radius * 0.9 and (best is None or along < best[2]):
                best = (prop, b, along)
        return best

    def pick_near(self, point):
        """A jointed part the hand is at wins over its prop's body, whose sphere usually contains it."""
        best = best_part = None
        best_gap, best_part_gap = GRAB_REACH, PART_REACH
        for prop, b, pose in self._bodies():
            center, radius = self.body_sphere(prop, b, pose)
            gap = np.linalg.norm(center - point) - radius
            if gap <= best_gap:
                best_gap, best = gap, (prop, b)
            if b < len(prop.movable) and prop.movable[b] and gap <= best_part_gap:
                best_part_gap, best_part = gap, (prop, b)
        return best_part or best

    # --- grabbing --------------------------------------------------------------------------------
    def interact(self, frame: InputFrame) -> None:
        t = self.now()
        for h in range(2):
            hand, hi = self.hands[h], frame.hands[h]
            if not hi.active:
                hand.ray_length = 0
                continue
            hand.history.append((t, hi.grip.p.copy()))
            del hand.history[:-6]
            was = hand.squeezing
            hand.squeezing = hi.squeeze > (GRAB_OFF if was else GRAB_ON)
            if hand.squeezing and not was:
                self.grab(h, hi)
            elif was and not hand.squeezing and hand.prop is not None:
                self.release(h)
            click = hi.trigger > (0.3 if hand.trigger_down else 0.7)
            hand.trigger_clicked = click and not hand.trigger_down
            hand.trigger_down = click
            hand.grip = hi.grip
            if hand.trigger_clicked and hand.prop is not None and hand.whole:
                self.set_fixed(hand.prop, not hand.prop.anchor)
            elif hand.trigger_clicked and hand.prop is None and not self.pending_spawn:
                self.poke_toaster(h, hi)
            hand.ray_length = 0
            if hand.prop is None and hi.aim_valid and hi.trigger > 0.1:
                hit = self.pick_ray(hi.aim.p, hi.aim.axis((0, 0, -1)))
                hand.ray_length = hit[2] if hit else RAY_MAX * 0.4

    def grab(self, h: int, hi: HandInput) -> None:
        hand, other = self.hands[h], self.hands[1 - h]
        hand.pulling = False
        ref = self.pick_near(hi.grip.p)
        if ref is None and hi.aim_valid:  # force-pull whatever the hand points at
            hit = self.pick_ray(hi.aim.p, hi.aim.axis((0, 0, -1)))
            if hit:
                ref, hand.pulling = hit[:2], True
        pose = self.body_pose(*ref) if ref else None
        if pose is None:
            return
        prop, b = ref
        fixed = bool(prop.anchor)
        movable = b < len(prop.movable) and prop.movable[b]
        # A part moves alone while the prop is fixed, or while the other hand holds the rest.
        part = movable and (fixed or (other.prop is prop and other.whole and other.body != b))
        if other.prop is prop and (other.body == b or (not part and not fixed)):  # take it over
            other.drop()
        hand.prop, hand.body, hand.whole = prop, b, not part
        hand.pulling = hand.pulling and not part
        hand.offset = hi.grip.inverse() * pose
        if hand.whole:
            self.capture_whole(hand)
        self.haptic(h, 0.6, 0.04)

    def capture_whole(self, hand: Hand) -> None:
        touched = self.body_pose(hand.prop, hand.body)
        hand.rel = []
        for b in range(len(hand.prop.body_paths)):
            pose = self.body_pose(hand.prop, b)
            hand.rel.append(touched.inverse() * pose if touched and pose else Pose())

    def release(self, h: int) -> None:
        hand, other = self.hands[h], self.hands[1 - h]
        prop = hand.prop
        if prop.anchor:
            # Fixed: the anchor keeps holding it; a released toaster part stops (no dial spin).
            if not hand.whole and prop.device:
                i = self.body_index(prop, hand.body)
                if i is not None:
                    self.physics.set_velocities({i: (np.zeros(3), np.zeros(3))})
        elif other.prop is prop:
            other.whole = True  # the other hand now holds everything; no throw
            self.capture_whole(other)
        elif not hand.pulling:
            self.throw(hand)
        self.last_prop = prop
        hand.drop()
        self.haptic(h, 0.3, 0.02)

    def throw(self, hand: Hand) -> None:
        """The hand's velocity, carried rigidly: each body gets v + w x (its offset from the touched one)."""
        touched = self.body_pose(hand.prop, hand.body)
        if touched is None:
            return
        v, w = hand.velocity() * THROW_SCALE, hand.last_angular
        out = {}
        for b in range(len(hand.prop.body_paths)):
            i, pose = self.body_index(hand.prop, b), self.body_pose(hand.prop, b)
            if i is not None:
                out[i] = (v + np.cross(w, pose.p - touched.p), w)
        self.physics.set_velocities(out)

    def set_fixed(self, prop: Prop, fixed: bool) -> None:
        prop.anchor = [self.body_pose(prop, b) or Pose() for b in range(len(prop.body_paths))] if fixed else []
        for h, hand in enumerate(self.hands):
            if hand.prop is prop:
                hand.pulling = False
                pose = self.body_pose(prop, hand.body)
                if pose:
                    hand.offset = hand.grip.inverse() * pose
                if hand.whole:
                    self.capture_whole(hand)
                self.haptic(h, 0.8 if fixed else 0.5, 0.08 if fixed else 0.04)
        self.link.send({"t": "fixed" if fixed else "unfixed", "name": prop.name, "parts": int(sum(prop.movable))})

    def fix_command(self, fixed: bool) -> None:
        """Voice "fix it" / "free it": a held prop, else the one last let go of, else the newest."""
        target = next((h.prop for h in self.hands if h.prop is not None and h.whole), None) or \
            next((h.prop for h in self.hands if h.prop is not None), None)
        if target is None:
            target = self.last_prop if self.last_prop in self.props else (self.props[-1] if self.props else None)
        if target is None:
            self.link.send({"t": "nothing_to_fix"})
        else:
            self.set_fixed(target, fixed)

    def turn_fixed(self, frame: InputFrame) -> None:
        """Thumbstick while holding a fixed prop turns it about its centre's vertical axis."""
        for h in range(2):
            hand, x = self.hands[h], frame.hands[h].thumb_x
            if hand.prop is None or not hand.whole or abs(x) < 0.25 or not hand.prop.anchor:
                continue
            prop = hand.prop
            centers = [a.p + a.axis(prop.spheres[b][:3]) if b < len(prop.spheres) else a.p
                       for b, a in enumerate(prop.anchor) if not (b < len(prop.movable) and prop.movable[b])]
            if not centers:
                continue
            center = np.mean(centers, axis=0)
            turn = axis_angle(Z, -x * TURN_RATE * self.dt)
            turned = []
            for a in prop.anchor:
                off = Pose((0, 0, 0), turn).axis(a.p - center)
                turned.append(Pose((center[0] + off[0], center[1] + off[1], a.p[2]), quat_mul(turn, a.q)))
            prop.anchor = turned

    def drives(self, frame: InputFrame) -> list[Drive]:
        """This frame's drive targets: fixed props' anchors, the toaster's latch/dial/cancel, hands."""
        horizon = PHYSICS_DT
        drives: dict[int, Drive] = {}
        for prop in self.props:
            if prop.fix_on_spawn:  # placed at the marker: fixed where it first appears
                poses = [self.body_pose(prop, b) for b in range(len(prop.body_paths))]
                if all(poses):
                    prop.anchor, prop.fix_on_spawn = poses, False
            for b, target in enumerate(prop.anchor):
                if b < len(prop.movable) and prop.movable[b]:
                    continue
                i = self.body_index(prop, b)
                if i is not None:
                    drives[i] = Drive(i, target, horizon, MAX_LINEAR, MAX_ANGULAR)
        self.toaster_drives(drives, horizon)
        for h in range(2):  # hands last: a hand on a part overrides its toaster drive
            hand = self.hands[h]
            if hand.prop is None or not frame.hands[h].active:
                continue
            if hand.whole and hand.prop.anchor:
                continue  # holding a fixed prop: the anchor drives it
            if self.body_pose(hand.prop, hand.body) is None:
                hand.drop()
                continue
            if hand.pulling:
                hand.offset = Pose(hand.offset.p * 0.85, hand.offset.q)
                hand.pulling = np.linalg.norm(hand.offset.p) > 0.03
            other = self.hands[1 - h]
            part_of_other = other.body if other.prop is hand.prop and not other.whole else -1
            target = frame.hands[h].grip * hand.offset
            for b in range(len(hand.prop.body_paths)):
                mine = b != part_of_other if hand.whole else b == hand.body
                i = self.body_index(hand.prop, b)
                if not mine or i is None:
                    continue
                goal = target * hand.rel[b] if hand.whole and b < len(hand.rel) else target
                drives[i] = Drive(i, goal, horizon, MAX_LINEAR, MAX_ANGULAR)
                if b == hand.body:
                    pose = self.physics.pose(i)
                    hand.last_angular = clamp_length(rotation_vector(quat_mul(goal.q, quat_conj(pose.q))) / horizon,
                                                     MAX_ANGULAR)
        return list(drives.values())

    # --- placement -------------------------------------------------------------------------------
    @staticmethod
    def _load_hulls(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not path.exists():
            log.info("Placement: no %s; new props drop instead", path)
            return []
        data = np.fromfile(path, dtype=np.float32)
        hulls, i = [], 1
        for _ in range(int(data[0]) if len(data) else 0):
            n = int(data[i])
            lo, hi = data[i + 1:i + 4].astype(float), data[i + 4:i + 7].astype(float)
            planes = data[i + 7:i + 7 + 4 * n].reshape(n, 4).astype(float)
            hulls.append((lo, hi, planes))
            i += 7 + 4 * n
        log.info("Placement: %d scene hulls", len(hulls))
        return hulls

    def cast_scene(self, origin, direction):
        """First scene collider hit by a ray: (distance, surface normal)."""
        best = None
        for lo, hi, planes in self.hulls:
            limit = best[0] if best else RAY_MAX
            with np.errstate(divide="ignore", invalid="ignore"):
                ta, tb = (lo - origin) / direction, (hi - origin) / direction
            near, far = np.nanmax(np.minimum(ta, tb)), np.nanmin(np.maximum(ta, tb))
            if max(near, 0) > min(far, limit) or np.any((np.abs(direction) < 1e-12) & ((origin < lo) | (origin > hi))):
                continue
            t0, t1, normal = 0.0, limit, np.array([0.0, 0.0, 1.0])
            den = planes[:, :3] @ direction
            num = -(planes[:, :3] @ origin + planes[:, 3])
            ok = True
            for k in range(len(planes)):
                if abs(den[k]) < 1e-12:
                    if num[k] < 0:
                        ok = False
                        break
                elif den[k] < 0:
                    if num[k] / den[k] > t0:
                        t0, normal = num[k] / den[k], planes[k, :3]
                else:
                    t1 = min(t1, num[k] / den[k])
                if t0 > t1:
                    ok = False
                    break
            if ok and 0 < t0 <= t1:
                best = (t0, normal)
        return best

    def place(self, frame: InputFrame) -> None:
        """A spawn waits for a spot: a red marker follows one hand's pointer on upward-facing
        surfaces, the thumbstick turns it, the trigger spawns the asset there, fixed."""
        if not self.pending_spawn:
            return
        for h in range(2):
            if frame.hands[h].trigger > 0.2 and frame.hands[1 - h].trigger < 0.05 and self.hands[h].prop is None:
                self.place_hand = h
        hi, hand = frame.hands[self.place_hand], self.hands[self.place_hand]
        if not hi.aim_valid or hand.prop is not None:
            return
        d = hi.aim.axis((0, 0, -1))
        hit = self.cast_scene(hi.aim.p, d)
        hand.ray_length = hit[0] if hit else RAY_MAX * 0.4
        if hit and hit[1][2] > 0.6:
            self.place_at = hi.aim.p + d * hit[0]
            side = np.cross(d, Z)
            self.place_side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-3 else np.array([1.0, 0, 0])
            self.place_valid = True
        if abs(hi.thumb_x) > 0.25:
            self.place_yaw -= hi.thumb_x * TURN_RATE * self.dt
        if hand.trigger_clicked and self.place_valid:
            msg, self.pending_spawn = dict(self.pending_spawn), None
            msg["at_marker"] = True
            self.spawn(msg)
            self.haptic(self.place_hand, 0.8, 0.08)

    def cancel_placement(self) -> None:
        self.pending_spawn = None
        self.link.send({"t": "placing_cancelled"})

    # --- toaster ---------------------------------------------------------------------------------
    def setup_toaster(self, prop: Prop, msg: dict) -> None:
        def body(rel: str) -> int:
            path = f"{prop.root}/{rel}"
            return prop.body_paths.index(path) if path in prop.body_paths else -1

        def joint(role: str) -> DeviceJoint:
            f = msg.get("joints", {}).get(role)
            j = DeviceJoint()
            if not f:
                return j
            j.body = body(f.get("body", ""))
            j.origin = np.asarray(f.get("origin", (0, 0, 0)), dtype=float)
            j.axis = np.asarray(f.get("axis", (1, 0, 0)), dtype=float)
            j.axis /= np.linalg.norm(j.axis)
            j.linear = f.get("kind", "linear") == "linear"
            j.offset = float(f.get("offset", 0.0))
            if isinstance(f.get("lower"), (int, float)):
                j.lower = float(f["lower"])
            if isinstance(f.get("upper"), (int, float)):
                j.upper = float(f["upper"])
            return j

        base = body(msg.get("base", "Asset/Geometry/body_obj_01"))
        lever = joint("lever")
        if base < 0 or lever.body < 0:
            log.warning("Toaster: base or lever body not found; it stays a plain prop")
            return
        r = prop.root
        self.toaster = Toaster(
            prop, base, lever, joint("dial"), joint("cancel"),
            toast=f"{r}/{msg.get('toast', 'Asset/Toast')}",
            toast_material=f"{r}/{msg.get('toast_material', 'Asset/Toast/Looks/ToastMat/Surface')}",
            glow_light=f"{r}/{msg.get('glow_light', 'Asset/ToasterECU/HeaterGlow')}",
            glow_shader=f"{r}/{msg.get('glow_shader', 'Asset/ToasterECU/SlotGlow/Glow/Shader')}",
            glow_max=float(msg.get("glow_max", 30000.0)),
            latched_below=float(msg.get("lever_latched_below", -0.03)))
        if isinstance(msg.get("monitor"), dict):
            self.setup_monitor(msg["monitor"])
        log.info("Toaster ready: lever body %d, dial %d, cancel %d", lever.body, self.toaster.dial.body,
                 self.toaster.cancel.body)

    def setup_monitor(self, m: dict) -> None:
        if not self.monitor.handle:
            layer = Path(m.get("layer", ""))
            if not layer.is_file():
                log.warning("Toaster monitor: layer %s not readable", layer)
                return
            self.monitor.handle, _ = self.rt.add_reference(layer.read_text(encoding="utf-8"), MONITOR)
        z = m.get("z", {})
        self.monitor.layers = [MonitorLayer(f"{MONITOR}/{name}", int(budget), float(z.get(name, 0.002)))
                               for name, budget in m.get("layers", {}).items()]
        self.monitor.offset = np.asarray(m.get("offset", (0, -0.34, 0.19)), dtype=float)
        self.monitor.scale = float(m.get("scale", 1.0))
        self.monitor.swatch = f"{MONITOR}/{m.get('swatch', 'Looks/Swatch/Surface')}"
        self.link.send({"t": "toaster_monitor", "on": self.monitor.on})  # and ask for a full frame
        log.info("Toaster monitor: %d layers (left menu button shows / hides it)", len(self.monitor.layers))

    def monitor_frame(self, msg: dict) -> None:
        for name, data in msg.get("layers", {}).items():
            for layer in self.monitor.layers:
                if layer.prim != f"{MONITOR}/{name}":
                    continue
                r = decode_floats(data)
                n = min(layer.budget, len(r) // 4)
                pts = np.zeros((layer.budget, 4, 3), dtype=np.float32)
                rects = r[:4 * n].reshape(n, 4)
                x0, y0, x1, y1 = rects.T
                pts[:n, 0, 0], pts[:n, 0, 1] = x0, y0
                pts[:n, 1, 0], pts[:n, 1, 1] = x1, y0
                pts[:n, 2, 0], pts[:n, 2, 1] = x1, y1
                pts[:n, 3, 0], pts[:n, 3, 1] = x0, y1
                pts[:n, :, 2] = layer.z
                layer.points, layer.pending = pts.reshape(-1, 3), True

    def toaster_rel(self, j: DeviceJoint) -> Pose | None:
        ts = self.toaster
        base = self.body_pose(ts.prop, ts.base)
        part = self.body_pose(ts.prop, j.body) if j.body >= 0 else None
        return base.inverse() * part if base and part else None

    def read_joint(self, j: DeviceJoint) -> None:
        rel = self.toaster_rel(j)
        if rel is None:
            return
        if j.linear:
            j.value = float(np.clip((rel.p - j.origin) @ j.axis + j.offset, j.lower - 1e-4, j.upper + 1e-4))
        else:
            q = rel.q
            angle = math.degrees(2.0 * math.atan2(float(q[:3] @ j.axis), q[3]))
            angle = math.remainder(angle, 360.0)
            j.turns = j.turns + math.remainder(angle - j.last_angle, 360.0) if j.seen else angle
            j.last_angle = angle
            j.value = j.turns + j.offset
            if not j.seen:
                j.hold = j.value
        j.seen = True

    def measure_toaster(self) -> None:
        """Joint readings to the sidecar's FMUs, up to 60 times a second."""
        ts = self.toaster
        if ts is None:
            return
        for j in (ts.lever, ts.dial, ts.cancel):
            self.read_joint(j)
        t = self.now()
        cancel = ts.cancel.value
        if t < ts.cancel_press_until:  # poked: count it as pressed even if physics lags the 3 mm
            cancel = min(cancel, ts.cancel.lower + 0.0003)
        if t - ts.last_sent < 1.0 / 60.0 or not ts.lever.seen:
            return
        ts.last_sent = t
        msg = {"t": "toaster_in", "lever": ts.lever.value, "dial": ts.dial.value, "cancel": cancel}
        if self.head is not None:  # world pose of the view: the sidecar's 3D sound (the ding) is heard from here
            msg["head"] = [*map(float, self.head.p), *map(float, self.head.q)]
        self.link.send(msg)

    def hand_holds(self, prop: Prop, b: int) -> bool:
        return any(h.prop is prop and h.body == b and not h.whole for h in self.hands)

    def joint_drive(self, j: DeviceJoint, target: float, horizon: float) -> Drive | None:
        """A pose target that moves the joint to ``target`` (m, or deg) along its axis."""
        ts = self.toaster
        base = self.body_pose(ts.prop, ts.base)
        i = self.body_index(ts.prop, j.body)
        if base is None or i is None:
            return None
        part = self.physics.pose(i)
        along = base.axis(j.axis)
        gap = target - j.value
        if j.linear:
            return Drive(i, Pose(part.p + along * gap, part.q), horizon, 0.4, 8.0)
        return Drive(i, Pose(part.p, quat_mul(axis_angle(along, math.radians(gap)), part.q)), horizon, 0.4, 8.0)

    def toaster_drives(self, drives: dict[int, Drive], horizon: float) -> None:
        """The ECU's latch holds the carriage down (against the lever spring); a poke on the cancel
        button pushes it in for a moment; the frictionless dial holds where the hand left it."""
        ts = self.toaster
        if ts is None:
            return
        out = []
        lever_free = ts.lever.seen and not self.hand_holds(ts.prop, ts.lever.body)
        if ts.lever_target < ts.latched_below and lever_free:
            out.append(self.joint_drive(ts.lever, max(ts.lever_target, ts.lever.lower + 0.0002), horizon))
        elif lever_free and ts.lever.value < ts.lever.lower + 0.001:
            # Released on its bottom stop (the latch let go, or a hand pushed it there): Newton's XPBD
            # enforces the limit and skips the joint's spring drive while the joint sits on it, so the
            # lever would stay down. Lift it off the stop; the spring takes it from there.
            out.append(self.joint_drive(ts.lever, ts.lever.lower + 0.004, horizon))
        if self.now() < ts.cancel_press_until and ts.cancel.body >= 0 and not self.hand_holds(ts.prop, ts.cancel.body):
            out.append(self.joint_drive(ts.cancel, ts.cancel.lower + 0.0001, horizon))
        if ts.dial.body >= 0 and ts.dial.seen:
            if self.hand_holds(ts.prop, ts.dial.body):
                ts.dial.hold = ts.dial.value
            else:
                out.append(self.joint_drive(ts.dial, ts.dial.hold, horizon))
        for d in out:
            if d is not None:
                drives[d.body] = d

    def poke_toaster(self, h: int, hi: HandInput) -> None:
        ts = self.toaster
        if ts is None or ts.cancel.body < 0:
            return
        button = self.body_pose(ts.prop, ts.cancel.body)
        if button is None or np.linalg.norm(button.p - hi.grip.p) > POKE_REACH:
            return
        ts.cancel_press_until = self.now() + 0.4
        self.haptic(h, 0.7, 0.05)

    def on_toaster(self, msg: dict) -> None:
        ts = self.toaster
        if ts is None:
            return
        before = ts.lever_target
        ts.lever_target = float(msg.get("lever_target", ts.lever_target))
        if len(msg.get("toast", [])) == 4:
            ts.toast_local = tuple(msg["toast"])
        if len(msg.get("toast_color", [])) == 3:
            color = tuple(msg["toast_color"])
            ts.color_dirty = ts.color_dirty or color != ts.toast_color
            ts.toast_color = color
        glow = float(msg.get("glow", 0.0))
        ts.glow_dirty = ts.glow_dirty or abs(glow - ts.glow) > 1.0
        ts.glow = glow
        latched, was = ts.lever_target < ts.latched_below, before < ts.latched_below
        if latched != was:  # click: the latch catches, or lets go with the ding
            for h in range(2):
                if self.hands[h].prop is ts.prop or not latched:
                    self.haptic(h, 0.5 if latched else 0.9, 0.03 if latched else 0.12)

    def toaster_transforms(self, prims: list, mats: list) -> None:
        ts = self.toaster
        if ts is None:
            return
        base = self.body_pose(ts.prop, ts.base)
        if base is None:
            return
        x, y, z, tumble = ts.toast_local
        prims.append(ts.toast)
        mats.append((base * Pose((x, y, z), axis_angle((1, 0, 0), math.radians(tumble)))).to_row_major())
        # The glow strips and heater light sit outside the physics bodies: keep them with the housing.
        prims.append(ts.glow_shader[:ts.glow_shader.rfind("/Glow/")])
        mats.append(base.to_row_major())
        prims.append(ts.glow_light)
        mats.append((base * Pose((0, 0, 0.16), axis_angle((1, 0, 0), math.pi))).to_row_major())
        if self.monitor.handle:
            p = (base * Pose(self.monitor.offset)).p
            prims.append(MONITOR)
            mats.append(facing(p, self.head, base.axis((1, 0, 0))).to_row_major(self.monitor.scale)
                        if self.monitor.on else HIDDEN)

    def toaster_writes(self, ordinal: int) -> None:
        ts = self.toaster
        if ts is None:
            return
        if ts.color_dirty:
            self.rt.write_floats(ts.toast_material, "inputs:diffuseColor", ts.toast_color, ordinal)
            if self.monitor.handle and self.monitor.on:
                c = np.asarray(ts.toast_color)
                self.rt.write_floats(self.monitor.swatch, "inputs:diffuseColor", c, ordinal)
                self.rt.write_floats(self.monitor.swatch, "inputs:emissiveColor", 0.5 * c, ordinal)
            ts.color_dirty = False
        if self.monitor.on and self.monitor.handle:  # handle 0: removed with the toaster
            for layer in self.monitor.layers:
                if layer.pending and layer.points is not None:
                    self.rt.write_points(layer.prim, layer.points, ordinal)
                    layer.pending = False
        if ts.glow_dirty:
            k = GLOW_EMISSION * min(max(ts.glow / ts.glow_max, 0.0), 1.0)
            self.rt.write_floats(ts.glow_shader, "inputs:emissiveColor", (k, 0.38 * k, 0.08 * k), ordinal)
            self.rt.write_floats(ts.glow_light, "inputs:intensity", (ts.glow,), ordinal)
            ts.glow_dirty = False

    # --- per-frame visuals -------------------------------------------------------------------------
    def write_visuals(self, frame: InputFrame, ordinal: int) -> None:
        """Everything that moves this frame, in one OVStage transform write: Newton's bodies (unless
        ovnewton publishes them), the rig, controllers, rays, marker, panels and toaster parts."""
        prims, mats = [RIG], [self.rig.to_row_major()]
        if not self.publish_ovnewton and self.physics.labels:
            prims += self.physics.labels
            mats += list(self.physics.body_matrices())
        for h in range(2):
            hi = frame.hands[h]
            prims += [HANDS[h], RAYS[h]]
            mats.append(hi.grip.to_row_major() if hi.active else HIDDEN)
            ray = self.hands[h].ray_length
            mats.append(hi.aim.to_row_major((1, 1, ray)) if hi.aim_valid and ray > 0 else HIDDEN)
        prims.append(MARKER)
        mats.append(Pose(self.place_at, axis_angle(Z, self.place_yaw)).to_row_major()
                    if self.pending_spawn and self.place_valid else HIDDEN)
        self.hud_transforms(prims, mats)
        self.toaster_transforms(prims, mats)
        b = self.browser
        if b.open:
            prims += [BROWSER, HIGHLIGHT]
            mats.append(b.pose.to_row_major())
            p2 = 2 * b.pad
            if b.hover_widget:
                r = b.close_rect if b.hover_widget == 1 else b.handle_rect
                at = b.pose * Pose(((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, 0))
                mats.append(at.to_row_major((r[2] - r[0] + p2, r[3] - r[1] + p2, 1)))
            elif b.hover >= 0:
                cx, cy = self.tile_center(b.hover)
                mats.append((b.pose * Pose((cx, cy, 0))).to_row_major((b.tile_w + p2, b.tile_h + p2, 1)))
            else:
                mats.append(HIDDEN)
        spinning = b.loading and b.placed
        prims += [LOADING, SPINNER]
        mats.append(b.pose.to_row_major() if spinning else HIDDEN)
        mats.append((b.pose * Pose((0, 0, 0), axis_angle(Z, -6.0 * self.now()))).to_row_major() if spinning else HIDDEN)
        self.rt.write_transforms(prims, np.asarray(mats), ordinal)
        self.toaster_writes(ordinal)
        for prim, attr, values in self.pending_floats:
            self.rt.write_floats(prim, attr, values, ordinal)
        self.pending_floats.clear()
        if self.hud.pending and self.hud.points is not None and len(self.hud.points) == HUD_FLOATS:
            self.rt.write_points(TEXT, self.hud.points, ordinal)
            self.hud.pending = False

    # --- info panel (port of vrsim/hud.py placement) ---------------------------------------------
    def hud_transforms(self, prims: list, mats: list) -> None:
        now = time.perf_counter()
        hud = self.hud
        visible = hud.points is not None and len(hud.points) > 0 and self.head is not None and now - hud.posted < hud.hold
        accent = {"ok": 1, "error": 2}.get(hud.kind, 0)
        panel = Pose()
        if visible:
            fwd = flat_forward(self.head)
            obstacles = self.hud_obstacles()
            if hud.pos is not None:
                to_panel = hud.pos - self.head.p
                to_panel[2] = 0
                if np.linalg.norm(to_panel) > 1e-3 and (to_panel / np.linalg.norm(to_panel)) @ fwd < math.cos(math.radians(70)):
                    hud.anchor = self.hud_target(fwd, obstacles)  # re-placed when it drifts ~70 deg out of view
                elif self.hud_overlap(hud.anchor, obstacles) > HUD_REPLACE_OVERLAP:
                    hud.anchor = self.hud_target(fwd, obstacles)  # or something now stands in it (monitor shown)
            else:
                hud.anchor = self.hud_target(fwd, obstacles)
            k = min(1.0, (now - self.hud_last) * 3.0)
            hud.pos = hud.anchor.copy() if hud.pos is None else hud.pos + (hud.anchor - hud.pos) * k
            panel = facing(hud.pos, self.head, -fwd)
            hud.pose = panel
        else:
            hud.pos = None
        hud.visible = visible
        self.hud_last = now
        prims.append(PANEL)
        mats.append(panel.to_row_major() if visible else HIDDEN)
        for i, path in enumerate(ACCENTS):
            prims.append(path)
            mats.append((panel * Pose((-0.36 + 0.018, 0, 0.003))).to_row_major() if visible and i == accent else HIDDEN)

    def hud_obstacles(self) -> list[tuple[np.ndarray, float]]:
        """What the info panel must not cut through, as world spheres: the toaster's bodies and,
        while shown, its FMU monitor."""
        ts = self.toaster
        if ts is None:
            return []
        out = []
        for b in range(len(ts.prop.body_paths)):
            pose = self.body_pose(ts.prop, b)
            if pose is not None:
                out.append(self.body_sphere(ts.prop, b, pose))
        base = self.body_pose(ts.prop, ts.base)
        if self.monitor.handle and self.monitor.on and base is not None:
            out.append(((base * Pose(self.monitor.offset)).p, 0.5 * math.hypot(*MONITOR_SIZE) * self.monitor.scale))
        return out

    def hud_overlap(self, pos, obstacles) -> float:
        """How far [m] the panel at ``pos`` (facing the head) reaches into the nearest obstacle (<= 0: clear)."""
        if not obstacles:
            return -1.0
        inv = facing(pos, self.head).inverse()
        worst = -1.0
        for centre, radius in obstacles:
            x, y, z = (inv * Pose(centre)).p
            gap = math.sqrt(max(abs(x) - HUD_HALF[0], 0.0) ** 2 + max(abs(y) - HUD_HALF[1], 0.0) ** 2 + z * z)
            worst = max(worst, radius + HUD_CLEARANCE - gap)
        return worst

    def hud_target(self, fwd, obstacles) -> np.ndarray:
        """Where the info panel goes: off to the left and a little low, unless that cuts through the
        toaster or its monitor; then the first clear spot of HUD_SPOTS (or the least blocked one)."""
        best, best_overlap = None, math.inf
        for yaw, drop in HUD_SPOTS:
            side = Pose((0, 0, 0), axis_angle(Z, math.radians(yaw))).axis(fwd)
            pos = self.head.p + side * HUD_DISTANCE - (0, 0, drop)
            overlap = self.hud_overlap(pos, obstacles)
            if overlap <= 0.0:
                return pos
            if overlap < best_overlap:
                best, best_overlap = pos, overlap
        return best

    def hud_hit(self, origin, direction):
        if not self.hud.visible:
            return None
        n = self.hud.pose.axis((0, 0, 1))
        denom = float(direction @ n)
        if denom > -1e-4:
            return None
        t = float((self.hud.pose.p - origin) @ n) / denom
        if t <= 0 or t > RAY_MAX:
            return None
        local = (self.hud.pose.inverse() * Pose(origin + direction * t)).p
        if abs(local[0]) > 0.36 or abs(local[1]) > 0.125:  # vrsim.hud PANEL_W/H
            return None
        return local, t

    def hud_in_front(self, hi: HandInput) -> bool:
        d = hi.aim.axis((0, 0, -1))
        hud = self.hud_hit(hi.aim.p, d)
        if hud is None:
            return False
        browser = self.browser_hit(hi.aim.p, d) if self.browser.open else None
        return browser is None or hud[1] < browser[1]

    def hud_interact(self, frame: InputFrame) -> None:
        if not self.hud.visible:
            return
        for h in range(2):
            hi = frame.hands[h]
            down = hi.trigger > (0.5 if self.hud.trigger_down[h] else 0.7)
            pressed = down and not self.hud.trigger_down[h]
            self.hud.trigger_down[h] = down
            if not hi.aim_valid or self.hands[h].prop is not None or not self.hud_in_front(hi):
                continue
            local, t = self.hud_hit(hi.aim.p, hi.aim.axis((0, 0, -1)))
            self.hands[h].ray_length = t
            half = 0.022 + 0.012  # CLOSE_HALF plus slack
            if pressed and abs(local[0] - (0.36 - 0.035)) <= half and abs(local[1] - (0.125 - 0.035)) <= half:
                self.hud.hold = 0  # hidden from the next frame on
                self.haptic(h, 0.4, 0.03)
                self.link.send({"t": "hud_closed"})  # the sidecar cancels a running download
            elif pressed and self.hud.nav:
                for delta, (cx, cy) in zip((-1, 1), HUD_NAV):
                    if abs(local[0] - cx) <= HUD_NAV_HALF + 0.012 and abs(local[1] - cy) <= HUD_NAV_HALF + 0.012:
                        self.haptic(h, 0.3, 0.03)
                        self.link.send({"t": "hud_page", "delta": delta})

    # --- asset browser ---------------------------------------------------------------------------
    def tile_center(self, index: int) -> tuple[float, float]:
        b = self.browser
        r, c = divmod(index, b.cols)
        return b.ox + c * (b.tile_w + b.gap) + b.tile_w / 2, b.oy - r * (b.tile_h + b.gap) - b.tile_h / 2

    def browser_hit(self, origin, direction):
        b = self.browser
        n = b.pose.axis((0, 0, 1))
        denom = float(direction @ n)
        if denom > -1e-4:
            return None
        t = float((b.pose.p - origin) @ n) / denom
        if t <= 0 or t > RAY_MAX:
            return None
        local = (b.pose.inverse() * Pose(origin + direction * t)).p
        half_w = -b.ox + 0.03
        grid_h = b.rows * b.tile_h + (b.rows - 1) * b.gap
        bottom = min(b.oy - grid_h - 0.03, b.handle_rect[1] - 0.02)
        if abs(local[0]) > half_w or local[1] < bottom or local[1] > b.oy + 0.12:
            return None
        return local, t

    def place_browser(self, msg: dict) -> None:
        """First appearance: standing on the table at the sidecar's anchor, else ahead of the user."""
        b = self.browser
        if b.placed or self.head is None:
            return
        anchor = msg.get("anchor") or []
        if len(anchor) == 3:
            handle_bottom = b.handle_rect[1] if b.handle_rect[1] != 0 else -0.597
            b.pose = facing((anchor[0], anchor[1], anchor[2] - (handle_bottom - 0.02)), self.head)
        else:
            b.pose = facing(self.head.p + flat_forward(self.head) * 1.2 - (0, 0, 0.1), self.head)
        b.placed = True

    def browser_grab(self, frame: InputFrame) -> None:
        b = self.browser
        if not b.open:
            return
        for h in range(2):
            hi = frame.hands[h]
            if b.moving_hand == h:
                if hi.squeeze < 0.35 or not hi.aim_valid:
                    b.moving_hand = -1
                continue
            if b.moving_hand >= 0 or self.hands[h].squeezing or hi.squeeze < 0.55 or not hi.aim_valid:
                continue
            hit = self.browser_hit(hi.aim.p, hi.aim.axis((0, 0, -1)))
            if hit and in_rect(hit[0], b.handle_rect):
                b.moving_hand = h
                b.move_offset = hi.aim.inverse() * b.pose
                self.hands[h].squeezing = True  # interact() then sees a held squeeze, not a new grab
                self.haptic(h, 0.5, 0.04)

    def tile_at(self, local) -> int:
        b = self.browser
        cx = (local[0] - b.ox) / (b.tile_w + b.gap)
        cy = (b.oy - local[1]) / (b.tile_h + b.gap)
        c, r = math.floor(cx), math.floor(cy)
        if c < 0 or r < 0 or c >= b.cols or r >= b.rows:
            return -1
        if (cx - c) * (b.tile_w + b.gap) > b.tile_w or (cy - r) * (b.tile_h + b.gap) > b.tile_h:
            return -1
        index = r * b.cols + c
        return index if index < b.count else -1

    def browser_interact(self, frame: InputFrame) -> None:
        b = self.browser
        for hi in frame.hands:
            if hi.thumb_click_pressed:
                self.link.send({"t": "browser_toggle"})
        if not b.open:
            return
        if b.moving_hand >= 0 and frame.hands[b.moving_hand].aim_valid:
            b.pose = facing((frame.hands[b.moving_hand].aim * b.move_offset).p, self.head)
        b.hover, b.hover_widget = -1, 0
        for h in (1, 0):  # right hand wins when both point at the panel
            hi = frame.hands[h]
            down = hi.trigger > (0.5 if b.trigger_down[h] else 0.7)
            pressed = down and not b.trigger_down[h]
            b.trigger_down[h] = down
            if not hi.aim_valid or self.hands[h].prop is not None or self.pending_spawn or self.hud_in_front(hi):
                continue
            hit = self.browser_hit(hi.aim.p, hi.aim.axis((0, 0, -1)))
            if hit is None:
                continue
            local, t = hit
            self.hands[h].ray_length = t
            if in_rect(local, b.close_rect):
                b.hover_widget = 1
                if pressed:
                    self.haptic(h, 0.4, 0.03)
                    self.close_browser()
                    self.link.send({"t": "browser_closed"})
                    return
                continue
            if in_rect(local, b.handle_rect) or b.moving_hand == h:
                b.hover_widget = b.hover_widget or 2
                continue
            tile = self.tile_at(local)
            if tile < 0:
                continue
            if b.hover < 0:
                b.hover = tile
            if pressed:
                self.link.send({"t": "browser_pick", "index": tile})
                self.haptic(h, 0.5, 0.04)
        if b.hover != b.reported_hover:
            b.reported_hover = b.hover
            self.link.send({"t": "browser_hover", "index": b.hover})
        for h in range(2):  # a thumbstick flick pages
            x = frame.hands[h].thumb_x
            if self.hands[h].prop is not None or self.pending_spawn:
                continue
            if b.stick_armed[h] and abs(x) > 0.7:
                self.link.send({"t": "browser_page", "delta": 1 if x > 0 else -1})
                b.stick_armed[h] = False
                self.haptic(h, 0.2, 0.02)
            elif abs(x) < 0.3:
                b.stick_armed[h] = True

    def show_browser(self, msg: dict) -> None:
        layer = Path(msg.get("layer", ""))
        if not layer.is_file():
            return
        b = self.browser
        if b.handle:
            self.rt.remove_references([b.handle])
        b.cols, b.rows, b.count = int(msg.get("cols", 4)), int(msg.get("rows", 3)), int(msg.get("count", 0))
        b.tile_w, b.tile_h = msg.get("tile", (0.26, 0.3))
        b.ox, b.oy = msg.get("grid_origin", (-0.565, 0.4875))
        b.gap, b.pad = float(msg.get("gap", 0.03)), float(msg.get("highlight_pad", 0.012))
        b.close_rect = tuple(msg.get("close", (0.495, 0.507, 0.565, 0.577)))
        b.handle_rect = tuple(msg.get("handle", (-0.22, -0.597, 0.22, -0.537)))
        b.loading = False
        b.handle, _ = self.rt.add_reference(layer.read_text(encoding="utf-8"), BROWSER)
        self.place_browser(msg)  # no-op once placed
        b.open = True
        b.hover = b.reported_hover = -1
        log.info("Browser page %d/%d (%d assets)", msg.get("page", 0) + 1, msg.get("pages", 1), b.count)

    def close_browser(self) -> None:
        if self.browser.handle:
            self.rt.remove_references([self.browser.handle])
        self.browser = Browser()

    # --- sidecar ---------------------------------------------------------------------------------
    def handle_sidecar(self) -> None:
        for msg in self.link.poll():
            t = msg.get("t")
            try:
                if t == "spawn":
                    self.spawn(msg)
                elif t == "undo":
                    self.cancel_placement() if self.pending_spawn else self.undo()
                elif t == "clear":
                    self.clear()
                elif t == "fix":
                    self.fix_command(bool(msg.get("on", True)))
                elif t == "toaster":
                    self.on_toaster(msg)
                elif t == "toaster_monitor_frame":
                    self.monitor_frame(msg)
                elif t == "toaster_monitor":
                    self.set_monitor(bool(msg.get("on", not self.monitor.on)))
                elif t == "remove_device":
                    self.remove_device(str(msg.get("device", "")))
                elif t == "realign":
                    self.realign()
                elif t == "browser":
                    self.show_browser(msg)
                elif t == "browser_close":
                    self.close_browser()
                elif t == "browser_loading":
                    self.place_browser(msg)
                    self.browser.loading = True
                elif t == "hud":
                    self.hud.points = decode_floats(msg.get("points", ""))
                    self.hud.kind = msg.get("kind", "info")
                    self.hud.hold = float(msg.get("hold", 6.0))
                    self.hud.nav = bool(msg.get("nav", False))
                    self.hud.posted = time.perf_counter()
                    self.hud.pending = True
            except Exception:
                log.exception("Sidecar message %r failed", t)

    def spawn_pose(self, msg: dict, i: int, side_sign: float) -> Pose:
        """Small props drop onto the desk ahead, off to one side; larger ones onto the floor."""
        at = msg.get("position") or []
        if len(at) == 3:  # a scene object made grabbable, or the toaster: exactly where it stood
            return Pose((at[0], at[1], at[2] + 0.002 + i * 0.2), axis_angle(Z, math.radians(float(msg.get("yaw", 0.0)))))
        size = msg.get("size", (0.2, 0.2, 0.2))
        extent = max(size)
        fwd = flat_forward(self.head)
        head = self.head.p if self.head is not None else np.array([0.0, 0.0, 1.6])
        right = np.cross(fwd, Z)
        spacing = max(0.25, max(size[0], size[1]) * 1.2)
        if extent <= DESK_PROP_MAX:
            p = head + fwd * (0.6 + 0.5 * extent) + right * (side_sign * (0.45 + i * spacing))
            p[2] = max(head[2] - 0.15, 0.4)
        else:
            p = head + right * (side_sign * (0.55 + 0.5 * extent)) + fwd * (0.15 - i * spacing)
            p[2] = max(head[2] - 0.7, 0.5)
        p[2] += i * 0.05 - float(msg.get("bottom_z", 0.0))
        return Pose(p, axis_angle(Z, math.atan2(fwd[1], fwd[0])))

    def spawn(self, msg: dict) -> None:
        count = min(max(int(msg.get("count", 1)), 1), 10)
        correction = np.asarray(msg.get("correction", []), dtype=float)
        if correction.size != 16:
            return
        correction = correction.reshape(4, 4)
        size = msg.get("size", (0.2, 0.2, 0.2))
        asset_ref = msg.get("asset_ref") or f"@{msg.get('root_usd', '')}@"
        side_sign = 1.0 if self.next_spawn_right else -1.0
        at_toast = bool(msg.get("at_toast")) and self.toaster is not None
        catalog_spawn = not msg.get("position") and not at_toast
        at_marker = bool(msg.get("at_marker"))
        if catalog_spawn and not at_marker and self.hulls:
            self.pending_spawn = msg
            fwd = self.head.axis((0, 0, -1)) if self.head is not None else np.array([0.0, 1.0, 0.0])
            self.place_yaw, self.place_valid = math.atan2(fwd[1], fwd[0]), False
            self.link.send({"t": "placing", "name": msg.get("name", "prop"), "count": count})
            return
        if catalog_spawn and not at_marker:
            self.next_spawn_right = not self.next_spawn_right
        spacing = max(0.25, max(size[0], size[1]) * 1.2)
        new, ordinal = [], None
        for i in range(count):
            if len(self.props) >= MAX_PROPS:
                self.remove_props([self.oldest_removable()], rebuild=False)
            pose = self.spawn_pose(msg, i, side_sign)
            if at_toast:  # the sidecar's copy of the landed toast: exactly where the FMU toast lies
                base = self.body_pose(self.toaster.prop, self.toaster.base)
                x, y, z, tumble = self.toaster.toast_local
                pose = base * Pose((x, y, z + 0.002), axis_angle((1, 0, 0), math.radians(tumble)))
            if at_marker:  # standing on the marked spot, 2 mm up, facing the marker's arrow
                pose = Pose(self.place_at + self.place_side * ((i - (count - 1) / 2.0) * spacing)
                            + (0, 0, 0.002 - float(msg.get("bottom_z", 0.0))), axis_angle(Z, self.place_yaw))
            m = correction @ np.asarray(pose.to_row_major()).reshape(4, 4)
            rows = ", ".join("(" + ", ".join(f"{v:.9g}" for v in r) + ")" for r in m)
            layer = ('#usda 1.0\n(\n    defaultPrim = "Prop"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n'
                     f'def Xform "Prop"\n{{\n    matrix4d xformOp:transform = ({rows})\n'
                     '    uniform token[] xformOpOrder = ["xformOp:transform"]\n'
                     f'    def "Asset" (prepend references = {asset_ref}) {{}}\n}}\n')
            root = f"/World/Props/prop_{self.next_prop:04d}"
            self.next_prop += 1
            handle, ordinal = self.rt.add_reference(layer, root)
            spheres = [s for s in msg.get("body_spheres", []) if len(s) == 4]
            bodies = [f"{root}/{rel}" for rel in msg.get("body_paths", [])]
            prop = Prop(root, msg.get("name", "prop"), handle, bodies,
                        spheres if len(spheres) == len(bodies) else [], list(msg.get("body_movable", [])),
                        fix_on_spawn=at_marker or bool(msg.get("fixed")),
                        radius=0.5 * float(np.linalg.norm(size)), device=msg.get("device", ""))
            self.props.append(prop)
            new.append(prop)
            tint = msg.get("tint")
            if isinstance(tint, dict) and at_toast:  # the copy keeps the toast's browning colour
                self.pending_floats.append((f"{root}/{tint['prim']}", tint.get("attr", "inputs:diffuseColor"),
                                            tuple(self.toaster.toast_color)))
        self.physics.rebuild(ordinal)
        missing = [p for prop in new for p in prop.body_paths if p not in self.physics.index]
        if missing:
            log.warning("Bodies not in the Newton model: %s", missing[:4])
        for prop in new:
            if prop.device == "toaster":
                self.setup_toaster(prop, msg)
        log.info("Spawned %d x %s", count, msg.get("name", "prop"))
        self.link.send({"t": "spawned", "name": msg.get("name", "prop"), "count": count, "total": len(self.props)})
        self.haptic(1, 0.4, 0.06)

    def remove_props(self, props: list[Prop], rebuild: bool = True) -> None:
        if not props:
            return
        for h in self.hands:
            if h.prop in props:
                h.drop()
        if self.last_prop in props:
            self.last_prop = None
        ordinal = self.rt.remove_references([p.handle for p in props])
        self.props = [p for p in self.props if p not in props]
        if self.toaster and self.toaster.prop in props:
            self.toaster = None
        if rebuild and ordinal is not None:
            self.physics.rebuild(ordinal)

    def remove_device(self, device: str) -> None:
        """Remove a device prop (the toaster: "hide the toaster"), which undo / clear keep, and its monitor."""
        doomed = [p for p in self.props if p.device == device]
        if device == "toaster" and self.monitor.handle:
            self.rt.remove_references([self.monitor.handle])  # placed only with the toaster: would freeze in the air
            self.monitor.handle = 0  # setup_monitor adds it again with the next toaster
        self.remove_props(doomed)
        log.info("Removed device %s (%d props)", device, len(doomed))

    def oldest_removable(self) -> Prop:
        return next((p for p in self.props if not p.device), self.props[0])

    def newest_removable(self) -> Prop | None:
        return next((p for p in reversed(self.props) if not p.device), None)

    def undo(self) -> None:
        prop = self.newest_removable()
        if prop is None:
            self.link.send({"t": "nothing_to_undo"})
            return
        self.remove_props([prop])
        self.link.send({"t": "removed", "name": prop.name, "total": len(self.props)})

    def clear(self) -> None:
        doomed = [p for p in self.props if not p.device]
        self.remove_props(doomed)
        self.link.send({"t": "cleared", "n": len(doomed)})

    # --- teardown --------------------------------------------------------------------------------
    def shutdown(self) -> None:
        log.info("Shutting down...")
        self.link.close()
        if self.xr:
            try:
                self.xr.destroy()
            except Exception as exc:
                log.warning("XR input teardown: %s", exc)
        try:
            self.ctx.destroy(stage_ordinal=self.rt.current())
        except Exception as exc:
            log.warning("ovxr destroy: %s", exc)
        self.physics.destroy()
        self.rt.destroy()


def ensure_openxr_runtime() -> None:
    """The Oculus -> "Meta Horizon" update can leave the machine-wide ActiveRuntime pointing at a
    deleted manifest (XR_ERROR_RUNTIME_UNAVAILABLE); point this process at an existing one."""
    if sys.platform != "win32" or os.environ.get("XR_RUNTIME_JSON"):
        return
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\OpenXR\1") as key:
            registered = os.path.expandvars(winreg.QueryValueEx(key, "ActiveRuntime")[0])
    except OSError:
        registered = ""
    if registered and os.path.exists(registered):
        return
    for candidate in (r"C:\Program Files\Meta Horizon\Support\oculus-runtime\oculus_openxr_64.json",
                      r"C:\Program Files\Oculus\Support\oculus-runtime\oculus_openxr_64.json"):
        if os.path.exists(candidate):
            os.environ["XR_RUNTIME_JSON"] = candidate
            log.warning("Registered OpenXR runtime %r is missing; using %s for this process", registered, candidate)
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="VR SimReady, Newton edition (ovxr + ovrtx + ovstage + ovnewton)")
    parser.add_argument("--render", choices=("rt2", "minimal"), default="rt2")
    parser.add_argument("--env", help="render-stage environment USD (default: the sidecar's environment_newton*.usda)")
    parser.add_argument("--frames-in-flight", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--no-foveation", dest="foveation", action="store_false")
    parser.add_argument("--device", default="cuda:0", help="Warp device for Newton")
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--port", type=int, default=SIDECAR_PORT)
    parser.add_argument("--publish", choices=("app", "ovnewton"), default="app",
                        help="who writes Newton body poses into the stage: the app's per-frame transform batch "
                             "(default, ~0.1 ms) or ovnewton's update_to_ovstage (~3 ms, also velocities and joint state)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")
    ensure_openxr_runtime()
    app = App(args)
    signal.signal(signal.SIGINT, lambda *_: setattr(app, "running", False))
    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
