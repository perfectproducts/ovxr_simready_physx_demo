"""VR SimReady Physics: say "add a mug", grab it, throw it.

Per frame (ovxr split loop, application-owned stage ordinals):

    begin_frame -> sync controllers at predicted display time -> hand interaction -> Newton step
    -> update_prim_transforms(bodies + hands) at ordinal N -> bind_frame(N) -> submit_frame

Spawns are searched/downloaded/prepared on a worker thread and applied between frames.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import ovxr

from .assets import NotDynamicError, PreparedAsset, prepare_asset
from .catalog import SimReadyCatalog
from .config import Config
from .hud import TEXT_PATH, Feedback, Hud, download_progress
from .mathutil import Pose, quat_from_axis_angle
from .physics import GrabTarget, PhysicsWorld, PropInstance
from .scene import HAND_PATHS, RAY_PATHS, RIG_PATH, SceneStage
from .voice import Command, VoiceListener, start_console
from .xr_input import LEFT, RIGHT, InputFrame, XrInput

log = logging.getLogger("vrsim")

HIDDEN = Pose((0.0, 0.0, -100.0)).to_row_major(1e-4)  # parks visuals far below the floor
GRAB_ON, GRAB_OFF = 0.55, 0.35  # squeeze hysteresis


_RUNTIME_CANDIDATES = (
    r"C:\Program Files\Meta Horizon\Support\oculus-runtime\oculus_openxr_64.json",
    r"C:\Program Files\Oculus\Support\oculus-runtime\oculus_openxr_64.json",
)


def _ensure_openxr_runtime() -> None:
    """Point this process at an existing OpenXR runtime manifest if the registered one is gone.

    The Oculus -> "Meta Horizon" app update moves the runtime but can leave the machine-wide
    ActiveRuntime registry value pointing at the old path (XR_ERROR_RUNTIME_UNAVAILABLE). Fix it
    process-locally via XR_RUNTIME_JSON rather than touching machine-wide settings.
    """
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
    for candidate in _RUNTIME_CANDIDATES:
        if os.path.exists(candidate):
            os.environ["XR_RUNTIME_JSON"] = candidate
            log.warning("Registered OpenXR runtime %r is missing; using %s for this process. "
                        "Fix it permanently in the Meta Horizon app: Settings > General > OpenXR Runtime.",
                        registered or "(none)", candidate)
            return


# --- spawning ------------------------------------------------------------------------------
class Spawner:
    """Turns voice commands into prepared assets on a worker thread."""

    def __init__(self, config: Config, catalog: SimReadyCatalog, feedback: Feedback):
        self.config = config
        self.catalog = catalog
        self.feedback = feedback
        self.requests: "queue.Queue[Command]" = queue.Queue()
        self.ready: "queue.Queue[tuple[PreparedAsset, int] | str]" = queue.Queue()
        self._cache: dict[str, PreparedAsset] = {}
        threading.Thread(target=self._run, daemon=True, name="spawner").start()

    def _run(self) -> None:
        while True:
            command = self.requests.get()
            try:
                match = self.catalog.best_match(command.query)
                if not match:
                    alternatives = ", ".join(e.name for e, _ in self.catalog.search(command.query, 3))
                    log.warning("No SimReady asset matches '%s' (closest: %s)", command.query, alternatives)
                    self.feedback.post("error", f"No SimReady asset for '{command.query}'",
                                       f"Closest: {alternatives}" if alternatives else "")
                    self.ready.put("miss")
                    continue
                entry, score = match
                log.info("'%s' -> %s (score %.0f, %s)", command.query, entry.name, score, entry.repo_id)
                asset = self._cache.get(entry.key)
                if asset is None:
                    self.feedback.post("thinking", f"Found: {entry.name}", f"Downloading from {entry.repo_id}…")
                    root = self.catalog.fetch(entry, download_progress(self.feedback, entry.name))
                    asset = prepare_asset(entry, root, self.config.generated_dir)
                    self._cache[entry.key] = asset
                self.ready.put((asset, command.count))
            except NotDynamicError:
                log.warning("'%s' has no rigid bodies", command.query)
                self.feedback.post("error", f"Can't pick up '{command.query}'",
                                   "That asset is static (no rigid body). Try something smaller")
                self.ready.put("miss")
            except Exception as exc:
                log.exception("Spawning '%s' failed", command.query)
                self.feedback.post("error", f"Couldn't load '{command.query}'", str(exc))
                self.ready.put("miss")


# --- hand interaction ----------------------------------------------------------------------
class Hand:
    def __init__(self, index: int):
        self.index = index
        self.body: int | None = None
        self.offset = Pose()  # body pose in the hand frame
        self.pulling = False  # true while a ray-grabbed body is flying to the hand
        self.squeezing = False
        self.history: collections.deque = collections.deque(maxlen=6)  # (time, position)
        self.ray_length = 0.0

    def velocity(self) -> np.ndarray:
        if len(self.history) < 2:
            return np.zeros(3)
        (t0, p0), (t1, p1) = self.history[0], self.history[-1]
        return (p1 - p0) / max(t1 - t0, 1e-3)


class Interaction:
    def __init__(self, config: Config, physics: PhysicsWorld):
        self.config = config
        self.physics = physics
        self.hands = [Hand(LEFT), Hand(RIGHT)]
        self.haptics: list[tuple[int, float, float]] = []  # (hand, amplitude, seconds)

    def release_all(self) -> None:
        for hand in self.hands:
            hand.body = None

    def update(self, frame: InputFrame, now: float) -> list[GrabTarget]:
        grabs = []
        for hand, inp in zip(self.hands, frame.hands):
            if not inp.active:
                hand.ray_length = 0.0
                continue
            grip = inp.grip
            hand.history.append((now, grip.p.copy()))

            was = hand.squeezing
            hand.squeezing = inp.squeeze > (GRAB_OFF if was else GRAB_ON)
            if hand.squeezing and not was:
                self._try_grab(hand, inp)
            elif was and not hand.squeezing and hand.body is not None:
                self._throw(hand)

            if hand.body is not None:
                if hand.pulling:  # reel the offset in so the body flies into the palm
                    hand.offset = Pose(hand.offset.p * 0.85, hand.offset.q)
                    hand.pulling = np.linalg.norm(hand.offset.p) > 0.03
                grabs.append(GrabTarget(hand.body, grip * hand.offset))
                hand.ray_length = 0.0
            elif inp.aim is not None and inp.trigger > 0.1:
                d = inp.aim.axis((0, 0, -1))
                hit = self.physics.pick_ray(inp.aim.p, d, self.config.ray_max_distance)
                hand.ray_length = hit[1] if hit else self.physics.ray_floor_distance(
                    inp.aim.p, d, self.config.ray_max_distance)
            else:
                hand.ray_length = 0.0
        return grabs

    def _try_grab(self, hand: Hand, inp) -> None:
        body = self.physics.pick_near(inp.grip.p, self.config.grab_radius)
        hand.pulling = False
        if body is None and inp.aim is not None:  # force-pull whatever the hand points at
            hit = self.physics.pick_ray(inp.aim.p, inp.aim.axis((0, 0, -1)), self.config.ray_max_distance)
            if hit:
                body, hand.pulling = hit[0], True
        if body is None:
            return
        for other in self.hands:  # steal from the other hand
            if other is not hand and other.body == body:
                other.body = None
        hand.body = body
        hand.offset = inp.grip.inverse() * self.physics.body_pose(body)
        self.haptics.append((hand.index, 0.6, 0.04))

    def _throw(self, hand: Hand) -> None:
        if not hand.pulling:
            self.physics.set_linear_velocity(hand.body, hand.velocity() * self.config.throw_scale)
        hand.body = None
        hand.pulling = False
        self.haptics.append((hand.index, 0.3, 0.02))

    def forget_bodies(self) -> None:
        """Body indices change when the physics model is rebuilt."""
        self.release_all()


# --- app -----------------------------------------------------------------------------------
class App:
    def __init__(self, config: Config, refresh_catalog: bool, console: bool):
        self.config = config
        self.running = True
        self.commands: "queue.Queue[Command]" = queue.Queue()

        self.feedback = Feedback()
        self.hud = Hud(self.feedback)
        self.catalog = SimReadyCatalog(config).load(refresh=refresh_catalog)
        self.spawner = Spawner(config, self.catalog, self.feedback)
        self.physics = PhysicsWorld(config)
        self.interaction = Interaction(config, self.physics)
        self.scene = SceneStage(config)
        self.voice = VoiceListener(config, self.commands, self.feedback)
        if console:
            start_console(self.commands)
            log.info("Console: type commands like 'add a mug', 'spawn 3 footballs', 'undo', 'clear all'")

        self.ctx = ovxr.Context(ovxr.ContextDesc(
            ovrtx_renderer=self.scene.renderer,
            ovstage=self.scene.stage,
            application_name=config.application_name,
            display_mode=ovxr.DisplayMode.STEREO,
            session_mode=ovxr.SessionMode.VR,
            local_foveation_mode=ovxr.LocalFoveationMode.WARPED if config.foveation else ovxr.LocalFoveationMode.DISABLED,
            runtime_foveated_submission_policy=ovxr.FeaturePolicy.AUTO if config.foveation else ovxr.FeaturePolicy.DISABLED,
            foveation_unwarp=config.foveation,
            graphics_api=ovxr.GraphicsApi.VULKAN,
            create_render_products=True,
            tracking_anchor_path=RIG_PATH,
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
            log.warning("OpenXR probe: %s (%s). Is Quest Link / Air Link connected?",
                        ovxr.result_name(probe.result), self.ctx.last_error)
        self.xr: XrInput | None = None
        self.head: Pose | None = None
        self.spawn_order: list[PropInstance] = []
        self.pending_spawns = 0

    # --- commands --------------------------------------------------------------------------
    def _handle_commands(self) -> None:
        while not self.commands.empty():
            command = self.commands.get()
            log.info("Command: %s", command)
            heard = f"“{command.text}”" if command.text else None
            if command.kind == "spawn":
                self.pending_spawns += 1
                self.spawner.requests.put(command)
                self.feedback.post("thinking", heard or f"Add {command.count} × {command.query}",
                                   f"Searching SimReady for '{command.query}'…")
            elif command.kind == "undo":
                if self.spawn_order:
                    last = self.spawn_order[-1]
                    self._remove([last])
                    self.feedback.post("ok", f"Removed {last.asset.name}", self._scene_summary())
                else:
                    self.feedback.post("info", "Nothing to undo", "Say “add a mug” to spawn something")
            elif command.kind == "query":
                self._answer_query(command.query)
            elif command.kind == "clear":
                count = len(self.spawn_order)
                self._remove(list(self.spawn_order))
                self.feedback.post("ok" if count else "info",
                                   f"Cleared {count} object{'s' if count != 1 else ''}" if count else "Nothing to clear")

        while not self.spawner.ready.empty():
            item = self.spawner.ready.get()
            self.pending_spawns = max(0, self.pending_spawns - 1)
            if item == "miss":
                self._buzz_error()
                continue
            asset, count = item
            for i in range(count):
                self._spawn(asset, i, count)
            self.feedback.post("ok", f"Added {count} × {asset.name}" if count > 1 else f"Added {asset.name}",
                               f"{self._scene_summary()} · grip to grab, release to throw")

    def _answer_query(self, query: str) -> None:
        names = self.catalog.find(query)
        shown = ", ".join(names[:8]) + (f" +{len(names) - 8} more" if len(names) > 8 else "")
        log.info("Query '%s': %d match(es): %s", query, len(names), ", ".join(names))
        if not query:
            self.feedback.post("info", f"{len(names)} SimReady assets available",
                               f"e.g. {shown}", hold=12.0)
        elif names:
            self.feedback.post("info", f"{len(names)} match{'es' if len(names) != 1 else ''} for '{query}'",
                               f"{shown} · say “add a …”", hold=12.0)
        else:
            closest = ", ".join(e.name for e, _ in self.catalog.search(query, 3))
            self.feedback.post("error", f"No SimReady assets for '{query}'",
                               f"Closest: {closest}" if closest else "")

    def _scene_summary(self) -> str:
        n = len(self.spawn_order)
        return f"{n} object{'s' if n != 1 else ''} in the scene"

    def _spawn_pose(self, asset: PreparedAsset, i: int, count: int) -> Pose:
        """In front of the user's head, dropped from a little above eye level, fanned out sideways."""
        if self.head is not None:
            fwd = self.head.axis((0, 0, -1))
            fwd[2] = 0.0
            fwd = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 1e-3 else np.array([0.0, 1.0, 0.0])
            base = self.head.p + fwd * (0.6 + 0.5 * float(np.max(asset.size)))
            z = max(self.head.p[2] - 0.15, 0.4)
        else:
            fwd, base, z = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.7, 0.0]), 1.3
        side = np.cross(fwd, (0.0, 0.0, 1.0))
        spread = (i - (count - 1) / 2) * max(0.25, float(np.max(asset.size[:2])) * 1.2)
        p = base + side * spread
        p[2] = z + i * 0.05 - asset.bottom_z
        yaw = np.arctan2(fwd[1], fwd[0]) + np.random.uniform(-0.4, 0.4)
        return Pose(p, quat_from_axis_angle((0, 0, 1), yaw))

    def _spawn(self, asset: PreparedAsset, i: int, count: int) -> None:
        if len(self.spawn_order) >= self.config.max_props:
            self._remove([self.spawn_order[0]])
        self.interaction.forget_bodies()
        prop = self.physics.add_prop(asset, self._spawn_pose(asset, i, count))
        prop.usd_handle = self.scene.add_reference(asset.wrapper_path, prop.prim_root)
        self.spawn_order.append(prop)
        log.info("Spawned %s at %s", asset.name, np.round(prop.spawn.p, 2).tolist())
        if self.xr:
            self.xr.haptic(RIGHT, 0.4, 0.06)

    def _remove(self, props: list[PropInstance]) -> None:
        if not props:
            return
        self.interaction.forget_bodies()
        self.physics.remove_props(props)
        self.scene.remove_references([p.usd_handle for p in props if p.usd_handle is not None])
        doomed = {p.uid for p in props}
        self.spawn_order = [p for p in self.spawn_order if p.uid not in doomed]
        log.info("Removed %d prop(s); %d left", len(props), len(self.spawn_order))

    def _buzz_error(self) -> None:
        if self.xr:
            for hand in (LEFT, RIGHT):
                self.xr.haptic(hand, 0.8, 0.15, frequency=80.0)

    # --- per frame -------------------------------------------------------------------------
    def _visual_transforms(self, frame: InputFrame | None) -> dict[str, list[float]]:
        out = {}
        for h in (LEFT, RIGHT):
            inp = frame.hands[h] if frame else None
            hand = self.interaction.hands[h]
            out[HAND_PATHS[h]] = inp.grip.to_row_major() if inp and inp.active else HIDDEN
            if inp and inp.aim is not None and hand.ray_length > 0:
                out[RAY_PATHS[h]] = inp.aim.to_row_major((1.0, 1.0, hand.ray_length))
            else:
                out[RAY_PATHS[h]] = HIDDEN
        return out

    def _buttons(self, frame: InputFrame) -> None:
        hands = frame.hands
        if any(h.primary_pressed for h in hands):
            self.voice.start()
            if self.xr:
                self.xr.haptic(RIGHT if hands[RIGHT].primary_pressed else LEFT, 0.25, 0.03)
        if any(h.primary_released for h in hands) and not any(h.primary for h in hands):
            self.voice.stop()
        if hands[RIGHT].secondary_pressed:
            self.commands.put(Command("undo"))
        if hands[LEFT].secondary_pressed:
            self.commands.put(Command("clear"))

    def run(self) -> None:
        waiting_since = None
        last = last_status = time.perf_counter()
        frames = 0
        while self.running:
            result, plan = self.ctx.begin_frame()
            if result != ovxr.Result.SUCCESS:
                # NOT_READY / TIMEOUT: no headset yet (Link not started, headset asleep or not worn).
                now = time.perf_counter()
                if waiting_since is None or now - waiting_since > 10.0:
                    waiting_since = now
                    log.info("Waiting for headset (%s: %s)", ovxr.result_name(result), self.ctx.last_error)
                self._handle_commands()  # keep downloads/console flowing while waiting
                time.sleep(0.05)
                continue
            waiting_since = None

            try:
                timing = plan.timing
                now = time.perf_counter()
                # Wall-clock frame time drives physics; the runtime's suggested delta was far too
                # small on Quest Link (sim ran ~4x slow).
                dt = min(now - last, 0.1)
                last = now

                if self.xr is None:
                    handles = self.ctx.get_openxr_handles()
                    if handles is not None:
                        self.xr = XrInput(self.ctx, handles)
                        self.feedback.post("info", "Hold A or X and speak",
                                           "“add a mug” · “spawn three footballs” · “undo” · “clear all”", hold=12.0)
                frame = self.xr.sync(timing.predicted_display_time_ns) if self.xr else None
                if frame is not None:
                    self.head = frame.head or self.head
                    self._buttons(frame)
                    grabs = self.interaction.update(frame, now)
                    for hand, amp, dur in self.interaction.haptics:
                        self.xr.haptic(hand, amp, dur)
                    self.interaction.haptics.clear()
                else:
                    grabs = []

                self.physics.step(dt, grabs)

                # Every ovxr call that takes an ordinal also seals it, so a frame uses two: one for
                # the scene writes (our ovstage HUD write + ovxr's transforms, published together
                # by its seal), then a fresh one for bind_frame's camera state.
                ordinal = self.scene.clock.current()
                transforms = self.physics.prim_transforms()
                transforms.update(self._visual_transforms(frame))
                hud_transforms, hud_points = self.hud.update(self.head, HIDDEN)
                transforms.update(hud_transforms)
                if hud_points is not None:  # message changed: rewrite the text mesh at this ordinal
                    self.scene.write_points(TEXT_PATH, hud_points, ordinal)
                try:
                    self.ctx.update_prim_transforms(transforms, stage_ordinal=ordinal)
                except ovxr.OvxrError as exc:
                    log.warning("update_prim_transforms: %s", exc)
                    if hud_points is not None:
                        self.hud.invalidate()  # rewrite the text next frame
                finally:
                    self.scene.clock.resolve()  # sealed on success; abandoned (never reused) on failure

                ordinal = self.scene.clock.current()
                bind_result, sealed = self.ctx.bind_frame(plan, dt, ordinal)
                if sealed:
                    self.scene.clock.resolve()
                if bind_result != ovxr.Result.SUCCESS:
                    self.ctx.cancel_frame(plan)
                    plan = None
                    if bind_result != ovxr.Result.NOT_READY:
                        log.warning("bind_frame: %s (%s)", ovxr.result_name(bind_result), self.ctx.last_error)
                    continue
                submit_result, _ = self.ctx.submit_frame(plan)
                plan = None
                if submit_result != ovxr.Result.SUCCESS:
                    log.warning("submit_frame: %s (%s)", ovxr.result_name(submit_result), self.ctx.last_error)
            finally:
                if plan is not None:  # an exception left the plan live
                    self.ctx.cancel_frame(plan)

            # Population changes (spawn/remove) happen between frames, at their own ordinals.
            self._handle_commands()
            frames += 1
            if now - last_status > 2.0:
                last_status = now
                self._log_status(frame, dt, timing)
            if frames % 5400 == 0:
                log.info("frame %d | %d props | sim %.1fs", frames, len(self.spawn_order), self.physics.sim_time)

    def _log_status(self, frame: InputFrame | None, dt: float, timing) -> None:
        """Periodic diagnostics: tracking, controller and physics state as the app sees it."""
        def fmt(pose):
            return np.round(pose.p, 2).tolist() if pose is not None else None

        flags = self.xr.last_flags if self.xr else None
        hands = [(h.active, fmt(h.grip)) for h in frame.hands] if frame else None
        body = None
        if self.spawn_order:
            body = fmt(self.physics.body_pose(self.spawn_order[-1].body_start))
        log.info("status | dt %.4f (suggested %.4f) | focused %s | head %s | hands %s | locate flags %s | "
                 "last prop %s | sim %.1fs", dt, timing.suggested_delta_time_seconds,
                 frame.focused if frame else None, fmt(self.head), hands, flags, body, self.physics.sim_time)

    def shutdown(self) -> None:
        log.info("Shutting down...")
        self.voice.close()
        if self.xr:
            try:
                self.xr.destroy()
            except Exception as exc:
                log.warning("XR input teardown: %s", exc)
        try:
            self.ctx.destroy(stage_ordinal=self.scene.clock.current())
        except ovxr.OvxrError as exc:
            log.warning("ovxr destroy: %s", exc)
        self.scene.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="VR SimReady Physics (ovxr + Newton + Wit.ai)")
    parser.add_argument("--refresh-catalog", action="store_true", help="Rebuild the SimReady index")
    parser.add_argument("--wit-token", help="Wit.ai client access token (or set WIT_AI_TOKEN)")
    parser.add_argument("--mic", help="Input device name or index (see --list-mics)")
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument("--no-console", action="store_true", help="Disable typed commands on stdin")
    parser.add_argument("--no-foveation", action="store_true")
    parser.add_argument("--throw-scale", type=float)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):  # HUD messages use “quotes” and …; Windows consoles default to cp1252
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx2", "httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_mics:
        import sounddevice as sd
        print(sd.query_devices())
        return

    config = Config()
    if args.wit_token:
        config.wit_token = args.wit_token
    if args.mic:
        config.mic_device = int(args.mic) if args.mic.isdigit() else args.mic
    if args.no_foveation:
        config.foveation = False
    if args.throw_scale:
        config.throw_scale = args.throw_scale

    _ensure_openxr_runtime()
    app = App(config, refresh_catalog=args.refresh_catalog, console=not args.no_console)
    signal.signal(signal.SIGINT, lambda *_: setattr(app, "running", False))
    try:
        app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
