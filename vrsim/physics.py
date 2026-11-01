"""Newton rigid-body world with spawnable props and velocity-driven grabbing.

Newton models are immutable once finalized, so spawning or removing a prop rebuilds the model and
carries every surviving body's pose and velocity across. Spawns are rare (voice commands), so a
short rebuild hitch is acceptable.

Grabbing drives a body's velocity toward the hand target every substep instead of teleporting it:
contacts stay physical, distant "force pull" grabs fly to the hand, and releasing leaves the body
with the hand's velocity, which is what makes throwing feel right.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import warp as wp
import newton

from .assets import PreparedAsset
from .config import Config
from .mathutil import Pose

log = logging.getLogger(__name__)


@dataclass
class PropInstance:
    uid: int
    asset: PreparedAsset
    prim_root: str  # e.g. "/World/Props/prop_0003"
    spawn: Pose
    body_start: int = -1  # first global body index in the current model
    usd_handle: int | None = None  # renderer-side reference handle

    @property
    def body_indices(self) -> range:
        return range(self.body_start, self.body_start + len(self.asset.bodies))


@dataclass
class GrabTarget:
    body: int
    target: Pose


@wp.kernel
def _drive_bodies(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    ids: wp.array(dtype=wp.int32),
    targets: wp.array(dtype=wp.transform),
    inv_dt: float,
    max_lin: float,
    max_ang: float,
):
    t = wp.tid()
    b = ids[t]
    if b < 0:  # empty hand slot
        return
    x = body_q[b]
    goal = targets[t]
    p = wp.transform_get_translation(x)
    q = wp.transform_get_rotation(x)

    v = (wp.transform_get_translation(goal) - p) * inv_dt
    dq = wp.transform_get_rotation(goal) * wp.quat_inverse(q)
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])
    axis = wp.vec3(dq[0], dq[1], dq[2])
    s = wp.length(axis)
    w = wp.vec3(0.0)
    if s > 1.0e-7:
        w = axis * (2.0 * wp.atan2(s, dq[3]) / s) * inv_dt

    if wp.length(v) > max_lin:
        v = wp.normalize(v) * max_lin
    if wp.length(w) > max_ang:
        w = wp.normalize(w) * max_ang

    # body_qd's linear part is the center-of-mass velocity; v above is the body-origin velocity.
    r = wp.transform_vector(x, body_com[b])
    body_qd[b] = wp.spatial_vector(v + wp.cross(w, r), w)


MAX_GRABS = 2  # one per hand


class PhysicsWorld:
    """Fixed-timestep simulation; on CUDA one frame's substeps replay as a captured CUDA graph.

    Launching ~20 small kernels per substep from Python cost ~34 ms per frame for a single prop,
    far over the Quest's 13.9 ms budget. The graph reads grab targets from fixed device buffers,
    and is re-captured whenever the model is rebuilt.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = wp.get_device(config.device)
        self.props: list[PropInstance] = []
        self._next_uid = 0
        self.model: newton.Model | None = None
        self.sim_time = 0.0
        self.fixed_dt = 1.0 / config.physics_hz
        self.substeps = config.substeps + config.substeps % 2  # even: state buffers end where they began
        self._accum = 0.0
        self._grab_ids = wp.full(MAX_GRABS, -1, dtype=wp.int32, device=self.device)
        self._grab_targets = wp.zeros(MAX_GRABS, dtype=wp.transform, device=self.device)
        self._grab_host_ids = np.full(MAX_GRABS, -1, dtype=np.int32)
        self._grab_host_targets = np.zeros((MAX_GRABS, 7), dtype=np.float32)
        self.stream = wp.Stream(self.device) if self.device.is_cuda else None
        self._capture_failed = False
        self._rebuild()

    # --- world construction ------------------------------------------------------------------
    def _add_environment(self, builder: newton.ModelBuilder) -> None:
        builder.add_ground_plane()
        t = self.config.table
        cx, cy, _ = t.center
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(cx, cy, t.height - t.top_thickness * 0.5), wp.quat_identity()),
            hx=t.size[0] * 0.5, hy=t.size[1] * 0.5, hz=t.top_thickness * 0.5,
            label="table_top",
        )

    def _rebuild(self) -> None:
        t0 = time.perf_counter()
        saved_q = saved_qd = None
        if self.model is not None and self.model.body_count:
            saved_q = self.state_0.body_q.numpy()
            saved_qd = self.state_0.body_qd.numpy()

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = self.config.friction
        self._add_environment(builder)

        restore_q, restore_qd = [], []
        for prop in self.props:
            old_start = prop.body_start
            prop.body_start = builder.body_count
            builder.add_builder(prop.asset.builder, xform=wp.transform(prop.spawn.p, prop.spawn.q))
            if old_start >= 0 and saved_q is not None:
                n = len(prop.asset.bodies)
                restore_q.append((prop.body_start, saved_q[old_start:old_start + n]))
                restore_qd.append((prop.body_start, saved_qd[old_start:old_start + n]))

        self.model = builder.finalize(device=self.device)
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=self.config.solver_iterations)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision = newton.CollisionPipeline(self.model)
        self.contacts = self.collision.contacts()

        if restore_q:
            q = self.state_0.body_q.numpy()
            qd = self.state_0.body_qd.numpy()
            for start, values in restore_q:
                q[start:start + len(values)] = values
            for start, values in restore_qd:
                qd[start:start + len(values)] = values
            self.state_0.body_q.assign(q)
            self.state_0.body_qd.assign(qd)

        self._radius = np.zeros(self.model.body_count)
        self._center = np.zeros((self.model.body_count, 3))
        for prop in self.props:
            for idx, body in zip(prop.body_indices, prop.asset.bodies):
                self._radius[idx] = body.radius
                self._center[idx] = body.center
        self._q_host = self.state_0.body_q.numpy() if self.model.body_count else np.zeros((0, 7))
        self._capture()
        log.info("Physics rebuilt: %d props, %d bodies, %d shapes in %.0f ms%s",
                 len(self.props), self.model.body_count, self.model.shape_count,
                 (time.perf_counter() - t0) * 1000, " (CUDA graph)" if self.graph else "")

    def _simulate_frame(self) -> None:
        """One fixed physics frame: all substeps. Captured into a CUDA graph when possible."""
        dt = self.fixed_dt / self.substeps
        for _ in range(self.substeps):
            wp.launch(_drive_bodies, dim=MAX_GRABS, device=self.device,
                      inputs=[self.state_0.body_q, self.state_0.body_qd, self.model.body_com,
                              self._grab_ids, self._grab_targets,
                              # Close the gap over the whole frame, not one substep, to avoid jitter.
                              1.0 / self.fixed_dt, self.config.grab_max_linear_speed,
                              self.config.grab_max_angular_speed])
            self.state_0.clear_forces()
            self.collision.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self) -> None:
        self.graph = None
        if not self.device.is_cuda or self.model.body_count == 0 or self._capture_failed:
            return
        # The renderer shares this process and leaves its own CUDA context current, so record on a
        # stream Warp created itself ("unknown stream" otherwise).
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            self._simulate_frame()  # warm up: compile/load every kernel before recording
            self.state_0.body_q.assign(self._q_host)  # undo the warm-up's motion
            try:
                with wp.ScopedCapture(device=self.device, stream=self.stream) as capture:
                    self._simulate_frame()
                self.graph = capture.graph
            except RuntimeError as exc:
                self._capture_failed = True
                log.warning("CUDA graph capture unavailable (%s); stepping with regular launches", exc)

    # --- props -------------------------------------------------------------------------------
    def add_prop(self, asset: PreparedAsset, spawn: Pose, prim_root_fmt: str = "/World/Props/prop_{:04d}") -> PropInstance:
        prop = PropInstance(self._next_uid, asset, prim_root_fmt.format(self._next_uid), spawn)
        self._next_uid += 1
        self.props.append(prop)
        self._rebuild()
        return prop

    def remove_props(self, props: list[PropInstance]) -> None:
        if not props:
            return
        doomed = {p.uid for p in props}
        self.props = [p for p in self.props if p.uid not in doomed]
        self._rebuild()

    def prop_for_body(self, body: int) -> PropInstance | None:
        for prop in self.props:
            if body in prop.body_indices:
                return prop
        return None

    # --- simulation --------------------------------------------------------------------------
    def step(self, frame_dt: float, grabs: list[GrabTarget]) -> None:
        """Advance by real elapsed time in fixed physics frames (at most 3 per call to avoid spirals)."""
        if self.model.body_count == 0:
            return
        self._accum = min(self._accum + frame_dt, 3 * self.fixed_dt)
        frames = int(self._accum / self.fixed_dt + 1e-6)
        if frames == 0:
            return
        self._accum -= frames * self.fixed_dt

        self._grab_host_ids[:] = -1
        for slot, grab in enumerate(grabs[:MAX_GRABS]):
            self._grab_host_ids[slot] = grab.body
            self._grab_host_targets[slot, :3] = grab.target.p
            self._grab_host_targets[slot, 3:] = grab.target.q
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            self._grab_ids.assign(self._grab_host_ids)
            self._grab_targets.assign(self._grab_host_targets)
            for _ in range(frames):
                if self.graph is not None:
                    wp.capture_launch(self.graph, stream=self.stream)
                else:
                    self._simulate_frame()
        self.sim_time += frames * self.fixed_dt
        self._q_host = self.state_0.body_q.numpy()

    def body_pose(self, body: int) -> Pose:
        v = self._q_host[body]
        return Pose(v[:3], v[3:7])

    def set_linear_velocity(self, body: int, velocity: np.ndarray) -> None:
        qd = self.state_0.body_qd.numpy()
        qd[body, :3] = velocity
        self.state_0.body_qd.assign(qd)

    def prim_transforms(self) -> dict[str, list[float]]:
        """World transforms (ovxr row-major layout) for every simulated body prim."""
        out = {}
        for prop in self.props:
            for idx, body in zip(prop.body_indices, prop.asset.bodies):
                v = self._q_host[idx]
                out[f"{prop.prim_root}/{body.rel_path}"] = Pose(v[:3], v[3:7]).to_row_major(body.scale)
        return out

    # --- picking -----------------------------------------------------------------------------
    def _centers_world(self) -> np.ndarray:
        if not len(self._q_host):
            return np.zeros((0, 3))
        return np.array([self.body_pose(i).p + self.body_pose(i).axis(self._center[i])
                         if self._radius[i] > 0 else np.full(3, np.inf)
                         for i in range(len(self._q_host))])

    def pick_near(self, point: np.ndarray, reach: float) -> int | None:
        centers = self._centers_world()
        if not len(centers):
            return None
        gap = np.linalg.norm(centers - point, axis=1) - self._radius
        best = int(np.argmin(gap))
        return best if gap[best] <= reach else None

    def pick_ray(self, origin: np.ndarray, direction: np.ndarray, max_dist: float) -> tuple[int, float] | None:
        """Nearest bounding sphere hit by the ray. Returns (body, distance)."""
        centers = self._centers_world()
        if not len(centers):
            return None
        d = direction / np.linalg.norm(direction)
        rel = centers - origin
        along = rel @ d
        perp = np.linalg.norm(rel - np.outer(along, d), axis=1)
        hit = (along > 0) & (along < max_dist) & (perp <= self._radius * 0.9)
        if not hit.any():
            return None
        candidates = np.where(hit)[0]
        best = candidates[np.argmin(along[candidates])]
        return int(best), float(along[best])

    def ray_floor_distance(self, origin: np.ndarray, direction: np.ndarray, max_dist: float) -> float:
        """Distance until the ray hits the floor (z = 0) or the table top, capped at max_dist."""
        t = self.config.table
        best = max_dist
        if direction[2] < -1e-6:
            for plane_z, bounds in ((0.0, None), (t.height, t)):
                dist = (plane_z - origin[2]) / direction[2]
                if 0 < dist < best:
                    hit = origin + direction * dist
                    if bounds is None or (abs(hit[0] - t.center[0]) <= t.size[0] / 2
                                          and abs(hit[1] - t.center[1]) <= t.size[1] / 2):
                        best = dist
        return best
