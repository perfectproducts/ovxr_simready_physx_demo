"""Newton rigid-body physics built from the app's ovstage through ovnewton.

The stage is the single scene: props are referenced into it (ovstage population) and ovnewton
builds the Newton model from what is populated there, so Newton never parses USD itself. After
each step, ``binding.update_to_ovstage`` publishes every body pose into the same stage at the
frame's ordinal, which ovrtx then renders.

Two things keep this fast enough for 72 Hz VR:

- **Static environment once.** ovnewton imports one ovstage into an empty builder, and Newton
  models are immutable, so a spawn means a rebuild. Re-importing the Studierstube's ~350 convex
  colliders took 2 s per spawn. The environment's colliders therefore live in a separate,
  physics-only ovstage, imported once into a cached builder, and appended to each rebuilt model
  after the props (``add_builder``). The render stage carries no environment colliders.
- **One CUDA graph per frame.** All substeps, including the hand/anchor drive kernel, replay as
  one captured graph. Drive targets sit in fixed device buffers the host refreshes each frame.

Hands and anchors drive bodies by velocity, not position (same approach as the C++ app): every
substep the kernel sets the velocity that closes the gap to the target over ``horizon`` seconds,
from the body's current pose. Contacts stay physical and the last velocity becomes the throw.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import newton
import numpy as np
import ovnewton
import ovstage
import warp as wp
from ovstage import population

from vrsim.mathutil import Pose

log = logging.getLogger(__name__)

MAX_DRIVES = 96  # bodies driven per frame: two hands' props, fixed props' anchors, toaster parts


@dataclass
class Drive:
    """Drive one body toward ``target`` over ``horizon`` seconds, speed-capped."""

    body: int
    target: Pose
    horizon: float
    max_linear: float = 12.0
    max_angular: float = 30.0


@wp.kernel
def _drive_bodies(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    ids: wp.array(dtype=wp.int32),
    targets: wp.array(dtype=wp.transform),
    params: wp.array(dtype=wp.vec3),  # (1 / horizon, max linear, max angular)
):
    t = wp.tid()
    b = ids[t]
    if b < 0:
        return
    x = body_q[b]
    goal = targets[t]
    inv_h = params[t][0]
    p = wp.transform_get_translation(x)
    q = wp.transform_get_rotation(x)
    v = (wp.transform_get_translation(goal) - p) * inv_h
    dq = wp.transform_get_rotation(goal) * wp.quat_inverse(q)
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])
    axis = wp.vec3(dq[0], dq[1], dq[2])
    s = wp.length(axis)
    w = wp.vec3(0.0)
    if s > 1.0e-7:
        w = axis * (2.0 * wp.atan2(s, dq[3]) / s) * inv_h
    if wp.length(v) > params[t][1]:
        v = wp.normalize(v) * params[t][1]
    if wp.length(w) > params[t][2]:
        w = wp.normalize(w) * params[t][2]
    # body_qd's linear part is the centre-of-mass velocity; v is the body-origin velocity.
    r = wp.transform_vector(x, body_com[b])
    body_qd[b] = wp.spatial_vector(v + wp.cross(w, r), w)


class NewtonPhysics:
    """The app's Newton world. ``rebuild(ordinal)`` after every population change."""

    def __init__(self, stage: ovstage.Stage, env_physics_usd: str | None, *, device: str = "cuda:0",
                 physics_dt: float = 1.0 / 72.0, substeps: int = 2, iterations: int = 8,
                 friction: float = 0.8, collide_every_substep: bool = True, broad_phase: str = "sap"):
        self.stage = stage
        self.device = wp.get_device(device)
        self.dt = physics_dt
        self.substeps = substeps + substeps % 2  # even: the state buffers end where they began
        self.iterations = iterations
        self.friction = friction
        self.collide_every_substep = collide_every_substep
        self.broad_phase = broad_phase
        self.stream = wp.Stream(self.device) if self.device.is_cuda else None
        self._env_builder = self._import_environment(env_physics_usd)
        self.model: newton.Model | None = None
        self.binding = None
        self.graph = None
        self._capture_failed = False
        self.index: dict[str, int] = {}
        self.labels: list[str] = []
        self.q = np.zeros((0, 7), dtype=np.float32)  # host copy of body_q after the last step
        self._drive_ids = wp.full(MAX_DRIVES, -1, dtype=wp.int32, device=self.device)
        self._drive_targets = wp.zeros(MAX_DRIVES, dtype=wp.transform, device=self.device)
        self._drive_params = wp.zeros(MAX_DRIVES, dtype=wp.vec3, device=self.device)
        self._host_ids = np.full(MAX_DRIVES, -1, dtype=np.int32)
        self._host_targets = np.zeros((MAX_DRIVES, 7), dtype=np.float32)
        self._host_params = np.zeros((MAX_DRIVES, 3), dtype=np.float32)

    # --- model construction ------------------------------------------------------------------
    def _new_builder(self) -> newton.ModelBuilder:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg.mu = self.friction
        return builder

    def _import_environment(self, usd: str | None) -> newton.ModelBuilder | None:
        """Static colliders of the environment, imported once through their own ovstage."""
        if not usd:
            return None
        t0 = time.perf_counter()
        env_stage = ovstage.Stage("vrnx_environment_physics", config=ovstage.StageConfig(
            runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL))
        try:
            population.open_usd_with_desc(env_stage, usd, ordinal=1, time=population.DEFAULT_TIME,
                                          descs=(ovnewton.population_desc(),))
            env_stage.advance_write_floor(1).wait()
            builder = self._new_builder()
            ovnewton.add_ovstage(builder, env_stage, ordinal=1)
        finally:
            env_stage.destroy()
        if builder.body_count:
            raise RuntimeError(f"{usd}: the environment must hold only static colliders, found "
                               f"{builder.body_count} rigid bodies")
        log.info("Environment colliders: %d shapes from %s in %.2f s", builder.shape_count, usd,
                 time.perf_counter() - t0)
        return builder

    def _hull_dynamic_meshes(self, builder: newton.ModelBuilder) -> None:
        """Triangle-mesh colliders on moving bodies -> convex hulls.

        Newton collides two triangle meshes triangle by triangle; the toaster's housing alone is
        17k vertices, and its parts made one collision pass cost 3.5 ms of GPU time per substep.
        PhysX cooks such meshes (SDF / convex) for dynamic bodies; here they become convex hulls,
        which is close enough for a prop that stands on the desk with its parts on joints. Meshes
        are hulled once per source mesh and reused across rebuilds.
        """
        mesh = int(newton.GeoType.MESH)
        indices = [i for i, (t, b) in enumerate(zip(builder.shape_type, builder.shape_body))
                   if int(t) == mesh and b >= 0]
        if not indices:
            return
        cache = self.__dict__.setdefault("_hull_cache", {})
        todo = []
        for i in indices:
            label = builder.shape_label[i]
            if label in cache:
                builder.shape_source[i] = cache[label]
                builder.shape_type[i] = newton.GeoType.CONVEX_MESH
            else:
                todo.append(i)
        if todo:
            builder.approximate_meshes("convex_hull", shape_indices=todo)
            for i in todo:
                cache[builder.shape_label[i]] = builder.shape_source[i]
            log.info("Convex hulls for %d dynamic triangle-mesh colliders", len(todo))

    @staticmethod
    def _filter_jointed_parts(builder: newton.ModelBuilder) -> None:
        """No contacts between the bodies of one jointed asset (the toaster's housing, lever, dial,
        button and tray), nor between its jointed parts and the static environment. The hulled
        colliders overlap each other and reach into the desk's hulls, and those contacts jammed the
        lever at its bottom stop. PhysX does not collide jointed bodies either; the parts' housing
        still rests on the desk."""
        parent = list(range(builder.body_count))

        def root(b: int) -> int:
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b

        for a, b in zip(builder.joint_parent, builder.joint_child):
            if a >= 0 and b >= 0:
                parent[root(a)] = root(b)
        shapes: dict[int, list[tuple[int, int]]] = {}
        for s, b in enumerate(builder.shape_body):
            if b >= 0:
                shapes.setdefault(root(b), []).append((s, b))
        static = [s for s, b in enumerate(builder.shape_body) if b < 0]
        parts = {b for a, b in zip(builder.joint_parent, builder.joint_child) if a >= 0 and b >= 0}
        pairs = 0
        for s0, b0 in enumerate(builder.shape_body):
            if b0 in parts:
                for s1 in static:
                    builder.add_shape_collision_filter_pair(s0, s1)
                    pairs += 1
        for group in shapes.values():
            for i, (s0, b0) in enumerate(group):
                for s1, b1 in group[i + 1:]:
                    if b0 != b1:
                        builder.add_shape_collision_filter_pair(s0, s1)
                        pairs += 1
        if pairs:
            log.debug("Filtered %d shape pairs inside jointed assets", pairs)

    def rebuild(self, ordinal: int) -> None:
        """Rebuild the model from the stage sealed at ``ordinal``. Bodies that survive keep their
        pose and velocity; new ones start where the stage puts them."""
        t0 = time.perf_counter()
        saved: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.model is not None and self.model.body_count:
            q, qd = self.state_0.body_q.numpy(), self.state_0.body_qd.numpy()
            saved = {label: (q[i], qd[i]) for i, label in enumerate(self.labels)}

        builder = self._new_builder()
        result = ovnewton.add_ovstage(builder, self.stage, ordinal=ordinal)
        self._hull_dynamic_meshes(builder)
        t_import = time.perf_counter()
        if self._env_builder is not None:
            builder.add_builder(self._env_builder)
        self._filter_jointed_parts(builder)
        self.model = builder.finalize(device=self.device, skip_validation_joints=result.has_orphan_joints)
        t_final = time.perf_counter()
        self.binding = ovnewton.attach_ovstage(self.stage, model=self.model, ordinal=ordinal)
        self.labels = list(self.model.body_label)
        self.index = {label: i for i, label in enumerate(self.labels)}

        self.solver = newton.solvers.SolverXPBD(self.model, iterations=self.iterations)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        if self.model.joint_count:
            # Newton replaces a drive target outside the joint limits with the limits' midpoint.
            # SimReady springs rely on such targets (the toaster lever's 0.04 m above its top stop
            # holds it up), so take the authored targets from the stage as they are.
            self.binding.update_from_ovstage(self.state_0, self.control, attributes=("jointPositionTarget",))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.collision = newton.CollisionPipeline(self.model, broad_phase=self.broad_phase)
        self.contacts = self.collision.contacts()
        if saved and self.model.body_count:
            q, qd = self.state_0.body_q.numpy(), self.state_0.body_qd.numpy()
            for i, label in enumerate(self.labels):
                if label in saved:
                    q[i], qd[i] = saved[label]
            self.state_0.body_q.assign(q)
            self.state_0.body_qd.assign(qd)
        self.q = self.state_0.body_q.numpy() if self.model.body_count else np.zeros((0, 7), dtype=np.float32)
        # Newton bodies carry no scale; the body prims do (SimReady packages in centimetres sit
        # under a 0.01 correction). ovnewton records each body's signed world scale when binding.
        try:
            self.scales = np.asarray(self.binding._runtime.body_scale.numpy(), dtype=np.float64).reshape(-1, 3)
        except (AttributeError, TypeError):
            self.scales = np.ones((self.model.body_count, 3))
        self._capture()
        log.info("Physics rebuilt at ordinal %d: %d bodies, %d joints, %d shapes in %.0f ms "
                 "(import %.0f, finalize %.0f)%s", ordinal, self.model.body_count, self.model.joint_count,
                 self.model.shape_count, (time.perf_counter() - t0) * 1000, (t_import - t0) * 1000,
                 (t_final - t_import) * 1000, " · CUDA graph" if self.graph else "")

    # --- stepping ----------------------------------------------------------------------------
    def _frame(self) -> None:
        dt = self.dt / self.substeps
        for i in range(self.substeps):
            wp.launch(_drive_bodies, dim=MAX_DRIVES, device=self.device,
                      inputs=[self.state_0.body_q, self.state_0.body_qd, self.model.body_com,
                              self._drive_ids, self._drive_targets, self._drive_params])
            self.state_0.clear_forces()
            if self.collide_every_substep or i == 0:
                self.collision.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self) -> None:
        self.graph = None
        if not self.device.is_cuda or self.model.body_count == 0 or self._capture_failed:
            return
        # ovrtx leaves its own CUDA context current: record on a stream Warp created itself.
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            q, qd = wp.clone(self.state_0.body_q), wp.clone(self.state_0.body_qd)
            self._frame()  # warm-up: load every kernel before recording
            self.state_0.body_q.assign(q)
            self.state_0.body_qd.assign(qd)
            try:
                with wp.ScopedCapture(device=self.device, stream=self.stream) as capture:
                    self._frame()
                self.graph = capture.graph
            except RuntimeError as exc:
                self._capture_failed = True
                log.warning("CUDA graph capture unavailable (%s); stepping with regular launches", exc)
            self.state_0.body_q.assign(q)  # the recorded frame ran once too: undo it
            self.state_0.body_qd.assign(qd)

    def step(self, frames: int, drives: list[Drive]) -> None:
        """Advance ``frames`` fixed physics frames with these drives active."""
        if self.model is None or self.model.body_count == 0 or frames <= 0:
            return
        self._host_ids[:] = -1
        for slot, d in enumerate(drives[:MAX_DRIVES]):
            self._host_ids[slot] = d.body
            self._host_targets[slot, :3] = d.target.p
            self._host_targets[slot, 3:] = d.target.q
            self._host_params[slot] = (1.0 / max(d.horizon, 1e-4), d.max_linear, d.max_angular)
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            self._drive_ids.assign(self._host_ids)
            self._drive_targets.assign(self._host_targets)
            self._drive_params.assign(self._host_params)
            for _ in range(frames):
                if self.graph is not None:
                    wp.capture_launch(self.graph, stream=self.stream)
                else:
                    self._frame()
            self.q = self.state_0.body_q.numpy()

    def publish(self, ordinal: int) -> None:
        """ovnewton's own publication: every body pose (``omni:xform``), velocity and joint state
        into the stage at ``ordinal``; the caller seals it. The app's default instead writes the
        poses in its own per-frame transform batch (``body_matrices``): ~0.1 ms against ~3 ms."""
        if self.binding is not None and self.model.body_count:
            with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
                self.binding.update_to_ovstage(self.state_0, ordinal=ordinal)

    def body_matrices(self) -> np.ndarray:
        """Row-major, row-vector world matrices (N x 16, prim scale included) of every body, for
        the app's batched OVStage transform write."""
        q = self.q.astype(np.float64)
        n = len(q)
        if n == 0:
            return np.zeros((0, 16))
        x, y, z, w = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        r = np.empty((n, 3, 3))  # column-vector rotation
        r[:, 0, 0], r[:, 0, 1], r[:, 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
        r[:, 1, 0], r[:, 1, 1], r[:, 1, 2] = 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
        r[:, 2, 0], r[:, 2, 1], r[:, 2, 2] = 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
        m = np.zeros((n, 4, 4))
        m[:, :3, :3] = np.transpose(r, (0, 2, 1)) * self.scales[:, :, None]  # row i = scaled local axis i
        m[:, 3, :3] = q[:, :3]
        m[:, 3, 3] = 1.0
        return m.reshape(n, 16)

    # --- body access -------------------------------------------------------------------------
    def pose(self, body: int) -> Pose:
        v = self.q[body]
        return Pose(v[:3], v[3:7])

    def set_velocities(self, velocities: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        """One-off body-origin velocities (a throw, stopping a released part): {body: (v, w)}."""
        if not velocities or self.model is None:
            return
        qd = self.state_0.body_qd.numpy()
        com = self.model.body_com.numpy()
        for body, (v, w) in velocities.items():
            pose = self.pose(body)
            r = pose.axis(com[body])
            qd[body, :3] = np.asarray(v) + np.cross(w, r)
            qd[body, 3:] = w
        self.state_0.body_qd.assign(qd)

    def destroy(self) -> None:
        self.binding = None
        self.model = None
