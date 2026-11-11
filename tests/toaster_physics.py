"""Headless ovphysx check of the toaster on the Studierstube desk (no headset, no renderer).

    uv run python tests/toaster_physics.py

Loads the C++ app's physics environment, spawns the toaster station the way vrsim_cpp does (the
sidecar's wrapper under a placed Prop), and drives it the way the app does: velocity writes on the
lever and dial bodies, joint positions measured from body poses relative to the housing
(vrsim.toaster.joint_frames). Then it checks:

  1. the toaster rests on the desk (it does not fall through or tip over),
  2. a hand pushing the lever down takes it to the bottom of its travel,
  3. released, the lever spring brings it back up,
  4. the latch (velocity drive toward the bottom, as the app applies it while the ECU latches) holds it,
  5. turning the dial changes its reading, which the ECU decodes into a browning level,
  6. the measured joints, fed to the FMUs, start the heater (lever latched by the ECU itself).
"""

import logging
import sys

import numpy as np
import ovphysx
import ovstage
from ovphysx import PhysX
from ovphysx.samples.python_samples.schema_support import physx_schema_paths
from ovphysx.types import SimObjectType

from vrsim.assets import describe_asset
from vrsim.config import Config
from vrsim.scene import write_environment
from vrsim.toaster import STEP, CHAMBER, LATCH, LEVER, CoSimulation, ToasterScenario

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
DT = 1.0 / 72.0
ROOT = "/World/Props/prop_toaster"


def quat_rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


class Rig:
    def __init__(self, config: Config, scenario: ToasterScenario):
        self.scenario = scenario
        ovstage.population.register_usd_schemas([*[str(p) for p in physx_schema_paths()], str(ovphysx.newton_schema_root())])
        self.stage = ovstage.Stage("toaster-physics-test")
        env = write_environment(config, config.generated_dir / "environment_physics_test.usda", physics=True)
        ovstage.population.open_usd(self.stage, str(env), ordinal=1, domains=ovstage.PopulationDomain.ALL)
        self.stage.advance_write_floor(1).wait()
        self.physx = PhysX()
        self.physx.wait_task(self.physx.attach_ovstage(self.stage, read_ordinal=1))
        self.paths = ovstage.PathDictionary(self.stage)

        info = describe_asset(scenario.toaster_usd, config.generated_dir)
        ref = info.asset_ref.replace("@.cache", "@" + (config.generated_dir.parent.parent / ".cache").as_posix())
        m = scenario.placement_matrix()
        m[3, 2] += 0.002
        rows = "(" + ", ".join("(" + ", ".join(f"{v:.9g}" for v in r) + ")" for r in m) + ")"
        layer = (f'#usda 1.0\n(\n    defaultPrim = "Prop"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n'
                 f'def Xform "Prop"\n{{\n    matrix4d xformOp:transform = {rows}\n'
                 f'    uniform token[] xformOpOrder = ["xformOp:transform"]\n'
                 f'    def "Asset" (prepend references = {ref}) {{}}\n}}\n')
        ovstage.population.add_usd_reference_from_string(self.stage, layer, ROOT)
        ovstage.population.apply_usd_changes(self.stage, 2)
        self.stage.advance_write_floor(2).wait()
        self.physx.wait_task(self.physx.update_from_ovstage(2, 2))
        self.base = f"{ROOT}/{scenario.base}"
        self.bodies = {role: f"{ROOT}/{f['body']}" for role, f in scenario.joints.items()}
        self.turns, self.last_angle = 0.0, None
        self.home = None

    def read(self) -> dict:
        out: dict = {}
        ids = [self.paths.intern_path(p) for p in (self.base, *self.bodies.values())]
        with self.physx.read_sync(SimObjectType.RIGID_BODY, ["position", "orientation"], prim_paths=ids) as r:
            for g in r.groups:
                name = r.dictionary.token_to_string(g.attribute)
                paths = r.dictionary.get_path_strings(g.prim_list)
                arr = g.tensors[0].numpy().copy()
                for p, row in zip(paths, arr):
                    out.setdefault(p, {})[name] = row.astype(float)
        return out

    def joints(self, poses: dict) -> dict:
        b = poses[self.base]
        values = {}
        for role, frame in self.scenario.joints.items():
            part = poses[self.bodies[role]]
            rel_p = quat_rotate(quat_conj(b["orientation"]), part["position"] - b["position"])
            axis = np.array(frame["axis"])
            if frame["kind"] == "linear":
                values[role] = float(np.dot(rel_p - frame["origin"], axis) + frame["offset"])
            else:
                q = quat_mul(quat_conj(b["orientation"]), part["orientation"])
                angle = np.degrees(2 * np.arctan2(np.dot(q[:3], axis), q[3]))
                angle = (angle + 180) % 360 - 180
                if self.last_angle is not None:
                    self.turns += (angle - self.last_angle + 180) % 360 - 180
                else:
                    self.turns = angle
                self.last_angle = angle
                values[role] = self.turns + frame["offset"]
        return values

    def drive(self, poses: dict, linear: dict, angular: dict) -> None:
        """Velocities in the toaster frame per role (linear [m/s] along the joint axis, angular [rad/s])."""
        q = poses[self.base]["orientation"]
        rows_l, rows_a = {}, {}
        for role in set(linear) | set(angular):
            axis = np.array(self.scenario.joints[role]["axis"])
            rows_l[self.bodies[role]] = quat_rotate(q, axis * linear.get(role, 0.0))
            rows_a[self.bodies[role]] = quat_rotate(q, axis * angular.get(role, 0.0))
        for attr, rows in (("linearVelocity", rows_l), ("angularVelocity", rows_a)):
            ids = [self.paths.intern_path(p) for p in rows]
            with self.physx.write_sync(SimObjectType.RIGID_BODY, [attr], prim_paths=ids) as w:
                for g in w.groups:
                    paths = self.paths.get_path_strings(g.prim_list)
                    g.tensors[0].assign(np.stack([rows[p] for p in paths]).astype(np.float32))
                    w.commit_sync(g)

    def anchor(self, poses: dict) -> None:
        """Hold the housing where it settled, as the app's fixed-prop anchor does (velocity drive)."""
        b = poses[self.base]
        if self.home is None:
            self.home = (b["position"].copy(), b["orientation"].copy())
        dq = quat_mul(self.home[1], quat_conj(b["orientation"]))
        dq = dq if dq[3] >= 0 else -dq
        w = 2 * dq[:3] / DT
        for attr, v in (("linearVelocity", (self.home[0] - b["position"]) / DT), ("angularVelocity", w)):
            with self.physx.write_sync(SimObjectType.RIGID_BODY, [attr], prim_paths=[self.paths.intern_path(self.base)]) as wr:
                for g in wr.groups:
                    g.tensors[0].assign(np.asarray([v], dtype=np.float32))
                    wr.commit_sync(g)

    def step(self, seconds: float, drive=None) -> dict:
        poses = self.read()
        for _ in range(int(round(seconds / DT))):
            if self.home is not None:
                self.anchor(poses)
            if drive:
                drive(poses, self.joints(poses))
            self.physx.step_sync(DT)
            poses = self.read()
        self.values = self.joints(poses)
        return poses

    def close(self):
        self.physx.wait_all()
        self.physx.wait_task(self.physx.detach_ovstage())
        self.physx.destroy()


def main() -> int:
    config = Config()
    t = config.environment.toaster
    scenario = ToasterScenario.load(t.position, t.yaw)
    rig = Rig(config, scenario)
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    poses = rig.step(1.0)
    rig.anchor(poses)  # from here on the housing is held, as the app fixes it on spawn
    z0 = poses[rig.base]["position"][2]
    lever0, dial_start = rig.values["lever"], rig.values["dial"]
    up = quat_rotate(poses[rig.base]["orientation"], np.array([0.0, 0.0, 1.0]))[2]
    check("rests on the desk", abs(z0 - t.position[2]) < 0.02 and up > 0.99,
          f"housing z {z0:.3f} m (desk {t.position[2]:.3f}), up {up:.3f}, lever {lever0 * 1000:.1f} mm")

    lever_lo = scenario.joints["lever"]["lower"]
    def toward(role, target):  # the hand: a velocity that closes the gap this frame
        return lambda p, j: rig.drive(p, {role: float(np.clip((target - j[role]) / DT, -0.4, 0.4))}, {})             if role != "dial" else rig.drive(p, {}, {role: float(np.clip(np.radians(target - j[role]) / DT, -8, 8))})

    hold_dial = toward("dial", dial_start)
    rig.step(0.6, lambda p, j: (toward("lever", lever_lo - 0.005)(p, j), hold_dial(p, j)))
    down = rig.values["lever"]
    check("hand pushes the lever down", down < -0.05, f"{down * 1000:.1f} mm (limit {lever_lo * 1000:.1f})")

    rig.step(1.0)
    back = rig.values["lever"]
    check("spring returns the lever", back > -0.01, f"{back * 1000:.1f} mm after 1 s")

    rig.step(0.6, toward("lever", lever_lo - 0.005))
    latch = lever_lo + 0.0002
    rig.step(2.0, lambda p, j: rig.drive(p, {"lever": float(np.clip((latch - j["lever"]) / DT, -0.4, 0.4))}, {}))
    held = rig.values["lever"]
    check("latch holds the lever down", held < -0.055, f"{held * 1000:.1f} mm after 2 s latched")
    rig.step(1.0)

    dial0 = rig.values["dial"]
    rig.step(0.8, toward("dial", dial0 + 90))
    dial1 = rig.values["dial"]
    rig.step(2.0, toward("dial", dial1))  # the app holds a released dial where it was left
    dial2 = rig.values["dial"]
    check("dial turns and stays", abs(dial1 - dial0 - 90) < 10 and abs(dial2 - dial1) < 3,
          f"{dial0:.0f} -> {dial1:.0f} deg, {dial2:.0f} deg 2 s after letting go")

    # closed loop with the FMUs: push the lever, the ECU latches it and heats
    sim = CoSimulation(scenario.instances, scenario.initial)
    state = {"pushing": 1.0}

    def closed_loop(p, j):
        for role, (prim, attr) in (("lever", (LEVER, "state:linear:physics:position")),
                                   ("dial", ("/Toaster/Joints/joint_dial_joint_00", "state:angular:physics:position")),
                                   ("cancel", ("/Toaster/Joints/joint_cancel_button_joint_00", "state:linear:physics:position"))):
            sim.usd.setdefault(prim, {})[attr] = j[role]
        sim.step(DT)
        target = sim.usd[LEVER].get(LATCH, -0.0005)
        if state["pushing"] > 0:
            state["pushing"] -= DT
            toward("lever", lever_lo - 0.005)(p, j)
        elif target < -0.03:
            rig.drive(p, {"lever": float(np.clip((max(target, latch) - j["lever"]) / DT, -0.4, 0.4))}, {})

    rig.step(4.0, closed_loop)
    heater = sim.usd[CHAMBER].get("toaster:heaterPower", 0.0)
    check("FMUs latch and heat from measured joints", heater > 0 and rig.values["lever"] < -0.05,
          f"heater {heater:.0f} W, latch target {sim.usd[LEVER].get(LATCH):.4f} m, lever {rig.values['lever'] * 1000:.1f} mm, "
          f"dial {rig.values['dial']:.0f} deg, T {sim.usd[CHAMBER].get('toaster:temperature', 0):.1f} degC")
    sim.close()
    rig.close()
    print("OK" if all(checks) else "FAILED")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
