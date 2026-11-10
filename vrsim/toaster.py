"""The dbeyond smart toaster on the Studierstube desk: its FMUs, run live against the VR physics.

The toaster product (assets/dbeyond, from the dSimula x SyncTwin demonstrator) carries its behaviour
as an ovfmi USD-FMI binding: ``FmuInstance`` prims name an FMU, their ``FmuConnection`` children
target prims, and ``FmuMapping`` grandchildren route one FMU variable to one USD attribute
(component). This module reads that binding from the composed station layer with pxr, steps the
FMUs with FMPy (the backend ovfmi uses) in real time and routes the values through a dictionary
that stands in for the stage attributes, in ovfmi's order: inputs -> step every FMU -> outputs.

The C++ app owns the physics. It measures the lever, dial and cancel-button joints from the body
poses and sends them (``toaster_in``); this side answers with the outputs it needs (``toaster``):
the latch target for the lever, the toast's pose and colour, and the heater glow.

    app -> sidecar   {"t":"toaster_in","lever":m,"dial":deg,"cancel":m}
    sidecar -> app   {"t":"toaster","lever_target":m,"toast":[x,y,z,rot_deg],"toast_color":[r,g,b],
                      "glow":intensity,"heater":W,"temp":degC,"remaining":s,"level":n}

    uv run python -m vrsim.toaster        (offline check: 600 s with a scripted user, no headset)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT

log = logging.getLogger(__name__)

DBEYOND = PROJECT_ROOT / "assets" / "dbeyond"
STATION_USD = DBEYOND / "Scenarios" / "StudierstubeToaster" / "ToasterStation.usda"
ROOT = "/Toaster"  # the station layer's defaultPrim; the app references it at <prop>/Asset
LEVER = ROOT + "/Joints/joint_lever_joint_00"
DIAL = ROOT + "/Joints/joint_dial_joint_00"
CANCEL = ROOT + "/Joints/joint_cancel_button_joint_00"
CHAMBER = ROOT + "/ToasterECU/Chamber"
GLOW = ROOT + "/ToasterECU/HeaterGlow"
DISPLAY = ROOT + "/ToasterECU/Display"
TOAST = ROOT + "/Toast"
TOAST_MATERIAL = "/Looks/ToastMat/Surface"  # below the toast prim
GLOW_SHADER = ROOT + "/ToasterECU/SlotGlow/Glow/Shader"
LINEAR, ANGULAR = "state:linear:physics:position", "state:angular:physics:position"
LATCH = "drive:linear:physics:targetPosition"

STEP = 0.01  # [s] co-simulation step; the ECU samples every 0.05 s inside its FMU
JOINTS = {"lever": (LEVER, LINEAR), "dial": (DIAL, ANGULAR), "cancel": (CANCEL, LINEAR)}


# --- ovfmi binding ----------------------------------------------------------------------------
@dataclass
class Mapping:
    fmu_var: str
    usd_attr: str
    direction: str  # "input" (USD -> FMU) or "output"
    offset: int = 0  # component of the USD attribute (fmi:usdMapping = (offset, count))
    count: int = 1


@dataclass
class Connection:
    targets: list[str]
    mappings: list[Mapping]


@dataclass
class Instance:
    path: str
    fmu: Path
    connections: list[Connection]


def read_binding(stage) -> list[Instance]:
    """Every enabled FmuInstance on the composed stage, with its connections and mappings."""
    instances = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "FmuInstance":
            continue
        enabled = prim.GetAttribute("fmi:enabled")
        if enabled and enabled.Get() is False:
            continue
        asset = prim.GetAttribute("fmi:fmu").Get()
        fmu = Path(asset.resolvedPath) if asset and asset.resolvedPath else None
        if fmu is None or not fmu.exists():
            raise FileNotFoundError(f"{prim.GetPath()}: FMU {asset} not found")
        connections = []
        for conn in prim.GetChildren():
            if conn.GetTypeName() != "FmuConnection":
                continue
            targets = [str(t) for t in conn.GetRelationship("fmi:targets").GetTargets()]
            mappings = []
            for m in conn.GetChildren():
                if m.GetTypeName() != "FmuMapping":
                    continue
                usd_map = m.GetAttribute("fmi:usdMapping").Get() or (0, 1)
                mappings.append(Mapping(m.GetAttribute("fmi:fmuAttribute").Get(), m.GetAttribute("fmi:usdAttribute").Get(),
                                        str(m.GetAttribute("fmi:direction").Get()), int(usd_map[0]), int(usd_map[1])))
            connections.append(Connection(targets, mappings))
        instances.append(Instance(str(prim.GetPath()), fmu, connections))
    return instances


class Fmu:
    """Minimal FMI 2.0 / 3.0 co-simulation wrapper around FMPy."""

    def __init__(self, path: Path, name: str):
        from fmpy import extract, instantiate_fmu, read_model_description

        self.name = name
        self.md = read_model_description(str(path))
        self.fmi3 = str(self.md.fmiVersion).startswith("3")
        self.dir = extract(str(path), unzipdir=tempfile.mkdtemp(prefix=f"vrsim_{name}_"))
        self.vars = {v.name: v for v in self.md.modelVariables}
        self.inst = instantiate_fmu(self.dir, self.md, fmi_type="CoSimulation")
        if self.fmi3:
            self.inst.enterInitializationMode(startTime=0.0)
        else:
            self.inst.setupExperiment(startTime=0.0)
            self.inst.enterInitializationMode()
        self.inst.exitInitializationMode()

    def set(self, var: str, value: float) -> None:
        v = self.vars[var]
        vr = [v.valueReference]
        if v.type == "Boolean":
            self.inst.setBoolean(vr, [bool(value)])
        elif v.type in ("Int32", "Integer"):
            (self.inst.setInt32 if self.fmi3 else self.inst.setInteger)(vr, [int(round(value))])
        else:
            (self.inst.setFloat64 if self.fmi3 else self.inst.setReal)(vr, [float(value)])

    def get(self, var: str) -> float:
        v = self.vars[var]
        vr = [v.valueReference]
        if v.type == "Boolean":
            return float(self.inst.getBoolean(vr)[0])
        if v.type in ("Int32", "Integer"):
            return float((self.inst.getInt32 if self.fmi3 else self.inst.getInteger)(vr)[0])
        return float((self.inst.getFloat64 if self.fmi3 else self.inst.getReal)(vr)[0])

    def step(self, t: float, dt: float) -> None:
        self.inst.doStep(currentCommunicationPoint=t, communicationStepSize=dt)

    def close(self) -> None:
        try:
            self.inst.terminate()
            self.inst.freeInstance()
        finally:
            shutil.rmtree(self.dir, ignore_errors=True)


class CoSimulation:
    """The FMUs of one binding, routed through ``usd`` ({prim path: {attribute: value}})."""

    def __init__(self, instances: list[Instance], initial: dict[str, dict]):
        self.usd = {prim: dict(attrs) for prim, attrs in initial.items()}
        self.fmus = [(Fmu(inst.fmu, inst.path.rsplit("/", 1)[-1]), inst.connections) for inst in instances]
        self.t = 0.0

    def fmu(self, name: str) -> Fmu:
        return next(f for f, _ in self.fmus if f.name == name)

    def step(self, dt: float) -> None:
        for fmu, conns in self.fmus:
            for c in conns:
                for m in c.mappings:
                    if m.direction == "input":
                        value = self.usd.get(c.targets[0], {}).get(m.usd_attr)
                        if isinstance(value, list):
                            value = value[m.offset]
                        if value is not None:
                            fmu.set(m.fmu_var, value)
        for fmu, _ in self.fmus:
            fmu.step(self.t, dt)
        for fmu, conns in self.fmus:
            for c in conns:
                for m in c.mappings:
                    if m.direction != "output":
                        continue
                    value = fmu.get(m.fmu_var)
                    for target in c.targets:
                        attrs = self.usd.setdefault(target, {})
                        current = attrs.get(m.usd_attr)
                        if isinstance(current, list) or m.offset > 0:
                            vec = current if isinstance(current, list) else [0.0] * (m.offset + 1)
                            vec.extend([0.0] * (m.offset + 1 - len(vec)))
                            vec[m.offset] = value  # ovfmi usdMapping: one vector component
                            attrs[m.usd_attr] = vec
                        else:
                            attrs[m.usd_attr] = value
        self.t += dt

    def close(self) -> None:
        for fmu, _ in self.fmus:
            fmu.close()


# --- the station ------------------------------------------------------------------------------
def _value(v):
    if v is None:
        return None
    if hasattr(v, "__len__") and not isinstance(v, str):
        return [float(x) for x in v]
    return float(v)


@dataclass
class ToasterScenario:
    """The station layer, read once: its binding, start values and joint frames."""

    stage: object
    instances: list[Instance]
    initial: dict[str, dict] = field(default_factory=dict)
    joints: dict[str, dict] = field(default_factory=dict)  # role -> frame for the app (see joint_frames)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    base: str = "Asset/Geometry/body_obj_01"

    @property
    def toaster_usd(self) -> Path:
        return STATION_USD

    @property
    def toast_usd(self) -> Path:
        return DBEYOND / "Assets" / "Products" / "Toast_Slice" / "Toast_Slice.usda"

    @classmethod
    def load(cls, position=(0.0, 0.0, 0.0), yaw: float = 0.0) -> "ToasterScenario":
        from pxr import Usd

        stage = Usd.Stage.Open(str(STATION_USD))
        instances = read_binding(stage)
        initial: dict[str, dict] = {}
        for inst in instances:  # every routed attribute starts at its authored value
            for c in inst.connections:
                for m in c.mappings:
                    for target in c.targets:
                        attr = stage.GetPrimAtPath(target).GetAttribute(m.usd_attr)
                        value = _value(attr.Get()) if attr else None
                        if value is not None:
                            initial.setdefault(target, {})[m.usd_attr] = value
        # The latch output starts released (lever up). The station's own drive target is a spring preload.
        initial.setdefault(LEVER, {})[LATCH] = initial.get(LEVER, {}).get(LINEAR, -0.0005)
        initial.setdefault(TOAST, {}).setdefault("xformOp:translate", [0.0, -0.026, 0.17])
        initial[TOAST].setdefault("xformOp:rotateXYZ", [0.0, 0.0, 0.0])
        scenario = cls(stage, instances, initial, position=tuple(position), yaw=float(yaw))
        scenario.joints = scenario.joint_frames()
        return scenario

    def joint_frames(self) -> dict[str, dict]:
        """Each operated joint as the app measures it: the moving body (relative to the wrapper's
        Prop prim), the joint origin and axis in the toaster body's frame, and the offset from that
        geometric reading to the authored joint state (the dial's 150 degrees)."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        xf = UsdGeom.XformCache(Usd.TimeCode.Default())

        def rigid(path):
            prim = self.stage.GetPrimAtPath(path)
            while prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim = prim.GetParent()
            return prim

        frames = {}
        for role, (path, state_attr) in JOINTS.items():
            prim = self.stage.GetPrimAtPath(path)
            joint = UsdPhysics.Joint(prim)
            body0_target = joint.GetBody0Rel().GetTargets()[0]
            body0, body1 = rigid(body0_target), rigid(joint.GetBody1Rel().GetTargets()[0])
            world0 = xf.GetLocalToWorldTransform(body0)
            # localPos0 is in the body0 target's frame (a mesh below the body); move it to the body's
            target_xf = xf.GetLocalToWorldTransform(self.stage.GetPrimAtPath(body0_target))
            local_pos = Gf.Vec3d(prim.GetAttribute("physics:localPos0").Get())
            origin = world0.GetInverse().Transform(target_xf.Transform(local_pos))
            axis_token = str(prim.GetAttribute("physics:axis").Get() or "X")
            axis = np.eye(3)["XYZ".index(axis_token)]
            rest = world0.GetInverse().Transform(xf.GetLocalToWorldTransform(body1).ExtractTranslation())
            linear = state_attr == LINEAR
            geometric = float(np.dot(np.array(rest) - np.array(origin), axis)) if linear else 0.0
            state = prim.GetAttribute(state_attr).Get()
            lower, upper = prim.GetAttribute("physics:lowerLimit").Get(), prim.GetAttribute("physics:upperLimit").Get()
            self.base = "Asset" + str(body0.GetPath()).removeprefix(ROOT)  # the housing: the toaster frame
            frames[role] = {
                "body": "Asset" + str(body1.GetPath()).removeprefix(ROOT),
                "origin": [float(v) for v in origin],
                "axis": axis.tolist(),
                "kind": "linear" if linear else "angular",
                "offset": float(state) - geometric if state is not None else 0.0,
                "lower": float(lower) if lower is not None and np.isfinite(lower) else None,
                "upper": float(upper) if upper is not None and np.isfinite(upper) else None,
            }
        return frames

    def placement_matrix(self) -> np.ndarray:
        """Toaster frame -> world (USD row-vector convention)."""
        c, s = np.cos(np.radians(self.yaw)), np.sin(np.radians(self.yaw))
        m = np.eye(4)
        m[0, :2], m[1, :2] = (c, s), (-s, c)
        m[3, :3] = self.position
        return m

    def spawn_extras(self) -> dict:
        """Spawn-message fields that make the app treat this prop as the toaster."""
        latch = self.stage.GetPrimAtPath(LEVER)
        return {
            "device": "toaster",
            "fixed": True,
            "yaw": self.yaw,
            "base": self.base,
            "joints": self.joints,
            "toast": "Asset" + TOAST.removeprefix(ROOT),
            "toast_material": "Asset" + TOAST.removeprefix(ROOT) + TOAST_MATERIAL,
            "glow_light": "Asset" + GLOW.removeprefix(ROOT),
            "glow_shader": "Asset" + GLOW_SHADER.removeprefix(ROOT),
            "glow_max": 30000.0,  # ECU glow_max: HeaterGlow intensity at full heater power
            "lever_latched_below": -0.03,
            "lever_stiffness": float(latch.GetAttribute("drive:linear:physics:stiffness").Get() or 0.0),
        }


def browning_word(b: float) -> str:
    return ("untoasted" if b < 0.15 else "barely warm" if b < 0.5 else "light" if b < 0.85 else
            "golden" if b < 1.3 else "dark" if b < 1.7 else "burnt")


class ToasterDevice:
    """Real-time co-simulation thread for one app connection.

    ``send(dict)`` delivers ``toaster`` messages to the app; ``feedback.post`` shows events
    (heating starts, the dial changes level, the toast pops up) on the info panel."""

    def __init__(self, scenario: ToasterScenario, send, feedback=None):
        self.scenario = scenario
        self.send = send
        self.feedback = feedback
        self.sim = CoSimulation(scenario.instances, scenario.initial)
        self._lock = threading.Lock()
        self._inputs: dict[str, float] = {}
        self._stop = threading.Event()
        self._last_sent: dict | None = None
        self._last_send_time = 0.0
        self._heating = False
        self._level = None
        self._cancel_seen = False
        self._toast_mode = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="toaster")
        self._thread.start()
        log.info("Toaster co-simulation running: %s", ", ".join(f.name for f, _ in self.sim.fmus))

    def set_inputs(self, lever: float | None = None, dial: float | None = None, cancel: float | None = None) -> None:
        with self._lock:
            for role, value in (("lever", lever), ("dial", dial), ("cancel", cancel)):
                if value is not None and np.isfinite(value):
                    self._inputs[role] = float(value)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.sim.close()

    # --- thread ---------------------------------------------------------------------------------
    def _run(self) -> None:
        start = time.perf_counter()
        try:
            while not self._stop.is_set():
                now = time.perf_counter() - start
                steps = 0
                while self.sim.t + STEP <= now and steps < 25:
                    self._step()
                    steps += 1
                if self.sim.t + STEP <= now:  # fell behind (debugger, sleep): don't try to catch up
                    start = time.perf_counter() - self.sim.t
                if steps:
                    self._publish()
                time.sleep(STEP * 0.5)
        except Exception:
            log.exception("Toaster co-simulation stopped")

    def _step(self) -> None:
        with self._lock:
            inputs = dict(self._inputs)
        usd = self.sim.usd
        for role, (prim, attr) in JOINTS.items():
            if role in inputs:
                usd.setdefault(prim, {})[attr] = inputs[role]
        self.sim.step(STEP)
        self._events()

    def _events(self) -> None:
        if self.feedback is None:
            return
        usd, ecu = self.sim.usd, self.sim.fmu("ToasterECU")
        ch = usd.get(CHAMBER, {})
        heating = ch.get("toaster:heaterPower", 0.0) > 0
        level = int(ecu.get("browning_level"))
        remaining = usd.get(DISPLAY, {}).get("toaster:timeRemaining", 0.0)
        cancel = usd.get(CANCEL, {}).get(LINEAR, 0.0) < -0.0025
        browning = ch.get("toaster:browning", 0.0)
        if heating and not self._heating:
            self.feedback.post("ok", f"Toasting · level {level}",
                               f"about {remaining:.0f} s · the cancel button (below the dial) stops it", hold=6.0)
        elif self._heating and not heating:
            if cancel or self._cancel_seen:
                self.feedback.post("info", "Toaster cancelled", f"the toast is {browning_word(browning)}", hold=5.0)
            else:
                self.feedback.post("ok", "Ding!", f"the toast is {browning_word(browning)} "
                                                  f"(browning {browning:.2f}) · push the lever to put it back", hold=8.0)
        elif self._level is not None and level != self._level and not heating:
            self.feedback.post("info", f"Browning level {level}", "push the lever down to start", hold=3.0)
        self._cancel_seen = (self._cancel_seen or cancel) and heating
        self._heating, self._level = heating, level

    def _publish(self) -> None:
        usd = self.sim.usd
        ch = usd.get(CHAMBER, {})
        toast = usd.get(TOAST, {})
        msg = {
            "t": "toaster",
            "lever_target": round(usd.get(LEVER, {}).get(LATCH, -0.0005), 5),
            "toast": [round(v, 4) for v in (*toast.get("xformOp:translate", [0, -0.026, 0.17])[:3],
                                            toast.get("xformOp:rotateXYZ", [0, 0, 0])[0])],
            "toast_color": [round(v, 3) for v in usd.get(TOAST + TOAST_MATERIAL, {}).get("inputs:diffuseColor", [0.93, 0.85, 0.68])[:3]],
            "glow": round(usd.get(GLOW, {}).get("inputs:intensity", 0.0), 1),
            "heater": round(ch.get("toaster:heaterPower", 0.0), 1),
            "temp": round(ch.get("toaster:temperature", 20.0), 1),
            "remaining": round(usd.get(DISPLAY, {}).get("toaster:timeRemaining", 0.0), 1),
        }
        now = time.perf_counter()
        changed = {k: v for k, v in msg.items() if k not in ("temp", "remaining")} != (
            {k: v for k, v in self._last_sent.items() if k not in ("temp", "remaining")} if self._last_sent else None)
        if changed or now - self._last_send_time > 0.5:
            self.send(msg)
            self._last_sent, self._last_send_time = msg, now


# --- offline check --------------------------------------------------------------------------------
def _offline(stop: float = 600.0) -> None:
    """Scripted user as in the dbeyond headless run: lever at 1 s, level 7 at 140 s, lever at
    150 s and 400 s, cancel at 420 s. The lever follows the latch target, as the PhysX joint would."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    scenario = ToasterScenario.load()
    for role, frame in scenario.joints.items():
        print(f"{role:6s} {frame}")
    sim = CoSimulation(scenario.instances, scenario.initial)
    heating, mode = False, "in slot"
    for k in range(int(round(stop / STEP))):
        t = k * STEP
        pushing = any(t0 <= t < t0 + 0.5 for t0 in (1.0, 150.0, 400.0))
        sim.usd[LEVER][LINEAR] = -0.0595 if pushing else sim.usd[LEVER].get(LATCH, -0.0005)
        sim.usd[DIAL][ANGULAR] = 150.0 if t < 140.0 else 290.0
        sim.usd[CANCEL][LINEAR] = -0.0032 if 420.0 <= t < 420.5 else -0.0002
        sim.step(STEP)
        ch = sim.usd[CHAMBER]
        on = ch.get("toaster:heaterPower", 0) > 0
        pos = sim.usd[TOAST]["xformOp:translate"]
        now_mode = "in slot" if abs(pos[1] + 0.026) < 1e-6 else "landed" if pos[2] < 0.01 else "flying"
        if on != heating or now_mode != mode:
            print(f"{t + STEP:8.2f} s  {'heater on ' if on else 'heater off'}  toast {now_mode:8s} "
                  f"T {ch.get('toaster:temperature', 0):6.1f} degC  browning {ch.get('toaster:browning', 0):.2f}")
            heating, mode = on, now_mode
    print("toast colour", [round(v, 3) for v in sim.usd[TOAST + TOAST_MATERIAL]["inputs:diffuseColor"]])
    sim.close()


if __name__ == "__main__":
    _offline()
