"""Offscreen check of the toaster's FMU monitor panel (no headset): layer + ``points`` writes -> ovrtx.

    uv run python tests/render_toaster_monitor.py [out_dir]

Runs the three FMUs for 40 s (lever pushed at 1 s, level 1), puts the toaster and the monitor panel on
the Studierstube desk where the C++ app places them (toaster frame offset, turned to a seated user's
head) and writes each colour layer's quads as the app does. Renders from the user's eye and closer.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import ovstage
import ovstage.population as population

sys.path.insert(0, str(Path(__file__).parent))
from render_toaster import TOAST, TOASTER, _placed, _render_layer, render  # noqa: E402

from vrsim.config import Config  # noqa: E402
from vrsim.scene import SceneStage  # noqa: E402
from vrsim.toaster import (ANGULAR, CANCEL, DIAL, GLOW_SHADER, LATCH, LEVER, LINEAR, ROOT, STEP, TOAST_MATERIAL,  # noqa: E402
                           CoSimulation, ToasterScenario)
from vrsim.toaster_monitor import ALL_LAYERS, PRIM, MonitorRenderer, monitor_extras, snapshot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

HEAD_LOCAL = np.array([0.6, 0.15, 0.45])  # a seated user's eye in the toaster frame [m] (+X faces them)


def to_world(place: np.ndarray, p) -> np.ndarray:
    return np.append(np.asarray(p, float), 1.0) @ place


def write_toast(scene, place: np.ndarray, v: dict, ordinal: int) -> None:
    """Toast pose and colour and the slot glow from a monitor snapshot, as the apps write them."""
    paths = ovstage.PathDictionary(scene.stage)
    query = lambda path: scene.stage.query_from_path_list(paths.create_path_list_from_strings([path]))
    local = np.eye(4)
    local[3, :3] = (v["fl.toast_x"], v["fl.toast_y"], v["fl.toast_z"])
    m = np.ascontiguousarray((local @ place)[None], dtype=np.float64)
    f64x16 = ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 64, 16)
    f32x3 = ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 32, 3)
    toast = query(TOAST)
    for name in ("omni:fabric:localMatrix", "omni:fabric:worldMatrix"):
        scene.stage.write_attribute(toast, name, ordinal, [ovstage.make_dltensor(m, dtype=f64x16, shape=[1])],
                                    semantic=ovstage.AttributeSemantic.MATRIX, is_array=False).wait()
    reset = ovstage.make_dltensor(np.ones(1, dtype=np.bool_), dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLBool, 8, 1), shape=[1])
    scene.stage.write_attribute(toast, "omni:fabric:resetXformStack", ordinal, [reset], is_array=False).wait()
    color = np.ascontiguousarray(np.asarray([v["th.toast_r"], v["th.toast_g"], v["th.toast_b"]], dtype=np.float32)[None])
    scene.stage.write_attribute(query(TOAST + TOAST_MATERIAL), "inputs:diffuseColor", ordinal,
                                [ovstage.make_dltensor(color, dtype=f32x3, shape=[1])], is_array=False).wait()
    k = 6.0 * v["ecu.heater_power"] / 900.0  # the apps' slot glow at full heater power
    glow = np.ascontiguousarray(np.asarray([k, 0.38 * k, 0.08 * k], dtype=np.float32)[None])
    scene.stage.write_attribute(query(TOASTER + GLOW_SHADER.removeprefix(ROOT)), "inputs:emissiveColor", ordinal,
                                [ovstage.make_dltensor(glow, dtype=f32x3, shape=[1])], is_array=False).wait()


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config()
    t = config.environment.toaster
    scenario = ToasterScenario.load(t.position, t.yaw)

    sim = CoSimulation(scenario.instances, scenario.initial)
    mon = MonitorRenderer()
    v = {}
    for k in range(int(40 / STEP)):
        tt = k * STEP
        sim.usd[LEVER][LINEAR] = -0.0595 if 1.0 <= tt < 1.5 else sim.usd[LEVER].get(LATCH, -0.0005)
        sim.usd[DIAL][ANGULAR] = 20.0
        sim.usd[CANCEL][LINEAR] = -0.0002
        sim.step(STEP)
        if k % 10 == 0:
            v = snapshot(sim)
            mon.record(v)
    sim.close()
    rects = mon.rects(v)
    print("quads per layer:", {n: len(r) for n, r in rects.items()}, "ECU state", v["ecu.state"])

    extras = monitor_extras(config.generated_dir, t.monitor_offset)
    place = scenario.placement_matrix()
    head = to_world(place, HEAD_LOCAL)[:3]
    p = to_world(place, extras["offset"])[:3]
    z = head - p
    z[2] = 0
    z /= np.linalg.norm(z)
    x = np.cross((0.0, 0.0, 1.0), z)
    panel = np.array([[*x, 0], [0, 0, 1, 0], [*z, 0], [*p, 1]])

    scene = SceneStage(config)
    origin = place[3, :3]
    ordinal = scene.clock.current()
    look = (origin + p) / 2 + (0, 0, 0.08)
    eye = look + 1.4 * (head - look)  # a little behind the seated eye, so the whole panel is in frame
    population.add_usd_reference_from_string(scene.stage, _render_layer(eye, look), "/Render")
    population.add_usd_reference_from_string(scene.stage, _placed(f"@{scenario.toaster_usd.as_posix()}@", place), TOASTER)
    population.add_usd_reference_from_string(scene.stage, _placed(f"@{Path(extras['layer']).as_posix()}@", panel), PRIM)
    population.apply_usd_changes(scene.stage, ordinal)
    scene.clock.commit()

    ordinal = scene.clock.current()
    for name, r in rects.items():
        budget, zl = ALL_LAYERS[name][2], extras["z"][name]
        pts = np.zeros((budget, 4, 3), np.float32)
        pts[:, :, 2] = zl
        n = min(budget, len(r))
        pts[:n, 0, :2] = r[:n][:, [0, 1]]
        pts[:n, 1, :2] = r[:n][:, [2, 1]]
        pts[:n, 2, :2] = r[:n][:, [2, 3]]
        pts[:n, 3, :2] = r[:n][:, [0, 3]]
        scene.write_points(f"{PRIM}/{name}", pts.reshape(-1, 3), ordinal)
    write_toast(scene, place, v, ordinal)  # the toast where the FMUs have it, so it matches the panel
    scene.clock.commit()
    seated = render(scene, ordinal, out_dir / "monitor_seated.png")
    scene.destroy()

    lit = int((seated.min(axis=2) > 170).sum())  # white text and grid on the dark plate
    print(f"bright pixels: {lit}")
    ok = lit > 800
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".cache/toaster_test")))
