"""GPU time per physics frame for the app's startup scene (desk environment, toaster, mug).

    cd newton_app && .venv\\Scripts\\python tests\\bench_physics.py

Times the captured frame graph with CUDA events for a few configurations and splits one frame into
collision and solver time.
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vrnx  # noqa: E402,F401
import ovnewton  # noqa: E402
import ovstage  # noqa: E402
from ovstage import PopulationDomain, population  # noqa: E402

from tests.physics_headless import layer  # noqa: E402
from vrnx.physics import NewtonPhysics  # noqa: E402
from vrsim.mathutil import Pose, quat_from_axis_angle  # noqa: E402

GEN = vrnx.REPO_ROOT / ".cache" / "generated"
logging.basicConfig(level=logging.WARNING)


def scene(**kw) -> NewtonPhysics:
    wrappers = [next(GEN.glob(f"{s}_*.usda")) for s in ("prop_ToasterStation", "prop_desk_prop_nvidia_mug")]
    toaster, mug = subprocess.run([sys.executable, "-m", "vrnx.usd_fixups", *map(str, wrappers)], check=True,
                                  capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.split()
    stage = ovstage.Stage("bench", config=ovstage.StageConfig(
        runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL))
    population.open_usd_with_desc(stage, str(GEN / "environment_newton.usda"), ordinal=1, time=population.DEFAULT_TIME,
                                  descs=(population.Desc(domains=PopulationDomain.RENDERING), ovnewton.population_desc()))
    stage.advance_write_floor(1).wait()
    physics = NewtonPhysics(stage, str(GEN / "environment_physics.usda"), **kw)
    for i, (path, pose) in enumerate([(toaster, Pose((-0.62, 0.64, 0.785), quat_from_axis_angle((0, 0, 1), -math.pi / 2))),
                                      (mug, Pose((-0.25, 0.55, 0.785)))]):
        population.add_usd_reference_from_string(stage, layer(f"@{Path(path).as_posix()}@</Prop/Asset>", pose),
                                                 f"/World/Props/prop_{i:04d}")
        population.apply_usd_changes(stage, i + 2)
        stage.advance_write_floor(i + 2).wait()
    physics.rebuild(3)
    physics._stage_keepalive = stage
    return physics


def gpu_ms(physics: NewtonPhysics, frames: int = 200) -> float:
    times = []
    with wp.ScopedDevice(physics.device), wp.ScopedStream(physics.stream):
        for _ in range(frames):
            e0, e1 = wp.Event(physics.device, enable_timing=True), wp.Event(physics.device, enable_timing=True)
            wp.record_event(e0)
            wp.capture_launch(physics.graph, stream=physics.stream)
            wp.record_event(e1)
            wp.synchronize_event(e1)
            times.append(wp.get_event_elapsed_time(e0, e1))
    return float(np.median(times[20:]))


def split(physics: NewtonPhysics) -> tuple[float, float]:
    """Collision vs solver GPU time of one substep (no graph)."""
    col, sol = [], []
    with wp.ScopedDevice(physics.device), wp.ScopedStream(physics.stream):
        for _ in range(50):
            e = [wp.Event(physics.device, enable_timing=True) for _ in range(3)]
            wp.record_event(e[0])
            physics.collision.collide(physics.state_0, physics.contacts)
            wp.record_event(e[1])
            physics.solver.step(physics.state_0, physics.state_1, physics.control, physics.contacts, physics.dt / physics.substeps)
            physics.state_0, physics.state_1 = physics.state_1, physics.state_0
            wp.record_event(e[2])
            wp.synchronize_event(e[2])
            col.append(wp.get_event_elapsed_time(e[0], e[1]))
            sol.append(wp.get_event_elapsed_time(e[1], e[2]))
    return float(np.median(col[10:])), float(np.median(sol[10:]))


def main() -> None:
    configs = {
        "default (2 substeps, collide each, 8 iters, sap)": {},
        "collide once per frame": {"collide_every_substep": False},
        "4 solver iterations": {"iterations": 4},
        "nxn broad phase": {"broad_phase": "nxn"},
    }
    for name, kw in configs.items():
        physics = scene(**kw)
        c, s = split(physics)
        print(f"{name:52s} frame {gpu_ms(physics):5.2f} ms | one substep: collide {c:.2f} ms, solve {s:.2f} ms "
              f"| {physics.model.shape_count} shapes, contacts max {physics.contacts.rigid_contact_max}")


if __name__ == "__main__":
    main()
