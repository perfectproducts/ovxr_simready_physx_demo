"""Headless check of vrnx.physics on the real scene: no headset, no renderer.

Spawns the toaster, the NVIDIA mug and a football the way the app does (ovstage references, then a
rebuild through ovnewton), lets them settle on the Studierstube desk, drives the mug to a hand
target, throws it, and reports rebuild and per-frame step times.

    cd newton_app && .venv\\Scripts\\python tests\\physics_headless.py
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vrnx  # noqa: E402,F401
import ovnewton  # noqa: E402
import ovstage  # noqa: E402
from ovstage import PopulationDomain, population  # noqa: E402

from vrnx.physics import Drive, NewtonPhysics  # noqa: E402
from vrsim.mathutil import Pose, quat_from_axis_angle  # noqa: E402

GEN = vrnx.REPO_ROOT / ".cache" / "generated"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def layer(asset_ref: str, pose: Pose) -> str:
    m = np.asarray(pose.to_row_major()).reshape(4, 4)
    rows = ", ".join("(" + ", ".join(f"{v:.9g}" for v in r) + ")" for r in m)
    return ('#usda 1.0\n(\n    defaultPrim = "Prop"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n'
            f'def Xform "Prop"\n{{\n    matrix4d xformOp:transform = ({rows})\n'
            '    uniform token[] xformOpOrder = ["xformOp:transform"]\n'
            f'    def "Asset" (prepend references = {asset_ref}) {{}}\n}}\n')


def main() -> int:
    wrappers = [next(GEN.glob(f"{stem}_*.usda")) for stem in
                ("prop_ToasterStation", "prop_desk_prop_nvidia_mug", "prop_football")]
    fixed = subprocess.run([sys.executable, "-m", "vrnx.usd_fixups", *map(str, wrappers)], check=True,
                           capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.split()
    toaster, mug, football = fixed

    ovnewton.register_usd_schemas()
    stage = ovstage.Stage("physics_headless", config=ovstage.StageConfig(
        runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL))
    population.open_usd_with_desc(stage, str(GEN / "environment_newton.usda"), ordinal=1,
                                  time=population.DEFAULT_TIME,
                                  descs=(population.Desc(domains=PopulationDomain.RENDERING), ovnewton.population_desc()))
    stage.advance_write_floor(1).wait()
    physics = NewtonPhysics(stage, str(GEN / "environment_physics.usda"))
    ordinal = 2
    physics.rebuild(1)

    spawns = [(toaster, Pose((-0.62, 0.64, 0.785), quat_from_axis_angle((0, 0, 1), math.radians(-90)))),
              (mug, Pose((-0.25, 0.55, 0.80))),
              (football, Pose((0.1, 0.5, 1.1)))]
    rebuild_ms = []
    for i, (path, pose) in enumerate(spawns):
        population.add_usd_reference_from_string(stage, layer(f"@{Path(path).as_posix()}@</Prop/Asset>", pose),
                                                 f"/World/Props/prop_{i:04d}")
        population.apply_usd_changes(stage, ordinal)
        stage.advance_write_floor(ordinal).wait()
        t0 = time.perf_counter()
        physics.rebuild(ordinal)
        rebuild_ms.append((time.perf_counter() - t0) * 1000)
        ordinal += 1

    def body(suffix: str) -> int:
        return next(i for i, label in enumerate(physics.labels) if label.endswith(suffix))

    mug_body, ball = body("prop_0001/Asset"), body("prop_0002/Asset/bodies/body_000")
    toaster_body = body("prop_0000/Asset/Geometry/body_obj_01")
    lever = body("prop_0000/Asset/Geometry/lever_obj_01")

    def run(frames: int, drives=()) -> float:
        times = []
        for _ in range(frames):
            t0 = time.perf_counter()
            physics.step(1, list(drives))
            physics.publish(ordinal_box[0])
            stage.advance_write_floor(ordinal_box[0]).wait()
            ordinal_box[0] += 1
            times.append(time.perf_counter() - t0)
        return float(np.median(times) * 1000)

    ordinal_box = [ordinal]
    # Fix the toaster where it stands (as the app does), let everything settle.
    anchor = physics.pose(toaster_body)
    settle_ms = run(144, [Drive(toaster_body, anchor, 1 / 72)])
    ok = True
    mug_z = physics.pose(mug_body).p[2]
    print(f"settled: mug z {mug_z:.3f} (desk 0.783), ball z {physics.pose(ball).p[2]:.3f}, "
          f"toaster z {physics.pose(toaster_body).p[2]:.3f}, lever rel z "
          f"{(physics.pose(toaster_body).inverse() * physics.pose(lever)).p[2]:.4f}")
    ok &= 0.77 < mug_z < 0.85

    # Grab: carry the mug to a hand 30 cm above the desk and hold it there.
    target = Pose(physics.pose(mug_body).p + (0, 0, 0.3), physics.pose(mug_body).q)
    run(72, [Drive(toaster_body, anchor, 1 / 72), Drive(mug_body, target, 1 / 72)])
    err = np.linalg.norm(physics.pose(mug_body).p - target.p)
    print(f"grab: mug {err * 1000:.1f} mm from the hand target")
    ok &= err < 0.01

    # Throw: release with 3 m/s forward, 1 m/s up.
    start = physics.pose(mug_body).p.copy()
    physics.set_velocities({mug_body: (np.array([0.0, -3.0, 1.0]), np.zeros(3))})
    run(144, [Drive(toaster_body, anchor, 1 / 72)])
    flown = np.linalg.norm((physics.pose(mug_body).p - start)[:2])
    print(f"throw: mug flew {flown:.2f} m, now z {physics.pose(mug_body).p[2]:.2f}")
    ok &= flown > 1.0

    step_ms = run(144, [Drive(toaster_body, anchor, 1 / 72)])
    print(f"rebuild per spawn: {', '.join(f'{t:.0f}' for t in rebuild_ms)} ms; "
          f"frame (step + publish + seal): median {step_ms:.2f} ms (settle {settle_ms:.2f} ms)")
    print("PASS" if ok else "FAIL")
    physics.destroy()
    stage.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
