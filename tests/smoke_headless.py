"""Headless check of the asset -> physics pipeline (no headset, no renderer).

    uv run python tests/smoke_headless.py mug football

Spawns each named asset above the table, lets it settle, then grabs the first one with a simulated
hand, carries it, and throws it. Exits non-zero if anything falls through the world.
"""

import logging
import sys

import numpy as np

from vrsim.assets import prepare_asset
from vrsim.catalog import SimReadyCatalog
from vrsim.config import Config
from vrsim.mathutil import Pose
from vrsim.physics import GrabTarget, PhysicsWorld

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx2").setLevel(logging.WARNING)


def main(queries: list[str]) -> int:
    cfg = Config()
    catalog = SimReadyCatalog(cfg).load()
    world = PhysicsWorld(cfg)
    table_top = cfg.table.height

    for i, query in enumerate(queries):
        entry, score = catalog.best_match(query) or (None, 0)
        if entry is None:
            print(f"no match for {query!r}")
            return 1
        asset = prepare_asset(entry, catalog.fetch(entry), cfg.generated_dir)
        spawn = Pose((cfg.table.center[0] - 0.3 + 0.3 * i, cfg.table.center[1], table_top + 0.3 - asset.bottom_z))
        prop = world.add_prop(asset, spawn)
        print(f"spawned {entry.name!r} (score {score:.0f}) as {prop.prim_root}")
        for _ in range(60):
            world.step(1 / 90, [])

    for _ in range(180):
        world.step(1 / 90, [])
    ok = True
    for prop in world.props:
        z = world.body_pose(prop.body_start).p[2]
        print(f"  {prop.asset.name:20s} rest z={z:.3f} (table top {table_top})")
        ok &= z > table_top - 0.05

    # Grab the first prop, lift it 40 cm over 0.5 s, then throw forward (+Y) at 4 m/s.
    body = world.props[0].body_start
    start = world.body_pose(body)
    hand = Pose(start.p, start.q)
    for k in range(45):
        hand = Pose(start.p + np.array([0.0, -0.3 * k / 45, 0.4 * k / 45]), start.q)
        world.step(1 / 90, [GrabTarget(body, hand)])
    held = world.body_pose(body).p
    print(f"  carried to {np.round(held, 3).tolist()}, hand at {np.round(hand.p, 3).tolist()}")
    ok &= np.linalg.norm(held - hand.p) < 0.05
    world.set_linear_velocity(body, np.array([0.0, 4.0, 2.0]))
    for _ in range(180):
        world.step(1 / 90, [])
    landed = world.body_pose(body).p
    print(f"  thrown, landed at {np.round(landed, 3).tolist()}")
    ok &= landed[1] > held[1] + 1.0 and landed[2] > -0.05

    print("prim transforms:", list(world.prim_transforms())[:3])
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["mug", "football"]))
