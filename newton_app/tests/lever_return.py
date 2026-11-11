"""Physics only: the toaster lever, held at its bottom stop and released, must spring back up.

    cd newton_app && .venv\Scripts\python tests\lever_return.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.bench_physics import scene  # noqa: E402
from vrnx.physics import Drive  # noqa: E402
from vrsim.mathutil import Pose  # noqa: E402


def main() -> int:
    p = scene()
    base = next(i for i, l in enumerate(p.labels) if l.endswith("body_obj_01"))
    lever = next(i for i, l in enumerate(p.labels) if l.endswith("lever_obj_01"))
    anchor = p.pose(base)

    def rel_z() -> float:
        return float((p.pose(base).inverse() * p.pose(lever)).p[2])

    p.step(72, [Drive(base, anchor, 1 / 72)])
    rest = rel_z()
    for _ in range(72):  # hold it 5.5 cm lower for 1 s, as the latch does
        target = Pose(p.pose(base).p, p.pose(base).q) * Pose((p.pose(base).inverse() * p.pose(lever)).p * (1, 1, 0) + (0, 0, rest - 0.06))
        p.step(1, [Drive(base, anchor, 1 / 72), Drive(lever, Pose(target.p, p.pose(lever).q), 1 / 72)])
    down = rel_z()
    for _ in range(72):
        p.step(1, [Drive(base, anchor, 1 / 72)])
    back = rel_z()
    print(f"lever rel z: rest {rest:.4f}, held {down:.4f}, 1 s after release {back:.4f}")
    ok = down < rest - 0.04 and back > rest - 0.005
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
