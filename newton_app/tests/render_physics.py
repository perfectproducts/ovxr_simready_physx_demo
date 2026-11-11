"""Offscreen check (no headset): Newton poses published by ovnewton reach the ovrtx image.

Renders the Studierstube desk with a mug held 25 cm above it, then lets Newton drop it for one
second, publishing each frame through ``binding.update_to_ovstage`` (``omni:xform``) and, for the
right-hand controller sphere, through the app's own ``write_transforms``. Both renders are saved;
the test passes when the mug's fall and the hand's move both change the image.

    cd newton_app && .venv\\Scripts\\python tests\\render_physics.py [out_dir]
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vrnx  # noqa: E402,F401

from vrnx.physics import NewtonPhysics  # noqa: E402
from vrnx.stage import RuntimeStage  # noqa: E402
from vrsim.mathutil import Pose  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
GEN = vrnx.REPO_ROOT / ".cache" / "generated"
W, H = 960, 540
MUG_AT = np.array([-0.25, 0.55, 0.783 + 0.25])


def look_at(eye, target) -> str:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross((0.0, 0.0, 1.0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return "(" + ", ".join("(" + ", ".join(f"{v:.6f}" for v in r) + ")"
                           for r in [(*x, 0), (*y, 0), (*z, 0), (*eye, 1)]) + ")"


def camera_layer() -> str:
    return f"""#usda 1.0
(
    defaultPrim = "Render"
)
def Scope "Render"
{{
    def Camera "Camera"
    {{
        float focalLength = 18
        float2 clippingRange = (0.01, 100)
        float horizontalAperture = 20.955
        float verticalAperture = 11.787
        matrix4d xformOp:transform = {look_at(MUG_AT + (0.0, -0.75, 0.05), MUG_AT - (0, 0, 0.15))}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
    def RenderProduct "Product"
    {{
        rel camera = </Render/Camera>
        uniform int2 resolution = ({W}, {H})
        rel orderedVars = </Render/LdrColor>
    }}
    def RenderVar "LdrColor"
    {{
        uniform string sourceName = "LdrColor"
    }}
}}
"""


def prop_layer(asset_ref: str, pose: Pose) -> str:
    m = np.asarray(pose.to_row_major()).reshape(4, 4)
    rows = ", ".join("(" + ", ".join(f"{v:.9g}" for v in r) + ")" for r in m)
    return ('#usda 1.0\n(\n    defaultPrim = "Prop"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n'
            f'def Xform "Prop"\n{{\n    matrix4d xformOp:transform = ({rows})\n'
            '    uniform token[] xformOpOrder = ["xformOp:transform"]\n'
            f'    def "Asset" (prepend references = {asset_ref}) {{}}\n}}\n')


def render(rt: RuntimeStage, ordinal: int, path: Path) -> np.ndarray:
    image = None
    for _ in range(16):
        outputs = rt.renderer.step({"/Render/Product"}, 1.0 / 60.0, ordinal=ordinal)  # keep alive while mapping
        for product in outputs.values():
            for frame in product.frames:
                for var in frame.render_vars.values():
                    with var.map() as mapped:
                        image = np.array(np.from_dlpack(mapped), copy=True)
    image = image.reshape(H, W, -1)[..., :3].astype(np.uint8)
    Image.fromarray(image).save(path)
    print("wrote", path)
    return image


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    mug = subprocess.run([sys.executable, "-m", "vrnx.usd_fixups", str(next(GEN.glob("prop_desk_prop_nvidia_mug_*.usda")))],
                         check=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.split()[0]
    rt = RuntimeStage(GEN / "environment_newton.usda")
    physics = NewtonPhysics(rt.stage, str(GEN / "environment_physics.usda"))
    rt.add_reference(camera_layer(), "/Render")
    _, ordinal = rt.add_reference(prop_layer(f"@{Path(mug).as_posix()}@</Prop/Asset>", Pose(MUG_AT)), "/World/Props/prop_0000")
    physics.rebuild(ordinal)

    hand = "/World/Hands/Right"
    o = rt.current()
    rt.write_transforms([hand], [Pose(MUG_AT + (0.18, 0, 0)).to_row_major()], o)
    rt.commit()
    before = render(rt, o, out / "newton_before.png")

    for _ in range(72):
        physics.step(1, [])
        o = rt.current()
        physics.publish(o)
        rt.write_transforms([hand], [Pose(MUG_AT + (0.18, 0, -0.2)).to_row_major()], o)
        rt.commit()
    after = render(rt, o, out / "newton_after.png")
    z = physics.pose(0).p[2]
    changed = int((np.abs(before.astype(int) - after.astype(int)).sum(axis=2) > 60).sum())
    print(f"mug body z {z:.3f} (desk top 0.783); pixels changed: {changed}")
    ok = z < 0.80 and changed > 3000
    print("PASS" if ok else "FAIL")
    physics.destroy()
    rt.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else vrnx.REPO_ROOT / ".cache" / "newton_test"))
