"""Offscreen check of the toaster demo's runtime writes (no headset): ovstage writes -> ovrtx render.

    uv run python tests/render_toaster.py [out_dir]

Puts the dbeyond toaster and its toast on the Studierstube desk (where the sidecar spawns them),
then renders twice: the toast down in the slot and pale, then popped up and browned. The toast pose
goes in as a transform write and its colour as an ``inputs:diffuseColor`` write on its material,
the same writes the C++ app makes from the FMU outputs.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import ovstage
import ovstage.population as population
from PIL import Image

from vrsim.config import Config
from vrsim.scene import SceneStage
from vrsim.toaster import GLOW_SHADER, ROOT, TOAST_MATERIAL, ToasterScenario

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

WIDTH, HEIGHT = 1280, 720
TOASTER = "/World/TestToaster"
TOAST = TOASTER + "/Toast"  # the station layer's toast, below the toaster


def _look_at(eye, target) -> str:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross((0.0, 0.0, 1.0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rows = [(*x, 0), (*y, 0), (*z, 0), (*eye, 1)]  # USD row-vector convention
    return "(" + ", ".join("(" + ", ".join(f"{v:.6f}" for v in r) + ")" for r in rows) + ")"


def _render_layer(eye, target) -> str:
    return f"""#usda 1.0
(
    defaultPrim = "Render"
)
def Scope "Render"
{{
    def Camera "Camera"
    {{
        float focalLength = 24
        float2 clippingRange = (0.01, 100)
        float horizontalAperture = 20.955
        float verticalAperture = 11.787
        matrix4d xformOp:transform = {_look_at(eye, target)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
    def RenderProduct "Product"
    {{
        rel camera = </Render/Camera>
        uniform int2 resolution = ({WIDTH}, {HEIGHT})
        rel orderedVars = </Render/LdrColor>
    }}
    def RenderVar "LdrColor"
    {{
        uniform string sourceName = "LdrColor"
    }}
}}
"""


def _placed(layer_ref: str, matrix: np.ndarray) -> str:
    rows = "(" + ", ".join("(" + ", ".join(f"{v:.9g}" for v in r) + ")" for r in matrix) + ")"
    return f"""#usda 1.0
(
    defaultPrim = "P"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "P" (prepend references = {layer_ref})
{{
    matrix4d xformOp:transform = {rows}
    uniform token[] xformOpOrder = ["xformOp:transform"]
}}
"""


def render(scene: SceneStage, ordinal: int, path: Path) -> np.ndarray:
    image = None
    for _ in range(24):
        outputs = scene.renderer.step({"/Render/Product"}, 1.0 / 60.0, ordinal=ordinal)
        for product in outputs.values():
            for frame in product.frames:
                for var in frame.render_vars.values():
                    with var.map() as mapped:
                        image = np.array(np.from_dlpack(mapped), copy=True)
    if image is None:
        raise RuntimeError("renderer produced no LdrColor output")
    image = image.reshape(HEIGHT, WIDTH, -1)[..., :3].astype(np.uint8)
    Image.fromarray(image).save(path)
    print("wrote", path)
    return image


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config()
    scenario = ToasterScenario.load(config.environment.toaster.position, config.environment.toaster.yaw)
    scene = SceneStage(config)
    place = scenario.placement_matrix()  # toaster frame -> world, row-vector convention
    origin = place[3, :3]

    ordinal = scene.clock.current()
    population.add_usd_reference_from_string(scene.stage, _render_layer(origin + (0.15, -0.35, 0.45), origin + (0, 0, 0.12)), "/Render")
    population.add_usd_reference_from_string(scene.stage, _placed(f"@{scenario.toaster_usd.as_posix()}@", place), TOASTER)
    population.apply_usd_changes(scene.stage, ordinal)
    scene.clock.commit()

    paths = ovstage.PathDictionary(scene.stage)
    toast_list = paths.create_path_list_from_strings([TOAST])
    toast_query = scene.stage.query_from_path_list(toast_list)
    mat_list = paths.create_path_list_from_strings([TOAST + TOAST_MATERIAL])
    mat_query = scene.stage.query_from_path_list(mat_list)
    glow_list = paths.create_path_list_from_strings([TOASTER + GLOW_SHADER.removeprefix(ROOT)])
    glow_query = scene.stage.query_from_path_list(glow_list)

    def write(toast_local: np.ndarray, color, glow: float = 0.0) -> int:
        o = scene.clock.current()
        m = np.ascontiguousarray((toast_local @ place)[None], dtype=np.float64)
        t = ovstage.make_dltensor(m, dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 64, 16), shape=[1])
        reset = ovstage.make_dltensor(np.ones(1, dtype=np.bool_), dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLBool, 8, 1), shape=[1])
        for name in ("omni:fabric:localMatrix", "omni:fabric:worldMatrix"):
            scene.stage.write_attribute(toast_query, name, o, [t], semantic=ovstage.AttributeSemantic.MATRIX, is_array=False).wait()
        scene.stage.write_attribute(toast_query, "omni:fabric:resetXformStack", o, [reset], is_array=False).wait()
        c = np.ascontiguousarray(np.asarray(color, dtype=np.float32)[None])
        ct = ovstage.make_dltensor(c, dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 32, 3), shape=[1])
        scene.stage.write_attribute(mat_query, "inputs:diffuseColor", o, [ct], is_array=False).wait()
        k = 6.0 * glow  # cpp kGlowEmission x glow / glow_max
        g = np.ascontiguousarray(np.asarray([1.0 * k, 0.38 * k, 0.08 * k], dtype=np.float32)[None])
        gt = ovstage.make_dltensor(g, dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 32, 3), shape=[1])
        scene.stage.write_attribute(glow_query, "inputs:emissiveColor", o, [gt], is_array=False).wait()
        scene.clock.commit()
        return o

    def local(z: float) -> np.ndarray:
        m = np.eye(4)
        m[3, :3] = (0.0, -0.026, z)
        return m

    a = render(scene, write(local(0.17 - 0.059), (0.93, 0.85, 0.68), glow=1.0), out_dir / "toaster_down.png")
    b = render(scene, write(local(0.24), (0.45, 0.22, 0.06)), out_dir / "toaster_up_brown.png")
    scene.destroy()
    changed = int((np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 60).sum())
    print(f"pixels changed between the two states: {changed}")
    ok = changed > 2000
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".cache/toaster_test")))
