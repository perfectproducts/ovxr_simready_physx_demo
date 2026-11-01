"""Offscreen check of the HUD text path (no headset): ovstage ``points`` writes -> ovrtx render.

    uv run python tests/render_hud.py [out_dir]

Outside the frame loop nothing places the panel, so it sits at the world origin facing +Z with its
backing inside the floor. The test lifts the text 30 cm and views it from an angled camera, then
renders two different messages to PNGs to prove the text mesh updates in place.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import ovstage.population as population
from PIL import Image

from vrsim.config import Config
from vrsim.hud import TEXT_PATH, TextLayout
from vrsim.scene import SceneStage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

WIDTH, HEIGHT = 1280, 720
LIFT = 0.3


def _look_at(eye, target) -> str:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross((0.0, 0.0, 1.0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rows = [(*x, 0), (*y, 0), (*z, 0), (*eye, 1)]  # USD row-vector convention
    return "(" + ", ".join("(" + ", ".join(f"{v:.6f}" for v in r) + ")" for r in rows) + ")"


RENDER = f"""#usda 1.0
(
    defaultPrim = "Render"
)
def Scope "Render"
{{
    def Camera "Camera"
    {{
        float focalLength = 24
        float2 clippingRange = (0.01, 100)  # default near plane is 1 m, closer than the text
        float horizontalAperture = 20.955
        float verticalAperture = 11.787
        matrix4d xformOp:transform = {_look_at((0.0, -0.75, 0.75), (0.0, 0.0, LIFT))}
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


def render(scene: SceneStage, ordinal: int, path: Path) -> np.ndarray:
    image = None
    for _ in range(24):  # let temporal accumulation settle on the new text
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
    scene = SceneStage(Config())
    layout = TextLayout()

    population.add_usd_reference_from_string(scene.stage, RENDER, "/Render")
    population.apply_usd_changes(scene.stage, scene.clock.current())
    scene.clock.commit()

    messages = [("“add three mugs”", "Searching SimReady for 'mug'…"),
                ("No SimReady asset for 'rubber duck'", "Closest: dumbbell, beer stein, computer mouse")]
    images = []
    for i, (title, detail) in enumerate(messages):
        points = layout.points(title, detail)
        points[:, 2] += LIFT
        ordinal = scene.clock.current()
        scene.write_points(TEXT_PATH, points, ordinal)
        scene.clock.commit()
        images.append(render(scene, ordinal, out_dir / f"hud_{i}.png"))
    scene.destroy()

    bright = [int((img.min(axis=2) > 150).sum()) for img in images]  # white text on a dark floor
    changed = int((np.abs(images[0].astype(int) - images[1].astype(int)).sum(axis=2) > 90).sum())
    print(f"text pixels: {bright}, pixels changed between messages: {changed}")
    ok = all(b > 1000 for b in bright) and changed > 1000
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".cache/hud_test")))
