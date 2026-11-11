"""The real app's frame loop without a headset: ovxr replaced by an offscreen stub, scripted hands.

Needs the sidecar running (python -m vrnx.sidecar --no-console): it spawns the NVIDIA mug and the
toaster on connect and runs the toaster's FMUs. The script then
  1. pushes the toaster lever down with the right hand and lets go: the ECU must latch it,
  2. lifts the mug with the left hand and throws it,
and renders the desk at the end. Everything between input and image is the app's own code path
(App.run: drives, Newton frame, ovnewton publish, data-plane writes, sealing).

    cd newton_app && .venv\\Scripts\\python tests\\app_headless.py [out.png]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vrnx  # noqa: E402,F401
import ovxr  # noqa: E402

from vrnx import app as appmod  # noqa: E402
from vrnx.xr_input import InputFrame  # noqa: E402
from vrsim.mathutil import Pose, quat_from_axis_angle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("app_headless")
W, H = 960, 540


class FakeContext:
    """Stands in for ovxr.Context: every frame begins, bind seals the ordinal like ovxr does."""

    def __init__(self, desc):
        self.rt_stage = desc.ovstage
        self.last_error = ""

    def probe_openxr(self):
        return SimpleNamespace(result=ovxr.Result.NOT_READY)

    def begin_frame(self):
        import time  # noqa: PLC0415  pace like a 72 Hz headset
        now = time.perf_counter()
        self._next = max(getattr(self, "_next", now), now - 0.05) + 1 / 72
        if self._next > now:
            time.sleep(self._next - now)
        return ovxr.Result.SUCCESS, SimpleNamespace(timing=SimpleNamespace(predicted_display_time_ns=0))

    def get_openxr_handles(self):
        return None

    def bind_frame(self, plan, dt, ordinal):
        self.rt_stage.advance_write_floor(ordinal).wait()
        return ovxr.Result.SUCCESS, True

    def submit_frame(self, plan):
        return ovxr.Result.SUCCESS, None

    def cancel_frame(self, plan):
        pass

    def destroy(self, stage_ordinal=None):
        pass


class ScriptedXr:
    """Plays a hand script; ``app`` is consulted for live body positions (where the lever is)."""

    def __init__(self, app):
        self.app = app
        self.frame = 0
        self.report = {}
        self.head = Pose((0.0, 0.0, 1.25), quat_from_axis_angle((1, 0, 0), math.pi / 2))  # looking along +Y
        self.lever_start = None
        self.mug_start = None

    def haptic(self, *a):
        pass

    def destroy(self):
        pass

    def _body(self, prop_name: str, suffix: str):
        for prop in self.app.props:
            if prop.name == prop_name:
                for b, path in enumerate(prop.body_paths):
                    if path.endswith(suffix):
                        return prop, b
        return None

    def sync(self, _t) -> InputFrame:
        f = InputFrame(focused=True, head=self.head)
        for h in f.hands:
            h.active, h.aim_valid = True, True
            h.grip = Pose((0.3, 0.2, 0.9))
            h.aim = Pose((0.3, 0.2, 0.9))
        f.hands[0].grip = Pose((-0.3, 0.2, 0.9))
        n, self.frame = self.frame, self.frame + 1
        app, ts = self.app, self.app.toaster
        right, left = f.hands[1], f.hands[0]

        # 1. Lever: hand to the lever, grip, push down 6 cm over 1 s, hold, let go.
        if ts is not None and 60 <= n < 260:
            ref = self._body("Smart toaster", "lever_obj_01")
            pose = app.body_pose(*ref) if ref else None
            if pose is not None and self.lever_start is None:
                center, _ = app.body_sphere(ref[0], ref[1], pose)
                self.lever_start = center
                self.report["lever_before"] = ts.lever.value
            if self.lever_start is not None:
                k = min(max((n - 80) / 72.0, 0.0), 1.0)
                right.grip = Pose(self.lever_start - (0, 0, 0.065 * k))
                right.squeeze = 1.0 if 70 <= n < 200 else 0.0
        if n == 199:
            self.report["lever_pushed"] = ts.lever.value if ts else None
            self.report["hand_on_lever"] = app.hands[1].prop is not None and not app.hands[1].whole
        if n == 330:
            self.report["lever_after_release"] = ts.lever.value if ts else None
            self.report["lever_target"] = ts.lever_target if ts else None

        # 2. Mug: grab, lift 30 cm, swing forward fast, let go.
        ref = self._body("NVIDIA mug", "Asset")
        if ref and 340 <= n < 520:
            pose = app.body_pose(*ref)
            if self.mug_start is None and pose is not None:
                self.mug_start = pose.p.copy()
            if self.mug_start is not None:
                lift = min(max((n - 360) / 60.0, 0.0), 1.0) * 0.3
                swing = max(n - 470, 0) * 0.04  # 0.04 m/frame = 2.9 m/s toward -Y (toward the user)
                left.grip = Pose(self.mug_start + (0, -swing, 0.05 + lift))
                left.squeeze = 1.0 if 350 <= n < 480 else 0.0
        if n == 420 and ref:
            self.report["mug_lifted_z"] = float(app.body_pose(*ref).p[2])
            self.report["mug_held"] = app.hands[0].prop is not None
        if n == 640 and ref:
            p = app.body_pose(*ref).p
            self.report["mug_thrown_distance"] = float(np.linalg.norm((p - self.mug_start)[:2]))
            self.report["mug_end"] = np.round(p, 3).tolist()
        # 3. After the cycle (10 s with the sidecar's default): latch released, lever back up.
        if ts is not None and n > 400 and ts.lever_target > ts.latched_below:
            self.report.setdefault("released_at", n)
            if n - self.report["released_at"] == 108:  # 1.5 s later
                self.report["lever_after_eject"] = ts.lever.value
                self.report["lever_after_eject_target"] = ts.lever_target
        # 4. The toast flies (higher), lands, a copy stays on the desk, a fresh slice is in the slot.
        if ts is not None and n > 400:
            self.report["toast_peak"] = max(self.report.get("toast_peak", 0.0), ts.toast_local[2])
        if "released_at" in self.report and n - self.report["released_at"] == 360:  # 5 s after the eject
            copies = [p for p in app.props if p.name == "Toast"]
            self.report["toast_copies"] = len(copies)
            if copies:
                pose = app.body_pose(copies[0], 0)
                self.report["toast_copy_z"] = float(pose.p[2]) if pose else None
            self.report["toast_in_slot"] = [round(v, 3) for v in ts.toast_local]
            self.report["toast_color_after"] = list(ts.toast_color)
        if n >= 660 and ("toast_copies" in self.report or n >= 1800):
            app.running = False
        return f


def render(app, out: Path) -> None:
    from tests.render_physics import look_at  # noqa: PLC0415

    eye, target = np.array([-0.35, 0.05, 1.25]), np.array([-0.45, 0.6, 0.85])
    layer = f"""#usda 1.0
(
    defaultPrim = "Render"
)
def Scope "Render"
{{
    def Camera "Camera"
    {{
        float focalLength = 16
        float2 clippingRange = (0.01, 100)
        float horizontalAperture = 20.955
        float verticalAperture = 11.787
        matrix4d xformOp:transform = {look_at(eye, target)}
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
    _, ordinal = app.rt.add_reference(layer, "/Render")
    image = None
    for _ in range(16):
        outputs = app.rt.renderer.step({"/Render/Product"}, 1.0 / 60.0, ordinal=ordinal)
        for product in outputs.values():
            for frame in product.frames:
                for var in frame.render_vars.values():
                    with var.map() as mapped:
                        image = np.array(np.from_dlpack(mapped), copy=True)
    Image.fromarray(image.reshape(H, W, -1)[..., :3].astype(np.uint8)).save(out)
    log.info("wrote %s", out)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else vrnx.REPO_ROOT / ".cache" / "newton_test" / "app_headless.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ovxr.Context = FakeContext
    args = argparse.Namespace(render="rt2", env=None, frames_in_flight=2, foveation=False, device="cuda:0",
                              substeps=2, port=int(os.environ.get("VRNX_TEST_PORT", appmod.SIDECAR_PORT)),
                              publish=sys.argv[2] if len(sys.argv) > 2 else "app")
    app = appmod.App(args)
    script = ScriptedXr(app)
    app.xr = script
    # Wait (inside the loop) for the sidecar's toaster and mug before the script starts.
    original_sync = script.sync

    def sync(t):
        if app.toaster is None or not any(p.name == "NVIDIA mug" for p in app.props):
            return InputFrame(focused=True, head=script.head)
        return original_sync(t)

    script.sync = sync
    try:
        app.run()
        render(app, out)
    finally:
        app.shutdown()
    r = script.report
    print("report:", r)
    checks = {
        "hand took the lever (part)": bool(r.get("hand_on_lever")),
        "lever pushed below the latch": (r.get("lever_pushed") or 0) < -0.03,
        "ECU latched the lever": (r.get("lever_target") or 0) < -0.03 and (r.get("lever_after_release") or 0) < -0.03,
        "mug lifted in hand": r.get("mug_held") and r.get("mug_lifted_z", 0) > 0.95,
        "mug thrown > 0.5 m": r.get("mug_thrown_distance", 0) > 0.5,
        "lever back up after the eject": r.get("lever_after_eject", -1) > -0.01,
        "toast rose > 0.8 m above the slot": r.get("toast_peak", 0) > 0.17 + 0.8,
        "toasted copy left on the desk": r.get("toast_copies", 0) == 1 and 0.75 < (r.get("toast_copy_z") or 0) < 0.85,
        # in the slot: y on the slot centre, z on the carriage (it rides on the lever, 5 mm below the top)
        "fresh toast back in the slot": r.get("toast_in_slot", [0, 0, 0])[1] == -0.026
                                        and abs(r.get("toast_in_slot", [0, 0, 0])[2] - 0.17) < 0.01,
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
