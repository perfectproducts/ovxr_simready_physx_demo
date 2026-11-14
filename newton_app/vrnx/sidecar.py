"""The Newton app's sidecar: the C++ app's sidecar (vrsim.sidecar) plus what ovnewton needs.

Voice, catalog, downloads, the asset browser, HUD text and the toaster's FMUs are unchanged; the
app talks to this process with the same JSON-lines protocol (see vrsim/sidecar.py). Differences:

- It listens on port 47801, so the C++ app's sidecar can run at the same time.
- It writes the environment twice: ``environment_newton.usda`` for the render stage (no physics:
  ovnewton would re-import ~350 colliders on every spawn) and ``environment_physics.usda`` with the
  static colliders, which the app imports once into Newton (vrnx.physics).
- Every spawned asset goes through ``vrnx.usd_fixups.newton_wrapper``, which normalizes the USD
  physics PhysX tolerates but ovnewton rejects, in an override layer.
- Toaster defaults for VR: a 10 s cycle, the FMU monitor at twice its size, a higher eject, and after
  each eject the toasted slice stays on the desk (a grabbable copy) while a fresh one goes into the slot.

    python -m vrnx.sidecar [--mic NAME] [--prepare-only]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import vrnx  # noqa: F401  (puts the repository's vrsim package on sys.path)
from vrsim import sidecar as base
from vrsim.config import Config
from vrsim.scene import write_environment

from .usd_fixups import newton_wrapper

log = logging.getLogger("vrnx.sidecar")
PORT = 47801
DEFAULT_TOAST_SECONDS = 10.0  # a toasting cycle in VR: done after 10 s instead of 60-200 s
DEFAULT_MONITOR_SCALE = 2.0  # the FMU monitor at twice its 0.40 x 0.30 m, readable from the seat
# The monitor standing on the book stack behind and left of the toaster (world centre about
# (-0.82, 0.98), top 0.969 m): toaster frame x, y (+X toward the seat) and the stack's height above
# the toaster's base. At scale 2 the panel spans about 1.05-1.65 m, above the toaster.
DEFAULT_MONITOR_OFFSET = (-0.313, -0.240, 0.186)
DEFAULT_EJECT_HEIGHT = 0.9  # [m] the toast rises this far above the slot (authored: 0.46 m)
TOAST_LOOK = "Asset/Looks/ToastMat/Surface"  # the toast copy's shader, below its prop root


def write_toast_prop(config: Config) -> Path:
    """The toast slice as a free rigid body (30 g, convex-hull collider), for the copy left on the desk."""
    from vrsim.toaster import ToasterScenario

    slice_usd = ToasterScenario.toast_usd.fget(None)  # the station's Toast_Slice asset
    path = config.generated_dir / "toast_prop.usda"
    path.write_text(f'''#usda 1.0
(
    defaultPrim = "ToastProp"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
)

def Xform "ToastProp" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    prepend references = @{slice_usd.resolve().as_posix()}@
)
{{
    float physics:mass = 0.03
    over "Slice" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token physics:approximation = "convexHull"
    }}
}}
''', encoding="utf-8")
    return path


def write_environments(config: Config, loading_layer: Path | None) -> dict[str, Path]:
    gen = config.generated_dir
    out = {
        "render": write_environment(config, gen / "environment_newton.usda", physics=False, loading_layer=loading_layer),
        "render_minimal": write_environment(config, gen / "environment_newton_minimal.usda", physics=False,
                                            loading_layer=loading_layer, fake_sky=True),
        "physics": gen / "environment_physics.usda",
    }
    if not out["physics"].exists():
        write_environment(config, out["physics"], physics=True, loading_layer=loading_layer)
    return out


class NewtonSidecar(base.Sidecar):
    def __init__(self, config: Config, refresh: bool):
        super().__init__(config, refresh)
        self.environments = write_environments(config, self.browser.loading_layer())
        log.info("Environments for the Newton app: %s", {k: v.name for k, v in self.environments.items()})
        self.toast_prop = None
        if self.toaster is not None and config.environment.toaster.new_toast_after_eject:
            from vrsim.assets import describe_asset

            self.toast_prop = describe_asset(write_toast_prop(config), config.generated_dir)

    def _toast_landed(self) -> None:
        """Leave the toasted slice on the desk as a grabbable copy (the app places it where the
        FMU toast lies and paints it its browning colour), then reset the toaster with a fresh one.
        The copy's spawn goes out first, so the app still has the landed pose and colour."""
        if self.toast_prop is not None and self.toaster_shown:
            self._quiet_spawns += 1
            self.link.send({"t": "spawn", "name": "Toast", "count": 1, **self._spawn_payload(self.toast_prop),
                            "at_toast": True, "tint": {"prim": TOAST_LOOK, "attr": "inputs:diffuseColor"}})
        super()._toast_landed()

    def _on_app_message(self, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "talk_cancel":  # A + X: the press that started push-to-talk was a recenter chord
            voice = self.voice
            with voice._lock:
                voice._recording = False
                voice._chunks = []
        elif kind == "realigned":
            self.feedback.post("ok", "Room realigned", "the desk is in front of you again · A + X any time", hold=4.0)
        else:
            super()._on_app_message(msg)

    def _spawn_payload(self, info) -> dict:  # base: a staticmethod; here it also fixes the physics
        payload = base.Sidecar._spawn_payload(info)
        ref = payload["asset_ref"]  # "@<wrapper>@</Prop/Asset>"
        wrapper = Path(ref.split("@")[1])
        fixed = newton_wrapper(wrapper, self.config.generated_dir)
        if fixed != wrapper:
            payload["asset_ref"] = f"@{fixed.as_posix()}@</Prop/Asset>"
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice/catalog sidecar for the Newton VR app")
    parser.add_argument("--mic", help="Input device name or index")
    parser.add_argument("--speaker", help="Output device for 3D sound, name or index (default: the system default; or VRSIM_SPEAKER)")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--no-console", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="write the environments and exit")
    parser.add_argument("--port", type=int, default=PORT, help="TCP port the app connects to (default 47801)")
    parser.add_argument("--monitor-offset", type=lambda v: tuple(float(x) for x in v.split(",")), default=None,
                        help="FMU monitor position 'x,y[,h]' in the toaster frame [m], +x toward the seat, h = height of "
                             "the surface it stands on (default -0.313,-0.24,0.186: on the books; or VRSIM_MONITOR_OFFSET)")
    parser.add_argument("--monitor-scale", type=float, default=None,
                        help="size factor of the toaster's FMU monitor panel (default 2, or VRSIM_MONITOR_SCALE)")
    parser.add_argument("--eject-height", type=float, default=None,
                        help="how high the ejected toast rises above the slot [m] (default 0.9, or VRSIM_EJECT_HEIGHT; "
                             "0 = the FMU's authored 0.46 m)")
    parser.add_argument("--no-new-toast", action="store_true",
                        help="after an eject, leave the toaster as it is (default: fresh slice, toasted one stays on the desk)")
    parser.add_argument("--toast-seconds", type=float, default=None,
                        help="real seconds a toasting cycle takes, whatever the dial level (default 10, "
                             "or VRSIM_TOAST_SECONDS; 0 = the FMUs' real time, 60-200 s)")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx2", "httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = Config()
    toaster = config.environment.toaster
    # This app's toaster defaults, unless given on the command line or in the environment.
    for attr, value, env, default in (("toast_seconds", args.toast_seconds, "VRSIM_TOAST_SECONDS", DEFAULT_TOAST_SECONDS),
                                      ("monitor_scale", args.monitor_scale, "VRSIM_MONITOR_SCALE", DEFAULT_MONITOR_SCALE),
                                      ("eject_height", args.eject_height, "VRSIM_EJECT_HEIGHT", DEFAULT_EJECT_HEIGHT),
                                      ("monitor_offset", args.monitor_offset, "VRSIM_MONITOR_OFFSET", DEFAULT_MONITOR_OFFSET)):
        if value is not None:
            setattr(toaster, attr, value)
        elif not os.environ.get(env):
            setattr(toaster, attr, default)
    if args.no_new_toast:
        toaster.new_toast_after_eject = False
    elif not os.environ.get("VRSIM_NEW_TOAST"):
        toaster.new_toast_after_eject = True
    if args.mic:
        config.mic_device = int(args.mic) if args.mic.isdigit() else args.mic
    if args.speaker:
        config.speaker_device = int(args.speaker) if args.speaker.isdigit() else args.speaker
    if args.prepare_only:
        from vrsim.browser import BrowserBuilder

        envs = write_environments(config, BrowserBuilder(config.generated_dir).loading_layer())
        for kind, path in envs.items():
            print(f"{kind}: {path}")
        return
    base.PORT = args.port  # Sidecar.serve() binds base.PORT
    sidecar = NewtonSidecar(config, refresh=args.refresh_catalog)
    if not args.no_console:
        base.start_console(sidecar.commands)
    sidecar.serve()


if __name__ == "__main__":
    main()
