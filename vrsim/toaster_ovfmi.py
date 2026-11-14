"""The toaster's FMUs run by NVIDIA's ovfmi (FmiHost), behind the interface of toaster.CoSimulation.

ovfmi discovers the FMUs from the station's USD-FMI binding, steps them with FMPy and routes values
by USD prim path and attribute. The sidecar has no ovstage: the apps own the physics, and their joint
readings arrive over the socket. So values go in through ``FmiHost.write`` (``AttributeWrite``) and
come out through ``FmiHost.read``, the host's documented data plane; the stage it is attached to is
only borrowed for ``update_from_ovstage`` / ``write_to_ovstage``, which are not used here.

What the VR experience needs beyond the authored binding is added the USD way, in a generated
overlay on the station (``vr_binding_layer``): output mappings for the values the device reacts to
(browning level, latch, toast mode) and the eject speed as authored parameter values, which ovfmi
passes to the FMU as start values.

ovfmi behaviour this works with:
- The first step after attaching or ``reset()`` initializes the FMUs from the authored start values;
  inputs written before it take effect from the second step on (10 ms later).
- Each step is a ``fmpy.simulate_fmu`` call (~0.5 ms for the three FMUs), so fast-forwarding uses
  larger steps (toaster.FAST_STEP); this model gives identical results at 10, 50 and 100 ms.
- Attaching parses the source in a helper process (~1.4 s), so one host is kept per scenario and
  ``reset()`` for every new toast.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Device-facing outputs the authored binding doesn't route: FMU variable -> attribute on SIM_PRIM.
VR_OUTPUTS = {"ToasterECU": ("browning_level", "latch_engaged"), "ToastFlight": ("toast_mode",)}
LAUNCH_ATTRS = {"v_launch[2]": "vr:launchVy", "v_launch[3]": "vr:launchVz"}  # ToastFlight parameters


class _BorrowedStage:
    """Stands in for the ovstage FmiHost borrows: this process has none, and write()/read() don't use it."""


def vr_binding_layer(station: Path, root: str, launch: dict[str, float] | None, directory: Path) -> Path:
    """The station plus the VR mappings, as a USD layer ovfmi can parse (it needs a file)."""
    sim = f"{root}/Simulation"
    launch = launch or {}
    lines = ["#usda 1.0", "(", f"    subLayers = [@{Path(station).resolve().as_posix()}@]", ")", "",
             f'over "{root.strip("/")}"', "{", '    over "Simulation"', "    {"]
    for var, attr in LAUNCH_ATTRS.items():
        if var in launch:
            lines.append(f"        double {attr} = {float(launch[var])!r}")

    def mapping(name: str, var: str, attr: str, direction: str) -> list[str]:
        return [f'                def FmuMapping "{name}"', "                {",
                f'                    string fmi:fmuAttribute = "{var}"',
                f'                    string fmi:usdAttribute = "{attr}"',
                f'                    token fmi:direction = "{direction}"',
                "                    int2 fmi:usdMapping = (0, 1)", "                }"]

    for instance in ("ToasterECU", "ToasterThermal", "ToastFlight"):
        body = []
        outputs = VR_OUTPUTS.get(instance, ())
        if outputs:
            body += ['            def FmuConnection "VrOutputs"', "            {",
                     f"                rel fmi:targets = <{sim}>"]
            for var in outputs:
                body += mapping(var, var, f"vr:{instance}:{var}", "output")
            body.append("            }")
        if instance == "ToastFlight" and any(v in launch for v in LAUNCH_ATTRS):
            body += ['            def FmuConnection "VrLaunch"', "            {",
                     f"                rel fmi:targets = <{sim}>"]
            for var, attr in LAUNCH_ATTRS.items():
                if var in launch:
                    body += mapping(var.replace("[", "_").replace("]", ""), var, attr, "input")
            body.append("            }")
        if body:
            lines += [f'        over "{instance}"', "        {", *body, "        }"]
    lines += ["    }", "}", ""]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "toaster_station_vr_ovfmi.usda"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class _FmuView:
    """``fmu(name).get(var)`` for the monitor and the device: a routed USD value, or the input written."""

    def __init__(self, sim: "OvfmiCoSimulation", instance):
        self.sim, self.name = sim, instance.path.rsplit("/", 1)[-1]
        self._routes = {m.fmu_var: (c.targets[0], m.usd_attr, m.offset)
                        for c in instance.connections for m in c.mappings if c.targets}

    def get(self, var: str) -> float:
        prim, attr, offset = self._routes[var]
        value = self.sim.usd.get(prim, {}).get(attr, 0.0)
        return float(value[offset] if isinstance(value, (list, tuple)) else value)


class OvfmiCoSimulation:
    """toaster.CoSimulation's interface (``usd``, ``t``, ``step``, ``fmu``, ``fmus``, ``close``) on ovfmi."""

    def __init__(self, station: Path, root: str, initial: dict[str, dict], launch: dict[str, float] | None = None):
        from ovfmi import FmiHost
        from pxr import Usd

        from .toaster import read_binding

        layer = vr_binding_layer(station, root, launch, Path(tempfile.gettempdir()) / "vrsim")
        self.instances = read_binding(Usd.Stage.Open(str(layer)))  # the same binding, for the value lookup
        self.host = FmiHost()
        report = self.host.attach_ovstage(_BorrowedStage(), source_asset=layer)
        log.info("ovfmi: %d FMU instance(s) from %s: %s", len(report.instances), layer.name,
                 ", ".join(i.prim_path.rsplit("/", 1)[-1] for i in report.instances))
        self._initial = {prim: dict(attrs) for prim, attrs in initial.items()}
        self._inputs = sorted({(c.targets[0], m.usd_attr) for i in self.instances for c in i.connections
                               for m in c.mappings if m.direction == "input" and c.targets
                               and not m.usd_attr.startswith("vr:launch")})
        self.fmus = [(_FmuView(self, i), i.connections) for i in self.instances]
        self.usd: dict[str, dict] = {}
        self.t = 0.0
        self.reset()

    def reset(self) -> None:
        """Back to t = 0 and the authored start values (a fresh toast), without parsing again."""
        self.host.reset()
        self.usd = {prim: dict(attrs) for prim, attrs in self._initial.items()}
        self.t = 0.0

    def fmu(self, name: str) -> _FmuView:
        return next(f for f, _ in self.fmus if f.name == name)

    def step(self, dt: float) -> None:
        from ovfmi import AttributeWrite

        by_attr: dict[str, list[tuple[str, float]]] = {}
        for prim, attr in self._inputs:
            value = self.usd.get(prim, {}).get(attr)
            if value is not None:
                by_attr.setdefault(attr, []).append((prim, value))
        writes = [AttributeWrite(tuple(p for p, _ in rows), attr, [[v] if np.isscalar(v) else list(v) for _, v in rows])
                  for attr, rows in by_attr.items()]
        if writes:
            self.host.write(writes)
        self.host.step_sync(dt)
        with self.host.read() as result:
            for group in result.groups:
                rows = np.asarray(group.tensors[0], dtype=np.float64).reshape(len(group.prim_paths), -1)
                for prim, row in zip(group.prim_paths, rows):
                    attrs = self.usd.setdefault(prim, {})
                    current = attrs.get(group.attribute_name)
                    if isinstance(current, list):  # a vector read back as its mapped leading components
                        merged = list(current)
                        merged[: row.size] = row.tolist()
                        attrs[group.attribute_name] = merged
                    else:
                        attrs[group.attribute_name] = float(row[0]) if row.size == 1 else row.tolist()
        self.t += dt

    def close(self) -> None:
        """Kept for the next toast (ToasterScenario caches it); ``release`` ends it."""

    def release(self) -> None:
        self.host.release()
