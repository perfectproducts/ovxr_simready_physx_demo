"""Live FMU monitor beside the toaster: the values and states of its three FMUs, with 60 s plots.

Built like the info panel (vrsim/hud.py): real scene geometry, no textures. Pillow draws the panel
once per colour; each colour's pixels become merged rectangles, one quad each, in a mesh with a fixed
quad budget. The app only rewrites ``points``.

    /World/ToasterMonitor            Xform; the app places it beside the toaster, turned to the head
        Backing, Stand               dark plate on a thin post down to the desk
        Text Dim Red Amber Cyan Green  one quad mesh per colour (LAYERS)
        Swatch                       quad in the toast's colour (the app writes its material)

    sidecar -> app   {"t":"toaster_monitor_frame","layers":{name: base64 float32 [x0,y0,x1,y1]*n}}
                     panel-local metres, +Z faces the viewer; only the layers that changed
    app -> sidecar   {"t":"toaster_monitor","on":bool}   (left menu button) - drawn only while shown

    uv run python -m vrsim.toaster_monitor   (renders a sample panel to .cache/toaster_monitor.png)
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .hud import _font, _mask_rects

PRIM = "/World/ToasterMonitor"
W_PX, H_PX = 400, 300
PX_M = 0.001  # [m] per pixel: the panel is 0.40 x 0.30 m
PANEL_W, PANEL_H = W_PX * PX_M, H_PX * PX_M
Z = 0.002  # quads float just in front of the backing; each colour layer a little further (no z-fighting)
Z_STEP = 0.0004
STAND_DROP = 0.04  # [m] panel bottom above the desk
HISTORY_S, SAMPLE_S = 60.0, 0.1

# name: (diffuse, emissive, quad budget)
LAYERS = {
    "Text": ((0.92, 0.94, 0.96), (0.85, 0.87, 0.9), 2000),
    "Dim": ((0.4, 0.43, 0.48), (0.28, 0.3, 0.34), 1600),
    "Red": ((1.0, 0.2, 0.1), (0.9, 0.12, 0.05), 900),
    "Amber": ((1.0, 0.68, 0.1), (0.9, 0.5, 0.04), 900),
    "Cyan": ((0.35, 0.78, 1.0), (0.25, 0.65, 0.95), 900),
    "Green": ((0.4, 0.92, 0.5), (0.3, 0.8, 0.38), 700),
}
COLS = (8, 138, 268)  # column x [px]
VALUE_DX = 62  # value column inside a column [px]
ROW0, ROW_H = 52, 16
SWATCH_PX = (COLS[1] + VALUE_DX, ROW0 + 3 * ROW_H + 3, COLS[1] + VALUE_DX + 40, ROW0 + 4 * ROW_H - 2)  # x0, y0, x1, y1
PLOTS = ((168, 222), (246, 282))  # y top, bottom [px]
PLOT_X = (34, 392)
ECU_STATES = {0: ("IDLE", "Text"), 1: ("HEATING", "Red"), 2: ("RELEASE", "Amber")}
FLIGHT_MODES = {0: ("in slot", "Text"), 1: ("flying", "Amber"), 2: ("landed", "Green")}


@dataclass
class Sample:
    t: float
    temp: float
    browning: float
    heater: float
    lever: float  # [m]
    toast_z: float  # [m]
    eject: float


def snapshot(sim) -> dict:
    """All monitored FMU variables of a vrsim.toaster.CoSimulation, read after a step."""
    ecu, th, fl = sim.fmu("ToasterECU"), sim.fmu("ToasterThermal"), sim.fmu("ToastFlight")
    v = {"t": sim.t}
    for prefix, fmu, names in (
        ("ecu", ecu, ("lever_pos", "dial_angle", "cancel_pos", "chamber_temp", "heater_power", "lever_target",
                      "eject", "time_remaining", "browning_level", "latch_engaged")),
        ("th", th, ("heater_power", "lever_pos", "chamber_temp", "browning", "toast_r", "toast_g", "toast_b")),
        ("fl", fl, ("lever_pos", "eject", "toast_x", "toast_y", "toast_z", "toast_rot", "toast_mode")),
    ):
        for n in names:
            v[f"{prefix}.{n}"] = fmu.get(n)
    latched, lever = v["ecu.latch_engaged"] > 0.5, v["ecu.lever_pos"]
    # The ECU's state variable isn't exposed; it follows from its outputs and the lever (ToasterECU.mo).
    v["ecu.state"] = 1 if latched else 2 if (v["ecu.eject"] > 0.5 or lever < -0.05) else 0
    return v


class MonitorRenderer:
    def __init__(self):
        self.title = _font(17, bold=True)
        self.head = _font(13, bold=True)
        self.body = _font(13, bold=False)
        self.small = _font(12, bold=False)
        self.history: deque[Sample] = deque(maxlen=int(HISTORY_S / SAMPLE_S) + 1)
        self._sent: dict[str, bytes] = {}

    def record(self, v: dict) -> None:
        if self.history and v["t"] - self.history[-1].t < SAMPLE_S - 1e-6:
            return
        self.history.append(Sample(v["t"], v["th.chamber_temp"], v["th.browning"], v["ecu.heater_power"],
                                   v["ecu.lever_pos"], v["fl.toast_z"], v["ecu.eject"]))

    def reset_sent(self) -> None:
        self._sent.clear()

    # --- drawing --------------------------------------------------------------------------------
    def draw(self, v: dict, history: list[Sample] | None = None) -> dict[str, Image.Image]:
        canvas = {name: Image.new("L", (W_PX, H_PX), 0) for name in LAYERS}
        d = {name: ImageDraw.Draw(img) for name, img in canvas.items()}

        def text(layer, xy, s, font=None):
            d[layer].text(xy, s, font=font or self.body, fill=255)

        def row(col, i, label, value, layer="Text"):
            x, y = COLS[col], ROW0 + i * ROW_H
            text("Dim", (x, y), label)
            text(layer, (x + VALUE_DX, y), value)

        text("Text", (8, 5), "Smart toaster · FMU co-simulation", self.title)
        t_str = f"t {v['t']:.0f} s"
        text("Dim", (W_PX - 8 - self.body.getlength(t_str), 9), t_str)

        for col, name in enumerate(("ToasterECU", "ToasterThermal", "ToastFlight")):
            x = COLS[col]
            text("Cyan", (x, 28), name, self.head)
            d["Dim"].line((x, 46, x + 124, 46), fill=255, width=1)

        state, layer = ECU_STATES[int(v["ecu.state"])]
        level = int(v["ecu.browning_level"])
        row(0, 0, "state", state, layer)
        row(0, 1, "level", f"{level} · {v['ecu.dial_angle'] % 360:.0f}°")
        row(0, 2, "left", f"{v['ecu.time_remaining']:.0f} s" if v["ecu.state"] == 1 else "–")
        heater = v["ecu.heater_power"]
        row(0, 3, "heater", f"{heater:.0f} W", "Red" if heater > 0 else "Text")
        row(0, 4, "latch", "engaged" if v["ecu.latch_engaged"] > 0.5 else "free",
            "Red" if v["ecu.latch_engaged"] > 0.5 else "Text")
        row(0, 5, "eject", "PULSE" if v["ecu.eject"] > 0.5 else "0", "Amber" if v["ecu.eject"] > 0.5 else "Text")

        temp = v["th.chamber_temp"]
        row(1, 0, "chamber", f"{temp:.0f} °C", "Red" if temp > 60 else "Text")
        row(1, 1, "browning", f"{v['th.browning']:.2f}", "Amber")
        row(1, 2, "toast", browning_word(v["th.browning"]))
        row(1, 3, "colour", "")
        d["Dim"].rectangle((SWATCH_PX[0] - 1, SWATCH_PX[1] - 1, SWATCH_PX[2] + 1, SWATCH_PX[3] + 1), outline=255)
        row(1, 4, "in: heat", f"{v['th.heater_power']:.0f} W")
        row(1, 5, "in: lever", _mm(v['th.lever_pos']), "Cyan")

        mode, layer = FLIGHT_MODES.get(int(v["fl.toast_mode"]), ("?", "Text"))
        row(2, 0, "mode", mode, layer)
        row(2, 1, "carriage", _mm(v['fl.lever_pos']), "Cyan")
        row(2, 2, "height", f"{v['fl.toast_z'] * 100:.1f} cm", "Green")
        row(2, 3, "out", f"{v['fl.toast_y'] * 100 + 2.6:.1f} cm")
        row(2, 4, "tumble", f"{v['fl.toast_rot'] % 360:.0f}°")
        row(2, 5, "in: eject", "PULSE" if v["fl.eject"] > 0.5 else "0", "Amber" if v["fl.eject"] > 0.5 else "Text")

        self._plots(d, text, v["t"], list(self.history) if history is None else history)
        return canvas

    def _plots(self, d, text, now: float, hist: list[Sample]) -> None:
        x0, x1 = PLOT_X
        ts = np.array([s.t for s in hist]) if hist else np.zeros(0)
        xs = x1 - (now - ts) / HISTORY_S * (x1 - x0)

        def frame(y0, y1, ticks):
            for value, y, label in ticks:
                d["Dim"].line((x0, y, x1, y), fill=255, width=1) if label else None
                d["Dim"].text((x0 - 4, y), label, font=self.small, fill=255, anchor="rm")
            d["Dim"].line((x0, y0, x0, y1), fill=255, width=1)
            d["Dim"].line((x0, y1, x1, y1), fill=255, width=1)

        def trace(layer, values, lo, hi, y0, y1):
            if len(values) < 2:
                return
            ys = y1 - (np.clip(values, lo, hi) - lo) / (hi - lo) * (y1 - y0)
            pts = [(float(x), float(y)) for x, y in zip(xs, ys) if x >= x0]
            if len(pts) >= 2:
                d[layer].line(pts, fill=255, width=2, joint="curve")

        # thermal: chamber 0..250 degC (red), browning 0..2 (amber), heater on as a bar on the axis
        y0, y1 = PLOTS[0]
        frame(y0, y1, [(t, y1 - t / 250 * (y1 - y0), f"{t}") for t in (100, 200)] + [(0, y1, "0")])
        text("Red", (x0, y0 - 15), "chamber °C", self.small)
        text("Amber", (x0 + 70, y0 - 15), "browning ×100", self.small)
        text("Red", (x0 + 160, y0 - 15), "▬ heater on", self.small)
        if hist:
            trace("Red", np.array([s.temp for s in hist]), 0, 250, y0, y1)
            trace("Amber", np.array([s.browning for s in hist]) * 100, 0, 250, y0, y1)
            on = np.array([s.heater > 0 for s in hist])
            for i in np.flatnonzero(on):
                if xs[i] >= x0:
                    d["Red"].rectangle((xs[i] - 1, y1 + 2, xs[i] + 2, y1 + 4), fill=255)

        # mechanics: lever -60..0 mm (cyan), toast height 0..30 cm (green), eject pulses (amber)
        y0, y1 = PLOTS[1]
        frame(y0, y1, [(0, y0, "up"), (0, y1, "down")])
        text("Cyan", (x0, y0 - 15), "lever / carriage", self.small)
        text("Green", (x0 + 96, y0 - 15), "toast height", self.small)
        text("Amber", (x0 + 176, y0 - 15), "| eject", self.small)
        if hist:
            trace("Cyan", np.array([s.lever for s in hist]), -0.06, 0.0, y0, y1)
            trace("Green", np.array([s.toast_z for s in hist]), 0.0, 0.30, y0, y1)
            ej = np.array([s.eject > 0.5 for s in hist])
            for i in np.flatnonzero(ej & ~np.r_[False, ej[:-1]]):
                if xs[i] >= x0:
                    d["Amber"].line((xs[i], y0, xs[i], y1), fill=255, width=1)
        text("Dim", (x0, y1 + 2), f"-{HISTORY_S:.0f} s", self.small)
        text("Dim", (x1 - self.small.getlength("now"), y1 + 2), "now", self.small)

    # --- quads ----------------------------------------------------------------------------------
    def rects(self, v: dict, history: list[Sample] | None = None) -> dict[str, np.ndarray]:
        """Panel-local rectangles per layer, (n, 4) float32 [x0, y0, x1, y1] in metres."""
        out = {}
        for name, img in self.draw(v, history).items():
            r = _mask_rects(np.asarray(img) >= 128)
            budget = LAYERS[name][2]
            r = np.asarray(r[:budget], dtype=np.float32).reshape(-1, 4)
            # pixel (x right, y down) -> panel metres (x right, y up), centred
            m = np.empty_like(r)
            m[:, 0] = r[:, 0] * PX_M - PANEL_W / 2
            m[:, 2] = r[:, 2] * PX_M - PANEL_W / 2
            m[:, 1] = PANEL_H / 2 - r[:, 3] * PX_M
            m[:, 3] = PANEL_H / 2 - r[:, 1] * PX_M
            out[name] = m
        return out

    def frame_message(self, v: dict, history: list[Sample] | None = None) -> dict | None:
        """The changed layers as a toaster_monitor_frame message, or None when nothing changed."""
        layers = {}
        for name, r in self.rects(v, history).items():
            raw = r.tobytes()
            if self._sent.get(name) != raw:
                layers[name] = base64.b64encode(raw).decode("ascii")
                self._sent[name] = raw
        return {"t": "toaster_monitor_frame", "layers": layers} if layers else None


def _mm(m: float) -> str:
    return f"{round(m * 1000) + 0:.0f} mm"


def browning_word(b: float) -> str:
    return ("untoasted" if b < 0.15 else "barely warm" if b < 0.5 else "light" if b < 0.85 else
            "golden" if b < 1.3 else "dark" if b < 1.7 else "burnt")


# --- USD -------------------------------------------------------------------------------------
def _flat_material(name: str, color, emissive) -> str:
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = </Monitor/Looks/{name}/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {tuple(color)}
                color3f inputs:emissiveColor = {tuple(emissive)}
                float inputs:roughness = 0.9
                token outputs:surface
            }}
        }}'''


def _quads_mesh(name: str, budget: int) -> str:
    counts = ", ".join(["4"] * budget)
    indices = ", ".join(str(i) for i in range(budget * 4))
    points = ", ".join(["(0, 0, 0)"] * (budget * 4))
    return f'''    def Mesh "{name}" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        point3f[] points = [{points}]
        normal3f[] normals = [(0, 0, 1)] (interpolation = "constant")
        float3[] extent = [({-PANEL_W / 2}, {-PANEL_H / 2}, 0), ({PANEL_W / 2}, {PANEL_H / 2}, {Z + Z_STEP * len(LAYERS)})]
        uniform bool doubleSided = 1
        bool primvars:doNotCastShadows = 1
        uniform token subdivisionScheme = "none"
        rel material:binding = </Monitor/Looks/{name}>
    }}'''


def _box(name: str, x0, y0, z0, x1, y1, z1, material: str) -> str:
    from .scene import _box as box, _mesh

    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    h = ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)
    return _mesh(name, *box(c, h), f"/Monitor/Looks/{material}", indent="    ", cast_shadows=False,
                 api_schemas=("MaterialBindingAPI",))


def swatch_rect() -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = SWATCH_PX
    return (x0 * PX_M - PANEL_W / 2, PANEL_H / 2 - y1 * PX_M, x1 * PX_M - PANEL_W / 2, PANEL_H / 2 - y0 * PX_M)


def monitor_usda() -> str:
    """The panel layer (defaultPrim Monitor), referenced by the app at PRIM. Panel origin: its centre."""
    sx0, sy0, sx1, sy1 = swatch_rect()
    bottom = -PANEL_H / 2
    prims = [
        _box("Backing", -PANEL_W / 2, -PANEL_H / 2, -0.006, PANEL_W / 2, PANEL_H / 2, -0.001, "Backing"),
        _box("Stand", -0.006, bottom - STAND_DROP, -0.012, 0.006, bottom, -0.006, "Stand"),
        _box("Foot", -0.05, bottom - STAND_DROP, -0.04, 0.05, bottom - STAND_DROP + 0.004, 0.02, "Stand"),
        _box("Swatch", sx0, sy0, 0.0, sx1, sy1, Z, "Swatch"),
    ] + [_quads_mesh(name, budget) for name, (_, _, budget) in LAYERS.items()]
    looks = [_flat_material("Backing", (0.025, 0.03, 0.04), (0.008, 0.01, 0.014)),
             _flat_material("Stand", (0.12, 0.12, 0.13), (0, 0, 0)),
             _flat_material("Swatch", (0.93, 0.85, 0.68), (0.4, 0.36, 0.28))]
    looks += [_flat_material(name, color, emissive) for name, (color, emissive, _) in LAYERS.items()]
    body, mats = "\n".join(prims), "\n".join(looks)
    return f'''#usda 1.0
(
    defaultPrim = "Monitor"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Monitor"
{{
{body}

    def Scope "Looks"
    {{
{mats}
    }}
}}
'''


def write_layer(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "toaster_monitor.usda"
    text = monitor_usda()
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path


def monitor_extras(directory: Path, offset, scale: float = 1.0) -> dict:
    """Spawn-message field for the app: where the layer is and how to fill and place it.

    ``offset`` is (x, y) in the toaster frame, or (x, y, h) to stand the panel on a surface h metres
    above the toaster's base (the books). ``scale`` enlarges the whole panel (the app scales its
    transform; the C++ app ignores it). With an (x, y) offset the centre moves away from the toaster
    by the extra half width, so the near edge stays where it was; the foot always stays on its surface."""
    x, y = float(offset[0]), float(offset[1])
    stand = float(offset[2]) if len(offset) > 2 else 0.0
    if scale != 1.0 and (x or y) and len(offset) == 2:
        grow = PANEL_W * (scale - 1.0) / 2
        x, y = x + grow * x / np.hypot(x, y), y + grow * y / np.hypot(x, y)
    return {
        "layer": str(write_layer(directory)),
        "layers": {name: budget for name, (_, _, budget) in LAYERS.items()},
        "z": {name: Z + i * Z_STEP for i, name in enumerate(LAYERS)},
        # panel centre in the toaster frame (z: above the desk top, which is the toaster's z = 0)
        "offset": [x, y, stand + (STAND_DROP + PANEL_H / 2) * scale],
        "scale": float(scale),
        "swatch": "Looks/Swatch/Surface",
    }


# --- offline preview -----------------------------------------------------------------------------
def _preview() -> None:
    import time

    from .config import PROJECT_ROOT
    from .toaster import ANGULAR, CANCEL, DIAL, LATCH, LEVER, LINEAR, STEP, CoSimulation, ToasterScenario

    scenario = ToasterScenario.load()
    sim = CoSimulation(scenario.instances, scenario.initial)
    mon = MonitorRenderer()
    v = {}
    for k in range(int(75 / STEP)):
        t = k * STEP
        sim.usd[LEVER][LINEAR] = -0.0595 if 1.0 <= t < 1.5 else sim.usd[LEVER].get(LATCH, -0.0005)
        sim.usd[DIAL][ANGULAR] = 20.0  # level 1: 60 s
        sim.usd[CANCEL][LINEAR] = -0.0002
        sim.step(STEP)
        if k % 10 == 0:
            v = snapshot(sim)
            mon.record(v)
    t0 = time.perf_counter()
    rects = mon.rects(v)
    dt = time.perf_counter() - t0
    img = Image.new("RGB", (W_PX, H_PX), (6, 8, 10))
    for name, layer in mon.draw(v).items():
        color = tuple(int(c * 255) for c in LAYERS[name][0])
        img.paste(Image.new("RGB", img.size, color), mask=layer.point(lambda p: 255 if p >= 128 else 0))
    out = PROJECT_ROOT / ".cache" / "toaster_monitor.png"
    out.parent.mkdir(exist_ok=True)
    img.resize((W_PX * 2, H_PX * 2), Image.NEAREST).save(out)
    print("render", f"{dt * 1000:.0f} ms", {n: (len(r), LAYERS[n][2]) for n, r in rects.items()}, "->", out)
    sim.close()


if __name__ == "__main__":
    _preview()
