"""In-headset feedback panel for voice commands.

The panel is real scene geometry, so it needs no extra swapchain or compositor layer:

    /World/HUD/Panel            Xform, placed every frame (lazily follows the head)
        Backing                 dark plate
        Text                    Mesh with MAX_QUADS fixed quads; only ``points`` is rewritten
        AccentBusy/Ok/Error     colored status bar; the inactive ones are parked out of sight

Text is rasterized with Pillow and converted into merged pixel rectangles, one quad each. The
topology never changes: unused quads collapse to a point. A message change is therefore a single
``points`` array write to ovstage at the frame's ordinal.

``Feedback`` is the thread-safe message board. The voice thread, the spawner thread and the frame
loop all post to it.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .mathutil import Pose, mat_to_quat

log = logging.getLogger(__name__)

PANEL_PATH = "/World/HUD/Panel"
TEXT_PATH = PANEL_PATH + "/Text"
ACCENTS = {"busy": PANEL_PATH + "/AccentBusy", "ok": PANEL_PATH + "/AccentOk", "error": PANEL_PATH + "/AccentError"}
KIND_ACCENT = {"listening": "busy", "thinking": "busy", "info": "busy", "ok": "ok", "error": "error"}
HOLD_S = {"ok": 5.0, "error": 7.0, "info": 6.0, "listening": 30.0, "thinking": 30.0}

MAX_QUADS = 6000
PANEL_W, PANEL_H = 0.72, 0.25  # [m]
MARGIN_L, MARGIN_T = 0.05, 0.03  # [m]
PX_M = 0.0008  # metres per font pixel: 46 px title ~ 3.7 cm, legible at ~1 m
TITLE_PX, BODY_PX = 46, 36
DETAIL_LINES = 3
TEXT_Z = 0.003  # text floats just in front of the backing
ACCENT_OFFSET = Pose((-PANEL_W / 2 + 0.018, 0.0, TEXT_Z))
FOLLOW_DISTANCE, FOLLOW_DROP, FOLLOW_RATE = 0.85, 0.25, 3.0
FOLLOW_YAW = np.radians(35.0)  # off to the user's left, clear of the catalog and the table
# Close button in the panel's top-right corner (panel-local centre, half size) [m].
CLOSE_CENTER = (PANEL_W / 2 - 0.035, PANEL_H / 2 - 0.035)
CLOSE_HALF = 0.022

_FONT_DIR = Path("C:/Windows/Fonts")


@dataclass(frozen=True)
class Message:
    version: int
    kind: str  # listening | thinking | info | ok | error
    title: str
    detail: str
    posted: float
    hold: float


class Feedback:
    def __init__(self):
        self._lock = threading.Lock()
        self._msg = Message(0, "info", "", "", 0.0, 0.0)

    def post(self, kind: str, title: str, detail: str = "", hold: float | None = None) -> None:
        with self._lock:
            self._msg = Message(self._msg.version + 1, kind, title, detail, time.perf_counter(),
                                HOLD_S.get(kind, 6.0) if hold is None else hold)
        log.info("HUD [%s] %s%s", kind, title, f" | {detail}" if detail else "")

    def current(self) -> Message:
        with self._lock:
            return self._msg


def download_progress(feedback: Feedback, name: str, cancel: threading.Event | None = None,
                      interval: float = 0.4):
    """A catalog.fetch progress callback that posts throttled 'Downloading … 42%' HUD messages,
    with the ways to cancel; silent once ``cancel`` is set."""
    last = [0.0]
    lock = threading.Lock()

    def report(done: int, total: int, files: int, total_files: int) -> None:
        if cancel is not None and cancel.is_set():
            return
        now = time.perf_counter()
        with lock:
            if files < total_files and now - last[0] < interval:
                return
            last[0] = now
        pct = 100.0 * done / total if total else 100.0
        feedback.post("thinking", f"Downloading {name}… {pct:.0f}%",
                      f"{done / 1e6:.1f} / {total / 1e6:.1f} MB · say “cancel” or press × to stop")

    return report


# --- text -> quads ---------------------------------------------------------------------------
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    names = ("seguisb.ttf", "segoeui.ttf", "arialbd.ttf", "arial.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        for candidate in (_FONT_DIR / name, Path(name)):
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _mask_rects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover the True pixels with rectangles (x0, y0, x1, y1), exclusive ends: row runs, merged down."""
    h = mask.shape[0]
    open_runs: dict[tuple[int, int], int] = {}
    rects = []
    for y in range(h + 1):
        runs = set()
        if y < h:
            d = np.diff(np.concatenate(([0], mask[y].astype(np.int8), [0])))
            runs = set(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
        for run in [r for r in open_runs if r not in runs]:
            rects.append((run[0], open_runs.pop(run), run[1], y))
        for run in runs:
            open_runs.setdefault(run, y)
    return rects


class TextLayout:
    def __init__(self):
        self.title_font = _font(TITLE_PX, bold=True)
        self.body_font = _font(BODY_PX, bold=False)
        self.max_width_px = (PANEL_W - MARGIN_L - 0.09) / PX_M  # leave room for the close button

    def _fit(self, text: str, font) -> str:
        if font.getlength(text) <= self.max_width_px:
            return text
        while text and font.getlength(text + "…") > self.max_width_px:
            text = text[:-1]
        return text + "…"

    def _wrap(self, text: str, font, max_lines: int) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if font.getlength(trial) <= self.max_width_px:
                line = trial
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = self._fit(lines[-1] + " …", font)
        return [self._fit(line, font) for line in lines]

    def points(self, title: str, detail: str) -> np.ndarray:
        """Panel-local quad corners, shape (MAX_QUADS * 4, 3), float32; panel plane is XY, +Z faces the viewer."""
        rows = [(self._fit(title, self.title_font), self.title_font, TITLE_PX)]
        rows += [(line, self.body_font, BODY_PX) for line in self._wrap(detail, self.body_font, DETAIL_LINES)]

        quads = []
        x_left, y_cursor = -PANEL_W / 2 + MARGIN_L, PANEL_H / 2 - MARGIN_T
        for text, font, size in rows:
            ascent, descent = font.getmetrics()
            if text:
                img = Image.new("L", (int(font.getlength(text)) + 4, ascent + descent + 4), 0)
                ImageDraw.Draw(img).text((2, 2), text, font=font, fill=255)
                for x0, y0, x1, y1 in _mask_rects(np.asarray(img) >= 128):
                    quads.append((x_left + x0 * PX_M, y_cursor - y1 * PX_M,
                                  x_left + x1 * PX_M, y_cursor - y0 * PX_M))
            y_cursor -= (ascent + descent) * PX_M * 1.05

        if len(quads) > MAX_QUADS:
            log.debug("HUD text needs %d quads, truncated to %d", len(quads), MAX_QUADS)
            quads = quads[:MAX_QUADS]
        pts = np.zeros((MAX_QUADS, 4, 3), dtype=np.float32)
        pts[:, :, 2] = TEXT_Z
        if quads:
            q = np.asarray(quads, dtype=np.float32)
            pts[: len(q), 0, :2] = q[:, [0, 1]]
            pts[: len(q), 1, :2] = q[:, [2, 1]]
            pts[: len(q), 2, :2] = q[:, [2, 3]]
            pts[: len(q), 3, :2] = q[:, [0, 3]]
        return pts.reshape(-1, 3)


# --- USD authoring -------------------------------------------------------------------------
def _close_button(indent: str) -> str:
    """Red square with a white X (two thin rotated quads); plain meshes, so no texture is needed."""
    from .scene import _box, _mesh

    cx, cy = CLOSE_CENTER
    parts = [_mesh("CloseBox", *_box((cx, cy, TEXT_Z), (CLOSE_HALF, CLOSE_HALF, 0.002)),
                   "/World/Looks/HudError", indent=indent, cast_shadows=False)]
    length, width = CLOSE_HALF * 0.8, 0.004
    for i, angle in enumerate((np.pi / 4, -np.pi / 4)):
        ax, ay = np.cos(angle), np.sin(angle)  # stroke direction
        px, py = -ay, ax                         # stroke normal
        corners = [(cx + sx * length * ax + sy * width * px, cy + sx * length * ay + sy * width * py, TEXT_Z + 0.0025)
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        parts.append(_mesh(f"CloseStroke{i}", corners, [4], [0, 1, 2, 3], [(0, 0, 1)] * 4,
                           "/World/Looks/HudText", indent=indent, cast_shadows=False))
    return "\n".join(parts)


def _box_mesh(name: str, center, half, material: str, indent: str) -> str:
    from .scene import _box, _mesh  # shared geometry helpers

    return _mesh(name, *_box(center, half), material, indent=indent, cast_shadows=False)


def hud_usda(indent: str = "    ") -> str:
    """The /World/HUD prims, to be embedded in the environment layer."""
    i1, i2, i3 = indent, indent * 2, indent * 3
    accents = "\n".join(
        _box_mesh(path.rsplit("/", 1)[1], (0, 0, 0), (0.006, PANEL_H / 2 - 0.02, 0.002),
                  f"/World/Looks/Hud{name.capitalize()}", i3)
        for name, path in ACCENTS.items()
    )
    counts = ", ".join(["4"] * MAX_QUADS)
    indices = ", ".join(str(i) for i in range(MAX_QUADS * 4))
    points = ", ".join(["(0, 0, 0)"] * (MAX_QUADS * 4))
    return f'''{i1}def Xform "HUD"
{i1}{{
{i2}def Xform "Panel"
{i2}{{
{_box_mesh("Backing", (0, 0, -0.004), (PANEL_W / 2, PANEL_H / 2, 0.003), "/World/Looks/HudBacking", i3)}
{accents}
{_close_button(i3)}
{i3}def Mesh "Text"
{i3}{{
{i3}    int[] faceVertexCounts = [{counts}]
{i3}    int[] faceVertexIndices = [{indices}]
{i3}    point3f[] points = [{points}]
{i3}    normal3f[] normals = [(0, 0, 1)] (interpolation = "constant")
{i3}    float3[] extent = [({-PANEL_W / 2}, {-PANEL_H / 2}, 0), ({PANEL_W / 2}, {PANEL_H / 2}, {TEXT_Z * 2})]
{i3}    uniform bool doubleSided = 1
{i3}    bool primvars:doNotCastShadows = 1
{i3}    uniform token subdivisionScheme = "none"
{i3}    rel material:binding = </World/Looks/HudText>
{i3}}}
{i2}}}
{i1}}}'''


HUD_MATERIALS = [  # (name, diffuse, kwargs) for scene._material
    ("HudBacking", (0.02, 0.025, 0.035), {"roughness": 0.9, "emissive": (0.01, 0.012, 0.018)}),
    ("HudText", (1.0, 1.0, 1.0), {"roughness": 0.5, "emissive": (0.95, 0.95, 0.95)}),
    ("HudBusy", (0.2, 0.55, 1.0), {"roughness": 0.4, "emissive": (0.15, 0.45, 1.0)}),
    ("HudOk", (0.2, 0.9, 0.4), {"roughness": 0.4, "emissive": (0.15, 0.8, 0.3)}),
    ("HudError", (1.0, 0.25, 0.2), {"roughness": 0.4, "emissive": (1.0, 0.2, 0.15)}),
]


# --- runtime -------------------------------------------------------------------------------
class Hud:
    def __init__(self, feedback: Feedback):
        self.feedback = feedback
        self.layout = TextLayout()
        self._version = -1
        self._pos: np.ndarray | None = None
        self._last = time.perf_counter()

    def invalidate(self) -> None:
        """Force the text points to be produced again on the next update."""
        self._version = -1

    def update(self, head: Pose | None, hidden: list[float]) -> tuple[dict[str, list[float]], np.ndarray | None]:
        """Returns (prim world transforms, new text points or None if unchanged)."""
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        msg = self.feedback.current()

        points = None
        if msg.version != self._version:
            self._version = msg.version
            points = self.layout.points(msg.title, msg.detail)

        visible = bool(msg.title) and head is not None and now - msg.posted < msg.hold
        transforms = {path: hidden for path in ACCENTS.values()}
        if not visible:
            self._pos = None  # reappear in front of the user next time
            transforms[PANEL_PATH] = hidden
            return transforms, points

        fwd = head.axis((0, 0, -1))
        fwd[2] = 0.0
        norm = np.linalg.norm(fwd)
        fwd = fwd / norm if norm > 1e-3 else np.array([0.0, 1.0, 0.0])
        c, s_ = np.cos(FOLLOW_YAW), np.sin(FOLLOW_YAW)
        side = np.array([c * fwd[0] - s_ * fwd[1], s_ * fwd[0] + c * fwd[1], 0.0])
        target = head.p + side * FOLLOW_DISTANCE - np.array([0.0, 0.0, FOLLOW_DROP])
        # World-anchored while visible, so turning to read it doesn't push it away; it is re-placed
        # only on (re)appear or when it drifts more than ~70 degrees out of view.
        if self._pos is None:
            self._anchor = self._pos = target
        else:
            to_panel = self._pos - head.p
            to_panel[2] = 0.0
            norm = np.linalg.norm(to_panel)
            if norm > 1e-3 and float(to_panel @ fwd) / norm < np.cos(np.radians(70.0)):
                self._anchor = target
            self._pos = self._pos + (self._anchor - self._pos) * min(1.0, dt * FOLLOW_RATE)

        # Face the head, stay upright: x = right, y = world up, z = toward the viewer.
        z = head.p - self._pos
        z[2] = 0.0
        z = z / np.linalg.norm(z) if np.linalg.norm(z) > 1e-3 else -fwd
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z)
        panel = Pose(self._pos, mat_to_quat(np.column_stack([x, up, z])))
        transforms[PANEL_PATH] = panel.to_row_major()
        transforms[ACCENTS[KIND_ACCENT.get(msg.kind, "busy")]] = (panel * ACCENT_OFFSET).to_row_major()
        return transforms, points
