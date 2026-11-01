"""In-VR asset browser pages: SimReady thumbnails as a textured grid the app references into the scene.

Every SimReady package ships a render at ``<root dir>/.thumbs/256x256/<root file>.png``. For a page of
catalog entries we download those (small), composite each with its name into one tile image, and write
a USD layer with one textured quad per tile, a header, a dark backing and a highlight quad:

    /Browser                       (the app places this prim; panel plane is XY, +Z faces the viewer)
        Backing, Header            static quads
        Tile_00 .. Tile_11         textured quads, row-major from the top-left
        Highlight                  bright quad the app moves behind the hovered tile

The layout numbers are sent to the app with the layer so it can map a pointer hit to a tile.
"""

from __future__ import annotations

import hashlib
import logging
import math
import posixpath
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from .catalog import CatalogEntry
from .hud import _font

log = logging.getLogger(__name__)

COLS, ROWS = 4, 3
PAGE_SIZE = COLS * ROWS
TILE_W = 0.26  # [m]
TILE_H = TILE_W * 300 / 256  # tile image is 256 x 300 (thumbnail + label)
GAP = 0.03
HEADER_H = 0.07
GRID_W = COLS * TILE_W + (COLS - 1) * GAP
GRID_H = ROWS * TILE_H + (ROWS - 1) * GAP
# Grid top-left in panel coordinates; the panel origin is the grid centre.
GRID_ORIGIN = (-GRID_W / 2, GRID_H / 2)
HIGHLIGHT_PAD = 0.012
# Panel-local rectangles (x0, y0, x1, y1): a grab bar under the backing and a close button at the
# right end of the header row.
CLOSE_RECT = (-GRID_ORIGIN[0] - 0.07, GRID_ORIGIN[1] + 0.02, -GRID_ORIGIN[0], GRID_ORIGIN[1] + 0.02 + HEADER_H)
HANDLE_RECT = (-0.22, GRID_ORIGIN[1] - GRID_H - 0.11, 0.22, GRID_ORIGIN[1] - GRID_H - 0.05)


@dataclass
class Page:
    layer: Path
    entries: list[CatalogEntry]
    page: int
    pages: int
    title: str

    def message(self) -> dict:
        return {"t": "browser", "layer": self.layer.as_posix(), "cols": COLS, "rows": ROWS,
                "count": len(self.entries), "tile": [TILE_W, TILE_H], "gap": GAP,
                "grid_origin": list(GRID_ORIGIN), "highlight_pad": HIGHLIGHT_PAD,
                "page": self.page, "pages": self.pages, "title": self.title,
                "close": list(CLOSE_RECT), "handle": list(HANDLE_RECT)}


def thumbnail_path(entry: CatalogEntry) -> str:
    folder, name = posixpath.split(entry.root_usd)
    return f"{folder}/.thumbs/256x256/{name}.png"


class BrowserBuilder:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir / "browser"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.label_font = _font(22, bold=True)
        self.title_font = _font(40, bold=True)

    def _tile(self, entry: CatalogEntry) -> Path:
        key = hashlib.sha1(entry.key.encode()).hexdigest()[:12]
        out = self.out_dir / f"tile_{key}.png"
        if out.exists():
            return out
        tile = Image.new("RGB", (256, 300), (27, 31, 39))
        try:
            thumb = Image.open(hf_hub_download(entry.repo_id, thumbnail_path(entry), repo_type="dataset",
                                               revision=entry.revision))
            thumb = thumb.convert("RGBA").resize((256, 256), Image.LANCZOS)
            card = Image.new("RGBA", (256, 256), (205, 210, 218, 255))  # light card behind transparent renders
            card.alpha_composite(thumb)
            tile.paste(card.convert("RGB"), (0, 0))
        except Exception as exc:  # missing thumbnail: keep a labelled placeholder
            log.debug("No thumbnail for %s: %s", entry.name, exc)
            ImageDraw.Draw(tile).text((128, 128), "?", font=self.title_font, fill=(120, 130, 145), anchor="mm")
        name = entry.name
        while name and self.label_font.getlength(name) > 244:
            name = name[:-1]
        if name != entry.name:
            name = name.rstrip() + "…"
        ImageDraw.Draw(tile).text((128, 278), name, font=self.label_font, fill=(235, 238, 242), anchor="mm")
        tile.save(out)
        return out

    def _widget(self, name: str, size: tuple[int, int], color, label: str, font) -> Path:
        out = self.out_dir / f"{name}.png"
        if not out.exists():
            img = Image.new("RGB", size, color)
            ImageDraw.Draw(img).text((size[0] // 2, size[1] // 2), label, font=font, fill=(245, 246, 248),
                                     anchor="mm")
            img.save(out)
        return out

    def close_image(self) -> Path:
        return self._widget("close_button_v2", (96, 96), (200, 50, 45), "×", _font(80, bold=True))

    def handle_image(self) -> Path:
        return self._widget("move_handle_v2", (640, 88), (62, 68, 80), "grip here to move", _font(40, bold=True))

    def _header(self, title: str) -> Path:
        out = self.out_dir / f"header_{hashlib.sha1(title.encode()).hexdigest()[:12]}.png"
        if not out.exists():
            img = Image.new("RGB", (1024, round(1024 * HEADER_H / GRID_W)), (20, 24, 31))
            ImageDraw.Draw(img).text((16, img.height // 2), title, font=self.title_font, fill=(240, 242, 245),
                                     anchor="lm")
            img.save(out)
        return out

    def loading_layer(self) -> Path:
        """Static 'page loading' overlay: a dim quad over the grid plus a spinner ring (root /Loading).

        The app shows it at the browser pose and rotates /Loading/Spinner while a page is built.
        The ring's background matches the dim quad, so the rotating square's corners are invisible.
        """
        bg = (8, 9, 12)
        ring = self.out_dir / "spinner.png"
        if not ring.exists():
            img = Image.new("RGB", (256, 256), bg)
            ImageDraw.Draw(img).arc((28, 28, 228, 228), start=-90, end=180, fill=(90, 170, 255), width=22)
            img.save(ring)
        out = self.out_dir / "loading.usda"
        ox, oy = GRID_ORIGIN
        s = 0.09
        prims = "\n".join([
            _quad("Dim", ox, oy - GRID_H, -ox, oy, 0.004, "/Loading/Looks/Dim"),
            _quad("Spinner", -s, -s, s, s, 0.006, "/Loading/Looks/Spinner"),
        ]).replace("/Browser/", "/Loading/")
        looks = "\n".join([_flat_material("Dim", (0.03, 0.035, 0.045), (0.01, 0.012, 0.016)),
                           _textured_material("Spinner", ring)]).replace("/Browser/", "/Loading/")
        out.write_text(f'''#usda 1.0
(
    defaultPrim = "Loading"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Loading"
{{
{prims}

    def Scope "Looks"
    {{
{looks}
    }}
}}
''', encoding="utf-8")
        return out

    def prefetch(self, entries: list[CatalogEntry], page: int) -> None:
        """Download/compose the tiles of a page ahead of time (no layer, no message)."""
        chunk = entries[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self._tile, chunk))

    def build(self, entries: list[CatalogEntry], page: int, title: str) -> Page:
        pages = max(1, math.ceil(len(entries) / PAGE_SIZE))
        page = min(max(page, 0), pages - 1)
        chunk = entries[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        with ThreadPoolExecutor(max_workers=8) as pool:
            tiles = list(pool.map(self._tile, chunk))
        heading = f"{title}  ·  page {page + 1}/{pages}  ·  trigger: add · stick: page"
        header = self._header(heading)
        layer = self.out_dir / f"page_{hashlib.sha1((heading + ''.join(e.key for e in chunk)).encode()).hexdigest()[:12]}.usda"
        layer.write_text(_page_usda(tiles, header, self.close_image(), self.handle_image()), encoding="utf-8")
        return Page(layer, chunk, page, pages, title)


# --- USD authoring -----------------------------------------------------------------------------
def _quad(name: str, x0: float, y0: float, x1: float, y1: float, z: float, material: str, hidden: bool = False) -> str:
    # Hidden = parked below the floor at ~zero scale until the app places it.
    xform = ('\n        matrix4d xformOp:transform = ((0.0001, 0, 0, 0), (0, 0.0001, 0, 0), (0, 0, 0.0001, 0), (0, 0, -100, 1))'
             '\n        uniform token[] xformOpOrder = ["xformOp:transform"]') if hidden else ""
    return f'''    def Mesh "{name}"
    {{{xform}
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [({x0:.5f}, {y0:.5f}, {z:.5f}), ({x1:.5f}, {y0:.5f}, {z:.5f}), ({x1:.5f}, {y1:.5f}, {z:.5f}), ({x0:.5f}, {y1:.5f}, {z:.5f})]
        normal3f[] normals = [(0, 0, 1)] (interpolation = "constant")
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "vertex")
        float3[] extent = [({x0:.5f}, {y0:.5f}, {z:.5f}), ({x1:.5f}, {y1:.5f}, {z:.5f})]
        uniform bool doubleSided = 1
        bool primvars:doNotCastShadows = 1
        uniform token subdivisionScheme = "none"
        rel material:binding = <{material}>
    }}'''


def _textured_material(name: str, image: Path) -> str:
    # Emissive as well as diffuse so thumbnails read clearly regardless of the scene lighting.
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = </Browser/Looks/{name}/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Browser/Looks/{name}/Image.outputs:rgb>
                color3f inputs:emissiveColor.connect = </Browser/Looks/{name}/Image.outputs:rgb>
                float inputs:roughness = 0.9
                token outputs:surface
            }}
            def Shader "Image"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{image.as_posix()}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = </Browser/Looks/{name}/Uv.outputs:result>
                float3 outputs:rgb
            }}
            def Shader "Uv"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
        }}'''


def _flat_material(name: str, color, emissive) -> str:
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = </Browser/Looks/{name}/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {color}
                color3f inputs:emissiveColor = {emissive}
                float inputs:roughness = 0.9
                token outputs:surface
            }}
        }}'''


def _page_usda(tiles: list[Path], header: Path, close: Path, handle: Path) -> str:
    ox, oy = GRID_ORIGIN
    prims = [
        _quad("Backing", ox - 0.03, oy - GRID_H - 0.03, -ox + 0.03, oy + HEADER_H + 0.05, -0.006, "/Browser/Looks/Backing"),
        _quad("Header", ox, oy + 0.02, CLOSE_RECT[0] - 0.01, oy + 0.02 + HEADER_H, 0.0, "/Browser/Looks/Header"),
        _quad("Close", *CLOSE_RECT, 0.0, "/Browser/Looks/Close"),
        _quad("Handle", *HANDLE_RECT, 0.0, "/Browser/Looks/Handle"),
        # Unit-sized highlight centred on the origin; the app scales/places it behind the hovered tile.
        _quad("Highlight", -0.5, -0.5, 0.5, 0.5, -0.003, "/Browser/Looks/Highlight", hidden=True),
    ]
    materials = [_flat_material("Backing", (0.03, 0.035, 0.045), (0.01, 0.012, 0.016)),
                 _flat_material("Highlight", (0.25, 0.6, 1.0), (0.2, 0.5, 1.0)),
                 _textured_material("Header", header),
                 _textured_material("Close", close),
                 _textured_material("Handle", handle)]
    for i, image in enumerate(tiles):
        r, c = divmod(i, COLS)
        x0 = ox + c * (TILE_W + GAP)
        y1 = oy - r * (TILE_H + GAP)
        prims.append(_quad(f"Tile_{i:02d}", x0, y1 - TILE_H, x0 + TILE_W, y1, 0.0, f"/Browser/Looks/Tile_{i:02d}"))
        materials.append(_textured_material(f"Tile_{i:02d}", image))
    body = "\n".join(prims)
    looks = "\n".join(materials)
    return f'''#usda 1.0
(
    defaultPrim = "Browser"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Browser"
{{
{body}

    def Scope "Looks"
    {{
{looks}
    }}
}}
'''
