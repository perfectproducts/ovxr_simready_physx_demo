"""Render-side scene: the ovrtx renderer, the ovstage it borrows, and stage ordinals.

ovxr runs in ``StageOrdinalMode.APPLICATION`` because the app co-writes the stage (prop
transforms every frame, prop references on spawn). ``OrdinalClock`` hands out one ordinal at a
time and keeps it until it is resolved, so ovstage never sees a gap: an ordinal is resolved either
by ``bind_frame`` sealing it or by ``advance_write_floor`` after a population change.

The environment (floor, table, lights, controller visuals) is generated from ``Config`` so the
rendered table always matches the physics table. Geometry is authored as explicit meshes.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import ovrtx
import ovstage
import ovstage.population as population

from .config import Config
from .hud import HUD_MATERIALS, hud_usda

log = logging.getLogger(__name__)

PROPS_ROOT = "/World/Props"
RIG_PATH = "/World/Rig"
HAND_PATHS = ("/World/Hands/Left", "/World/Hands/Right")
RAY_PATHS = ("/World/Hands/LeftRay", "/World/Hands/RightRay")




class OrdinalClock:
    def __init__(self, stage: ovstage.Stage):
        self.stage = stage
        query = stage.get_attribute_write_floor()
        self._next = stage.fetch_ordinal(query) + 1
        stage.release_ordinal_query(query).wait()
        self._held: int | None = None

    def current(self) -> int:
        """The ordinal to write at; stays the same until ``resolve()``."""
        if self._held is None:
            self._held = self._next
            self._next += 1
        return self._held

    def resolve(self) -> None:
        self._held = None

    def commit(self) -> None:
        """Publish population writes made at ``current()`` and resolve it."""
        self.stage.advance_write_floor(self.current()).wait()
        self.resolve()


# --- generated geometry ---------------------------------------------------------------------
def _fmt(values) -> str:
    return ", ".join(f"({', '.join(f'{c:.6g}' for c in v)})" if isinstance(v, (tuple, list)) else str(v)
                     for v in values)


def _mesh(name: str, points, counts, indices, normals, material: str, indent: str = "    ",
          xform: tuple | None = None, cast_shadows: bool = True, api_schemas: tuple = ()) -> str:
    quoted = ", ".join(f'"{s}"' for s in api_schemas)
    schemas = f" (prepend apiSchemas = [{quoted}])" if api_schemas else ""
    lines = [
        f'{indent}def Mesh "{name}"{schemas}',
        f"{indent}{{",
        f"{indent}    int[] faceVertexCounts = [{_fmt(counts)}]",
        f"{indent}    int[] faceVertexIndices = [{_fmt(indices)}]",
        f"{indent}    point3f[] points = [{_fmt(points)}]",
        f"{indent}    normal3f[] normals = [{_fmt(normals)}] (interpolation = \"faceVarying\")",
        f'{indent}    uniform token subdivisionScheme = "none"',
        f"{indent}    rel material:binding = <{material}>",
    ]
    if not cast_shadows:  # UI / pointer visuals should not darken the world
        lines.append(f"{indent}    bool primvars:doNotCastShadows = 1")
    if xform:
        lines.append(f"{indent}    double3 xformOp:translate = ({_fmt(xform)})")
        lines.append(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]')
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def _box(center, half) -> tuple:
    cx, cy, cz = center
    hx, hy, hz = half
    faces = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
             ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
             ((0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]),
             ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
             ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
             ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])]
    points, normals, indices = [], [], []
    for n, corners in faces:
        base = len(points)
        points += [(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sx, sy, sz in corners]
        normals += [n] * 4
        indices += [base, base + 1, base + 2, base + 3]
    return points, [4] * 6, indices, normals


def _sphere(radius: float, rings: int = 10, segments: int = 16) -> tuple:
    def vert(r, s):
        th, ph = math.pi * r / rings, 2 * math.pi * s / segments
        return (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))

    points, normals, indices, counts = [], [], [], []
    for r in range(rings):
        for s in range(segments):
            quad = [vert(r, s), vert(r + 1, s), vert(r + 1, s + 1), vert(r, s + 1)]
            base = len(points)
            points += [(x * radius, y * radius, z * radius) for x, y, z in quad]
            normals += quad
            indices += [base, base + 1, base + 2, base + 3]
            counts.append(4)
    return points, counts, indices, normals


def _material(name: str, color, roughness: float = 0.6, metallic: float = 0.0, emissive=(0, 0, 0)) -> str:
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = </World/Looks/{name}/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({_fmt(color)})
                color3f inputs:emissiveColor = ({_fmt(emissive)})
                float inputs:roughness = {roughness}
                float inputs:metallic = {metallic}
                token outputs:surface
            }}
        }}'''


def _physics_scene() -> str:
    # CPU simulation: a few dozen props don't need the GPU pipeline, and CPU scenes accept
    # velocity writes on every body type.
    return '''    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["PhysxSceneAPI"]
    )
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
        bool physxScene:enableGPUDynamics = 0
        token physxScene:broadphaseType = "MBP"
        uint physxScene:timeStepsPerSecond = 144
    }'''


def _authored_environment(config: Config, physics: bool, fake_sky: bool = False) -> str:
    """Reference the authored scene, moved so its standing spot is the tracking origin. Its own
    lights and camera are deactivated, and with ``physics`` every solid mesh gets a static convex
    collider (the scene carries no USD physics). With ``fake_sky`` (Minimal rendering) the
    ``shadowless_prims`` meshes stop casting shadows."""
    from fnmatch import fnmatch

    from pxr import Usd, UsdGeom  # usd-core; only needed for an authored environment

    env = config.environment
    stage = Usd.Stage.Open(str(env.usd))
    root = stage.GetDefaultPrim()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    overs: dict[tuple[str, ...], str] = {}
    shadowless: set[tuple[str, ...]] = set()
    if fake_sky:
        for child in root.GetChildren():
            if any(fnmatch(child.GetName(), pattern) for pattern in env.shadowless_prims):
                for prim in Usd.PrimRange(child):
                    if prim.IsA(UsdGeom.Mesh):
                        rel = tuple(prim.GetPath().MakeRelativePath(root.GetPath()).pathString.split("/"))
                        shadowless.add(rel)
                        overs.setdefault(rel, "")
    for name in env.hidden_prims:
        if root.GetChild(name):
            overs[(name,)] = "active = false"
    if physics:
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            size = cache.ComputeLocalBound(prim).ComputeAlignedRange().GetSize()
            if min(size) < 0.003:  # flat decals (screen text, coffee surface): no hull volume
                continue
            rel = prim.GetPath().MakeRelativePath(root.GetPath()).pathString.split("/")
            overs[tuple(rel)] = 'prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]'

    def tree(prefix: tuple[str, ...], depth: int) -> list[str]:
        lines, pad = [], "    " * depth
        children = sorted({key[len(prefix)] for key in overs if key[:len(prefix)] == prefix and len(key) > len(prefix)})
        for child in children:
            key = prefix + (child,)
            meta = overs.get(key, "")
            collider = "PhysicsCollisionAPI" in meta
            lines.append(f'{pad}over "{child}"' + (f" ({meta})" if meta else ""))
            lines.append(f"{pad}{{")
            if collider:
                lines.append(f'{pad}    uniform token physics:approximation = "convexHull"')
            if key in shadowless:
                lines.append(f"{pad}    bool primvars:doNotCastShadows = 1")
            if meta != "active = false":
                lines += tree(key, depth + 1)
            lines.append(f"{pad}}}")
        return lines

    ox, oy, oz = env.offset
    return f'''    def Xform "Environment" (
        prepend references = @{Path(env.usd).resolve().as_posix()}@
    )
    {{
        double3 xformOp:translate = ({ox}, {oy}, {oz})
        uniform token[] xformOpOrder = ["xformOp:translate"]
{chr(10).join(tree((), 2))}
    }}'''


def _toward(elevation: float, azimuth: float) -> str:
    """matrix4d whose +Z points toward (elevation, azimuth) [deg]: a DistantLight shines along -Z."""
    e, a = math.radians(elevation), math.radians(azimuth)
    z = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    x = np.cross((0.0, 0.0, 1.0), z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    return "(" + ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, 0)" for v in (x, y, z)) + ", (0, 0, 0, 1))"


MINIMAL_SOURCE_PRODUCT = "/World/Render/MinimalSource"


def _sky_palette(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """The backdrop image at sphere-face resolution, quantized: (rings x segments palette indices,
    linear RGB palette)."""
    from PIL import Image

    light = config.lighting
    img = Image.open(light.backdrop).convert("RGB").resize(
        (light.backdrop_segments, light.backdrop_rings), Image.LANCZOS)
    quantized = img.quantize(light.backdrop_colors, method=Image.Quantize.MEDIANCUT)
    srgb = np.asarray(quantized.getpalette()[:light.backdrop_colors * 3], dtype=np.float32).reshape(-1, 3) / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return np.asarray(quantized), linear


def _fake_sky(config: Config) -> str:
    """Stand-in for the dome under ovrtx Minimal rendering, which lights only with the first
    DistantLight: no dome, no other lights, and no working ambient term. Emission does show in
    its mode 3, but only as a constant colour (texture or primvar-driven emission comes out
    unlit/black). So the sky is a sphere of faces, each bound to one of N constant-emissive
    materials: the backdrop image quantized to a palette. It casts no shadows, so it doesn't block
    the sun from the scene inside it."""
    light = config.lighting
    rings, segments, r = light.backdrop_rings, light.backdrop_segments, light.backdrop_radius
    palette_index, _ = _sky_palette(config)
    points = []
    for i in range(rings + 1):
        for j in range(segments + 1):
            theta, u = math.pi * i / rings, j / segments
            # Image u grows clockwise seen from above; the sun's column lands at sun_azimuth.
            phi = math.radians(light.sun_azimuth) - (u - light.backdrop_sun_u) * 2 * math.pi
            points.append((r * math.sin(theta) * math.cos(phi), r * math.sin(theta) * math.sin(phi),
                           r * math.cos(theta)))
    indices, faces_by_color = [], {}
    for i in range(rings):
        for j in range(segments):
            a, b = i * (segments + 1) + j, (i + 1) * (segments + 1) + j
            indices += [a, a + 1, b + 1, b]  # wound to face the centre
            faces_by_color.setdefault(int(palette_index[i, j]), []).append(i * segments + j)
    subsets = "\n".join(f'''        def GeomSubset "c{c:03d}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [{_fmt(faces)}]
            rel material:binding = </World/Looks/Sky/c{c:03d}>
        }}''' for c, faces in sorted(faces_by_color.items()))
    return f'''    def Mesh "Backdrop" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [{_fmt([4] * (rings * segments))}]
        int[] faceVertexIndices = [{_fmt(indices)}]
        point3f[] points = [{_fmt(points)}]
        uniform token subdivisionScheme = "none"
        uniform token subsetFamily:materialBind:familyType = "partition"
        bool primvars:doNotCastShadows = 1
{subsets}
    }}

    def Scope "Render"
    {{
        # ovxr clones this for each eye (its source_render_product): mode 3 evaluates emission.
        # It needs exactly one camera target, or every ovxr_bind_frame fails ("camera relationship
        # ... has an invalid tensor shape") and the headset stays black; the clones get eye cameras.
        def Camera "MinimalSourceCamera"
        {{
        }}

        def RenderProduct "MinimalSource" (
            prepend apiSchemas = ["OmniRtxSettingsMinimalAPI_1"]
        )
        {{
            rel camera = </World/Render/MinimalSourceCamera>
            uniform int2 resolution = (1024, 1024)
            token omni:rtx:rendermode = "MinimalRendering"
            int omni:rtx:minimal:mode = 3
        }}
    }}'''


def _backdrop_material(config: Config) -> str:
    _, linear = _sky_palette(config)
    k = config.lighting.backdrop_brightness
    materials = "\n".join(f'''            def Material "c{c:03d}"
            {{
                token outputs:surface.connect = </World/Looks/Sky/c{c:03d}/Shader.outputs:surface>
                def Shader "Shader"
                {{
                    uniform token info:id = "UsdPreviewSurface"
                    color3f inputs:diffuseColor = (0, 0, 0)
                    color3f inputs:emissiveColor = ({rgb[0] * k:.5g}, {rgb[1] * k:.5g}, {rgb[2] * k:.5g})
                    token outputs:surface
                }}
            }}''' for c, rgb in enumerate(linear))
    return f'''        def Scope "Sky"
        {{
{materials}
        }}'''


def _lights(config: Config, fake_sky: bool = False) -> str:
    """Sky dome (the HDRI when configured: image-based light and backdrop) plus a sun for crisp
    shadows. ``fake_sky`` replaces the dome with _fake_sky for Minimal rendering."""
    light = config.lighting
    sun = _toward(light.sun_elevation, light.sun_azimuth)
    sun_light = f'''    def DistantLight "Sun"
    {{
        float inputs:intensity = {light.sun_intensity}
        float inputs:angle = 1.0
        matrix4d xformOp:transform = {sun}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}'''
    if fake_sky and light.backdrop:  # the sun first: Minimal mode uses only the first DistantLight
        return sun_light + "\n\n" + _fake_sky(config)
    if light.hdri:
        sky = f'''        asset inputs:texture:file = @{Path(light.hdri).resolve().as_posix()}@
        token inputs:texture:format = "latlong"
        float3 xformOp:rotateXYZ = (0, 0, {light.dome_rotation})
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]'''
    else:
        sky = "        color3f inputs:color = (0.85, 0.9, 1.0)"
    return f'''    def DomeLight "Sky"
    {{
        float inputs:intensity = {light.sky_intensity or (1000.0 if light.hdri else 60.0)}
{sky}
    }}

{sun_light}'''


def _loading_overlay(layer: Path | None) -> str:
    """The asset browser's page-loading overlay (vrsim.browser), parked out of sight until shown."""
    if layer is None:
        return ""
    return f'''    def Xform "BrowserLoading" (
        prepend references = @{layer.as_posix()}@
    )
    {{
        matrix4d xformOp:transform = ((0.0001, 0, 0, 0), (0, 0.0001, 0, 0), (0, 0, 0.0001, 0), (0, 0, -100, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}'''


def write_environment(config: Config, path: Path, physics: bool = False,
                      loading_layer: Path | None = None, fake_sky: bool = False) -> Path:
    """Generate the environment layer. ``physics`` adds a CPU PhysicsScene and static colliders on
    the floor and table top, for simulators that read USD physics from the stage (ovphysx).
    ``fake_sky`` lights it for ovrtx Minimal rendering, which ignores dome lights."""
    fake_sky = fake_sky and config.lighting.backdrop is not None
    colliders = ("PhysicsCollisionAPI",) if physics else ()
    t = config.table
    cx, cy, _ = t.center
    top = _box((cx, cy, t.height - t.top_thickness / 2), (t.size[0] / 2, t.size[1] / 2, t.top_thickness / 2))
    legs = []
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        lx, ly = cx + sx * (t.size[0] / 2 - 0.05), cy + sy * (t.size[1] / 2 - 0.05)
        legs.append(_mesh(f"Leg{i}", *_box((lx, ly, (t.height - t.top_thickness) / 2),
                                           (0.025, 0.025, (t.height - t.top_thickness) / 2)), "/World/Looks/TableLegs",
                          indent="        "))
    authored = config.environment.usd is not None
    ground = config.environment.ground_z if authored else 0.0
    # Wide enough to reach the horizon: past its edge a sky-only HDRI shows its blurred lower half.
    # Inside the fake sky's sphere (corners at 283 m < 300 m).
    floor = _box((0, 0, ground - 0.05), (200, 200, 0.05))
    palm = _sphere(0.03)
    # Pointer ray: unit length along local -Z (OpenXR "forward"); scaled per frame along Z.
    ray = _box((0, 0, -0.5), (0.003, 0.003, 0.5))

    table = f'''    def Xform "Table"
    {{
{_mesh("Top", *top, "/World/Looks/TableTop", indent="        ", api_schemas=colliders)}
{chr(10).join(legs)}
    }}'''

    def hand(name: str, material: str) -> str:
        return f'''        def Xform "{name}"
        {{
{_mesh("Palm", *palm, material, indent="            ", cast_shadows=False)}
        }}'''

    text = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "Rig"
    {{
    }}

{_lights(config, fake_sky)}

{_mesh("Floor", *floor, "/World/Looks/Floor", api_schemas=colliders)}
{_physics_scene() if physics else ""}

{_authored_environment(config, physics, fake_sky) if authored else table}

    def Xform "Hands"
    {{
{hand("Left", "/World/Looks/HandLeft")}
{hand("Right", "/World/Looks/HandRight")}
        def Xform "LeftRay"
        {{
{_mesh("Beam", *ray, "/World/Looks/Ray", indent="            ", cast_shadows=False)}
        }}
        def Xform "RightRay"
        {{
{_mesh("Beam", *ray, "/World/Looks/Ray", indent="            ", cast_shadows=False)}
        }}
    }}

{hud_usda()}
{_loading_overlay(loading_layer)}

    def Scope "Props"
    {{
    }}

    def Scope "Looks"
    {{
{_material("Floor", (0.2, 0.27, 0.14) if config.lighting.hdri or fake_sky else (0.32, 0.34, 0.37), roughness=0.95)}
{_material("TableTop", (0.55, 0.38, 0.22), roughness=0.45)}
{_material("TableLegs", (0.15, 0.15, 0.16), roughness=0.3, metallic=1.0)}
{_material("HandLeft", (0.1, 0.5, 1.0), roughness=0.3, emissive=(0.02, 0.1, 0.25))}
{_material("HandRight", (1.0, 0.45, 0.1), roughness=0.3, emissive=(0.25, 0.1, 0.02))}
{_material("Ray", (0.9, 0.95, 1.0), roughness=0.2, emissive=(0.6, 0.7, 0.9))}
{chr(10).join(_material(name, color, **kw) for name, color, kw in HUD_MATERIALS)}
{_backdrop_material(config) if fake_sky else ""}
    }}
}}
'''
    path.write_text(text, encoding="utf-8")
    return path


class SceneStage:
    def __init__(self, config: Config):
        self.config = config
        log.info("Creating renderer (first run compiles and caches shaders, this can take minutes)...")
        self.renderer = ovrtx.Renderer()
        # GPU hierarchy model: with the CPU-incremental default, ovrtx 0.6 picks up a prim's
        # transform only on its first change; later per-frame writes (physics bodies, controllers,
        # HUD) never reach the image. Verified offscreen with both ovxr and direct ovstage writes.
        self.stage = ovstage.Stage(config.application_name, config=ovstage.StageConfig(
            runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL))
        self.renderer.attach_ovstage(self.stage)
        self.clock = OrdinalClock(self.stage)

        env = write_environment(config, config.generated_dir / "environment.usda")
        population.open_usd(self.stage, str(env), ordinal=self.clock.current())
        self.clock.commit()
        log.info("Environment loaded: %s", env)
        self._paths = ovstage.PathDictionary(self.stage)
        self._point_queries: dict[str, tuple] = {}

    def write_points(self, prim_path: str, points: np.ndarray, ordinal: int) -> None:
        """Overwrite a mesh's ``points`` (same element count as authored) at ``ordinal``."""
        if prim_path not in self._point_queries:  # keep the path list alive as long as its query
            plist = self._paths.create_path_list_from_strings([prim_path])
            self._point_queries[prim_path] = (plist, self.stage.query_from_path_list(plist))
        query = self._point_queries[prim_path][1]
        points = np.ascontiguousarray(points, dtype=np.float32)
        tensor = ovstage.make_dltensor(
            points, dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 32, 3), shape=[len(points)])
        self.stage.write_attribute(query, "points", ordinal, [tensor], is_array=True).wait()

    def add_reference(self, usd_path: Path, prim_path: str) -> int:
        handle = population.add_usd_reference(self.stage, Path(usd_path).as_posix(), prim_path)
        population.apply_usd_changes(self.stage, self.clock.current())
        self.clock.commit()
        return handle

    def remove_references(self, handles: list[int]) -> None:
        if not handles:
            return
        for handle in handles:
            population.remove_usd(self.stage, handle)
        population.apply_usd_changes(self.stage, self.clock.current())
        self.clock.commit()

    def destroy(self) -> None:
        for plist, query in self._point_queries.values():
            query.release()
            plist.release()
        self._point_queries.clear()
        self.renderer.destroy()
        self.stage.destroy()
