"""Turn a downloaded SimReady package into something both the renderer and Newton can spawn.

Each asset gets a small wrapper layer that normalizes it to the app's Z-up, metres stage::

    def Xform "Prop" { xformOp:transform = <up-axis + metersPerUnit correction>
        def "Asset" ( references = @<package root usd>@ ) {}
    }

The same wrapper is referenced into the ovstage scene and parsed by ``ModelBuilder.add_usd``, so
rendered meshes and simulated bodies share one source of truth. Rigid bodies are tracked by their
prim path relative to the wrapper root (e.g. ``Asset/bodies/body_000``).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import newton
import scipy.spatial  # noqa: F401  Newton imports this lazily for hull remeshing; on Windows that
#                     DLL load deadlocks against the console thread blocked on stdin, so load it now.
from pxr import Gf, Usd, UsdGeom

from .catalog import CatalogEntry

log = logging.getLogger(__name__)

WRAPPER_ROOT = "/Prop"


class NotDynamicError(RuntimeError):
    """The asset has no rigid bodies, so it cannot be thrown around."""


@dataclass
class BodyInfo:
    rel_path: str  # prim path relative to the wrapper root, e.g. "Asset/bodies/body_000"
    local_index: int  # body index inside the asset's sub-builder
    scale: np.ndarray  # world scale of the body prim (Newton bodies carry no scale; the prim does)
    center: np.ndarray  # bounding-sphere center in the body frame [m]
    radius: float  # bounding-sphere radius [m]


@dataclass
class PreparedAsset:
    entry: CatalogEntry
    wrapper_path: Path
    builder: newton.ModelBuilder
    bodies: list[BodyInfo] = field(default_factory=list)
    bottom_z: float = 0.0  # lowest point relative to the wrapper origin [m]
    size: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def name(self) -> str:
        return self.entry.name


def _correction_matrix(stage: Usd.Stage) -> Gf.Matrix4d:
    scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    m = Gf.Matrix4d(1.0).SetScale(Gf.Vec3d(scale))
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        # Y-up -> Z-up: rotate +90 degrees about X.
        m = m * Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0))
    return m


def _asset_overs(stage: Usd.Stage, indent: str) -> str:
    """Overs under the default prim that make a package spawnable into our scene:

    - ``instanceable = false`` on the outermost instances. nvidia/simready-assets packages put each
      collision mesh inside an instanceable reference, and ovphysx 0.7 skips colliders in instance
      prototypes: the rigid body is simulated without a shape and falls through everything. Nested
      instances (materials) can stay instanced.
    - ``active = false`` on PhysicsScene prims. Some packages (ForgeCAD) ship their own scene, which
      would become a second PhysX scene with a CPU broadphase next to the app's.
    """
    root = stage.GetDefaultPrim()
    tree: dict = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstanceProxy():
            continue
        meta = ("instanceable = false" if prim.IsInstance() else
                "active = false" if prim.GetTypeName() == "PhysicsScene" else None)
        if meta:
            node = tree
            for name in prim.GetPath().MakeRelativePath(root.GetPath()).pathString.split("/"):
                node = node.setdefault(name, {})
            node[None] = meta  # this prim itself gets the metadata

    def emit(node: dict, depth: int) -> list[str]:
        lines, pad = [], indent + "    " * depth
        for name, child in node.items():
            if name is None:
                continue
            meta = f" (\n{pad}    {child[None]}\n{pad})" if child.get(None) else ""
            lines += [f'{pad}over "{name}"{meta}', f"{pad}{{", *emit(child, depth + 1), f"{pad}}}"]
        return lines

    return "\n".join(emit(tree, 0))


def _write_wrapper(root_usd: Path, out_dir: Path) -> Path:
    source = Usd.Stage.Open(str(root_usd))
    if not source.GetDefaultPrim():
        raise ValueError(f"{root_usd.name} has no defaultPrim to reference")
    m = _correction_matrix(source)
    overs = _asset_overs(source, "        ")
    rows = ", ".join("(" + ", ".join(f"{m[r][c]:.9g}" for c in range(4)) + ")" for r in range(4))
    digest = hashlib.sha1(str(root_usd).encode()).hexdigest()[:10]
    path = out_dir / f"prop_{root_usd.stem}_{digest}.usda"
    path.write_text(
        "#usda 1.0\n"
        '(\n    defaultPrim = "Prop"\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        'def Xform "Prop"\n{\n'
        f"    matrix4d xformOp:transform = ({rows})\n"
        '    uniform token[] xformOpOrder = ["xformOp:transform"]\n\n'
        f'    def "Asset" (\n        prepend references = @{root_usd.as_posix()}@\n    )\n    {{\n{overs}\n    }}\n'
        "}\n",
        encoding="utf-8",
    )
    return path


@dataclass
class AssetInfo:
    """What a stage-native simulator (the C++ app with ovphysx) needs to spawn a package."""

    root_usd: Path
    asset_ref: str  # USD reference to spawn under a prim: the wrapper's Asset prim (de-instanced)
    correction: list[float]  # 16 floats, row-major row-vector: package units/up-axis -> Z-up metres
    body_paths: list[str]  # rigid-body prims relative to the wrapper root, e.g. "Asset/bodies/body_000"
    bottom_z: float
    size: list[float]
    # Per body, its bounding sphere in the body frame [m]: cx, cy, cz, r. Hands pick the nearest body.
    body_spheres: list[list[float]]
    # Per body, whether it is a movable part: body1 of a non-fixed joint (a lid, a temple). While one
    # hand holds the asset, the other hand can grab such a part and move it against the joint.
    body_movable: list[bool]


def describe_asset(root_usd: Path, out_dir: Path) -> AssetInfo:
    """Like prepare_asset, but without building a Newton model (USD physics is read natively)."""
    from pxr import UsdPhysics

    wrapper = _write_wrapper(root_usd, out_dir)
    stage = Usd.Stage.Open(str(wrapper))
    prims = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not prims:
        raise NotDynamicError(f"'{root_usd.stem}' has no rigid bodies (static asset)")
    bodies = [str(p.GetPath()).removeprefix(WRAPPER_ROOT + "/") for p in prims]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    xf_cache = UsdGeom.XformCache()
    spheres = []
    for prim in prims:  # like prepare_asset's BodyInfo: local bound scaled to metres
        world_xf = xf_cache.GetLocalToWorldTransform(prim)
        scale = np.array([Gf.Vec3d(world_xf.GetRow3(i)).GetLength() for i in range(3)])
        local = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if local.IsEmpty():
            spheres.append([0.0, 0.0, 0.0, 0.1])
            continue
        lo, hi = np.array(local.GetMin()) * scale, np.array(local.GetMax()) * scale
        spheres.append([*((lo + hi) * 0.5).tolist(), max(float(np.linalg.norm(hi - lo) * 0.5), 0.02)])
    # Joint targets may name a mesh under the body: resolve to the nearest rigid-body ancestor.
    index = {p.GetPath(): i for i, p in enumerate(prims)}
    movable = [False] * len(prims)
    for joint in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not joint.IsA(UsdPhysics.Joint) or joint.IsA(UsdPhysics.FixedJoint):
            continue
        for target in UsdPhysics.Joint(joint).GetBody1Rel().GetTargets():
            while not target.isEmpty and target not in index:
                target = target.GetParentPath()
            if target in index:
                movable[index[target]] = True
    world = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(WRAPPER_ROOT)).ComputeAlignedRange()
    bottom, size = (0.0, [0.2] * 3) if world.IsEmpty() else (float(world.GetMin()[2]), list(world.GetSize()))
    m = _correction_matrix(Usd.Stage.Open(str(root_usd), load=Usd.Stage.LoadNone))
    correction = [float(m[r][c]) for r in range(4) for c in range(4)]
    asset_ref = f"@{wrapper.as_posix()}@<{WRAPPER_ROOT}/Asset>"
    return AssetInfo(root_usd, asset_ref, correction, bodies, bottom, [float(s) for s in size], spheres, movable)


def prepare_asset(entry: CatalogEntry, root_usd: Path, out_dir: Path) -> PreparedAsset:
    """Build the wrapper layer and a Newton sub-builder. Safe to call off the main thread."""
    wrapper = _write_wrapper(root_usd, out_dir)
    stage = Usd.Stage.Open(str(wrapper))

    builder = newton.ModelBuilder()
    result = builder.add_usd(str(wrapper), load_visual_shapes=False, skip_mesh_approximation=False)
    if builder.body_count == 0:
        raise NotDynamicError(f"'{entry.name}' has no rigid bodies (static asset)")

    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bodies = []
    for path, index in sorted(result["path_body_map"].items(), key=lambda kv: kv[1]):
        prim = stage.GetPrimAtPath(path)
        world = xf_cache.GetLocalToWorldTransform(prim)
        scale = np.array([Gf.Vec3d(world.GetRow3(i)).GetLength() for i in range(3)])
        local = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if local.IsEmpty():
            center, radius = np.zeros(3), 0.1
        else:
            lo, hi = np.array(local.GetMin()) * scale, np.array(local.GetMax()) * scale
            center, radius = (lo + hi) * 0.5, float(np.linalg.norm(hi - lo) * 0.5)
        bodies.append(BodyInfo(path.removeprefix(WRAPPER_ROOT + "/"), index, scale, center, max(radius, 0.02)))

    world_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(WRAPPER_ROOT)).ComputeAlignedRange()
    if world_range.IsEmpty():
        bottom, size = 0.0, np.full(3, 0.2)
    else:
        bottom = float(world_range.GetMin()[2])
        size = np.array(world_range.GetSize())

    mass = sum(builder.body_mass)
    log.info("Prepared '%s': %d bodies, %d shapes, %.2f kg, size %s m",
             entry.name, builder.body_count, builder.shape_count, mass, np.round(size, 2).tolist())
    return PreparedAsset(entry, wrapper, builder, bodies, bottom, size)
