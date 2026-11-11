"""USD physics that PhysX accepts but ovnewton rejects, normalized in a small override layer.

ovnewton validates authored physics strictly instead of guessing (its support guide), so a few
SimReady packages need a non-destructive fix before they can be imported. The source package and
the sidecar's wrapper stay untouched: this writes ``newton_<wrapper>.usda``, which references the
wrapper's ``/Prop`` and overrides only what has to change.

- **The asset's Newton physics.** SimReady packages carry ``PhysX`` / ``MuJoCo`` / ``Newton`` variant
  sets. Where a ``Newton`` set exists, the override selects ``Newton = Enabled`` and turns the
  others off. The toaster's PhysX colliders are SDF triangle meshes (17k vertices on a moving body),
  which Newton would collide triangle by triangle; its Newton variant uses convex hulls.
- **Joint targets below a rigid body.** The dbeyond toaster's joints name a mesh under each body
  (``body_obj_01/body_mesh_01``). PhysX resolves the owning body; ovnewton raises
  ``InvalidPhysicsError``. The override retargets ``physics:body0/1`` to the rigid-body ancestor
  and re-expresses ``localPos/localRot`` in that body's frame, so the joint frame is unchanged.

Runs in the sidecar process (it uses ``pxr``), never in the process that owns ovstage + ovrtx.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

WRAPPER_ROOT = "/Prop"


def _fmt(values) -> str:
    return "(" + ", ".join(f"{float(v):.9g}" for v in values) + ")"


def _rigid_ancestor(stage, path, bodies: set):
    while not path.isEmpty and path not in bodies:
        path = path.GetParentPath()
    return None if path.isEmpty else path


def _joint_overs(stage) -> dict[str, list[str]]:
    """Per joint path (relative to /Prop): the override lines it needs."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    bodies = {p.GetPath() for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
              if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    xforms = UsdGeom.XformCache()
    overs: dict[str, list[str]] = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        lines = []
        for side, rel, pos_attr, rot_attr in (
                (0, joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                (1, joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr())):
            targets = rel.GetTargets()
            if len(targets) != 1 or targets[0] in bodies:
                continue
            body = _rigid_ancestor(stage, targets[0], bodies)
            if body is None:
                continue  # a static (world) target: ovnewton reports it if it matters
            target_world = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(targets[0]))
            body_world = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(body))
            target_in_body = target_world * body_world.GetInverse()  # row-vector: target -> body
            pos = Gf.Vec3d(pos_attr.Get() or Gf.Vec3f(0, 0, 0))
            rot = rot_attr.Get() or Gf.Quatf(1, 0, 0, 0)
            frame = Gf.Matrix4d().SetRotate(Gf.Quatd(rot)) * Gf.Matrix4d().SetTranslate(pos)
            frame_in_body = frame * target_in_body
            new_pos = frame_in_body.ExtractTranslation()
            ortho = Gf.Matrix4d(frame_in_body)
            ortho.Orthonormalize()
            q = ortho.ExtractRotationQuat()
            # Scale between the mesh and its body moves the anchor; express it in body units.
            lines += [f"rel physics:body{side} = <{body}>",
                      f"point3f physics:localPos{side} = {_fmt(new_pos)}",
                      f"quatf physics:localRot{side} = {_fmt((q.GetReal(), *q.GetImaginary()))}"]
            log.info("ovnewton fixup: %s body%d %s -> %s", prim.GetPath(), side, targets[0], body)
        if lines:
            overs[str(prim.GetPath())] = lines
    return overs


VARIANT_CHOICES = (("Newton", "Enabled"), ("PhysX", "Disabled"), ("MuJoCo", "Disabled"))


def _select_newton_variants(stage) -> dict[str, str]:
    """Select the Newton physics variant on the package's root prim (in the session layer)."""
    from pxr import Usd

    asset = stage.GetPrimAtPath(f"{WRAPPER_ROOT}/Asset")
    if not asset:
        return {}
    sets = asset.GetVariantSets()
    names = set(sets.GetNames())
    if "Newton" not in names or "Enabled" not in sets.GetVariantSet("Newton").GetVariantNames():
        return {}
    chosen = {}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, value in VARIANT_CHOICES:
            if name in names and value in sets.GetVariantSet(name).GetVariantNames():
                sets.GetVariantSet(name).SetVariantSelection(value)
                chosen[name] = value
    return chosen


def newton_wrapper(wrapper: Path, out_dir: Path) -> Path:
    """The wrapper as ovnewton can import it: itself when nothing needs fixing, else an override
    layer next to it. Reused while newer than the wrapper."""
    from pxr import Usd

    out = out_dir / f"newton_{wrapper.name}"
    if out.exists() and out.stat().st_mtime >= wrapper.stat().st_mtime and out.stat().st_mtime >= Path(__file__).stat().st_mtime:
        return out
    stage = Usd.Stage.Open(str(wrapper))
    variants = _select_newton_variants(stage)
    overs = _joint_overs(stage)
    if not overs and not variants:
        return wrapper

    tree: dict = {"Asset": {}}
    for path, lines in overs.items():
        node = tree
        for name in path.removeprefix(WRAPPER_ROOT + "/").split("/"):
            node = node.setdefault(name, {})
        node[None] = lines

    def emit(node: dict, depth: int) -> list[str]:
        pad, out_lines = "    " * depth, []
        for name, child in node.items():
            if name is None:
                continue
            meta = ""
            if depth == 1 and name == "Asset" and variants:
                choices = "; ".join(f'string {k} = "{v}"' for k, v in variants.items())
                meta = f" (\n{pad}    variants = {{ {choices} }}\n{pad})"
            out_lines += [f'{pad}over "{name}"{meta}', f"{pad}{{"]
            out_lines += [f"{pad}    {line}" for line in child.get(None, [])]
            out_lines += emit(child, depth + 1)
            out_lines.append(f"{pad}}}")
        return out_lines

    text = "\n".join([
        "#usda 1.0",
        '(\n    defaultPrim = "Prop"\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n    upAxis = "Z"\n)',
        "",
        f'def Xform "Prop" (\n    prepend references = @{wrapper.resolve().as_posix()}@</Prop>\n)',
        "{",
        *emit(tree, 1),
        "}",
        "",
    ])
    out.write_text(text, encoding="utf-8")
    log.info("ovnewton fixups for %s: variants %s, %d joints -> %s", wrapper.name, variants or "-", len(overs), out.name)
    return out


if __name__ == "__main__":  # python -m vrnx.usd_fixups <wrapper.usda>... : print the importable layers
    import sys

    logging.basicConfig(level=logging.INFO)
    for arg in sys.argv[1:]:
        print(newton_wrapper(Path(arg), Path(arg).parent).as_posix())
