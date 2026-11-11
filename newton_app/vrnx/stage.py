"""The runtime owner: one ovstage, the ovrtx renderer attached to it, and the ordinal clock.

Following the realtime-viewer architecture, this module is the only one that creates the stage,
populates USD into it, hands out ordinals and seals them. Everything else (Newton physics through
ovnewton, ovxr's eye cameras, the app's controller/HUD/toaster visuals) writes runtime state into
this stage at the ordinal it is given; ovrtx renders the sealed result.

Ordinals (ovxr runs in ``StageOrdinalMode.APPLICATION``): ``current()`` hands out one ordinal and
keeps it until it is resolved, either by ``commit()`` (population changes between frames) or by
ovxr's ``bind_frame`` sealing it (the frame's writes). ovstage therefore never sees a gap.

Runtime writes go through the OVStage data plane:
- transforms: ``omni:fabric:localMatrix`` + ``worldMatrix`` + ``resetXformStack`` (the GPU
  hierarchy model only moves child prims with local + reset; see the repository's notes),
- mesh ``points`` for the HUD text and the toaster monitor,
- float tuples for material inputs and light intensity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import ovnewton
import ovrtx
import ovstage
from ovstage import PopulationDomain, population

log = logging.getLogger(__name__)

_F64x16 = ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 64, 16)
_BOOL = ovstage.DLDataType(ovstage.DLDataTypeCode.kDLBool, 8, 1)


class RuntimeStage:
    def __init__(self, environment_usd: Path, *, minimal_rendering: bool = False, name: str = "vrnx"):
        # Newton's and PhysX's schema definitions must be known before the first population, or
        # their attributes are dropped and ovnewton cannot import (or reject) them.
        ovnewton.register_usd_schemas()
        log.info("Creating renderer (first run compiles shaders; with Quest Link active this can take minutes)...")
        config = ovrtx.RendererConfig(enable_minimal=True, enable_realtimepathtracing=False,
                                      enable_pathtracing=False) if minimal_rendering else None
        self.renderer = ovrtx.Renderer(config)
        log.info("Render mode: %s", "Minimal" if minimal_rendering else "real-time path tracing (RT2)")
        # GPU hierarchy model: with the CPU-incremental default, ovrtx 0.6 applies a prim's transform
        # only on its first change, so per-frame writes never reach the image. ovnewton requires the
        # same setting for rendered stages.
        self.stage = ovstage.Stage(name, config=ovstage.StageConfig(
            runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL))
        self.renderer.attach_ovstage(self.stage)
        self.paths = ovstage.PathDictionary(self.stage)
        query = self.stage.get_attribute_write_floor()
        self._next = self.stage.fetch_ordinal(query) + 1
        self.stage.release_ordinal_query(query).wait()
        self._held: int | None = None
        self._queries: dict[str, tuple] = {}  # key -> (path list, query); the list must outlive the query

        # One population for every consumer: rendering data for ovrtx, physics data and composed
        # schema names for ovnewton (so unsupported physics is detected, not silently dropped).
        self.descs = (population.Desc(domains=PopulationDomain.RENDERING), ovnewton.population_desc())
        ordinal = self.current()
        population.open_usd_with_desc(self.stage, str(environment_usd), ordinal=ordinal,
                                      time=population.DEFAULT_TIME, descs=self.descs)
        self.commit()
        self.population_ordinal = ordinal
        log.info("Environment loaded at ordinal %d: %s", ordinal, environment_usd)

    # --- ordinals ----------------------------------------------------------------------------
    def current(self) -> int:
        if self._held is None:
            self._held = self._next
            self._next += 1
        return self._held

    def resolve(self) -> None:
        self._held = None

    def commit(self) -> int:
        """Seal population changes written at ``current()``."""
        ordinal = self.current()
        self.stage.advance_write_floor(ordinal).wait()
        self.resolve()
        return ordinal

    # --- population (between frames) ----------------------------------------------------------
    def add_reference(self, layer_text: str, prim_path: str) -> tuple[int, int]:
        """Reference inline USDA at ``prim_path`` and seal it. Returns (handle, ordinal)."""
        handle = population.add_usd_reference_from_string(self.stage, layer_text, prim_path)
        ordinal = self.current()
        population.apply_usd_changes(self.stage, ordinal)
        self.commit()
        return handle, ordinal

    def add_reference_file(self, usd: Path, prim_path: str) -> tuple[int, int]:
        handle = population.add_usd_reference(self.stage, Path(usd).as_posix(), prim_path)
        ordinal = self.current()
        population.apply_usd_changes(self.stage, ordinal)
        self.commit()
        return handle, ordinal

    def remove_references(self, handles: list[int]) -> int | None:
        if not handles:
            return None
        for handle in handles:
            population.remove_usd(self.stage, handle)
        ordinal = self.current()
        population.apply_usd_changes(self.stage, ordinal)
        self.commit()
        self._drop_queries()  # removed prims: cached queries may cover dead paths
        return ordinal

    # --- runtime writes (at the frame's ordinal) ------------------------------------------------
    def _query(self, prims: tuple[str, ...]):
        key = ";".join(prims)
        if key not in self._queries:
            plist = self.paths.create_path_list_from_strings(list(prims))
            self._queries[key] = (plist, self.stage.query_from_path_list(plist))
        return self._queries[key][1]

    def _drop_queries(self) -> None:
        for plist, query in self._queries.values():
            query.release()
            plist.release()
        self._queries.clear()

    def write_transforms(self, prims: list[str], matrices: np.ndarray, ordinal: int) -> None:
        """Row-major row-vector world matrices (N x 16) for prims, written as local + world with
        the xform stack reset."""
        if not prims:
            return
        m = np.ascontiguousarray(np.asarray(matrices, dtype=np.float64).reshape(len(prims), 16))
        t = ovstage.make_dltensor(m, dtype=_F64x16, shape=[len(prims)])
        reset = ovstage.make_dltensor(np.ones(len(prims), dtype=np.bool_), dtype=_BOOL, shape=[len(prims)])
        force = ovstage.AttributeMode.FORCE
        writes = [
            ovstage.WriteDesc("omni:fabric:localMatrix", [t], is_array=False,
                              semantic=ovstage.AttributeSemantic.MATRIX, attribute_mode=force),
            ovstage.WriteDesc("omni:fabric:worldMatrix", [t], is_array=False,
                              semantic=ovstage.AttributeSemantic.MATRIX, attribute_mode=force),
            ovstage.WriteDesc("omni:fabric:resetXformStack", [reset], is_array=False, attribute_mode=force),
        ]
        self.stage.write_attributes(self._query(tuple(prims)), writes, ordinal).wait()

    def write_points(self, prim: str, xyz: np.ndarray, ordinal: int) -> None:
        points = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32).reshape(-1, 3))
        t = ovstage.make_dltensor(points, dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 32, 3),
                                  shape=[len(points)])
        self.stage.write_attribute(self._query((prim,)), "points", ordinal, [t], is_array=True).wait()

    def write_floats(self, prim: str, attribute: str, values, ordinal: int) -> None:
        v = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(1, -1))
        t = ovstage.make_dltensor(v, dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 32, v.shape[1]),
                                  shape=[1])
        self.stage.write_attribute(self._query((prim,)), attribute, ordinal, [t], is_array=False,
                                   attribute_mode=ovstage.AttributeMode.FORCE).wait()

    def destroy(self) -> None:
        self._drop_queries()
        self.paths = None
        self.renderer.detach_ovstage() if hasattr(self.renderer, "detach_ovstage") else None
        self.renderer.destroy()
        self.stage.destroy()
