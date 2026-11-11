// The shared ovstage: renderer attachment, population, ordinals and per-frame transform writes.
#pragma once

#include "common.hpp"

#include <ovrtx/ovrtx.h>
#include <ovstage/ovstage.h>
#include <ovstage/ovstage_population.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace vrsim
{

class Stage
{
public:
    // Must run before anything else touches ovstage: the hierarchy model is process-scoped.
    static void initializeProcess();
    static void shutdownProcess();

    // minimalRendering: create the renderer with only ovrtx's Minimal mode enabled; products that
    // request real-time path tracing (RT2) then fall back to it. Much cheaper per frame.
    Stage(const std::string& environmentUsd, const std::string& physxSchemasRoot, bool minimalRendering = false);
    ~Stage();

    ovstage_instance_t* handle() const { return m_stage; }
    ovrtx_renderer_t* renderer() const { return m_renderer; }
    path_dictionary_instance_t* paths() const { return m_paths; }

    // One ordinal at a time; it stays current until resolve(). Every lower ordinal is resolved.
    ovstage_ordinal_t current();
    void resolve() { m_held = 0; }
    // Seal the current ordinal (population changes) and resolve it.
    ovstage_ordinal_t commit();

    // Reference a USDA layer at primPath, publish it and return the reference handle.
    ovstage_population_usd_reference_handle_t addReference(const std::string& layer, const std::string& primPath,
                                                           ovstage_ordinal_t ordinal);
    void removeReference(ovstage_population_usd_reference_handle_t handle, ovstage_ordinal_t ordinal);

    ovx_primpath_t internPath(const std::string& path);
    ovx_primpath_list_t pathList(const std::vector<std::string>& paths);

    // World transforms for prims in `list` order, authored like ovxr's prim transforms (local == world
    // plus resetXformStack) so the GPU hierarchy update carries children (meshes under bodies) along.
    void writeTransforms(ovx_primpath_list_t list, const std::vector<Mat16>& matrices, ovstage_ordinal_t ordinal);
    // Same, for named prims (controllers, rays, HUD).
    void writeTransforms(const std::vector<std::string>& prims, const std::vector<Mat16>& matrices,
                         ovstage_ordinal_t ordinal);
    // Overwrite a mesh's points (fixed element count) at `ordinal`.
    void writePoints(const std::string& prim, const std::vector<float>& xyz, ovstage_ordinal_t ordinal);
    // Overwrite a scalar float (1 value) or a float3/color3f (3 values) attribute of one prim.
    void writeFloats(const std::string& prim, const char* attribute, const std::vector<float>& values,
                     ovstage_ordinal_t ordinal);

private:
    ovstage_query_handle_t queryFor(ovx_primpath_list_t list);

    ovrtx_renderer_t* m_renderer = nullptr;
    ovstage_instance_t* m_stage = nullptr;
    path_dictionary_instance_t* m_paths = nullptr;
    ovstage_ordinal_t m_next = 0;
    ovstage_ordinal_t m_held = 0;
    std::unordered_map<ovx_primpath_list_t, ovstage_query_handle_t> m_queries;
    std::unordered_map<std::string, ovx_primpath_list_t> m_namedLists;
};

} // namespace vrsim
