#include "stage.hpp"

#include <ovrtx/ovrtx_config.h>
#include <ovstage/ovstage_config.h>

#include <cmath>
#include <cstring>

namespace vrsim
{
namespace
{

void checkOvrtx(ovrtx_result_t result, const char* what)
{
    if (result.status != OVRTX_API_SUCCESS)
        die(what, str(ovrtx_get_last_error()));
}

void waitPopulation(ovstage_instance_t* stage, ovstage_population_enqueue_result_t result, const char* what)
{
    if (result.status != OVSTAGE_OK ||
        (result.op_index != OVSTAGE_POPULATION_INVALID_OP_ID &&
         ovstage_population_wait_op(stage, result.op_index, OVSTAGE_TIMEOUT_INFINITE, nullptr) != OVSTAGE_OK))
        die(what, str(ovstage_population_get_last_error()));
}

DLTensor tensorF64x16(const Mat16* data, int64_t* shape)
{
    DLTensor t{};
    t.data = const_cast<Mat16*>(data);
    t.device.device_type = kDLCPU;
    t.ndim = 1;
    t.shape = shape;
    t.dtype = { kDLFloat, 64, 16 };
    return t;
}

} // namespace

void Stage::initializeProcess()
{
    // GPU hierarchy model: with the CPU-incremental default, ovrtx 0.6 applies a prim's transform
    // only on its first change, so per-frame body/controller updates never reach the image.
    const ovstage_config_entry_t entries[] = { ovstage_config_entry_runtime_default_hierarchy_computation_model(
        OVSTAGE_HIERARCHY_COMPUTATION_MODEL_GPU_INCREMENTAL) };
    const ovstage_config_t config{ entries, 1 };
    checkStage(ovstage_initialize(&config), "ovstage_initialize");
}

void Stage::shutdownProcess()
{
    ovstage_shutdown();
}

Stage::Stage(const std::string& environmentUsd, const std::string& physxSchemasRoot, bool minimalRendering)
{
    // ovphysx reads physx:* attributes only if their schemas are registered before population.
    const std::string modules[] = { physxSchemasRoot + "/physx_usd_schemas/PhysxSchema/resources",
                                    physxSchemasRoot + "/physx_usd_schemas/OmniUsdPhysicsDeformableSchema/resources" };
    const ovx_string_t schemaPaths[] = { view(modules[0]), view(modules[1]) };
    if (ovstage_population_register_usd_schemas(schemaPaths, 2) != OVSTAGE_OK)
        die("ovstage_population_register_usd_schemas", str(ovstage_population_get_last_error()) +
                                                           " (set PHYSX_USD_SCHEMAS_ROOT to the venv site-packages)");

    logf("Creating renderer (first run compiles shaders; with Quest Link active this can take minutes)...");
    const ovrtx_config_entry_t minimalOnly[] = { ovrtx_config_entry_enable_minimal(true),
                                                 ovrtx_config_entry_enable_realtimepathtracing(false),
                                                 ovrtx_config_entry_enable_pathtracing(false) };
    ovrtx_config_t rendererConfig{};
    if (minimalRendering)
        rendererConfig = { minimalOnly, 3 };
    logf("Render mode: %s", minimalRendering ? "Minimal" : "real-time path tracing (RT2)");
    checkOvrtx(ovrtx_create_renderer(&rendererConfig, &m_renderer), "ovrtx_create_renderer");
    ovstage_instance_desc_t desc{};
    desc.name = "vrsim_cpp";
    checkStage(ovstage_create_instance(&desc, &m_stage), "ovstage_create_instance");
    checkOvrtx(ovrtx_attach_ovstage(m_renderer, m_stage), "ovrtx_attach_ovstage");
    m_paths = ovstage_get_path_dictionary(m_stage);

    ovstage_ordinal_query_handle_t floorQuery = OVSTAGE_INVALID_ORDINAL_QUERY_HANDLE;
    checkStage(ovstage_get_attribute_write_floor(m_stage, ovx_string_or_token_t{}, &floorQuery).status,
               "ovstage_get_attribute_write_floor");
    ovstage_ordinal_t floor = 0;
    checkStage(ovstage_fetch_ordinal(m_stage, floorQuery, OVSTAGE_TIMEOUT_INFINITE, &floor), "ovstage_fetch_ordinal");
    waitStage(m_stage, ovstage_release_ordinal_query(m_stage, floorQuery), "ovstage_release_ordinal_query");
    m_next = floor + 1;

    // ALL domains: the renderer needs meshes/materials, ovphysx needs bodies/colliders/units.
    const ovstage_ordinal_t ordinal = current();
    waitPopulation(m_stage,
                   ovstage_population_open_usd_from_file(m_stage, view(environmentUsd), ordinal, NAN,
                                                         OVSTAGE_POPULATION_DOMAIN_ALL),
                   "open environment");
    waitStage(m_stage, ovstage_compute_hierarchy(m_stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_RUNTIME_DEFAULT,
                                                 ordinal, ordinal),
              "ovstage_compute_hierarchy");
    commit();
    logf("Environment loaded: %s", environmentUsd.c_str());
}

Stage::~Stage()
{
    for (auto& [list, query] : m_queries)
        ovstage_release_query(m_stage, query);
    for (auto& [name, list] : m_namedLists)
        path_dictionary_release_path_list_reference(m_paths, list);
    ovrtx_detach_ovstage(m_renderer);
    ovrtx_destroy_renderer(m_renderer);
    ovstage_destroy_instance(m_stage);
}

ovstage_ordinal_t Stage::current()
{
    if (!m_held)
        m_held = m_next++;
    return m_held;
}

ovstage_ordinal_t Stage::commit()
{
    const ovstage_ordinal_t ordinal = current();
    ovstage_write_floor_desc_t floor{};
    floor.ordinal = ordinal;
    floor.scope = OVSTAGE_SCOPE_ALL;
    waitStage(m_stage, ovstage_advance_write_floor(m_stage, &floor), "ovstage_advance_write_floor");
    resolve();
    return ordinal;
}

ovstage_population_usd_reference_handle_t Stage::addReference(const std::string& layer, const std::string& primPath,
                                                              ovstage_ordinal_t ordinal)
{
    ovstage_population_usd_reference_handle_t handle = OVSTAGE_POPULATION_INVALID_USD_REFERENCE_HANDLE;
    waitPopulation(m_stage,
                   ovstage_population_add_usd_reference_from_string(m_stage, view(layer), view(primPath), &handle),
                   "add_usd_reference_from_string");
    waitPopulation(m_stage, ovstage_population_apply_usd_changes(m_stage, ordinal), "apply_usd_changes");
    // ovphysx parses the new bodies' world matrices, so they must exist before the seal.
    waitStage(m_stage, ovstage_compute_hierarchy(m_stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_RUNTIME_DEFAULT,
                                                 ordinal, ordinal),
              "ovstage_compute_hierarchy");
    return handle;
}

void Stage::removeReference(ovstage_population_usd_reference_handle_t handle, ovstage_ordinal_t ordinal)
{
    waitPopulation(m_stage, ovstage_population_remove_usd_reference(m_stage, handle), "remove_usd_reference");
    waitPopulation(m_stage, ovstage_population_apply_usd_changes(m_stage, ordinal), "apply_usd_changes");
}

ovx_primpath_t Stage::internPath(const std::string& path)
{
    const ovx_string_t s = view(path);
    ovx_primpath_t out = 0;
    if (path_dictionary_create_paths_from_strings(m_paths, &s, 1, &out).status != OVX_API_SUCCESS)
        die("path_dictionary_create_paths_from_strings", path);
    return out;
}

ovx_primpath_list_t Stage::pathList(const std::vector<std::string>& paths)
{
    std::vector<ovx_string_t> views;
    views.reserve(paths.size());
    for (const auto& p : paths)
        views.push_back(view(p));
    ovx_primpath_list_t list = 0;
    if (path_dictionary_create_path_list_from_strings(m_paths, views.data(), views.size(), &list).status !=
        OVX_API_SUCCESS)
        die("path_dictionary_create_path_list_from_strings");
    return list;
}

ovstage_query_handle_t Stage::queryFor(ovx_primpath_list_t list)
{
    auto it = m_queries.find(list);
    if (it != m_queries.end())
        return it->second;
    ovstage_query_handle_t query{};
    checkStage(ovstage_query_from_path_list(m_stage, list, &query), "ovstage_query_from_path_list");
    m_queries.emplace(list, query);
    return query;
}

void Stage::writeTransforms(ovx_primpath_list_t list, const std::vector<Mat16>& matrices, ovstage_ordinal_t ordinal)
{
    if (matrices.empty())
        return;
    int64_t rows = static_cast<int64_t>(matrices.size());
    const DLTensor matrixTensor = tensorF64x16(matrices.data(), &rows);
    std::vector<uint8_t> resetFlags(matrices.size(), 1);
    DLTensor resetTensor{};
    resetTensor.data = resetFlags.data();
    resetTensor.device.device_type = kDLCPU;
    resetTensor.ndim = 1;
    resetTensor.shape = &rows;
    resetTensor.dtype = { kDLBool, 8, 1 };

    auto data = [](const DLTensor* t, ovstage_attribute_semantic_t semantic) {
        ovstage_write_data_t d{};
        d.tensors = t;
        d.tensor_count = 1;
        d.semantic = semantic;
        d.is_array = false;
        d.attribute_mode = OVSTAGE_ATTRIBUTE_MODE_FORCE;
        return d;
    };
    // localMatrix aliases xformOp:transform/omni:xform; with resetXformStack it is the world pose.
    const ovstage_attribute_write_t writes[] = {
        { name("omni:fabric:localMatrix"), data(&matrixTensor, OVSTAGE_SEMANTIC_MATRIX) },
        { name("omni:fabric:worldMatrix"), data(&matrixTensor, OVSTAGE_SEMANTIC_MATRIX) },
        { name("omni:fabric:resetXformStack"), data(&resetTensor, OVSTAGE_SEMANTIC_NONE) },
    };
    waitStage(m_stage, ovstage_write_attributes(m_stage, queryFor(list), writes, 3, ordinal, OVSTAGE_PRIM_MODE_UPSERT),
              "write transforms");
}

void Stage::writeTransforms(const std::vector<std::string>& prims, const std::vector<Mat16>& matrices,
                            ovstage_ordinal_t ordinal)
{
    std::string key;
    for (const auto& p : prims)
        key += p + ";";
    auto it = m_namedLists.find(key);
    if (it == m_namedLists.end())
        it = m_namedLists.emplace(key, pathList(prims)).first;
    writeTransforms(it->second, matrices, ordinal);
}

void Stage::writePoints(const std::string& prim, const std::vector<float>& xyz, ovstage_ordinal_t ordinal)
{
    auto it = m_namedLists.find(prim);
    if (it == m_namedLists.end())
        it = m_namedLists.emplace(prim, pathList({ prim })).first;
    int64_t count = static_cast<int64_t>(xyz.size() / 3);
    DLTensor t{};
    t.data = const_cast<float*>(xyz.data());
    t.device.device_type = kDLCPU;
    t.ndim = 1;
    t.shape = &count;
    t.dtype = { kDLFloat, 32, 3 };
    ovstage_write_data_t d{};
    d.tensors = &t;
    d.tensor_count = 1;  // one tensor per prim for array attributes
    d.is_array = true;
    waitStage(m_stage,
              ovstage_write_attribute(m_stage, queryFor(it->second), name("points"), ordinal, d, OVSTAGE_PRIM_MODE_UPSERT),
              "write points");
}

void Stage::writeFloats(const std::string& prim, const char* attribute, const std::vector<float>& values,
                        ovstage_ordinal_t ordinal)
{
    if (values.empty())
        return;
    auto it = m_namedLists.find(prim);
    if (it == m_namedLists.end())
        it = m_namedLists.emplace(prim, pathList({ prim })).first;
    int64_t rows = 1;
    DLTensor t{};
    t.data = const_cast<float*>(values.data());
    t.device.device_type = kDLCPU;
    t.ndim = 1;
    t.shape = &rows;
    t.dtype = { kDLFloat, 32, static_cast<uint16_t>(values.size()) };
    ovstage_write_data_t d{};
    d.tensors = &t;
    d.tensor_count = 1;
    d.is_array = false;
    d.attribute_mode = OVSTAGE_ATTRIBUTE_MODE_FORCE;
    waitStage(m_stage,
              ovstage_write_attribute(m_stage, queryFor(it->second), name(attribute), ordinal, d, OVSTAGE_PRIM_MODE_UPSERT),
              "write floats");
}

} // namespace vrsim
