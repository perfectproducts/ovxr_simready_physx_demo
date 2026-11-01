#include "physics.hpp"

#include <ovphysx/ovphysx_attributes.h>

#include <cstring>

namespace vrsim
{
namespace
{

void checkPx(ovphysx_result_t result, const char* what)
{
    if (result.status != OVPHYSX_API_SUCCESS)
    {
        const ovphysx_string_t e = ovphysx_get_last_error();
        die(what, e.ptr ? std::string(e.ptr, e.length) : std::string());
    }
}

ovx_token_t token(path_dictionary_instance_t* paths, const char* s)
{
    const ovx_string_t v{ s, std::strlen(s) };
    ovx_token_t t = 0;
    if (path_dictionary_create_tokens_from_strings(paths, &v, 1, &t).status != OVX_API_SUCCESS)
        die("path_dictionary_create_tokens_from_strings", s);
    return t;
}

// Prim handles a group covers, in row order.
std::vector<ovx_primpath_t> groupPaths(path_dictionary_instance_t* paths, const ovstage_prim_group_t& prims)
{
    std::vector<ovx_primpath_t> all;
    size_t total = 0;
    path_dictionary_get_num_paths_from_path_list(paths, prims.list, &total);
    all.resize(total);
    size_t got = 0;
    path_dictionary_get_paths_from_path_list(paths, prims.list, 0, total, all.data(), &got);
    std::vector<ovx_primpath_t> rows(prims.count);
    for (uint32_t i = 0; i < prims.count; ++i)
        rows[i] = all[prims.index_map ? prims.index_map[i] : prims.offset + i];
    return rows;
}

} // namespace

Physics::Physics(Stage& stage, ovstage_ordinal_t attachOrdinal) : m_stage(stage)
{
    checkPx(ovphysx_initialize(), "ovphysx_initialize");
    checkPx(ovphysx_create_instance(&m_px), "ovphysx_create_instance");
    wait(ovphysx_attach_ovstage(m_px, stage.handle(), attachOrdinal), "ovphysx_attach_ovstage");
    m_positionToken = token(stage.paths(), OVPHYSX_ATTR_POSITION);
    m_orientationToken = token(stage.paths(), OVPHYSX_ATTR_ORIENTATION);
    checkPx(ovphysx_warmup(m_px), "ovphysx_warmup");
    logf("ovphysx attached at ordinal %llu", static_cast<unsigned long long>(attachOrdinal));
}

Physics::~Physics()
{
    for (auto& [root, query] : m_selectionQueries)
        ovphysx_release_query(m_px, query);
    if (m_allBodies)
        ovphysx_release_query(m_px, m_allBodies);
    if (m_bodyList)
        path_dictionary_release_path_list_reference(m_stage.paths(), m_bodyList);
    ovphysx_detach_ovstage(m_px);
    ovphysx_destroy_instance(m_px);
    ovphysx_shutdown();
}

void Physics::wait(ovphysx_enqueue_result_t task, const char* what)
{
    if (task.status != OVPHYSX_API_SUCCESS)
        checkPx({ task.status }, what);
    ovphysx_wait_task_result_t result{};
    const ovphysx_result_t status = ovphysx_wait_task(m_px, task, OVPHYSX_TIMEOUT_INFINITE, &result);
    std::string error;
    for (size_t i = 0; i < result.num_errors; ++i)
    {
        const ovphysx_string_t e = ovphysx_get_last_task_error(m_px, result.error_tasks[i]);
        error += e.ptr ? std::string(e.ptr, e.length) : "?";
    }
    ovphysx_destroy_wait_task_result(&result);
    if (status.status != OVPHYSX_API_SUCCESS || !error.empty())
        die(what, error);
}

void Physics::sync(ovstage_ordinal_t ordinal)
{
    // Structural change: body membership and query layouts are rebuilt.
    for (auto& [root, query] : m_selectionQueries)
        ovphysx_release_query(m_px, query);
    m_selectionQueries.clear();
    if (m_allBodies)
        ovphysx_release_query(m_px, m_allBodies), m_allBodies = 0;
    wait(ovphysx_update_from_ovstage(m_px, ovstage_ordinal_range_t{ ordinal, ordinal }), "ovphysx_update_from_ovstage");
}

void Physics::step(float dt)
{
    checkPx(ovphysx_step_sync(m_px, dt), "ovphysx_step_sync");
}

void Physics::readBodies()
{
    if (!m_allBodies)
        wait(ovphysx_query(m_px, OVPHYSX_QUERY_TARGET_RIGID_BODY, OVPHYSX_SCOPE_ALL, &m_allBodies, nullptr),
             "ovphysx_query");
    const ovx_string_or_token_t attrs[] = { name(OVPHYSX_ATTR_POSITION), name(OVPHYSX_ATTR_ORIENTATION) };
    ovphysx_read_handle_t read = 0;
    wait(ovphysx_read(m_px, m_allBodies, attrs, 2, &read, nullptr), "ovphysx_read");

    std::vector<ovx_primpath_t> order;
    std::unordered_map<ovx_primpath_t, Pose> poses;
    for (;;)
    {
        const ovstage_read_group_t* g = nullptr;
        wait(ovphysx_fetch_read_next(m_px, read, &g), "ovphysx_fetch_read_next");
        if (!g)
            break;
        if (!g->is_delete && !g->is_array && g->data.tensor_count && g->data.tensors[0].device.device_type == kDLCPU)
        {
            const std::vector<ovx_primpath_t> rows = groupPaths(m_stage.paths(), g->prims);
            const float* v = static_cast<const float*>(g->data.tensors[0].data);
            for (size_t i = 0; i < rows.size(); ++i)
            {
                const uint32_t r = g->data.index_map ? g->data.index_map[i] : static_cast<uint32_t>(i);
                Pose& pose = poses[rows[i]];
                if (g->attribute == m_positionToken)
                {
                    pose.p = { v[r * 3], v[r * 3 + 1], v[r * 3 + 2] };
                    order.push_back(rows[i]);
                }
                else if (g->attribute == m_orientationToken)
                    pose.q = Quat{ v[r * 4], v[r * 4 + 1], v[r * 4 + 2], v[r * 4 + 3] }.normalized();
            }
        }
        ovphysx_release_group(m_px, read, g->read_group_id);
    }
    wait(ovphysx_release_read(m_px, read), "ovphysx_release_read");

    m_bodies.clear();
    for (ovx_primpath_t p : order)
        m_bodies.push_back({ p, poses[p] });
    if (order != m_listPaths)  // membership changed: rebuild the write list in this order
    {
        if (m_bodyList)
            path_dictionary_release_path_list_reference(m_stage.paths(), m_bodyList);
        m_bodyList = 0;
        if (!order.empty() &&
            path_dictionary_create_path_list_from_paths(m_stage.paths(), order.data(), order.size(), &m_bodyList)
                    .status != OVX_API_SUCCESS)
            die("path_dictionary_create_path_list_from_paths");
        m_listPaths = order;
    }
}

ovphysx_query_handle_t Physics::selectionQuery(const std::string& pattern)
{
    auto it = m_selectionQueries.find(pattern);
    if (it != m_selectionQueries.end())
        return it->second;
    ovphysx_query_selection_t selection{};
    selection.pattern = { pattern.c_str(), pattern.size() };
    ovphysx_query_handle_t query = 0;
    wait(ovphysx_query_selection(m_px, OVPHYSX_QUERY_TARGET_RIGID_BODY, &selection, &query, nullptr),
         "ovphysx_query_selection");
    return m_selectionQueries.emplace(pattern, query).first->second;
}

void Physics::setVelocity(const std::string& propRoot, const Vec3& linear, const Vec3& angular)
{
    writeVelocity(selectionQuery(propRoot + "/**"), linear, angular);
}

void Physics::setBodyVelocity(const std::string& bodyPath, const Vec3& linear, const Vec3& angular)
{
    writeVelocity(selectionQuery(bodyPath), linear, angular);
}

void Physics::writeVelocity(ovphysx_query_handle_t query, const Vec3& linear, const Vec3& angular)
{
    const ovx_string_or_token_t attrs[] = { name(OVPHYSX_ATTR_LINEAR_VELOCITY), name(OVPHYSX_ATTR_ANGULAR_VELOCITY) };
    ovphysx_write_handle_t write = 0;
    wait(ovphysx_write(m_px, query, attrs, 2, &write, nullptr), "ovphysx_write");
    for (;;)
    {
        ovphysx_fetched_write_group_t fg{};
        wait(ovphysx_fetch_write_next(m_px, write, &fg), "ovphysx_fetch_write_next");
        if (!fg.group)
            break;
        const Vec3& v = fg.attribute_index == 0 ? linear : angular;
        const DLTensor& t = fg.group->data.tensors[0];
        float* out = static_cast<float*>(t.data);
        const int64_t rows = t.ndim > 0 ? t.shape[0] : 1;
        for (int64_t i = 0; i < rows; ++i)
        {
            out[i * 3] = static_cast<float>(v.x);
            out[i * 3 + 1] = static_cast<float>(v.y);
            out[i * 3 + 2] = static_cast<float>(v.z);
        }
        wait(ovphysx_commit_group(m_px, write, fg.group, ovstage_cuda_sync_t{}), "ovphysx_commit_group");
    }
    wait(ovphysx_release_write(m_px, write), "ovphysx_release_write");
}

void Physics::forgetProp(const std::string& propRoot)
{
    for (auto it = m_selectionQueries.begin(); it != m_selectionQueries.end();)
        if (it->first.rfind(propRoot + "/", 0) == 0)  // the prop's "/**" and its body paths
        {
            ovphysx_release_query(m_px, it->second);
            it = m_selectionQueries.erase(it);
        }
        else
            ++it;
}

} // namespace vrsim
