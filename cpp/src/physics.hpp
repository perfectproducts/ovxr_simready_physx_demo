// ovphysx (PhysX) simulating the shared stage's USD physics.
//
// Contract with the stage (docs: ovphysx ovstage_integration.md): ovphysx drains only the
// ordinals we hand it (environment + spawn/remove population changes); the per-frame transform
// writes it produced are never drained back into physics.
#pragma once

#include "common.hpp"
#include "stage.hpp"

#include <ovphysx/ovphysx.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace vrsim
{

struct BodyState
{
    ovx_primpath_t path = 0;
    Pose pose;
};

class Physics
{
public:
    Physics(Stage& stage, ovstage_ordinal_t attachOrdinal);
    ~Physics();

    // Drain one sealed population ordinal (spawn/remove) into the simulation.
    void sync(ovstage_ordinal_t ordinal);
    void step(float dt);

    // Poses of every dynamic rigid body after the last step, plus the stage path list in the same
    // order (for writing their transforms back in one batch).
    void readBodies();
    const std::vector<BodyState>& bodies() const { return m_bodies; }
    ovx_primpath_list_t bodyList() const { return m_bodyList; }

    // Overwrite the velocities of every body under `propRoot` (COM linear [m/s], angular [rad/s]).
    void setVelocity(const std::string& propRoot, const Vec3& linear, const Vec3& angular);
    // Same for one rigid-body prim, e.g. the part of a jointed asset a hand holds.
    void setBodyVelocity(const std::string& bodyPath, const Vec3& linear, const Vec3& angular);
    void forgetProp(const std::string& propRoot);

private:
    void wait(ovphysx_enqueue_result_t task, const char* what);
    // Rigid bodies matching a selection pattern, cached until the next sync.
    ovphysx_query_handle_t selectionQuery(const std::string& pattern);
    void writeVelocity(ovphysx_query_handle_t query, const Vec3& linear, const Vec3& angular);

    Stage& m_stage;
    ovphysx_instance_handle_t m_px = OVPHYSX_INVALID_HANDLE;
    ovphysx_query_handle_t m_allBodies = 0;
    ovx_token_t m_positionToken = 0, m_orientationToken = 0;
    std::vector<BodyState> m_bodies;
    std::vector<ovx_primpath_t> m_listPaths;
    ovx_primpath_list_t m_bodyList = 0;
    std::unordered_map<std::string, ovphysx_query_handle_t> m_selectionQueries;
};

} // namespace vrsim
