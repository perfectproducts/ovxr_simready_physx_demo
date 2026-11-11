#include "xr_input.hpp"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vrsim
{
namespace
{

template <typename Fn>
void resolve(PFN_xrGetInstanceProcAddr gipa, XrInstance instance, const char* name, Fn& out)
{
    PFN_xrVoidFunction fn = nullptr;
    if (XR_FAILED(gipa(instance, name, &fn)) || !fn)
        die("xrGetInstanceProcAddr", name);
    out = reinterpret_cast<Fn>(fn);
}

void checkXr(XrResult r, const char* what)
{
    if (XR_FAILED(r))
        die(what, "XrResult " + std::to_string(static_cast<int>(r)));
}

} // namespace

XrPath XrInput::path(const char* s)
{
    XrPath p = XR_NULL_PATH;
    checkXr(m_xrStringToPath(m_instance, s, &p), s);
    return p;
}

XrInput::XrInput(ovxr_context_t* context, const ovxr_openxr_handles_t& handles)
    : m_ctx(context), m_instance(reinterpret_cast<XrInstance>(handles.xr_instance)),
      m_session(reinterpret_cast<XrSession>(handles.xr_session)), m_base(reinterpret_cast<XrSpace>(handles.xr_space)),
      m_gipa(reinterpret_cast<PFN_xrGetInstanceProcAddr>(handles.xr_get_instance_proc_addr))
{
    resolve(m_gipa, m_instance, "xrStringToPath", m_xrStringToPath);
    resolve(m_gipa, m_instance, "xrCreateActionSet", m_xrCreateActionSet);
    resolve(m_gipa, m_instance, "xrDestroyActionSet", m_xrDestroyActionSet);
    resolve(m_gipa, m_instance, "xrCreateAction", m_xrCreateAction);
    resolve(m_gipa, m_instance, "xrSuggestInteractionProfileBindings", m_xrSuggest);
    resolve(m_gipa, m_instance, "xrCreateActionSpace", m_xrCreateActionSpace);
    resolve(m_gipa, m_instance, "xrCreateReferenceSpace", m_xrCreateReferenceSpace);
    resolve(m_gipa, m_instance, "xrDestroySpace", m_xrDestroySpace);
    resolve(m_gipa, m_instance, "xrAttachSessionActionSets", m_xrAttach);
    resolve(m_gipa, m_instance, "xrSyncActions", m_xrSyncActions);
    resolve(m_gipa, m_instance, "xrLocateSpace", m_xrLocateSpace);
    resolve(m_gipa, m_instance, "xrGetActionStateFloat", m_xrGetFloat);
    resolve(m_gipa, m_instance, "xrGetActionStateBoolean", m_xrGetBool);
    resolve(m_gipa, m_instance, "xrApplyHapticFeedback", m_xrHaptic);
    resolve(m_gipa, m_instance, "xrGetActionStateVector2f", m_xrGetVec2);

    XrActionSetCreateInfo setInfo{ XR_TYPE_ACTION_SET_CREATE_INFO };
    std::snprintf(setInfo.actionSetName, sizeof(setInfo.actionSetName), "vrsim");
    std::snprintf(setInfo.localizedActionSetName, sizeof(setInfo.localizedActionSetName), "VR SimReady");
    checkXr(m_xrCreateActionSet(m_instance, &setInfo, &m_set), "xrCreateActionSet");

    m_hands = { path("/user/hand/left"), path("/user/hand/right") };
    auto action = [&](const char* name, XrActionType type) {
        XrActionCreateInfo info{ XR_TYPE_ACTION_CREATE_INFO };
        std::snprintf(info.actionName, sizeof(info.actionName), "%s", name);
        std::snprintf(info.localizedActionName, sizeof(info.localizedActionName), "%s", name);
        info.actionType = type;
        info.countSubactionPaths = 2;
        info.subactionPaths = m_hands.data();
        XrAction a = XR_NULL_HANDLE;
        checkXr(m_xrCreateAction(m_set, &info, &a), name);
        return a;
    };
    m_gripPose = action("grip_pose", XR_ACTION_TYPE_POSE_INPUT);
    m_aimPose = action("aim_pose", XR_ACTION_TYPE_POSE_INPUT);
    m_squeeze = action("squeeze", XR_ACTION_TYPE_FLOAT_INPUT);
    m_trigger = action("trigger", XR_ACTION_TYPE_FLOAT_INPUT);
    m_primary = action("primary", XR_ACTION_TYPE_BOOLEAN_INPUT);
    m_secondary = action("secondary", XR_ACTION_TYPE_BOOLEAN_INPUT);
    m_hapticOut = action("haptic", XR_ACTION_TYPE_VIBRATION_OUTPUT);
    m_thumb = action("thumbstick", XR_ACTION_TYPE_VECTOR2F_INPUT);
    m_thumbClick = action("thumbstick_click", XR_ACTION_TYPE_BOOLEAN_INPUT);
    m_menu = action("menu", XR_ACTION_TYPE_BOOLEAN_INPUT);

    std::vector<XrActionSuggestedBinding> touch;
    for (const char* hand : { "left", "right" })
    {
        const std::string u = std::string("/user/hand/") + hand;
        touch.push_back({ m_gripPose, path((u + "/input/grip/pose").c_str()) });
        touch.push_back({ m_aimPose, path((u + "/input/aim/pose").c_str()) });
        touch.push_back({ m_squeeze, path((u + "/input/squeeze/value").c_str()) });
        touch.push_back({ m_trigger, path((u + "/input/trigger/value").c_str()) });
        touch.push_back({ m_hapticOut, path((u + "/output/haptic").c_str()) });
        touch.push_back({ m_thumb, path((u + "/input/thumbstick").c_str()) });
        touch.push_back({ m_thumbClick, path((u + "/input/thumbstick/click").c_str()) });
    }
    touch.push_back({ m_primary, path("/user/hand/left/input/x/click") });
    touch.push_back({ m_primary, path("/user/hand/right/input/a/click") });
    touch.push_back({ m_secondary, path("/user/hand/left/input/y/click") });
    touch.push_back({ m_secondary, path("/user/hand/right/input/b/click") });
    touch.push_back({ m_menu, path("/user/hand/left/input/menu/click") });
    XrInteractionProfileSuggestedBinding suggested{ XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
    suggested.interactionProfile = path("/interaction_profiles/oculus/touch_controller");
    suggested.suggestedBindings = touch.data();
    suggested.countSuggestedBindings = static_cast<uint32_t>(touch.size());
    checkXr(m_xrSuggest(m_instance, &suggested), "xrSuggestInteractionProfileBindings");

    for (int h = 0; h < 2; ++h)
    {
        XrActionSpaceCreateInfo info{ XR_TYPE_ACTION_SPACE_CREATE_INFO };
        info.subactionPath = m_hands[h];
        info.poseInActionSpace.orientation.w = 1.0f;
        info.action = m_gripPose;
        checkXr(m_xrCreateActionSpace(m_session, &info, &m_gripSpaces[h]), "xrCreateActionSpace(grip)");
        info.action = m_aimPose;
        checkXr(m_xrCreateActionSpace(m_session, &info, &m_aimSpaces[h]), "xrCreateActionSpace(aim)");
    }
    XrReferenceSpaceCreateInfo view{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    view.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    view.poseInReferenceSpace.orientation.w = 1.0f;
    checkXr(m_xrCreateReferenceSpace(m_session, &view, &m_view), "xrCreateReferenceSpace(VIEW)");

    XrSessionActionSetsAttachInfo attach{ XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO };
    attach.countActionSets = 1;
    attach.actionSets = &m_set;
    checkXr(m_xrAttach(m_session, &attach), "xrAttachSessionActionSets");
    logf("XR input ready");
}

XrInput::~XrInput()
{
    for (XrSpace s : { m_gripSpaces[0], m_gripSpaces[1], m_aimSpaces[0], m_aimSpaces[1], m_view })
        if (s != XR_NULL_HANDLE)
            m_xrDestroySpace(s);
    if (m_set != XR_NULL_HANDLE)
        m_xrDestroyActionSet(m_set);
}

bool XrInput::locate(XrSpace space, XrTime time, Pose& out)
{
    XrSpaceLocation loc{ XR_TYPE_SPACE_LOCATION };
    if (XR_FAILED(m_xrLocateSpace(space, m_base, time, &loc)))
        return false;
    const XrSpaceLocationFlags need = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    if ((loc.locationFlags & need) != need)
        return false;
    ovxr_pose_t pose{};
    pose.position = { loc.pose.position.x, loc.pose.position.y, loc.pose.position.z };
    pose.orientation = { loc.pose.orientation.x, loc.pose.orientation.y, loc.pose.orientation.z,
                         loc.pose.orientation.w };
    double m[16];
    if (ovxr_transform_openxr_pose(m_ctx, &pose, m) != OVXR_SUCCESS)  // stage units, latched anchor
        return false;
    out = Pose::fromMatrix(m);
    return true;
}

float XrInput::floatState(XrAction action, int hand)
{
    XrActionStateGetInfo info{ XR_TYPE_ACTION_STATE_GET_INFO };
    info.action = action;
    info.subactionPath = m_hands[hand];
    XrActionStateFloat state{ XR_TYPE_ACTION_STATE_FLOAT };
    m_xrGetFloat(m_session, &info, &state);
    return state.isActive ? state.currentState : 0.0f;
}

bool XrInput::boolState(XrAction action, int hand)
{
    XrActionStateGetInfo info{ XR_TYPE_ACTION_STATE_GET_INFO };
    info.action = action;
    info.subactionPath = m_hands[hand];
    XrActionStateBoolean state{ XR_TYPE_ACTION_STATE_BOOLEAN };
    m_xrGetBool(m_session, &info, &state);
    return state.isActive && state.currentState;
}

InputFrame XrInput::sync(int64_t predictedDisplayTimeNs)
{
    InputFrame frame;
    const XrActiveActionSet active{ m_set, XR_NULL_PATH };
    XrActionsSyncInfo syncInfo{ XR_TYPE_ACTIONS_SYNC_INFO };
    syncInfo.countActiveActionSets = 1;
    syncInfo.activeActionSets = &active;
    frame.focused = m_xrSyncActions(m_session, &syncInfo) == XR_SUCCESS;
    const XrTime time = predictedDisplayTimeNs;
    frame.headValid = locate(m_view, time, frame.head);
    if (!frame.focused)  // e.g. the Quest system menu is open
    {
        m_prevPrimary = {};
        m_prevSecondary = {};
        m_prevThumbClick = {};
        m_prevMenu = {};
        return frame;
    }
    for (int h = 0; h < 2; ++h)
    {
        HandInput& hi = frame.hands[h];
        hi.active = locate(m_gripSpaces[h], time, hi.grip);
        hi.aimValid = locate(m_aimSpaces[h], time, hi.aim);
        hi.squeeze = floatState(m_squeeze, h);
        hi.trigger = floatState(m_trigger, h);
        hi.primary = boolState(m_primary, h);
        hi.secondary = boolState(m_secondary, h);
        hi.primaryPressed = hi.primary && !m_prevPrimary[h];
        hi.primaryReleased = !hi.primary && m_prevPrimary[h];
        hi.secondaryPressed = hi.secondary && !m_prevSecondary[h];
        m_prevPrimary[h] = hi.primary;
        m_prevSecondary[h] = hi.secondary;
        XrActionStateGetInfo info{ XR_TYPE_ACTION_STATE_GET_INFO };
        info.action = m_thumb;
        info.subactionPath = m_hands[h];
        XrActionStateVector2f stick{ XR_TYPE_ACTION_STATE_VECTOR2F };
        m_xrGetVec2(m_session, &info, &stick);
        hi.thumbX = stick.isActive ? stick.currentState.x : 0.0f;
        const bool click = boolState(m_thumbClick, h);
        hi.thumbClickPressed = click && !m_prevThumbClick[h];
        m_prevThumbClick[h] = click;
        const bool menu = boolState(m_menu, h);
        hi.menuPressed = menu && !m_prevMenu[h];
        m_prevMenu[h] = menu;
    }
    return frame;
}

void XrInput::haptic(int hand, float amplitude, float seconds, float frequency)
{
    XrHapticActionInfo info{ XR_TYPE_HAPTIC_ACTION_INFO };
    info.action = m_hapticOut;
    info.subactionPath = m_hands[hand];
    XrHapticVibration vibration{ XR_TYPE_HAPTIC_VIBRATION };
    vibration.amplitude = amplitude;
    vibration.duration = static_cast<XrDuration>(seconds * 1e9);
    vibration.frequency = frequency;
    m_xrHaptic(m_session, &info, reinterpret_cast<const XrHapticBaseHeader*>(&vibration));
}

} // namespace vrsim
