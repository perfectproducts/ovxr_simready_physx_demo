// Controller input on the OpenXR session ovxr owns (borrowed handles; we create only our own
// action set, action spaces and a VIEW space). Same bindings as the Python app:
//   squeeze grab/throw · trigger pointer ray · A/X push-to-talk · B undo · Y clear · left menu: toaster monitor
#pragma once

#include "common.hpp"

#include <ovxr/ovxr.h>

#include <openxr/openxr.h>

#include <array>

namespace vrsim
{

struct HandInput
{
    bool active = false;
    Pose grip, aim;
    bool aimValid = false;
    float squeeze = 0, trigger = 0;
    bool primary = false, secondary = false;
    bool primaryPressed = false, primaryReleased = false, secondaryPressed = false;
    float thumbX = 0;
    bool thumbClickPressed = false;
    bool menuPressed = false;  // left controller only (the right one's is the system button)
};

struct InputFrame
{
    std::array<HandInput, 2> hands;
    Pose head;
    bool headValid = false;
    bool focused = false;
};

class XrInput
{
public:
    XrInput(ovxr_context_t* context, const ovxr_openxr_handles_t& handles);
    ~XrInput();

    InputFrame sync(int64_t predictedDisplayTimeNs);
    void haptic(int hand, float amplitude, float seconds, float frequency = XR_FREQUENCY_UNSPECIFIED);

private:
    bool locate(XrSpace space, XrTime time, Pose& out);
    float floatState(XrAction action, int hand);
    bool boolState(XrAction action, int hand);
    XrPath path(const char* s);

    ovxr_context_t* m_ctx;
    XrInstance m_instance;
    XrSession m_session;
    XrSpace m_base;
    PFN_xrGetInstanceProcAddr m_gipa;

    PFN_xrStringToPath m_xrStringToPath = nullptr;
    PFN_xrCreateActionSet m_xrCreateActionSet = nullptr;
    PFN_xrDestroyActionSet m_xrDestroyActionSet = nullptr;
    PFN_xrCreateAction m_xrCreateAction = nullptr;
    PFN_xrSuggestInteractionProfileBindings m_xrSuggest = nullptr;
    PFN_xrCreateActionSpace m_xrCreateActionSpace = nullptr;
    PFN_xrCreateReferenceSpace m_xrCreateReferenceSpace = nullptr;
    PFN_xrDestroySpace m_xrDestroySpace = nullptr;
    PFN_xrAttachSessionActionSets m_xrAttach = nullptr;
    PFN_xrSyncActions m_xrSyncActions = nullptr;
    PFN_xrLocateSpace m_xrLocateSpace = nullptr;
    PFN_xrGetActionStateFloat m_xrGetFloat = nullptr;
    PFN_xrGetActionStateBoolean m_xrGetBool = nullptr;
    PFN_xrGetActionStateVector2f m_xrGetVec2 = nullptr;
    PFN_xrApplyHapticFeedback m_xrHaptic = nullptr;

    XrActionSet m_set = XR_NULL_HANDLE;
    XrAction m_gripPose = XR_NULL_HANDLE, m_aimPose = XR_NULL_HANDLE, m_squeeze = XR_NULL_HANDLE,
             m_trigger = XR_NULL_HANDLE, m_primary = XR_NULL_HANDLE, m_secondary = XR_NULL_HANDLE,
             m_hapticOut = XR_NULL_HANDLE, m_thumb = XR_NULL_HANDLE, m_thumbClick = XR_NULL_HANDLE, m_menu = XR_NULL_HANDLE;
    std::array<XrPath, 2> m_hands{};
    std::array<XrSpace, 2> m_gripSpaces{}, m_aimSpaces{};
    XrSpace m_view = XR_NULL_HANDLE;
    std::array<bool, 2> m_prevPrimary{}, m_prevSecondary{}, m_prevThumbClick{}, m_prevMenu{};
};

} // namespace vrsim
