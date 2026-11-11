"""Controller input on the OpenXR session that ovxr owns (same bindings as the C++ app).

ovxr owns the instance, session and frame loop; this creates only its own action set, action
spaces and a VIEW space, and resolves every entry point through ovxr's ``xrGetInstanceProcAddr``
(the handles belong to ovxr's loader, not pyopenxr's). Poses are located at the frame's predicted
display time and mapped into stage space with ``Context.transform_openxr_pose``.

Meta Quest Touch:
    squeeze          grab / throw, drag the catalog by its handle
    trigger          pointer ray, place / fix / press, pick catalog tiles
    A / X (hold)     push-to-talk          B   undo / cancel placement     Y   clear all props
    thumbstick X     turn a placement or a held fixed prop, page the catalog
    thumbstick click open / close the catalog        left menu   toaster FMU monitor
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass, field

import xr

from vrsim.mathutil import Pose
from vrsim.xr_input import _check, _Fn

log = logging.getLogger(__name__)

LEFT, RIGHT = 0, 1
HANDS = ("left", "right")
_VALID = xr.SpaceLocationFlags.POSITION_VALID_BIT | xr.SpaceLocationFlags.ORIENTATION_VALID_BIT


@dataclass
class HandInput:
    active: bool = False
    grip: Pose = field(default_factory=Pose)
    aim: Pose = field(default_factory=Pose)
    aim_valid: bool = False
    squeeze: float = 0.0
    trigger: float = 0.0
    primary: bool = False
    secondary: bool = False
    primary_pressed: bool = False
    primary_released: bool = False
    secondary_pressed: bool = False
    thumb_x: float = 0.0
    thumb_click: bool = False
    thumb_click_pressed: bool = False
    menu: bool = False
    menu_pressed: bool = False  # left controller only (the right one's is the system button)


@dataclass
class InputFrame:
    hands: list[HandInput] = field(default_factory=lambda: [HandInput(), HandInput()])
    head: Pose | None = None
    focused: bool = False


class XrInput:
    def __init__(self, context, handles):
        self.ctx = context
        self.instance = ctypes.cast(ctypes.c_void_p(handles.xr_instance), xr.Instance)
        self.session = ctypes.cast(ctypes.c_void_p(handles.xr_session), xr.Session)
        self.base_space = ctypes.cast(ctypes.c_void_p(handles.xr_space), xr.Space)
        self.fn = _Fn(self.instance, handles.xr_get_instance_proc_addr)
        self._prev = [HandInput(), HandInput()]
        self._warned = False
        self._create_actions()
        log.info("XR input ready (reference space %s)", handles.xr_reference_space_type)

    def _path(self, s: str) -> int:
        p = xr.Path()
        _check(self.fn.xrStringToPath(self.instance, s.encode(), ctypes.byref(p)), f"xrStringToPath({s})")
        return p.value

    def _create_actions(self) -> None:
        fn = self.fn
        self.action_set = xr.ActionSet()
        _check(fn.xrCreateActionSet(self.instance, ctypes.byref(xr.ActionSetCreateInfo(
            action_set_name="vrnx", localized_action_set_name="VR SimReady Newton", priority=0)),
            ctypes.byref(self.action_set)), "xrCreateActionSet")
        self.hand_paths = [self._path(f"/user/hand/{h}") for h in HANDS]
        sub = (xr.Path * 2)(*self.hand_paths)

        def action(name, kind):
            a = xr.Action()
            _check(fn.xrCreateAction(self.action_set, ctypes.byref(xr.ActionCreateInfo(
                action_name=name, action_type=kind, count_subaction_paths=2, subaction_paths=sub,
                localized_action_name=name.replace("_", " "))), ctypes.byref(a)), f"xrCreateAction({name})")
            return a

        T = xr.ActionType
        self.a_grip = action("grip_pose", T.POSE_INPUT)
        self.a_aim = action("aim_pose", T.POSE_INPUT)
        self.a_squeeze = action("squeeze", T.FLOAT_INPUT)
        self.a_trigger = action("trigger", T.FLOAT_INPUT)
        self.a_primary = action("primary", T.BOOLEAN_INPUT)
        self.a_secondary = action("secondary", T.BOOLEAN_INPUT)
        self.a_thumb = action("thumbstick", T.VECTOR2F_INPUT)
        self.a_thumb_click = action("thumbstick_click", T.BOOLEAN_INPUT)
        self.a_menu = action("menu", T.BOOLEAN_INPUT)
        self.a_haptic = action("haptic", T.VIBRATION_OUTPUT)

        def suggest(profile: str, pairs: list[tuple]) -> None:
            bindings = (xr.ActionSuggestedBinding * len(pairs))(
                *[xr.ActionSuggestedBinding(a, self._path(p)) for a, p in pairs])
            rc = fn.xrSuggestInteractionProfileBindings(self.instance, ctypes.byref(
                xr.InteractionProfileSuggestedBinding(interaction_profile=self._path(profile),
                                                      count_suggested_bindings=len(pairs),
                                                      suggested_bindings=bindings)))
            if rc < 0:
                log.warning("Bindings for %s rejected (%s)", profile, rc)

        touch = []
        for h in HANDS:
            u = f"/user/hand/{h}"
            touch += [(self.a_grip, f"{u}/input/grip/pose"), (self.a_aim, f"{u}/input/aim/pose"),
                      (self.a_squeeze, f"{u}/input/squeeze/value"), (self.a_trigger, f"{u}/input/trigger/value"),
                      (self.a_thumb, f"{u}/input/thumbstick"), (self.a_thumb_click, f"{u}/input/thumbstick/click"),
                      (self.a_haptic, f"{u}/output/haptic")]
        touch += [(self.a_primary, "/user/hand/left/input/x/click"), (self.a_primary, "/user/hand/right/input/a/click"),
                  (self.a_secondary, "/user/hand/left/input/y/click"), (self.a_secondary, "/user/hand/right/input/b/click"),
                  (self.a_menu, "/user/hand/left/input/menu/click")]
        suggest("/interaction_profiles/oculus/touch_controller", touch)
        simple = []
        for h in HANDS:
            u = f"/user/hand/{h}"
            simple += [(self.a_grip, f"{u}/input/grip/pose"), (self.a_aim, f"{u}/input/aim/pose"),
                       (self.a_primary, f"{u}/input/menu/click"), (self.a_squeeze, f"{u}/input/select/click"),
                       (self.a_haptic, f"{u}/output/haptic")]
        suggest("/interaction_profiles/khr/simple_controller", simple)

        self.grip_spaces, self.aim_spaces = [], []
        for hp in self.hand_paths:
            for act, out in ((self.a_grip, self.grip_spaces), (self.a_aim, self.aim_spaces)):
                space = xr.Space()
                _check(fn.xrCreateActionSpace(self.session, ctypes.byref(
                    xr.ActionSpaceCreateInfo(action=act, subaction_path=hp)), ctypes.byref(space)),
                    "xrCreateActionSpace")
                out.append(space)
        self.view_space = xr.Space()
        _check(fn.xrCreateReferenceSpace(self.session, ctypes.byref(
            xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.VIEW)),
            ctypes.byref(self.view_space)), "xrCreateReferenceSpace(VIEW)")
        sets = (xr.ActionSet * 1)(self.action_set)
        _check(fn.xrAttachSessionActionSets(self.session, ctypes.byref(
            xr.SessionActionSetsAttachInfo(count_action_sets=1, action_sets=sets))), "xrAttachSessionActionSets")

    # --- per frame ---------------------------------------------------------------------------
    def _locate(self, space, time_ns: int) -> Pose | None:
        loc = xr.SpaceLocation()
        rc = self.fn.xrLocateSpace(space, self.base_space, xr.Time(time_ns), ctypes.byref(loc))
        if rc < 0 or (int(loc.location_flags) & _VALID) != _VALID:
            return None
        p, o = loc.pose.position, loc.pose.orientation
        try:
            m = self.ctx.transform_openxr_pose((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        except Exception as exc:
            if not self._warned:
                self._warned = True
                log.warning("transform_openxr_pose failed (logged once): %s", exc)
            return None
        return Pose.from_row_major(m)

    def _info(self, action, hand: int):
        return ctypes.byref(xr.ActionStateGetInfo(action=action, subaction_path=self.hand_paths[hand]))

    def _float(self, action, hand: int) -> float:
        state = xr.ActionStateFloat()
        self.fn.xrGetActionStateFloat(self.session, self._info(action, hand), ctypes.byref(state))
        return float(state.current_state) if state.is_active else 0.0

    def _bool(self, action, hand: int) -> bool:
        state = xr.ActionStateBoolean()
        self.fn.xrGetActionStateBoolean(self.session, self._info(action, hand), ctypes.byref(state))
        return bool(state.is_active and state.current_state)

    def _vec2_x(self, action, hand: int) -> float:
        state = xr.ActionStateVector2f()
        self.fn.xrGetActionStateVector2f(self.session, self._info(action, hand), ctypes.byref(state))
        return float(state.current_state.x) if state.is_active else 0.0

    def sync(self, predicted_time_ns: int) -> InputFrame:
        frame = InputFrame()
        active = (xr.ActiveActionSet * 1)(xr.ActiveActionSet(self.action_set, xr.NULL_PATH))
        rc = self.fn.xrSyncActions(self.session, ctypes.byref(
            xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=active)))
        frame.focused = rc == xr.Result.SUCCESS
        frame.head = self._locate(self.view_space, predicted_time_ns)
        if not frame.focused:  # e.g. the Quest system menu is open
            self._prev = frame.hands
            return frame
        for h in (LEFT, RIGHT):
            hi, prev = frame.hands[h], self._prev[h]
            grip = self._locate(self.grip_spaces[h], predicted_time_ns)
            aim = self._locate(self.aim_spaces[h], predicted_time_ns)
            hi.active, hi.aim_valid = grip is not None, aim is not None
            hi.grip, hi.aim = grip or Pose(), aim or Pose()
            hi.squeeze = self._float(self.a_squeeze, h)
            hi.trigger = self._float(self.a_trigger, h)
            hi.primary = self._bool(self.a_primary, h)
            hi.secondary = self._bool(self.a_secondary, h)
            hi.thumb_x = self._vec2_x(self.a_thumb, h)
            hi.thumb_click = self._bool(self.a_thumb_click, h)
            hi.menu = self._bool(self.a_menu, h)
            hi.primary_pressed = hi.primary and not prev.primary
            hi.primary_released = prev.primary and not hi.primary
            hi.secondary_pressed = hi.secondary and not prev.secondary
            hi.thumb_click_pressed = hi.thumb_click and not prev.thumb_click
            hi.menu_pressed = hi.menu and not prev.menu
        self._prev = frame.hands
        return frame

    def haptic(self, hand: int, amplitude: float = 0.5, duration_s: float = 0.05) -> None:
        vib = xr.HapticVibration(duration=int(duration_s * 1e9), frequency=0.0, amplitude=amplitude)
        self.fn.xrApplyHapticFeedback(self.session, ctypes.byref(
            xr.HapticActionInfo(action=self.a_haptic, subaction_path=self.hand_paths[hand])),
            ctypes.cast(ctypes.byref(vib), ctypes.POINTER(xr.HapticBaseHeader)))

    def destroy(self) -> None:
        """Destroy only what this created; ovxr's handles are borrowed."""
        for space in [*self.grip_spaces, *self.aim_spaces, self.view_space]:
            self.fn.xrDestroySpace(space)
        self.fn.xrDestroyActionSet(self.action_set)
