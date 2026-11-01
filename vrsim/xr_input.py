"""Controller input on the OpenXR session that ovxr owns.

ovxr owns the instance, session and frame loop; we create only our own action set, action spaces
and a VIEW space, and resolve every entry point through ovxr's ``xrGetInstanceProcAddr`` (the
handles belong to ovxr's loader, not pyopenxr's). Poses are located at the frame's predicted
display time and mapped into stage space with ``Context.transform_openxr_pose``.

Bindings (Meta Quest Touch):
    squeeze (grip)  grab / throw          trigger   show pointer ray
    A / X (hold)    push-to-talk          B         undo last spawn
    Y               clear all props
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass, field

import numpy as np
import xr

from .mathutil import Pose

log = logging.getLogger(__name__)

LEFT, RIGHT = 0, 1
HANDS = ("left", "right")
_VALID = xr.SpaceLocationFlags.POSITION_VALID_BIT | xr.SpaceLocationFlags.ORIENTATION_VALID_BIT


@dataclass
class HandInput:
    active: bool = False
    grip: Pose | None = None
    aim: Pose | None = None
    squeeze: float = 0.0
    trigger: float = 0.0
    primary: bool = False  # A / X
    secondary: bool = False  # B / Y
    primary_pressed: bool = False  # edges since the previous sync
    primary_released: bool = False
    secondary_pressed: bool = False


@dataclass
class InputFrame:
    hands: list[HandInput] = field(default_factory=lambda: [HandInput(), HandInput()])
    head: Pose | None = None
    focused: bool = False


class _Fn:
    """Resolve OpenXR entry points through ovxr's instance."""

    def __init__(self, instance, get_proc_addr: int):
        self._instance = instance
        self._gipa = xr.PFN_xrGetInstanceProcAddr(get_proc_addr)
        self._cache = {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._cache:
            fn = xr.PFN_xrVoidFunction()
            rc = self._gipa(self._instance, name.encode(), ctypes.byref(fn))
            if rc < 0 or not fn:
                raise RuntimeError(f"xrGetInstanceProcAddr({name}) failed: {rc}")
            self._cache[name] = ctypes.cast(fn, getattr(xr, "PFN_" + name))
        return self._cache[name]


def _check(rc: int, what: str) -> int:
    if rc < 0:
        try:
            name = xr.Result(rc).name
        except ValueError:
            name = str(rc)
        raise RuntimeError(f"{what} failed: {name}")
    return rc


class XrInput:
    def __init__(self, context, handles):
        self.ctx = context
        self.instance = ctypes.cast(ctypes.c_void_p(handles.xr_instance), xr.Instance)
        self.session = ctypes.cast(ctypes.c_void_p(handles.xr_session), xr.Session)
        self.base_space = ctypes.cast(ctypes.c_void_p(handles.xr_space), xr.Space)
        self.fn = _Fn(self.instance, handles.xr_get_instance_proc_addr)
        self._prev = [HandInput(), HandInput()]
        self._warned_transform = False
        self.last_flags = [0, 0, 0]  # raw locate flags (left grip, right grip, head) for diagnostics
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
            action_set_name="vrsim", localized_action_set_name="VR SimReady", priority=0)),
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
        self.a_grip_pose = action("grip_pose", T.POSE_INPUT)
        self.a_aim_pose = action("aim_pose", T.POSE_INPUT)
        self.a_squeeze = action("squeeze", T.FLOAT_INPUT)
        self.a_trigger = action("trigger", T.FLOAT_INPUT)
        self.a_primary = action("primary", T.BOOLEAN_INPUT)
        self.a_secondary = action("secondary", T.BOOLEAN_INPUT)
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
            touch += [(self.a_grip_pose, f"{u}/input/grip/pose"), (self.a_aim_pose, f"{u}/input/aim/pose"),
                      (self.a_squeeze, f"{u}/input/squeeze/value"), (self.a_trigger, f"{u}/input/trigger/value"),
                      (self.a_haptic, f"{u}/output/haptic")]
        touch += [(self.a_primary, "/user/hand/left/input/x/click"),
                  (self.a_primary, "/user/hand/right/input/a/click"),
                  (self.a_secondary, "/user/hand/left/input/y/click"),
                  (self.a_secondary, "/user/hand/right/input/b/click")]
        suggest("/interaction_profiles/oculus/touch_controller", touch)
        # Generic fallback so the app still works on other runtimes/simulators.
        simple = []
        for h in HANDS:
            u = f"/user/hand/{h}"
            simple += [(self.a_grip_pose, f"{u}/input/grip/pose"), (self.a_aim_pose, f"{u}/input/aim/pose"),
                       (self.a_primary, f"{u}/input/menu/click"), (self.a_squeeze, f"{u}/input/select/click"),
                       (self.a_haptic, f"{u}/output/haptic")]
        suggest("/interaction_profiles/khr/simple_controller", simple)

        self.grip_spaces, self.aim_spaces = [], []
        for hp in self.hand_paths:
            for act, out in ((self.a_grip_pose, self.grip_spaces), (self.a_aim_pose, self.aim_spaces)):
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
            xr.SessionActionSetsAttachInfo(count_action_sets=1, action_sets=sets))),
            "xrAttachSessionActionSets (ovxr may already have attached its own action sets)")

    # --- per frame ---------------------------------------------------------------------------
    def _locate(self, space, time_ns: int, slot: int | None = None) -> Pose | None:
        loc = xr.SpaceLocation()
        rc = self.fn.xrLocateSpace(space, self.base_space, xr.Time(time_ns), ctypes.byref(loc))
        if slot is not None:
            self.last_flags[slot] = rc if rc < 0 else int(loc.location_flags)
        if rc < 0:
            return None
        if (int(loc.location_flags) & _VALID) != _VALID:
            return None
        p, o = loc.pose.position, loc.pose.orientation
        try:
            m = self.ctx.transform_openxr_pose((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        except Exception as exc:
            if not self._warned_transform:
                self._warned_transform = True
                log.warning("transform_openxr_pose failed (logged once): %s", exc)
            return None
        return Pose.from_row_major(m)

    def _float(self, action, hand: int) -> float:
        state = xr.ActionStateFloat()
        self.fn.xrGetActionStateFloat(self.session, ctypes.byref(
            xr.ActionStateGetInfo(action=action, subaction_path=self.hand_paths[hand])), ctypes.byref(state))
        return float(state.current_state) if state.is_active else 0.0

    def _bool(self, action, hand: int) -> bool:
        state = xr.ActionStateBoolean()
        self.fn.xrGetActionStateBoolean(self.session, ctypes.byref(
            xr.ActionStateGetInfo(action=action, subaction_path=self.hand_paths[hand])), ctypes.byref(state))
        return bool(state.is_active and state.current_state)

    def sync(self, predicted_time_ns: int) -> InputFrame:
        frame = InputFrame()
        active = (xr.ActiveActionSet * 1)(xr.ActiveActionSet(self.action_set, xr.NULL_PATH))
        rc = self.fn.xrSyncActions(self.session, ctypes.byref(
            xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=active)))
        frame.focused = rc == xr.Result.SUCCESS
        frame.head = self._locate(self.view_space, predicted_time_ns, slot=2)
        if not frame.focused:  # SESSION_NOT_FOCUSED: e.g. the Quest system menu is open
            self._prev = frame.hands
            return frame

        for h in (LEFT, RIGHT):
            hi = frame.hands[h]
            hi.grip = self._locate(self.grip_spaces[h], predicted_time_ns, slot=h)
            hi.aim = self._locate(self.aim_spaces[h], predicted_time_ns)
            hi.active = hi.grip is not None
            hi.squeeze = self._float(self.a_squeeze, h)
            hi.trigger = self._float(self.a_trigger, h)
            hi.primary = self._bool(self.a_primary, h)
            hi.secondary = self._bool(self.a_secondary, h)
            prev = self._prev[h]
            hi.primary_pressed = hi.primary and not prev.primary
            hi.primary_released = prev.primary and not hi.primary
            hi.secondary_pressed = hi.secondary and not prev.secondary
        self._prev = frame.hands
        return frame

    def haptic(self, hand: int, amplitude: float = 0.5, duration_s: float = 0.05, frequency: float = 0.0) -> None:
        vib = xr.HapticVibration(duration=int(duration_s * 1e9), frequency=frequency, amplitude=amplitude)
        self.fn.xrApplyHapticFeedback(self.session, ctypes.byref(
            xr.HapticActionInfo(action=self.a_haptic, subaction_path=self.hand_paths[hand])),
            ctypes.cast(ctypes.byref(vib), ctypes.POINTER(xr.HapticBaseHeader)))

    def destroy(self) -> None:
        """Destroy only what we created; ovxr's handles are borrowed."""
        for space in [*self.grip_spaces, *self.aim_spaces, self.view_space]:
            self.fn.xrDestroySpace(space)
        self.fn.xrDestroyActionSet(self.action_set)
