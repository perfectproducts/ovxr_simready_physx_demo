// VR SimReady Physics, native edition.
//
// Per frame (ovxr split loop, application-owned ordinals; validated offscreen first):
//   begin_frame -> controllers at predicted display time -> grab velocities into ovphysx
//   -> ovphysx step(s) -> read body poses -> write body/controller/HUD transforms at ordinal N
//   -> bind_frame(N) seals N and renders it -> submit_frame
// Spawns/removals happen between frames at their own sealed ordinals, which (and only which)
// ovphysx drains with update_from_ovstage.
#include "common.hpp"
#include "physics.hpp"
#include "sidecar.hpp"
#include "stage.hpp"
#include "xr_input.hpp"

#include <ovxr/ovxr.h>

#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace vrsim;
using Clock = std::chrono::steady_clock;

namespace
{

volatile std::sig_atomic_t g_running = 1;

constexpr int kSidecarPort = 47800;
constexpr double kPhysicsDt = 1.0 / 72.0;  // Quest 3 refresh; the scene substeps at 144 Hz
constexpr int kMaxProps = 40;
constexpr double kGrabOn = 0.55, kGrabOff = 0.35, kGrabReach = 0.12, kRayMax = 8.0;
constexpr double kMaxLinear = 12.0, kMaxAngular = 30.0, kThrowScale = 1.3;
constexpr double kTurnRate = 1.6;  // [rad/s] thumbstick turning of a placement or a held fixed prop
constexpr double kPartReach = 0.025;  // [m] grip within this of a jointed part takes the part, not its prop
constexpr double kPokeReach = 0.06;  // [m] controller to the toaster's cancel button for a trigger press
constexpr double kGlowEmission = 6.0;  // slot strip emissiveColor at full heater glow (Minimal rendering)
constexpr double kDeskPropMax = 0.35;  // [m] largest side of a prop that spawns on the desk
constexpr size_t kHudFloats = 6000 * 4 * 3;  // vrsim.hud.MAX_QUADS quads x 4 corners x xyz
const char* kHands[2] = { "/World/Hands/Left", "/World/Hands/Right" };
const char* kRays[2] = { "/World/Hands/LeftRay", "/World/Hands/RightRay" };
const char* kPanel = "/World/HUD/Panel";
const char* kText = "/World/HUD/Panel/Text";
const char* kBrowser = "/World/Browser";
const char* kHighlight = "/World/Browser/Highlight";
const char* kMarker = "/World/Hands/PlaceMarker";  // red sphere + arrow: where a new asset will stand
const char* kLoading = "/World/BrowserLoading";  // vrsim.browser loading overlay, in the environment
const char* kSpinner = "/World/BrowserLoading/Spinner";
const char* kAccents[3] = { "/World/HUD/Panel/AccentBusy", "/World/HUD/Panel/AccentOk", "/World/HUD/Panel/AccentError" };
const char* kRig = "/World/Rig";  // ovxr tracking anchor: world = rig * tracking pose
const char* kMonitor = "/World/ToasterMonitor";  // vrsim.toaster_monitor panel beside the toaster
// Info panel placement: (yaw left of the view [deg], drop below the eyes [m]) in order of preference,
// the first one that doesn't cut through the toaster or its monitor (vrsim.hud FOLLOW_* is the first).
const std::pair<double, double> kHudSpots[] = { { 35, 0.25 }, { 35, 0.05 }, { 55, 0.25 }, { 55, 0.05 },
                                                { -35, 0.25 }, { -35, 0.05 }, { 20, -0.1 } };
const double kHudDistance = 0.85;               // [m] from the head
const double kEyeHeight = 1.22;  // [m] seated eyes in the world: the 0.78 m desk plus ~0.44 m (VRSIM_EYE_HEIGHT)
const double kHudHalf[2] = { 0.36, 0.125 };     // [m] vrsim.hud PANEL_W / 2, PANEL_H / 2
const double kHudNav[2][2] = { { 0.36 - 0.165, -0.125 + 0.024 }, { 0.36 - 0.035, -0.125 + 0.024 } };  // vrsim.hud NAV_PREV / NEXT
const double kHudNavHalf = 0.018;               // vrsim.hud NAV_HALF
const double kHudClearance = 0.03;              // [m] gap kept to obstacles
const double kHudReplaceOverlap = 0.02;         // [m] an anchored panel moves when something reaches this far in
const double kMonitorSize[2] = { 0.40, 0.30 };  // [m] vrsim.toaster_monitor PANEL_W, PANEL_H
const Mat16 kHidden = Pose{ { 0, 0, -100 } }.matrix({ 1e-4, 1e-4, 1e-4 });

double seconds(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

Vec3 clampLength(Vec3 v, double maxLength)
{
    const double l = v.length();
    return l > maxLength ? v * (maxLength / l) : v;
}

std::vector<float> decodeBase64Floats(const std::string& in)
{
    static int table[256];
    static bool init = false;
    if (!init)
    {
        for (int& t : table)
            t = -1;
        const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i)
            table[static_cast<unsigned char>(chars[i])] = i;
        init = true;
    }
    std::vector<uint8_t> bytes;
    bytes.reserve(in.size() * 3 / 4);
    int val = 0, bits = -8;
    for (unsigned char c : in)
    {
        if (table[c] < 0)
            continue;
        val = (val << 6) + table[c];
        bits += 6;
        if (bits >= 0)
        {
            bytes.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
    return out;
}

// The Oculus -> "Meta Horizon" update moves the runtime but can leave the machine-wide
// ActiveRuntime pointing at the old path; fix that for this process only.
void ensureOpenXrRuntime()
{
    char existing[8];
    if (GetEnvironmentVariableA("XR_RUNTIME_JSON", existing, sizeof(existing)) > 0)
        return;
    char value[MAX_PATH] = {};
    DWORD size = sizeof(value);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Khronos\\OpenXR\\1", "ActiveRuntime", RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ,
                     nullptr, value, &size) == ERROR_SUCCESS &&
        std::filesystem::exists(value))
        return;
    for (const char* candidate : { "C:\\Program Files\\Meta Horizon\\Support\\oculus-runtime\\oculus_openxr_64.json",
                                   "C:\\Program Files\\Oculus\\Support\\oculus-runtime\\oculus_openxr_64.json" })
        if (std::filesystem::exists(candidate))
        {
            SetEnvironmentVariableA("XR_RUNTIME_JSON", candidate);
            logf("Registered OpenXR runtime '%s' is missing; using %s for this process", value, candidate);
            return;
        }
}

std::string envOr(const char* var, const std::string& fallback)
{
    char buf[1024];
    const DWORD n = GetEnvironmentVariableA(var, buf, sizeof(buf));
    return n > 0 && n < sizeof(buf) ? std::string(buf, n) : fallback;
}

bool fileContains(const std::string& path, const std::string& needle)
{
    std::ifstream file(path, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return text.find(needle) != std::string::npos;
}

struct Prop
{
    std::string root, name;
    ovstage_population_usd_reference_handle_t handle = 0;
    std::vector<ovx_primpath_t> bodies;
    std::vector<std::string> bodyPaths;
    std::vector<std::array<double, 4>> spheres;  // per body: center in the body frame, radius [m]
    std::vector<bool> movable;  // per body: a jointed part (lid, temple) the second hand may take
    // Fixed in place: per body, the pose its non-movable bodies are held at (velocity-driven every
    // frame); only the jointed parts still move. Empty = free.
    std::vector<Pose> anchor;
    bool fixOnSpawn = false;  // placed at the marker: anchor it once physics reports its bodies
    double radius = 0.1, scale = 1.0;
    std::string device;  // "toaster": a prop driven by FMUs in the sidecar (vrsim/toaster.py); kept by undo/clear
};

// One operated joint of a device prop, measured from body poses (vrsim.toaster.joint_frames):
// linear = (part position in the base body's frame - origin) . axis, angular = turn about axis.
struct DeviceJoint
{
    int body = -1;  // index into the prop's bodies
    Vec3 origin, axis{ 1, 0, 0 };
    bool linear = true;
    double offset = 0, lower = -1e9, upper = 1e9;
    double turns = 0, lastAngle = 0;  // angular: unwrapped reading [deg]
    bool seen = false;
    double value = 0;  // last reading in joint units (m or deg)
    double hold = 0;   // angular: held here while no hand turns it (the dial's friction)
};

// The dbeyond smart toaster: the app measures lever, dial and cancel button and applies the FMU
// outputs the sidecar sends back (latch, toast pose and colour, heater glow).
struct Toaster
{
    int prop = -1;
    int base = -1;  // the housing body: the toaster frame
    DeviceJoint lever, dial, cancel;
    std::string toast, toastMaterial, glowLight, glowShader;
    double glowMax = 30000, latchedBelow = -0.03;
    double leverTarget = -0.0005;
    std::array<double, 4> toastLocal{ 0, -0.026, 0.17, 0 };  // x, y, z [m], tumble about X [deg]
    std::array<float, 3> toastColor{ 0.93f, 0.85f, 0.68f };
    float glow = 0;
    bool colorDirty = true, glowDirty = true;
    double cancelPressUntil = -1;  // trigger poke on the cancel button: held in until then [s]
    double lastSent = -1;
};

// One rigid body of a prop. Jointed assets have several.
struct BodyRef
{
    int prop = -1, body = -1;
};

// A static scene collider as its convex hull (vrsim.scene.write_placement_hulls): placement rays.
struct Hull
{
    Vec3 lo, hi;
    std::vector<std::array<double, 4>> planes;  // n.p + d <= 0 inside
};

struct Hand
{
    int prop = -1;  // index into props, -1 = empty
    int body = -1;  // index into that prop's bodies: the one this hand touched
    Pose offset;    // that body's pose in the grip frame
    // The first hand on a prop holds all of it rigidly (rel: every body's pose relative to the
    // touched one). Only then can the other hand take a movable part, which it holds alone.
    bool whole = true;
    std::vector<Pose> rel;
    bool pulling = false, squeezing = false, triggerDown = false;
    bool triggerClicked = false;  // this frame
    Pose grip;
    std::deque<std::pair<double, Vec3>> history;  // (time, grip position)
    double rayLength = 0;
    Vec3 lastAngular;

    Vec3 velocity() const
    {
        if (history.size() < 2)
            return {};
        const auto& [t0, p0] = history.front();
        const auto& [t1, p1] = history.back();
        return (p1 - p0) * (1.0 / std::max(t1 - t0, 1e-3));
    }
};

// In-VR asset browser: a page layer from the sidecar (vrsim/browser.py) referenced at /World/Browser.
struct Browser
{
    bool open = false;
    ovstage_population_usd_reference_handle_t handle = 0;
    int cols = 4, rows = 3, count = 0;
    double tileW = 0.26, tileH = 0.3, gap = 0.03, ox = 0, oy = 0, pad = 0.012;
    std::array<double, 4> closeRect{}, handleRect{};  // panel-local x0, y0, x1, y1
    Pose pose;
    bool placed = false;
    int hover = -1;        // tile index under a pointer
    int reportedHover = -1;  // last hover sent to the sidecar (it shows that asset's details)
    int hoverWidget = 0;   // 1 = close button, 2 = move handle
    int movingHand = -1;   // hand dragging the panel by its handle
    Pose moveOffset;       // panel pose in that hand's aim frame
    bool loading = false;  // a page is being built: show the spinner
    std::array<bool, 2> triggerDown{}, stickArmed{ true, true };
};

// The toaster's FMU monitor (vrsim/toaster_monitor.py): a panel layer referenced at kMonitor with one
// fixed-budget quad mesh per colour; the sidecar sends rectangles, the app rewrites the points.
struct MonitorPanel
{
    struct Layer
    {
        std::string prim;
        size_t budget = 0;
        float z = 0.002f;
        std::vector<float> points;
        bool pending = false;
    };
    ovstage_population_usd_reference_handle_t handle = 0;
    bool on = false;  // hidden until the menu button or "toggle signal monitor"
    std::vector<Layer> layers;
    Vec3 offset{ 0, -0.34, 0.19 };  // panel centre in the toaster frame
    std::string swatch;             // its toast-colour shader
};

struct Hud
{
    std::vector<float> points;
    bool pending = false;
    std::string kind = "info";
    double hold = 0;
    Clock::time_point posted;
    std::optional<Vec3> pos;
    Vec3 anchor;            // where the panel settles; changes only on (re)appear or when out of view
    Pose pose;              // last placed panel pose (for pointer hits)
    bool visible = false;
    std::array<bool, 2> triggerDown{};
    bool nav = false;  // the text has page buttons (help): ◀ at kHudNav[0], ▶ at kHudNav[1]
};

class App
{
public:
    App(const std::string& envUsd, const std::string& schemasRoot, uint32_t framesInFlight, bool foveation,
        bool minimalRendering)
        : m_link(kSidecarPort)
    {
        m_stage = std::make_unique<Stage>(envUsd, schemasRoot, minimalRendering);
        m_physics = std::make_unique<Physics>(*m_stage, 1);  // environment was populated at ordinal 1
        loadHulls((std::filesystem::path(envUsd).parent_path() / "placement_hulls.bin").string());

        const ovxr_setting_entry_t entries[] = {
            ovxr_setting_entry_int(OVXR_SETTING_DISPLAY_MODE, OVXR_DISPLAY_MODE_STEREO),
            ovxr_setting_entry_int(OVXR_SETTING_SESSION_MODE, OVXR_SESSION_MODE_VR),
            ovxr_setting_entry_int(OVXR_SETTING_ENVIRONMENT_BLEND_MODE, OVXR_ENVIRONMENT_BLEND_MODE_AUTO),
            ovxr_setting_entry_int(OVXR_SETTING_LOCAL_FOVEATION_MODE,
                                   foveation ? OVXR_LOCAL_FOVEATION_WARPED : OVXR_LOCAL_FOVEATION_DISABLED),
            ovxr_setting_entry_int(OVXR_SETTING_RUNTIME_FOVEATED_SUBMISSION_POLICY,
                                   foveation ? OVXR_FEATURE_AUTO : OVXR_FEATURE_DISABLED),
            ovxr_setting_entry_int(OVXR_SETTING_GRAPHICS_API, OVXR_GRAPHICS_API_VULKAN),
        };
        ovxr_settings_t settings{ entries, sizeof(entries) / sizeof(entries[0]) };
        static const char anchor[] = "/World/Rig";
        ovxr_desc_t desc{};
        desc.ovrtx_renderer = m_stage->renderer();
        desc.ovstage = m_stage->handle();
        desc.application_name = "vrsim.cpp";
        desc.runtime_prim_root = "/OVXR";
        desc.settings = &settings;
        // 2 lets the CPU prepare the next frame while the GPU renders this one; at 1, render
        // (~9 ms in bind) plus submit (~4.6 ms) just overran 13.9 ms and halved the rate to 36 fps.
        desc.max_frames_in_flight = framesInFlight;
        desc.create_render_products = true;
        desc.tracking_anchor_path = { anchor, sizeof(anchor) - 1 };
        desc.scene_info_policy = OVXR_STAGE_SCENE_INFO_AUTO;
        desc.up_axis = OVXR_STAGE_UP_AXIS_Z;
        desc.meters_per_unit = 1.0f;
        desc.foveation_unwarp = foveation;
        desc.ordinal_mode = OVXR_STAGE_ORDINAL_APPLICATION;
        // Minimal rendering: clone the environment's MinimalSource product per eye. Its shading
        // mode 3 evaluates emission, which the faked sky needs (vrsim.scene._fake_sky).
        static const char minimalSource[] = "/World/Render/MinimalSource";
        if (minimalRendering && fileContains(envUsd, "def RenderProduct \"MinimalSource\""))
            desc.source_render_product = { minimalSource, sizeof(minimalSource) - 1 };
        if (ovxr_create(&desc, &m_ctx) != OVXR_SUCCESS)
            die("ovxr_create", ovxr_get_last_error());
        ovxr_openxr_probe_result_t probe{};
        if (ovxr_probe_openxr(m_ctx, &probe) == OVXR_SUCCESS)
            logf("OpenXR: runtime='%s' system='%s' %ux%u", probe.runtime_name, probe.system_name,
                 probe.recommended_swapchain_width, probe.recommended_swapchain_height);
        else
            logf("OpenXR probe: %s (is Quest Link active?)", ovxr_get_last_error());
    }

    ~App()
    {
        m_xr.reset();
        if (ovxr_destroy(m_ctx, m_stage->current()) != OVXR_SUCCESS)
            logf("ovxr_destroy: %s", ovxr_get_last_error());
        m_physics.reset();
        m_stage.reset();
    }

    void run()
    {
        auto last = Clock::now(), lastStatus = last, waitingLog = Clock::time_point{};
        double accumulator = 0;
        int frames = 0;
        while (g_running)
        {
            ovxr_frame_plan_t plan{};
            const ovxr_result_t begun = ovxr_begin_frame(m_ctx, &plan);
            const auto now = Clock::now();
            if (begun != OVXR_SUCCESS)
            {
                if (seconds(waitingLog, now) > 10.0)
                {
                    logf("Waiting for headset: %s", ovxr_get_last_error());
                    waitingLog = now;
                }
                handleSidecar();
                Sleep(50);
                last = now;
                continue;
            }
            const double dt = std::min(seconds(last, now), 0.1);
            m_dt = dt;
            last = now;

            if (!m_xr)
            {
                ovxr_openxr_handles_t handles{};
                if (ovxr_get_openxr_handles(m_ctx, &handles) == OVXR_SUCCESS)
                    m_xr = std::make_unique<XrInput>(m_ctx, handles);
            }
            InputFrame input;
            if (m_xr)
            {
                input = m_xr->sync(plan.timing.predicted_display_time_ns);
                if (input.headValid)
                    m_rawHead = input.head;
                if (input.headValid && !m_eyesLevelled)  // the first head pose: seated eye height, same spot and heading
                {
                    m_eyesLevelled = true;
                    levelEyes();
                }
                toWorld(input);
                if (input.headValid)
                    m_head = input.head;
                buttons(input);
                browserGrab(input);  // before interact(): grip on the handle must not grab props
                interact(input, seconds(m_start, now));
                place(input);
                turnFixed(input);
                browserInteract(input);
                hudInteract(input);
            }

            // Fixed-step physics against real time (at most 3 steps per frame).
            auto mark = Clock::now();
            auto lap = [&](double& bucket) {
                const auto t = Clock::now();
                bucket += seconds(mark, t) * 1000.0;
                mark = t;
            };
            accumulator = std::min(accumulator + dt, 3 * kPhysicsDt);
            const int steps = static_cast<int>(accumulator / kPhysicsDt + 1e-9);
            accumulator -= steps * kPhysicsDt;
            if (steps > 0)
            {
                // One grab velocity for the whole frame: it must close the gap over all of this
                // frame's steps. Re-applying a one-step velocity before every step from the same
                // (stale) body pose overshot by the step count and made held props jitter.
                driveGrabs(input, steps * kPhysicsDt);
                lap(m_prof.drive);
                for (int i = 0; i < steps; ++i)
                {
                    // The toaster's latch and dial hold fight springs and gravity on light parts:
                    // re-aim them before every step from fresh poses. Once per frame with 1-3
                    // steps, the latched lever jumped over a 13 mm range (the toast rides on it).
                    if (i > 0 && m_toaster)
                    {
                        m_physics->readBodies();
                        readToasterJoints();
                        driveToaster(kPhysicsDt);
                    }
                    m_physics->step(static_cast<float>(kPhysicsDt));
                }
                lap(m_prof.step);
                m_prof.steps += steps;
            }
            m_physics->readBodies();
            measureToaster(seconds(m_start, now));
            lap(m_prof.read);

            const ovstage_ordinal_t ordinal = m_stage->current();
            writeBodies(ordinal);
            writeVisuals(input, ordinal);
            lap(m_prof.write);
            bool sealed = false;
            const ovxr_result_t bound = ovxr_bind_frame(m_ctx, &plan, dt, ordinal, &sealed);
            lap(m_prof.bind);
            if (sealed)
                m_stage->resolve();
            if (bound != OVXR_SUCCESS)
            {
                ovxr_cancel_frame(m_ctx, &plan);
                if (bound != OVXR_NOT_READY)
                    logf("bind_frame: %s", ovxr_get_last_error());
                continue;
            }
            ovxr_frame_result_t result{};
            if (ovxr_submit_frame(m_ctx, &plan, nullptr, &result) != OVXR_SUCCESS)
                logf("submit_frame: %s", ovxr_get_last_error());
            lap(m_prof.submit);

            handleSidecar();  // population changes between frames, at their own ordinals
            ++frames;
            if (seconds(lastStatus, now) > 2.0)
            {
                const double n = std::max(frames, 1);
                logf("status | %.1f fps | dt %.1f ms | bodies %zu | props %zu | sidecar %s | per frame ms: "
                     "drive %.2f step %.2f (%.1f steps) read %.2f write %.2f bind %.2f submit %.2f",
                     frames / seconds(lastStatus, now), dt * 1000, m_physics->bodies().size(), m_props.size(),
                     m_link.connected() ? "up" : "down", m_prof.drive / n, m_prof.step / n, m_prof.steps / n,
                     m_prof.read / n, m_prof.write / n, m_prof.bind / n, m_prof.submit / n);
                lastStatus = now;
                frames = 0;
                m_prof = {};
            }
        }
    }

private:
    // --- input ---------------------------------------------------------------------------------
    // Tracking poses -> world through the rig (ovxr renders with the same anchor, see kRig).
    void toWorld(InputFrame& in) const
    {
        in.head = m_rig * in.head;
        for (HandInput& hi : in.hands)
        {
            hi.grip = m_rig * hi.grip;
            hi.aim = m_rig * hi.aim;
        }
    }

    // "realign": turn and shift the rig so the head stands over the scene's seat spot (the world
    // origin, where config.EnvironmentConfig.offset puts the user) and looks along +Y at the desk.
    // Heading only; the floor stays the floor.
    // Lift the rig so the eyes are at the seated eye height (it only turns about Z, so world eye z =
    // rig z + tracking eye z). VRSIM_EYE_HEIGHT overrides it; 0 keeps tracking's own height.
    void levelEyes()
    {
        if (!m_rawHead)
            return;
#pragma warning(suppress : 4996)  // read once per recenter, nothing else writes the environment
        const char* env = std::getenv("VRSIM_EYE_HEIGHT");
        const double eye = env && *env ? std::atof(env) : kEyeHeight;
        if (eye <= 0)
            return;
        m_rig.p.z = eye - m_rawHead->p.z;
        logf("Rig: eyes at %.2f m (tracking %.2f m)", eye, m_rawHead->p.z);
    }

    void realign()
    {
        if (!m_rawHead)
            return;
        Vec3 f = m_rawHead->axis({ 0, 0, -1 });
        f.z = 0;
        if (f.length() < 1e-3)
            return;
        f = f.normalized();
        const Quat q = Quat::axisAngle({ 0, 0, 1 }, 3.14159265358979 / 2 - std::atan2(f.y, f.x));
        const Vec3 head{ m_rawHead->p.x, m_rawHead->p.y, 0 };
        m_rig = Pose{ Vec3{ 0, 0, 0 } - q.rotate(head), q };
        levelEyes();
        for (Hand& h : m_hands)
            h.history.clear();  // no throw from the jump
        m_hud.pos.reset();      // the info panel settles in front again
        if (m_xr)
            for (int h = 0; h < 2; ++h)
                m_xr->haptic(h, 0.4f, 0.05f);
        logf("Realigned: rig at (%.2f, %.2f), heading %.0f deg", m_rig.p.x, m_rig.p.y,
             std::atan2(f.y, f.x) * 180 / 3.14159265358979);
    }

    void buttons(const InputFrame& in)
    {
        const auto& h = in.hands;
        if (h[0].primaryPressed || h[1].primaryPressed)
        {
            m_link.send({ { "t", "talk" }, { "on", true } });
            m_xr->haptic(h[1].primaryPressed ? 1 : 0, 0.25f, 0.03f);
        }
        if ((h[0].primaryReleased || h[1].primaryReleased) && !h[0].primary && !h[1].primary)
            m_link.send({ { "t", "talk" }, { "on", false } });
        if (h[1].secondaryPressed)
            m_pendingSpawn ? cancelPlacement() : undo();
        if (h[0].secondaryPressed)
            clear();
        if ((h[0].menuPressed || h[1].menuPressed) && m_monitor.handle)
        {
            setMonitor(!m_monitor.on);
            m_xr->haptic(0, 0.3f, 0.03f);
        }
    }

    // Show / hide the toaster's FMU monitor (menu button, or "toggle signal monitor" via the sidecar).
    void setMonitor(bool on)
    {
        if (!m_monitor.handle)
            return;
        m_monitor.on = on;
        m_link.send({ { "t", "toaster_monitor" }, { "on", m_monitor.on } });  // the sidecar starts / stops frames
        if (m_toaster)
            m_toaster->colorDirty = true;  // repaint the swatch with the panel
    }

    // What the info panel must not cut through, as world spheres: the toaster's bodies and, while
    // shown, its FMU monitor.
    std::vector<std::pair<Vec3, double>> hudObstacles() const
    {
        std::vector<std::pair<Vec3, double>> out;
        if (!m_toaster)
            return out;
        const Toaster& ts = *m_toaster;
        for (size_t b = 0; b < m_props[ts.prop].bodyPaths.size(); ++b)
            if (const BodyState* s = bodyState(ts.prop, static_cast<int>(b)))
                out.push_back(bodySphere(*s, BodyRef{ ts.prop, static_cast<int>(b) }));
        const BodyState* base = bodyState(ts.prop, ts.base);
        if (m_monitor.handle && m_monitor.on && base)
            out.push_back({ (base->pose * Pose{ m_monitor.offset }).p, 0.5 * std::hypot(kMonitorSize[0], kMonitorSize[1]) });
        return out;
    }

    // An upright panel at pos facing the head horizontally (as the info panel is drawn).
    Pose facingHead(const Vec3& pos) const
    {
        Vec3 z = m_head->p - pos;
        z.z = 0;
        z = z.length() > 1e-3 ? z.normalized() : Vec3{ 0, -1, 0 };
        const Vec3 up{ 0, 0, 1 }, x = up.cross(z);
        const double m[16] = { x.x, x.y, x.z, 0, up.x, up.y, up.z, 0, z.x, z.y, z.z, 0, pos.x, pos.y, pos.z, 1 };
        return Pose::fromMatrix(m);
    }

    // How far [m] the info panel at pos reaches into the nearest obstacle (<= 0: clear).
    double hudOverlap(const Vec3& pos, const std::vector<std::pair<Vec3, double>>& obstacles) const
    {
        if (obstacles.empty())
            return -1.0;
        const Pose inv = facingHead(pos).inverse();
        double worst = -1.0;
        for (const auto& [centre, radius] : obstacles)
        {
            const Vec3 l = (inv * Pose{ centre }).p;
            const double dx = std::max(std::abs(l.x) - kHudHalf[0], 0.0), dy = std::max(std::abs(l.y) - kHudHalf[1], 0.0);
            worst = std::max(worst, radius + kHudClearance - std::sqrt(dx * dx + dy * dy + l.z * l.z));
        }
        return worst;
    }

    // The first clear spot of kHudSpots, or the least blocked one.
    Vec3 hudTarget(const Vec3& fwd, const std::vector<std::pair<Vec3, double>>& obstacles) const
    {
        Vec3 best;
        double bestOverlap = 1e9;
        for (const auto& [yaw, drop] : kHudSpots)
        {
            const Vec3 side = Quat::axisAngle({ 0, 0, 1 }, yaw * 3.14159265358979 / 180.0).rotate(fwd);
            const Vec3 pos = m_head->p + side * kHudDistance - Vec3{ 0, 0, drop };
            const double overlap = hudOverlap(pos, obstacles);
            if (overlap <= 0.0)
                return pos;
            if (overlap < bestOverlap)
                best = pos, bestOverlap = overlap;
        }
        return best;
    }

    // A body's bounding sphere in the world; without per-body spheres, the whole prop's.
    std::pair<Vec3, double> bodySphere(const BodyState& b, const BodyRef& ref) const
    {
        const Prop& prop = m_props[ref.prop];
        if (ref.body >= static_cast<int>(prop.spheres.size()))
            return { b.pose.p, prop.radius };
        const auto& s = prop.spheres[ref.body];
        return { b.pose.p + b.pose.q.rotate(Vec3{ s[0], s[1], s[2] }), s[3] };
    }

    std::optional<std::pair<BodyRef, double>> pickRay(const Vec3& origin, const Vec3& dir) const
    {
        std::optional<std::pair<BodyRef, double>> best;
        for (const BodyState& b : m_physics->bodies())
        {
            const auto it = m_bodyProp.find(b.path);
            if (it == m_bodyProp.end())
                continue;
            const auto [center, radius] = bodySphere(b, it->second);
            const Vec3 rel = center - origin;
            const double along = rel.dot(dir);
            const double perp = (rel - dir * along).length();
            if (along > 0 && along < kRayMax && perp <= radius * 0.9 && (!best || along < best->second))
                best = std::make_pair(it->second, along);
        }
        return best;
    }

    std::optional<BodyRef> pickNear(const Vec3& point) const
    {
        // A jointed part the hand is at wins over its prop's body, whose bounding sphere usually
        // contains it (the toaster housing's 18 cm sphere swallows the dial and the lever).
        std::optional<BodyRef> best, bestPart;
        double bestGap = kGrabReach, bestPartGap = kPartReach;
        for (const BodyState& b : m_physics->bodies())
        {
            const auto it = m_bodyProp.find(b.path);
            if (it == m_bodyProp.end())
                continue;
            const auto [center, radius] = bodySphere(b, it->second);
            const double gap = (center - point).length() - radius;
            if (gap <= bestGap)
                bestGap = gap, best = it->second;
            const Prop& prop = m_props[it->second.prop];
            const bool movable = it->second.body < static_cast<int>(prop.movable.size()) && prop.movable[it->second.body];
            if (movable && gap <= bestPartGap)
                bestPartGap = gap, bestPart = it->second;
        }
        return bestPart ? bestPart : best;
    }

    const BodyState* bodyState(int prop, int body) const
    {
        if (prop < 0 || body < 0 || body >= static_cast<int>(m_props[prop].bodies.size()))
            return nullptr;
        const ovx_primpath_t path = m_props[prop].bodies[body];
        for (const BodyState& b : m_physics->bodies())
            if (b.path == path)
                return &b;
        return nullptr;
    }

    void interact(const InputFrame& in, double t)
    {
        for (int h = 0; h < 2; ++h)
        {
            Hand& hand = m_hands[h];
            const HandInput& hi = in.hands[h];
            if (!hi.active)
            {
                hand.rayLength = 0;
                continue;
            }
            hand.history.emplace_back(t, hi.grip.p);
            while (hand.history.size() > 6)
                hand.history.pop_front();

            const bool was = hand.squeezing;
            hand.squeezing = hi.squeeze > (was ? kGrabOff : kGrabOn);
            if (hand.squeezing && !was)
                grab(h, hi);
            else if (was && !hand.squeezing && hand.prop >= 0)
                release(h);

            // Trigger while holding a prop whole: fix it where it is, or free a fixed one.
            const bool click = hi.trigger > (hand.triggerDown ? 0.3f : 0.7f);
            hand.triggerClicked = click && !hand.triggerDown;
            hand.triggerDown = click;
            hand.grip = hi.grip;
            if (hand.triggerClicked && hand.prop >= 0 && hand.whole)
                setFixed(hand.prop, m_props[hand.prop].anchor.empty());
            else if (hand.triggerClicked && hand.prop < 0 && !m_pendingSpawn)
                pokeToaster(h, hi);

            hand.rayLength = 0;
            if (hand.prop < 0 && hi.aimValid && hi.trigger > 0.1f)
            {
                const Vec3 d = hi.aim.axis({ 0, 0, -1 });
                const auto hit = pickRay(hi.aim.p, d);
                hand.rayLength = hit ? hit->second : kRayMax * 0.4;
            }
        }
    }

    void grab(int h, const HandInput& hi)
    {
        Hand& hand = m_hands[h];
        hand.pulling = false;
        std::optional<BodyRef> ref = pickNear(hi.grip.p);
        if (!ref && hi.aimValid)  // force-pull whatever the hand points at
            if (auto hit = pickRay(hi.aim.p, hi.aim.axis({ 0, 0, -1 })))
                ref = hit->first, hand.pulling = true;
        const BodyState* body = ref ? bodyState(ref->prop, ref->body) : nullptr;
        if (!body)
            return;
        Hand& other = m_hands[1 - h];
        Prop& prop = m_props[ref->prop];
        const bool fixed = !prop.anchor.empty();
        const bool movable = ref->body < static_cast<int>(prop.movable.size()) && prop.movable[ref->body];
        // A part moves alone while the prop is fixed, or while the other hand holds the rest.
        const bool part = movable && (fixed || (other.prop == ref->prop && other.whole && other.body != ref->body));
        if (other.prop == ref->prop && (other.body == ref->body || (!part && !fixed)))  // take it over
            other.prop = other.body = -1;
        hand.prop = ref->prop;
        hand.body = ref->body;
        hand.whole = !part;
        hand.pulling = hand.pulling && !part;
        hand.offset = hi.grip.inverse() * body->pose;
        if (hand.whole)
            captureWhole(hand);
        m_xr->haptic(h, 0.6f, 0.04f);
    }

    // Record every body's pose relative to the touched one, so the hand carries them rigidly.
    void captureWhole(Hand& hand)
    {
        const Prop& prop = m_props[hand.prop];
        const BodyState* touched = bodyState(hand.prop, hand.body);
        hand.rel.assign(prop.bodies.size(), Pose{});
        for (size_t b = 0; touched && b < prop.bodies.size(); ++b)
            if (const BodyState* s = bodyState(hand.prop, static_cast<int>(b)))
                hand.rel[b] = touched->pose.inverse() * s->pose;
    }

    void release(int h)
    {
        Hand& hand = m_hands[h];
        Hand& other = m_hands[1 - h];
        if (hand.prop >= 0 && !m_props[hand.prop].anchor.empty())
        {
            // Fixed: the anchor keeps holding it; a released part swings on as it was moving,
            // except on the toaster, whose dial has no friction and would spin on.
            const BodyState* s = !hand.whole && !m_props[hand.prop].device.empty() ? bodyState(hand.prop, hand.body) : nullptr;
            if (s)
                m_physics->setBodyVelocities(m_props[hand.prop].root, { s->path }, { Vec3{} }, { Vec3{} });
        }
        else if (hand.prop >= 0 && other.prop == hand.prop)
        {
            // Letting go of a part: the holding hand keeps it at its new angle. Letting go of the
            // whole: the part's hand now holds everything. Either way, no throw.
            other.whole = true;
            captureWhole(other);
        }
        else if (!hand.pulling && hand.prop >= 0)
            throwProp(hand);
        m_lastProp = hand.prop;
        hand.prop = hand.body = -1;
        hand.pulling = false;
        m_xr->haptic(h, 0.3f, 0.02f);
    }

    // Fix a prop where it is now (its non-movable bodies stay put, its jointed parts still move),
    // or free it: a hand holding it whole then carries all of it again.
    void setFixed(int index, bool fixed)
    {
        Prop& prop = m_props[index];
        prop.anchor.clear();
        if (fixed)
            for (size_t b = 0; b < prop.bodies.size(); ++b)
            {
                const BodyState* s = bodyState(index, static_cast<int>(b));
                prop.anchor.push_back(s ? s->pose : Pose{});
            }
        for (int h = 0; h < 2; ++h)
            if (m_hands[h].prop == index)
            {
                m_hands[h].pulling = false;
                if (const BodyState* s = bodyState(index, m_hands[h].body))  // it may have been turned
                    m_hands[h].offset = m_hands[h].grip.inverse() * s->pose;
                if (m_hands[h].whole)
                    captureWhole(m_hands[h]);
                m_xr->haptic(h, fixed ? 0.8f : 0.5f, fixed ? 0.08f : 0.04f);
            }
        const int parts = static_cast<int>(std::count(prop.movable.begin(), prop.movable.end(), true));
        m_link.send({ { "t", fixed ? "fixed" : "unfixed" }, { "name", prop.name }, { "parts", parts } });
    }

    // Voice "fix it" / "free it": a held prop, else the one last let go of, else the newest.
    void fixCommand(bool fixed)
    {
        int index = -1;
        for (const Hand& hand : m_hands)
            if (hand.prop >= 0 && (index < 0 || hand.whole))
                index = hand.prop;
        if (index < 0)
            index = m_lastProp >= 0 && m_lastProp < static_cast<int>(m_props.size()) ? m_lastProp
                                                                                    : static_cast<int>(m_props.size()) - 1;
        if (index < 0)
            m_link.send({ { "t", "nothing_to_fix" } });
        else
            setFixed(index, fixed);
    }

    void loadHulls(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            logf("Placement: no %s; new props drop instead", path.c_str());
            return;
        }
        const auto bytes = static_cast<size_t>(file.tellg());
        std::vector<float> data(bytes / sizeof(float));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
        size_t i = 0;
        const size_t count = data.empty() ? 0 : static_cast<size_t>(data[i++]);
        for (size_t h = 0; h < count && i + 7 <= data.size(); ++h)
        {
            Hull hull;
            const size_t planes = static_cast<size_t>(data[i]);
            hull.lo = { data[i + 1], data[i + 2], data[i + 3] };
            hull.hi = { data[i + 4], data[i + 5], data[i + 6] };
            i += 7;
            for (size_t p = 0; p < planes && i + 4 <= data.size(); ++p, i += 4)
                hull.planes.push_back({ data[i], data[i + 1], data[i + 2], data[i + 3] });
            m_hulls.push_back(std::move(hull));
        }
        logf("Placement: %zu scene hulls from %s", m_hulls.size(), path.c_str());
    }

    // First scene collider a ray hits: distance and the surface normal there.
    std::optional<std::pair<double, Vec3>> castScene(const Vec3& origin, const Vec3& dir) const
    {
        std::optional<std::pair<double, Vec3>> best;
        for (const Hull& hull : m_hulls)
        {
            double t0 = 0, t1 = best ? best->first : kRayMax;
            // slab test on the bounds first
            bool miss = false;
            for (int a = 0; a < 3 && !miss; ++a)
            {
                const double o = a == 0 ? origin.x : a == 1 ? origin.y : origin.z;
                const double d = a == 0 ? dir.x : a == 1 ? dir.y : dir.z;
                const double lo = a == 0 ? hull.lo.x : a == 1 ? hull.lo.y : hull.lo.z;
                const double hi = a == 0 ? hull.hi.x : a == 1 ? hull.hi.y : hull.hi.z;
                if (std::abs(d) < 1e-12)
                    miss = o < lo || o > hi;
                else
                {
                    double ta = (lo - o) / d, tb = (hi - o) / d;
                    if (ta > tb)
                        std::swap(ta, tb);
                    t0 = std::max(t0, ta);
                    t1 = std::min(t1, tb);
                    miss = t0 > t1;
                }
            }
            if (miss)
                continue;
            t0 = 0;
            t1 = best ? best->first : kRayMax;
            Vec3 normal{ 0, 0, 1 };
            for (const auto& pl : hull.planes)
            {
                const Vec3 n{ pl[0], pl[1], pl[2] };
                const double den = n.dot(dir), num = -(n.dot(origin) + pl[3]);
                if (std::abs(den) < 1e-12)
                {
                    if (num < 0)
                    {
                        t0 = t1 + 1;
                        break;
                    }
                }
                else if (den < 0)
                {
                    if (num / den > t0)
                        t0 = num / den, normal = n;
                }
                else
                    t1 = std::min(t1, num / den);
                if (t0 > t1)
                    break;
            }
            if (t0 <= t1 && t0 > 0)
                best = std::make_pair(t0, normal);
        }
        return best;
    }

    // A catalog or voice spawn waits for a spot: a red marker follows the pointer ray of one hand
    // (the right, or the one whose trigger is touched alone) on upward-facing scene surfaces, the
    // thumbstick turns it (its arrow is the asset's facing), and the trigger spawns the asset
    // there, fixed. A batch stands in a row across the ray.
    void place(const InputFrame& in)
    {
        if (!m_pendingSpawn)
            return;
        for (int h = 0; h < 2; ++h)
            if (in.hands[h].trigger > 0.2f && in.hands[1 - h].trigger < 0.05f && m_hands[h].prop < 0)
                m_placeHand = h;
        const HandInput& hi = in.hands[m_placeHand];
        Hand& hand = m_hands[m_placeHand];
        if (!hi.aimValid || hand.prop >= 0)
            return;
        const Vec3 dir = hi.aim.axis({ 0, 0, -1 });
        const auto hit = castScene(hi.aim.p, dir);
        hand.rayLength = hit ? hit->first : kRayMax * 0.4;
        if (hit && hit->second.z > 0.6)
        {
            m_placeAt = hi.aim.p + dir * hit->first;
            const Vec3 side = dir.cross({ 0, 0, 1 });
            m_placeSide = side.length() > 1e-3 ? side.normalized() : Vec3{ 1, 0, 0 };
            m_placeValid = true;
        }
        if (std::abs(hi.thumbX) > 0.25f)
            m_placeYaw -= hi.thumbX * kTurnRate * m_dt;
        if (hand.triggerClicked && m_placeValid)
        {
            nlohmann::json msg = *m_pendingSpawn;
            m_pendingSpawn.reset();
            msg["at_marker"] = true;
            spawn(msg);
            m_xr->haptic(m_placeHand, 0.8f, 0.08f);
        }
    }

    void cancelPlacement()
    {
        m_pendingSpawn.reset();
        m_link.send({ { "t", "placing_cancelled" } });
    }

    // Thumbstick left/right while holding a fixed prop turns it about its centre's vertical axis,
    // so it stays standing where it is.
    void turnFixed(const InputFrame& in)
    {
        for (int h = 0; h < 2; ++h)
        {
            const Hand& hand = m_hands[h];
            const float x = in.hands[h].thumbX;
            if (hand.prop < 0 || !hand.whole || std::abs(x) < 0.25f)
                continue;
            Prop& prop = m_props[hand.prop];
            if (prop.anchor.empty())
                continue;
            Vec3 center;
            int n = 0;
            for (size_t b = 0; b < prop.anchor.size(); ++b)
            {
                if (b < prop.movable.size() && prop.movable[b])
                    continue;
                const Pose& a = prop.anchor[b];
                center = center + (b < prop.spheres.size() ? a.p + a.q.rotate({ prop.spheres[b][0], prop.spheres[b][1], prop.spheres[b][2] }) : a.p);
                ++n;
            }
            if (n == 0)
                continue;
            center = center * (1.0 / n);
            const Quat turn = Quat::axisAngle({ 0, 0, 1 }, -x * kTurnRate * m_dt);
            for (Pose& a : prop.anchor)
            {
                Vec3 off = a.p - center;
                off = turn.rotate(off);
                a = Pose{ Vec3{ center.x + off.x, center.y + off.y, a.p.z }, (turn * a.q).normalized() };
            }
        }
    }

    // Hold the non-movable bodies of fixed props at their anchor poses.
    void driveAnchors(double horizon)
    {
        for (size_t i = 0; i < m_props.size(); ++i)
        {
            Prop& prop = m_props[i];
            if (prop.fixOnSpawn)  // placed at the marker: fixed where it first appears
            {
                std::vector<Pose> anchor;
                for (size_t b = 0; b < prop.bodies.size(); ++b)
                    if (const BodyState* s = bodyState(static_cast<int>(i), static_cast<int>(b)))
                        anchor.push_back(s->pose);
                if (anchor.size() != prop.bodies.size())
                    continue;
                prop.anchor = std::move(anchor);
                prop.fixOnSpawn = false;
            }
            std::vector<ovx_primpath_t> paths;
            std::vector<Vec3> linear, angular;
            for (size_t b = 0; b < prop.anchor.size(); ++b)
            {
                const bool part = b < prop.movable.size() && prop.movable[b];
                const BodyState* s = part ? nullptr : bodyState(static_cast<int>(i), static_cast<int>(b));
                if (!s)
                    continue;
                const Pose& t = prop.anchor[b];
                paths.push_back(s->path);
                linear.push_back(clampLength((t.p - s->pose.p) * (1.0 / horizon), kMaxLinear));
                angular.push_back(clampLength((t.q * s->pose.q.conj()).rotationVector() * (1.0 / horizon), kMaxAngular));
            }
            m_physics->setBodyVelocities(prop.root, paths, linear, angular);
        }
    }

    // The hand's velocity, carried rigidly: each body gets v + w x (its offset from the touched body).
    void throwProp(const Hand& hand)
    {
        const Prop& prop = m_props[hand.prop];
        const BodyState* touched = bodyState(hand.prop, hand.body);
        if (!touched)
            return;
        const Vec3 v = hand.velocity() * kThrowScale, w = hand.lastAngular;
        std::vector<ovx_primpath_t> paths;
        std::vector<Vec3> linear, angular;
        for (size_t b = 0; b < prop.bodies.size(); ++b)
            if (const BodyState* s = bodyState(hand.prop, static_cast<int>(b)))
            {
                paths.push_back(s->path);
                linear.push_back(v + w.cross(s->pose.p - touched->pose.p));
                angular.push_back(w);
            }
        m_physics->setBodyVelocities(prop.root, paths, linear, angular);
    }

    // Velocity that reaches the hand target by the end of this frame's physics; contacts stay
    // physical, pulled props fly to the hand, and the last velocity becomes the throw.
    void driveGrabs(const InputFrame& in, double horizon)
    {
        driveAnchors(horizon);
        driveToaster(kPhysicsDt);  // one step ahead, re-aimed per step; a hand on a part overrides it
        for (int h = 0; h < 2; ++h)
        {
            Hand& hand = m_hands[h];
            if (hand.prop < 0 || !in.hands[h].active)
                continue;
            if (hand.whole && !m_props[hand.prop].anchor.empty())
                continue;  // holding a fixed prop: the anchor drives it
            if (!bodyState(hand.prop, hand.body))
            {
                hand.prop = hand.body = -1;
                continue;
            }
            if (hand.pulling)
            {
                hand.offset.p = hand.offset.p * 0.85;
                hand.pulling = hand.offset.p.length() > 0.03;
            }
            const Prop& prop = m_props[hand.prop];
            const Hand& other = m_hands[1 - h];
            const int partOfOther = other.prop == hand.prop && !other.whole ? other.body : -1;
            const Pose target = in.hands[h].grip * hand.offset;
            std::vector<ovx_primpath_t> paths;
            std::vector<Vec3> linear, angular;
            for (size_t b = 0; b < prop.bodies.size(); ++b)
            {
                const bool mine = hand.whole ? static_cast<int>(b) != partOfOther : static_cast<int>(b) == hand.body;
                const BodyState* s = mine ? bodyState(hand.prop, static_cast<int>(b)) : nullptr;
                if (!s)
                    continue;
                const Pose t = hand.whole && b < hand.rel.size() ? target * hand.rel[b] : target;
                paths.push_back(s->path);
                linear.push_back(clampLength((t.p - s->pose.p) * (1.0 / horizon), kMaxLinear));
                angular.push_back(clampLength((t.q * s->pose.q.conj()).rotationVector() * (1.0 / horizon), kMaxAngular));
                if (static_cast<int>(b) == hand.body)
                    hand.lastAngular = angular.back();
            }
            m_physics->setBodyVelocities(prop.root, paths, linear, angular);
        }
    }

    // --- smart toaster (vrsim/toaster.py runs its FMUs) -------------------------------------------
    int bodyIndex(const Prop& prop, const std::string& rel) const
    {
        const std::string path = prop.root + "/" + rel;
        for (size_t b = 0; b < prop.bodyPaths.size(); ++b)
            if (prop.bodyPaths[b] == path)
                return static_cast<int>(b);
        return -1;
    }

    void setupToaster(int index, const nlohmann::json& msg)
    {
        const Prop& prop = m_props[index];
        Toaster t;
        t.prop = index;
        t.base = bodyIndex(prop, msg.value("base", std::string("Asset/Geometry/body_obj_01")));
        const auto joints = msg.value("joints", nlohmann::json::object());
        auto joint = [&](const char* role, DeviceJoint& j) {
            if (!joints.contains(role))
                return;
            const auto& f = joints[role];
            j.body = bodyIndex(prop, f.value("body", std::string()));
            const auto o = f.value("origin", std::vector<double>{ 0, 0, 0 });
            const auto a = f.value("axis", std::vector<double>{ 1, 0, 0 });
            j.origin = { o[0], o[1], o[2] };
            j.axis = Vec3{ a[0], a[1], a[2] }.normalized();
            j.linear = f.value("kind", std::string("linear")) == "linear";
            j.offset = f.value("offset", 0.0);
            if (f.contains("lower") && f["lower"].is_number())
                j.lower = f["lower"];
            if (f.contains("upper") && f["upper"].is_number())
                j.upper = f["upper"];
        };
        joint("lever", t.lever);
        joint("dial", t.dial);
        joint("cancel", t.cancel);
        t.toast = prop.root + "/" + msg.value("toast", std::string("Asset/Toast"));
        t.toastMaterial = prop.root + "/" + msg.value("toast_material", std::string("Asset/Toast/Looks/ToastMat/Surface"));
        t.glowLight = prop.root + "/" + msg.value("glow_light", std::string("Asset/ToasterECU/HeaterGlow"));
        t.glowShader = prop.root + "/" + msg.value("glow_shader", std::string("Asset/ToasterECU/SlotGlow/Glow/Shader"));
        t.glowMax = msg.value("glow_max", 30000.0);
        t.latchedBelow = msg.value("lever_latched_below", -0.03);
        if (t.base < 0 || t.lever.body < 0)
        {
            logf("Toaster: base or lever body not found; it stays a plain prop");
            return;
        }
        m_toaster = t;
        if (msg.contains("monitor") && msg["monitor"].is_object())
            setupMonitor(msg["monitor"]);
        logf("Toaster ready: lever body %d, dial %d, cancel %d", t.lever.body, t.dial.body, t.cancel.body);
    }

    void setupMonitor(const nlohmann::json& m)
    {
        if (!m_monitor.handle)
        {
            std::ifstream file(m.value("layer", ""), std::ios::binary);
            const std::string layer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (layer.empty())
            {
                logf("Toaster monitor: layer %s not readable", m.value("layer", "").c_str());
                return;
            }
            m_monitor.handle = m_stage->addReference(layer, kMonitor, m_stage->current());
            m_stage->commit();
        }
        m_monitor.layers.clear();
        const auto z = m.value("z", nlohmann::json::object());
        const auto layers = m.value("layers", nlohmann::json::object());  // a named copy: items() of a temporary dangles
        for (const auto& [name, budget] : layers.items())
        {
            MonitorPanel::Layer l;
            l.prim = std::string(kMonitor) + "/" + name;
            l.budget = budget.get<size_t>();
            l.z = z.contains(name) ? z[name].get<float>() : 0.002f;
            m_monitor.layers.push_back(std::move(l));
        }
        const auto o = m.value("offset", std::vector<double>{ 0, -0.34, 0.19 });
        m_monitor.offset = { o[0], o[1], o[2] };
        m_monitor.swatch = std::string(kMonitor) + "/" + m.value("swatch", std::string("Looks/Swatch/Surface"));
        m_link.send({ { "t", "toaster_monitor" }, { "on", m_monitor.on } });  // and ask for a full frame
        logf("Toaster monitor: %zu layers (left menu button shows / hides it)", m_monitor.layers.size());
    }

    // Rectangles (x0, y0, x1, y1, panel-local metres) -> the layer's quad points; unused quads collapse.
    void monitorFrame(const nlohmann::json& msg)
    {
        const auto layers = msg.value("layers", nlohmann::json::object());
        for (const auto& [name, data] : layers.items())
        {
            const std::string prim = std::string(kMonitor) + "/" + name;
            for (MonitorPanel::Layer& l : m_monitor.layers)
            {
                if (l.prim != prim)
                    continue;
                const std::vector<float> r = decodeBase64Floats(data.get<std::string>());
                l.points.assign(l.budget * 12, 0.0f);
                const size_t n = std::min(l.budget, r.size() / 4);
                for (size_t i = 0; i < n; ++i)
                {
                    const float x0 = r[4 * i], y0 = r[4 * i + 1], x1 = r[4 * i + 2], y1 = r[4 * i + 3];
                    const float corners[12] = { x0, y0, l.z, x1, y0, l.z, x1, y1, l.z, x0, y1, l.z };
                    std::copy_n(corners, 12, l.points.begin() + 12 * i);
                }
                l.pending = true;
            }
        }
    }

    // A part's pose in the toaster frame (the housing body).
    std::optional<Pose> toasterRel(const DeviceJoint& j) const
    {
        const BodyState* base = bodyState(m_toaster->prop, m_toaster->base);
        const BodyState* part = j.body >= 0 ? bodyState(m_toaster->prop, j.body) : nullptr;
        if (!base || !part)
            return std::nullopt;
        return base->pose.inverse() * part->pose;
    }

    void readJoint(DeviceJoint& j)
    {
        const auto rel = toasterRel(j);
        if (!rel)
            return;
        if (j.linear)
            j.value = std::clamp((rel->p - j.origin).dot(j.axis) + j.offset, j.lower - 1e-4, j.upper + 1e-4);
        else
        {
            const Quat& q = rel->q;
            double angle = 2.0 * std::atan2(Vec3{ q.x, q.y, q.z }.dot(j.axis), q.w) * 180.0 / 3.14159265358979;
            angle = std::remainder(angle, 360.0);
            if (j.seen)
                j.turns += std::remainder(angle - j.lastAngle, 360.0);
            else
                j.turns = angle;
            j.lastAngle = angle;
            j.value = j.turns + j.offset;
            if (!j.seen)
                j.hold = j.value;
        }
        j.seen = true;
    }

    void readToasterJoints()
    {
        readJoint(m_toaster->lever);
        readJoint(m_toaster->dial);
        readJoint(m_toaster->cancel);
    }

    // Joint readings to the sidecar's FMUs, up to 60 times a second.
    void measureToaster(double t)
    {
        if (!m_toaster)
            return;
        Toaster& ts = *m_toaster;
        readToasterJoints();
        double cancel = ts.cancel.value;
        if (t < ts.cancelPressUntil)  // poked: count it as pressed even if physics lags the 3 mm
            cancel = std::min(cancel, ts.cancel.lower + 0.0003);
        if (t - ts.lastSent < 1.0 / 60.0 || !ts.lever.seen)
            return;
        ts.lastSent = t;
        nlohmann::json msg = { { "t", "toaster_in" }, { "lever", ts.lever.value }, { "dial", ts.dial.value }, { "cancel", cancel } };
        if (m_head)  // world pose of the view: the sidecar's 3D sound (the ding) is heard from here
            msg["head"] = { m_head->p.x, m_head->p.y, m_head->p.z, m_head->q.x, m_head->q.y, m_head->q.z, m_head->q.w };
        m_link.send(msg);
    }

    bool handHolds(int prop, int body) const
    {
        for (const Hand& h : m_hands)
            if (h.prop == prop && h.body == body && !h.whole)
                return true;
        return false;
    }

    // Velocity that brings a joint to `target` (joint units: m, or deg) over the horizon.
    void driveJoint(const DeviceJoint& j, double target, double horizon)
    {
        const BodyState* base = bodyState(m_toaster->prop, m_toaster->base);
        const BodyState* part = j.body >= 0 ? bodyState(m_toaster->prop, j.body) : nullptr;
        if (!base || !part)
            return;
        const double gap = target - j.value;
        const Vec3 along = base->pose.q.rotate(j.axis);
        const Vec3 v = j.linear ? along * std::clamp(gap / horizon, -0.4, 0.4) : Vec3{};
        const Vec3 w = j.linear ? Vec3{} : along * std::clamp(gap * 3.14159265358979 / 180.0 / horizon, -8.0, 8.0);
        m_physics->setBodyVelocities(m_props[m_toaster->prop].root, { part->path }, { v }, { w });
    }

    // The ECU's latch holds the carriage down (its drive target in the binding, here a velocity
    // drive against the lever spring); a poke on the cancel button pushes it in for a moment.
    void driveToaster(double horizon)
    {
        if (!m_toaster)
            return;
        const Toaster& ts = *m_toaster;
        const double t = seconds(m_start, Clock::now());
        if (ts.leverTarget < ts.latchedBelow && !handHolds(ts.prop, ts.lever.body) && ts.lever.seen)
            driveJoint(ts.lever, std::max(ts.leverTarget, ts.lever.lower + 0.0002), horizon);
        if (t < ts.cancelPressUntil && ts.cancel.body >= 0 && !handHolds(ts.prop, ts.cancel.body))
            driveJoint(ts.cancel, ts.cancel.lower + 0.0001, horizon);
        // The dial has no friction and its weight is off the axle: hold it where the hand left it.
        if (ts.dial.body >= 0 && ts.dial.seen)
        {
            if (handHolds(ts.prop, ts.dial.body))
                m_toaster->dial.hold = ts.dial.value;
            else
                driveJoint(ts.dial, ts.dial.hold, horizon);
        }
    }

    // Trigger with the controller at the cancel button (not holding anything): press it. It is
    // 3 mm of travel, too little to push reliably by grabbing it.
    void pokeToaster(int h, const HandInput& hi)
    {
        if (!m_toaster || m_toaster->cancel.body < 0)
            return;
        const BodyState* button = bodyState(m_toaster->prop, m_toaster->cancel.body);
        if (!button || (button->pose.p - hi.grip.p).length() > kPokeReach)
            return;
        m_toaster->cancelPressUntil = seconds(m_start, Clock::now()) + 0.4;
        m_xr->haptic(h, 0.7f, 0.05f);
    }

    void onToaster(const nlohmann::json& msg)
    {
        if (!m_toaster)
            return;
        Toaster& ts = *m_toaster;
        const double before = ts.leverTarget;
        ts.leverTarget = msg.value("lever_target", ts.leverTarget);
        if (const auto v = msg.value("toast", std::vector<double>{}); v.size() == 4)
            std::copy(v.begin(), v.end(), ts.toastLocal.begin());
        if (const auto c = msg.value("toast_color", std::vector<float>{}); c.size() == 3)
        {
            std::array<float, 3> color{ c[0], c[1], c[2] };
            ts.colorDirty = ts.colorDirty || color != ts.toastColor;
            ts.toastColor = color;
        }
        const float glow = msg.value("glow", 0.0f);
        ts.glowDirty = ts.glowDirty || std::abs(glow - ts.glow) > 1.0f;
        ts.glow = glow;
        const bool latched = ts.leverTarget < ts.latchedBelow, wasLatched = before < ts.latchedBelow;
        if (m_xr && latched != wasLatched)  // click: the latch catches, or lets go with the ding
            for (int h = 0; h < 2; ++h)
                if (m_hands[h].prop == ts.prop || !latched)
                    m_xr->haptic(h, latched ? 0.5f : 0.9f, latched ? 0.03f : 0.12f);
    }

    void toasterTransforms(std::vector<std::string>& prims, std::vector<Mat16>& matrices) const
    {
        if (!m_toaster)
            return;
        const Toaster& ts = *m_toaster;
        const BodyState* base = bodyState(ts.prop, ts.base);
        if (!base)
            return;
        const auto& l = ts.toastLocal;
        const Pose toast{ { l[0], l[1], l[2] }, Quat::axisAngle({ 1, 0, 0 }, l[3] * 3.14159265358979 / 180.0) };
        prims.push_back(ts.toast);
        matrices.push_back((base->pose * toast).matrix());
        // The glow strips and the heater light sit outside the physics bodies: keep them with the
        // housing when the fixed toaster is turned. HeaterGlow: (0, 0, 0.16), turned over about X.
        prims.push_back(ts.glowShader.substr(0, ts.glowShader.rfind("/Glow/")));
        matrices.push_back(base->pose.matrix());
        prims.push_back(ts.glowLight);
        matrices.push_back((base->pose * Pose{ { 0, 0, 0.16 }, Quat::axisAngle({ 1, 0, 0 }, 3.14159265358979) }).matrix());
        if (m_monitor.handle)
        {
            // Beside the toaster, standing upright, turned toward the head (about the vertical only).
            const Vec3 p = (base->pose * Pose{ m_monitor.offset }).p;
            Vec3 z = (m_head ? m_head->p : p + base->pose.axis({ 1, 0, 0 })) - p;
            z.z = 0;
            z = z.length() > 1e-3 ? z.normalized() : Vec3{ 1, 0, 0 };
            const Vec3 up{ 0, 0, 1 }, x = up.cross(z);
            const double m[16] = { x.x, x.y, x.z, 0, up.x, up.y, up.z, 0, z.x, z.y, z.z, 0, p.x, p.y, p.z, 1 };
            prims.push_back(kMonitor);
            matrices.push_back(m_monitor.on ? Pose::fromMatrix(m).matrix() : kHidden);
        }
    }

    void toasterWrites(ovstage_ordinal_t ordinal)
    {
        if (!m_toaster)
            return;
        Toaster& ts = *m_toaster;
        if (ts.colorDirty)
        {
            m_stage->writeFloats(ts.toastMaterial, "inputs:diffuseColor",
                                 { ts.toastColor[0], ts.toastColor[1], ts.toastColor[2] }, ordinal);
            if (m_monitor.handle && m_monitor.on)
            {
                const auto& c = ts.toastColor;
                m_stage->writeFloats(m_monitor.swatch, "inputs:diffuseColor", { c[0], c[1], c[2] }, ordinal);
                m_stage->writeFloats(m_monitor.swatch, "inputs:emissiveColor", { 0.5f * c[0], 0.5f * c[1], 0.5f * c[2] }, ordinal);
            }
            ts.colorDirty = false;
        }
        if (m_monitor.on && m_monitor.handle)  // handle 0: removed with the toaster
            for (MonitorPanel::Layer& l : m_monitor.layers)
                if (l.pending && l.points.size() == l.budget * 12)
                {
                    m_stage->writePoints(l.prim, l.points, ordinal);
                    l.pending = false;
                }
        if (ts.glowDirty)
        {
            const float k = static_cast<float>(kGlowEmission * std::clamp(ts.glow / ts.glowMax, 0.0, 1.0));
            m_stage->writeFloats(ts.glowShader, "inputs:emissiveColor", { 1.0f * k, 0.38f * k, 0.08f * k }, ordinal);
            m_stage->writeFloats(ts.glowLight, "inputs:intensity", { ts.glow }, ordinal);
            ts.glowDirty = false;
        }
    }

    // --- per-frame stage writes ------------------------------------------------------------------
    void writeBodies(ovstage_ordinal_t ordinal)
    {
        const auto& bodies = m_physics->bodies();
        if (bodies.empty())
            return;
        std::vector<Mat16> matrices;
        matrices.reserve(bodies.size());
        for (const BodyState& b : bodies)
        {
            const auto it = m_bodyProp.find(b.path);
            const double s = it != m_bodyProp.end() ? m_props[it->second.prop].scale : 1.0;
            matrices.push_back(b.pose.matrix({ s, s, s }));
        }
        m_stage->writeTransforms(m_physics->bodyList(), matrices, ordinal);
    }

    void writeVisuals(const InputFrame& in, ovstage_ordinal_t ordinal)
    {
        std::vector<std::string> prims;
        std::vector<Mat16> matrices;
        prims.push_back(kRig);
        matrices.push_back(m_rig.matrix());
        for (int h = 0; h < 2; ++h)
        {
            const HandInput& hi = in.hands[h];
            prims.push_back(kHands[h]);
            matrices.push_back(hi.active ? hi.grip.matrix() : kHidden);
            prims.push_back(kRays[h]);
            matrices.push_back(hi.aimValid && m_hands[h].rayLength > 0 ? hi.aim.matrix({ 1, 1, m_hands[h].rayLength })
                                                                     : kHidden);
        }
        prims.push_back(kMarker);
        matrices.push_back(m_pendingSpawn && m_placeValid
                               ? Pose{ m_placeAt, Quat::axisAngle({ 0, 0, 1 }, m_placeYaw) }.matrix()
                               : kHidden);
        hudTransforms(prims, matrices);
        toasterTransforms(prims, matrices);
        if (m_browser.open)
        {
            prims.push_back(kBrowser);
            matrices.push_back(m_browser.pose.matrix());
            prims.push_back(kHighlight);
            const double p2 = 2 * m_browser.pad;
            if (m_browser.hoverWidget)
            {
                const auto& r = m_browser.hoverWidget == 1 ? m_browser.closeRect : m_browser.handleRect;
                const Pose at = m_browser.pose * Pose{ { (r[0] + r[2]) / 2, (r[1] + r[3]) / 2, 0 } };
                matrices.push_back(at.matrix({ r[2] - r[0] + p2, r[3] - r[1] + p2, 1 }));
            }
            else if (m_browser.hover >= 0)
            {
                const auto [cx, cy] = tileCenter(m_browser.hover);
                matrices.push_back((m_browser.pose * Pose{ { cx, cy, 0 } }).matrix({ m_browser.tileW + p2, m_browser.tileH + p2, 1 }));
            }
            else
                matrices.push_back(kHidden);
        }
        const bool spinning = m_browser.loading && m_browser.placed;
        prims.push_back(kLoading);
        matrices.push_back(spinning ? m_browser.pose.matrix() : kHidden);
        prims.push_back(kSpinner);
        const double angle = -6.0 * seconds(m_start, Clock::now());
        matrices.push_back(spinning ? (m_browser.pose * Pose{ {}, Quat::axisAngle({ 0, 0, 1 }, angle) }).matrix() : kHidden);
        m_stage->writeTransforms(prims, matrices, ordinal);
        toasterWrites(ordinal);
        if (m_hud.pending && m_hud.points.size() == kHudFloats)
        {
            m_stage->writePoints(kText, m_hud.points, ordinal);
            m_hud.pending = false;
        }
    }

    // Panel floats ~0.95 m ahead, slightly low, lazily following the head (port of vrsim/hud.py).
    void hudTransforms(std::vector<std::string>& prims, std::vector<Mat16>& matrices)
    {
        const auto now = Clock::now();
        const bool visible = !m_hud.points.empty() && m_head && seconds(m_hud.posted, now) < m_hud.hold;
        const int accent = m_hud.kind == "ok" ? 1 : m_hud.kind == "error" ? 2 : 0;
        Pose panel;
        if (visible)
        {
            Vec3 fwd = m_head->axis({ 0, 0, -1 });
            fwd.z = 0;
            fwd = fwd.length() > 1e-3 ? fwd.normalized() : Vec3{ 0, 1, 0 };
            // Off to the left (35 deg) and a little low, clear of the catalog and the table
            // (vrsim.hud FOLLOW_YAW / FOLLOW_DISTANCE / FOLLOW_DROP), unless that cuts through the
            // toaster or its monitor: then the first clear spot of kHudSpots.
            const auto obstacles = hudObstacles();
            // World-anchored while visible, so turning to read it (or aim at its X) doesn't push it
            // away; it is re-placed only when it drifts more than ~70 degrees out of view, or when
            // something comes to stand in it (the monitor switched on).
            if (m_hud.pos)
            {
                Vec3 toPanel = *m_hud.pos - m_head->p;
                toPanel.z = 0;
                if (toPanel.length() > 1e-3 && toPanel.normalized().dot(fwd) < std::cos(70.0 * 3.14159265358979 / 180.0))
                    m_hud.anchor = hudTarget(fwd, obstacles);
                else if (hudOverlap(m_hud.anchor, obstacles) > kHudReplaceOverlap)
                    m_hud.anchor = hudTarget(fwd, obstacles);
            }
            else
                m_hud.anchor = hudTarget(fwd, obstacles);
            const double dt = seconds(m_hudLast, now);
            m_hud.pos = m_hud.pos ? *m_hud.pos + (m_hud.anchor - *m_hud.pos) * std::min(1.0, dt * 3.0) : m_hud.anchor;
            Vec3 z = m_head->p - *m_hud.pos;
            z.z = 0;
            z = z.length() > 1e-3 ? z.normalized() : fwd * -1.0;
            const Vec3 up{ 0, 0, 1 }, x = up.cross(z);
            const double m[16] = { x.x, x.y, x.z, 0, up.x, up.y, up.z, 0, z.x, z.y, z.z, 0, m_hud.pos->x, m_hud.pos->y, m_hud.pos->z, 1 };
            panel = Pose::fromMatrix(m);
            m_hud.pose = panel;
        }
        else
            m_hud.pos.reset();
        m_hud.visible = visible;
        m_hudLast = now;
        prims.push_back(kPanel);
        matrices.push_back(visible ? panel.matrix() : kHidden);
        const Pose accentOffset{ { -0.36 + 0.018, 0, 0.003 } };  // vrsim.hud.ACCENT_OFFSET
        for (int i = 0; i < 3; ++i)
        {
            prims.push_back(kAccents[i]);
            matrices.push_back(visible && i == accent ? (panel * accentOffset).matrix() : kHidden);
        }
    }

    // --- info panel close button --------------------------------------------------------------
    // Pointing at the panel stops the ray there; trigger on the X (vrsim.hud CLOSE_CENTER/HALF)
    // dismisses the current message until the next one arrives.
    // Panel-local hit of a ray on the info panel, or nothing.
    std::optional<std::pair<Vec3, double>> hudHit(const Vec3& origin, const Vec3& dir) const
    {
        if (!m_hud.visible)
            return std::nullopt;
        const Vec3 n = m_hud.pose.axis({ 0, 0, 1 });
        const double denom = dir.dot(n);
        if (denom > -1e-4)
            return std::nullopt;
        const double t = (m_hud.pose.p - origin).dot(n) / denom;
        if (t <= 0 || t > kRayMax)
            return std::nullopt;
        const Vec3 local = (m_hud.pose.inverse() * Pose{ origin + dir * t }).p;
        if (std::abs(local.x) > 0.36 || std::abs(local.y) > 0.125)  // vrsim.hud PANEL_W/H
            return std::nullopt;
        return std::make_pair(local, t);
    }

    // True when the info panel is the nearest panel along this hand's ray (it occludes the browser).
    bool hudInFront(const HandInput& hi) const
    {
        const Vec3 dir = hi.aim.axis({ 0, 0, -1 });
        const auto hud = hudHit(hi.aim.p, dir);
        if (!hud)
            return false;
        const auto browser = m_browser.open ? browserHit(hi.aim.p, dir) : std::nullopt;
        return !browser || hud->second < browser->second;
    }

    void hudInteract(const InputFrame& in)
    {
        if (!m_hud.visible)
            return;
        for (int h = 0; h < 2; ++h)
        {
            const HandInput& hi = in.hands[h];
            const bool down = hi.trigger > (m_hud.triggerDown[h] ? 0.5f : 0.7f);
            const bool pressed = down && !m_hud.triggerDown[h];
            m_hud.triggerDown[h] = down;
            if (!hi.aimValid || m_hands[h].prop >= 0 || !hudInFront(hi))
                continue;
            const auto [local, t] = *hudHit(hi.aim.p, hi.aim.axis({ 0, 0, -1 }));
            m_hands[h].rayLength = t;
            const double hitHalf = 0.022 + 0.012;  // CLOSE_HALF plus a little slack
            if (pressed && std::abs(local.x - (0.36 - 0.035)) <= hitHalf && std::abs(local.y - (0.125 - 0.035)) <= hitHalf)
            {
                m_hud.hold = 0;  // hidden from the next frame on
                m_xr->haptic(h, 0.4f, 0.03f);
                m_link.send({ { "t", "hud_closed" } });  // the sidecar cancels a running download
            }
            else if (pressed && m_hud.nav)
                for (int i = 0; i < 2; ++i)
                    if (std::abs(local.x - kHudNav[i][0]) <= kHudNavHalf + 0.012 && std::abs(local.y - kHudNav[i][1]) <= kHudNavHalf + 0.012)
                    {
                        m_xr->haptic(h, 0.3f, 0.03f);
                        m_link.send({ { "t", "hud_page" }, { "delta", i ? 1 : -1 } });
                    }
        }
    }

    // --- asset browser -----------------------------------------------------------------------
    std::pair<double, double> tileCenter(int index) const
    {
        const int r = index / m_browser.cols, c = index % m_browser.cols;
        return { m_browser.ox + c * (m_browser.tileW + m_browser.gap) + m_browser.tileW / 2,
                 m_browser.oy - r * (m_browser.tileH + m_browser.gap) - m_browser.tileH / 2 };
    }

    // Panel-local hit of a ray on the browser plane (+Z faces the viewer), or nothing.
    std::optional<std::pair<Vec3, double>> browserHit(const Vec3& origin, const Vec3& dir) const
    {
        const Vec3 n = m_browser.pose.axis({ 0, 0, 1 });
        const double denom = dir.dot(n);
        if (denom > -1e-4)
            return std::nullopt;
        const double t = (m_browser.pose.p - origin).dot(n) / denom;
        if (t <= 0 || t > kRayMax)
            return std::nullopt;
        const Vec3 local = (m_browser.pose.inverse() * Pose{ origin + dir * t }).p;
        const double halfW = -m_browser.ox + 0.03;
        const double gridH = m_browser.rows * m_browser.tileH + (m_browser.rows - 1) * m_browser.gap;
        const double bottom = std::min(m_browser.oy - gridH - 0.03, m_browser.handleRect[1] - 0.02);
        if (std::abs(local.x) > halfW || local.y < bottom || local.y > m_browser.oy + 0.12)
            return std::nullopt;
        return std::make_pair(local, t);
    }

    static bool inRect(const Vec3& local, const std::array<double, 4>& r)
    {
        return local.x >= r[0] && local.x <= r[2] && local.y >= r[1] && local.y <= r[3];
    }

    // Upright pose at `pos` facing the user (panel +Z toward the head, horizontally).
    Pose facingUser(const Vec3& pos) const
    {
        Vec3 z = m_head ? m_head->p - pos : Vec3{ 0, -1, 0 };
        z.z = 0;
        z = z.length() > 1e-3 ? z.normalized() : Vec3{ 0, -1, 0 };
        const Vec3 up{ 0, 0, 1 }, x = up.cross(z);
        const double m[16] = { x.x, x.y, x.z, 0, up.x, up.y, up.z, 0, z.x, z.y, z.z, 0, pos.x, pos.y, pos.z, 1 };
        return Pose::fromMatrix(m);
    }

    // First appearance: standing on the table at the sidecar's "anchor" (x, y, bottom z: its far
    // edge, just above the top), else 1.2 m in front of the user; it then stays put.
    void placeBrowser(const nlohmann::json& msg)
    {
        if (m_browser.placed || !m_head)
            return;
        const auto anchor = msg.value("anchor", std::vector<double>{});
        if (anchor.size() == 3)
        {
            // Panel-local bottom: just below the move handle (vrsim.browser HANDLE_RECT).
            const double handleBottom = m_browser.handleRect[1] != 0 ? m_browser.handleRect[1] : -0.597;
            m_browser.pose = facingUser({ anchor[0], anchor[1], anchor[2] - (handleBottom - 0.02) });
        }
        else
        {
            Vec3 fwd = m_head->axis({ 0, 0, -1 });
            fwd.z = 0;
            fwd = fwd.length() > 1e-3 ? fwd.normalized() : Vec3{ 0, 1, 0 };
            m_browser.pose = facingUser(m_head->p + fwd * 1.2 - Vec3{ 0, 0, 0.1 });
        }
        m_browser.placed = true;
    }

    // Grip on the move handle starts dragging the panel; runs before prop grabbing.
    void browserGrab(const InputFrame& in)
    {
        if (!m_browser.open)
            return;
        for (int h = 0; h < 2; ++h)
        {
            const HandInput& hi = in.hands[h];
            if (m_browser.movingHand == h)
            {
                if (hi.squeeze < 0.35f || !hi.aimValid)
                    m_browser.movingHand = -1;
                continue;
            }
            if (m_browser.movingHand >= 0 || m_hands[h].squeezing || hi.squeeze < 0.55f || !hi.aimValid)
                continue;
            const auto hit = browserHit(hi.aim.p, hi.aim.axis({ 0, 0, -1 }));
            if (hit && inRect(hit->first, m_browser.handleRect))
            {
                m_browser.movingHand = h;
                m_browser.moveOffset = hi.aim.inverse() * m_browser.pose;
                m_hands[h].squeezing = true;  // interact() then sees a held squeeze, not a new grab
                m_xr->haptic(h, 0.5f, 0.04f);
            }
        }
    }

    int tileAt(const Vec3& local) const
    {
        const double cx = (local.x - m_browser.ox) / (m_browser.tileW + m_browser.gap);
        const double cy = (m_browser.oy - local.y) / (m_browser.tileH + m_browser.gap);
        const int c = static_cast<int>(std::floor(cx)), r = static_cast<int>(std::floor(cy));
        if (c < 0 || r < 0 || c >= m_browser.cols || r >= m_browser.rows)
            return -1;
        if ((cx - c) * (m_browser.tileW + m_browser.gap) > m_browser.tileW ||
            (cy - r) * (m_browser.tileH + m_browser.gap) > m_browser.tileH)
            return -1;  // in the gap between tiles
        const int index = r * m_browser.cols + c;
        return index < m_browser.count ? index : -1;
    }

    bool placingAny() const { return m_pendingSpawn.has_value(); }

    void browserInteract(const InputFrame& in)
    {
        for (int h = 0; h < 2; ++h)
            if (in.hands[h].thumbClickPressed)
                m_link.send({ { "t", "browser_toggle" } });
        if (!m_browser.open)
            return;
        if (m_browser.movingHand >= 0 && in.hands[m_browser.movingHand].aimValid)
        {
            const Pose target = in.hands[m_browser.movingHand].aim * m_browser.moveOffset;
            m_browser.pose = facingUser(target.p);
        }
        m_browser.hover = -1;
        m_browser.hoverWidget = 0;
        for (int h = 1; h >= 0; --h)  // right hand wins when both point at the panel
        {
            const HandInput& hi = in.hands[h];
            const bool down = hi.trigger > (m_browser.triggerDown[h] ? 0.5f : 0.7f);
            const bool pressed = down && !m_browser.triggerDown[h];
            m_browser.triggerDown[h] = down;
            if (!hi.aimValid || m_hands[h].prop >= 0 || placingAny() || hudInFront(hi))  // the info panel occludes it
                continue;
            const auto hit = browserHit(hi.aim.p, hi.aim.axis({ 0, 0, -1 }));
            if (!hit)
                continue;
            m_hands[h].rayLength = hit->second;  // the pointer ray stops at the panel
            if (inRect(hit->first, m_browser.closeRect))
            {
                m_browser.hoverWidget = 1;
                if (pressed)
                {
                    m_xr->haptic(h, 0.4f, 0.03f);
                    closeBrowser();
                    m_link.send({ { "t", "browser_closed" } });
                    return;
                }
                continue;
            }
            if (inRect(hit->first, m_browser.handleRect) || m_browser.movingHand == h)
            {
                if (!m_browser.hoverWidget)
                    m_browser.hoverWidget = 2;
                continue;
            }
            const int tile = tileAt(hit->first);
            if (tile < 0)
                continue;
            if (m_browser.hover < 0)
                m_browser.hover = tile;
            if (pressed)
            {
                m_link.send({ { "t", "browser_pick" }, { "index", tile } });
                m_xr->haptic(h, 0.5f, 0.04f);
            }
        }
        if (m_browser.hover != m_browser.reportedHover)
        {
            m_browser.reportedHover = m_browser.hover;
            m_link.send({ { "t", "browser_hover" }, { "index", m_browser.hover } });
        }
        for (int h = 0; h < 2; ++h)  // a thumbstick flick pages
        {
            const float x = in.hands[h].thumbX;
            if (m_hands[h].prop >= 0 || placingAny())  // the stick turns a held fixed prop or a placement
                continue;
            if (m_browser.stickArmed[h] && std::abs(x) > 0.7f)
            {
                m_link.send({ { "t", "browser_page" }, { "delta", x > 0 ? 1 : -1 } });
                m_browser.stickArmed[h] = false;
                m_xr->haptic(h, 0.2f, 0.02f);
            }
            else if (std::abs(x) < 0.3f)
                m_browser.stickArmed[h] = true;
        }
    }

    void showBrowser(const nlohmann::json& msg)
    {
        std::ifstream file(msg.value("layer", ""), std::ios::binary);
        const std::string layer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (layer.empty())
            return;
        if (m_browser.handle)
        {
            m_stage->removeReference(m_browser.handle, m_stage->current());
            m_stage->commit();
        }
        m_browser.cols = msg.value("cols", 4);
        m_browser.rows = msg.value("rows", 3);
        m_browser.count = msg.value("count", 0);
        const auto tile = msg.value("tile", std::vector<double>{ 0.26, 0.3 });
        const auto origin = msg.value("grid_origin", std::vector<double>{ -0.565, 0.4875 });
        m_browser.tileW = tile[0], m_browser.tileH = tile[1];
        m_browser.ox = origin[0], m_browser.oy = origin[1];
        m_browser.gap = msg.value("gap", 0.03);
        m_browser.pad = msg.value("highlight_pad", 0.012);
        const auto closeRect = msg.value("close", std::vector<double>{ 0.495, 0.507, 0.565, 0.577 });
        const auto handleRect = msg.value("handle", std::vector<double>{ -0.22, -0.597, 0.22, -0.537 });
        std::copy_n(closeRect.begin(), 4, m_browser.closeRect.begin());
        std::copy_n(handleRect.begin(), 4, m_browser.handleRect.begin());
        m_browser.loading = false;
        m_browser.handle = m_stage->addReference(layer, kBrowser, m_stage->current());
        m_stage->commit();
        placeBrowser(msg);  // no-op once placed (it stays put while paging, or where it was dragged)
        m_browser.open = true;
        m_browser.hover = m_browser.reportedHover = -1;  // a new page: report the next hover again
        logf("Browser page %d/%d (%d assets)", msg.value("page", 0) + 1, msg.value("pages", 1), m_browser.count);
    }

    void closeBrowser()
    {
        if (m_browser.handle)
        {
            m_stage->removeReference(m_browser.handle, m_stage->current());
            m_stage->commit();
        }
        m_browser = Browser{};
    }

    // --- sidecar commands --------------------------------------------------------------------
    void handleSidecar()
    {
        nlohmann::json msg;
        while (m_link.poll(msg))
        {
            const std::string t = msg.value("t", "");
            if (t == "spawn")
                spawn(msg);
            else if (t == "undo")
                m_pendingSpawn ? cancelPlacement() : undo();
            else if (t == "clear")
                clear();
            else if (t == "fix")
                fixCommand(msg.value("on", true));
            else if (t == "toaster")
                onToaster(msg);
            else if (t == "toaster_monitor_frame")
                monitorFrame(msg);
            else if (t == "toaster_monitor")
                setMonitor(msg.value("on", !m_monitor.on));
            else if (t == "remove_device")
                removeDevice(msg.value("device", std::string()));
            else if (t == "realign")
                realign();
            else if (t == "browser")
                showBrowser(msg);
            else if (t == "browser_close")
                closeBrowser();
            else if (t == "browser_loading")
            {
                placeBrowser(msg);
                m_browser.loading = true;
            }
            else if (t == "hud")
            {
                m_hud.points = decodeBase64Floats(msg.value("points", ""));
                m_hud.kind = msg.value("kind", "info");
                m_hud.hold = msg.value("hold", 6.0);
                m_hud.nav = msg.value("nav", false);
                m_hud.posted = Clock::now();
                m_hud.pending = true;
            }
        }
    }

    // Where a new prop drops, by its largest side (the sidecar's bounding box, metres): small ones
    // onto the desk ahead, off to one side of the laptop; larger ones onto the floor beside the user.
    // Sides alternate right, left, ... per spawn; a batch lines up outward on the desk, or backward
    // on the floor.
    Pose spawnPose(const nlohmann::json& msg, int i, int count, double sideSign) const
    {
        // A scene object made grabbable (sidecar desk_props): exactly where it stood, 2 mm up.
        if (const auto at = msg.value("position", std::vector<double>{}); at.size() == 3)
            return { { at[0], at[1], at[2] + 0.002 + i * 0.2 },
                     Quat::axisAngle({ 0, 0, 1 }, msg.value("yaw", 0.0) * 3.14159265358979 / 180.0) };
        const auto size = msg.value("size", std::vector<double>{ 0.2, 0.2, 0.2 });
        const double extent = std::max({ size[0], size[1], size[2] });
        Vec3 fwd{ 0, 1, 0 }, head{ 0, 0, 1.6 };
        if (m_head)
        {
            fwd = m_head->axis({ 0, 0, -1 });
            fwd.z = 0;
            fwd = fwd.length() > 1e-3 ? fwd.normalized() : Vec3{ 0, 1, 0 };
            head = m_head->p;
        }
        const Vec3 right = fwd.cross({ 0, 0, 1 });
        const double spacing = std::max(0.25, std::max(size[0], size[1]) * 1.2);
        Vec3 p;
        if (extent <= kDeskPropMax)
        {
            p = head + fwd * (0.6 + 0.5 * extent) + right * (sideSign * (0.45 + i * spacing));
            p.z = std::max(head.z - 0.15, 0.4);
        }
        else  // waist height, so it drops to the floor (the deck) without bouncing far
        {
            p = head + right * (sideSign * (0.55 + 0.5 * extent)) + fwd * (0.15 - i * spacing);
            p.z = std::max(head.z - 0.7, 0.5);
        }
        p.z += i * 0.05 - msg.value("bottom_z", 0.0);
        return { p, Quat::axisAngle({ 0, 0, 1 }, std::atan2(fwd.y, fwd.x)) };
    }

    void spawn(const nlohmann::json& msg)
    {
        const int count = std::clamp(msg.value("count", 1), 1, 10);
        const auto correctionVec = msg.value("correction", std::vector<double>{});
        if (correctionVec.size() != 16)
            return;
        Mat16 correction;
        std::copy(correctionVec.begin(), correctionVec.end(), correction.begin());
        const double scale = Vec3{ correction[0], correction[1], correction[2] }.length();
        const auto size = msg.value("size", std::vector<double>{ 0.2, 0.2, 0.2 });
        // The sidecar's wrapper Asset prim: the package plus instanceable = false overs, since
        // ovphysx skips colliders inside instances (vrsim.assets._asset_overs).
        const std::string assetRef = msg.value("asset_ref", "@" + msg.value("root_usd", "") + "@");
        const double sideSign = m_nextSpawnRight ? 1.0 : -1.0;
        // Catalog and voice spawns wait for a spot picked with the marker (place()); scene props
        // (the mug) stand where they were.
        const bool catalogSpawn = msg.value("position", std::vector<double>{}).empty();
        const bool atMarker = msg.value("at_marker", false);
        if (catalogSpawn && !atMarker && !m_hulls.empty())
        {
            m_pendingSpawn = msg;
            const Vec3 fwd = m_head ? m_head->axis({ 0, 0, -1 }) : Vec3{ 0, 1, 0 };
            m_placeYaw = std::atan2(fwd.y, fwd.x);
            m_placeValid = false;
            m_link.send({ { "t", "placing" }, { "name", msg.value("name", "prop") }, { "count", count } });
            return;
        }
        if (catalogSpawn && !atMarker)
            m_nextSpawnRight = !m_nextSpawnRight;
        const double spacing = std::max(0.25, std::max(size[0], size[1]) * 1.2);
        for (int i = 0; i < count; ++i)
        {
            if (m_props.size() >= kMaxProps)
                removeProp(oldestRemovable());
            Pose pose = spawnPose(msg, i, count, sideSign);
            if (atMarker)  // standing on the marked spot, 2 mm up, facing the marker's arrow
                pose = { m_placeAt + m_placeSide * ((i - (count - 1) / 2.0) * spacing) +
                             Vec3{ 0, 0, 0.002 - msg.value("bottom_z", 0.0) },
                         Quat::axisAngle({ 0, 0, 1 }, m_placeYaw) };
            const Mat16 m = mul(correction, pose.matrix());
            char rows[512];
            std::snprintf(rows, sizeof(rows), "((%.9g, %.9g, %.9g, %.9g), (%.9g, %.9g, %.9g, %.9g), (%.9g, %.9g, %.9g, %.9g), (%.9g, %.9g, %.9g, %.9g))",
                          m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
            const std::string layer = std::string("#usda 1.0\n(\n    defaultPrim = \"Prop\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n") +
                                      "def Xform \"Prop\"\n{\n    matrix4d xformOp:transform = " + rows +
                                      "\n    uniform token[] xformOpOrder = [\"xformOp:transform\"]\n    def \"Asset\" (prepend references = " +
                                      assetRef + ") {}\n}\n";
            char root[64];
            std::snprintf(root, sizeof(root), "/World/Props/prop_%04d", m_nextProp++);
            Prop prop;
            prop.root = root;
            prop.name = msg.value("name", "prop");
            prop.scale = scale;
            prop.radius = 0.5 * Vec3{ size[0], size[1], size[2] }.length();
            const ovstage_ordinal_t ordinal = m_stage->current();
            prop.handle = m_stage->addReference(layer, prop.root, ordinal);
            m_stage->commit();
            m_physics->sync(ordinal);
            for (const auto& rel : msg.value("body_paths", std::vector<std::string>{}))
            {
                prop.bodyPaths.push_back(prop.root + "/" + rel);
                prop.bodies.push_back(m_stage->internPath(prop.bodyPaths.back()));
            }
            for (const auto& sphere : msg.value("body_spheres", std::vector<std::vector<double>>{}))
                if (sphere.size() == 4)
                    prop.spheres.push_back({ sphere[0], sphere[1], sphere[2], sphere[3] });
            if (prop.spheres.size() != prop.bodies.size())
                prop.spheres.clear();
            prop.movable = msg.value("body_movable", std::vector<bool>{});
            prop.fixOnSpawn = atMarker || msg.value("fixed", false);
            prop.device = msg.value("device", std::string());
            m_props.push_back(std::move(prop));
            if (m_props.back().device == "toaster")
                setupToaster(static_cast<int>(m_props.size()) - 1, msg);
        }
        rebuildBodyMap();
        logf("Spawned %d x %s", count, msg.value("name", "prop").c_str());
        m_link.send({ { "t", "spawned" }, { "name", msg.value("name", "prop") }, { "count", count }, { "total", m_props.size() } });
        if (m_xr)
            m_xr->haptic(1, 0.4f, 0.06f);
    }

    void removeProp(size_t index)
    {
        for (Hand& h : m_hands)
            h.prop = h.body = -1;
        m_lastProp = -1;
        const Prop prop = m_props[index];
        const ovstage_ordinal_t ordinal = m_stage->current();
        m_stage->removeReference(prop.handle, ordinal);
        m_stage->commit();
        m_physics->forgetProp(prop.root);
        m_physics->sync(ordinal);
        m_props.erase(m_props.begin() + static_cast<long long>(index));
        if (m_toaster && m_toaster->prop == static_cast<int>(index))
            m_toaster.reset();
        else if (m_toaster && m_toaster->prop > static_cast<int>(index))
            --m_toaster->prop;
        rebuildBodyMap();
    }

    // Remove a device prop (the toaster: "hide the toaster"), which undo / clear keep, and its monitor.
    void removeDevice(const std::string& device)
    {
        if (device == "toaster" && m_monitor.handle)
        {
            // Placed only with the toaster, it would freeze in the air; setupMonitor adds it again.
            m_stage->removeReference(m_monitor.handle, m_stage->current());
            m_stage->commit();
            m_monitor.handle = 0;
        }
        int n = 0;
        for (int i = static_cast<int>(m_props.size()) - 1; i >= 0; --i)
            if (m_props[i].device == device)
                removeProp(static_cast<size_t>(i)), ++n;
        logf("Removed device %s (%d props)", device.c_str(), n);
    }

    // Undo, clear and the prop limit leave device props (the toaster) alone.
    size_t oldestRemovable() const
    {
        for (size_t i = 0; i < m_props.size(); ++i)
            if (m_props[i].device.empty())
                return i;
        return 0;
    }

    int newestRemovable() const
    {
        for (int i = static_cast<int>(m_props.size()) - 1; i >= 0; --i)
            if (m_props[i].device.empty())
                return i;
        return -1;
    }

    void undo()
    {
        const int last = newestRemovable();
        if (last < 0)
        {
            m_link.send({ { "t", "nothing_to_undo" } });
            return;
        }
        const std::string name = m_props[last].name;
        removeProp(static_cast<size_t>(last));
        m_link.send({ { "t", "removed" }, { "name", name }, { "total", m_props.size() } });
    }

    void clear()
    {
        size_t n = 0;
        for (int i = newestRemovable(); i >= 0; i = newestRemovable(), ++n)
            removeProp(static_cast<size_t>(i));
        m_link.send({ { "t", "cleared" }, { "n", n } });
    }

    void rebuildBodyMap()
    {
        m_bodyProp.clear();
        for (size_t i = 0; i < m_props.size(); ++i)
            for (size_t b = 0; b < m_props[i].bodies.size(); ++b)
                m_bodyProp[m_props[i].bodies[b]] = { static_cast<int>(i), static_cast<int>(b) };
    }

    SidecarLink m_link;
    std::unique_ptr<Stage> m_stage;
    std::unique_ptr<Physics> m_physics;
    ovxr_context_t* m_ctx = nullptr;
    std::unique_ptr<XrInput> m_xr;
    std::optional<Pose> m_head;
    std::vector<Prop> m_props;
    std::unordered_map<ovx_primpath_t, BodyRef> m_bodyProp;
    Hand m_hands[2];
    Hud m_hud;
    Browser m_browser;
    MonitorPanel m_monitor;
    Pose m_rig;                     // tracking -> world (seated eye height at start, turned and shifted by "realign")
    bool m_eyesLevelled = false;    // m_rig's height set from the first head pose
    std::optional<Pose> m_rawHead;  // head in tracking space
    struct
    {
        double drive = 0, step = 0, read = 0, write = 0, bind = 0, submit = 0;
        int steps = 0;
    } m_prof;
    Clock::time_point m_hudLast = Clock::now();
    const Clock::time_point m_start = Clock::now();
    int m_nextProp = 0;
    bool m_nextSpawnRight = true;
    int m_lastProp = -1;  // the prop a hand last let go of (target of "fix it")
    std::vector<Hull> m_hulls;
    double m_dt = 1.0 / 72;
    std::optional<Toaster> m_toaster;
    std::optional<nlohmann::json> m_pendingSpawn;  // a spawn waiting for its spot (place())
    int m_placeHand = 1;
    bool m_placeValid = false;  // m_placeAt is a surface the pointer found
    Vec3 m_placeAt, m_placeSide{ 1, 0, 0 };
    double m_placeYaw = 0;
};

} // namespace

int main(int argc, char** argv)
{
    std::string root = VRSIM_ROOT;
    std::string env = root + "/.cache/generated/environment_physics.usda";
    std::string schemas = envOr("PHYSX_USD_SCHEMAS_ROOT", root + "/.venv/Lib/site-packages");
    uint32_t framesInFlight = 2;
    bool foveation = true;
    bool minimalRendering = false;
    bool envGiven = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--env" && i + 1 < argc)
        {
            env = argv[++i];
            envGiven = true;
        }
        else if (a == "--schemas" && i + 1 < argc)
            schemas = argv[++i];
        else if (a == "--frames-in-flight" && i + 1 < argc)
            framesInFlight = static_cast<uint32_t>(std::clamp(std::atoi(argv[++i]), 1, 3));
        else if (a == "--render" && i + 1 < argc)
            minimalRendering = std::string(argv[++i]) == "minimal";
        else if (a == "--no-foveation")
            foveation = false;
        else
        {
            std::fprintf(stderr, "usage: vrsim_cpp [--env environment_physics.usda] [--schemas site-packages-dir]\n"
                                 "                 [--frames-in-flight 1|2|3] [--no-foveation] [--render rt2|minimal]\n");
            return 2;
        }
    }
    if (!envGiven && minimalRendering)  // the sidecar writes a variant lit for Minimal rendering
        env = root + "/.cache/generated/environment_physics_minimal.usda";
    if (!std::filesystem::exists(env))
        die("open environment", env + " not found; start the sidecar first (uv run python -m vrsim.sidecar)");

    std::signal(SIGINT, [](int) { g_running = 0; });
    ensureOpenXrRuntime();
    Stage::initializeProcess();
    {
        App app(env, schemas, framesInFlight, foveation, minimalRendering);
        app.run();
        logf("Shutting down...");
    }
    Stage::shutdownProcess();
    return 0;
}
