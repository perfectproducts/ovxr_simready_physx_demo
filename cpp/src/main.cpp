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
constexpr size_t kHudFloats = 6000 * 4 * 3;  // vrsim.hud.MAX_QUADS quads x 4 corners x xyz
const char* kHands[2] = { "/World/Hands/Left", "/World/Hands/Right" };
const char* kRays[2] = { "/World/Hands/LeftRay", "/World/Hands/RightRay" };
const char* kPanel = "/World/HUD/Panel";
const char* kText = "/World/HUD/Panel/Text";
const char* kBrowser = "/World/Browser";
const char* kHighlight = "/World/Browser/Highlight";
const char* kLoading = "/World/BrowserLoading";  // vrsim.browser loading overlay, in the environment
const char* kSpinner = "/World/BrowserLoading/Spinner";
const char* kAccents[3] = { "/World/HUD/Panel/AccentBusy", "/World/HUD/Panel/AccentOk", "/World/HUD/Panel/AccentError" };
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
    double radius = 0.1, scale = 1.0;
};

// One rigid body of a prop. Jointed assets have several.
struct BodyRef
{
    int prop = -1, body = -1;
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
    bool pulling = false, squeezing = false;
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
    int hoverWidget = 0;   // 1 = close button, 2 = move handle
    int movingHand = -1;   // hand dragging the panel by its handle
    Pose moveOffset;       // panel pose in that hand's aim frame
    bool loading = false;  // a page is being built: show the spinner
    std::array<bool, 2> triggerDown{}, stickArmed{ true, true };
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
                    m_head = input.head;
                buttons(input);
                browserGrab(input);  // before interact(): grip on the handle must not grab props
                interact(input, seconds(m_start, now));
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
                    m_physics->step(static_cast<float>(kPhysicsDt));
                lap(m_prof.step);
                m_prof.steps += steps;
            }
            m_physics->readBodies();
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
            undo();
        if (h[0].secondaryPressed)
            clear();
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
        std::optional<BodyRef> best;
        double bestGap = kGrabReach;
        for (const BodyState& b : m_physics->bodies())
        {
            const auto it = m_bodyProp.find(b.path);
            if (it == m_bodyProp.end())
                continue;
            const auto [center, radius] = bodySphere(b, it->second);
            const double gap = (center - point).length() - radius;
            if (gap <= bestGap)
                bestGap = gap, best = it->second;
        }
        return best;
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
        const Prop& prop = m_props[ref->prop];
        const bool part = other.prop == ref->prop && other.whole && other.body != ref->body &&
                          ref->body < static_cast<int>(prop.movable.size()) && prop.movable[ref->body];
        if (other.prop == ref->prop && !part)  // anything else on a held prop: take it over
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
        if (hand.prop >= 0 && other.prop == hand.prop)
        {
            // Letting go of a part: the holding hand keeps it at its new angle. Letting go of the
            // whole: the part's hand now holds everything. Either way, no throw.
            other.whole = true;
            captureWhole(other);
        }
        else if (!hand.pulling && hand.prop >= 0)
            throwProp(hand);
        hand.prop = hand.body = -1;
        hand.pulling = false;
        m_xr->haptic(h, 0.3f, 0.02f);
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
        for (int h = 0; h < 2; ++h)
        {
            Hand& hand = m_hands[h];
            if (hand.prop < 0 || !in.hands[h].active)
                continue;
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
        for (int h = 0; h < 2; ++h)
        {
            const HandInput& hi = in.hands[h];
            prims.push_back(kHands[h]);
            matrices.push_back(hi.active ? hi.grip.matrix() : kHidden);
            prims.push_back(kRays[h]);
            matrices.push_back(hi.aimValid && m_hands[h].rayLength > 0 ? hi.aim.matrix({ 1, 1, m_hands[h].rayLength })
                                                                     : kHidden);
        }
        hudTransforms(prims, matrices);
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
            // (vrsim.hud FOLLOW_YAW / FOLLOW_DISTANCE / FOLLOW_DROP).
            const Vec3 side = Quat::axisAngle({ 0, 0, 1 }, 35.0 * 3.14159265358979 / 180.0).rotate(fwd);
            const Vec3 target = m_head->p + side * 0.85 - Vec3{ 0, 0, 0.25 };
            // World-anchored while visible, so turning to read it (or aim at its X) doesn't push it
            // away; it is re-placed only when it drifts more than ~70 degrees out of view.
            if (m_hud.pos)
            {
                Vec3 toPanel = *m_hud.pos - m_head->p;
                toPanel.z = 0;
                if (toPanel.length() > 1e-3 && toPanel.normalized().dot(fwd) < std::cos(70.0 * 3.14159265358979 / 180.0))
                    m_hud.anchor = target;
            }
            else
                m_hud.anchor = target;
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
            if (!hi.aimValid || m_hands[h].prop >= 0 || hudInFront(hi))  // the info panel occludes it
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
        for (int h = 0; h < 2; ++h)  // a thumbstick flick pages
        {
            const float x = in.hands[h].thumbX;
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
        m_browser.hover = -1;
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
                undo();
            else if (t == "clear")
                clear();
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
                m_hud.posted = Clock::now();
                m_hud.pending = true;
            }
        }
    }

    Pose spawnPose(const nlohmann::json& msg, int i, int count) const
    {
        // A scene object made grabbable (sidecar desk_props): exactly where it stood, 2 mm up.
        if (const auto at = msg.value("position", std::vector<double>{}); at.size() == 3)
            return { { at[0], at[1], at[2] + 0.002 + i * 0.2 } };
        const auto size = msg.value("size", std::vector<double>{ 0.2, 0.2, 0.2 });
        const double extent = std::max({ size[0], size[1], size[2] });
        Vec3 fwd{ 0, 1, 0 }, base{ 0, 0.7, 0 };
        double z = 1.3;
        if (m_head)
        {
            fwd = m_head->axis({ 0, 0, -1 });
            fwd.z = 0;
            fwd = fwd.length() > 1e-3 ? fwd.normalized() : Vec3{ 0, 1, 0 };
            base = m_head->p + fwd * (0.6 + 0.5 * extent);
            z = std::max(m_head->p.z - 0.15, 0.4);
        }
        const Vec3 side = fwd.cross({ 0, 0, 1 });
        Vec3 p = base + side * ((i - (count - 1) / 2.0) * std::max(0.25, std::max(size[0], size[1]) * 1.2));
        p.z = z + i * 0.05 - msg.value("bottom_z", 0.0);
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
        for (int i = 0; i < count; ++i)
        {
            if (m_props.size() >= kMaxProps)
                removeProp(0);
            const Mat16 m = mul(correction, spawnPose(msg, i, count).matrix());
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
            m_props.push_back(std::move(prop));
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
        const Prop prop = m_props[index];
        const ovstage_ordinal_t ordinal = m_stage->current();
        m_stage->removeReference(prop.handle, ordinal);
        m_stage->commit();
        m_physics->forgetProp(prop.root);
        m_physics->sync(ordinal);
        m_props.erase(m_props.begin() + static_cast<long long>(index));
        rebuildBodyMap();
    }

    void undo()
    {
        if (m_props.empty())
        {
            m_link.send({ { "t", "nothing_to_undo" } });
            return;
        }
        const std::string name = m_props.back().name;
        removeProp(m_props.size() - 1);
        m_link.send({ { "t", "removed" }, { "name", name }, { "total", m_props.size() } });
    }

    void clear()
    {
        const size_t n = m_props.size();
        while (!m_props.empty())
            removeProp(m_props.size() - 1);
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
    struct
    {
        double drive = 0, step = 0, read = 0, write = 0, bind = 0, submit = 0;
        int steps = 0;
    } m_prof;
    Clock::time_point m_hudLast = Clock::now();
    const Clock::time_point m_start = Clock::now();
    int m_nextProp = 0;
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
