// Shared helpers: error reporting, ovstage operation waits, and pose math.
//
// Conventions match the Python app: quaternions are (x, y, z, w); 4x4 matrices exchanged with
// ovxr/ovstage are row-major with row vectors (translation in elements 12..14).
#pragma once

#include <ovstage/ovstage.h>

#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace vrsim
{

inline void logf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
}

[[noreturn]] inline void die(const char* what, const std::string& detail = {})
{
    logf("FATAL: %s failed%s%s", what, detail.empty() ? "" : ": ", detail.c_str());
    std::exit(1);
}

inline std::string str(ovx_string_t s)
{
    return s.ptr && s.length ? std::string(s.ptr, s.length) : std::string();
}

inline ovx_string_t view(const std::string& s)
{
    return { s.c_str(), s.size() };
}

inline ovx_string_or_token_t name(const char* s)
{
    ovx_string_or_token_t n{};
    n.string = { s, std::strlen(s) };
    return n;
}

inline void checkStage(ovstage_api_status_t status, const char* what)
{
    if (status != OVSTAGE_OK)
        die(what, str(ovstage_get_last_error()));
}

// Wait for and release an enqueued ovstage operation; dies with the op's error on failure.
inline void waitStage(ovstage_instance_t* stage, ovstage_enqueue_result_t result, const char* what)
{
    checkStage(result.status, what);
    if (result.op_index == OVSTAGE_INVALID_OP_ID)
        return;
    ovstage_op_wait_result_t wait{};
    checkStage(ovstage_wait_op(stage, result.op_index, OVSTAGE_TIMEOUT_INFINITE, &wait), what);
    if (wait.error_op_id_count != 0)
        die(what, str(ovstage_get_last_op_error(stage, wait.error_op_ids[0])));
    ovstage_release_op(stage, result.op_index);
}

// --- math -------------------------------------------------------------------------------------
struct Vec3
{
    double x = 0, y = 0, z = 0;
    Vec3 operator+(const Vec3& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Vec3 operator-(const Vec3& o) const { return { x - o.x, y - o.y, z - o.z }; }
    Vec3 operator*(double s) const { return { x * s, y * s, z * s }; }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const { return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x }; }
    double length() const { return std::sqrt(dot(*this)); }
    Vec3 normalized() const
    {
        const double l = length();
        return l > 1e-12 ? *this * (1.0 / l) : Vec3{};
    }
};

struct Quat
{
    double x = 0, y = 0, z = 0, w = 1;
    Quat operator*(const Quat& b) const
    {
        return { w * b.x + x * b.w + y * b.z - z * b.y, w * b.y - x * b.z + y * b.w + z * b.x,
                 w * b.z + x * b.y - y * b.x + z * b.w, w * b.w - x * b.x - y * b.y - z * b.z };
    }
    Quat conj() const { return { -x, -y, -z, w }; }
    Quat normalized() const
    {
        const double l = std::sqrt(x * x + y * y + z * z + w * w);
        return l > 0 ? Quat{ x / l, y / l, z / l, w / l } : Quat{};
    }
    Vec3 rotate(const Vec3& v) const
    {
        const Vec3 u{ x, y, z };
        const Vec3 t = u.cross(v) * 2.0;
        return v + t * w + u.cross(t);
    }
    static Quat axisAngle(const Vec3& axis, double angle)
    {
        const Vec3 a = axis.normalized();
        const double s = std::sin(angle * 0.5);
        return { a.x * s, a.y * s, a.z * s, std::cos(angle * 0.5) };
    }
    // Rotation vector (axis * angle) of the shortest rotation this quaternion represents.
    Vec3 rotationVector() const
    {
        Quat q = (w < 0) ? Quat{ -x, -y, -z, -w } : *this;
        const Vec3 v{ q.x, q.y, q.z };
        const double s = v.length();
        return s > 1e-9 ? v * (2.0 * std::atan2(s, q.w) / s) : Vec3{};
    }
};

using Mat16 = std::array<double, 16>;

struct Pose
{
    Vec3 p;
    Quat q;

    Pose operator*(const Pose& o) const { return { p + q.rotate(o.p), (q * o.q).normalized() }; }
    Pose inverse() const
    {
        const Quat qi = q.conj();
        return { qi.rotate(p) * -1.0, qi };
    }
    Vec3 axis(const Vec3& local) const { return q.rotate(local); }

    // Row-major, row-vector 4x4 with per-axis scale (ovxr / omni:fabric layout).
    Mat16 matrix(Vec3 scale = { 1, 1, 1 }) const
    {
        const Vec3 cx = q.rotate({ 1, 0, 0 }) * scale.x;
        const Vec3 cy = q.rotate({ 0, 1, 0 }) * scale.y;
        const Vec3 cz = q.rotate({ 0, 0, 1 }) * scale.z;
        return { cx.x, cx.y, cx.z, 0, cy.x, cy.y, cy.z, 0, cz.x, cz.y, cz.z, 0, p.x, p.y, p.z, 1 };
    }

    static Pose fromMatrix(const double m[16])
    {
        // Rows 0..2 are the (possibly scaled) local axes in world space.
        Vec3 ax{ m[0], m[1], m[2] }, ay{ m[4], m[5], m[6] }, az{ m[8], m[9], m[10] };
        ax = ax.normalized(), ay = ay.normalized(), az = az.normalized();
        // Rotation matrix columns = local axes; convert to quaternion.
        const double r00 = ax.x, r10 = ax.y, r20 = ax.z, r01 = ay.x, r11 = ay.y, r21 = ay.z, r02 = az.x,
                     r12 = az.y, r22 = az.z;
        Quat q;
        const double tr = r00 + r11 + r22;
        if (tr > 0)
        {
            const double s = std::sqrt(tr + 1.0) * 2;
            q = { (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, 0.25 * s };
        }
        else if (r00 > r11 && r00 > r22)
        {
            const double s = std::sqrt(1.0 + r00 - r11 - r22) * 2;
            q = { 0.25 * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s };
        }
        else if (r11 > r22)
        {
            const double s = std::sqrt(1.0 + r11 - r00 - r22) * 2;
            q = { (r01 + r10) / s, 0.25 * s, (r12 + r21) / s, (r02 - r20) / s };
        }
        else
        {
            const double s = std::sqrt(1.0 + r22 - r00 - r11) * 2;
            q = { (r02 + r20) / s, (r12 + r21) / s, 0.25 * s, (r10 - r01) / s };
        }
        return { { m[12], m[13], m[14] }, q.normalized() };
    }
};

// Row-vector product: apply a, then b.
inline Mat16 mul(const Mat16& a, const Mat16& b)
{
    Mat16 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
    return r;
}

} // namespace vrsim
