"""Small pose helpers.

Quaternions are (x, y, z, w), matching Warp/Newton and OpenXR. Rotation matrices use the
column-vector convention (v' = R @ v). ovxr exchanges 16-float row-major *row-vector* matrices
(translation in elements 12..14), i.e. M[:3, :3] = R.T * scale and M[3, :3] = position.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conj(q) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return quat_normalize(q)


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)])


class Pose:
    """Rigid transform (position + unit quaternion)."""

    __slots__ = ("p", "q")

    def __init__(self, p=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0)):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = quat_normalize(q)

    def __mul__(self, other: "Pose") -> "Pose":
        return Pose(self.p + quat_to_mat(self.q) @ other.p, quat_mul(self.q, other.q))

    def inverse(self) -> "Pose":
        qi = quat_conj(self.q)
        return Pose(-(quat_to_mat(qi) @ self.p), qi)

    def axis(self, local_axis) -> np.ndarray:
        return quat_to_mat(self.q) @ np.asarray(local_axis, dtype=np.float64)

    def to_row_major(self, scale=1.0) -> list[float]:
        return row_major(self.p, self.q, scale)

    @staticmethod
    def from_row_major(m16) -> "Pose":
        m = np.asarray(m16, dtype=np.float64).reshape(4, 4)
        rot = m[:3, :3].T
        scale = np.linalg.norm(rot, axis=0)
        rot = rot / np.where(scale > 0, scale, 1.0)
        return Pose(m[3, :3], mat_to_quat(rot))

    def __repr__(self) -> str:
        return f"Pose(p={np.round(self.p, 3).tolist()}, q={np.round(self.q, 3).tolist()})"


def row_major(p, q, scale=1.0) -> list[float]:
    """Row-major row-vector 4x4 (ovxr layout). ``scale`` is a float or 3-vector of local scales."""
    m = np.eye(4)
    m[:3, :3] = (quat_to_mat(q) * np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))).T
    m[3, :3] = p
    return m.reshape(-1).tolist()
