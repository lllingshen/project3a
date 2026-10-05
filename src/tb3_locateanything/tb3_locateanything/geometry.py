"""Necessary camera/LiDAR helpers reused from Ling Shen's research localizer.

Source commit: 694fff69d87852d592be99f39f0ca77c9b04b610.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CameraCalibration:
    """Validated pinhole camera calibration."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str


@dataclass(frozen=True)
class Vector3:
    """Small immutable 3D vector used for pure tests and TF math."""

    x: float
    y: float
    z: float

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class Quaternion:
    """Quaternion with ROS field order."""

    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class Transform3D:
    """Rigid transform from a source frame into a target frame."""

    translation: Vector3
    rotation: Quaternion


def validate_camera_calibration(calibration: CameraCalibration) -> list[str]:
    """Return validation errors for a camera calibration."""

    errors: list[str] = []
    if calibration.width <= 0:
        errors.append("camera_info width must be positive")
    if calibration.height <= 0:
        errors.append("camera_info height must be positive")
    if not math.isfinite(calibration.fx) or calibration.fx <= 0.0:
        errors.append("camera_info fx must be finite and positive")
    if not math.isfinite(calibration.fy) or calibration.fy <= 0.0:
        errors.append("camera_info fy must be finite and positive")
    if not math.isfinite(calibration.cx):
        errors.append("camera_info cx must be finite")
    if not math.isfinite(calibration.cy):
        errors.append("camera_info cy must be finite")
    return errors


def camera_ray_from_pixel(u: float, v: float, calibration: CameraCalibration) -> Vector3:
    """Construct a normalized pinhole ray in camera optical coordinates."""

    return Vector3(
        x=(u - calibration.cx) / calibration.fx,
        y=(v - calibration.cy) / calibration.fy,
        z=1.0,
    )


def quaternion_rotate_vector(rotation: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector by a quaternion without applying translation."""

    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("invalid zero or non-finite quaternion")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # q * v * q_conjugate, expanded to avoid dependencies.
    uvx = y * vector.z - z * vector.y
    uvy = z * vector.x - x * vector.z
    uvz = x * vector.y - y * vector.x

    uuvx = y * uvz - z * uvy
    uuvy = z * uvx - x * uvz
    uuvz = x * uvy - y * uvx

    return Vector3(
        x=vector.x + 2.0 * (w * uvx + uuvx),
        y=vector.y + 2.0 * (w * uvy + uuvy),
        z=vector.z + 2.0 * (w * uvz + uuvz),
    )


def scan_bearing_from_ray(scan_frame_ray: Vector3) -> float:
    """Compute horizontal bearing in the LaserScan frame."""

    return math.atan2(scan_frame_ray.y, scan_frame_ray.x)


def point_from_bearing_range(bearing: float, range_m: float) -> Vector3:
    """Construct a planar scan-frame point from bearing and range."""

    return Vector3(
        x=range_m * math.cos(bearing),
        y=range_m * math.sin(bearing),
        z=0.0,
    )


def transform_point(transform: Transform3D, point: Vector3) -> Vector3:
    """Apply rotation and translation to a point."""

    rotated = quaternion_rotate_vector(transform.rotation, point)
    return Vector3(
        x=rotated.x + transform.translation.x,
        y=rotated.y + transform.translation.y,
        z=rotated.z + transform.translation.z,
    )
