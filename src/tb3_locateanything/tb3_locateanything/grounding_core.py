"""Small ROS-independent validation and camera/LiDAR fusion helpers.

geometry.py and output_parser.py are reused from Ling Shen's research checkout
694fff69d87852d592be99f39f0ca77c9b04b610. No simulator state enters this code.
"""

import math
import re
import statistics

from .geometry import (
    camera_ray_from_pixel, point_from_bearing_range, quaternion_rotate_vector,
    scan_bearing_from_ray, transform_point, validate_camera_calibration,
)
from .output_parser import parse_locateanything_output


def referring_expression(command):
    """Remove only a leading movement verb; preserve every descriptive word."""
    return re.sub(r"^(?:please\s+)?(?:move|go|navigate|drive)\s+to\s+", "",
                  command.strip(), flags=re.IGNORECASE).strip().rstrip(".")


def one_box(answer, width, height):
    parsed = parse_locateanything_output(answer, width, height)
    if not parsed["parse_success"]:
        raise ValueError("malformed model output: " + ",".join(parsed["errors"]))
    if not parsed["pixel_boxes"]:
        raise ValueError("LocateAnything returned no target")
    if len(parsed["pixel_boxes"]) != 1:
        raise ValueError("ambiguous model output: expected exactly one box")
    return parsed["pixel_boxes"][0]


def check_snapshot_times(now_ns, image_ns, scan_ns, info_ns, odom_ns,
                         max_age=0.75, tolerance=0.15):
    """Reject stale/future images and sensor records from a different instant."""
    age = (now_ns - image_ns) / 1e9
    if image_ns <= 0 or age < -0.05 or age > max_age:
        raise ValueError("stale or future camera image")
    if any(abs(stamp - image_ns) / 1e9 > tolerance
           for stamp in (scan_ns, info_ns, odom_ns)):
        raise ValueError("camera, scan, calibration and odometry are not synchronized")


def localize_box(box, calibration, camera_to_scan, scan_to_map,
                 ranges, angle_min, angle_increment, range_min, range_max,
                 half_window=2):
    """Pinhole ray -> scan bearing -> nearby median return -> frozen map TF.

    This planar approximation assumes the selected object intersects the laser
    plane. As in the reused research localizer, the camera-to-laser baseline is
    small and the ray direction uses rotation only. It is not depth estimation.
    """
    errors = validate_camera_calibration(calibration)
    if errors:
        raise ValueError("; ".join(errors))
    if not ranges or angle_increment <= 0:
        raise ValueError("invalid laser scan")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= calibration.width and 0 <= y1 < y2 <= calibration.height):
        raise ValueError("box outside original camera image")
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    if u < 5 or u > calibration.width - 5:
        raise ValueError("target too close to image edge")
    ray = quaternion_rotate_vector(camera_to_scan.rotation,
                                   camera_ray_from_pixel(u, v, calibration))
    if math.hypot(ray.x, ray.y) < 1e-6:
        raise ValueError("camera ray has no planar component; check optical frame")
    bearing = scan_bearing_from_ray(ray)
    n = len(ranges)
    full_circle = n * angle_increment >= 2 * math.pi - 2 * angle_increment
    candidate = round(((bearing - angle_min) % (2 * math.pi)) / angle_increment)
    if full_circle:
        candidate %= n
    elif candidate >= n:
        raise ValueError("target outside scan field of view")
    # Restrict the median window to the actual angular width of the box.
    bearings = [scan_bearing_from_ray(quaternion_rotate_vector(
        camera_to_scan.rotation, camera_ray_from_pixel(edge, v, calibration)))
        for edge in (x1, x2)]
    widths = [abs(math.atan2(math.sin(b - bearing), math.cos(b - bearing)))
              for b in bearings]
    window = min(half_window, int(min(widths) / angle_increment))
    indices = [((candidate + offset) % n) if full_circle else candidate + offset
               for offset in range(-window, window + 1)]
    valid = [ranges[i] for i in indices if 0 <= i < n
             and math.isfinite(ranges[i]) and range_min <= ranges[i] < range_max]
    if not valid:
        raise ValueError("no usable laser return for selected box")
    distance = statistics.median(valid)
    if max(valid) - min(valid) > 0.6:
        raise ValueError("mixed foreground/background laser returns")
    point = transform_point(scan_to_map, point_from_bearing_range(bearing, distance))
    return point, {"bearing_rad": bearing, "range_m": distance,
                   "scan_index": candidate, "valid_scan_rays": len(valid)}
