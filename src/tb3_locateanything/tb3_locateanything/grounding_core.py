"""Small ROS-independent validation and camera/LiDAR fusion helpers.

geometry.py and output_parser.py are reused from Ling Shen's research checkout
694fff69d87852d592be99f39f0ca77c9b04b610. No simulator state enters this code.
"""

import math
import re
import statistics

from .geometry import (
    Quaternion, Vector3,
    camera_ray_from_pixel, point_from_bearing_range, quaternion_rotate_vector,
    scan_bearing_from_ray, transform_point, validate_camera_calibration,
)
from .output_parser import parse_locateanything_output


def referring_expression(command):
    """Remove only a leading movement verb; preserve every descriptive word."""
    return re.sub(r"^(?:please\s+)?(?:move|go|navigate|drive)\s+to\s+", "",
                  command.strip(), flags=re.IGNORECASE).strip().rstrip(".")


def allows_any_instance(command):
    """Permit choosing one model match; never infer the target class or color.

    This is only a multiplicity policy. The entire referring expression still
    reaches LocateAnything, including every appearance and relational detail.
    """
    text = referring_expression(command).lower().rstrip(".! ")
    prefix = r"(?:(?:(?:a|an|one)\s+)?random|any(?:\s+one)?|a|an)"
    if re.fullmatch(prefix, text):
        return False
    if re.search(r"\b(?:not|no|never|instead|without|except|avoid|cannot|and|or|then)\b"
                 r"|\b\w+['’]t\b|[;\n]", text):
        return False
    return bool(re.match(r"^" + prefix + r"\s+\S", text))


def described_class(command, labels):
    """Return an explicit target-class hint, never a semantic target choice.

    Only a first class mention preceded by a short, adjective-like prefix is
    considered. Relational/generic references and negation remain entirely
    model-grounded. The caller can reject a conflicting known memory class;
    it must still send the full description to LocateAnything.
    """
    text = re.sub(r"[\s_]+", " ", referring_expression(command).lower())
    if re.search(r"\b(?:not|no|never|instead|without|except|avoid|cannot)\b"
                 r"|\b(?:don|doesn|didn|isn|aren|won|wouldn|shouldn|couldn|can)['’]t\b", text):
        return None
    matches = []
    for label in labels:
        normalized = re.sub(r"[\s_]+", " ", str(label).strip().lower())
        if not normalized or normalized in {"object", "objects", "thing", "things", "something", "anything"}:
            continue
        match = re.search(r"(?<!\w)" + re.escape(normalized) + r"(?!\w)", text)
        if match:
            matches.append((match.start(), -len(normalized), normalized))
    if not matches:
        return None
    start, _, label = min(matches)
    prefix = text[:start].strip()
    words = prefix.split()
    blocked = {"near", "beside", "next", "behind", "front",
               "with", "of", "that", "which", "object", "objects", "thing",
               "things", "something", "anything", "one", "ones", "to", "for",
               "from", "by", "at", "in", "on", "is", "are", "i", "you", "we",
               "it", "want", "need", "find", "choose", "select", "ignore"}
    if (len(words) > 4 or any(word in blocked for word in words)
            or (prefix and not re.fullmatch(r"[a-z]+(?:[- ][a-z]+)*", prefix))):
        return None
    return label.replace(" ", "_")


def memory_relation(command, labels):
    """Parse bounded robot-relative commands without discarding modifiers.

    Only a bare known class is supported here. A color, reference object or
    other predicate needs separate grounding; never silently drop that text.
    """
    text = re.sub(r"[\s_]+", " ", referring_expression(command).lower()).strip(" .!?")
    match = re.fullmatch(
        r"(?:the\s+)?(another|other|near|nearer|nearest|closer|closest|"
        r"far|farther|farthest|further|furthest)\s+(.+)", text)
    if not match:
        return None
    selector, target = match.groups()
    known = {str(label).replace("_", " ").strip().lower(): label for label in labels
             if label != "object"}
    if target not in known:
        raise ValueError("measured another/near/far queries currently require a bare known class "
                         "(for example 'another chair' or 'farther person'); "
                         "additional attributes or reference objects are not supported")
    kind = ("another" if selector in {"another", "other"} else
            "farthest" if selector in {"far", "farther", "farthest", "further", "furthest"}
            else "nearest")
    return kind, known[target]


def validate_localizable_box(box, width, height):
    """Reject boxes whose center ray is unreliable in this planar localizer.

    Horizontal clipping shifts the box center away from the object's bearing,
    even when only one side is cut off. A nearly full image box, or a box cut
    off at both vertical edges, is also unreliable. This is a geometry guard,
    not a test that the described object is actually present.
    """
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError("invalid box or image dimensions")
    if not all(math.isfinite(value) for value in box):
        raise ValueError("non-finite box coordinate")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box outside original camera image")
    if (x2 - x1) * (y2 - y1) >= 0.75 * width * height:
        raise ValueError("target box covers nearly the whole image")
    edge_x, edge_y = max(2.0, width * 0.005), max(2.0, height * 0.005)
    if x1 <= edge_x or x2 >= width - edge_x:
        raise ValueError("target box is clipped at a horizontal image edge")
    if y1 <= edge_y and y2 >= height - edge_y:
        raise ValueError("target box is clipped at opposite image edges")
    return box


def one_box(answer, width, height):
    parsed = parse_locateanything_output(answer, width, height)
    if not parsed["parse_success"]:
        raise ValueError("malformed model output: " + ",".join(parsed["errors"]))
    if not parsed["pixel_boxes"]:
        raise ValueError("LocateAnything returned no target")
    if len(parsed["pixel_boxes"]) != 1:
        raise ValueError("ambiguous model output: expected exactly one box")
    return validate_localizable_box(parsed["pixel_boxes"][0], width, height)


def catalog_boxes(answer, width, height):
    """Parse a class inventory, retaining only boxes usable for localization.

    Explicit ``<box>None</box>`` is a valid empty inventory. Malformed output
    fails as a whole; a well-formed but clipped object is skipped so that other
    complete objects in the same view can still enter visual memory.
    """
    parsed = parse_locateanything_output(answer, width, height)
    if not parsed["parse_success"]:
        raise ValueError("malformed model output: " + ",".join(parsed["errors"]))
    boxes = []
    for box in parsed["pixel_boxes"]:
        try:
            validate_localizable_box(box, width, height)
        except ValueError:
            continue
        boxes.append(box)
    return boxes


def check_snapshot_times(now_ns, image_ns, scan_ns, info_ns, odom_ns,
                         max_age=0.75, tolerance=0.15):
    """Reject stale/future images and sensor records from a different instant."""
    age = (now_ns - image_ns) / 1e9
    if image_ns <= 0 or age < -0.05 or age > max_age:
        raise ValueError("stale or future camera image")
    if any(abs(stamp - image_ns) / 1e9 > tolerance
           for stamp in (scan_ns, info_ns, odom_ns)):
        raise ValueError("camera, scan, calibration and odometry are not synchronized")


def validate_surface_visibility(box, laser_pixel, scene_boxes, support_pixels=None):
    """Decline a laser surface also covered by another inventoried instance.

    Clipped boxes still identify possible foreground occluders. Bounding boxes
    cannot tell which of two overlapping objects owns the hit; wait for a less
    occluded view rather than attach the nearer surface to the farther target.
    The median-based center can fall just outside an occluder's RGB boundary
    even when the actual supporting rays hit it. Reject when at least half of
    those rays project inside the other box; a small peripheral overlap alone
    does not invalidate a foreground object's otherwise visible surface.
    """
    u, v = laser_pixel
    for other in scene_boxes:
        overlap = max(0., min(box[2], other[2])-max(box[0], other[0])) * max(
            0., min(box[3], other[3])-max(box[1], other[1]))
        union = (box[2]-box[0])*(box[3]-box[1]) + (other[2]-other[0])*(other[3]-other[1])-overlap
        if overlap / union >= .6:
            continue  # Same detection/re-grounded pixels; class voting handles contradictions.
        if other[0] <= u <= other[2] and other[1] <= v <= other[3]:
            raise ValueError("laser surface overlaps another detected object; need an unoccluded view")
        if support_pixels:
            covered = sum(other[0] <= pu <= other[2] and other[1] <= pv <= other[3]
                          for pu, pv in support_pixels)
            if 2*covered >= len(support_pixels):
                raise ValueError("laser support overlaps another detected object; need an unoccluded view")


def _reject_continuous_background_plane(box, support, seed_indices, distance,
        calibration, camera_to_scan, scan_to_map, ranges,
        angle_min, angle_increment, range_min, range_max, full_circle):
    """Reject a flat laser surface visibly continuing past BOTH box sides.

    This catches an RGB shadow/mark on a wall without using class or scene
    knowledge. Fit a line to >=4 returns spanning 15 cm, tolerate 3 cm of
    scan noise, and require >=3 connected returns extending by at least
    max(20 px, 20% box width) on each side. A real object coplanar with its
    background can also fail this conservative test. Missing one side or a
    nonplanar surface is inconclusive, never proof of a real object.
    """
    if len(support) < 4:
        return
    mx = sum(p[0] for p in support) / len(support)
    my = sum(p[1] for p in support) / len(support)
    xx = sum((p[0]-mx)**2 for p in support)
    yy = sum((p[1]-my)**2 for p in support)
    xy = sum((p[0]-mx)*(p[1]-my) for p in support)
    angle = .5 * math.atan2(2*xy, xx-yy)
    tangent = (math.cos(angle), math.sin(angle))
    normal = (-tangent[1], tangent[0])
    tolerance = .03

    def residual(p):
        return abs((p[0]-mx)*normal[0] + (p[1]-my)*normal[1])

    along = [p[0]*tangent[0] + p[1]*tangent[1] for p in support]
    if max(along)-min(along) < .15 or max(map(residual, support)) > tolerance:
        return
    x1, y1, x2, y2 = box
    rotation, translation = camera_to_scan.rotation, camera_to_scan.translation
    inverse = Quaternion(-rotation.x, -rotation.y, -rotation.z, rotation.w)
    n = len(ranges)

    def sample(index):
        if not 0 <= index < n:
            return None
        value = ranges[index]
        if not math.isfinite(value) or not range_min <= value < range_max:
            return None
        angle = angle_min + index * angle_increment
        point = point_from_bearing_range(angle, value)
        camera = quaternion_rotate_vector(inverse, Vector3(
            point.x-translation.x, point.y-translation.y, point.z-translation.z))
        if camera.z <= 0:
            return None
        u = calibration.fx * camera.x / camera.z + calibration.cx
        v = calibration.fy * camera.y / camera.z + calibration.cy
        if not (0 <= u < calibration.width and y1-3 <= v <= y2+3):
            return None
        fixed = transform_point(scan_to_map, point)
        if residual((fixed.x, fixed.y)) > tolerance:
            return None
        direction = quaternion_rotate_vector(scan_to_map.rotation,
            Vector3(math.cos(angle), math.sin(angle), 0.))
        denominator = normal[0]*direction.x + normal[1]*direction.y
        if abs(denominator) < 1e-8:
            return None
        origin = scan_to_map.translation
        predicted_range = (normal[0]*(mx-origin.x) + normal[1]*(my-origin.y)) / denominator
        if predicted_range <= 0:
            return None
        predicted = transform_point(scan_to_map,
            point_from_bearing_range(angle, predicted_range))
        return index, value, u, (point.x, point.y), (predicted.x, predicted.y)

    seeds = [sample(i) for i in seed_indices]
    seeds = [p for p in seeds if p is not None and x1 <= p[2] <= x2]
    if not seeds:
        return
    seed = min(seeds, key=lambda p: abs(p[1]-distance))
    connected = {seed[0]: seed}
    for direction in (-1, 1):
        previous = seed
        for step in range(1, n):
            index = seed[0] + direction * step
            if full_circle:
                index %= n
            current = sample(index)
            if current is None:
                break
            # Grazing rays on one wall can be >10 cm apart. Compare their
            # measured spacing with the fitted plane's ray intersections,
            # allowing two 3 cm residuals, instead of imposing a fixed jump.
            if math.dist(current[3], previous[3]) > math.dist(current[4], previous[4]) + 2*tolerance:
                break
            connected[index] = current
            previous = current
    left = [p[2] for p in connected.values() if p[2] < x1]
    right = [p[2] for p in connected.values() if p[2] > x2]
    margin = max(20., .2*(x2-x1))
    if (len(left) >= 3 and len(right) >= 3
            and x1-min(left) >= margin and max(right)-x2 >= margin):
        raise ValueError("laser surface continues as a background plane beyond both sides of target box")


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
    if len(valid) < 2:
        raise ValueError("fewer than two usable laser returns near target center; need a clearer view")
    distance = statistics.median(valid)
    if max(valid) - min(valid) > 0.6:
        raise ValueError("mixed foreground/background laser returns")
    scan_point = point_from_bearing_range(bearing, distance)
    translation, rotation = camera_to_scan.translation, camera_to_scan.rotation
    camera_point = quaternion_rotate_vector(
        Quaternion(-rotation.x, -rotation.y, -rotation.z, rotation.w),
        Vector3(scan_point.x-translation.x, scan_point.y-translation.y, scan_point.z-translation.z))
    if camera_point.z <= 0:
        raise ValueError("laser surface is behind the observation camera")
    laser_u = calibration.fx * camera_point.x / camera_point.z + calibration.cx
    laser_v = calibration.fy * camera_point.y / camera_point.z + calibration.cy
    # Seeing the top of an object behind a shelf does not mean that the
    # low laser hit belongs to it. Reproject the measured surface into the
    # original RGB box, including the camera/laser baseline and frozen TF.
    if not (x1-3 <= laser_u <= x2+3 and y1-3 <= laser_v <= y2+3):
        raise ValueError("laser surface projects outside target box (occluded or above laser plane)")
    point = transform_point(scan_to_map, scan_point)
    # Identity needs more than the centre return: a long object presents a
    # different surface point from each direction. Keep the connected laser
    # surface inside the box, stopping at gaps/occlusions. The caller supplies
    # an odom transform so these points do not move with SLAM map corrections.
    support = []
    support_pixels = []
    seed = min((i for i in indices if 0 <= i < n and math.isfinite(ranges[i])
                and range_min <= ranges[i] < range_max),
               key=lambda i: abs(ranges[i] - distance))
    angular_limit = min(widths)
    chosen = {seed}
    for direction in (-1, 1):
        previous = seed
        for step in range(1, int(angular_limit / angle_increment) + 2):
            index = seed + direction * step
            if full_circle:
                index %= n
            if index < 0 or index >= n:
                break
            angle = angle_min + index * angle_increment
            offset = abs(math.atan2(math.sin(angle-bearing), math.cos(angle-bearing)))
            value = ranges[index]
            if (offset > angular_limit or not math.isfinite(value)
                    or not range_min <= value < range_max):
                break
            # Include continuous faces, never jump a depth discontinuity to
            # a wall/background merely because it falls within the RGB box.
            separation = math.sqrt(max(0.0, value*value + ranges[previous]**2
                - 2*value*ranges[previous]*math.cos(angle_increment)))
            if separation > max(0.10, 2.5 * min(value, ranges[previous]) * angle_increment):
                break
            chosen.add(index)
            previous = index
    ordered = sorted(chosen)
    if len(ordered) < 2:
        raise ValueError("fewer than two continuous laser returns for target; need a clearer view")
    if len(ordered) > 32:
        ordered = [ordered[round(i * (len(ordered)-1) / 31)] for i in range(32)]
    for index in ordered:
        scan_sample = point_from_bearing_range(angle_min + index * angle_increment, ranges[index])
        p = transform_point(scan_to_map, scan_sample)
        support.append([p.x, p.y])
        camera_sample = quaternion_rotate_vector(
            Quaternion(-rotation.x, -rotation.y, -rotation.z, rotation.w),
            Vector3(scan_sample.x-translation.x, scan_sample.y-translation.y,
                    scan_sample.z-translation.z))
        if camera_sample.z <= 0:
            raise ValueError("laser support is behind the observation camera")
        support_pixels.append([calibration.fx*camera_sample.x/camera_sample.z + calibration.cx,
                               calibration.fy*camera_sample.y/camera_sample.z + calibration.cy])
    # Rotation-only bearings approximate a small camera/laser baseline. Near
    # occlusion boundaries that approximation can select mostly off-box rays
    # even though the synthetic median center passes the 3 px tolerance.
    # Require a strict majority of actual support to project inside the box,
    # with no pixel padding. This establishes geometric support, not ownership
    # when the model has omitted a foreground occluder's box altogether.
    inside = sum(x1 <= pu <= x2 and y1 <= pv <= y2 for pu, pv in support_pixels)
    if 2*inside <= len(support_pixels):
        raise ValueError("most laser support projects outside target box; need a clearer view")
    _reject_continuous_background_plane(box, support, indices, distance,
        calibration, camera_to_scan, scan_to_map, ranges,
        angle_min, angle_increment, range_min, range_max, full_circle)
    return point, {"bearing_rad": bearing, "range_m": distance,
                   "scan_index": candidate, "valid_scan_rays": len(valid),
                   "laser_pixel": [laser_u, laser_v],
                   "support_points": support, "support_pixels": support_pixels}
