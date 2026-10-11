"""Reject ambiguous class assignments sharing almost the same image region.

This is a prefilter for parsed inventory observations, not an object-identity
classifier. Nested boxes are rejected unless independently valid aligned-depth
measurements place their visible support in clearly separated spatial ranges.
This narrow exception allows one object seen through another object's frame.
No class names, simulator geometry or known object positions are used here.
"""

import math
from numbers import Real


def _depth_bounds(geometry):
    """Fail closed on missing or malformed measured support, without mutation."""
    if not isinstance(geometry, dict) or geometry.get("localization_source") != "aligned_depth":
        return None
    depth = geometry.get("depth_m")
    points = geometry.get("support_points")
    if (not isinstance(depth, Real) or isinstance(depth, bool) or
            not isinstance(points, (list, tuple))):
        return None
    try:
        depth = float(depth)
        if not math.isfinite(depth) or depth <= 0:
            return None
        distinct = set()
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None
            if any(not isinstance(v, Real) or isinstance(v, bool) for v in point):
                return None
            xy = tuple(float(v) for v in point)
            if not all(math.isfinite(v) for v in xy):
                return None
            distinct.add(xy)
    except (OverflowError, TypeError, ValueError):
        return None
    if len(distinct) < 3:
        return None
    return (float(depth), tuple(min(p[axis] for p in distinct) for axis in (0, 1)),
            tuple(max(p[axis] for p in distinct) for axis in (0, 1)))


def depth_separated(first, second):
    """Whether two valid aligned-depth supports clearly occupy different ranges.

    Requiring separation of the entire sampled XY bounds is stronger than a
    distance between representative pixels or sparse support samples. A second
    axial-depth gap prevents treating lateral pieces of one object as proof of
    foreground/background separation. This does not verify either class label.
    """
    a, b = _depth_bounds(first), _depth_bounds(second)
    if a is None or b is None:
        return False
    depth_gap = abs(a[0]-b[0])
    if not math.isfinite(depth_gap) or depth_gap < .5:
        return False
    gaps = [max(0., a[1][axis]-b[2][axis], b[1][axis]-a[2][axis]) for axis in (0, 1)]
    return all(math.isfinite(v) for v in gaps) and math.hypot(*gaps) >= .35


def filter_ambiguous_inventory(observations, min_containment=0.9, geometries=None):
    """Return ``(accepted, rejected)`` without modifying parsed observations.

    Inputs have the parser's ``{"label": str, "box": [x1,y1,x2,y2]}`` form.
    Cross-class boxes are ambiguous when their intersection covers at least
    ``min_containment`` of the smaller box. This catches nested detections even
    when their IoU is small. Both ends of every ambiguous pair are rejected, so
    a connected group cannot keep an arbitrary winner based on detector order.
    Optional ``geometries`` maps observation indices to independently successful
    aligned-depth localization metadata. Only an edge satisfying both measured
    separation checks is exempt; all remaining conflicts retain group rejection.

    Accepted items retain their original objects and order. Rejections contain
    the original index, label and box, a human-readable reason, and the connected
    ambiguous group indices. Same-class boxes and lesser partial occlusions are
    preserved; this helper does not merge or assign identities to them.
    """
    if not math.isfinite(min_containment) or not 0 < min_containment <= 1:
        raise ValueError("min_containment must be finite and in (0, 1]")
    geometries = geometries if isinstance(geometries, dict) else {}
    items = list(observations)
    labels, boxes, areas = [], [], []
    for item in items:
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("inventory observation needs a nonempty label")
        labels.append(" ".join(label.lower().replace("_", " ").split()))
        try:
            box = tuple(float(v) for v in item["box"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("inventory observation needs a finite positive-area box") from exc
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("inventory observation needs a finite positive-area box")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("inventory observation needs a finite positive-area box")
        boxes.append(box)
        areas.append((x2-x1)*(y2-y1))

    conflicts = [[] for _ in items]
    for i, first in enumerate(boxes):
        for j in range(i+1, len(boxes)):
            if labels[i] == labels[j]:
                continue
            second = boxes[j]
            intersection = (max(0., min(first[2], second[2])-max(first[0], second[0])) *
                            max(0., min(first[3], second[3])-max(first[1], second[1])))
            fraction = intersection / min(areas[i], areas[j])
            if fraction + 1e-12 >= min_containment:
                if depth_separated(geometries.get(i), geometries.get(j)):
                    continue
                conflicts[i].append({"index": j, "intersection_over_smaller": fraction})
                conflicts[j].append({"index": i, "intersection_over_smaller": fraction})

    groups = {}
    for start, peers in enumerate(conflicts):
        if not peers or start in groups:
            continue
        pending, found = [start], set()
        while pending:
            index = pending.pop()
            if index in found:
                continue
            found.add(index)
            pending.extend(peer["index"] for peer in conflicts[index])
        group = sorted(found)
        groups.update((index, group) for index in group)

    accepted, rejected = [], []
    for index, item in enumerate(items):
        if index not in groups:
            accepted.append(item)
            continue
        rejected.append({
            "index": index, "label": item["label"], "box": list(item["box"]),
            "reason": (f"ambiguous cross-class boxes cover at least {min_containment:.0%} "
                       "of the smaller box; wait for another view before assigning identities"),
            "ambiguous_indices": groups[index], "overlaps": conflicts[index],
        })
    return accepted, rejected
