"""Locate an RGB box using aligned, measured depth and visible surface ownership.

No model labels, simulator entities or known object positions enter this code.
Depth surfaces extending well outside the box are foreground/background, not
evidence that the enclosed target owns their range. Ambiguous views are skipped.
"""
import math

import cv2
import numpy as np

from .geometry import Vector3, transform_point


class AmbiguousDepthError(ValueError):
    """Rejected localization with bounded evidence for another observation.

    Groups are measured, mutually disconnected surfaces, not object identities
    or selectable target positions. An unusable hint retains the rejection and
    the total group count; no subset of an overfull hint is exposed.
    """
    def __init__(self, box, groups, group_count, hint_reason=None):
        super().__init__("multiple separated depth surfaces inside target box; wait for a clearer view")
        self.box = tuple(float(v) for v in box)
        self.groups = tuple(groups)
        self.group_count = group_count
        self.hint_reason = hint_reason
        self.hint_usable = hint_reason is None and len(self.groups) >= 2


def _ambiguity_error(box, components, clouds, us, vs):
    """Describe every 3D connected group without changing the depth verdict."""
    remaining = set(range(len(clouds)))
    groups = []
    while remaining:
        group = {min(remaining)}
        remaining.difference_update(group)
        while remaining:
            adjacent = {j for j in remaining if any(_touching(clouds[i], clouds[j]) for i in group)}
            if not adjacent:
                break
            group.update(adjacent)
            remaining.difference_update(adjacent)
        groups.append(sorted(group))
    if len(groups) > 8:
        return AmbiguousDepthError(box, (), len(groups), "too_many_surface_groups")
    evidence = []
    for group in groups:
        xyz = np.concatenate([clouds[i] for i in group])
        pixels = np.concatenate([np.stack((us[components[i]["rows"], components[i]["cols"]],
                                           vs[components[i]["rows"], components[i]["cols"]]), axis=-1)
                                 for i in group])
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(pixels)):
            return AmbiguousDepthError(box, (), len(groups), "nonfinite_surface_groups")
        # Retain actual XYZ/pixel extrema so bounded sampling does not shrink
        # the measured extent; every pixel still corresponds to its own point.
        combined = np.column_stack((xyz, pixels))
        indices = set(np.argmin(combined, axis=0)) | set(np.argmax(combined, axis=0))
        for index in np.linspace(0, len(xyz)-1, min(64, len(xyz)), dtype=int):
            if len(indices) >= 64:
                break
            indices.add(index)
        selected = sorted(indices)
        evidence.append({"odom_xyz": xyz[selected].tolist(),
                         "pixels": pixels[selected].astype(int).tolist(), "count": len(xyz)})
    return AmbiguousDepthError(box, evidence, len(groups))


def _plane(points):
    """A measured plane, only when the patch has two spatial dimensions."""
    if len(points) < 12:
        return None
    sample = points[np.linspace(0, len(points)-1, min(512, len(points)), dtype=int)]
    center = np.mean(sample, axis=0)
    _, singular, axes = np.linalg.svd(sample-center, full_matrices=False)
    if singular[1] / math.sqrt(len(sample)) < .008:
        return None
    normal = axes[-1]
    if np.quantile(np.abs((sample-center) @ normal), .9) > .025:
        return None
    return center, normal


def _same_plane(a, b):
    if a is None or b is None:
        return False
    ac, an = a
    bc, bn = b
    return (abs(float(an @ bn)) > math.cos(math.radians(5)) and
            abs(float((ac-bc) @ bn)) < .04 and abs(float((ac-bc) @ an)) < .04)


def _outside_planes(points, minimum):
    """Fit a few substantial outside surfaces even across a room corner.

    Sampling is deterministic and only actual measured pixels outside the
    box vote. A thin one-pixel box overshoot cannot establish a plane.
    """
    if len(points) < minimum:
        return []
    sample = points[np.linspace(0, len(points)-1, min(1600, len(points)), dtype=int)]
    minimum = max(18, int(minimum*len(sample)/len(points)))
    rng = np.random.default_rng(0)
    planes = []
    for _ in range(4):
        if len(sample) < minimum:
            break
        best = None
        best_count = 0
        for _ in range(70):
            a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
            normal = np.cross(b-a, c-a)
            length = np.linalg.norm(normal)
            if length < 1e-6:
                continue
            normal /= length
            inliers = np.abs((sample-a) @ normal) < .02
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best, best_count = inliers, count
        if best is None or best_count < minimum:
            break
        plane = _plane(sample[best])
        if plane is not None:
            planes.append(plane)
        sample = sample[~best]
    return planes


def depth_in_metres(array, encoding):
    if encoding == "32FC1":
        result = np.asarray(array, dtype=np.float32).copy()
    elif encoding == "16UC1":
        result = np.asarray(array, dtype=np.float32) * .001
    else:
        raise ValueError("depth must be 32FC1 metres or 16UC1 millimetres")
    if result.ndim != 2:
        raise ValueError("depth image must have one channel")
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


def _touching(a, b, tolerance=.12):
    """Measured 3D proximity, without equating depth medians on a deep object."""
    # Bounding boxes cheaply reject separated surfaces before exact distances.
    gap = np.maximum(0., np.maximum(a.min(axis=0)-b.max(axis=0),
                                   b.min(axis=0)-a.max(axis=0)))
    if np.linalg.norm(gap) > tolerance:
        return False
    for offset in range(0, len(a), 96):
        delta = a[offset:offset+96, None, :] - b[None, :, :]
        if np.any(np.sum(delta*delta, axis=-1) <= tolerance*tolerance):
            return True
    return False


def _single_dense_component(points, tolerance=.12):
    """Exact radius connectivity of all selected ROI-stride measurements.

    Spatial buckets only accelerate neighbor lookup. They never join points
    by voxel membership, image flood-fill membership or the depth allowance.
    This is measured connectivity, not proof of a semantic object identity.
    """
    if len(points) == 0 or not np.all(np.isfinite(points)):
        return False
    cells = np.floor(points / tolerance).astype(np.int64)
    buckets = {}
    for index, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(index)
    buckets = {key: np.asarray(value, dtype=int) for key, value in buckets.items()}
    todo, reached = [0], 1
    first = tuple(cells[0])
    buckets[first] = buckets[first][buckets[first] != 0]
    while todo:
        index = todo.pop()
        cell = cells[index]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
                    remaining = buckets.get(key)
                    if remaining is None or not len(remaining):
                        continue
                    delta = points[remaining]-points[index]
                    nearby = np.sum(delta*delta, axis=1) <= tolerance*tolerance
                    found = remaining[nearby]
                    if len(found):
                        todo.extend(found.tolist())
                        reached += len(found)
                        buckets[key] = remaining[~nearby]
    return reached == len(points)


def localize_depth_box(box, calibration, depth, camera_to_odom):
    """Return an observed odom surface and its evidence, never a bbox ray guess.

    A floating-range flood fill follows depth-continuous surfaces in a padded
    ROI. Floor returns are excluded in the saved odom frame. A surface mostly
    continuing outside the model's box cannot supply the target distance.
    Adjacent parts of one deep object can have different depth medians. Join
    measured surfaces within 12 cm; disconnected internal surfaces must fit
    the measured outer footprint, not merely receive fewer pixels/votes.
    """
    if depth.shape != (calibration.height, calibration.width):
        raise ValueError("RGB and aligned depth dimensions differ")
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in box) or not (
            0 <= x1 < x2 <= calibration.width and 0 <= y1 < y2 <= calibration.height):
        raise ValueError("invalid depth target box")
    bw, bh = x2-x1, y2-y1
    left, top = max(0, int(x1-.3*bw)), max(0, int(y1-.3*bh))
    right = min(calibration.width, int(math.ceil(x2+.3*bw)))
    bottom = min(calibration.height, int(math.ceil(y2+.3*bh)))
    stride = max(1, int(math.ceil(max(right-left, bottom-top) / 220)))
    us, vs = np.meshgrid(np.arange(left, right, stride), np.arange(top, bottom, stride))
    z = depth[top:bottom:stride, left:right:stride].copy()
    valid = np.isfinite(z) & (z >= .12) & (z <= 8.)
    # Convert optical xyz to odom with the frozen image-time rigid transform.
    xyz = np.stack(((us-calibration.cx)*z/calibration.fx,
                    (vs-calibration.cy)*z/calibration.fy, z), axis=-1)
    origin = transform_point(camera_to_odom, Vector3(0., 0., 0.))
    axes = [transform_point(camera_to_odom, Vector3(*axis)) for axis in
            ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
    matrix = np.array([[p.x-origin.x, p.y-origin.y, p.z-origin.z] for p in axes]).T
    world = xyz @ matrix.T + np.array([origin.x, origin.y, origin.z])
    valid &= world[..., 2] > .04
    inside = (us >= x1) & (us <= x2) & (vs >= y1) & (vs <= y2)
    total_inside = int(np.count_nonzero(inside))
    if np.count_nonzero(valid & inside) < max(12, .05*total_inside):
        raise ValueError("insufficient valid depth on the target")
    # Inspect outside pixels even when they are disconnected by a closed
    # object frame. Plane fitting is robust to two walls meeting at a corner.
    background_planes = _outside_planes(world[valid & ~inside], max(24, int(.04*total_inside)))
    for center, normal in background_planes:
        valid &= np.abs((world-center) @ normal) > .025
    image = np.where(valid, z, 0.).astype(np.float32)
    flood_mask = np.pad((~valid).astype(np.uint8), 1, constant_values=1)
    # Pixel removal can leave near-threshold fragments of an outside plane.
    # Keep those measured planes for the existing whole-component coplanarity
    # check too; otherwise the deleted outside pixels cannot prove continuity.
    components, rejected, outside_planes = [], [], list(background_planes)
    minimum = max(12, int(.035*total_inside))
    for row, col in np.argwhere(valid & inside):
        if flood_mask[row+1, col+1]:
            continue
        before = flood_mask.copy()
        # Relative tolerance allows slanted continuous surfaces, while an
        # actual foreground/background discontinuity remains disconnected.
        tolerance = max(.06, float(z[row, col])*.025)
        count, _, flood_mask, _ = cv2.floodFill(
            image, flood_mask, (int(col), int(row)), 0.,
            loDiff=tolerance, upDiff=tolerance,
            flags=4 | cv2.FLOODFILL_MASK_ONLY | (2 << 8))
        component = (flood_mask[1:-1, 1:-1] != before[1:-1, 1:-1])
        own = component & inside
        n_inside = int(np.count_nonzero(own))
        if n_inside < minimum:
            continue
        n_outside = int(count)-n_inside
        if n_outside > max(12, .25*n_inside):
            rejected.append({"inside": n_inside, "outside": n_outside,
                             "depth_m": float(np.median(z[own]))})
            outside_planes.append(_plane(world[component]))
            continue
        rows, cols = np.nonzero(own)
        # A one-dimensional streak is not a visible object surface.
        if np.ptp(us[own]) < min(6., bw*.15) or np.ptp(vs[own]) < min(6., bh*.15):
            continue
        components.append({"rows": rows, "cols": cols, "count": n_inside,
                           "depth": float(np.median(z[own])), "plane": _plane(world[own])})
    # A closed chair/rack frame disconnects the wall visible through its hole
    # from the same wall outside the box. Image connectivity alone would call
    # that hole a bounded target. Reject measured coplanar continuations too.
    components = [c for c in components if not any(
        _same_plane(c["plane"], outside) for outside in outside_planes)]
    if not components:
        raise ValueError("no depth surface is bounded by this object box; occluded or background")
    components.sort(key=lambda c: c["count"], reverse=True)
    main = components[0]
    clouds = [world[c["rows"], c["cols"]] for c in components]
    connected, remaining = {0}, set(range(1, len(components)))
    while remaining:
        found = {j for j in remaining if any(_touching(clouds[i], clouds[j]) for i in connected)}
        if not found:
            break
        connected.update(found)
        remaining.difference_update(found)
    # Pieces separated by a crossing pole can still belong to the same broad
    # plane; retain the previous narrow same-depth allowance only when there
    # are no competing depth layers.
    if all(abs(c["depth"]-main["depth"]) <= .35 for c in components):
        connected = set(range(len(components)))
        remaining = set()
    if remaining:
        shell = np.concatenate([clouds[i] for i in sorted(connected)])
        hull = cv2.convexHull(shell[:, :2].astype(np.float32))
        zmin, zmax = shell[:, 2].min(), shell[:, 2].max()
        for i in remaining:
            cloud = clouds[i]
            sample = cloud[np.linspace(0, len(cloud)-1, min(256, len(cloud)), dtype=int)]
            # Use the same 12 cm continuity tolerance as direct surface
            # joining. This accommodates unobserved edges of a physical
            # frame; it does not admit a separate foreground/background layer.
            enclosed = [cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True) >= -.12
                        and zmin-.12 <= p[2] <= zmax+.12 for p in sample]
            if np.mean(enclosed) < .95:
                raise _ambiguity_error(box, components, clouds, us, vs)
        connected.update(remaining)
        remaining = set()
    rows = np.concatenate([components[i]["rows"] for i in sorted(connected)])
    cols = np.concatenate([components[i]["cols"] for i in sorted(connected)])
    selected_world = world[rows, cols]
    median = np.median(selected_world, axis=0)
    # Pick an actual measured pixel closest to the surface median.
    index = int(np.argmin(np.linalg.norm(selected_world-median, axis=1)))
    point = Vector3(*map(float, selected_world[index]))
    sample = np.linspace(0, len(rows)-1, min(64, len(rows)), dtype=int)
    supports = selected_world[sample, :2].tolist()
    pixels = np.stack((us[rows[sample], cols[sample]], vs[rows[sample], cols[sample]]), axis=-1).tolist()
    # A valid surface can still have conflicting class predictions. Preserve
    # measured 3D/pixel provenance for looking again, without changing the
    # localized point or the support used for object association.
    all_pixels = np.stack((us[rows, cols], vs[rows, cols]), axis=-1)
    combined = np.column_stack((selected_world, all_pixels))
    indices = set(np.argmin(combined, axis=0)) | set(np.argmax(combined, axis=0))
    for sample_index in sample:
        if len(indices) >= 64:
            break
        indices.add(sample_index)
    indices = sorted(indices)
    surface_group = {"odom_xyz": selected_world[indices].tolist(),
                     "pixels": all_pixels[indices].astype(int).tolist(),
                     "count": len(selected_world)}
    return point, {"localization_source": "aligned_depth", "depth_m": main["depth"],
                   "depth_surface_pixels": main["count"], "depth_roi_stride": stride,
                   "depth_rejected_surfaces": rejected,
                   "depth_outside_planes": len(background_planes),
                   "depth_discarded_components": len(remaining),
                   "support_points": supports, "support_pixels": pixels,
                   "surface_group": surface_group,
                   "identity_connectivity": {"box": list(map(float, box)),
                       "contact_m": .12, "selected_count": len(selected_world),
                       "roi_stride": stride,
                       "single_component": _single_dense_component(selected_world)},
                   "depth_pixel": [int(us[rows[index], cols[index]]), int(vs[rows[index], cols[index]])]}
