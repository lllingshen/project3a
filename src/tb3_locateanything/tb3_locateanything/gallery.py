"""One image of retained LA instances, with reversible source associations.

Only the model sees the gallery. Navigation must use the selected entry's
original View and source_box, never gallery pixel coordinates. A gallery helps
compare observed instances; it does not prove that an absent description will
be rejected by the model. Spatial relations need the original scene views.
"""
import base64
from dataclasses import dataclass, field
import math
import re

import cv2
import numpy as np

from .visual_memory import View
from .depth_core import AmbiguousDepthError


@dataclass
class GalleryEntry:
    object_id: str
    view: View
    source_box: list
    tile_box: list
    content_box: list
    object_box: list
    source_crop: list


@dataclass
class Gallery:
    image_bgr: np.ndarray
    image_png: str  # base64 PNG, directly accepted by the existing worker
    entries: list
    excluded_objects: dict = field(default_factory=dict)


def has_spatial_relation(command):
    """Conservatively keep viewpoint relations/coreference out of a tile grid."""
    return bool(re.search(
        r"\b(?:right(?:most)?|left(?:most)?|behind|front|beside|next|between|"
        r"near(?:er|est)?|far(?:ther|thest)?|close(?:r|st)?|furth(?:er|est)|"
        r"corners?|previous(?:ly)?|another|other|same|again)\b",
        str(command), re.IGNORECASE))


def _box(values):
    box = [float(value) for value in values]
    if (len(box) != 4 or not all(math.isfinite(v) for v in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError("invalid gallery box")
    return box


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))


def _decode(view):
    try:
        raw = base64.b64decode(view.payload["image_png"], validate=True)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (KeyError, ValueError, TypeError, cv2.error) as exc:
        raise ValueError("view has no readable PNG evidence") from exc
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("view has no readable PNG evidence")
    return image


def build_gallery(memory, requested_label=None, max_objects=12, *, allow_partial=False):
    """Return one padded tile per known retained instance, without mutating memory.

    Select each instance's largest usable box, preferring a vertically complete
    observation over a close-up cut off at the top/bottom. Horizontally clipped
    evidence cannot supply a reliable laser bearing and is excluded. Tiny boxes
    are excluded rather than enlarged into misleading detail. Class filtering
    uses LA's stored labels only; no color, simulator pose or pixel classifier.

    Empty evidence and overflow fail explicitly. Callers may request an eligible
    subset with allow_partial; omissions remain recorded and cannot be selected.
    A gallery describes retained reliable evidence, not an exhaustive room scan.
    """
    if int(max_objects) != max_objects or max_objects < 1:
        raise ValueError("max_objects must be a positive integer")
    wanted = str(requested_label).strip().lower().replace(" ", "_") if requested_label else None
    pending = [oid for oid, landmark in memory.landmarks.items()
               if not landmark.confirmed and (wanted is None or wanted == landmark.label
                                               or wanted in landmark.seen_known_labels)]
    if pending and not allow_partial:
        raise ValueError("unconfirmed class evidence in gallery candidates: " + ", ".join(sorted(pending))
                         + "; continue exploration from different positions")
    conflicts = [oid for oid, landmark in memory.landmarks.items()
                 if landmark.conflicted and (wanted is None or wanted == landmark.label
                                             or wanted in landmark.seen_known_labels)]
    if conflicts and not allow_partial:
        raise ValueError("conflicting class evidence in gallery candidates: " + ", ".join(sorted(conflicts))
                         + "; clear memory and re-observe with more reliable localization")
    expected = {oid for oid, landmark in memory.landmarks.items()
                if landmark.confirmed and not landmark.conflicted
                and (wanted is None or landmark.label == wanted)}
    best = {}
    for view in list(memory.views):
        try:
            image = _decode(view)
        except ValueError:
            continue
        height, width = image.shape[:2]
        for observation in view.observations:
            oid = observation.get("object_id")
            landmark = memory.landmarks.get(oid)
            if landmark is None or oid not in expected:
                continue
            if memory.confirm_classes and landmark.label not in observation.get("labels", [landmark.label]):
                # Promotion changes the track's class, not the original model
                # call. A large mislabelled close-up must not replace the views
                # that actually supported the confirmed identity.
                continue
            try:
                box = _box(observation["box"])
            except (KeyError, TypeError, ValueError):
                continue
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                continue
            if x1 <= max(2., width * .005) or x2 >= width - max(2., width * .005):
                continue
            if min(x2 - x1, y2 - y1) < 8 or _area(box) < 256:
                continue
            if _area(box) >= .75 * width * height:
                continue
            edge_y = max(2., height * .005)
            full_height = y1 > edge_y and y2 < height - edge_y
            score = (full_height, _area(box), view.stamp_ns)
            if oid not in best or score > best[oid][0]:
                best[oid] = (score, view, box, image)
    if not best:
        raise ValueError("no usable retained visual instance for gallery")
    missing = expected - best.keys()
    if missing and not allow_partial:
        raise ValueError("incomplete retained visual evidence for gallery: " + ", ".join(sorted(missing)))
    if len(best) > max_objects:
        raise ValueError("too many gallery candidates; use a more specific object class")
    excluded = {oid: "unconfirmed" for oid in pending}
    excluded.update({oid: "conflicting_class_evidence" for oid in conflicts})
    excluded.update({oid: "no_usable_retained_image" for oid in sorted(missing)})

    count = len(best)
    columns = 1 if count == 1 else (2 if count <= 4 else 3)
    rows = math.ceil(count / columns)
    size, gutter, outer = 320, 16, 16
    canvas = np.full((rows * size + (rows - 1) * gutter + 2 * outer,
                      columns * size + (columns - 1) * gutter + 2 * outer, 3), 238, np.uint8)
    entries = []
    for index, oid in enumerate(sorted(best)):
        _, view, source_box, image = best[oid]
        left = outer + (index % columns) * (size + gutter)
        top = outer + (index // columns) * (size + gutter)
        x1, y1, x2, y2 = source_box
        px, py = .12 * (x2 - x1), .12 * (y2 - y1)
        crop_box = [max(0, math.floor(x1 - px)), max(0, math.floor(y1 - py)),
                    min(image.shape[1], math.ceil(x2 + px)), min(image.shape[0], math.ceil(y2 + py))]
        crop = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        scale = min((size - 24) / crop.shape[1], (size - 44) / crop.shape[0])
        width, height = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
        scaled = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        dx = left + (size - width) // 2
        dy = top + 32 + (size - 44 - height) // 2
        canvas[dy:dy + height, dx:dx + width] = scaled
        cv2.putText(canvas, oid, (left + 12, top + 21), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (30, 30, 30), 1, cv2.LINE_AA)
        sx, sy = width / crop.shape[1], height / crop.shape[0]
        rendered_box = [dx + (x1 - crop_box[0]) * sx, dy + (y1 - crop_box[1]) * sy,
                        dx + (x2 - crop_box[0]) * sx, dy + (y2 - crop_box[1]) * sy]
        entries.append(GalleryEntry(oid, view, list(source_box),
            [left, top, left + size, top + size], [dx, dy, dx + width, dy + height],
            rendered_box, crop_box))
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("gallery PNG encoding failed")
    return Gallery(canvas, base64.b64encode(encoded.tobytes()).decode("ascii"), entries, excluded)


def select_gallery_box(gallery, box):
    """Map a model box to one retained instance, or fail on grid/background boxes.

    At least 80% of its area must lie in one tile and half in its image content;
    its centre must lie on that tile's catalogued object, with object IoU >= .2.
    Tile order and artificial grid positions never define scene relationships.
    """
    box = _box(box)
    height, width = gallery.image_bgr.shape[:2]
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("model box outside gallery")
    area = _area(box)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    matches = []
    for entry in gallery.entries:
        target = entry.object_box
        overlap = _intersection(box, target)
        iou = overlap / (area + _area(target) - overlap)
        if (_intersection(box, entry.tile_box) / area >= .8
                and _intersection(box, entry.content_box) / area >= .5
                and target[0] <= cx <= target[2] and target[1] <= cy <= target[3]
                and iou >= .2):
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("gallery result spans tiles, selects background, or has ambiguous identity")
    return matches[0]


def select_gallery_part(gallery, box):
    """Return only a contained *candidate*, never an identity decision.

    Unlike the full-object IoU path, a part must be wholly inside exactly one
    tile, its actual image content, and its catalogued object rectangle. The
    caller must validate its original aligned-depth support before selection.
    """
    box = _box(box)
    height, width = gallery.image_bgr.shape[:2]
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("model part outside gallery")
    def contains(outer):
        return outer[0] <= box[0] and outer[1] <= box[1] and box[2] <= outer[2] and box[3] <= outer[3]
    matches = [entry for entry in gallery.entries if all(
        contains(_box(bounds)) for bounds in (entry.tile_box, entry.content_box, entry.object_box))]
    if len(matches) != 1:
        raise ValueError("gallery part is not contained in one unique object tile")
    entry = matches[0]
    content, crop = _box(entry.content_box), _box(entry.source_crop)
    sx = (crop[2]-crop[0])/(content[2]-content[0])
    sy = (crop[3]-crop[1])/(content[3]-content[1])
    source = [crop[0]+(box[0]-content[0])*sx, crop[1]+(box[1]-content[1])*sy,
              crop[0]+(box[2]-content[0])*sx, crop[1]+(box[3]-content[1])*sy]
    return entry, _box(source)


def validate_gallery_part(memory, entry, geometry, now_ns):
    """Require unique measured support in fixed odom; no labels/votes are added.

    The .18 m distance is Memory's existing surface-association gate. This
    fallback is stricter: every measured part support must match the target,
    and none may match any other retained track, regardless of class/status.
    Missing support prevents proving uniqueness. Sparse evidence can therefore
    reject a valid part; the normal full-object selection remains unchanged.
    """
    if (not any(view is entry.view for view in memory.views)
            or not 0 <= int(now_ns)-entry.view.stamp_ns <= memory.max_age_ns):
        raise ValueError("gallery part source view expired or belongs to a different clock epoch")
    target = memory.landmarks.get(entry.object_id)
    if target is None or not target.confirmed or target.conflicted:
        raise ValueError("gallery part target is absent, unconfirmed, or conflicted")
    if geometry.get("localization_source") != "aligned_depth":
        raise ValueError("gallery part requires original aligned depth evidence")
    def support(values):
        try:
            points = np.asarray(values, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError
            points = np.unique(points, axis=0)
            if len(points) < 3:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("gallery part identity requires at least three distinct finite odom supports per track") from exc
        return points
    part = support(geometry.get("support_points"))
    records = {}
    for oid, track in memory.landmarks.items():
        retained = support(track.support_points)
        nearest = np.linalg.norm(part[:, None, :]-retained[None, :, :], axis=2).min(axis=1)
        compatible = nearest <= .18
        records[oid] = {"min_distance_m": float(nearest.min()), "max_distance_m": float(nearest.max()),
                        "compatible_part_points": int(compatible.sum()), "part_points": len(part),
                        "confirmed": bool(track.confirmed), "conflicted": bool(track.conflicted)}
        if oid == entry.object_id and not compatible.all():
            raise ValueError("gallery part depth does not agree with the complete retained target support")
        if oid != entry.object_id and compatible.any():
            raise ValueError("gallery part depth also matches another retained instance")
    return {"method": "contained_part_unique_aligned_depth_support", "support_gate_m": .18,
            "view_id": entry.view.view_id, "image_stamp_ns": entry.view.stamp_ns,
            "support_frame": entry.view.payload["odom_frame"], "object_id": entry.object_id,
            "track_support_checks": records}


def validate_gallery_part_groups(memory, entry, error, part_box, now_ns):
    """Prove one retained identity using every usable measured part group.

    Disconnected surfaces may belong to one object. This does not localize a
    merged object: each complete group must independently pass the unchanged
    single-surface identity check, including every competing retained track.
    No group is selected, dropped or averaged into a navigation coordinate.
    """
    part_box = _box(part_box)
    if (not isinstance(error, AmbiguousDepthError) or not error.hint_usable
            or error.box != tuple(part_box)
            or type(error.group_count) is not int
            or error.group_count != len(error.groups)
            or not 2 <= len(error.groups) <= 8):
        raise ValueError("gallery part requires all usable same-box depth groups")
    clouds = []
    for group in error.groups:
        try:
            cloud = np.asarray(group["odom_xyz"], dtype=float)
            pixels = np.asarray(group["pixels"], dtype=float)
            count = group["count"]
            valid = (cloud.ndim == 2 and cloud.shape[1] == 3
                     and 3 <= len(cloud) <= 64 and np.isfinite(cloud).all()
                     and pixels.shape == (len(cloud), 2) and np.isfinite(pixels).all()
                     and np.all(pixels >= 0) and np.all(pixels == np.floor(pixels))
                     and np.all(pixels >= part_box[:2]) and np.all(pixels <= part_box[2:])
                     and len(np.unique(pixels, axis=0)) == len(pixels)
                     and type(count) is int and count >= len(cloud))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid gallery part group measurement provenance") from exc
        if not valid:
            raise ValueError("invalid gallery part group measurement provenance")
        clouds.append(cloud)
    proofs = [validate_gallery_part(memory, entry,
              {"localization_source": "aligned_depth", "support_points": cloud[:, :2].tolist()}, now_ns)
              for cloud in clouds]
    return {"method": "contained_part_all_measured_groups_unique_aligned_depth_support",
            "support_gate_m": proofs[0]["support_gate_m"],
            "view_id": entry.view.view_id, "image_stamp_ns": entry.view.stamp_ns,
            "support_frame": entry.view.payload["odom_frame"], "object_id": entry.object_id,
            "group_count": len(proofs),
            "group_support_checks": [proof["track_support_checks"] for proof in proofs]}
