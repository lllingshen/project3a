"""Bounded visual history and LA-grounded instances, with no ROS/model imports.

The caller validates synchronization before adding a snapshot and owns its
payload (RGB, scan, CameraInfo and observation-time transforms). Treat stored
payloads as immutable. Memory lasts for this process/map session only: a clock
rollback clears it, and serialize() is a status summary, not a disk checkpoint.
"""

from dataclasses import dataclass, field
import math
import copy
import statistics
import re


@dataclass
class View:
    view_id: str
    stamp_ns: int
    x: float
    y: float
    yaw: float
    payload: dict
    observations: list = field(default_factory=list)
    bridge_observations: list = field(default_factory=list)


@dataclass
class Landmark:
    object_id: str
    label: str
    x: float
    y: float
    observations: int
    last_seen_ns: int
    view_ids: set = field(default_factory=set)
    descriptions: set = field(default_factory=set)
    support_points: list = field(default_factory=list)
    label_votes: dict = field(default_factory=dict)
    seen_known_labels: set = field(default_factory=set)
    conflicted: bool = False
    cannot_link: set = field(default_factory=set)
    confirmed: bool = True
    label_viewpoints: dict = field(default_factory=dict)
    confirmed_class_complete: bool = False
    confirmed_class_evidence: dict = field(default_factory=dict)
    conflict_votes: dict = field(default_factory=dict)
    identity_resolutions: list = field(default_factory=list)


class UnknownObjectId(ValueError):
    """An explicit, well-formed instance ID is absent from this memory."""


class InvalidObjectReference(ValueError):
    """A command contains an invalid or ambiguous explicit instance ID."""


def _box(box):
    values = [float(value) for value in box]
    if (len(values) != 4 or not all(math.isfinite(v) for v in values)
            or values[0] >= values[2] or values[1] >= values[3]):
        raise ValueError("box must have four finite, ordered coordinates")
    return values


def class_box_complete(box, width, height):
    """Box-border evidence quality, separate from geometric localizability.

    Use the same edge margin as validate_localizable_box. An interior box is
    stronger than an image-truncated box; it does not certify correct semantics
    or rule out an occluder within the image.
    """
    x1, y1, x2, y2 = _box(box)
    width, height = float(width), float(height)
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("image dimensions must be finite and positive")
    edge_x, edge_y = max(2.0, width * .005), max(2.0, height * .005)
    return x1 > edge_x and y1 > edge_y and x2 < width-edge_x and y2 < height-edge_y


def _complete(observation, label):
    # Older callers omit quality and retain their existing class-vote behavior.
    return observation.get("label_completeness", {}).get(label, True)


def _iou(a, b):
    area = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - area
    return area / union


def _support(points):
    """Finite planar surface evidence in the fixed odom frame, never map XY."""
    if points is None:
        return []
    result = []
    for point in points:
        if len(point) != 2:
            raise ValueError("support points must be finite odom XY pairs")
        x, y = (float(value) for value in point)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("support points must be finite odom XY pairs")
        result.append((x, y))
    return result


def _bounded_support(old, new, limit=64):
    # Deduplicate on a 2 cm grid, then retain spatial coverage rather than just
    # the newest face. The fixed budget cannot grow with exploration duration.
    cells = {}
    for x, y in old + new:
        cells.setdefault((round(x / .02), round(y / .02)), (x, y))
    remaining = sorted(cells.values())
    if len(remaining) <= limit:
        return remaining
    kept = [remaining.pop(0)]
    distances = [(x-kept[0][0])**2 + (y-kept[0][1])**2 for x, y in remaining]
    while len(kept) < limit:
        index = max(range(len(remaining)), key=distances.__getitem__)
        point = remaining.pop(index)
        distances.pop(index)
        kept.append(point)
        distances = [min(distance, (x-point[0])**2 + (y-point[1])**2)
                     for (x, y), distance in zip(remaining, distances)]
    return kept


def _surface_distance(first, second):
    return min(math.hypot(ax-bx, ay-by) for ax, ay in first for bx, by in second)


class Memory:
    def __init__(self, max_views=12, max_age_sec=1800, merge_distance=.45,
                 view_distance=.6, view_angle_rad=.5, class_labels=(), confirm_classes=False):
        if int(max_views) != max_views or max_views < 1:
            raise ValueError("max_views must be a positive integer")
        limits = (max_age_sec, merge_distance, view_distance, view_angle_rad)
        if any(not math.isfinite(v) or v <= 0 for v in limits):
            raise ValueError("memory ages and distances must be finite and positive")
        self.max_views = int(max_views)
        self.max_age_ns = int(max_age_sec * 1e9)
        self.merge_distance = float(merge_distance)
        self.view_distance = float(view_distance)
        self.view_angle_rad = float(view_angle_rad)
        self.confirm_classes = bool(confirm_classes)
        self.class_labels = {str(label).strip().lower().replace(" ", "_")
                             for label in class_labels} | {"person", "chair", "trash_can", "object"}
        self.clear()

    def clear(self):
        self.views = []
        self.landmarks = {}
        self._next_view = 0
        self._next_object = {}
        self._next_candidate = 0
        self._capture_clock_ns = None
        self._query_clock_ns = None
        self._identity_relations = {}
        # Lifetime success cap keeps retained correction audits bounded, even
        # through later ordinary merges. Only a session reset restores it.
        self._identity_resolution_count = 0

    def _time(self, stamp_ns, capture=False):
        stamp_ns = int(stamp_ns)
        if stamp_ns < 0:
            raise ValueError("negative simulation time")
        previous = self._capture_clock_ns if capture else self._query_clock_ns
        # Capture stamps normally precede query 'now' by a sensor delay: those
        # different clocks must not be compared as consecutive observations.
        if ((previous is not None and stamp_ns < previous)
                or (not capture and self._capture_clock_ns is not None
                    and stamp_ns < self._capture_clock_ns)):
            self.clear()
        if capture:
            self._capture_clock_ns = stamp_ns
        else:
            self._query_clock_ns = stamp_ns
        self._prune(stamp_ns)
        return stamp_ns

    def _prune(self, now_ns):
        self.views = [v for v in self.views if now_ns-v.stamp_ns <= self.max_age_ns]
        self.landmarks = {oid: obj for oid, obj in self.landmarks.items()
                          if now_ns-obj.last_seen_ns <= self.max_age_ns}
        for view in self.views:
            view.observations[:] = [o for o in view.observations
                                    if o["object_id"] in self.landmarks]
        retained = {}
        for view in self.views:
            for observation in view.observations:
                retained.setdefault(observation["object_id"], set()).add(view.view_id)
        # An ID without sensor evidence cannot be queried. Do not leave it in
        # the published map, and do not rewind counters when removing it.
        self.landmarks = {oid: obj for oid, obj in self.landmarks.items() if oid in retained}
        self._identity_relations = {key: record for key, record in self._identity_relations.items()
                                    if all(oid in self.landmarks for oid in key)}
        for oid, obj in self.landmarks.items():
            obj.view_ids = retained[oid]
            obj.observations = len(obj.view_ids)
            obj.cannot_link.intersection_update(self.landmarks)
            self._update_label(obj)

    def _class_votes(self, obj, complete_only=False, views=None):
        votes, viewpoints = {}, {}
        for view in self.views if views is None else views:
            labels = {label for observation in view.observations
                      if observation["object_id"] == obj.object_id
                      for label in observation.get("labels", [obj.label])
                      if not complete_only or _complete(observation, label)}
            for label in labels - {"object"}:
                if self.confirm_classes:
                    sources = viewpoints.setdefault(label, [])
                    # Quality-filter first, then independently count positions.
                    # A nearby weak view must not suppress a stronger one.
                    if any(math.hypot(view.x-source["x"], view.y-source["y"]) < .25
                           for source in sources):
                        continue
                    sources.append({"view_id": view.view_id, "x": view.x, "y": view.y})
                votes[label] = votes.get(label, 0) + 1
        return votes, viewpoints

    def _retain_confirmed_quality(self, obj):
        if not obj.confirmed or obj.confirmed_class_complete:
            return
        for view in self.views:
            for observation in view.observations:
                if (observation["object_id"] != obj.object_id
                        or obj.label not in observation.get("labels", [])):
                    continue
                complete = _complete(observation, obj.label)
                if complete or not obj.confirmed_class_evidence:
                    obj.confirmed_class_evidence = {
                        "view_id": view.view_id, "label": obj.label,
                        "box": list(observation.get("label_boxes", {}).get(obj.label, observation["box"])),
                        "complete": complete}
                    obj.confirmed_class_complete = complete
                if complete:
                    return

    def _check_class_conflict(self, obj):
        self._retain_confirmed_quality(obj)
        eligible, _ = self._class_votes(obj, complete_only=obj.confirmed_class_complete)
        obj.conflict_votes = {label: count for label, count in eligible.items() if label != obj.label}
        obj.conflicted = obj.conflicted or any(count >= 2 for count in obj.conflict_votes.values())

    def _update_label(self, obj, promotion_label=None, previous_votes=None):
        votes, viewpoints = self._class_votes(obj)
        obj.label_votes = votes
        obj.label_viewpoints = viewpoints
        if self.confirm_classes:
            obj.seen_known_labels.update(votes)
            if obj.label != "object":
                obj.seen_known_labels.add(obj.label)
            if obj.confirmed:
                # A single bad close-up is audited, not a permanent veto.
                # Two independently positioned contradictory observations
                # quarantine the stable identity; later eviction cannot undo it.
                # Image-truncated alternatives cannot overturn a class with
                # complete-box support. Retain their raw labels/votes for audit.
                self._check_class_conflict(obj)
            elif promotion_label in votes:
                ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
                winner, count = ranked[0]
                runner_up = ranked[1][1] if len(ranked) > 1 else 0
                # Only a newly accepted independent observation for the
                # winning class can promote. Pruning and repeated calls may
                # change tallies, but cannot allocate a public instance ID.
                if (winner == promotion_label and count >= 2 and count-runner_up >= 2
                        and count > (previous_votes or {}).get(winner, 0)):
                    obj.label = winner
                    obj.confirmed = True
                    # Apply the quality-aware identity rule atomically: two
                    # sufficiently complete contradictory views must not leave
                    # a newly promoted identity briefly navigable.
                    self._rename_track(obj, self._allocate_id(winner))
                    self._check_class_conflict(obj)
            return
        # Several views can repeat the same occlusion mistake: more votes do
        # not establish that the nearer laser surface belongs to that class.
        # Keep the first known class and a small lifetime audit set. Expiring
        # a contradictory view must not silently make this track navigable.
        obj.seen_known_labels.update(votes)
        if obj.label != "object":
            obj.seen_known_labels.add(obj.label)
        obj.conflicted = obj.conflicted or len(obj.seen_known_labels) > 1
        if obj.label == "object" and len(obj.seen_known_labels) == 1:
            obj.label = next(iter(obj.seen_known_labels))

    def _allocate_id(self, label):
        index = self._next_object.get(label, 0)
        self._next_object[label] = index + 1
        return "%s_%d" % (label, index)

    def _rename_track(self, obj, new_id):
        old_id = obj.object_id
        obj.object_id = new_id
        self._remap_identity_relations({old_id}, new_id)
        # Preserve physical track creation order when changing its public ID.
        self.landmarks = {new_id if oid == old_id else oid: track
                          for oid, track in self.landmarks.items()}
        for view in self.views:
            for observation in view.observations:
                if observation["object_id"] == old_id:
                    observation["object_id"] = new_id
        for track in self.landmarks.values():
            if old_id in track.cannot_link:
                track.cannot_link.remove(old_id)
                track.cannot_link.add(new_id)

    def _eviction_key(self, view):
        ids = {o["object_id"] for o in view.observations if o["object_id"] in self.landmarks}
        other_ids = {o["object_id"] for other in self.views if other is not view
                     for o in other.observations if o["object_id"] in self.landmarks}
        def conflict_evidence(views):
            return {(o["object_id"], label) for source in views for o in source.observations
                    if o["object_id"] in self.landmarks and self.landmarks[o["object_id"]].conflicted
                    for label in o.get("labels", []) if label != "object"}
        evidence = conflict_evidence([view])
        other_evidence = conflict_evidence(v for v in self.views if v is not view)
        # Prefer redundant evidence over the sole retained picture explaining
        # one side of a class conflict. If capacity forces it out, the lifetime
        # seen-label set still keeps the conflict; no-view tracks are pruned.
        coverage = (len(ids - other_ids), len(evidence - other_evidence), len(ids), view.stamp_ns)
        if not self.confirm_classes:
            return coverage
        # Compare actual retained votes with/without this image. Close redundant
        # views may substitute for one another; protect the last independent
        # source for every label, then its complete-box quality. This preserves
        # pictures, not historical votes, and uses the unchanged .25 m rule.
        remaining = [other for other in self.views if other is not view]
        losses = []
        for complete in (False, True):
            lost = 0
            for oid in ids:
                obj = self.landmarks[oid]
                before, _ = self._class_votes(obj, complete)
                after, _ = self._class_votes(obj, complete, remaining)
                lost += sum(max(0, count-after.get(label, 0)) for label, count in before.items())
            losses.append(lost)
        return coverage[:1] + tuple(losses) + coverage[1:]

    def _merge_tracks(self, tracks, *, _soft_pairs=frozenset()):
        """Join supported fragments only when no retained frame separates them."""
        ids = {obj.object_id for obj in tracks}
        if any(tuple(sorted((obj.object_id, other))) not in _soft_pairs
               for obj in tracks for other in obj.cannot_link & ids):
            raise ValueError("surface evidence matches multiple physical tracks separated by an earlier view")
        for view in self.views:
            observations = [o for o in view.observations if o["object_id"] in ids]
            for index, observation in enumerate(observations):
                if any(_iou(observation["box"], other["box"]) < .2
                       for other in observations[index + 1:]
                       if other["object_id"] != observation["object_id"]
                       and tuple(sorted((observation["object_id"], other["object_id"]))) not in _soft_pairs):
                    raise ValueError("surface evidence matches multiple physical tracks with distinct boxes in one view")
        # Dict insertion order is track creation order, unlike lexicographic ID
        # order (chair_10 is newer than chair_2). Never reuse the retired IDs.
        confirmed = {obj.object_id for obj in tracks if obj.confirmed}
        eligible = confirmed if self.confirm_classes and confirmed else ids
        survivor = next(obj for oid, obj in self.landmarks.items() if oid in eligible)
        count = sum(obj.observations for obj in tracks)
        survivor.x = sum(obj.x * obj.observations for obj in tracks) / count
        survivor.y = sum(obj.y * obj.observations for obj in tracks) / count
        survivor.last_seen_ns = max(obj.last_seen_ns for obj in tracks)
        survivor.descriptions = set().union(*(obj.descriptions for obj in tracks))
        survivor.cannot_link = set().union(*(obj.cannot_link for obj in tracks)) - ids
        survivor.seen_known_labels = set().union(*(obj.seen_known_labels for obj in tracks))
        survivor.conflicted = any(obj.conflicted for obj in tracks)
        for obj in tracks:
            if (obj.confirmed and obj.label == survivor.label
                    and obj.confirmed_class_complete and not survivor.confirmed_class_complete):
                survivor.confirmed_class_complete = True
                survivor.confirmed_class_evidence = dict(obj.confirmed_class_evidence)
        if self.confirm_classes and len({obj.label for obj in tracks if obj.confirmed}) > 1:
            survivor.conflicted = True
        survivor.support_points = _bounded_support([], [point for obj in tracks for point in obj.support_points])
        survivor.identity_resolutions = [item for obj in tracks for item in obj.identity_resolutions]
        self._remap_identity_relations(ids, survivor.object_id)
        for view in self.views:
            grouped = [o for o in view.observations if o["object_id"] in ids]
            if not grouped:
                continue
            # Retain an actual model box, not an invented union bounding box.
            best = max(grouped, key=lambda o: (o["box"][2]-o["box"][0]) * (o["box"][3]-o["box"][1]))
            labels = {label for o in grouped
                      for label in o.get("labels", [self.landmarks[o["object_id"]].label])}
            completeness = {label: any(_complete(o, label) for o in grouped
                                              if label in o.get("labels", [])) for label in labels}
            label_boxes = {}
            for label in labels:
                source = max((o for o in grouped if label in o.get("labels", [])),
                             key=lambda o: _complete(o, label))
                label_boxes[label] = list(source.get("label_boxes", {}).get(label, source["box"]))
            merged = {"object_id": survivor.object_id, "box": list(best["box"]),
                      "labels": sorted(labels), "label_completeness": completeness,
                      "label_boxes": label_boxes}
            if 'identity_depth' in best:
                merged['identity_depth'] = copy.deepcopy(best['identity_depth'])
            view.observations[:] = [o for o in view.observations if o["object_id"] not in ids] + [merged]
        for oid in ids - {survivor.object_id}:
            del self.landmarks[oid]
        for obj in self.landmarks.values():
            if obj.cannot_link & ids:
                obj.cannot_link.difference_update(ids)
                obj.cannot_link.add(survivor.object_id)
        survivor.view_ids = {v.view_id for v in self.views
                             if any(o["object_id"] == survivor.object_id for o in v.observations)}
        survivor.observations = len(survivor.view_ids)
        self._update_label(survivor)
        return survivor

    @staticmethod
    def _pose(x, y, yaw):
        pose = tuple(float(v) for v in (x, y, yaw))
        if not all(math.isfinite(v) for v in pose):
            raise ValueError("view pose must be finite")
        return pose

    def _near_views(self, x, y, yaw):
        return [v for v in self.views
                if math.hypot(v.x-x, v.y-y) < self.view_distance
                and abs(math.atan2(math.sin(v.yaw-yaw), math.cos(v.yaw-yaw))) < self.view_angle_rad]

    def should_capture(self, x, y, yaw, stamp_ns):
        x, y, yaw = self._pose(x, y, yaw)
        self._time(stamp_ns, capture=True)
        return not self._near_views(x, y, yaw)

    def add_view(self, payload, stamp_ns, x, y, yaw, *, dedicated_capture=False):
        """Keep a snapshot, or return existing evidence from the same viewpoint.

        An ungrounded near-duplicate may be replaced. An already-grounded view
        is never overwritten by a newer, possibly occluded frame at that pose.
        Callers must use the returned View.payload, which can be an older view.
        """
        x, y, yaw = self._pose(x, y, yaw)
        stamp_ns = self._time(stamp_ns, capture=True)
        # A bounded active observation explicitly needs a new exposure even
        # when an older nearby image grounded a different object. Keep both;
        # the normal per-object independent-position vote rule still applies.
        nearby = [] if dedicated_capture else self._near_views(x, y, yaw)
        grounded = [v for v in nearby if v.observations]
        if grounded:
            return max(grounded, key=lambda v: (len(v.observations), v.stamp_ns))
        for old in nearby:
            self.views.remove(old)
        if len(self.views) >= self.max_views:
            # Preserve sole evidence for an instance before redundant rich
            # views; empty views remain the first candidates for eviction.
            victim = min(self.views, key=self._eviction_key)
            self.views.remove(victim)
        view = View("view_%d" % self._next_view, stamp_ns, x, y, yaw, payload)
        self._next_view += 1
        self.views.append(view)
        self._prune(stamp_ns)
        return view

    def _require_view(self, view):
        if not any(stored is view for stored in self.views):
            raise ValueError("view is expired, evicted, or from another map session")

    def remember(self, view, label, box, x, y, description="", support_points=None,
                 class_evidence_complete=True, identity_depth=None):
        """Associate observations, using odom surface support when both have it.

        Map-point distance is only a fallback for legacy observations lacking
        support. Votes summarize all retained labels. Qualifying contradictory
        evidence creates a lifetime conflict until the track is removed/cleared.
        """
        self._require_view(view)
        if not isinstance(class_evidence_complete, bool):
            raise ValueError("class evidence completeness must be boolean")
        box = _box(box)
        x, y, _ = self._pose(x, y, 0)
        label = str(label).strip().lower().replace(" ", "_")
        if not label:
            raise ValueError("empty object label")
        support = _support(support_points)
        identity_depth = self._identity_depth(view, box, identity_depth) if self.confirm_classes else None
        if identity_depth is not None and support != _support(
                view.payload['depth_geometry'][tuple(box)][1].get('support_points')):
            identity_depth = None
        matched = self.match_box(view, box)
        matched_overlap = max((_iou(o["box"], box) for o in view.observations
                               if matched is not None and o["object_id"] == matched.object_id), default=0.)
        compatible = lambda obj: obj.label == label or "object" in (label, obj.label)
        matched_geometry = (matched is not None and
                            (not support or not matched.support_points or
                             _surface_distance(support, matched.support_points) <= .18))
        # Strong overlap is physical-track evidence even when class calls
        # disagree. Keep one track and expose the semantic conflict.
        if matched_geometry and (compatible(matched) or matched_overlap >= .6):
            obj = matched
        else:
            supported, fallback = [], []
            for candidate in self.landmarks.values():
                same_view = [o for o in view.observations if o["object_id"] == candidate.object_id]
                if any(_iou(o["box"], box) < .2 for o in same_view):
                    continue  # Distinct boxes in one frame are a cannot-link.
                if same_view and not compatible(candidate) and max(_iou(o["box"], box) for o in same_view) < .6:
                    continue
                if support and candidate.support_points:
                    distance = _surface_distance(support, candidate.support_points)
                    if distance <= .18:
                        supported.append((distance, candidate))
                elif compatible(candidate):
                    distance = math.hypot(candidate.x-x, candidate.y-y)
                    if distance <= self.merge_distance:
                        fallback.append((distance, candidate.label == "object", candidate))
            if len(supported) > 1:
                tracks = [item[1] for item in supported]
                try:
                    obj = self._merge_tracks(tracks)
                except ValueError:
                    corrected = self._bridge_identity(view, label, box, x, y, description, support,
                                                      class_evidence_complete, identity_depth, tracks)
                    if corrected is not None:
                        return corrected
                    raise
            else:
                obj = (supported[0][1] if supported else
                       min(fallback, key=lambda item: item[:2])[2] if fallback else None)
        if obj is None:
            if self.confirm_classes:
                oid = "candidate_%d" % self._next_candidate
                self._next_candidate += 1
            else:
                oid = self._allocate_id(label)
            obj = Landmark(oid, label, x, y, 0, view.stamp_ns, confirmed=not self.confirm_classes)
            self.landmarks[oid] = obj
        return self._record_observation(obj, view, label, box, x, y, description, support,
                                        class_evidence_complete, identity_depth)

    def _record_observation(self, obj, view, label, box, x, y, description, support,
                            class_evidence_complete, identity_depth):
        previous_votes = dict(obj.label_votes)
        existing = next((o for o in view.observations if o["object_id"] == obj.object_id), None)
        new_evidence = existing is None or label not in existing.get("labels", [])
        if existing is None:
            count = obj.observations
            obj.x = (obj.x * count + x) / (count + 1)
            obj.y = (obj.y * count + y) / (count + 1)
            obj.observations += 1
            existing = {"object_id": obj.object_id, "box": box, "labels": []}
            view.observations.append(existing)
        if label not in existing.setdefault("labels", []):
            existing["labels"].append(label)
        qualities = existing.setdefault("label_completeness", {})
        label_boxes = existing.setdefault("label_boxes", {})
        if label not in label_boxes or (class_evidence_complete and not qualities.get(label, False)):
            label_boxes[label] = list(box)
        qualities[label] = qualities.get(label, False) or class_evidence_complete
        if identity_depth is not None and existing['box'] == box:
            existing['identity_depth'] = copy.deepcopy(identity_depth)
        obj.support_points = _bounded_support(obj.support_points, support)
        obj.last_seen_ns = max(obj.last_seen_ns, view.stamp_ns)
        obj.view_ids.add(view.view_id)
        # Distinct tracks observed together stay distinct even after this
        # frame is evicted. A later broad box cannot erase that evidence.
        for observation in view.observations:
            other = self.landmarks.get(observation["object_id"])
            if other is not None and other is not obj:
                obj.cannot_link.add(other.object_id)
                other.cannot_link.add(obj.object_id)
                self._record_separation(view, obj, other)
        if description.strip():
            obj.descriptions.add(description.strip())
        self._update_label(obj, label if new_evidence else None, previous_votes)
        return obj

    def _identity_depth(self, view, box, proof):
        """Optional exact-cache provenance; invalid metadata disables correction."""
        try:
            if (not isinstance(proof, dict) or type(proof['epoch']) is not int or proof['epoch'] < 0
                    or proof['view_id'] != view.view_id or proof['stamp_ns'] != view.stamp_ns
                    or type(proof['depth_stamp_ns']) is not int or proof['depth_stamp_ns'] <= 0
                    or abs(proof['depth_stamp_ns']-view.stamp_ns) > 5_000_000
                    or view.payload['metadata']['image_stamp_ns'] != view.stamp_ns
                    or view.payload['metadata']['depth_stamp_ns'] != proof['depth_stamp_ns']
                    or proof['box'] != list(box) or not isinstance(proof['odom_frame'], str) or not proof['odom_frame']
                    or proof['odom_frame'] != view.payload['odom_frame']):
                return None
            cached = view.payload['depth_geometry'][tuple(box)][1]
            group, connected = proof['surface_group'], proof['connectivity']
            if (cached.get('localization_source') != 'aligned_depth'
                    or group != cached['surface_group'] or connected != cached['identity_connectivity']
                    or connected['box'] != list(box) or connected['contact_m'] != .12
                    or type(connected['single_component']) is not bool
                    or type(connected['selected_count']) is not int
                    or connected['selected_count'] != group['count']
                    or type(connected['roi_stride']) is not int or connected['roi_stride'] < 1):
                return None
            xyz, pixels = group['odom_xyz'], group['pixels']
            if (not 1 <= len(xyz) <= 64 or len(xyz) != len(pixels)
                    or type(group['count']) is not int or group['count'] < len(xyz)):
                return None
            for point, pixel in zip(xyz, pixels):
                if (len(point) != 3 or len(pixel) != 2
                        or not all(math.isfinite(float(v)) for v in list(point)+list(pixel))
                        or any(float(v) != int(v) for v in pixel)
                        or not box[0] <= pixel[0] <= box[2] or not box[1] <= pixel[1] <= box[3]):
                    return None
            return copy.deepcopy(proof)
        except (KeyError, TypeError, ValueError, OverflowError, IndexError):
            return None

    def _record_separation(self, view, first, second):
        if not self.confirm_classes:
            return
        key = tuple(sorted((first.object_id, second.object_id)))
        observations = [next(o for o in view.observations if o['object_id'] == oid) for oid in key]
        proofs = [self._identity_depth(view, o['box'], o.get('identity_depth')) for o in observations]
        valid = all(proofs) and len({(p['epoch'], p['odom_frame']) for p in proofs}) == 1
        record = self._identity_relations.setdefault(key, {'hard': False, 'sources': []})
        # Missing depth/legacy evidence remains an irrevocable separation.
        record['hard'] |= not valid or first.confirmed or second.confirmed
        if any(source['view_id'] == view.view_id for source in record['sources']):
            return
        if len(record['sources']) >= self.max_views:
            record['hard'] = True
            return
        source = {'view_id': view.view_id, 'stamp_ns': view.stamp_ns,
                  'x': view.x, 'y': view.y,
                  'observations': copy.deepcopy(observations)}
        if any(math.hypot(view.x-old['x'], view.y-old['y']) >= .25 for old in record['sources']):
            record['hard'] = True
        record['sources'].append(source)

    def _remap_identity_relations(self, old_ids, new_id):
        changed = {}
        for key, record in self._identity_relations.items():
            key = tuple(sorted({new_id if oid in old_ids else oid for oid in key}))
            if len(key) != 2:
                continue
            if key not in changed:
                changed[key] = record
            else:
                # Combining relation histories can only strengthen a barrier.
                changed[key]['hard'] = True
        self._identity_relations = changed

    def _bridge_identity(self, view, label, box, x, y, description, support, complete, proof, tracks):
        """Retain non-voting whole evidence; two independent measurements may
        correct only a single-view soft negative among unconfirmed fragments.
        """
        ids = tuple(sorted(obj.object_id for obj in tracks))
        if (not self.confirm_classes or self._identity_resolution_count >= self.max_views
                or not 2 <= len(ids) <= 8 or proof is None
                or label == 'object' or not proof['connectivity']['single_component']
                or any(obj.confirmed or obj.conflicted for obj in tracks)):
            return None
        by_id = {v.view_id: v for v in self.views}
        pairs = {tuple(sorted((a, b))) for i, a in enumerate(ids) for b in ids[i+1:]}
        relations = {}
        for key in pairs:
            record = self._identity_relations.get(key)
            if record is None or record['hard'] or not record['sources']:
                return None
            # Every original negative witness must still exist, in this epoch.
            for source in record['sources']:
                old_view = by_id.get(source['view_id'])
                if old_view is None or old_view.stamp_ns != source['stamp_ns']:
                    return None
                for observation in source['observations']:
                    old = self._identity_depth(old_view, observation['box'], observation.get('identity_depth'))
                    if old is None or (old['epoch'], old['odom_frame']) != (proof['epoch'], proof['odom_frame']):
                        return None
            relations[key] = record
        originals = []
        for obj in tracks:
            sources = [(old_view, observation) for old_view in self.views for observation in old_view.observations
                       if observation['object_id'] == obj.object_id]
            points = []
            for old_view, observation in sources:
                old = self._identity_depth(old_view, observation['box'], observation.get('identity_depth'))
                if old is None or (old['epoch'], old['odom_frame']) != (proof['epoch'], proof['odom_frame']):
                    return None
                points.extend(old['surface_group']['odom_xyz'])
            if not points:
                return None
            originals.append(points)
        negative_sources = [s for record in relations.values() for s in record['sources']]
        def valid_whole(source_view, observation):
            evidence = self._identity_depth(source_view, observation['box'], observation.get('identity_depth'))
            if (evidence is None or not evidence['connectivity']['single_component']
                    or (evidence['epoch'], evidence['odom_frame']) != (proof['epoch'], proof['odom_frame'])
                    or any(source_view.stamp_ns <= s['stamp_ns'] or
                           math.hypot(source_view.x-s['x'], source_view.y-s['y']) < .25 for s in negative_sources)):
                return False
            points = evidence['surface_group']['odom_xyz']
            if any(statistics.median(min(math.dist(p, q) for q in points) for p in old) > .18
                   for old in originals):
                return False
            # Check all current tracks; never drop a competing third match.
            matches = {obj.object_id for obj in self.landmarks.values() if obj.support_points
                       and _surface_distance(observation['support_points'], obj.support_points) <= .18}
            return matches == set(ids)
        current = {'track_ids': list(ids), 'label': label, 'box': list(box),
                   'x': x, 'y': y, 'description': description, 'support_points': list(support),
                   'class_evidence_complete': complete, 'identity_depth': proof}
        if not valid_whole(view, current):
            return None
        existing = [(v, o) for v in self.views for o in v.bridge_observations
                    if tuple(o['track_ids']) == ids]
        previous = [(v, o) for v, o in existing if v is not view and v.stamp_ns < view.stamp_ns
                    and math.hypot(v.x-view.x, v.y-view.y) >= .25 and valid_whole(v, o)]
        if not any(o['box'] == list(box) and o['label'] == label and tuple(o['track_ids']) == ids
                   for o in view.bridge_observations):
            if sum(len(v.bridge_observations) for v in self.views) >= self.max_views:
                return None
            view.bridge_observations.append(copy.deepcopy(current))
        # Preserve conflicting raw whole labels as non-voting evidence too.
        if any(o['label'] != label for _, o in existing) or not previous:
            return None
        first_view, first = min(previous, key=lambda item: item[0].stamp_ns)
        # All gates above are read-only. Commit only these verified soft edges,
        # then record two actual observations through the unchanged vote path.
        audit = {'track_ids': list(ids), 'label': label,
                 'negative_relations': [{'track_ids': list(key), **copy.deepcopy(record)}
                                        for key, record in sorted(relations.items())],
                 'whole_observations': [copy.deepcopy(first), copy.deepcopy(current)]}
        # Save only bounded identity state, never RGB/depth payload arrays.
        # A callback exception must not expose a half merge or invented vote.
        old_landmarks = dict(self.landmarks)
        track_state = [(obj, copy.deepcopy(vars(obj))) for obj in self.landmarks.values()]
        view_state = [(v, copy.deepcopy(v.observations), copy.deepcopy(v.bridge_observations))
                      for v in self.views]
        old_relations = copy.deepcopy(self._identity_relations)
        old_ids = dict(self._next_object), self._next_candidate
        old_resolution_count = self._identity_resolution_count
        try:
            survivor = self._merge_tracks(tracks, _soft_pairs=pairs)
            survivor.identity_resolutions.append(audit)
            self._identity_resolution_count += 1
            for old_view, observation in ((first_view, first), (view, current)):
                self._record_observation(survivor, old_view, observation['label'], observation['box'],
                                         observation['x'], observation['y'], observation['description'],
                                         observation['support_points'], observation['class_evidence_complete'],
                                         observation['identity_depth'])
            for retained in self.views:
                retained.bridge_observations[:] = [o for o in retained.bridge_observations
                                                  if tuple(o['track_ids']) != ids]
            return survivor
        except Exception:
            for obj, attributes in track_state:
                vars(obj).clear()
                vars(obj).update(attributes)
            self.landmarks = old_landmarks
            self._identity_relations = old_relations
            self._next_object, self._next_candidate = old_ids
            self._identity_resolution_count = old_resolution_count
            for retained, observations, bridges in view_state:
                retained.observations[:], retained.bridge_observations[:] = observations, bridges
            raise

    def match_box(self, view, box, min_iou=.2):
        self._require_view(view)
        box = _box(box)
        if not 0 < min_iou <= 1:
            raise ValueError("min_iou must be in (0, 1]")
        ranked = sorted(((_iou(box, o["box"]), o["object_id"]) for o in view.observations
                         if o["object_id"] in self.landmarks), reverse=True)
        if not ranked or ranked[0][0] < min_iou:
            return None
        if len(ranked) > 1 and abs(ranked[0][0]-ranked[1][0]) < 1e-12:
            return None  # Equally overlapping instances are not an identity.
        return self.landmarks[ranked[0][1]]

    def lookup_id(self, text):
        """Return explicit known ID; None means normal language, never ID fallback."""
        labels = self.class_labels | {obj.label for obj in self.landmarks.values()}
        if self.confirm_classes:
            labels.add("candidate")
        names = "|".join(re.escape(label).replace("_", "[ _]*")
                         for label in sorted(labels, key=len, reverse=True))
        pattern = r"\b(" + names + r")[ _]*(?:(?:number|no\.?)[ _]*)?([+-]?\d+(?:\.\d+)?)\b"
        matches = re.findall(pattern, text.lower())
        if not matches:
            return None
        if len(matches) != 1 or not matches[0][1].isdigit():
            raise InvalidObjectReference("expected one nonnegative integer object ID")
        simple = r"(?:please\s+)?(?:(?:go|move|navigate|drive)\s+to\s+)?(?:(?:the|a|an)\s+)?"
        if re.fullmatch(simple + pattern + r"[.!]?", text.strip().lower()) is None:
            raise InvalidObjectReference("use a bare object ID or a simple 'go to ID' command; additional descriptions or negation are ambiguous")
        label, index = matches[0]
        compact = re.sub(r"[ _]+", "", label)
        label = next(name for name in labels if name.replace("_", "") == compact)
        oid = "%s_%d" % (label, int(index))
        if oid not in self.landmarks:
            raise UnknownObjectId("%s not in visual memory" % oid)
        if not self.landmarks[oid].confirmed:
            raise InvalidObjectReference("%s is pending independent class confirmation" % oid)
        if self.landmarks[oid].conflicted:
            raise InvalidObjectReference("%s has conflicting class evidence; clear memory and re-observe with more reliable localization" % oid)
        return self.landmarks[oid]

    def views_for_query(self, now_ns):
        self._time(now_ns)
        return sorted(self.views, key=lambda v: (len(v.observations), v.stamp_ns), reverse=True)

    def representative_views(self, now_ns, max_views=4):
        """Return (primary, fallback) with at most max_views total snapshots.

        Prefer rich views, retaining another grounded view only when it adds an
        instance not already covered. Subset views add no identity coverage and
        may omit context needed by a relation query. Ungrounded views are a
        separate fallback for callers to try only if primary grounding fails.
        This is retrieval, not proof that a model's answer is semantically right.
        """
        if int(max_views) != max_views or max_views < 1:
            raise ValueError("max_views must be a positive integer")
        primary, fallback, covered = [], [], set()
        for view in self.views_for_query(now_ns):
            ids = {item["object_id"] for item in view.observations
                   if item["object_id"] in self.landmarks
                   and self.landmarks[item["object_id"]].confirmed
                   and not self.landmarks[item["object_id"]].conflicted}
            if ids:
                if ids - covered and len(primary) < max_views:
                    primary.append(view)
                    covered.update(ids)
            else:
                fallback.append(view)
        return primary, fallback[:int(max_views) - len(primary)]

    def serialize(self):
        """JSON-safe status only. Raw images, scans and ROS objects stay in payload."""
        result = {
            "views": [{"view_id": v.view_id, "stamp_ns": v.stamp_ns,
                       "x": v.x, "y": v.y, "yaw": v.yaw,
                       "identity_bridge_observations": copy.deepcopy(v.bridge_observations),
                       "observations": [{"object_id": o["object_id"], "box": list(o["box"]),
                                         "labels": list(o.get("labels", [])),
                                         "label_completeness": dict(o.get("label_completeness", {})),
                                         "label_boxes": {label: list(box) for label, box in o.get("label_boxes", {}).items()}}
                                        for o in v.observations]} for v in self.views],
            "landmarks": [],
        }
        if self.confirm_classes:
            result["pending_landmarks"] = []
            result['identity_resolution_count'] = self._identity_resolution_count
            result['identity_resolution_capacity'] = self.max_views
            result['identity_relations'] = [{'track_ids': list(key), **copy.deepcopy(value)}
                                            for key, value in sorted(self._identity_relations.items())
                                            if all(oid in self.landmarks for oid in key)]
        for o in self.landmarks.values():
            record = {"object_id": o.object_id, "label": o.label,
                           "x": o.x, "y": o.y, "observations": o.observations,
                           "last_seen_ns": o.last_seen_ns, "view_ids": sorted(o.view_ids),
                           "descriptions": sorted(o.descriptions), "conflicted": o.conflicted,
                           "support_points": [list(point) for point in o.support_points],
                           "evidence": {"label_votes": dict(o.label_votes),
                                        "seen_known_labels": sorted(o.seen_known_labels),
                                        "retained_views": len(o.view_ids),
                                        "cannot_link_ids": sorted(o.cannot_link)}}
            if self.confirm_classes:
                record["confirmed"] = o.confirmed
                record["evidence"].update({
                    "confirmed_class_complete": o.confirmed_class_complete,
                    "confirmed_class_evidence": dict(o.confirmed_class_evidence),
                    "conflict_votes": dict(o.conflict_votes),
                    "identity_resolutions": copy.deepcopy(o.identity_resolutions)})
                record["evidence"]["independent_viewpoints"] = {
                    label: [dict(viewpoint) for viewpoint in sources]
                    for label, sources in o.label_viewpoints.items()}
            result["landmarks" if o.confirmed else "pending_landmarks"].append(record)
        return result
