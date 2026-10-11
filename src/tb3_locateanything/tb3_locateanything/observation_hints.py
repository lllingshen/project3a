"""Bounded observation tasks for ambiguous measured evidence, never objects.

A task preserves every plausible measured surface. Its search anchor is not a
localized target and cannot be selected by a user command or vote in Memory.
"""
import copy
import math
import numpy as np


def _label(value):
    return '_'.join(str(value).lower().replace('_', ' ').split())


def _points(values, dimensions):
    points = np.asarray(values, dtype=float)
    if (points.ndim != 2 or points.shape[1] != dimensions or not 1 <= len(points) <= 64
            or not np.isfinite(points).all()):
        raise ValueError('invalid bounded measured surface')
    return points


def _distance(a, b):
    gap = np.maximum(0., np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)))
    lower = float(np.linalg.norm(gap))
    if lower > .25:
        return lower
    return float(np.sqrt(np.min(np.sum((a[:, None, :]-b[None, :, :])**2, axis=2))))


def _same_surface_set(near):
    # Each new group needs its own old partner. Row/column coverage alone can
    # reuse one surface twice and incorrectly share a different task's budget.
    masks = {0}
    for row in near:
        masks = {mask | (1 << index) for mask in masks
                 for index, compatible in enumerate(row)
                 if compatible and not mask & (1 << index)}
        if not masks:
            return False
    return bool(masks)


class HintMemory:
    def __init__(self, max_hints=32, max_age_sec=1800., epoch=0):
        if (isinstance(max_hints, bool) or int(max_hints) != max_hints or max_hints <= 0
                or not math.isfinite(max_age_sec) or max_age_sec <= 0):
            raise ValueError('hint capacity and lifetime must be positive')
        self.max_hints = int(max_hints)
        self.max_age_ns = int(max_age_sec*1e9)
        self.clear(epoch)

    def clear(self, epoch):
        self.epoch = int(epoch)
        self.hints = []
        self.known_surfaces = {}
        self.rejected_capacity = 0

    def add(self, label, view_id, stamp_ns, source_pose, source_camera, box, groups, landmarks):
        if not 2 <= len(groups) <= 8:
            raise ValueError('ambiguity needs all two to eight measured surfaces')
        label = _label(label)
        return self._add(label, view_id, stamp_ns, source_pose, source_camera,
                         box, groups, landmarks)

    def add_class(self, label_candidates, view_id, stamp_ns, source_pose,
                  source_camera, box, groups, landmarks):
        """Preserve conflicting current labels without selecting one of them.

        A single real measured surface may have conflicting class predictions.
        Do not duplicate it to manufacture a depth ambiguity. The caller must
        supply every measured group and its original XYZ/pixel provenance.
        """
        if isinstance(label_candidates, (str, bytes)):
            raise ValueError('class ambiguity needs two to eight distinct labels')
        candidates = list(label_candidates)
        if any(not isinstance(label, str) or not _label(label) for label in candidates):
            raise ValueError('invalid class ambiguity labels')
        candidates = sorted({_label(label) for label in candidates})
        if not 2 <= len(candidates) <= 8:
            raise ValueError('class ambiguity needs two to eight distinct labels')
        if not 1 <= len(groups) <= 8:
            raise ValueError('class ambiguity needs all one to eight measured surfaces')
        return self._add('', view_id, stamp_ns, source_pose, source_camera,
                         box, groups, landmarks, label_candidates=candidates)

    def _add(self, label, view_id, stamp_ns, source_pose, source_camera,
             box, groups, landmarks, label_candidates=None):
        landmarks = list(landmarks)
        is_class = label_candidates is not None
        if ((not label and not is_class) or not isinstance(view_id, str)
                or not view_id or stamp_ns <= 0):
            raise ValueError('invalid ambiguity provenance')
        for pose in (source_pose, source_camera):
            if len(pose) != 3 or not all(math.isfinite(float(v)) for v in pose):
                raise ValueError('invalid ambiguity source pose')
        if (len(box) != 4 or not all(math.isfinite(float(v)) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('invalid ambiguity box')
        measured = [_points(group['odom_xyz'], 3) for group in groups]
        for group, cloud in zip(groups, measured):
            if is_class and (type(group['count']) is not int or group['count'] < 1):
                raise ValueError('class surface count must be the measured pixel count')
            if len(group['pixels']) != len(cloud) or group['count'] < len(cloud):
                raise ValueError('surface pixel/sample provenance mismatch')
            pixels = np.asarray(group['pixels'], dtype=float)
            if pixels.shape != (len(cloud), 2) or not np.isfinite(pixels).all():
                raise ValueError('invalid surface pixels')
            if is_class and (np.any(pixels < 0) or np.any(pixels != np.floor(pixels))):
                raise ValueError('class surface pixels must be measured image indices')
        # Share a scheduling budget for repeated observations of the same set
        # of surfaces. This does not assert that the surfaces are one object.
        for hint in self.hints:
            if is_class:
                if (hint.get('hint_kind') != 'class_ambiguity'
                        or hint.get('label_candidates') != label_candidates):
                    continue
            elif hint.get('hint_kind') == 'class_ambiguity':
                continue
            if hint['label'] != label or len(hint['surface_groups']) != len(groups):
                continue
            old = [_points(group['odom_xyz'], 3) for group in hint['surface_groups']]
            near = np.array([[_distance(a, b) <= .25 for b in old] for a in measured])
            if _same_surface_set(near):
                hint['last_seen_ns'] = max(hint['last_seen_ns'], int(stamp_ns))
                return hint['hint_id']
        # Never evict and recreate a task to reset its external attempt budget.
        if len(self.hints) >= self.max_hints:
            self.rejected_capacity += 1
            return None
        prefix = 'class' if is_class else 'depth'
        hint_id = f'{prefix}_hint_{self.epoch}_{len(self.hints)}'
        known = []
        for obj in landmarks:
            if obj.confirmed and obj.support_points:
                known.append([list(p) for p in obj.support_points])
        hint = {'hint_id': hint_id, 'label': label, 'epoch': self.epoch,
                'stamp_ns': int(stamp_ns), 'last_seen_ns': int(stamp_ns),
                'source_view_id': view_id, 'source_pose': list(source_pose),
                'source_camera': list(source_camera), 'box': list(box),
                'surface_groups': copy.deepcopy(list(groups)),
                'known_object_ids': sorted(obj.object_id for obj in landmarks if obj.confirmed),
                'status': 'unresolved', 'resolved_by': [],
                'scope': 'Measured depth hypotheses for observation only; no object coordinate or semantic vote.'}
        if is_class:
            hint.update(hint_kind='class_ambiguity', label_candidates=list(label_candidates),
                        scope='Measured region with unresolved classes for observation only; '
                              'no object identity, chosen class or semantic vote.')
        self.hints.append(hint)
        self.known_surfaces[hint_id] = known
        return hint_id

    def reconcile(self, landmarks, view_id, pose, stamp_ns):
        """Hand a new, independently observed track to ordinary Memory checks.

        An already-confirmed background object cannot dismiss an ambiguity.
        Resolution only retires the observation task; it never confirms a track.
        """
        for hint in self.hints:
            if hint['status'] != 'unresolved':
                continue
            if stamp_ns-hint['stamp_ns'] > self.max_age_ns:
                hint['status'] = 'expired'
                continue
            if (view_id == hint['source_view_id'] or stamp_ns <= hint['stamp_ns'] or
                    math.dist(pose[:2], hint['source_pose'][:2]) < .25):
                continue
            groups = [np.asarray(group['odom_xyz'])[:, :2] for group in hint['surface_groups']]
            known = [np.asarray(p) for p in self.known_surfaces[hint['hint_id']]]
            matches = []
            labels = (hint['label_candidates'] if hint.get('hint_kind') == 'class_ambiguity'
                      else [hint['label']])
            for obj in landmarks:
                if (obj.conflicted or obj.object_id in hint['known_object_ids']
                        or _label(obj.label) not in labels or view_id not in obj.view_ids
                        or not obj.support_points):
                    continue
                points = np.asarray(obj.support_points, dtype=float)
                if any(_distance(points, old) <= .18 for old in known):
                    continue
                compatible = [i for i, group in enumerate(groups) if _distance(points, group) <= .18]
                if len(compatible) == 1:
                    matches.append(obj.object_id)
            if len(set(matches)) == 1:
                hint['status'] = 'resolved'
                hint['resolved_by'] = sorted(set(matches))
                hint['resolved_stamp_ns'] = int(stamp_ns)
                hint['resolved_view_id'] = view_id

    def serialize(self, now_ns):
        for hint in self.hints:
            if hint['status'] == 'unresolved' and now_ns-hint['stamp_ns'] > self.max_age_ns:
                hint['status'] = 'expired'
        return copy.deepcopy(self.hints)
