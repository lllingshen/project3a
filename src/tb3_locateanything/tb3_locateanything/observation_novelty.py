"""Measured novelty is a scheduling preference, never object/absence evidence."""
import math
import numpy as np


def ranked_hint_groups(tasks, landmarks, attempts, focus_dispatched):
    """Untried groups first; more unexplained support first across all tasks.

    Only current confirmed, nonconflicted odom support explains a point, using
    Memory's existing .18 m association gate. Labels and hint creation-time
    known_object_ids are deliberately irrelevant. No task is resolved or
    removed, including groups entirely covered by existing measured surfaces.
    """
    known = []
    for obj in landmarks:
        if not obj.get('confirmed', False) or obj.get('conflicted', False):
            continue
        try:
            points = np.asarray(obj.get('support_points', []), dtype=float)
            if points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all():
                known.extend(points.tolist())
        except (TypeError, ValueError, OverflowError):
            continue  # Missing proof cannot reduce another surface's novelty.
    # Exact .18 m queries with a small spatial index keep global ranking cheap.
    # Neighbor buckets only accelerate the same measured-distance predicate.
    buckets = {}
    for x, y in known:
        buckets.setdefault((math.floor(x/.18), math.floor(y/.18)), []).append((x, y))

    def explained(point):
        x, y = point
        col, row = math.floor(x/.18), math.floor(y/.18)
        return any(math.hypot(x-a, y-b) <= .18
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for a, b in buckets.get((col+dx, row+dy), ()))
    ranked = []
    for track_id, task in tasks.items():
        for index, group in enumerate(task.get('surface_groups', [])):
            try:
                points = np.asarray(group['odom_xyz'], dtype=float)
                if (points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 64
                        or not np.isfinite(points).all()):
                    continue  # The ordinary geometry validator rejects this input too.
                points = np.unique(points[:, :2], axis=0)
                novelty = sum(not explained(point) for point in points)/len(points)
                tried = index in focus_dispatched.get(track_id, [])
                ranked.append((tried, -novelty, attempts.get(track_id, 0), track_id, index))
            except (TypeError, ValueError, OverflowError):
                continue
    return [(track_id, index, -negative, tried)
            for tried, negative, _, track_id, index in sorted(ranked)]


class SearchProgress:
    """A bounded search cursor; retain only identities, never stale safe poses."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.token = None
        self.seen = set()
        self.uncertain = set()

    def begin(self, options, token):
        if self.token != token:
            self.reset()
            self.token = token
        valid = {(row[0], row[1]) for row in options}
        self.seen.intersection_update(valid)
        self.uncertain.intersection_update(valid)

    def next(self, options):
        # Localized tasks first, then untried hint groups, then tried groups.
        for tier in (0, 1, 2):
            rows = [r for r in options if (0 if r[1] is None else 1+int(r[3])) == tier]
            for row in rows:
                if (row[0], row[1]) not in self.seen:
                    return row, False
            # All clear searches in this tier finished. Replan any remembered
            # uncertain identity against this callback's current safety inputs.
            for row in rows:
                if (row[0], row[1]) in self.uncertain:
                    return row, True
        return None

    def tested(self, row, uncertain=False):
        key = (row[0], row[1])
        self.seen.add(key)
        if uncertain:
            self.uncertain.add(key)
        else:
            self.uncertain.discard(key)
