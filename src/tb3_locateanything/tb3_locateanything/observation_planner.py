"""Choose a bounded second-view goal from measured evidence and occupancy grids.

All coordinates must already share the current map frame. This module neither
dispatches navigation nor confirms objects. Map visibility is only a candidate
filter: the caller still needs a fresh, independently positioned RGB-D
observation through the normal perception and memory checks after arrival.
"""

from collections import deque
from dataclasses import dataclass
from heapq import nsmallest
import math
from statistics import median
from typing import Sequence

CLEAR_VISIBILITY = "clear_2d_map_sightline"
UNKNOWN_VISIBILITY = "unknown_3d_visibility"


@dataclass(frozen=True)
class CameraFraming:
    """Horizontal pinhole calibration and level optical camera in base frame.

    Arrival tolerances come from the active Nav2 goal checker. This models only
    measured XY support, not unseen object surfaces, height, or recognition.
    """

    width: int
    fx: float
    cx: float
    offset_x: float
    offset_y: float
    yaw_offset: float
    xy_tolerance: float
    yaw_tolerance: float
    projection_margin: float = 0.
    vertical_calibration: tuple = ()

    def limits(self):
        values = (self.fx, self.cx, self.offset_x, self.offset_y,
                  self.yaw_offset, self.xy_tolerance, self.yaw_tolerance,
                  self.projection_margin)
        if (not isinstance(self.width, int) or self.width < 8 or
                not all(math.isfinite(v) for v in values) or self.fx <= 0 or
                self.xy_tolerance < 0 or self.projection_margin < 0 or
                not 0 <= self.yaw_tolerance < math.pi/2):
            raise ValueError("invalid horizontal camera framing or navigation tolerances")
        edge = max(2., self.width*.005)  # Same edge guard as box localization.
        if not edge < self.cx < self.width-edge:
            raise ValueError("camera principal point leaves no horizontal framing margin")
        limits = (math.atan((self.width-edge-self.cx)/self.fx)-self.projection_margin,
                  math.atan((self.cx-edge)/self.fx)-self.projection_margin)
        if min(limits) <= 0:
            raise ValueError("camera projection margin leaves no horizontal field of view")
        return limits  # right, left, in radians

    def camera_xy(self, point, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return (point[0]+c*self.offset_x-s*self.offset_y,
                point[1]+s*self.offset_x+c*self.offset_y)

    def position_error(self):
        # A changed final yaw also rotates the camera's base-frame offset.
        return self.xy_tolerance + 2*math.hypot(self.offset_x, self.offset_y)*math.sin(self.yaw_tolerance/2)


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def framing_fits(camera, point, yaw, support):
    """All measured support stays within horizontal FOV at arrival tolerance.

    Translation in a disc of radius e changes a point's bearing by at most
    asin(e / range); add the goal checker's yaw error. Bounds are conservative
    and independent of class, object count, and simulator coordinates.
    """
    right, left = camera.limits()
    origin = camera.camera_xy(point, yaw)
    error = camera.position_error()
    for xy in support:
        distance = math.dist(origin, xy)
        if distance <= error:
            return False
        margin = camera.yaw_tolerance + math.asin(min(1., error/distance))
        bearing = _angle(math.atan2(xy[1]-origin[1], xy[0]-origin[0])-yaw-camera.yaw_offset)
        if bearing-margin <= -right or bearing+margin >= left:
            return False
    return len(support) > 0


def _framed_heading(camera, point, target, support):
    right, left = camera.limits()
    heading = math.atan2(target[1]-point[1], target[0]-point[0])-camera.yaw_offset
    # Camera offset depends on base yaw. A few fixed-point steps center the
    # measured angular extent in the asymmetric image, followed by an exact
    # conservative fit test; failure never relaxes its safety margins.
    for _ in range(4):
        origin = camera.camera_xy(point, heading)
        bearings = [_angle(math.atan2(p[1]-origin[1], p[0]-origin[0])-
                           heading-camera.yaw_offset) for p in support]
        heading += (min(bearings)+max(bearings))/2 - (left-right)/2
    heading = _angle(heading)
    return heading if framing_fits(camera, point, heading, support) else None


def _framing_search_limit(camera, target, support):
    right, left = camera.limits()
    # Heading centers the angular window, including an asymmetric cx.
    usable = (right+left)/2-camera.yaw_tolerance
    if usable <= 0:
        raise ValueError("navigation yaw tolerance exceeds horizontal field of view")
    extent = max(math.dist(target, p) for p in support)
    limit = max(2., 1.02*((extent+camera.position_error())/math.sin(usable) +
                         math.hypot(camera.offset_x, camera.offset_y)))
    if not math.isfinite(limit):
        raise ValueError("nonfinite framing search limit")
    return limit


def _directional_radius(camera, target, support, angle, limit):
    """One extra radius from twelve bounded framing-only bisection steps.

    A circumscribed-disc radius can miss an end-on view in a small room.
    Search each direction independently, then run the unchanged navigation
    safety checks on the resulting point. This is not a completeness claim.
    """
    def fits(radius):
        point = (target[0]+radius*math.cos(angle), target[1]+radius*math.sin(angle))
        return _framed_heading(camera, point, target, support) is not None

    low, high = 1., limit
    if fits(low):
        return low
    if not fits(high):
        return None
    for _ in range(12):
        middle = (low+high)/2
        if fits(middle):
            high = middle
        else:
            low = middle
    return high  # This bound was tested, not extrapolated from an invalid point.


@dataclass(frozen=True)
class Grid:
    """Row-major OccupancyGrid values: -1 unknown, 0 free, 100 occupied.

    Cost grids use the same 0..100 representation, not raw Nav2 0..255 bytes.
    The two grids may have different sizes, resolutions, origins and rotations.
    """

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: Sequence[int]

    def valid(self):
        try:
            return (isinstance(self.width, int) and isinstance(self.height, int)
                    and self.width > 0 and self.height > 0
                    and math.isfinite(self.resolution) and self.resolution > 0
                    and all(math.isfinite(v) for v in
                            (self.origin_x, self.origin_y, self.origin_yaw))
                    and len(self.data) == self.width * self.height
                    and all(-1 <= v <= 100 and int(v) == v for v in self.data))
        except (TypeError, ValueError, OverflowError):
            return False

    def inside(self, col, row):
        return 0 <= col < self.width and 0 <= row < self.height

    def grid_xy(self, x, y):
        dx, dy = x - self.origin_x, y - self.origin_y
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return ((c * dx + s * dy) / self.resolution,
                (-s * dx + c * dy) / self.resolution)

    def cell(self, x, y):
        gx, gy = self.grid_xy(x, y)
        return math.floor(gx), math.floor(gy)

    def world_center(self, col, row):
        dx, dy = (col + .5) * self.resolution, (row + .5) * self.resolution
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return self.origin_x + c * dx - s * dy, self.origin_y + s * dx + c * dy

    def value(self, x, y):
        try:
            col, row = self.cell(x, y)
        except (TypeError, ValueError, OverflowError):
            return -1
        return self.data[row * self.width + col] if self.inside(col, row) else -1


def _point(value, dimensions=2):
    if isinstance(value, (str, bytes)) or len(value) != dimensions:
        raise ValueError("expected finite coordinates")
    point = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in point):
        raise ValueError("expected finite coordinates")
    return point


def convex_footprint(points):
    """Validate an ordered finite convex footprint, never an inferred radius."""
    if not 3 <= len(points) <= 64:
        raise ValueError("footprint requires 3..64 vertices")
    polygon = tuple(_point(p) for p in points)
    if len(set(polygon)) != len(polygon):
        raise ValueError("footprint requires 3..64 distinct finite vertices")
    area = sum(a[0]*b[1]-a[1]*b[0]
               for a, b in zip(polygon, polygon[1:]+polygon[:1]))
    if not math.isfinite(area) or abs(area) <= 1e-9:
        raise ValueError("degenerate footprint")
    if area < 0:
        polygon = tuple(reversed(polygon))
    for index, a in enumerate(polygon):
        b, p = polygon[(index+1) % len(polygon)], polygon[(index+2) % len(polygon)]
        turn = (b[0]-a[0])*(p[1]-b[1])-(b[1]-a[1])*(p[0]-b[0])
        if not math.isfinite(turn) or turn <= 1e-9:
            raise ValueError("footprint must have strictly convex nondegenerate turns")
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        sides = [(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) for p in polygon]
        if any(not math.isfinite(side) or side < -1e-9 for side in sides):
            raise ValueError("footprint must be ordered and convex")
    return polygon


def _inside_footprint(point, polygon):
    return all((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0]) >= -1e-9
               for a, b in zip(polygon, polygon[1:]+polygon[:1]))


def _cell_corners(grid, col, row):
    c, s = math.cos(grid.origin_yaw), math.sin(grid.origin_yaw)
    return [(grid.origin_x+c*x*grid.resolution-s*y*grid.resolution,
             grid.origin_y+s*x*grid.resolution+c*y*grid.resolution)
            for x, y in ((col, row), (col+1, row), (col+1, row+1), (col, row+1))]


def _positive_overlap(a, b):
    # Convex separating-axis test in cost-grid units. A shared boundary alone
    # has zero area; every positive-area overlapping cell must be checked.
    for polygon in (a, b):
        for p, q in zip(polygon, polygon[1:]+polygon[:1]):
            axis = (q[1]-p[1], p[0]-q[0])
            length = math.hypot(*axis)
            axis = (axis[0]/length, axis[1]/length)
            pa = [x*axis[0]+y*axis[1] for x, y in a]
            pb = [x*axis[0]+y*axis[1] for x, y in b]
            if min(max(pa), max(pb))-max(min(pa), min(pb)) <= 1e-9:
                return False
    return True


def traversable_cost(cost):
    """Nav2 OccupancyGrid inflation is traversable; unknown/99/100 is not."""
    return 0 <= cost < 99


def _cleared_footprint_cell(grid, costs, col, row, polygon):
    # Inflation can refill an obstacle-cleared footprint with nonzero costs.
    # Validate the whole cell against the same bound as a known-free route.
    corners = _cell_corners(grid, col, row)
    if not all(_inside_footprint(p, polygon) for p in corners):
        return False
    projected = [costs.grid_xy(*p) for p in corners]
    # Coverage outside the cost grid cannot certify free space, even if its
    # cell centre happens to be inside. Origins/resolutions may differ.
    if any(x < -1e-9 or y < -1e-9 or x > costs.width+1e-9 or y > costs.height+1e-9
           for x, y in projected):
        return False
    covered = False
    for y in range(max(0, math.floor(min(p[1] for p in projected))),
                   min(costs.height, math.ceil(max(p[1] for p in projected)))):
        for x in range(max(0, math.floor(min(p[0] for p in projected))),
                       min(costs.width, math.ceil(max(p[0] for p in projected)))):
            square = [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]
            if _positive_overlap(projected, square):
                covered = True
                if not traversable_cost(costs.data[y*costs.width+x]):
                    return False
    return covered


def _reachable(grid, costs, robot, start_footprint=()):
    """Shortest four-connected known-free routes, as in safe_frontier.hpp."""
    distances = [-1] * len(grid.data)

    start = grid.cell(*robot)
    polygon = ()
    if grid.value(*robot) == -1 and traversable_cost(costs.value(*robot)):
        try:
            polygon = convex_footprint(start_footprint)
            if not _inside_footprint(robot, polygon):
                polygon = ()
        except (TypeError, ValueError, OverflowError):
            polygon = ()
    checked = {}

    def usable(col, row):
        if not grid.inside(col, row):
            return False
        raw = grid.data[row*grid.width+col]
        if raw == 0:
            return traversable_cost(costs.value(*grid.world_center(col, row)))
        if raw != -1 or not polygon:
            return False
        key = col, row
        if key not in checked:
            checked[key] = _cleared_footprint_cell(grid, costs, col, row, polygon)
        return checked[key]

    if not usable(*start) or not traversable_cost(costs.value(*robot)):
        return distances
    distances[start[1] * grid.width + start[0]] = 0
    pending = deque([start])
    while pending:
        col, row = pending.popleft()
        distance = distances[row * grid.width + col] + 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = col + dx, row + dy
            if usable(nx, ny) and distances[ny * grid.width + nx] < 0:
                distances[ny * grid.width + nx] = distance
                pending.append((nx, ny))
    # The bridge only connects the seed to real known-free cells. It is not
    # newly observed map space, a goal candidate, or evidence of coverage.
    return [distance if raw == 0 else -1 for raw, distance in zip(grid.data, distances)]


def _supercover(grid, start, end):
    """Grid cells touched by the segment, including both sides of a corner."""
    ax, ay = grid.grid_xy(*start)
    bx, by = grid.grid_xy(*end)
    x, y = math.floor(ax), math.floor(ay)
    finish = math.floor(bx), math.floor(by)
    dx, dy = bx - ax, by - ay
    on_vertical_edge = dx == 0 and ax == math.floor(ax)
    on_horizontal_edge = dy == 0 and ay == math.floor(ay)

    def touched(col, row):
        yield col, row
        if on_vertical_edge:
            yield col - 1, row
        if on_horizontal_edge:
            yield col, row - 1

    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    step_x = 1 / abs(dx) if dx else math.inf
    step_y = 1 / abs(dy) if dy else math.inf
    next_x = ((x + 1 - ax) if dx > 0 else (ax - x)) * step_x if dx else math.inf
    next_y = ((y + 1 - ay) if dy > 0 else (ay - y)) * step_y if dy else math.inf
    yield from touched(x, y)
    # A segment ending exactly at a grid corner can cross an axis in the
    # negative direction: stepping past t=1 would overshoot floor(end) forever.
    # Traverse the open segment, then include its actual terminal cell once.
    while (x, y) != finish and min(next_x, next_y) < 1.:
        if math.isclose(next_x, next_y, rel_tol=0, abs_tol=1e-12):
            yield x + sx, y
            yield x, y + sy
            x, y = x + sx, y + sy
            next_x, next_y = next_x + step_x, next_y + step_y
        elif next_x < next_y:
            x, next_x = x + sx, next_x + step_x
        else:
            y, next_y = y + sy, next_y + step_y
        yield from touched(x, y)
    if (x, y) != finish:
        yield finish


def _visible(grid, start, end):
    try:
        terminal = grid.cell(*end)
    except (TypeError, ValueError, OverflowError):
        return False
    if not grid.inside(*terminal):
        return False
    for col, row in _supercover(grid, start, end):
        if not grid.inside(col, row):
            return False
        value = grid.data[row * grid.width + col]
        # Looking is not driving: a measured RGB-D surface may lie above the
        # scan plane, in a terminal cell labelled occupied OR still unknown.
        # Every earlier cell must be known-free; no final distance is skipped.
        if (col, row) != terminal and value != 0:
            return False
    return True


def _surface_separation(origin, groups):
    """Mean pairwise angular separation of hypotheses, not identities.

    A lateral view can separate surfaces that overlap along the source ray.
    This ranking does not assert that any group belongs to the detected object.
    All group points still have to fit the calibrated view separately below.
    """
    centers = [(sum(p[0] for p in group)/len(group),
                sum(p[1] for p in group)/len(group)) for group in groups]
    bearings = [math.atan2(p[1]-origin[1], p[0]-origin[0]) for p in centers]
    separations = [abs(_angle(a-b)) for i, a in enumerate(bearings) for b in bearings[i+1:]]
    return sum(separations)/len(separations)


def _fallback_points(grid, costs, distances, robot, previous, attempted, separation, seen):
    """At most 64 nearby reachable cell centers, before calibrated view checks.

    Circular samples can miss a narrow free corridor. Select from the actual
    map, with bounded expensive FOV/LOS work; this is not exhaustive search.
    """
    def eligible():
        for index, route in enumerate(distances):
            if route < 0 or grid.data[index] != 0:
                continue
            point = grid.world_center(index % grid.width, index // grid.width)
            if (tuple(round(v, 9) for v in point) in seen
                    or not 0 <= costs.value(*point) <= 60
                    or any(math.dist(point, p) < separation for p in previous)
                    or any(math.dist(point, p) < .65 for p in attempted)):
                continue
            yield route, math.dist(point, robot), index, point

    return [entry[3] for entry in nsmallest(64, eligible())]


def choose_observation_pose(map_grid, cost_grid, robot_xy, target_xy,
                            previous_view_xy, attempted_xy, support_xy=(),
                            capture_viewpoints=(), camera=None,
                            hypothesis_groups=(), source_camera=None,
                            cached_viewpoints=(), start_footprint=()):
    """Compatibility API: return only a pose with a clear 2D map sightline."""
    candidate = choose_observation_candidate(
        map_grid, cost_grid, robot_xy, target_xy, previous_view_xy, attempted_xy,
        support_xy, capture_viewpoints, camera, hypothesis_groups, source_camera,
        cached_viewpoints, start_footprint=start_footprint)
    return candidate["pose"] if candidate else None


def _minimum_viewpoint_angle(point, target, references):
    """Target-relative base-position angle; a coincident point has no bearing."""
    if point == target:
        return 0.
    bearing = math.atan2(point[1] - target[1], point[0] - target[0])
    angles = [math.atan2(p[1] - target[1], p[0] - target[0])
              for p in references if p != target]
    return min((abs(math.atan2(math.sin(bearing - angle), math.cos(bearing - angle)))
                for angle in angles), default=0.)


def choose_observation_candidate(map_grid, cost_grid, robot_xy, target_xy,
                                 previous_view_xy, attempted_xy, support_xy=(),
                                 capture_viewpoints=(), camera=None,
                                 hypothesis_groups=(), source_camera=None,
                                 cached_viewpoints=(), allow_uncertain_visibility=False,
                                 dedicated_capture=False, focus_hypotheses=False,
                                 attempted_focus_groups=(), start_footprint=(),
                                 forced_focus_index=None, _reachability_cache=None,
                                 vertical_framing=None, support_xyz=(), hypothesis_xyz_groups=(),
                                 vertical_support_xyz=(), prefer_viewpoint_diversity=False):
    """Return a candidate retaining navigation, framing and independence guards.

    Sample 16 directions at each of 1, 1.5 and 2 metres from the target.
    Planned positions must clear retained observation positions by the capture
    threshold (.6 m), actual XY goal tolerance and .05 m buffer,
    >= .65 m from attempted positions, using the legacy .9 m margin when no
    camera is supplied. Distance from the current robot is not evidence
    independence: frontier navigation may already have moved to a useful view.
    Attempts and
    arrival/perception validation belong to caller.
    ``capture_viewpoints`` contains ALL retained (x, y, yaw) observations, not
    only this track's views. Avoid their capture-duplicate regions with matching
    heading, including 15 degrees capture threshold + actual yaw goal tolerance
    + .05 rad buffer. A sufficiently different heading may still provide a view.
    If ``camera`` is supplied, all known support must fit its horizontal field
    of view despite navigation tolerances. Each direction adds at most one
    framing-derived radius (64 nominal candidates total). The ROS caller
    requires this calibration. Only measured horizontal geometry is covered.
    Revalidate up to 32 prior unattempted positions from ``cached_viewpoints``;
    no previous heading is reused. If these and the rings fail, check up to 64
    reachable map-cell centers (at most 160 candidates across both phases).
    Optional depth hypotheses are observation regions, never localized objects.
    Keep their entire union in frame and prefer increased angular separation
    over shorter travel. Clear map sightlines always take precedence, including
    in the fallback phase. Explicitly enabled uncertain visibility is considered
    only after both phases have no clear candidate. It does not imply that a
    camera can see the target: a 2D scan grid cannot establish 3D visibility.
    With ``dedicated_capture=True``, the caller promises one bounded new RGB-D
    capture after successful arrival. Use the target evidence independence
    distance (.25 m), plus arrival uncertainty, and omit the global cache
    duplicate gate. This does not bypass later Memory voting or depth checks.
    Its per-task dispatched budget belongs to the caller. Every driving,
    framing and independent-view guard still applies.
    Explicit ``focus_hypotheses`` instead frames one measured surface group,
    never an identified object. All input groups remain available. Groups not
    yet dispatched are preferred, then clear visibility and smaller measured
    source-camera range. Each of at most eight groups keeps the existing 160
    candidate bound; no other geometry or voting rule changes. Optional
    ``forced_focus_index`` searches exactly one of those fully validated groups.
    The observer's private per-callback cache reuses only reachability distances;
    no candidate, framing or visibility result is reused. Optional measured
    XYZ and calibrated vertical_framing softly prefer full-height measured
    support within each visibility tier, across both candidate phases. If
    unavailable or no such pose exists, retain the original partial-view pose.
    No extra radius, candidate, budget, or unseen object height is introduced.
    Optional ``vertical_support_xyz`` is a separately provenance-validated
    pending observation sample, used only for vertical preference; original
    horizontal supports and target remain unchanged. Bad optional data falls
    back to the original XY selection. Default callers keep their behavior.
    Explicit ``prefer_viewpoint_diversity`` for a localized pending target
    prefers the largest minimum target-relative bearing change from all source
    and attempted base positions, before the existing route tie-breaks. It
    only changes ranking within the existing visibility/height policy and
    candidate phases; uncertain candidates still compete across both phases.
    Missing or target-coincident history retains the route order. Hypothesis
    groups and recursive focused hypotheses keep their existing parallax order.
    """
    if not isinstance(map_grid, Grid) or not isinstance(cost_grid, Grid):
        return None
    if not map_grid.valid() or not cost_grid.valid():
        return None
    try:
        robot, target = _point(robot_xy), _point(target_xy)
        previous = [_point(p) for p in previous_view_xy]
        attempted = [_point(p) for p in attempted_xy]
        support = [_point(p) for p in support_xy]
        captures = [_point(p, 3) for p in capture_viewpoints]
        cached = []
        for raw in cached_viewpoints:
            point = _point(raw)
            if point not in cached:
                cached.append(point)
            if len(cached) == 32:
                break
        groups = [[_point(p) for p in group] for group in hypothesis_groups]
        xyz = [_point(p, 3) for p in support_xyz]
        xyz_groups = [[_point(p, 3) for p in group] for group in hypothesis_xyz_groups]
        if xyz and (len(xyz) != len(support) or any(p[:2] != xy for p, xy in zip(xyz, support))):
            return None
        if xyz_groups and (len(xyz_groups) != len(groups) or any(
                len(a) != len(b) or any(p[:2] != xy for p, xy in zip(a, b))
                for a, b in zip(xyz_groups, groups))):
            return None
        if forced_focus_index is not None and not focus_hypotheses:
            return None
        if focus_hypotheses:
            if (camera is None or not isinstance(camera, CameraFraming)
                    or not 1 <= len(groups) <= 8
                    or any(not 1 <= len(group) <= 64 for group in groups)):
                return None
            source = _point(source_camera, 3)
            tried = set(attempted_focus_groups)
            if any(type(index) is not int or not 0 <= index < len(groups) for index in tried):
                return None
            if (forced_focus_index is not None and
                    (type(forced_focus_index) is not int or not 0 <= forced_focus_index < len(groups))):
                return None
            return _focused_candidate(map_grid, cost_grid, robot, previous, attempted,
                                      groups, source, tried, captures, camera, cached,
                                      allow_uncertain_visibility, dedicated_capture, start_footprint,
                                      forced_focus_index, _reachability_cache, vertical_framing, xyz_groups)
        if groups:
            if (camera is None or not 2 <= len(groups) <= 8
                    or any(not 1 <= len(group) <= 64 for group in groups)):
                return None
            source = _point(source_camera, 3)
            # Do not trust a caller-supplied subset to frame an ambiguity.
            support = [point for group in groups for point in group]
            xyz = [point for group in xyz_groups for point in group]
            original_separation = _surface_separation(source[:2], groups)
        if camera is not None and not isinstance(camera, CameraFraming):
            return None
        limit = _framing_search_limit(camera, target, support or [target]) if camera else None
    except (TypeError, ValueError, OverflowError):
        return None
    if map_grid.value(*robot) not in (0, -1):
        return None
    # The observer creates this private cache for one callback only. It saves
    # BFS work, never a goal, heading, framing or visibility conclusion.
    key = (id(map_grid), id(cost_grid), robot, tuple(tuple(p) for p in start_footprint))
    if _reachability_cache is not None and key in _reachability_cache:
        distances = _reachability_cache[key]
    else:
        distances = _reachable(map_grid, cost_grid, robot, start_footprint)
        if _reachability_cache is not None:
            _reachability_cache[key] = distances
    if not any(distance >= 0 for distance in distances):
        return None
    start_angle = math.atan2(robot[1] - target[1], robot[0] - target[0])
    capture_separation = ((.25 if dedicated_capture else .6) +
                          (camera.xy_tolerance if camera else .25) + .05)
    nominal = []
    for index in range(16):
        angle = start_angle + index * math.tau / 16
        radii = [1., 1.5, 2.]
        if camera:
            radius = _directional_radius(camera, target, support or [target], angle, limit)
            if radius is not None and radius not in radii:
                radii.append(radius)
        for radius in radii:
            nominal.append((target[0] + radius * math.cos(angle),
                            target[1] + radius * math.sin(angle)))
    seen = set()
    uncertain = []
    partial_clear = []
    # Pending Memory's horizontal support and one accepted depth sample use
    # different sampling schemes. Optional height never replaces XY guards.
    height_support = xyz
    if not groups and not xyz:
        try:
            if not 1 <= len(vertical_support_xyz) <= 64:
                raise ValueError("invalid bounded optional height sample")
            height_support = [_point(point, 3) for point in vertical_support_xyz]
        except (TypeError, ValueError, OverflowError):
            height_support = []
    vertical_preference = vertical_framing is not None and bool(height_support)

    def select(points):
        candidates = []
        for index, point in enumerate(points):
            key = tuple(round(v, 9) for v in point)
            if key in seen:
                continue
            seen.add(key)
            heading = math.atan2(target[1] - point[1], target[0] - point[0])
            if camera is not None:
                heading = _framed_heading(camera, point, target, support or [target])
                if heading is None:
                    continue
            if (any(math.dist(point, p) < capture_separation for p in previous)
                    or any(math.dist(point, p) < .65 for p in attempted)
                    or (not dedicated_capture and any(math.dist(point, p[:2]) < capture_separation and
                           abs(math.atan2(math.sin(heading-p[2]), math.cos(heading-p[2])))
                           < math.radians(15) + (camera.yaw_tolerance if camera else .25) + .05 for p in captures))
                    or map_grid.value(*point) != 0
                    or not 0 <= cost_grid.value(*point) <= 60):
                continue
            col, row = map_grid.cell(*point)
            route = distances[row * map_grid.width + col]
            if route >= 0:
                priority = 0.
                if groups:
                    origin = camera.camera_xy(point, heading)
                    separation = _surface_separation(origin, groups)
                    # Prefer parallax, but it is not a safety condition.
                    # Vertically separated groups may have the same XY center;
                    # changed scale or context can still help a later model view.
                    priority = -(separation-original_separation)
                elif prefer_viewpoint_diversity:
                    priority = -_minimum_viewpoint_angle(point, target, previous + attempted)
                candidates.append((priority, route, math.dist(point, robot),
                                   -math.dist(point, target), index, point, heading))
        for rank in sorted(candidates):
            point, heading = rank[-2:]
            camera_origin = camera.camera_xy(point, heading) if camera else point
            visible = [any(_visible(map_grid, camera_origin, end) for end in group)
                       for group in (groups or [support or [target]])]
            candidate = {"pose": (*point, heading), "map_sightline_groups": visible,
                         "visibility_reason": CLEAR_VISIBILITY if all(visible) else UNKNOWN_VISIBILITY}
            complete = False
            if vertical_preference:
                try:
                    prediction = vertical_framing.predict(point, heading, height_support)
                except (ValueError, OverflowError) as exc:
                    prediction = {"available": False, "arrival_tolerance_fits": False,
                                  "reason": str(exc),
                                  "scope": "Optional height prediction unavailable; retain partial-view policy."}
                candidate["vertical_framing"] = prediction
                complete = prediction["arrival_tolerance_fits"]
            if all(visible):
                if not vertical_preference or complete:
                    return candidate
                partial_clear.append((rank, candidate))
            elif allow_uncertain_visibility:
                uncertain.append((rank, candidate))
        return None

    result = select(nominal + cached)
    if result is not None:
        return result
    result = select(_fallback_points(map_grid, cost_grid, distances, robot,
                                    previous, attempted, capture_separation, seen))
    if result is not None:
        return result
    if partial_clear:
        # Nominal/cached are appended before fallback, each in the old rank
        # order. With no complete option retain the original phase preference.
        return partial_clear[0][1]
    if vertical_preference:
        complete = [item for item in uncertain if item[1]["vertical_framing"]["arrival_tolerance_fits"]]
        if complete:
            return min(complete, key=lambda item: item[0])[1]
    return min(uncertain, key=lambda item: item[0])[1] if uncertain else None


def _focused_candidate(maps, costs, robot, previous, attempted, groups, source,
                       tried, captures, camera, cached, allow_uncertain, dedicated, start_footprint=(),
                       forced_focus_index=None, reachability_cache=None, vertical_framing=None, xyz_groups=()):
    """Investigate a measured hypothesis without assigning it an identity.

    At most eight ordinary bounded searches. Within each untried/tried tier,
    search nearer measured groups first and retain unknown visibility only if
    no group in that tier has a clear candidate. Group indices are stable in
    the immutable hint record; only the caller's actual dispatch consumes one.
    """
    anchors = [tuple(median(p[axis] for p in group) for axis in (0, 1)) for group in groups]
    ranges = [math.dist(anchor, source[:2]) for anchor in anchors]
    for was_tried in (False, True):
        uncertain = None
        indices = sorted((i for i in range(len(groups)) if (i in tried) == was_tried
                          and (forced_focus_index is None or i == forced_focus_index)),
                         key=lambda i: (ranges[i], i))
        for index in indices:
            candidate = choose_observation_candidate(
                maps, costs, robot, anchors[index], previous, attempted, groups[index],
                capture_viewpoints=captures, camera=camera, cached_viewpoints=cached,
                allow_uncertain_visibility=allow_uncertain, dedicated_capture=dedicated,
                start_footprint=start_footprint, _reachability_cache=reachability_cache,
                vertical_framing=vertical_framing, support_xyz=xyz_groups[index] if xyz_groups else ())
            if candidate is None:
                continue
            candidate.update(focus_group_index=index, hypothesis_group_count=len(groups),
                             focus_source_range_m=ranges[index],
                             framing_scope="one_measured_hypothesis_group")
            if candidate["visibility_reason"] == CLEAR_VISIBILITY:
                return candidate
            if uncertain is None:
                uncertain = candidate
        if uncertain is not None:
            return uncertain
    return None
