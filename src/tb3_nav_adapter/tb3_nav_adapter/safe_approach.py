"""Select a reachable parking point from a full Nav2 OccupancyGrid.

Values here are the published -1..100 occupancy encoding, not Costmap2D's
0..255 bytes: 99 is inscribed and 100 is lethal. Inflation/robot footprint
remain Nav2's responsibility; this helper does not invent object extents.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

from tb3_nav_adapter.goal_adapter_core import compute_approach_pose


@dataclass(frozen=True)
class CostGrid:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: Sequence[int]

    def validate(self):
        if (self.width <= 0 or self.height <= 0
                or self.width * self.height != len(self.data)
                or self.width * self.height > 1_000_000
                or not all(math.isfinite(v) for v in (
                    self.resolution, self.origin_x, self.origin_y, self.origin_yaw))
                or self.resolution <= 0
                or any(v < -1 or v > 100 for v in self.data)):
            raise ValueError("invalid or oversized full global costmap")

    def cell(self, x, y):
        dx, dy = x - self.origin_x, y - self.origin_y
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        ix = math.floor((c * dx + s * dy) / self.resolution)
        iy = math.floor((-s * dx + c * dy) / self.resolution)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return iy * self.width + ix
        return None

    def center(self, index):
        x = (index % self.width + .5) * self.resolution
        y = (index // self.width + .5) * self.resolution
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return self.origin_x + c*x - s*y, self.origin_y + s*x + c*y


@dataclass(frozen=True)
class SafeApproach:
    x: float
    y: float
    yaw: float
    cost: int
    target_distance: float
    route_distance: float
    nominal_offset: float
    reachable_cells: int
    candidate_cells: int


def select_safe_approach(grid, target_xy, robot_xy, approach_distance=.5,
                         min_standoff=.3, max_approach_distance=1.,
                         max_endpoint_cost=60):
    """Prefer the original standoff, then smaller radius and shortest route.

    All routes use known cells below the inscribed threshold. Endpoints must
    also meet the stricter published-cost bound. Search is finite (at most
    one million input cells), with no scene/category/ground-truth inputs.
    """
    grid.validate()
    tx, ty = target_xy
    rx, ry = robot_xy
    if (not all(math.isfinite(v) for v in (
            tx, ty, rx, ry, approach_distance, min_standoff, max_approach_distance))
            or not 0 < min_standoff <= approach_distance <= max_approach_distance
            or not 0 <= max_endpoint_cost < 99):
        raise ValueError("invalid target, robot or approach limits")
    nominal = compute_approach_pose(tx, ty, approach_distance, min_standoff, rx, ry)
    if nominal is None:
        raise ValueError("target is inside minimum standoff distance")
    start = grid.cell(rx, ry)
    if start is None or not 0 <= grid.data[start] < 99:
        raise ValueError("robot start is outside known traversable costmap")
    direct_distance = math.hypot(tx-rx, ty-ry)
    if direct_distance <= approach_distance and grid.data[start] <= max_endpoint_cost:
        return SafeApproach(rx, ry, nominal[2], int(grid.data[start]),
                            direct_distance, 0., 0., 1, 1)

    # Four-connected routes cannot cut diagonally across obstacle corners.
    distances = {start: 0}
    queue = deque([start])
    while queue:
        index = queue.popleft()
        ix, iy = index % grid.width, index // grid.width
        for nx, ny in ((ix-1, iy), (ix+1, iy), (ix, iy-1), (ix, iy+1)):
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            other = ny * grid.width + nx
            if other not in distances and 0 <= grid.data[other] < 99:
                distances[other] = distances[index] + 1
                queue.append(other)

    best, count = None, 0
    for index, steps in distances.items():
        if grid.data[index] > max_endpoint_cost:
            continue
        x, y = grid.center(index)
        radius = math.hypot(tx-x, ty-y)
        if not approach_distance-1e-9 <= radius <= max_approach_distance+1e-9:
            continue
        count += 1
        offset = math.hypot(x-nominal[0], y-nominal[1])
        rank = (offset, radius, steps, index)
        if best is None or rank < best[0]:
            best = (rank, index, x, y, radius, steps)
    if best is None:
        raise ValueError("no safe reachable approach within declared distance range")
    rank, index, x, y, radius, steps = best
    return SafeApproach(x, y, math.atan2(ty-y, tx-x), int(grid.data[index]),
                        radius, steps * grid.resolution, rank[0], len(distances), count)
