"""Reachability and endpoint safety of the semantic parking adapter."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/tb3_nav_adapter'))
from tb3_nav_adapter.safe_approach import CostGrid, select_safe_approach
from tb3_nav_adapter.goal_adapter_core import compute_approach_pose


def grid():
    return CostGrid(61, 61, .05, -1.525, -1.525, 0., [0]*3721)


def test_legacy_nominal_math_is_unchanged():
    assert compute_approach_pose(0, 0, robot_x=-2) == (-.5, 0., 0.)


def test_free_nominal_goal_is_selected_and_faces_target():
    g = grid()
    result = select_safe_approach(g, (0, 0), (-1.2, 0))
    assert (result.x, result.y, result.yaw) == pytest.approx((-.5, 0., 0.))
    assert result.cost == 0


@pytest.mark.parametrize('cost', [-1, 99, 100])
def test_unknown_inscribed_and_lethal_nominal_cells_are_not_endpoints(cost):
    g = grid()
    g.data[g.cell(-.5, 0)] = cost
    result = select_safe_approach(g, (0, 0), (-1.2, 0))
    assert g.data[g.cell(result.x, result.y)] == 0
    assert .5 <= result.target_distance <= 1.
    assert result.nominal_offset > 0


def test_endpoint_high_cost_is_rejected_but_route_may_use_known_inflation():
    g = grid()
    for row in range(g.height):
        g.data[row*g.width + 15] = 80
    g.data[g.cell(-.5, 0)] = 80
    result = select_safe_approach(g, (0, 0), (-1.2, 0))
    assert result.cost <= 60
    assert result.reachable_cells == len(g.data)


def test_disconnected_nearest_cells_cannot_win():
    g = grid()
    for row in range(g.height):
        g.data[row*g.width + 15] = 100  # x=-.75 separates robot from nominal.
    result = select_safe_approach(g, (0, 0), (-1.2, 0))
    assert result.x < -.75
    assert .5 <= result.target_distance <= 1.


def test_unknown_wall_is_not_a_route():
    g = grid()
    for row in range(g.height):
        g.data[row*g.width + 9] = -1  # x=-1.05 keeps all annulus unreachable.
    with pytest.raises(ValueError, match='no safe reachable'):
        select_safe_approach(g, (0, 0), (-1.2, 0))


def test_no_diagonal_corner_cutting():
    g = CostGrid(3, 3, .5, -1.25, -.75, 0., [100]*9)
    g.data[0] = g.data[4] = 0
    with pytest.raises(ValueError, match='no safe reachable'):
        select_safe_approach(g, (0, 0), (-1., -.5), .5, .3, .8)


@pytest.mark.parametrize('cost', [-1, 99, 100])
def test_unsafe_robot_start_is_not_silently_cleared(cost):
    g = grid()
    g.data[g.cell(-1.2, 0)] = cost
    with pytest.raises(ValueError, match='robot start'):
        select_safe_approach(g, (0, 0), (-1.2, 0))


def test_stationary_rotation_only_when_current_cell_is_safe_and_standoff_valid():
    g = grid()
    result = select_safe_approach(g, (0, 0), (-.4, 0))
    assert (result.x, result.y, result.route_distance) == (-.4, 0, 0.)
    with pytest.raises(ValueError, match='minimum standoff'):
        select_safe_approach(g, (0, 0), (-.2, 0))
    g.data[g.cell(-.4, 0)] = 80
    assert select_safe_approach(g, (0, 0), (-.4, 0)).target_distance == pytest.approx(.5)


def test_bounded_range_cannot_select_distant_safe_cell():
    g = grid()
    for i in range(len(g.data)):
        if math.hypot(*g.center(i)) <= 1.01:
            g.data[i] = 99
    with pytest.raises(ValueError, match='declared distance range'):
        select_safe_approach(g, (0, 0), (-1.2, 0))


def test_rotated_grid_is_equivariant():
    g = grid()
    reference = select_safe_approach(g, (0, 0), (-1.2, 0))
    rotated = CostGrid(g.width, g.height, g.resolution,
                       3-g.origin_y, -2+g.origin_x, math.pi/2, g.data)
    got = select_safe_approach(rotated, (3, -2), (3, -3.2))
    assert (got.x, got.y) == pytest.approx((3-reference.y, -2+reference.x))
    assert got.yaw == pytest.approx(math.pi/2)


def test_single_cell_narrow_endpoint_is_found_without_angle_sampling():
    g = grid()
    # One narrow endpoint off all original 16 radial angles; high known costs
    # are traversable but do not satisfy the parking clearance bound.
    for i in range(len(g.data)):
        g.data[i] = 80
    endpoint = g.cell(-.7, .15)
    g.data[endpoint] = 0
    got = select_safe_approach(g, (0, 0), (-1.2, 0))
    assert (got.x, got.y) == pytest.approx(g.center(endpoint))
    assert got.candidate_cells == 1


@pytest.mark.parametrize('target', [(float('nan'), 0), (0, float('inf'))])
def test_nonfinite_coordinates_fail_closed(target):
    with pytest.raises(ValueError, match='invalid target'):
        select_safe_approach(grid(), target, (-1.2, 0))
