"""Geometric safety and independent-view regressions; no ROS or world truth."""

from dataclasses import replace
import math
import json
from pathlib import Path

import pytest

from tb3_locateanything.observation_planner import (
    CameraFraming, Grid, _supercover, _visible, choose_observation_pose,
    framing_fits, _framed_heading, _framing_search_limit, _directional_radius,
    _surface_separation, _fallback_points, _reachable,
)
from tb3_locateanything import observation_planner as planner
from tb3_locateanything.visual_memory import Memory


def room(resolution=.1, width_m=8., yaw=0., origin=(-4., -4.)):
    size = round(width_m / resolution)
    return Grid(size, size, resolution, *origin, yaw, [0] * (size * size))


def paint(grid, predicate, value):
    data = list(grid.data)
    for row in range(grid.height):
        for col in range(grid.width):
            if predicate(*grid.world_center(col, row)):
                data[row * grid.width + col] = value
    return replace(grid, data=data)


def set_cell(grid, xy, value):
    data = list(grid.data)
    col, row = grid.cell(*xy)
    data[row * grid.width + col] = value
    return replace(grid, data=data)


def test_free_room_selects_translated_reachable_view_facing_target_without_mutation():
    grid = room()
    original = list(grid.data)
    robot, target, previous = (-3., -.03), (.03, -.03), [(-1., -.03)]
    goal = choose_observation_pose(grid, grid, robot, target, previous, [])
    assert goal is not None
    x, y, yaw = goal
    assert math.dist((x, y), previous[0]) >= .9
    assert math.dist((x, y), robot) > .5
    assert math.dist((x, y), target) == pytest.approx(2.)
    assert yaw == pytest.approx(math.atan2(target[1] - y, target[0] - x))
    assert grid.value(x, y) == 0 and grid.data == original


def test_ambiguous_layers_choose_lateral_parallax_while_framing_entire_union():
    grid = room(width_m=12., origin=(-6., -6.))
    camera = CameraFraming(640, 530., 319.5, .077, 0., 0., .25, .25)
    groups = [[(x, -.15), (x, .15)] for x in (-.7, .7)]
    source = (-3., 0., 0.)
    goal = choose_observation_pose(grid, grid, source[:2], (0., 0.), [source[:2]], [],
                                  camera=camera, hypothesis_groups=groups, source_camera=source)
    assert goal is not None
    origin = camera.camera_xy(goal[:2], goal[2])
    assert _surface_separation(origin, groups) > _surface_separation(source[:2], groups)
    assert framing_fits(camera, goal[:2], goal[2], [p for group in groups for p in group])
    assert abs(goal[1]) > .5


def test_ambiguity_does_not_drop_far_group_to_fit_small_room():
    grid = room(width_m=3., origin=(-1.5, -1.5))
    camera = CameraFraming(640, 530., 319.5, .077, 0., 0., .25, .25)
    groups = [[(-.8, -2.), (-.8, 2.)], [(.8, -2.), (.8, 2.)]]
    assert choose_observation_pose(grid, grid, (0., -1.), (0., 0.), [], [],
                                  support_xy=[(0., 0.)], camera=camera,
                                  hypothesis_groups=groups, source_camera=(0., -3., 0.)) is None


def test_xy_coincident_vertical_layers_do_not_veto_other_pair_parallax():
    grid = room(width_m=12., origin=(-6., -6.))
    camera = CameraFraming(640, 530., 319.5, .077, 0., 0., .25, .25)
    # Two separate XYZ surfaces share this XY footprint. A third layer can
    # separate horizontally; a minimum-pair score would stay zero everywhere.
    groups = [[(-.7, -.1), (-.7, .1)], [(-.7, -.1), (-.7, .1)],
              [(.7, -.1), (.7, .1)]]
    source = (-3., 0., 0.)
    goal = choose_observation_pose(grid, grid, source[:2], (0., 0.), [source[:2]], [],
                                  camera=camera, hypothesis_groups=groups, source_camera=source)
    assert goal is not None
    assert _surface_separation(camera.camera_xy(goal[:2], goal[2]), groups) > 0.
    same = [groups[0], groups[1]]
    assert choose_observation_pose(grid, grid, source[:2], (-.7, 0.), [source[:2]], [],
                                   camera=camera, hypothesis_groups=same, source_camera=source) is not None


def test_ambiguity_requires_visibility_to_each_hypothesis_not_only_nearest():
    grid = room(width_m=12., origin=(-6., -6.))
    # The second measured hypothesis is enclosed behind occupied cells. The
    # nearest group stays visible, but cannot stand in for the missing group.
    grid = paint(grid, lambda x, y: .35 < math.hypot(x-1., y) < .6, 100)
    camera = CameraFraming(640, 530., 319.5, .077, 0., 0., .25, .25)
    groups = [[(-1., -.1), (-1., .1)], [(1., -.1), (1., .1)]]
    assert choose_observation_pose(grid, grid, (-3., 0.), (0., 0.), [], [], camera=camera,
                                  hypothesis_groups=groups, source_camera=(-3., 0., 0.)) is None


@pytest.mark.parametrize('groups', [[[(0., 0.)]], [[(0., 0.)], []],
                                  [[(float('nan'), 0.)], [(1., 0.)]],
                                  [[(0., 0.)]]*9])
def test_malformed_or_unbounded_hypotheses_fail_closed(groups):
    camera = CameraFraming(640, 530., 319.5, 0., 0., 0., .25, .25)
    assert choose_observation_pose(room(), room(), (-3., 0.), (0., 0.), [], [],
                                  camera=camera, hypothesis_groups=groups,
                                  source_camera=(-3., 0., 0.)) is None


@pytest.mark.parametrize("resolution", [.05, .1])
@pytest.mark.parametrize("barrier", [-1, 100])
def test_unknown_or_occupied_wall_separates_robot_from_all_target_viewpoints(resolution, barrier):
    grid = room(resolution, 12., origin=(-6., -6.))
    grid = paint(grid, lambda x, y: -.3 < x < .3, barrier)
    # Even the leftmost 2 m target-ring sample lies on the other side.
    assert choose_observation_pose(grid, room(resolution, 12., origin=(-6., -6.)),
                                   (-3., .03), (3., .03), [], []) is None


@pytest.mark.parametrize("terminal_value", [0, 100, -1])
def test_target_terminal_value_does_not_block_a_look_ray(terminal_value):
    grid = room()
    target = (.03, .03)
    grid = set_cell(grid, target, terminal_value)
    assert _visible(grid, (-1.03, .03), target)
    assert choose_observation_pose(grid, room(), (-3.03, .03), target, [], []) is not None


@pytest.mark.parametrize("blocker", [100, -1])
def test_obstacle_or_unknown_immediately_before_target_is_not_skipped(blocker):
    grid = room()
    grid = set_cell(grid, (-.03, .03), blocker)
    assert not _visible(grid, (-1.03, .03), (.03, .03))


def test_supercover_includes_both_sides_of_corner_and_exact_grid_edge():
    grid = Grid(4, 4, 1., 0., 0., 0., [0] * 16)
    assert {(1, 0), (0, 1)} <= set(_supercover(grid, (.5, .5), (2.5, 2.5)))
    corner = set_cell(grid, (1.5, .5), 100)
    assert not _visible(corner, (.5, .5), (2.5, 2.5))
    edge = set_cell(grid, (1.5, .5), 100)
    assert not _visible(edge, (.5, 1.), (2.5, 1.))


@pytest.mark.parametrize("start,end", [
    ((3.5, 1.5), (2., 3.)), ((1.5, 3.5), (3., 2.)),
    ((3.5, 3.5), (2., 2.)), ((.5, .5), (2., 2.)),
])
def test_corner_endpoint_with_negative_direction_never_overshoots(start, end):
    grid = Grid(5, 5, 1., 0., 0., 0., [0] * 25)
    # islice makes a non-terminating traversal a finite test failure.
    from itertools import islice
    cells = list(islice(_supercover(grid, start, end), 30))
    assert len(cells) < 30
    assert cells[-1] == grid.cell(*end)
    assert _visible(grid, start, end)


def test_ray_to_a_measured_front_surface_can_succeed_when_center_is_occluded():
    grid = paint(room(), lambda x, y: -.2 < x < .1 and abs(y) < .3, 100)
    start, center, surface = (-2., .03), (.33, .03), (-.13, .03)
    assert not _visible(grid, start, center)
    assert _visible(grid, start, surface)
    # A reachable side remains usable when the target's current box centre
    # lies behind its observed front surface.
    assert choose_observation_pose(grid, room(), (-3.03, .03), center,
                                   [], [], [surface]) is not None


def test_all_view_samples_occluded_even_though_navigation_endpoints_are_free():
    grid = paint(room(), lambda x, y: .25 < math.hypot(x, y) < .5, 100)
    assert choose_observation_pose(grid, room(), (-3.03, .03), (.03, .03), [], []) is None


def test_high_cost_endpoints_and_cost_barrier_fail_closed():
    grid = room()
    costs = paint(grid, lambda x, y: math.hypot(x - .03, y - .03) <= 2.2, 61)
    goal = choose_observation_pose(grid, costs, (-3.03, .03), (.03, .03), [], [])
    assert goal and costs.value(*goal[:2]) <= 60  # Safe map cells outside the rings remain legal.
    costs = paint(grid, lambda x, y: -2.5 < x < -2.2, 99)
    walled_map = paint(grid, lambda x, y: -2.5 < x < -2.2, 100)
    assert choose_observation_pose(walled_map, costs, (-3.03, .03), (.03, .03), [], []) is None


def test_reachability_cannot_cut_a_diagonal_costmap_corner():
    grid = Grid(4, 4, 1., 0., 0., 0., [0] * 16)
    costs = replace(grid, data=[99] * 16)
    costs = set_cell(set_cell(costs, (.5, .5), 0), (1.5, 1.5), 0)
    assert choose_observation_pose(grid, costs, (.5, .5), (2.5, 2.5), [(.5, .5)], []) is None


def test_failed_viewpoints_are_not_repeated_and_budget_can_exhaust_candidates():
    grid = room()
    robot, target = (-3.03, .03), (.03, .03)
    first = choose_observation_pose(grid, grid, robot, target, [], [])
    second = choose_observation_pose(grid, grid, robot, target, [], [first[:2]])
    assert first and second and math.dist(first[:2], second[:2]) >= .65
    # Cover the full free room, including the new bounded map-cell fallback.
    samples = [(x/2, y/2) for x in range(-8, 9) for y in range(-8, 9)]
    assert choose_observation_pose(grid, grid, robot, target, [], samples) is None
    assert choose_observation_pose(grid, grid, robot, target, samples, []) is None


def test_sampling_and_pose_transform_with_a_rotated_translated_map():
    angle = .63
    c, s = math.cos(angle), math.sin(angle)

    def transform(point):
        x, y = point
        return 5. + c * x - s * y, -2. + s * x + c * y

    plain = room()
    rotated = replace(plain, origin_x=transform((-4., -4.))[0],
                      origin_y=transform((-4., -4.))[1], origin_yaw=angle)
    robot, target, previous = (-3., -.03), (.03, -.03), [(-1., -.03)]
    first = choose_observation_pose(plain, plain, robot, target, previous, [])
    second = choose_observation_pose(rotated, rotated, transform(robot), transform(target),
                                    list(map(transform, previous)), [])
    assert second[:2] == pytest.approx(transform(first[:2]))
    assert math.sin(second[2] - first[2] - angle) == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("resolution", [.05, .1])
def test_map_and_cost_grid_can_have_distinct_resolution_and_origin(resolution):
    grid = room(resolution)
    costs = room(.2, 10., origin=(-5., -5.))
    goal = choose_observation_pose(grid, costs, (-3., -.03), (.03, -.03), [], [])
    assert goal and costs.value(*goal[:2]) == 0 and grid.value(*goal[:2]) == 0


def test_rotated_costmap_barrier_is_checked_in_world_coordinates():
    grid = room()
    costs = room(.2, 10., yaw=math.pi / 2, origin=(5., -5.))
    costs = paint(costs, lambda x, y: -2.6 < x < -2.2, 99)
    goal = choose_observation_pose(grid, costs, (-3.03, .03), (.03, .03), [], [])
    # A distant look from the robot's side is legal; crossing the cost barrier is not.
    assert goal and goal[0] < -2.6 and costs.value(*goal[:2]) <= 60


@pytest.mark.parametrize("bad_grid", [
    Grid(1, 1, 0., 0., 0., 0., [0]),
    Grid(2, 2, 1., 0., 0., 0., [0]),
    Grid(1, 1, 1., math.nan, 0., 0., [0]),
    Grid(1, 1, 1., 0., 0., 0., [255]),
    None,
])
def test_invalid_grids_return_none(bad_grid):
    assert choose_observation_pose(bad_grid, room(), (-3., .03), (.03, .03), [], []) is None


@pytest.mark.parametrize("points", [[(math.nan, 0.)], [None], [(0.,)], None])
def test_invalid_history_returns_none(points):
    assert choose_observation_pose(room(), room(), (-3., .03), (.03, .03), points, []) is None


def test_unknown_robot_cell_is_not_snapped_to_nearby_free_space():
    robot = (-3.03, .03)
    grid = set_cell(room(), robot, -1)
    assert choose_observation_pose(grid, room(), robot, (.03, .03), [], []) is None


def test_other_objects_retained_view_cannot_consume_a_revisit_without_new_capture():
    memory = Memory(max_views=32, confirm_classes=True, view_angle_rad=math.radians(15))
    old = memory.add_view({}, 1_000_000_000, -2., 0., 0.)
    memory.remember(old, 'object_a', [0, 0, 10, 10], 0., 0.,
                    support_points=[(0., 0.), (.01, 0.), (.02, 0.)])
    other = memory.add_view({}, 2_000_000_000, 2., 0., math.pi)
    memory.remember(other, 'object_b', [30, 30, 40, 40], 0., 3.,
                    support_points=[(0., 3.), (.01, 3.), (.02, 3.)])
    grid = room()
    costs = replace(grid, data=[98] * len(grid.data))
    for point in ((3., 0.), (2., 0.), (math.sqrt(2), math.sqrt(2))):
        costs = set_cell(costs, point, 0)
    before = choose_observation_pose(grid, costs, (3., 0.), (0., 0.), [(-2., 0.)], [])
    assert before == pytest.approx((2., 0., math.pi))
    assert not memory.should_capture(*before, 3_000_000_000)
    after = choose_observation_pose(
        grid, costs, (3., 0.), (0., 0.), [(-2., 0.)], [],
        capture_viewpoints=[(v.x, v.y, v.yaw) for v in memory.views])
    assert after is not None and math.dist(after[:2], before[:2]) >= .9
    assert memory.should_capture(*after, 4_000_000_000)


def test_same_position_with_sufficiently_different_camera_heading_remains_eligible():
    grid = room()
    goal = choose_observation_pose(grid, grid, (3., 0.), (0., 0.), [], [],
                                   capture_viewpoints=[(2., 0., 0.)])
    assert goal == pytest.approx((2., 0., math.pi))


def test_heading_duplicate_guard_wraps_at_pi_and_includes_arrival_tolerance():
    grid = room()
    limit = math.radians(15) + .25 + .05
    blocked = choose_observation_pose(grid, grid, (3., 0.), (0., 0.), [], [],
                                      capture_viewpoints=[(2., 0., -math.pi + limit - .01)])
    assert blocked is not None and math.dist(blocked[:2], (2., 0.)) > .01
    allowed = choose_observation_pose(grid, grid, (3., 0.), (0., 0.), [], [],
                                      capture_viewpoints=[(2., 0., -math.pi + limit + .01)])
    assert allowed == pytest.approx((2., 0., math.pi))


@pytest.mark.parametrize("captures", [[(0., 0.)], [(0., 0., math.nan)], [None], None])
def test_invalid_capture_viewpoints_fail_closed(captures):
    assert choose_observation_pose(room(), room(), (3., 0.), (0., 0.), [], [],
                                   capture_viewpoints=captures) is None


def calibrated(**changes):
    return replace(CameraFraming(640, 530., 319.5, .077, 0., 0., .25, .25), **changes)


@pytest.mark.parametrize('half_width', [.08, .65, 1.1])
def test_small_and_wide_support_fit_actual_calibration_and_arrival_uncertainty(half_width):
    support = [(-half_width, -.2), (-half_width, .2), (half_width, -.2), (half_width, .2)]
    grid = room(width_m=12., origin=(-6., -6.))
    camera = calibrated()
    goal = choose_observation_pose(grid, grid, (0., -4.), (0., 0.), [], [], support, camera=camera)
    assert goal is not None and framing_fits(camera, goal[:2], goal[2], support)
    # Check real pixel projections at an independent finite set of arrival
    # perturbations, not just a second invocation of the acceptance predicate.
    edge = max(2., camera.width*.005)
    for turn in (-camera.yaw_tolerance, 0., camera.yaw_tolerance):
        for index in range(16):
            angle = math.tau*index/16
            base = (goal[0]+camera.xy_tolerance*math.cos(angle),
                    goal[1]+camera.xy_tolerance*math.sin(angle))
            yaw = goal[2]+turn
            origin = camera.camera_xy(base, yaw)
            for x, y in support:
                bearing = math.atan2(y-origin[1], x-origin[0])-yaw-camera.yaw_offset
                u = camera.cx-camera.fx*math.tan(bearing)
                assert edge < u < camera.width-edge


def test_noncentral_principal_point_and_lateral_rotated_camera_change_final_heading():
    camera = calibrated(cx=220., offset_y=.2, yaw_offset=.31)
    support = [(-.4, -.1), (.4, .1)]
    point, target = (0., -3.), (0., 0.)
    heading = _framed_heading(camera, point, target, support)
    assert heading is not None and framing_fits(camera, point, heading, support)
    assert abs(heading-math.pi/2) > .1
    centered = _framed_heading(replace(camera, cx=320.), point, target, support)
    assert abs(centered-heading) > .1


def test_frozen_view198_is_clipped_even_without_navigation_error():
    data = json.loads((Path(__file__).parent/'fixtures/observation_view198.json').read_text())
    camera = CameraFraming(**data['camera'], xy_tolerance=0., yaw_tolerance=0.)
    assert not framing_fits(camera, data['robot_pose'][:2], data['robot_pose'][2], data['support_xy'])
    # A controlled open-grid replay verifies a new framing is possible, not
    # that this replacement was safe in the original live global costmap.
    camera = replace(camera, xy_tolerance=.25, yaw_tolerance=.25)
    target = (0., .18)
    grid = room(width_m=12., origin=(-6., -6.))
    goal = choose_observation_pose(grid, grid, data['robot_pose'][:2], target,
                                  [(-1.3489, 2.1698)], [], data['support_xy'], camera=camera)
    assert goal is not None and framing_fits(camera, goal[:2], goal[2], data['support_xy'])


def test_camera_aware_search_remains_rotation_equivariant():
    angle = .73
    def rotate(p):
        return (math.cos(angle)*p[0]-math.sin(angle)*p[1],
                math.sin(angle)*p[0]+math.cos(angle)*p[1])
    grid = room()
    origin = rotate((-4., -4.))
    turned = replace(grid, origin_x=origin[0], origin_y=origin[1], origin_yaw=angle)
    support = [(-.7, -.2), (-.7, .2), (.7, -.2), (.7, .2)]
    camera = calibrated(offset_y=.1, yaw_offset=.2)
    first = choose_observation_pose(grid, grid, (0., -3.), (0., 0.), [], [], support, camera=camera)
    second = choose_observation_pose(turned, turned, rotate((0., -3.)), (0., 0.), [], [],
                                    list(map(rotate, support)), camera=camera)
    assert first is not None and second is not None
    assert second[:2] == pytest.approx(rotate(first[:2]))
    assert math.sin(second[2]-first[2]-angle) == pytest.approx(0., abs=1e-10)


def test_narrow_free_corridor_cannot_relax_fov_to_manufacture_a_goal():
    grid = paint(room(), lambda x, y: not (-.2 < x < .2 and -1.7 < y < -.65), 100)
    # Candidate support need not be a navigable cell, but no safe endpoint
    # in this short corridor can fit the wide horizontal extent.
    support = [(-.9, 0.), (.9, 0.)]
    assert choose_observation_pose(grid, grid, (0., -1.6), (0., 0.), [], [], support,
                                   camera=calibrated()) is None


def test_directional_radius_recovers_end_on_view_missed_by_global_size_rings():
    grid = Grid(100, 60, .05, -2.5, -1.5, 0., [0]*6000)
    support = [(-1., -.03), (-1., .03), (1., -.03), (1., .03)]
    camera = calibrated(fx=530.46694)
    # The circumscribed-disc bound exceeds the room, but this long thin object
    # fits end-on at an intermediate radius just beyond the original 2 m ring.
    limit = _framing_search_limit(camera, (0., 0.), support)
    assert limit > 4.
    radius = _directional_radius(camera, (0., 0.), support, math.pi, limit)
    assert 2. < radius < 2.5
    goal = choose_observation_pose(grid, grid, (-2.3, -1.), (0., 0.), [], [], support,
                                   camera=camera)
    assert goal is not None and framing_fits(camera, goal[:2], goal[2], support)
    assert grid.value(*goal[:2]) == 0


def test_offcenter_cx_uses_whole_angular_window_without_false_empty_search():
    camera = calibrated(cx=100.)
    assert camera.limits()[1] < camera.yaw_tolerance
    support = [(0., -.05), (0., .05)]
    goal = choose_observation_pose(room(), room(), (-3., 0.), (0., 0.), [], [], support,
                                   camera=camera)
    assert goal is not None and framing_fits(camera, goal[:2], goal[2], support)


def test_larger_actual_xy_tolerance_expands_capture_duplicate_region():
    grid = room()
    camera = calibrated(xy_tolerance=.5, yaw_tolerance=.1)
    prior = (1.0, 0., math.pi)
    goal = choose_observation_pose(grid, grid, (3., 0.), (0., 0.), [], [],
                                   capture_viewpoints=[prior], camera=camera)
    assert goal and math.dist(goal[:2], prior[:2]) >= .6+.5+.05


@pytest.mark.parametrize('changes', [{'fx': 0.}, {'cx': math.nan}, {'width': 0},
                                   {'xy_tolerance': -1.}, {'yaw_tolerance': math.pi}])
def test_bad_framing_fails_closed(changes):
    camera = calibrated(**changes)
    assert choose_observation_pose(room(), room(), (-3., 0.), (0., 0.), [], [], camera=camera) is None


def test_current_robot_proximity_does_not_replace_source_view_independence():
    grid = room()
    robot, cached, target = (-2.85, .05), (-2.95, .05), (.05, .05)
    costs = replace(grid, data=[61]*len(grid.data))
    costs = set_cell(set_cell(costs, robot, 0), cached, 0)
    camera = calibrated()
    goal = choose_observation_pose(grid, costs, robot, target, [(2., .05)], [],
                                  camera=camera, cached_viewpoints=[cached])
    assert goal and goal[:2] == pytest.approx(cached)
    assert math.dist(robot, goal[:2]) < .5
    assert framing_fits(camera, goal[:2], goal[2], [target])
    # A turn near the original source is still not a translated viewpoint.
    assert choose_observation_pose(grid, costs, robot, target, [robot], [],
                                   camera=camera, cached_viewpoints=[cached]) is None


def narrow_endpoint_setup():
    grid = room()
    robot, target, endpoint = (-3.05, -1.05), (.05, .05), (-1.25, -1.05)
    # A traversable but non-goal cost corridor ends at a small safe patch.
    costs = replace(grid, data=[99]*len(grid.data))
    costs = paint(costs, lambda x, y: -3.1 < x < -1.2 and -1.1 < y < -1., 80)
    costs = set_cell(set_cell(costs, robot, 0), endpoint, 0)
    return grid, costs, robot, target, endpoint


def test_map_cell_fallback_finds_safe_corridor_endpoint_missed_by_all_rings(monkeypatch):
    grid, costs, robot, target, endpoint = narrow_endpoint_setup()
    camera = calibrated()
    with monkeypatch.context() as patch:
        patch.setattr(planner, '_fallback_points', lambda *args: [])
        assert choose_observation_pose(grid, costs, robot, target, [robot], [], camera=camera) is None
    goal = choose_observation_pose(grid, costs, robot, target, [robot], [], camera=camera)
    assert goal and goal[:2] == pytest.approx(endpoint)
    assert costs.value(*goal[:2]) == 0
    assert framing_fits(camera, goal[:2], goal[2], [target])
    assert _visible(grid, camera.camera_xy(goal[:2], goal[2]), target)


@pytest.mark.parametrize('change', ['cost', 'capture', 'attempt', 'unreachable', 'field_of_view'])
def test_cached_and_map_cell_positions_recheck_every_current_guard(change):
    grid, costs, robot, target, endpoint = narrow_endpoint_setup()
    camera, captures, attempted, support = calibrated(), [], [], [target]
    if change == 'cost':
        costs = set_cell(costs, endpoint, 61)
    elif change == 'capture':
        captures = [(*endpoint, _framed_heading(camera, endpoint, target, support))]
    elif change == 'attempt':
        attempted = [endpoint]
    elif change == 'unreachable':
        costs = paint(costs, lambda x, y: -2.4 < x < -2.2, 99)
    else:
        support = [(0., -3.), (0., 3.)]
    assert choose_observation_pose(grid, costs, robot, target, [robot], attempted, support,
                                   capture_viewpoints=captures, camera=camera,
                                   cached_viewpoints=[endpoint]) is None


def test_map_cell_fallback_is_bounded_route_ordered_and_prescreens_unsafe_cells():
    grid = room()
    robot, previous, attempted = (-3.05, .05), [(-2., .05)], [(-3., 1.)]
    costs = paint(grid, lambda x, y: x > 1. or y < -1., 99)
    costs = paint(costs, lambda x, y: -.5 < x < .5, 61)
    distances = _reachable(grid, costs, robot)
    points = _fallback_points(grid, costs, distances, robot, previous, attempted, .9, set())
    assert len(points) == 64 and len(set(points)) == 64
    ranking = []
    for point in points:
        col, row = grid.cell(*point)
        index = row*grid.width+col
        ranking.append((distances[index], math.dist(point, robot), index))
        assert distances[index] >= 0 and grid.value(*point) == 0 and 0 <= costs.value(*point) <= 60
        assert all(math.dist(point, p) >= .9 for p in previous)
        assert all(math.dist(point, p) >= .65 for p in attempted)
    assert ranking == sorted(ranking)


def test_bounded_candidate_phases_and_cached_duplicates_never_expand_work(monkeypatch):
    grid, origins = room(), []
    def invisible(_grid, start, _end):
        origins.append(start)
        return False
    monkeypatch.setattr(planner, '_visible', invisible)
    cached = [(i*.01-3., 0.) for i in range(100)]
    assert choose_observation_pose(grid, grid, (-3.5, 0.), (0., 0.), [], [],
                                   cached_viewpoints=cached+cached) is None
    # No calibration => 48 rings; with calibration the upper bound is 64.
    assert len(origins) <= 48+32+64 <= 160
    assert len(origins) == len(set(origins))
    assert all(p not in origins for p in cached[32:])



def test_unknown_visibility_is_explicit_opt_in_and_old_api_stays_clear_only():
    grid = room()
    grid = paint(grid, lambda x, y: .35 < math.hypot(x, y) < .6, 100)
    robot, target = (-3., 0.), (0., 0.)
    camera = calibrated()
    assert choose_observation_pose(grid, grid, robot, target, [robot], [], camera=camera) is None
    assert planner.choose_observation_candidate(grid, grid, robot, target, [robot], [], camera=camera) is None
    result = planner.choose_observation_candidate(grid, grid, robot, target, [robot], [],
                camera=camera, allow_uncertain_visibility=True)
    assert result and result['visibility_reason'] == planner.UNKNOWN_VISIBILITY
    assert result['map_sightline_groups'] == [False]
    pose = result['pose']
    col, row = grid.cell(*pose[:2])
    assert _reachable(grid, grid, robot)[row*grid.width+col] >= 0
    assert grid.value(*pose[:2]) == 0 and math.dist(pose[:2], robot) >= .9
    assert framing_fits(camera, pose[:2], pose[2], [target])


def test_clear_fallback_has_priority_over_every_nominal_unknown_candidate(monkeypatch):
    grid = room()
    fallback = (2.25, .75)
    monkeypatch.setattr(planner, '_fallback_points', lambda *args: [fallback])
    monkeypatch.setattr(planner, '_visible', lambda _grid, start, end: math.dist(start, fallback) < 1e-10)
    result = planner.choose_observation_candidate(grid, grid, (-3., 0.), (0., 0.), [], [],
                                                   allow_uncertain_visibility=True)
    assert result['pose'][:2] == fallback
    assert result['visibility_reason'] == planner.CLEAR_VISIBILITY
    assert result['map_sightline_groups'] == [True]
    assert choose_observation_pose(grid, grid, (-3., 0.), (0., 0.), [], []) == result['pose']


@pytest.mark.parametrize('change', ['none', 'cost', 'capture', 'attempt', 'unreachable', 'field_of_view', 'source'])
def test_unknown_visibility_preserves_all_navigation_and_independence_guards(monkeypatch, change):
    grid, costs, robot, target, endpoint = narrow_endpoint_setup()
    camera, captures, attempted, support, previous = calibrated(), [], [], [target], [robot]
    monkeypatch.setattr(planner, '_visible', lambda *args: False)
    if change == 'cost':
        costs = set_cell(costs, endpoint, 61)
    elif change == 'capture':
        captures = [(*endpoint, _framed_heading(camera, endpoint, target, support))]
    elif change == 'attempt':
        attempted = [endpoint]
    elif change == 'unreachable':
        costs = paint(costs, lambda x, y: -2.4 < x < -2.2, 99)
    elif change == 'field_of_view':
        support = [(0., -3.), (0., 3.)]
    elif change == 'source':
        previous.append(endpoint)
    result = planner.choose_observation_candidate(grid, costs, robot, target, previous, attempted, support,
                  capture_viewpoints=captures, camera=camera, cached_viewpoints=[endpoint],
                  allow_uncertain_visibility=True)
    if change == 'none':
        assert result and result['pose'][:2] == pytest.approx(endpoint)
        assert result['visibility_reason'] == planner.UNKNOWN_VISIBILITY
    else:
        assert result is None


def capture_pocket():
    """One safe camera pocket, so a rejected view cannot hide behind a new ring."""
    maps = room()
    costs = paint(maps, lambda x, y: not (-1.06 < x < -.94 and -.04 < y < .09), 100)
    camera = CameraFraming(640, 530., 319.5, 0., 0., 0., .25, .25)
    return maps, costs, camera


def test_dedicated_capture_allows_target_new_evidence_despite_unrelated_global_view():
    maps, costs, camera = capture_pocket()
    args = (maps, costs, (-.99, .03), (.03, .03), [(-3., .03)], [])
    kwargs = dict(camera=camera, capture_viewpoints=[(-.97, .03, 0.)])
    assert planner.choose_observation_candidate(*args, **kwargs) is None
    assert choose_observation_pose(*args, **kwargs) is None  # Legacy stays clear-only and deduped.
    result = planner.choose_observation_candidate(*args, **kwargs, dedicated_capture=True)
    assert result is not None and result['visibility_reason'] == planner.CLEAR_VISIBILITY
    assert maps.value(*result['pose'][:2]) == 0 and costs.value(*result['pose'][:2]) <= 60
    assert math.dist(result['pose'][:2], args[4][0]) >= .25+camera.xy_tolerance+.05
    assert framing_fits(camera, result['pose'][:2], result['pose'][2], [args[3]])


def test_dedicated_capture_uses_class_independence_plus_arrival_error_not_cache_radius():
    maps, costs, camera = capture_pocket()
    args = (maps, costs, (-.99, .03), (.03, .03), [(-1.6, .03)], [])
    assert planner.choose_observation_candidate(*args, camera=camera) is None
    result = planner.choose_observation_candidate(*args, camera=camera, dedicated_capture=True)
    assert result is not None
    distance = math.dist(result['pose'][:2], args[4][0])
    assert .25+camera.xy_tolerance+.05 <= distance < .6+camera.xy_tolerance+.05
    assert distance-camera.xy_tolerance > .25
    # The uncertainty buffer still forbids an effectively repeated source pose.
    assert planner.choose_observation_candidate(maps, costs, args[2], args[3], [(-1.3, .03)], [],
                                               camera=camera, dedicated_capture=True) is None


def test_dedicated_capture_retains_cost_attempt_and_visibility_rejections():
    maps, costs, camera = capture_pocket()
    args = (maps, costs, (-.99, .03), (.03, .03), [(-3., .03)])
    assert planner.choose_observation_candidate(*args, [(-.99, .03)], camera=camera,
                                               dedicated_capture=True) is None
    blocked_costs = replace(costs, data=[99 if c == 0 else c for c in costs.data])
    assert planner.choose_observation_candidate(maps, blocked_costs, args[2], args[3], args[4], [],
                                               camera=camera, dedicated_capture=True) is None
    occluded = paint(maps, lambda x, y: -.7 < x < -.4, 100)
    assert planner.choose_observation_candidate(occluded, costs, args[2], args[3], args[4], [],
                                               camera=camera, dedicated_capture=True) is None
    uncertain = planner.choose_observation_candidate(occluded, costs, args[2], args[3], args[4], [],
                              camera=camera, dedicated_capture=True, allow_uncertain_visibility=True)
    assert uncertain is not None and uncertain['visibility_reason'] == planner.UNKNOWN_VISIBILITY


def focus_scene():
    maps = room(width_m=6., origin=(-3., -3.))
    camera = CameraFraming(640, 530., 319.5, .077, 0., 0., .25, .25)
    groups = [[(-2., -.1), (-2., .1)], [(2., -.1), (2., .1)]]
    args = (maps, maps, (-2.5, -2.5), (0., 0.), [(-2.5, -2.5)], [])
    kwargs = dict(camera=camera, hypothesis_groups=groups, source_camera=(-3., -3., 0.),
                  dedicated_capture=True, allow_uncertain_visibility=True)
    return args, kwargs


def test_focused_hypothesis_can_fit_when_union_cannot_without_changing_input():
    import copy
    args, kwargs = focus_scene()
    original = copy.deepcopy(kwargs['hypothesis_groups'])
    assert planner.choose_observation_candidate(*args, **kwargs) is None
    candidate = planner.choose_observation_candidate(*args, **kwargs, focus_hypotheses=True)
    assert candidate and candidate['focus_group_index'] == 0
    assert candidate['hypothesis_group_count'] == 2
    assert candidate['framing_scope'] == 'one_measured_hypothesis_group'
    assert kwargs['hypothesis_groups'] == original
    assert framing_fits(kwargs['camera'], candidate['pose'][:2], candidate['pose'][2], original[0])
    assert not framing_fits(kwargs['camera'], candidate['pose'][:2], candidate['pose'][2], sum(original, []))
    assert args[0].value(*candidate['pose'][:2]) == 0
    assert math.dist(candidate['pose'][:2], args[4][0]) >= .55
    assert 'object_id' not in candidate and 'label' not in candidate
    second = planner.choose_observation_candidate(*args, **kwargs, focus_hypotheses=True,
                                                  attempted_focus_groups=[0],
                                                  cached_viewpoints=[candidate['pose'][:2]])
    assert second and second['focus_group_index'] == 1
    # A retained proposal is only a position: B gets a newly computed heading,
    # framing and sightline, never the old A group's visibility conclusion.
    assert framing_fits(kwargs['camera'], second['pose'][:2], second['pose'][2], original[1])


def test_one_class_surface_focus_allowed_but_default_union_still_requires_two():
    args, kwargs = focus_scene()
    kwargs['hypothesis_groups'] = kwargs['hypothesis_groups'][:1]
    assert planner.choose_observation_candidate(*args, **kwargs) is None
    candidate = planner.choose_observation_candidate(*args, **kwargs, focus_hypotheses=True)
    assert candidate['focus_group_index'] == 0 and candidate['hypothesis_group_count'] == 1


@pytest.mark.parametrize('mutation', ['empty', 'too_many', 'nan', 'index', 'bool_index', 'no_camera'])
def test_focused_inputs_fail_closed_without_silently_dropping_group(mutation):
    args, kwargs = focus_scene()
    if mutation == 'empty': kwargs['hypothesis_groups'] = []
    if mutation == 'too_many': kwargs['hypothesis_groups'] *= 5
    if mutation == 'nan': kwargs['hypothesis_groups'][1][0] = (math.nan, 0.)
    if mutation == 'index': kwargs['attempted_focus_groups'] = [2]
    if mutation == 'bool_index': kwargs['attempted_focus_groups'] = [True]
    if mutation == 'no_camera': kwargs['camera'] = None
    assert planner.choose_observation_candidate(*args, **kwargs, focus_hypotheses=True) is None


@pytest.mark.parametrize('block', ['endpoint', 'unreachable', 'source'])
def test_focus_mode_cannot_bypass_navigation_or_independence(block):
    args, kwargs = focus_scene()
    args = list(args)
    if block == 'endpoint': args[1] = replace(args[0], data=[61]*len(args[0].data))
    if block == 'unreachable': args[1] = replace(args[0], data=[100]*len(args[0].data))
    if block == 'source':
        args[4] = [args[0].world_center(x, y) for y in range(args[0].height)
                   for x in range(args[0].width)]
    assert planner.choose_observation_candidate(*args, **kwargs, focus_hypotheses=True) is None


def test_focus_tiers_clear_then_source_range_and_bounded_group_calls(monkeypatch):
    # Isolate only the investigation ordering; real geometry tested above.
    calls = []
    outcomes = {0: planner.UNKNOWN_VISIBILITY, 1: planner.CLEAR_VISIBILITY}
    def candidate(*args, **kwargs):
        index = round(args[3][0])
        calls.append(index)
        assert not kwargs.get('focus_hypotheses', False)
        if index not in outcomes: return None
        return dict(pose=(0., 0., 0.), visibility_reason=outcomes[index], map_sightline_groups=[False])
    monkeypatch.setattr(planner, 'choose_observation_candidate', candidate)
    groups = [[(float(i), 0.)] for i in range(8)]
    camera = CameraFraming(640, 530., 319.5, 0., 0., 0., .25, .25)
    def choose(tried):
        return planner._focused_candidate(room(), room(), (-3., 0.), [], [], groups,
                                          (-1., 0., 0.), tried, [], camera, [], True, True)
    assert choose(set())['focus_group_index'] == 1  # farther clear beats near unknown
    calls.clear()
    assert choose({1})['focus_group_index'] == 0  # untried tier before tried tier
    assert 1 not in calls
    calls.clear(); outcomes.clear()
    assert choose(set()) is None and calls == list(range(8))
