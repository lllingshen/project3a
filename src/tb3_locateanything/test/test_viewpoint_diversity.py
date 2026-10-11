"""Pending-only viewpoint ranking; actual planner and node callback regressions."""
import copy
import math

import pytest

from tb3_locateanything import observation_planner as p
from test_observation_planner import room
from test_observation_node import node, depth_hint


def choose(previous=(), attempted=(), **kwargs):
    grid = room(width_m=10., origin=(-5., -5.))
    return p.choose_observation_candidate(grid, grid, (3., 0.), (0., 0.),
                                          previous, attempted, **kwargs)


def bearing(point):
    return math.atan2(point[1], point[0])


def test_pending_prefers_opposite_direction_over_short_same_direction():
    old = choose([(2., 0.)])
    new = choose([(2., 0.)], prefer_viewpoint_diversity=True)
    assert old['pose'][0] > 0 and new['pose'][0] < 0
    assert abs(bearing(new['pose'])) == pytest.approx(math.pi)
    assert old['visibility_reason'] == new['visibility_reason'] == p.CLEAR_VISIBILITY
    assert math.dist(new['pose'][:2], (2., 0.)) >= .9


def test_attempt_and_every_source_constrain_minimum_bearing():
    result = choose([(2., 0.)], [(0., 2.)], prefer_viewpoint_diversity=True)
    assert bearing(result['pose']) == pytest.approx(-3 * math.pi / 4)
    with_second_source = choose([(2., 0.), (0., 2.)], prefer_viewpoint_diversity=True)
    assert with_second_source == result


def test_wraparound_uses_small_difference_not_full_revolution():
    target = (0., 0.)
    assert p._minimum_viewpoint_angle((-1., 1e-8), target, [(-1., -1e-8)]) == pytest.approx(2e-8)
    assert p._minimum_viewpoint_angle((1., 0.), target, [(-1., 0.)]) == pytest.approx(math.pi)


@pytest.mark.parametrize('previous', [[], [(0., 0.)], [(0., 0.), (0., 0.)]])
def test_missing_or_target_coincident_history_preserves_legacy(previous):
    assert choose(previous) == choose(previous, prefer_viewpoint_diversity=True)
    assert p._minimum_viewpoint_angle((0., 0.), (0., 0.), [(1., 1.)]) == 0.


def test_duplicate_sources_and_coincident_extra_reference_do_not_change_rank():
    previous = [(2., 0.)]
    assert choose(previous, prefer_viewpoint_diversity=True) == choose(
        previous * 3 + [(0., 0.)], prefer_viewpoint_diversity=True)


def test_bearing_ranking_is_rigid_transform_invariant():
    refs, target, point = [(2., 1.), (-3., 2.)], (.3, -.7), (-1., -2.)
    yaw = 2.93
    def transform(v):
        return (5 + math.cos(yaw) * v[0] - math.sin(yaw) * v[1],
                -7 + math.sin(yaw) * v[0] + math.cos(yaw) * v[1])
    assert p._minimum_viewpoint_angle(point, target, refs) == pytest.approx(
        p._minimum_viewpoint_angle(transform(point), transform(target), [transform(q) for q in refs]))


def test_clear_lower_angle_beats_uncertain_higher_angle(monkeypatch):
    monkeypatch.setattr(p, '_visible', lambda grid, start, end: start[0] > 0)
    selected = choose([(2., 0.)], prefer_viewpoint_diversity=True, allow_uncertain_visibility=True)
    assert selected['pose'][0] > 0
    assert selected['visibility_reason'] == p.CLEAR_VISIBILITY


def test_full_measured_height_beats_partial_higher_angle():
    class Height:
        def predict(self, point, yaw, support):
            return {'arrival_tolerance_fits': point[0] > .5}
    selected = choose([(2., 0.)], prefer_viewpoint_diversity=True,
                      vertical_framing=Height(), vertical_support_xyz=[(0., 0., 1.)])
    assert selected['pose'][0] > .5
    assert selected['vertical_framing']['arrival_tolerance_fits']


def test_clear_nominal_keeps_phase_priority_over_more_diverse_fallback(monkeypatch):
    fallback = (-2.3 * math.cos(.1), -2.3 * math.sin(.1))
    calls = []
    monkeypatch.setattr(p, '_fallback_points', lambda *args: calls.append(True) or [fallback])
    source = [(2 * math.cos(.1), 2 * math.sin(.1))]
    selected = choose(source, prefer_viewpoint_diversity=True)
    assert math.dist(selected['pose'][:2], fallback) > .1
    assert not calls


def test_full_fallback_beats_partial_nominal_even_with_lower_angle(monkeypatch):
    fallback = (.7, 2.3)
    monkeypatch.setattr(p, '_fallback_points', lambda *args: [fallback])
    class Height:
        def predict(self, point, yaw, support):
            return {'arrival_tolerance_fits': point == fallback}
    selected = choose([(2., 0.)], prefer_viewpoint_diversity=True,
                      vertical_framing=Height(), vertical_support_xyz=[(0., 0., 1.)])
    assert selected['pose'][:2] == fallback


def test_uncertain_candidates_still_compete_across_both_phases(monkeypatch):
    fallback = (-2.3 * math.cos(.1), -2.3 * math.sin(.1))
    monkeypatch.setattr(p, '_fallback_points', lambda *args: [fallback])
    monkeypatch.setattr(p, '_visible', lambda *args: False)
    previous = [(2 * math.cos(.1), 2 * math.sin(.1))]
    old = choose(previous, allow_uncertain_visibility=True)
    new = choose(previous, allow_uncertain_visibility=True, prefer_viewpoint_diversity=True)
    assert math.dist(old['pose'][:2], fallback) > .1
    assert new['pose'][:2] == fallback
    assert new['visibility_reason'] == p.UNKNOWN_VISIBILITY
    assert choose(previous, allow_uncertain_visibility=False, prefer_viewpoint_diversity=True) is None


@pytest.mark.parametrize('focus', [False, True])
def test_hypothesis_union_and_focused_recursion_do_not_use_pending_rank(focus):
    grid = room(width_m=12., origin=(-6., -6.))
    camera = p.CameraFraming(640, 530., 319.5, .077, 0., 0., .25, .1)
    groups = [[(-.7, -.1), (-.7, .1)], [(.7, -.1), (.7, .1)]]
    kwargs = dict(camera=camera, hypothesis_groups=groups, source_camera=(-3., 0., 0.),
                  focus_hypotheses=focus, dedicated_capture=True,
                  allow_uncertain_visibility=True)
    if focus:
        kwargs['forced_focus_index'] = 0
    args = (grid, grid, (-3., 0.), (0., 0.), [(-3., 0.)], [])
    before = copy.deepcopy(groups)
    old = p.choose_observation_candidate(*args, **kwargs)
    new = p.choose_observation_candidate(*args, **kwargs, prefer_viewpoint_diversity=True)
    assert old is not None and old == new and groups == before


@pytest.mark.parametrize('is_hint', [False, True])
def test_actual_node_tick_enables_diversity_only_for_pending(node, is_hint):
    if is_hint:
        node.memory['pending_landmarks'] = []
        node.memory['observation_hints'] = [depth_hint()]
    before = copy.deepcopy(node.memory)
    node._tick()
    assert node.planner_calls
    assert all(call[1]['prefer_viewpoint_diversity'] is (not is_hint) for call in node.planner_calls)
    assert node.memory == before and node.state.total_attempts == 0
    assert len(node.request_pub.records) == 1
    assert set(node.request_pub.records[0]) == {
        'request_id', 'track_id', 'frame_id', 'stamp_ns', 'x', 'y', 'yaw'}
