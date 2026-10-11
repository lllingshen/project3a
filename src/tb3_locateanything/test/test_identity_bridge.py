"""Pure saved-evidence identity correction; no ROS, model or ground truth."""
import copy
import json
import math

import numpy as np
import pytest
from tb3_locateanything.visual_memory import Memory
from tb3_locateanything.depth_core import _single_dense_component

S = 10**9
LEFT = [10., 10., 30., 30.]
RIGHT = [40., 10., 60., 30.]
WHOLE = [10., 10., 60., 30.]


def points(x):
    return [[x+dx, y, 1.] for dx in (-.03, 0., .03) for y in (-.03, 0., .03)]


def expose(memory, stamp, position, box, xyz, *, single=True, epoch=0, source_view=None):
    view = source_view or memory.add_view({'odom_frame': 'odom', 'depth_geometry': {}},
                                         stamp*S, position, 0., 0., dedicated_capture=True)
    view.payload['metadata'] = {'image_stamp_ns': view.stamp_ns, 'depth_stamp_ns': view.stamp_ns}
    group = {'odom_xyz': copy.deepcopy(xyz), 'pixels': [[int(box[0]+1), int(box[1]+1)]]*len(xyz), 'count': len(xyz)}
    connectivity = {'box': list(box), 'contact_m': .12, 'selected_count': len(xyz),
                    'roi_stride': 1, 'single_component': single}
    geometry = {'localization_source': 'aligned_depth', 'surface_group': group,
                'identity_connectivity': connectivity, 'support_points': [p[:2] for p in xyz]}
    view.payload['depth_geometry'][tuple(box)] = (None, geometry)
    proof = dict(epoch=epoch, view_id=view.view_id, stamp_ns=view.stamp_ns, depth_stamp_ns=view.stamp_ns,
                 odom_frame='odom', box=list(box), surface_group=group, connectivity=connectivity)
    return view, proof


def observe(memory, view, proof, label='chair'):
    xyz = proof['surface_group']['odom_xyz']
    return memory.remember(view, label, proof['box'], xyz[0][0], xyz[0][1],
                           support_points=[p[:2] for p in xyz], identity_depth=proof)


def fragments(*, legacy=False):
    m = Memory(max_views=32, confirm_classes=not legacy, class_labels=['chair', 'storage_rack'])
    v, p = expose(m, 1, 0., LEFT, points(0.))
    a = observe(m, v, p)
    _, q = expose(m, 1, 0., RIGHT, points(.6), source_view=v)
    b = observe(m, v, q, 'storage_rack')
    return m, v, a, b


def whole(m, stamp, position, label='chair', single=True, epoch=0):
    xyz = [[x, y, 1.] for x in np.linspace(-.03, .63, 16) for y in (-.03, 0., .03)]
    v, p = expose(m, stamp, position, WHOLE, xyz, single=single, epoch=epoch)
    return v, p


def test_two_wholes_resolve_soft_negative_without_inventing_votes():
    m, source, a, b = fragments()
    first, p = whole(m, 2, 1.)
    with pytest.raises(ValueError, match='separated by an earlier view'):
        observe(m, first, p)
    assert len(first.bridge_observations) == 1 and first.observations == []
    assert a.label_votes == {'chair': 1} and b.label_votes == {'storage_rack': 1}
    second, q = whole(m, 3, 2.)
    joined = observe(m, second, q)
    assert len(m.landmarks) == 1 and joined.confirmed and joined.object_id == 'chair_0'
    assert joined.label_votes == {'chair': 3, 'storage_rack': 1}
    assert len(joined.identity_resolutions) == 1
    assert len(joined.view_ids) == 3
    assert source.observations[0]['label_boxes']['storage_rack'] == RIGHT
    assert source.observations[0]['box'] in (LEFT, RIGHT)  # No invented union.
    assert not any(v.bridge_observations for v in m.views)
    json.dumps(m.serialize(), allow_nan=False)
    observe(m, second, q)
    assert joined.label_votes == {'chair': 3, 'storage_rack': 1}


@pytest.mark.parametrize('position', [0., .249, 1., 1.249])
def test_source_or_whole_positions_are_not_independent(position):
    m, source, a, b = fragments()
    first, p = whole(m, 2, 1.)
    with pytest.raises(ValueError): observe(m, first, p)
    other, q = whole(m, 3, position)
    with pytest.raises(ValueError): observe(m, other, q)
    assert len(m.landmarks) == 2 and not a.confirmed and not b.confirmed


def test_same_view_and_new_prompt_do_not_create_second_vote():
    m, source, a, b = fragments()
    first, p = whole(m, 2, 1.)
    for _ in range(3):
        with pytest.raises(ValueError): observe(m, first, p)
    assert len(first.bridge_observations) == 1 and not a.confirmed


@pytest.mark.parametrize('bad', ['disconnected', 'epoch', 'frame', 'stamp', 'box', 'pixels', 'nan', 'count', 'cache'])
def test_bad_current_provenance_keeps_old_rejection(bad):
    m, source, a, b = fragments()
    first, p = whole(m, 2, 1.)
    with pytest.raises(ValueError): observe(m, first, p)
    second, q = whole(m, 3, 2.)
    q = copy.deepcopy(q)
    if bad == 'disconnected':
        q['connectivity']['single_component'] = False
        second.payload['depth_geometry'][tuple(WHOLE)][1]['identity_connectivity']['single_component'] = False
    elif bad == 'epoch': q['epoch'] += 1
    elif bad == 'frame': q['odom_frame'] = 'map'
    elif bad == 'stamp': q['depth_stamp_ns'] -= 5_000_001
    elif bad == 'box': q['box'][0] += 1
    elif bad == 'pixels': q['surface_group']['pixels'][0][0] = 999
    elif bad == 'nan': q['surface_group']['odom_xyz'][0][0] = math.nan
    elif bad == 'count': q['surface_group']['count'] = 0
    elif bad == 'cache': second.payload['depth_geometry'].clear()
    with pytest.raises(ValueError): observe(m, second, q)
    assert len(m.landmarks) == 2 and not a.confirmed


def test_different_whole_labels_retained_but_cannot_be_cherry_picked():
    m, source, a, b = fragments()
    for stamp, label in [(2, 'chair'), (3, 'storage_rack'), (4, 'chair')]:
        v, p = whole(m, stamp, float(stamp))
        with pytest.raises(ValueError): observe(m, v, p, label)
    assert {o['label'] for v in m.views for o in v.bridge_observations} == {'chair', 'storage_rack'}
    assert len(m.landmarks) == 2


def test_two_independent_negatives_become_hard_and_eviction_never_softens():
    m, source, a, b = fragments()
    second, p = expose(m, 2, .5, LEFT, points(0.))
    observe(m, second, p, 'storage_rack')
    _, q = expose(m, 2, .5, RIGHT, points(.6), source_view=second)
    observe(m, second, q, 'chair')
    assert not a.confirmed and not b.confirmed
    assert next(iter(m._identity_relations.values()))['hard']
    m.views.remove(source)
    m._prune(2*S)
    for stamp in (3, 4):
        v, p = whole(m, stamp, stamp)
        with pytest.raises(ValueError): observe(m, v, p)
    assert len(m.landmarks) == 2


@pytest.mark.parametrize('flag', ['confirmed', 'conflicted'])
def test_confirmed_or_sticky_conflicted_tracks_never_override(flag):
    m, source, a, b = fragments()
    setattr(a, flag, True)
    for stamp in (2, 3):
        v, p = whole(m, stamp, stamp)
        with pytest.raises(ValueError): observe(m, v, p)
    assert len(m.landmarks) == 2


def test_third_track_cannot_be_ignored():
    m, source, a, b = fragments()
    v, p = expose(m, 2, .5, LEFT, points(.3))
    third = observe(m, v, p)
    assert third not in (a, b)
    for stamp in (3, 4):
        v, p = whole(m, stamp, stamp)
        with pytest.raises(ValueError): observe(m, v, p)
    assert len(m.landmarks) == 3


def test_first_whole_eviction_cannot_supply_a_historical_vote():
    m, source, a, b = fragments()
    first, p = whole(m, 2, 1.)
    with pytest.raises(ValueError): observe(m, first, p)
    m.views.remove(first)
    second, q = whole(m, 3, 2.)
    with pytest.raises(ValueError): observe(m, second, q)
    assert len(m.landmarks) == 2


def test_rollback_clears_relations_and_pending_bridges():
    m, source, a, b = fragments()
    first, p = whole(m, 2, 1.)
    with pytest.raises(ValueError): observe(m, first, p)
    m.views_for_query(1*S)
    assert not m.views and not m.landmarks and not m._identity_relations


def test_legacy_confirmation_mode_never_gains_override():
    m, source, a, b = fragments(legacy=True)
    for stamp in (2, 3):
        v, p = whole(m, stamp, stamp)
        with pytest.raises(ValueError): observe(m, v, p)
    assert len(m.landmarks) == 2


def test_dense_connectivity_uses_all_measured_points_not_sparse64_or_equal_depth():
    dense = np.array([[x, 0., 1.] for x in np.linspace(0, 2, 100)])
    assert _single_dense_component(dense)
    assert not _single_dense_component(dense[::10])
    assert not _single_dense_component(np.concatenate([dense, dense+[0., .5, 0.]]))
    # Contact is an explicit remaining identity risk, not semantic proof.
    assert _single_dense_component(np.concatenate([dense, dense+[0., .1, 0.]]))


def test_dead_identity_relations_are_pruned_not_only_hidden():
    m, source, a, b = fragments()
    assert m._identity_relations
    m.views.clear()
    m._prune(2*S)
    assert not m.landmarks and not m._identity_relations


def test_exact_saved_depth_stamp_not_just_nearby_is_required():
    m, source, a, b = fragments()
    first, p = whole(m, 2, 1.)
    with pytest.raises(ValueError): observe(m, first, p)
    second, q = whole(m, 3, 2.)
    q['depth_stamp_ns'] -= 1
    with pytest.raises(ValueError): observe(m, second, q)
    assert not second.bridge_observations and len(m.landmarks) == 2


def test_commit_exception_restores_objects_votes_relations_and_id_counters(monkeypatch):
    m, source, a, b = fragments()
    first, p = whole(m, 2, 1.)
    with pytest.raises(ValueError): observe(m, first, p)
    second, q = whole(m, 3, 2.)
    old_relations = copy.deepcopy(m._identity_relations)
    original = m._record_observation
    calls = []
    def failing(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('commit failure after promotion')
        return result
    monkeypatch.setattr(m, '_record_observation', failing)
    with pytest.raises(ValueError, match='commit failure'):
        observe(m, second, q)
    assert m.landmarks == {a.object_id: a, b.object_id: b}
    assert a.label_votes == {'chair': 1} and b.label_votes == {'storage_rack': 1}
    assert not a.confirmed and not b.confirmed and not m._next_object
    assert m._identity_relations == old_relations
    assert first.observations == [] and second.observations == []


def test_successful_correction_audit_budget_does_not_reset_on_prune():
    m, source, a, b = fragments()
    m._identity_resolution_count = m.max_views
    for stamp in (2, 3):
        v, p = whole(m, stamp, stamp)
        with pytest.raises(ValueError): observe(m, v, p)
    m.views.clear(); m._prune(4*S)
    assert m._identity_resolution_count == 32 and not m.landmarks
    m.clear()
    assert m._identity_resolution_count == 0


def test_one_close_point_does_not_establish_majority_correspondence():
    m = Memory(max_views=32, confirm_classes=True)
    v, p = expose(m, 1, 0., LEFT, [[i*.05, 0., 1.] for i in range(18)])
    a = observe(m, v, p)
    _, q = expose(m, 1, 0., RIGHT, points(2.), source_view=v)
    b = observe(m, v, q, 'storage_rack')
    # Deliberately isolate correspondence from the separately tested upstream
    # dense proof: a single near point passes min-distance but not the median.
    for stamp in (2, 3):
        w, proof = expose(m, stamp, stamp, WHOLE, [[0., 0., 1.]]+points(2.))
        with pytest.raises(ValueError): observe(m, w, proof)
    assert len(m.landmarks) == 2 and not a.confirmed and not b.confirmed


@pytest.mark.parametrize('name', ['two_chairs', 'front_back_classes'])
def test_real_native_distinct_object_controls_do_not_support_a_bridge(name):
    from pathlib import Path
    fixture = json.loads(Path(__file__).with_name('identity_bridge_controls.json').read_text())
    case = next(c for c in fixture['cases'] if c['name'] == name)
    first, second = case['measurements']
    import statistics
    xyz1, xyz2 = (o['geometry']['surface_group']['odom_xyz'] for o in (first, second))
    assert statistics.median(min(math.dist(p, q) for q in xyz2) for p in xyz1) > .18
    assert statistics.median(min(math.dist(p, q) for q in xyz1) for p in xyz2) > .18
    whole = case['union_abuse_control']
    if name == 'two_chairs':
        assert whole['rejected'].startswith('multiple separated depth surfaces')
    else:
        # The original outer rack box selects its own measured surface;
        # it cannot represent the distinct extinguisher behind it.
        union = whole['geometry']['surface_group']['odom_xyz']
        assert statistics.median(min(math.dist(p, q) for q in union) for p in xyz1) > .18
