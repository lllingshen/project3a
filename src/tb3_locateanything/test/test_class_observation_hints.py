"""Class uncertainty creates observation tasks, never object identities or votes."""
import copy
import json
from types import SimpleNamespace

import pytest

from tb3_locateanything.observation_hints import HintMemory

SECOND = 1_000_000_000
BOX = [10., 10., 100., 90.]


def surface(x=2., y=0.):
    return {'odom_xyz': [[x, y, 1.], [x+.02, y, 1.], [x, y+.02, 1.]],
            'pixels': [[20, 20], [21, 20], [20, 21]], 'count': 300}


def add(hints, labels=('chair', 'table'), groups=None, view='source', stamp=10*SECOND,
        landmarks=(), pose=(0., 0., 0.)):
    return hints.add_class(labels, view, stamp, pose, (.1, 0., 0.), BOX,
                           groups if groups is not None else [surface()], landmarks)


def track(oid='candidate_0', label='table', x=2., confirmed=False,
          conflicted=False, views=('independent',)):
    return SimpleNamespace(object_id=oid, label=label, confirmed=confirmed,
                           conflicted=conflicted, view_ids=set(views), label_votes={},
                           support_points=[(x, 0.), (x+.02, 0.), (x, .02)])


def test_one_real_surface_preserves_provenance_without_chosen_label_or_object():
    hints = HintMemory(epoch=4)
    groups = [surface()]
    oid = add(hints, (' TABLE ', 'Chair', 'table'), groups=groups)
    item = hints.serialize(10*SECOND)[0]
    assert oid == 'class_hint_4_0'
    assert item['hint_kind'] == 'class_ambiguity'
    assert item['label_candidates'] == ['chair', 'table'] and item['label'] == ''
    assert item['surface_groups'] == groups and len(item['surface_groups']) == 1
    assert item['status'] == 'unresolved' and item['resolved_by'] == []
    assert not ({'object_id', 'position', 'confirmed', 'label_votes'} & item.keys())
    groups[0]['odom_xyz'][0][0] = 999.
    item['surface_groups'][0]['pixels'][0][0] = 999
    assert hints.serialize(10*SECOND)[0]['surface_groups'] == [surface()]
    json.dumps(hints.serialize(10*SECOND), allow_nan=False)


def test_normalized_label_set_and_unordered_geometry_keep_original_task_identity():
    hints = HintMemory()
    first = add(hints, ('Storage rack', 'forklift'), groups=[surface(2.), surface(4.)])
    second = add(hints, ('FORKLIFT', 'storage_rack', 'storage rack'),
                 groups=[surface(4.03), surface(2.03)], view='later',
                 stamp=12*SECOND, pose=(1., 0., 1.))
    assert first == second
    item = hints.serialize(12*SECOND)[0]
    assert item['label_candidates'] == ['forklift', 'storage_rack']
    assert item['surface_groups'] == [surface(2.), surface(4.)]
    assert item['source_view_id'] == 'source' and item['source_pose'] == [0., 0., 0.]
    assert item['stamp_ns'] == 10*SECOND and item['last_seen_ns'] == 12*SECOND


def test_different_label_set_or_physical_surface_does_not_absorb_existing_task():
    hints = HintMemory()
    ids = [add(hints), add(hints, ('chair', 'desk')), add(hints, groups=[surface(5.)])]
    assert len(set(ids)) == 3


def test_group_matching_still_requires_one_to_one_not_reused_partner():
    hints = HintMemory()
    a = add(hints, groups=[surface(0.), surface(1.), surface(1.4)])
    b = add(hints, groups=[surface(-.15), surface(.15), surface(1.2)])
    assert a != b


@pytest.mark.parametrize('labels', [[], ['table'], ['table', 'TABLE'], 'chair table',
                                    ['chair', ''], ['chair', None], ['chair', 3],
                                    [str(i) for i in range(9)]])
def test_invalid_or_not_actually_ambiguous_labels_rejected(labels):
    hints = HintMemory()
    with pytest.raises((TypeError, ValueError)):
        add(hints, labels)
    assert hints.hints == []


@pytest.mark.parametrize('count', [1, 8])
def test_class_supports_one_through_eight_real_groups(count):
    hints = HintMemory()
    groups = [surface(2.+i) for i in range(count)]
    add(hints, groups=groups)
    assert hints.hints[0]['surface_groups'] == groups


@pytest.mark.parametrize('change', ['zero_groups', 'nine_groups', 'bad_xyz', 'bad_dimensions',
                                    'too_many_points', 'missing_pixel', 'nonfinite_pixel',
                                    'small_count', 'missing_count', 'nan_count', 'bool_count',
                                    'fractional_count', 'negative_pixel', 'fractional_pixel'])
def test_invalid_measured_provenance_does_not_create_class_task(change):
    groups = [surface()]
    if change == 'zero_groups': groups = []
    elif change == 'nine_groups': groups *= 9
    elif change == 'bad_xyz': groups[0]['odom_xyz'][0][0] = float('nan')
    elif change == 'bad_dimensions': groups[0]['odom_xyz'][0] = [2., 0.]
    elif change == 'too_many_points':
        groups[0]['odom_xyz'] *= 22; groups[0]['pixels'] *= 22
    elif change == 'missing_pixel': groups[0]['pixels'].pop()
    elif change == 'nonfinite_pixel': groups[0]['pixels'][0][0] = float('nan')
    elif change == 'small_count': groups[0]['count'] = 1
    elif change == 'missing_count': del groups[0]['count']
    elif change == 'nan_count': groups[0]['count'] = float('nan')
    elif change == 'bool_count': groups[0]['count'] = True
    elif change == 'fractional_count': groups[0]['count'] = 300.5
    elif change == 'negative_pixel': groups[0]['pixels'][0][0] = -1
    elif change == 'fractional_pixel': groups[0]['pixels'][0][0] = 20.5
    hints = HintMemory()
    with pytest.raises((ValueError, TypeError, KeyError)):
        add(hints, groups=groups)
    assert hints.hints == []


def test_depth_and_class_share_capacity_but_never_conflate_task_kinds():
    hints = HintMemory(max_hints=2)
    groups = [surface(2.), surface(4.)]
    depth = hints.add('table', 'source', 10*SECOND, (0., 0., 0.), (.1, 0., 0.), BOX, groups, [])
    cls = add(hints, groups=groups)
    assert depth == 'depth_hint_0_0' and cls == 'class_hint_0_1'
    assert add(hints, groups=[surface(8.)]) is None
    assert hints.add('chair', 'source', 10*SECOND, (0., 0., 0.), (.1, 0., 0.), BOX, groups, []) is None
    assert hints.rejected_capacity == 2 and len(hints.hints) == 2
    assert 'hint_kind' not in hints.hints[0] and 'label_candidates' not in hints.hints[0]
    with pytest.raises(ValueError):
        hints.add('table', 'source', 10*SECOND, (0., 0., 0.), (.1, 0., 0.), BOX, [surface()], [])


@pytest.mark.parametrize('retirement', ['resolved', 'expired'])
def test_retired_class_task_never_reopens_or_resets_external_attempt_identity(retirement):
    hints = HintMemory(max_hints=1, max_age_sec=5.)
    original = add(hints)
    if retirement == 'resolved':
        hints.reconcile([track()], 'independent', (1., 0., 0.), 11*SECOND)
    else:
        hints.serialize(16*SECOND)
    assert hints.hints[0]['status'] == retirement
    again = add(hints, view='new_view', stamp=18*SECOND, pose=(3., 0., 0.))
    assert original == again and hints.hints[0]['status'] == retirement
    assert hints.hints[0]['stamp_ns'] == 10*SECOND
    assert add(hints, groups=[surface(9.)]) is None


@pytest.mark.parametrize('label', ['table', 'CHAIR'])
def test_any_candidate_label_from_independent_ordinary_track_resolves_without_voting(label):
    hints = HintMemory(); add(hints)
    obj = track(label=label)
    before = copy.deepcopy(vars(obj))
    hints.reconcile([obj], 'independent', (.25, 0., 0.), 11*SECOND)
    item = hints.hints[0]
    assert item['resolved_by'] == [obj.object_id] and item['status'] == 'resolved'
    assert vars(obj) == before and not obj.confirmed and obj.label_votes == {}


@pytest.mark.parametrize('fault', ['wrong_label', 'conflict', 'missing_view', 'same_view',
                                  'pure_rotation', 'too_close', 'old_stamp', 'two_tracks',
                                  'two_groups', 'known_id', 'known_surface'])
def test_resolution_preserves_physical_uniqueness_independence_and_background_guards(fault):
    hints = HintMemory()
    groups = [surface(2.), surface(4.)]
    background = track('table_0', x=4., confirmed=True)
    add(hints, groups=groups, landmarks=[background])
    obj = track(); objects = [obj]; view = 'independent'; pose = (1., 0., 0.); stamp = 11*SECOND
    if fault == 'wrong_label': obj.label = 'person'
    elif fault == 'conflict': obj.conflicted = True
    elif fault == 'missing_view': obj.view_ids = {'other'}
    elif fault == 'same_view': view = 'source'; obj.view_ids = {view}
    elif fault == 'pure_rotation': pose = (0., 0., 3.)
    elif fault == 'too_close': pose = (.249, 0., 0.)
    elif fault == 'old_stamp': stamp = 10*SECOND
    elif fault == 'two_tracks': objects.append(track('candidate_1', 'chair', 2.05))
    elif fault == 'two_groups': obj.support_points += track(x=4.).support_points
    elif fault == 'known_id': obj.object_id = 'table_0'
    elif fault == 'known_surface': obj.support_points = track(x=4.).support_points
    hints.reconcile(objects, view, pose, stamp)
    assert hints.hints[0]['status'] == 'unresolved'


def test_expiry_reconcile_and_clock_epoch_clear_apply_equally_to_class_tasks():
    hints = HintMemory(max_age_sec=5., epoch=7)
    original = add(hints)
    hints.reconcile([track()], 'independent', (1., 0., 0.), 16*SECOND)
    assert hints.hints[0]['status'] == 'expired'
    hints.clear(8)
    assert hints.hints == [] and hints.known_surfaces == {} and hints.rejected_capacity == 0
    assert add(hints, stamp=SECOND) == 'class_hint_8_0' != original
