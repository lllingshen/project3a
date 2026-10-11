"""A part can prove an existing instance, never create one or relocate it."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tb3_locateanything.gallery import (Gallery, GalleryEntry, select_gallery_box,
                                      select_gallery_part, validate_gallery_part, validate_gallery_part_groups)
from tb3_locateanything.depth_core import AmbiguousDepthError
from tb3_locateanything.geometry import Vector3
from tb3_locateanything.visual_memory import Landmark
from test_visual_memory_node import (node, stored_view, active_query, gallery_answer,
                                    module, SECOND)


DATA = json.loads((Path(__file__).with_name('gallery_part_measurement.json')).read_text())


def scenario(node):
    entries = []
    for i, saved in enumerate(DATA['entries']):
        view = stored_view(node, 90+i, (i, 0, 0))
        view.payload['depth_m'] = np.ones((480, 640))
        view.observations = copy.deepcopy(saved['view_observations'])
        entries.append(GalleryEntry(saved['object_id'], view, saved['source_box'][:],
                                   saved['tile_box'][:], saved['content_box'][:],
                                   saved['object_box'][:], saved['crop_box'][:]))
    for saved in DATA['landmarks']:
        obj = Landmark(saved['object_id'], saved['label'], 0., 0., 3, 91*SECOND,
                       support_points=copy.deepcopy(saved['support_points']))
        node.memory.landmarks[obj.object_id] = obj
    gallery = Gallery(np.zeros(DATA['gallery_shape'], np.uint8), '', entries)
    return gallery, entries[1]


def response(node, answer=None):
    return {'request_id': node.pending['id'], 'answer': answer or DATA['raw_answer'],
            'prompt': 'the object I can sit on', 'inference_seconds': .2}


def test_saved_part_fails_unchanged_iou_but_proves_unique_original_depth(node):
    gallery, target = scenario(node)
    before = copy.deepcopy(node.memory.serialize())
    with pytest.raises(ValueError):
        select_gallery_box(gallery, DATA['pixel_box'])
    entry, part = select_gallery_part(gallery, DATA['pixel_box'])
    assert entry is target
    assert part == pytest.approx(DATA['entries'][1]['model_box_in_source_pixels'])
    proof = validate_gallery_part(node.memory, entry, DATA['part_geometry'], node.now_ns)
    checks = proof['track_support_checks']
    assert checks[entry.object_id]['compatible_part_points'] == 64
    assert checks[entry.object_id]['max_distance_m'] == pytest.approx(.056268354192024715, abs=1e-6)
    assert checks[gallery.entries[0].object_id]['compatible_part_points'] == 0
    assert node.memory.serialize() == before


def test_part_mapping_is_tile_order_independent(node):
    gallery, entry = scenario(node)
    original = select_gallery_part(gallery, DATA['pixel_box'])
    gallery.entries.reverse()
    result = select_gallery_part(gallery, DATA['pixel_box'])
    assert result[0] is original[0] and result[1] == original[1]


@pytest.mark.parametrize('box', [
    [-1, 100, 20, 150], [0, 0, 688, 352], [350, 170, 430, 200],
    [420, 20, 450, 30], [383, 90, 407, 120], [600, 150, 640, 200],
    [409, 160, float('nan'), 180], [409, 160, 409, 180],
])
def test_part_cross_tile_caption_padding_background_invalid_rejected(node, box):
    gallery, _ = scenario(node)
    with pytest.raises(ValueError):
        select_gallery_part(gallery, box)


def test_duplicate_tile_associations_never_choose_by_order(node):
    gallery, target = scenario(node)
    gallery.entries.append(copy.copy(target))
    with pytest.raises(ValueError, match='unique'):
        select_gallery_part(gallery, DATA['pixel_box'])


@pytest.mark.parametrize('status', ['confirmed', 'pending', 'conflicted', 'unrelated_class'])
def test_any_other_track_support_can_make_part_ambiguous(node, status):
    gallery, target = scenario(node)
    other = node.memory.landmarks[gallery.entries[0].object_id]
    # Even one common point is enough to block uniqueness; no label filter.
    other.support_points[0] = DATA['part_geometry']['support_points'][0][:]
    if status == 'pending':
        other.confirmed = False
    elif status == 'conflicted':
        other.conflicted = True
    elif status == 'unrelated_class':
        other.label = 'table'
    with pytest.raises(ValueError, match='another retained'):
        validate_gallery_part(node.memory, target, DATA['part_geometry'], node.now_ns)


@pytest.mark.parametrize('kind', ['absent', 'pending', 'conflicted', 'evicted', 'expired', 'future'])
def test_part_target_must_still_be_current_confirmed_retained_evidence(node, kind):
    gallery, target = scenario(node)
    obj = node.memory.landmarks[target.object_id]
    if kind == 'absent':
        del node.memory.landmarks[target.object_id]
    elif kind == 'pending':
        obj.confirmed = False
    elif kind == 'conflicted':
        obj.conflicted = True
    elif kind == 'evicted':
        node.memory.views = [gallery.entries[0].view]
    elif kind == 'expired':
        node.now_ns = target.view.stamp_ns + node.memory.max_age_ns + 1
    else:
        node.now_ns = target.view.stamp_ns - 1
    with pytest.raises(ValueError):
        validate_gallery_part(node.memory, target, DATA['part_geometry'], node.now_ns)


@pytest.mark.parametrize('where', ['part', 'target', 'other'])
@pytest.mark.parametrize('bad', [[], [[0., 1.]]*3, [[0., 1.], [1., 2.], [float('nan'), 3.]]])
def test_missing_or_degenerate_support_is_not_unique_evidence(node, where, bad):
    gallery, target = scenario(node)
    geometry = copy.deepcopy(DATA['part_geometry'])
    if where == 'part':
        geometry['support_points'] = bad
    else:
        oid = target.object_id if where == 'target' else gallery.entries[0].object_id
        node.memory.landmarks[oid].support_points = bad
    with pytest.raises(ValueError, match='distinct finite'):
        validate_gallery_part(node.memory, target, geometry, node.now_ns)


def test_single_target_contact_does_not_validate_a_background_or_mixed_part(node):
    gallery, target = scenario(node)
    geometry = copy.deepcopy(DATA['part_geometry'])
    geometry['support_points'][-1] = [30., 30.]
    with pytest.raises(ValueError, match='complete retained target support'):
        validate_gallery_part(node.memory, target, geometry, node.now_ns)


def test_wrong_tile_cannot_borrow_another_instances_depth_identity(node):
    gallery, _ = scenario(node)
    with pytest.raises(ValueError):
        validate_gallery_part(node.memory, gallery.entries[0], DATA['part_geometry'], node.now_ns)


def test_node_part_validates_original_pixels_but_selects_original_whole_target(node):
    gallery, entry = scenario(node)
    active_query(node)
    before = copy.deepcopy(node.memory.serialize())
    calls, selections = [], []
    def localize(view, box):
        calls.append((view, list(box)))
        return (Vector3(1, 2, 3), copy.deepcopy(DATA['part_geometry'])) if len(calls) == 1 else (Vector3(9, 8, 0), {'localization_source': 'aligned_depth', 'whole': True})
    node._localize = localize
    node._select = lambda *args: selections.append(args)
    node._gallery_result({'gallery': gallery, 'epoch': node.epoch}, response(node))
    assert len(calls) == 2 and all(view is entry.view for view, box in calls)
    assert calls[0][1] == pytest.approx(DATA['entries'][1]['model_box_in_source_pixels'])
    assert calls[1][1] == entry.source_box
    view, box, point, geometry, obj, _, kind = selections[0]
    assert view is entry.view and box == entry.source_box and point == Vector3(9, 8, 0)
    assert kind == 'gallery_description' and geometry['whole']
    proof = geometry['gallery_part_evidence']
    assert proof['source_object_box_xyxy'] == entry.source_box
    assert proof['depth_geometry']['support_points'] == DATA['part_geometry']['support_points']
    assert proof['image_stamp_ns'] == entry.view.stamp_ns
    after = node.memory.serialize()
    for item in before['landmarks'] + after['landmarks']:
        item['descriptions'] = []
    assert before == after  # identity, votes, surfaces, observations, positions unchanged
    assert not node.inference_calls


@pytest.mark.parametrize('failure', ['no_depth', 'ambiguous_depth', 'background', 'stale_tf', 'expired', 'epoch', 'deadline'])
def test_node_part_failure_never_selects_or_adds_votes(node, failure):
    gallery, entry = scenario(node)
    active_query(node)
    before = copy.deepcopy(node.memory.serialize())
    selections = []
    node._select = lambda *args: selections.append(args)
    def localize(view, box):
        if failure == 'ambiguous_depth':
            raise AmbiguousDepthError(box, [], 2)
        if failure in ('background', 'stale_tf'):
            raise ValueError(failure)
        if failure == 'deadline':
            node.pending['wall_start'] -= 100
        return Vector3(1, 2, 3), copy.deepcopy(DATA['part_geometry'])
    node._localize = localize
    if failure == 'no_depth':
        del entry.view.payload['depth_m']
    if failure == 'expired':
        entry.view.stamp_ns = node.now_ns-node.memory.max_age_ns-1
    job = {'gallery': gallery, 'epoch': node.epoch-(failure == 'epoch')}
    with pytest.raises(ValueError):
        node._gallery_result(job, response(node))
    assert not selections and not node.inference_calls
    if failure == 'expired':
        before['views'] = node.memory.serialize()['views']
    assert node.memory.serialize() == before


def test_full_gallery_path_does_not_require_or_execute_part_depth(node):
    gallery, entry = scenario(node)
    active_query(node)
    node.memory.landmarks[gallery.entries[0].object_id].support_points = []
    del entry.view.payload['depth_m']
    calls, selections = [], []
    node._localize = lambda view, box: calls.append((view, box)) or (Vector3(9, 8, 0), {})
    node._select = lambda *args: selections.append(args)
    node._gallery_result({'gallery': gallery, 'epoch': node.epoch},
                         response(node, gallery_answer(gallery, entry.object_box)))
    assert calls == [(entry.view, entry.source_box)]
    assert 'gallery_part_evidence' not in selections[0][3]


def part_groups(number=2):
    """Synthetic disconnected groups whose XY are actual fixture target supports."""
    points = np.asarray(DATA['part_geometry']['support_points'])
    u, v = np.ceil(DATA['entries'][1]['model_box_in_source_pixels'][:2]).astype(int)
    return [{'odom_xyz': np.column_stack((part, np.full(len(part), .2))).tolist(),
             'pixels': [[int(u+i), int(v+j)] for j in range(len(part))], 'count': len(part)+100}
            for i, part in enumerate(np.array_split(points, number))]


def group_error(box, number=2):
    return AmbiguousDepthError(box, part_groups(number), number)


@pytest.mark.parametrize('number', [2, 3, 8])
def test_all_measured_groups_independently_prove_same_instance_without_memory_mutation(node, number):
    gallery, target = scenario(node)
    _, part_box = select_gallery_part(gallery, DATA['pixel_box'])
    error = group_error(part_box, number)
    original_groups = copy.deepcopy(error.groups)
    before = copy.deepcopy(node.memory.serialize())
    proof = validate_gallery_part_groups(node.memory, target, error, part_box, node.now_ns)
    assert proof['group_count'] == number
    assert len(proof['group_support_checks']) == number
    assert proof['support_gate_m'] == .18
    for group, checks in zip(error.groups, proof['group_support_checks']):
        assert checks[target.object_id]['compatible_part_points'] == len(group['odom_xyz'])
        assert checks[gallery.entries[0].object_id]['compatible_part_points'] == 0
    assert error.groups == original_groups and node.memory.serialize() == before


@pytest.mark.parametrize('failure', ['ordinary_error', 'unusable', 'wrong_box', 'count_mismatch',
    'one_group', 'overflow', 'count_bool', 'count_float', 'too_few_samples', 'too_many_samples',
    'nonfinite_xyz', 'wrong_xyz_shape', 'nonfinite_pixels', 'pixel_mismatch', 'negative_pixel',
    'fractional_pixel', 'duplicate_pixel', 'count_less_than_samples', 'degenerate_xy', 'missing_field'])
def test_multi_group_provenance_cannot_drop_or_repair_bad_evidence(node, failure):
    gallery, target = scenario(node)
    _, part_box = select_gallery_part(gallery, DATA['pixel_box'])
    error = group_error(part_box)
    groups = list(error.groups)
    if failure == 'ordinary_error':
        error = ValueError('unrelated localization failure')
    elif failure == 'unusable':
        error.hint_usable = False
    elif failure == 'wrong_box':
        error.box = tuple([part_box[0]+1e-9, *part_box[1:]])
    elif failure == 'count_mismatch':
        error.group_count = 3
    elif failure == 'one_group':
        error.groups, error.group_count = tuple(groups[:1]), 1
    elif failure == 'overflow':
        error.groups, error.group_count = tuple(groups*5), 10
    elif failure == 'count_bool':
        error.group_count = True
    elif failure == 'count_float':
        groups[0]['count'] = 100.0
    elif failure in ('too_few_samples', 'too_many_samples'):
        count = 2 if failure == 'too_few_samples' else 65
        groups[0]['odom_xyz'] = [groups[0]['odom_xyz'][0][:] for _ in range(count)]
        groups[0]['pixels'] = [[0, i] for i in range(count)]
    elif failure == 'nonfinite_xyz':
        groups[0]['odom_xyz'][0][2] = float('nan')
    elif failure == 'wrong_xyz_shape':
        groups[0]['odom_xyz'][0].pop()
    elif failure == 'nonfinite_pixels':
        groups[0]['pixels'][0][0] = float('inf')
    elif failure == 'pixel_mismatch':
        groups[0]['pixels'].pop()
    elif failure == 'negative_pixel':
        groups[0]['pixels'][0][0] = -1
    elif failure == 'fractional_pixel':
        groups[0]['pixels'][0][0] = .5
    elif failure == 'duplicate_pixel':
        groups[0]['pixels'][1] = groups[0]['pixels'][0][:]
    elif failure == 'count_less_than_samples':
        groups[0]['count'] = len(groups[0]['pixels'])-1
    elif failure == 'degenerate_xy':
        for xyz in groups[0]['odom_xyz']:
            xyz[:2] = [0., 0.]
    elif failure == 'missing_field':
        del groups[0]['pixels']
    before = copy.deepcopy(node.memory.serialize())
    with pytest.raises(ValueError):
        validate_gallery_part_groups(node.memory, target, error, part_box, node.now_ns)
    assert node.memory.serialize() == before


@pytest.mark.parametrize('other_status', ['confirmed', 'pending', 'conflicted', 'different_class', 'missing_support'])
def test_any_group_competitor_including_non_gallery_track_rejects_whole_proof(node, other_status):
    gallery, target = scenario(node)
    _, part_box = select_gallery_part(gallery, DATA['pixel_box'])
    error = group_error(part_box)
    other = node.memory.landmarks[gallery.entries[0].object_id]
    # Only the LAST group's last point competes; an early successful group is insufficient.
    other.support_points[0] = error.groups[-1]['odom_xyz'][-1][:2]
    if other_status == 'pending': other.confirmed = False
    if other_status == 'conflicted': other.conflicted = True
    if other_status == 'different_class': other.label = 'forklift'
    if other_status == 'missing_support': other.support_points = []
    with pytest.raises(ValueError):
        validate_gallery_part_groups(node.memory, target, error, part_box, node.now_ns)


def test_one_unexplained_group_point_rejects_all_groups(node):
    gallery, target = scenario(node)
    _, part_box = select_gallery_part(gallery, DATA['pixel_box'])
    error = group_error(part_box)
    error.groups[-1]['odom_xyz'][-1][:2] = [40., 40.]
    with pytest.raises(ValueError, match='complete retained target support'):
        validate_gallery_part_groups(node.memory, target, error, part_box, node.now_ns)


@pytest.mark.parametrize('failure', ['expired', 'evicted', 'pending', 'conflicted'])
def test_multi_group_identity_needs_current_retained_confirmed_target(node, failure):
    gallery, target = scenario(node)
    _, part_box = select_gallery_part(gallery, DATA['pixel_box'])
    if failure == 'expired': target.view.stamp_ns = node.now_ns-node.memory.max_age_ns-1
    if failure == 'evicted': node.memory.views = [gallery.entries[0].view]
    if failure == 'pending': node.memory.landmarks[target.object_id].confirmed = False
    if failure == 'conflicted': node.memory.landmarks[target.object_id].conflicted = True
    with pytest.raises(ValueError):
        validate_gallery_part_groups(node.memory, target, group_error(part_box), part_box, node.now_ns)


def test_node_multi_group_proof_still_navigates_original_cached_whole_box(node):
    gallery, entry = scenario(node)
    active_query(node)
    before = copy.deepcopy(node.memory.serialize())
    calls, selections, errors = [], [], []
    original = (Vector3(9., 8., .7), {'localization_source': 'aligned_depth', 'whole': True})
    entry.view.payload['depth_geometry'] = {tuple(entry.source_box): original}
    def localize(view, box):
        calls.append((view, list(box)))
        if tuple(box) not in view.payload['depth_geometry']:
            error = group_error(box); errors.append(error); raise error
        return view.payload['depth_geometry'][tuple(box)]
    node._localize = localize
    node.memory.remember = lambda *_a, **_k: pytest.fail('Identity proof must not add votes')
    node._select = lambda *args: selections.append(args)
    node._gallery_result({'gallery': gallery, 'epoch': node.epoch}, response(node))
    assert len(calls) == 2 and calls[-1] == (entry.view, entry.source_box)
    view, box, point, meta, obj, _, kind = selections[0]
    assert view is entry.view and box == entry.source_box and point is original[0]
    assert obj.object_id == entry.object_id and kind == 'gallery_description'
    proof = meta['gallery_part_evidence']
    assert proof['group_count'] == 2 and len(proof['group_support_checks']) == 2
    assert proof['source_object_box_xyxy'] == entry.source_box
    assert proof['depth_geometry']['surface_groups'] == list(errors[0].groups)
    assert 'support_points' not in proof['depth_geometry']  # No merged/selected target geometry.
    after = node.memory.serialize()
    for item in before['landmarks']+after['landmarks']: item['descriptions'] = []
    assert before == after and not node.inference_calls


@pytest.mark.parametrize('failure', ['no_depth', 'epoch', 'ordinary', 'unusable', 'foreign_group',
                                     'expired', 'deadline', 'motion', 'whole_tf'])
def test_node_multi_group_never_masks_original_hard_failure(node, failure):
    gallery, entry = scenario(node)
    active_query(node)
    before = copy.deepcopy(node.memory.serialize())
    calls, selections = [], []
    def localize(view, box):
        calls.append(list(box))
        if box == entry.source_box:
            raise module.tf2_ros.TransformException('current projection unavailable')
        if failure == 'ordinary': raise ValueError('No usable depth')
        error = group_error(box)
        if failure == 'unusable': error.hint_usable = False
        if failure == 'foreign_group': error.groups[-1]['odom_xyz'][-1][:2] = [30., 30.]
        if failure == 'deadline': node.pending['wall_start'] -= 100
        if failure == 'motion': node.pending['robot_moved'] = True
        raise error
    node._localize = localize
    node._select = lambda *args: selections.append(args)
    if failure == 'no_depth': del entry.view.payload['depth_m']
    if failure == 'expired':
        entry.view.stamp_ns = node.now_ns-node.memory.max_age_ns-1
        before['views'] = node.memory.serialize()['views']
    with pytest.raises((ValueError, module.tf2_ros.TransformException)):
        node._gallery_result({'gallery': gallery, 'epoch': node.epoch-(failure == 'epoch')}, response(node))
    assert not selections and not node.inference_calls and node.memory.serialize() == before
    assert (len(calls) == 2) == (failure == 'whole_tf')


@pytest.mark.parametrize('axis,side', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_multi_group_pixels_must_stay_inside_exact_part_box(node, axis, side):
    gallery, target = scenario(node)
    _, part_box = select_gallery_part(gallery, DATA['pixel_box'])
    error = group_error(part_box)
    # The identity supports are all valid; only the actual pixel provenance is
    # outside the non-rounded query box. Never quietly filter that sample.
    bound = part_box[axis+2*side]
    error.groups[-1]['pixels'][-1][axis] = int(np.floor(bound)-1 if side == 0 else np.ceil(bound)+1)
    with pytest.raises(ValueError, match='measurement provenance'):
        validate_gallery_part_groups(node.memory, target, error, part_box, node.now_ns)
