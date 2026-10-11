"""Real catalog/depth callbacks: uncertain classes schedule views, not objects."""
import copy

import numpy as np
import pytest

from test_visual_memory_node import node, depth_hint_catalog, measured_depth_ambiguity
from tb3_locateanything import locateanything_node as module
from tb3_locateanything.depth_core import localize_depth_box
from tb3_locateanything.geometry import CameraCalibration, Quaternion, Transform3D, Vector3, transform_point


CAL = CameraCalibration(120, 100, 100., 100., 60., 50., 'optical')
TF = Transform3D(Vector3(1., 2., 1.5), Quaternion(-.5, .5, -.5, .5))


@pytest.mark.parametrize('slanted', [False, True])
def test_valid_surface_retains_only_actual_frozen_depth_pixels(slanted):
    depth = np.full((100, 120), 6., dtype=np.float32)
    depth[18:82, 38:82] = np.linspace(2.5, 3.5, 44)[None, :] if slanted else 3.
    _, meta = localize_depth_box((35., 15., 85., 85.), CAL, depth, TF)
    group = meta['surface_group']
    assert group['count'] >= len(group['odom_xyz']) and 3 <= len(group['pixels']) <= 64
    assert len(group['odom_xyz']) == len(group['pixels'])
    for xyz, (u, v) in zip(group['odom_xyz'], group['pixels']):
        assert type(u) is type(v) is int
        z = float(depth[v, u])
        measured = transform_point(TF, Vector3((u-CAL.cx)*z/CAL.fx, (v-CAL.cy)*z/CAL.fy, z))
        assert xyz == pytest.approx([measured.x, measured.y, measured.z], abs=1e-12)
        assert 38 <= u < 82 and 18 <= v < 82
    # Sampling preserves all three measured extent axes, not just a center.
    all_xyz = []
    for v in range(18, 82):
        for u in range(38, 82):
            z = float(depth[v, u])
            p = transform_point(TF, Vector3((u-CAL.cx)*z/CAL.fx, (v-CAL.cy)*z/CAL.fy, z))
            all_xyz.append([p.x, p.y, p.z])
    assert np.min(group['odom_xyz'], axis=0) == pytest.approx(np.min(all_xyz, axis=0))
    assert np.max(group['odom_xyz'], axis=0) == pytest.approx(np.max(all_xyz, axis=0))


def geometry(x=2., depth=2.):
    return {'localization_source': 'aligned_depth', 'depth_m': depth,
            'support_points': [[x, 1.], [x+.1, 1.], [x, 1.1]],
            'surface_group': {'odom_xyz': [[x, 1., .2], [x+.1, 1., .4], [x, 1.1, .6]],
                              'pixels': [[140, 140], [160, 160], [180, 180]], 'count': 100}}


def ambiguous_catalog(node, *, same_box=True):
    box = '<100><100><800><900>' if same_box else '<200><200><600><700>'
    return depth_hint_catalog(node,
        '<ref>table</ref><box><100><100><800><900></box>'
        '<ref>chair</ref><box>'+box+'</box>', categories=('table', 'chair'))


@pytest.mark.parametrize('same_box', [False, True])
def test_same_measured_surface_class_conflict_is_one_nonsemantic_task(node, same_box):
    view, job, response = ambiguous_catalog(node, same_box=same_box)
    measured = geometry()
    node._localize = lambda *_: (Vector3(2., 1., .2), copy.deepcopy(measured))
    node._catalog_result(job, response)
    hint, = node.observation_hints.serialize(node.now_ns)
    assert hint['hint_kind'] == 'class_ambiguity'
    assert hint['label_candidates'] == ['chair', 'table'] and hint['label'] == ''
    assert hint['surface_groups'] == [measured['surface_group']]
    assert hint['source_view_id'] == view.view_id
    assert hint['source_pose'] == [4., 5., .3] and hint['source_camera'][:2] == [4.1, 5.2]
    assert not node.memory.landmarks and not view.observations and not node.result_pub.messages
    event = node.events[-1]
    assert event['class_observation_hints'] == [hint['hint_id']]
    assert not event['depth_observation_hints'] and not event['object_ids']
    assert len(event['ambiguous_detections']) == 2
    # Repeated callbacks cannot recreate the task and reset its external budget.
    node._catalog_result(job, response)
    assert len(node.observation_hints.serialize(node.now_ns)) == 1
    assert node.events[-1]['class_observation_hints'] == [hint['hint_id']]


@pytest.mark.parametrize('failure', ['ordinary_depth', 'missing_xyz',
                                    'missing_camera', 'wrong_frame', 'malformed_count'])
def test_incomplete_class_provenance_never_creates_hint_or_memory(node, failure):
    view, job, response = ambiguous_catalog(node, same_box=False)
    if failure == 'missing_camera':
        del view.payload['camera_to_odom']
    if failure == 'wrong_frame':
        view.payload['odom_frame'] = 'map'
    def locate(source, box):
        if box[0] != 64.:
            if failure == 'ordinary_depth':
                raise ValueError('no bounded depth')
        meta = geometry()
        if failure == 'missing_xyz':
            del meta['surface_group']
        if failure == 'malformed_count':
            meta['surface_group']['count'] = 1
        return Vector3(2., 1., .2), meta
    node._localize = locate
    node._catalog_result(job, response)
    assert not node.memory.landmarks and not view.observations
    assert not node.events[-1]['class_observation_hints']
    assert not node.events[-1]['depth_observation_hints']
    if getattr(node, 'observation_hints', None):
        assert node.observation_hints.serialize(node.now_ns) == []


def test_distinct_measured_groups_all_retained_without_selecting_one(node):
    view, job, response = ambiguous_catalog(node, same_box=False)
    # Below strict foreground/background separation: preserve both possible
    # surfaces as a task; neither is a selectable coordinate or a class vote.
    first, second = geometry(), geometry(2.2, 2.1)
    node._localize = lambda source, box: (Vector3(2., 1., .2),
                                         first if box[0] == 64. else second)
    node._catalog_result(job, response)
    hint, = node.observation_hints.serialize(node.now_ns)
    assert hint['surface_groups'] == [first['surface_group'], second['surface_group']]
    assert not node.memory.landmarks and not view.observations


def test_actual_depth_separation_stays_ordinary_memory_not_class_hint(node):
    view, job, response = ambiguous_catalog(node, same_box=False)
    node._localize = lambda source, box: (Vector3(2., 1., .2),
        geometry() if box[0] == 64. else geometry(4., 4.))
    node._catalog_result(job, response)
    assert node.events[-1]['depth_resolved_indices'] == [0, 1]
    assert not node.events[-1]['class_observation_hints']
    assert {obj.label for obj in node.memory.landmarks.values()} == {'table', 'chair'}
    assert all(not obj.confirmed for obj in node.memory.landmarks.values())


def test_clipped_member_cannot_become_class_hint(node):
    view, job, response = depth_hint_catalog(node,
        '<ref>table</ref><box><0><100><800><900></box>'
        '<ref>chair</ref><box><200><200><600><700></box>', categories=('table', 'chair'))
    calls = []
    node._localize = lambda source, box: calls.append(box) or (Vector3(2., 1., .2), geometry())
    node._catalog_result(job, response)
    assert len(calls) == 1 and calls[0][0] > 0
    assert not node.events[-1]['class_observation_hints'] and not node.memory.landmarks


@pytest.mark.parametrize('ambiguous_member', [0, 1])
def test_mixed_class_depth_task_keeps_every_label_and_group_without_admission(node, ambiguous_member):
    view, job, response = ambiguous_catalog(node, same_box=False)
    valid = geometry(6., 4.)
    captured = []
    def locate(source, box):
        index = 0 if box[0] == 64. else 1
        if index == ambiguous_member:
            error = measured_depth_ambiguity(box)
            captured.extend(copy.deepcopy(error.groups))
            raise error
        return Vector3(6., 1., .2), copy.deepcopy(valid)
    node._localize = locate
    node.memory.remember = lambda *_args, **_kwargs: pytest.fail('Conflict cannot add a Memory vote')
    node._catalog_result(job, response)
    hint, = node.observation_hints.serialize(node.now_ns)
    expected = captured + [valid['surface_group']] if ambiguous_member == 0 else [valid['surface_group']] + captured
    assert hint['surface_groups'] == expected
    assert hint['label_candidates'] == ['chair', 'table'] and hint['label'] == ''
    assert hint['hint_kind'] == 'class_ambiguity' and hint['source_view_id'] == view.view_id
    assert not node.memory.landmarks and not view.observations and not node.result_pub.messages
    assert len(node.events[-1]['ambiguous_detections']) == 2
    assert node.events[-1]['depth_resolved_indices'] == []
    assert node.events[-1]['depth_observation_hints'] == []
    assert node.events[-1]['class_observation_hints'] == [hint['hint_id']]
    # Repeated mixed evidence keeps the old task identity; no budget reset.
    node._catalog_result(job, response)
    again, = node.observation_hints.serialize(node.now_ns)
    assert again['hint_id'] == hint['hint_id'] and again['surface_groups'] == expected


def test_all_ambiguous_members_keep_union_and_only_exact_group_duplicates_collapse(node):
    view, job, response = ambiguous_catalog(node, same_box=False)
    common = geometry(2.)['surface_group']
    near = geometry(2.01)['surface_group']  # Close is not the same measurement.
    last = geometry(5.)['surface_group']
    def locate(source, box):
        groups = [common, near] if box[0] == 64. else [common, last]
        raise module.AmbiguousDepthError(box, copy.deepcopy(groups), 2)
    node._localize = locate
    node._catalog_result(job, response)
    hint, = node.observation_hints.serialize(node.now_ns)
    assert hint['surface_groups'] == [common, near, last]
    assert hint['label_candidates'] == ['chair', 'table']
    assert not node.memory.landmarks and not view.observations and not node.result_pub.messages


@pytest.mark.parametrize('failure', ['ordinary', 'tf', 'unusable', 'wrong_box', 'wrong_count',
                                     'nonfinite', 'bad_pixels', 'overflow', 'clipped'])
def test_mixed_hint_needs_every_members_complete_usable_provenance(node, failure):
    view, job, response = ambiguous_catalog(node, same_box=False)
    if failure == 'clipped':
        response['answer'] = response['answer'].replace('<100><100><800><900>', '<0><100><800><900>')
    def locate(source, box):
        if box[0] < 100.:
            if failure == 'ordinary':
                raise ValueError('No bounded depth surface')
            if failure == 'tf':
                raise module.tf2_ros.TransformException('No same-frame transform')
            if failure == 'unusable':
                raise measured_depth_ambiguity(box, usable=False)
            error = measured_depth_ambiguity(box)
            if failure == 'wrong_box':
                error.box = tuple([box[0]+1., *box[1:]])
            if failure == 'wrong_count':
                error.group_count = 3
            if failure == 'nonfinite':
                error.groups[0]['odom_xyz'][0][2] = float('nan')
            if failure == 'bad_pixels':
                error.groups[0]['pixels'].pop()
            if failure == 'overflow':
                error = module.AmbiguousDepthError(box,
                    [geometry(float(i))['surface_group'] for i in range(8)], 8)
            raise error
        return Vector3(20., 1., .2), geometry(20., 5.)
    node._localize = locate
    node._catalog_result(job, response)
    assert not node.events[-1]['class_observation_hints']
    assert not node.events[-1]['depth_observation_hints']
    assert len(node.events[-1]['ambiguous_detections']) == 2
    assert not node.memory.landmarks and not view.observations and not node.result_pub.messages
    if getattr(node, 'observation_hints', None):
        assert node.observation_hints.serialize(node.now_ns) == []


def test_eight_identical_groups_are_not_counted_twice_or_truncated(node):
    view, job, response = ambiguous_catalog(node, same_box=False)
    groups = [geometry(float(i))['surface_group'] for i in range(8)]
    def locate(source, box):
        raise module.AmbiguousDepthError(box, copy.deepcopy(groups), 8)
    node._localize = locate
    node._catalog_result(job, response)
    hint, = node.observation_hints.serialize(node.now_ns)
    assert hint['surface_groups'] == groups and len(hint['surface_groups']) == 8
    assert not node.memory.landmarks and not view.observations


def test_unrelated_valid_detection_still_remembered_beside_mixed_class_hint(node):
    view, job, response = depth_hint_catalog(node,
        '<ref>table</ref><box><100><100><800><900></box>'
        '<ref>chair</ref><box><200><200><600><700></box>'
        '<ref>person</ref><box><850><100><950><900></box>',
        categories=('table', 'chair', 'person'))
    def locate(source, box):
        if box[0] == 64.:
            raise measured_depth_ambiguity(box)
        return Vector3(6., 1., .2), geometry(6., 4.)
    node._localize = locate
    node._catalog_result(job, response)
    obj, = node.memory.landmarks.values()
    assert obj.label == 'person' and not obj.confirmed
    assert len(view.observations) == 1
    hint, = node.observation_hints.serialize(node.now_ns)
    assert hint['label_candidates'] == ['chair', 'table']
    assert node.events[-1]['object_ids'] == [obj.object_id]
    assert not node.result_pub.messages
