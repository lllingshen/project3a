"""Accepted-cache publication and opt-in soft height, without ROS/model execution."""
import copy
import json
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest
from rclpy.time import Time

from tb3_locateanything import pending_framing as proof
from tb3_locateanything import observation_node as observer
from tb3_locateanything import observation_planner as planner
from tb3_locateanything.visual_memory import Memory
from tb3_locateanything.vertical_framing import VerticalFraming
from test_visual_memory_node import node as perception_node, stored_view, BOX
from test_observation_node import node as observer_node, transform
from test_vertical_framing import model, VerticalPreferenceOracle

SECOND = 1_000_000_000
GROUP = {'odom_xyz': [[2., 3., .2], [2.1, 3.1, .9], [2.2, 3.2, .4]],
         'pixels': [[101, 101], [120, 130], [249, 399]], 'count': 12}


def accepted(node, seconds=10., pose=(0., 0., 0.)):
    view = stored_view(node, seconds=seconds, pose=pose)
    obj = node.memory.remember(view, 'chair', BOX, 2., 3.,
        support_points=[[2., 3.], [2.1, 3.1]], class_evidence_complete=False)
    payload = view.payload
    payload['depth_m'] = np.ones((480, 640))
    payload['metadata'].update(localization_source='aligned_depth', depth_stamp_ns=view.stamp_ns)
    payload['depth_geometry'] = {tuple(BOX): (NS(x=2., y=3., z=.2),
        {'localization_source': 'aligned_depth', 'surface_group': copy.deepcopy(GROUP)})}
    return view, obj


def prepare_node(node):
    node.memory = Memory(confirm_classes=True)
    view, obj = accepted(node)
    return view, obj, node.memory.serialize()['pending_landmarks'][0]


def test_actual_publish_retains_original_memory_and_only_one_exact_accepted_cache(perception_node):
    node = perception_node
    view, obj, _ = prepare_node(node)
    # Plausible but unaccepted/query-part cache must never be exported.
    view.payload['depth_geometry'][(10., 10., 20., 20.)] = (
        NS(x=99., y=99., z=99.), {'localization_source': 'aligned_depth',
                                  'surface_group': {**GROUP, 'odom_xyz': [[99., 99., 99.]]*3}})
    before = copy.deepcopy(node.memory.serialize())
    node._localize = lambda *_a, **_k: pytest.fail('publication cannot need new localization')
    node._refresh_positions = lambda: {'valid': True}  # Isolate already-existing map projection.
    node._publish_memory()
    published = json.loads(node.memory_pub.messages[-1].data)
    source = published['pending_landmarks'][0]['framing_source']
    assert source['surface_group'] == GROUP and source['box'] == BOX
    assert source['epoch'] == published['epoch'] == 0
    assert node.memory.serialize() == before and not node.inference_calls
    assert obj.observations == 1 and not obj.confirmed
    assert published['pending_landmarks'][0]['support_points'] == before['pending_landmarks'][0]['support_points']


@pytest.mark.parametrize('bad', ['missing_cache', 'unaccepted_box', 'image_stamp', 'depth_stamp', 'frame',
    'not_depth', 'cache_source', 'nan', 'missing_z', 'pixels', 'pixel_outside', 'fractional_pixel', 'count', 'oversize'])
def test_bad_source_wholly_disables_optional_height_without_mutating_memory(perception_node, bad):
    node = perception_node
    view, obj, track = prepare_node(node)
    payload = view.payload
    if bad == 'missing_cache': payload['depth_geometry'].clear()
    elif bad == 'unaccepted_box':
        payload['depth_geometry'][(101., 100., 250., 400.)] = payload['depth_geometry'].pop(tuple(BOX))
    elif bad == 'image_stamp': payload['metadata']['image_stamp_ns'] += 1
    elif bad == 'depth_stamp': payload['metadata']['depth_stamp_ns'] += 5_000_001
    elif bad == 'frame': payload['image'].header.frame_id = 'wrong_optical'
    elif bad == 'not_depth': payload['metadata']['localization_source'] = 'laser'
    else:
        geometry = payload['depth_geometry'][tuple(BOX)][1]
        group = geometry['surface_group']
        if bad == 'cache_source': geometry['localization_source'] = 'laser'
        elif bad == 'nan': group['odom_xyz'][2][2] = float('nan')
        elif bad == 'missing_z': group['odom_xyz'][2].pop()
        elif bad == 'pixels': group['pixels'].pop()
        elif bad == 'pixel_outside': group['pixels'][2][0] = 251
        elif bad == 'fractional_pixel': group['pixels'][2][0] = 201.5
        elif bad == 'count': group['count'] = 2
        elif bad == 'oversize': group['odom_xyz'] *= 22; group['pixels'] *= 22; group['count'] = 100
    before = copy.deepcopy(node.memory.serialize())
    assert proof.retained_framing_source(node.memory, track, 0, node.now_ns) is None
    assert node.memory.serialize() == before and obj.observations == 1


def test_original_five_ms_sync_and_top_bottom_clip_are_not_new_rejections(perception_node):
    node = perception_node
    view, _, track = prepare_node(node)
    old = tuple(BOX); box = [100., 0., 250., 480.]
    view.observations[0]['box'] = box
    view.payload['depth_geometry'][tuple(box)] = view.payload['depth_geometry'].pop(old)
    view.payload['metadata']['depth_stamp_ns'] += 5_000_000
    assert proof.retained_framing_source(node.memory, track, 0, node.now_ns)['box'] == box


def test_latest_retained_source_selection_and_eviction_no_easier_old_fallback(perception_node):
    node = perception_node
    older, _, _ = prepare_node(node)
    newer, obj = accepted(node, 11., (.01, 0., 1.))
    track = node.memory.serialize()['pending_landmarks'][0]
    assert proof.retained_framing_source(node.memory, track, 0, node.now_ns)['view_id'] == newer.view_id
    # Newest accepted measurement is damaged: old easier height is not substituted.
    newer.payload['depth_geometry'].clear()
    assert proof.retained_framing_source(node.memory, track, 0, node.now_ns) is None
    node.memory.views.remove(newer)
    assert proof.retained_framing_source(node.memory, track, 0, node.now_ns)['view_id'] == older.view_id
    node.memory.views.clear()
    assert proof.retained_framing_source(node.memory, track, 0, node.now_ns) is None


def serialized(node):
    _, _, track = prepare_node(node)
    views = node.memory.serialize()['views']
    track['framing_source'] = proof.retained_framing_source(node.memory, track, 0, node.now_ns)
    return track, views


@pytest.mark.parametrize('bad', ['epoch', 'frame', 'object', 'evicted', 'stamp', 'box', 'not_track_view', 'conflict', 'confirmed', 'nan'])
def test_consumer_provenance_damage_preserves_default_three_tuple_xy(perception_node, bad):
    track, views = serialized(perception_node)
    tf = transform().transform
    original = observer.track_geometry(track, views, tf)
    if bad == 'epoch': track['framing_source']['epoch'] = 1
    elif bad == 'frame': track['framing_source']['odom_frame'] = 'world'
    elif bad == 'object': track['framing_source']['object_id'] = 'other'
    elif bad == 'evicted': views = []
    elif bad == 'stamp': track['framing_source']['stamp_ns'] += 1
    elif bad == 'box': track['framing_source']['box'][0] += .01
    elif bad == 'not_track_view': track['view_ids'] = ['other']
    elif bad == 'conflict': track['conflicted'] = True
    elif bad == 'confirmed': track['confirmed'] = True
    elif bad == 'nan': track['framing_source']['surface_group']['odom_xyz'][-1][2] = float('nan')
    result = observer.track_geometry(track, views, tf, include_xyz=True, memory_epoch=0)
    assert result[3] == []
    # Removing views can alter duplicated prior entries; compare the exact old API on same input.
    assert result[:3] == observer.track_geometry(track, views, tf)
    if bad not in ('evicted', 'not_track_view'): assert result[:3] == original


def test_independent_xyz_full_rotation_keeps_all_original_xy_support(perception_node):
    track, views = serialized(perception_node)
    tf = transform().transform
    tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = 0., math.sin(.2), 0., math.cos(.2)
    tf.translation.z = .3
    before = copy.deepcopy(track)
    result = observer.track_geometry(track, views, tf, include_xyz=True, memory_epoch=0)
    assert result[:3] == observer.track_geometry(track, views, tf)
    assert len(result[2]) == 2 and len(result[3]) == 3
    for original, actual in zip(GROUP['odom_xyz'], result[3]):
        assert actual == pytest.approx(observer.transform_xyz(original, tf))
    assert track == before


def test_real_tick_passes_separate_height_without_xy_or_budget_change(perception_node, observer_node, monkeypatch):
    track, views = serialized(perception_node)
    node = observer_node
    track['object_id'] = 'candidate_0'; track['framing_source']['object_id'] = 'candidate_0'
    views[0]['observations'][0]['object_id'] = 'candidate_0'
    node.memory.update(epoch=0, pending_landmarks=[track], views=views)
    old = observer.choose_observation_candidate
    def planner_spy(*args, **kwargs):
        assert len(kwargs['support_xy']) == 2 and len(kwargs['vertical_support_xyz']) == 3
        assert isinstance(kwargs['vertical_framing'], VerticalFraming)
        result = old(*args, **kwargs)
        result['vertical_framing'] = {'arrival_tolerance_fits': True, 'support_count': 3}
        return result
    monkeypatch.setattr(observer, 'choose_observation_candidate', planner_spy)
    before = copy.deepcopy(node.memory)
    node._tick()
    request = node.request_pub.records[-1]
    assert set(request) == {'request_id', 'track_id', 'frame_id', 'stamp_ns', 'x', 'y', 'yaw'}
    record = node.state.records[request['request_id']]
    assert record['vertical_framing']['source']['view_id'] == views[0]['view_id']
    assert node.memory == before and node.state.total_attempts == 0


def selection(vertical=None, height=(), **kwargs):
    grid = planner.Grid(100,100,.1,-5.,-5.,0.,[0]*10000)
    camera = planner.CameraFraming(640,530.,319.5,0.,0.,0.,.25,.1)
    return planner.choose_observation_candidate(grid,grid,(-3.,0.),(0.,0.),[(-4.,3.)],[],
        support_xy=[(0.,-.05),(0.,.05)],camera=camera,vertical_framing=vertical,
        vertical_support_xyz=height,**kwargs)


def test_separate_xyz_soft_preference_can_change_pose_without_shrinking_horizontal_support(monkeypatch):
    monkeypatch.setattr(planner, '_fallback_points', lambda *args: [(-3., .05)])
    oracle = VerticalPreferenceOracle(lambda point: point[0] < -2.5)
    xyz = [(0., 0., .1), (0., 0., .4), (0., 0., 1.2)]
    assert selection(oracle, xyz)['pose'][:2] == (-3., .05)
    assert all(call[2] == xyz for call in oracle.calls) and len(oracle.calls) <= 160
    assert selection(VerticalPreferenceOracle(lambda point: False), xyz)['pose'] == selection()['pose']


@pytest.mark.parametrize('bad', [None, [(0., 0.)], [(0., 0., float('nan'))], [(0.,0.,1.)]*65, 'bad'])
def test_entire_bad_optional_height_keeps_original_xy_selection(bad):
    assert selection(model(), bad) == selection()


def test_vertical_fit_cannot_rescue_invalid_horizontal_coverage():
    grid = planner.Grid(100,100,.1,-5.,-5.,0.,[0]*10000)
    camera = planner.CameraFraming(640,530.,319.5,0.,0.,0.,.25,.1)
    oracle = VerticalPreferenceOracle(lambda point: True)
    assert planner.choose_observation_candidate(grid,grid,(-3.,0.),(0.,0.),[(-4.,3.)],[],
        support_xy=[(-100., -100.), (100., 100.)], camera=camera, vertical_framing=oracle,
        vertical_support_xyz=[(0., 0., .2)]) is None
    assert not oracle.calls
