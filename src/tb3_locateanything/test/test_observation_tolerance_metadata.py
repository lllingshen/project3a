"""Offline real callback regression: planning tolerances are immutable request evidence."""
import copy
import json
from types import SimpleNamespace

import pytest
from std_msgs.msg import String
from test_observation_node import node, module


def test_actual_camera_tolerances_bind_to_request_not_later_dispatch_cache(node, monkeypatch):
    logs = []
    node.get_logger = lambda: SimpleNamespace(info=lambda value: logs.append(json.loads(value)))
    monkeypatch.setattr(module.time, 'monotonic', lambda: 20.)
    node.parameter_success_wall = 18.
    node.goal_checker, node.goal_tolerances = 'general_goal_checker', (.25, .10)
    before = copy.deepcopy(node.memory)
    node._tick()
    request = node.request_pub.records[-1]
    metadata = node.state.records[request['request_id']]['planning_goal_tolerances']
    camera = node.planner_calls[-1][1]['camera']
    assert metadata == dict(xy_tolerance_m=camera.xy_tolerance,
                            yaw_tolerance_rad=camera.yaw_tolerance,
                            goal_checker='general_goal_checker',
                            parameter_read_age_wall_sec=2., planning_stamp_ns=node.now_ns)
    assert camera.xy_tolerance == .25 and camera.yaw_tolerance == .1
    assert set(request) == {'request_id', 'track_id', 'frame_id', 'stamp_ns', 'x', 'y', 'yaw'}
    assert (request['x'], request['y'], request['yaw']) == (11., 22., .4)
    node.goal_checker, node.goal_tolerances, node.parameter_success_wall = 'later_checker', (.25, .25), 20.
    node._tick()  # Republish the existing immutable request, not a new proposal.
    assert node.request_pub.records[-1] == request
    dispatch = String(data=json.dumps({**request, 'event': 'dispatched'}))
    node._event_cb(dispatch)
    node._event_cb(dispatch)
    events = [row for row in logs if row.get('state') == 'observation_visibility_dispatch']
    assert len(events) == 1 and events[0]['request_id'] == request['request_id']
    assert events[0]['planning_goal_tolerances'] == metadata
    assert metadata['yaw_tolerance_rad'] == .1 and metadata['goal_checker'] == 'general_goal_checker'
    assert node.memory == before and node.state.total_attempts == 1
    node._event_cb(String(data=json.dumps({**request, 'event': 'aborted'})))
    node._tick()
    second = node.request_pub.records[-1]
    assert second['request_id'] != request['request_id']
    assert node.state.records[second['request_id']]['planning_goal_tolerances']['yaw_tolerance_rad'] == .25
    assert node.state.records[request['request_id']]['planning_goal_tolerances'] == metadata


def test_missing_read_age_is_explicit_unknown_not_nonfinite_json(node):
    camera = node._camera_model(node.now_ns)
    node._camera_model = lambda _: camera
    del node.parameter_success_wall
    node._tick()
    request = node.request_pub.records[-1]
    metadata = node.state.records[request['request_id']]['planning_goal_tolerances']
    assert metadata['parameter_read_age_wall_sec'] is None
    json.dumps(metadata, allow_nan=False)
