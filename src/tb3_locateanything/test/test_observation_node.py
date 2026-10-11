"""Budget/state and real ROS-message callback tests; no simulator or model."""
import copy
from concurrent.futures import Future
from dataclasses import replace
import json
import math
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.time import Time
from rcl_interfaces.msg import ParameterValue
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, String

from tb3_locateanything import observation_node as module
from tb3_locateanything.observation_node import RevisitState, SECOND, transform_xy, track_geometry, hint_geometry


def state(*tracks):
    result = RevisitState(namespace="test")
    result.clock(100*SECOND)
    result.reconcile(tracks or ["candidate_0"], 100*SECOND)
    return result


def offer(s, track="candidate_0", second=100, point=(1., 2.)):
    return s.offer(track, (*point, .3), point, int(second*SECOND))


def event(s, request, kind, second=100):
    return s.event({"event": kind, "request_id": request["request_id"],
                    "track_id": request["track_id"], "stamp_ns": int(second*SECOND)},
                   int(second*SECOND))


def test_proposals_are_immutable_then_expire_without_spending_attempts():
    s = state()
    first = offer(s)
    assert offer(s, second=101, point=(8., 9.)) == first
    second = offer(s, second=102, point=(8., 9.))
    assert second["request_id"] != first["request_id"]
    assert first["request_id"] in s.records
    assert s.total_attempts == 0 and s.first_dispatch_ns is None


def test_cached_positions_are_recent_bounded_deduplicated_and_task_local():
    s = state('candidate_0', 'another')
    for index in range(40):
        offer(s, second=100+2*index, point=(float(index), 0.))
    offer(s, second=180, point=(39.01, 0.))
    offer(s, track='another', second=182, point=(90., 0.))
    points = s.cached_positions('candidate_0')
    assert len(points) == 32 and points[0] == (39.01, 0.)
    assert points[-1] == (8., 0.) and (39., 0.) not in points
    assert s.cached_positions('another') == [(90., 0.)]
    assert s.total_attempts == 0


@pytest.mark.parametrize('kind', ['dispatched', 'geometry', 'expired'])
def test_cached_positions_exclude_actual_attempts_and_geometry_rejections(kind):
    s = state()
    request = offer(s)
    if kind == 'dispatched':
        event(s, request, kind)
        event(s, request, 'aborted', 101)
    else:
        s.event({**request, 'event': 'rejected', 'reason_code': kind}, 101*SECOND)
    assert s.cached_positions('candidate_0') == ([(1., 2.)] if kind == 'expired' else [])
    s.clock(SECOND)
    assert s.cached_positions('candidate_0') == []


def test_late_dispatch_of_archived_proposal_pins_original_and_counts_once():
    s = state()
    first = offer(s)
    newer = offer(s, second=102)
    event(s, first, "dispatched", 102.1)
    event(s, first, "dispatched", 102.2)
    assert s.active == first["request_id"] and s.proposal is None
    assert s.total_attempts == 1 and s.attempted_odom == {"candidate_0": [(1., 2.)]}
    event(s, newer, "rejected", 102.3)
    assert s.active == first["request_id"]
    assert offer(s, second=103) is None


def test_rejected_unselected_proposal_costs_no_attempt_and_old_result_cannot_clear_new_goal():
    s = state()
    first = offer(s)
    event(s, first, "rejected")
    second = offer(s, second=101)
    event(s, second, "dispatched", 101)
    event(s, first, "canceled", 102)
    assert s.total_attempts == 1 and s.active == second["request_id"]


@pytest.mark.parametrize("code,excluded", [("geometry", True), ("expired", False),
    ("protocol", False), ("busy", False), ("nav2", False)])
def test_only_geometric_predispatch_rejection_excludes_position_without_spending_budget(code, excluded):
    s = state()
    request = offer(s)
    s.event({**request, "event": "rejected", "reason_code": code}, 101*SECOND)
    assert s.total_attempts == 0
    assert bool(s.attempted_odom.get("candidate_0")) is excluded
    assert offer(s, second=102) is not None


@pytest.mark.parametrize("terminal", ["rejected", "aborted", "canceled", "timeout", "paused"])
def test_dispatched_failures_keep_attempts_and_attempt_positions(terminal):
    s = state()
    first = offer(s)
    event(s, first, "dispatched")
    event(s, first, terminal, 101)
    assert s.total_attempts == 1 and s.active is None
    assert s.attempted_odom["candidate_0"] == [(1., 2.)]


def test_success_requires_five_seconds_dwell_and_does_not_confirm_memory():
    s = state()
    request = offer(s)
    event(s, request, "dispatched")
    event(s, request, "succeeded", 102)
    event(s, request, "succeeded", 104)  # Duplicate must not extend dwell.
    assert s.status(106*SECOND)["state"] == "settling"
    assert offer(s, second=106) is None
    assert offer(s, second=107) is not None
    assert s.pending_ids == {"candidate_0"} and s.status(107*SECOND)["incomplete"]


def test_user_pause_retracts_and_requires_explicit_enable_even_after_completion():
    s = state()
    request = offer(s)
    event(s, request, "dispatched")
    s.set_enabled(False)
    assert s.cancel_requested and offer(s, second=101) is None
    event(s, request, "canceled", 101)
    assert offer(s, second=110) is None
    s.set_enabled(True)
    assert offer(s, second=110) is not None


@pytest.mark.parametrize("dispatched", [False, True])
def test_confirmed_or_removed_track_withdraws_without_creating_new_identity(dispatched):
    s = state()
    request = offer(s)
    if dispatched:
        event(s, request, "dispatched")
    assert s.reconcile([], 101*SECOND)
    assert s.proposal is None
    if dispatched:
        assert s.cancel_requested and s.active == request["request_id"]
        event(s, request, "canceled", 102)
    assert s.status(103*SECOND)["state"] == "no_pending_tracks"


def test_two_dispatches_per_track_and_eight_global_allow_last_goal_to_finish():
    s = state(*(f"candidate_{i}" for i in range(5)))
    for i in range(8):
        track = f"candidate_{i//2}"
        request = offer(s, track, second=100+i)
        assert request
        event(s, request, "dispatched", 100+i)
        assert not s.reconcile(s.pending_ids, (100+i)*SECOND)
        event(s, request, "aborted", 100+i)
        if i % 2:
            assert offer(s, track, second=100+i) is None
    assert s.total_attempts == 8 and offer(s, "candidate_4", second=110) is None
    assert s.status(110*SECOND)["state"] == "budget_exhausted"


def test_120_second_budget_starts_at_first_dispatch_and_requests_active_cancellation():
    s = state()
    request = offer(s, second=1000)
    assert s.first_dispatch_ns is None
    event(s, request, "dispatched", 1000)
    assert not s.reconcile(s.pending_ids, 1119*SECOND)
    assert s.reconcile(s.pending_ids, 1120*SECOND)
    assert s.cancel_requested
    event(s, request, "timeout", 1120)
    assert offer(s, second=1121) is None


def test_clock_rollback_clears_budgets_and_old_events_cannot_affect_new_epoch():
    s = state()
    first = offer(s)
    event(s, first, "dispatched")
    assert s.clock(1*SECOND)
    assert s.total_attempts == 0 and s.active is None and not s.records
    s.reconcile(["candidate_0"], SECOND)
    second = offer(s, second=1)
    assert first["request_id"] != second["request_id"]
    assert not event(s, first, "dispatched", 1.1)
    assert s.total_attempts == 0


def test_unknown_or_mismatched_event_cannot_spend_budget():
    s = state()
    request = offer(s)
    assert not s.event({"event": "dispatched", "request_id": request["request_id"],
                        "track_id": "another"}, 100*SECOND)
    assert not s.event({"event": "dispatched", "request_id": "unknown",
                        "track_id": "candidate_0"}, 100*SECOND)
    assert s.total_attempts == 0


def transform(seconds=100, xy=(10., 20.), yaw=math.pi/2):
    msg = TransformStamped()
    msg.header.stamp = Time(nanoseconds=int(seconds*SECOND)).to_msg()
    msg.header.frame_id = "map"
    msg.transform.translation.x, msg.transform.translation.y = xy
    msg.transform.rotation.z, msg.transform.rotation.w = math.sin(yaw/2), math.cos(yaw/2)
    return msg


def camera_info(seconds=100):
    msg = CameraInfo()
    msg.header.frame_id = 'camera_optical'
    msg.header.stamp = Time(nanoseconds=int(seconds*SECOND)).to_msg()
    msg.width, msg.height = 640, 480
    msg.k = [530., 0., 319.5, 0., 530., 239.5, 0., 0., 1.]
    return msg


def camera_transform():
    msg = TransformStamped()
    msg.transform.translation.x = .077
    q = msg.transform.rotation
    q.x, q.y, q.z, q.w = -.5, .5, -.5, .5
    return msg


def lookup(seconds=100):
    return lambda target, source, *args: camera_transform() if target == 'base_link' else transform(seconds)


def memory():
    return {"pending_landmarks": [{"object_id": "candidate_0", "x": 99., "y": 99.,
        "confirmed": False, "conflicted": False, "view_ids": ["view_0"],
        "support_points": [[2., 0.], [3., 0.], [4., 0.]],
        "evidence": {"independent_viewpoints": {"arbitrary_label": [{"x": 0., "y": 1.}]}}}],
        "views": [{"view_id": "view_0", "x": 0., "y": 1., "yaw": 0.}],
        "catalog_pending": 0, "catalog_idle": True}


def depth_hint(status="unresolved", label="unseen category"):
    return {"hint_id": "depth_hint_0_0", "epoch": 0, "label": label,
            "status": status, "stamp_ns": 90*SECOND, "last_seen_ns": 90*SECOND,
            "source_view_id": "view_0", "source_pose": [0., 1., 0.],
            "source_camera": [.077, 1., 0.], "box": [10., 10., 100., 100.],
            "surface_groups": [{"odom_xyz": [[x, -.1, .5], [x, .1, .8]],
                                "pixels": [[20, 20], [30, 30]], "count": 2}
                               for x in (2., 4.)],
            "known_object_ids": ["other_0"], "resolved_by": []}


def test_support_previous_views_and_attempts_transform_in_one_current_map_frame():
    data = memory()
    target, previous, support = track_geometry(data["pending_landmarks"][0], data["views"], transform().transform)
    assert target == pytest.approx((10., 23.))
    assert previous[0] == pytest.approx((9., 20.))
    assert support[0] == pytest.approx((10., 22.))
    point = (1., 2.)
    assert transform_xy(transform_xy(point, transform().transform), transform().transform, True) == pytest.approx(point)


def grid(seconds=100):
    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.header.stamp = Time(nanoseconds=int(seconds*SECOND)).to_msg()
    msg.info.width = msg.info.height = 10
    msg.info.resolution = 1.
    msg.info.origin.orientation.w = 1.
    msg.data = [0]*100
    return msg


@pytest.fixture
def node(monkeypatch):
    result = SimpleNamespace()
    for name, value in vars(module.ObservationNode).items():
        if isinstance(value, staticmethod):
            setattr(result, name, value.__func__)
        elif callable(value) and not name.startswith("__"):
            setattr(result, name, MethodType(value, result))
    result.now_ns = 100*SECOND
    result.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=result.now_ns))
    result.state = state()
    result.memory, result.map_msg, result.cost_msg = memory(), grid(1), grid()
    result.last_status = result.catalog_wait_ns = None
    result.camera_info = camera_info()
    result.goal_checker, result.goal_tolerances = 'general_goal_checker', (.25, .25)
    result.parameter_success_wall = module.time.monotonic()
    result.parameter_request_wall, result.parameter_future = -math.inf, None
    result.parameter_names = []
    result._refresh_goal_tolerances = lambda: None
    result.get_logger = lambda: SimpleNamespace(info=lambda _: None)
    result.get_parameter = lambda name: SimpleNamespace(value={"odom_frame": "odom", "robot_base_frame": "base_link"}[name])
    result.tf_buffer = SimpleNamespace(lookup_transform=lookup())
    for name in ("request_pub", "status_pub", "capture_pub"):
        records = []
        setattr(result, name, SimpleNamespace(publish=lambda m, out=records: out.append(json.loads(m.data)), records=records))
    result.busy_pub = SimpleNamespace(records=[])
    result.busy_pub.publish = lambda message: result.busy_pub.records.append(message.data)
    result.planner_calls = []
    def planner(*args, **kwargs):
        result.planner_calls.append((args, kwargs))
        return dict(pose=(11., 22., .4), visibility_reason=module.CLEAR_VISIBILITY,
                    map_sightline_groups=[True])
    monkeypatch.setattr(module, "choose_observation_candidate", planner)
    return result


def test_node_proposes_using_fresh_costmap_and_tf_without_requiring_recent_map(node):
    before = copy.deepcopy(node.memory)
    node._tick()
    request = node.request_pub.records[-1]
    assert request["request_id"] and request["track_id"] == "candidate_0"
    assert request["frame_id"] == "map" and request["stamp_ns"] == 100*SECOND
    args, kwargs = node.planner_calls[-1]
    assert args[3] == pytest.approx((10., 23.))
    assert kwargs["support_xy"][0] == pytest.approx((10., 22.))
    assert kwargs['camera'].offset_x == .077 and kwargs['camera'].yaw_tolerance == .25
    assert node.memory == before and node.state.total_attempts == 0
    assert node.busy_pub.records[-1] is True


def test_node_passes_every_retained_camera_with_current_map_yaw_to_planner(node):
    node.memory['views'].append({'view_id': 'another_object_view', 'x': 2., 'y': 3., 'yaw': math.pi})
    node._tick()
    _, kwargs = node.planner_calls[-1]
    assert len(kwargs['capture_viewpoints']) == 2
    assert kwargs['capture_viewpoints'][0] == pytest.approx((9., 20., math.pi/2))
    assert kwargs['capture_viewpoints'][1] == pytest.approx((7., 22., -math.pi/2))


def test_node_transforms_cached_odom_position_with_current_tf_and_replans_heading(node):
    old = node.state.offer('candidate_0', (999., 999., -2.), (1., 2.), 98*SECOND)
    node._tick()
    _, kwargs = node.planner_calls[-1]
    assert kwargs['cached_viewpoints'] == pytest.approx([(8., 21.)])
    request = node.request_pub.records[-1]
    assert request['request_id'] != old['request_id']
    assert (request['x'], request['y'], request['yaw']) == (11., 22., .4)
    assert request['stamp_ns'] == 100*SECOND
    assert node.state.total_attempts == 0


def test_hint_geometry_transforms_every_group_and_source_without_object_coordinate():
    hint = depth_hint()
    before = copy.deepcopy(hint)
    target, previous, support, groups, camera = hint_geometry(hint, transform().transform)
    assert target == pytest.approx((10., 23.))
    assert previous == pytest.approx([(9., 20.)])
    assert groups[0][0] == pytest.approx((10.1, 22.))
    assert groups[1][1] == pytest.approx((9.9, 24.))
    assert len(support) == 4 and camera == pytest.approx((9., 20.077, math.pi/2))
    assert hint == before and 'x' not in hint and 'object_id' not in hint


@pytest.mark.parametrize('label', ['storage rack', 'arbitrary custom category', 'person'])
def test_depth_hint_proposes_without_semantic_track_or_class_specific_rule(node, label):
    node.memory['pending_landmarks'] = []
    node.memory['observation_hints'] = [depth_hint(label=label)]
    before = copy.deepcopy(node.memory)
    node._tick()
    request = node.request_pub.records[-1]
    assert request['track_id'] == 'depth_hint_0_0' and request['request_id']
    _, kwargs = node.planner_calls[-1]
    assert len(kwargs['hypothesis_groups']) == 2
    assert len(kwargs['support_xy']) == 4
    assert kwargs['source_camera'] == pytest.approx((9., 20.077, math.pi/2))
    assert node.busy_pub.records[-1] and node.state.total_attempts == 0
    assert node.status_pub.records[-1]['pending_tracks_total'] == 0
    assert node.status_pub.records[-1]['observation_hints_unresolved'] == 1
    assert node.memory == before


@pytest.mark.parametrize('status', ['resolved', 'expired'])
def test_resolved_or_expired_hint_clears_active_task_without_claiming_expired_complete(node, status):
    node.memory['pending_landmarks'] = []
    node.memory['observation_hints'] = [depth_hint()]
    node._tick()
    request = node.request_pub.records[-1]
    node._event_cb(String(data=json.dumps({**request, 'event': 'dispatched'})))
    new = copy.deepcopy(node.memory)
    new['observation_hints'][0]['status'] = status
    node._memory_cb(String(data=json.dumps(new)))
    assert node.request_pub.records[-1] == {'request_id': ''}
    node._event_cb(String(data=json.dumps({**request, 'event': 'canceled'})))
    node._tick()
    assert node.state.total_attempts == 1 and node.busy_pub.records[-1] is False
    assert node.status_pub.records[-1]['incomplete'] is (status == 'expired')


def test_mixed_hints_and_tracks_share_two_per_task_eight_total_budget():
    tasks = ['candidate_0', 'depth_hint_0_0', 'candidate_1', 'depth_hint_0_1']
    s = state(*tasks)
    for i in range(8):
        task = tasks[i//2]
        request = offer(s, task, 100+i, point=(float(i), 2.))
        assert request is not None
        event(s, request, 'dispatched', 100+i)
        event(s, request, 'aborted', 100+i)
    assert s.total_attempts == 8 and all(count == 2 for count in s.attempts.values())
    s.reconcile(tasks+['depth_hint_0_2'], 110*SECOND)
    assert not s.ready('depth_hint_0_2', 110*SECOND)
    assert s.status(110*SECOND)['incomplete']


def test_hint_pause_reset_and_delayed_event_follow_existing_cancellation_rules(node):
    node.memory['pending_landmarks'] = []
    node.memory['observation_hints'] = [depth_hint()]
    node._tick()
    request = node.request_pub.records[-1]
    node._command_cb(String(data='go to a custom object'))
    node._tick()
    assert not node.busy_pub.records[-1] and node.state.total_attempts == 0
    node.now_ns = 2*SECOND
    node._now()
    node._event_cb(String(data=json.dumps({**request, 'event': 'dispatched'})))
    assert node.memory is None and node.state.total_attempts == 0 and node.state.epoch == 1


def test_no_safe_hint_pose_reports_incomplete_without_spending_budget(node, monkeypatch):
    node.memory['pending_landmarks'] = []
    node.memory['observation_hints'] = [depth_hint()]
    monkeypatch.setattr(module, 'choose_observation_candidate', lambda *args, **kwargs: None)
    node._tick()
    assert node.status_pub.records[-1]['incomplete'] and not node.busy_pub.records[-1]
    assert node.state.total_attempts == 0


def test_invalid_group_is_not_dropped_to_manufacture_a_hint_pose(node):
    node.memory['pending_landmarks'] = []
    hint = depth_hint()
    hint['surface_groups'][1]['odom_xyz'][0][0] = float('nan')
    node.memory['observation_hints'] = [hint]
    node._tick()
    assert node.planner_calls == [] and not node.busy_pub.records[-1]
    assert node.status_pub.records[-1]['incomplete']


@pytest.mark.parametrize("bad", ["costmap", "tf", "catalog", "frame"])
def test_node_fails_closed_on_stale_or_incomplete_inputs(node, bad):
    if bad == "costmap":
        node.cost_msg = grid(96)
    elif bad == "tf":
        node.tf_buffer.lookup_transform = lambda *args: transform(99)
    elif bad == "catalog":
        node.memory["catalog_pending"] = 1
    else:
        node.map_msg.header.frame_id = "odom"
    node._tick()
    assert node.request_pub.records[-1] == {"request_id": ""}
    assert node.status_pub.records[-1]["incomplete"]
    assert not node.planner_calls


def test_command_callback_withdraws_immediately_and_timer_does_not_resume(node):
    node._tick()
    node._command_cb(String(data="go to any object"))
    assert node.request_pub.records[-1] == {"request_id": ""}
    assert node.busy_pub.records[-1] is False
    node._tick()
    assert node.status_pub.records[-1]["state"] == "paused"
    node._enabled_cb(Bool(data=True))
    node._tick()
    assert node.request_pub.records[-1]["request_id"]


def test_dispatched_request_is_not_republished_while_active(node):
    node._tick()
    request = node.request_pub.records[-1]
    node._event_cb(String(data=json.dumps({**request, "event": "dispatched"})))
    count = len(node.request_pub.records)
    node._tick()
    assert len(node.request_pub.records) == count and node.state.total_attempts == 1
    assert node.busy_pub.records[-1] is True


def test_memory_promotion_clears_proposal_and_preserves_confirmed_memory(node):
    node._tick()
    new_memory = {"landmarks": [{"object_id": "anything_0", "confirmed": True}], "pending_landmarks": []}
    node._memory_cb(String(data=json.dumps(new_memory)))
    assert node.request_pub.records[-1] == {"request_id": ""}
    node._tick()
    assert not node.status_pub.records[-1]["incomplete"]
    assert node.busy_pub.records[-1] is False
    assert node.memory == new_memory


def test_conflicted_pending_remains_reported_as_incomplete_without_navigation(node):
    node.memory["pending_landmarks"][0]["conflicted"] = True
    node._tick()
    assert not node.planner_calls
    assert node.status_pub.records[-1]["pending_tracks_total"] == 1
    assert node.status_pub.records[-1]["incomplete"]


def test_no_reachable_view_does_not_spend_attempt_or_claim_completion(node, monkeypatch):
    monkeypatch.setattr(module, "choose_observation_candidate", lambda *args, **kwargs: None)
    node._tick()
    assert node.state.total_attempts == 0
    assert node.status_pub.records[-1]["incomplete"]
    assert "no safe reachable" in node.status_pub.records[-1]["detail"]
    assert node.busy_pub.records[-1] is False


def test_catalog_wait_busy_is_bounded_before_any_dispatch_and_recovers_when_idle(node):
    node.memory['catalog_pending'] = 1
    node._tick()
    assert node.busy_pub.records[-1] is True
    node.now_ns = 130*SECOND
    node._tick()
    assert node.busy_pub.records[-1] is False
    assert '30 seconds' in node.status_pub.records[-1]['detail']
    assert node.state.total_attempts == 0 and node.status_pub.records[-1]['incomplete']
    node.memory['catalog_pending'] = 0
    node.cost_msg = grid(130)
    node.tf_buffer.lookup_transform = lookup(130)
    node.camera_info = camera_info(130)
    node._tick()
    assert node.busy_pub.records[-1] is True and node.request_pub.records[-1]['request_id']


def test_success_dwell_keeps_busy_but_user_pause_releases_it(node):
    node._tick()
    request = node.request_pub.records[-1]
    node._event_cb(String(data=json.dumps({**request, 'event': 'dispatched'})))
    node._event_cb(String(data=json.dumps({**request, 'event': 'succeeded'})))
    node._tick()
    assert node.status_pub.records[-1]['state'] == 'settling'
    assert node.busy_pub.records[-1] is True
    node._enabled_cb(Bool(data=False))
    assert node.busy_pub.records[-1] is False


def test_expired_proposal_clear_precedes_replacement_without_spending_attempt(node):
    node._tick()
    previous = node.request_pub.records[-1]
    node.now_ns = 102*SECOND
    node.cost_msg = grid(102)
    node.tf_buffer.lookup_transform = lookup(102)
    node._tick()
    assert node.request_pub.records[-2] == {'request_id': ''}
    assert node.request_pub.records[-1]['request_id'] != previous['request_id']
    assert node.state.total_attempts == 0


def test_node_clock_rollback_discards_cached_grids_and_memory(node):
    node._tick()
    node.now_ns = SECOND
    node._tick()
    assert node.memory is node.map_msg is node.cost_msg is None
    assert node.state.epoch == 1 and node.request_pub.records[-1] == {"request_id": ""}


@pytest.mark.parametrize('missing', ['calibration', 'tolerances', 'stale_calibration', 'stale_tolerances', 'nan_skew', 'pitched_camera'])
def test_no_proposal_without_current_calibration_tolerances_and_level_extrinsic(node, missing):
    if missing == 'calibration':
        node.camera_info = None
    elif missing == 'tolerances':
        node.goal_tolerances = None
    elif missing == 'stale_calibration':
        node.camera_info = camera_info(90)
    elif missing == 'stale_tolerances':
        node.parameter_success_wall -= 10.
    elif missing == 'nan_skew':
        node.camera_info.k[1] = math.nan
    else:
        original = node.tf_buffer.lookup_transform
        node.tf_buffer.lookup_transform = lambda target, source, *args: (
            transform() if target == 'base_link' else original(target, source, *args))
    node._tick()
    assert not node.planner_calls and node.request_pub.records[-1] == {'request_id': ''}
    assert node.busy_pub.records[-1] is False


def calibrated_rotation(roll, pitch, yaw):
    """Independent Euler construction of an optical-to-base test transform."""
    result = camera_transform()
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    q = result.transform.rotation
    q.x, q.y = sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy
    q.z, q.w = cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy
    return result


def test_actual_upstream_rounded_optical_joint_gets_nonzero_projection_margin(node):
    # turtlebot3_waffle_pi.urdf optical joint uses -1.57, not exact -pi/2.
    optical = calibrated_rotation(-1.57, 0., -1.57)
    model = module.camera_framing(camera_info(), optical.transform, (.25, .25))
    assert 0. < model.projection_margin < .001
    assert model.yaw_offset == pytest.approx(math.pi/2-1.57, abs=1e-10)
    original = node.tf_buffer.lookup_transform
    node.tf_buffer.lookup_transform = lambda target, source, *args: (
        optical if target == 'base_link' else original(target, source, *args))
    node._tick()
    assert node.planner_calls and node.request_pub.records[-1]['request_id']
    assert node.planner_calls[-1][1]['camera'].projection_margin > 0.


@pytest.mark.parametrize('axis', ['roll', 'pitch'])
def test_deliberate_camera_tilt_still_rejected(axis):
    angles = [-math.pi/2, 0., -math.pi/2]
    angles[0 if axis == 'roll' else 1] += math.radians(2.)
    optical = calibrated_rotation(*angles)
    with pytest.raises(ValueError, match='pitch/roll is unsupported'):
        module.camera_framing(camera_info(), optical.transform, (.25, .25))


def test_rounded_tilt_margin_rejects_a_boundary_point_that_really_projects_outside():
    from tb3_locateanything.observation_planner import framing_fits
    info = camera_info()
    optical = calibrated_rotation(-1.57, 0., -1.57)
    model = module.camera_framing(info, optical.transform, (0., 0.))
    naive = replace(model, projection_margin=0.)
    q = optical.transform.rotation
    # Convert independently with the quaternion skew-matrix identity.
    vector = np.array([q.x, q.y, q.z])
    skew = np.array([[0., -q.z, q.y], [q.z, 0., -q.x], [-q.y, q.x, 0.]])
    inverse = ((q.w*q.w-vector@vector)*np.eye(3)+2*np.outer(vector, vector)+2*q.w*skew).T
    origin = model.camera_xy((0., 0.), 0.)
    edge = max(2., info.width*.005)
    clipped = []
    right, left = naive.limits()
    for relative in (-right+model.projection_margin/2, left-model.projection_margin/2):
        angle = model.yaw_offset+relative
        point = (origin[0]+3*math.cos(angle), origin[1]+3*math.sin(angle))
        assert framing_fits(naive, (0., 0.), 0., [point])
        assert not framing_fits(model, (0., 0.), 0., [point])
        dx, dy = point[0]-origin[0], point[1]-origin[1]
        for v in (0., float(info.height)):
            vertical = (v-info.k[5])/info.k[4]
            dz = ((vertical*inverse[2, :2]-inverse[1, :2]) @ [dx, dy]
                  / (inverse[1, 2]-vertical*inverse[2, 2]))
            px, _, pz = inverse @ [dx, dy, dz]
            u = info.k[0]*px/pz+info.k[2]
            clipped.append(not edge < u < info.width-edge)
    assert any(clipped), 'The rounding margin must cover a real projection error.'


@pytest.mark.parametrize('angles,cx', [((-1.57, 0., -1.57), 319.5),
    ((-math.pi/2+.006, .004, -math.pi/2+.13), 220.),
    ((-math.pi/2-.006, -.004, -math.pi/2-.09), 420.)])
def test_small_calibration_tilt_margin_bounds_actual_3d_pixels(angles, cx):
    from tb3_locateanything.observation_planner import _framed_heading, framing_fits
    info = camera_info()
    info.k[2], info.k[4] = cx, 390.
    optical = calibrated_rotation(*angles)
    model = module.camera_framing(info, optical.transform, (.25, .25))
    support = [(-.5, -.2), (-.5, .2), (.5, -.2), (.5, .2)]
    base = (0., -4.)
    heading = _framed_heading(model, base, (0., 0.), support)
    assert heading is not None and framing_fits(model, base, heading, support)
    # Independent Rz*Ry*Rx matrix, not the node's quaternion column formulas.
    roll, pitch, yaw = angles
    rx = np.array([[1., 0., 0.], [0., math.cos(roll), -math.sin(roll)],
                   [0., math.sin(roll), math.cos(roll)]])
    ry = np.array([[math.cos(pitch), 0., math.sin(pitch)], [0., 1., 0.],
                   [-math.sin(pitch), 0., math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                   [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
    rotation = rz @ ry @ rx
    edge = max(2., model.width*.005)
    for turn in (-model.yaw_tolerance, 0., model.yaw_tolerance):
        actual_yaw = heading+turn
        base_rotation = np.array([[math.cos(actual_yaw), -math.sin(actual_yaw), 0.],
                                  [math.sin(actual_yaw), math.cos(actual_yaw), 0.], [0., 0., 1.]])
        inverse = (base_rotation @ rotation).T
        for angle in np.linspace(0., math.tau, 16, endpoint=False):
            actual_base = (base[0]+model.xy_tolerance*math.cos(angle),
                           base[1]+model.xy_tolerance*math.sin(angle))
            origin = model.camera_xy(actual_base, actual_yaw)
            for point in support:
                dx, dy = point[0]-origin[0], point[1]-origin[1]
                for v in np.linspace(0., info.height, 9):
                    vertical = (v-info.k[5])/info.k[4]
                    # Solve height for this actual image row. This covers
                    # possible object heights within vertical FOV, without
                    # claiming a planar planner guarantees vertical visibility.
                    dz = ((vertical*inverse[2, :2]-inverse[1, :2]) @ [dx, dy]
                          / (inverse[1, 2]-vertical*inverse[2, 2]))
                    px, py, pz = inverse @ [dx, dy, dz]
                    assert pz > 0
                    assert info.k[4]*py/pz+info.k[5] == pytest.approx(v)
                    assert edge < info.k[0]*px/pz+info.k[2] < info.width-edge


def test_goal_tolerances_are_discovered_asynchronously_and_never_assumed(node, monkeypatch):
    wall = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: wall[0])
    node.goal_checker = node.goal_tolerances = None
    calls, futures = [], []
    def request(message):
        calls.append(list(message.names))
        future = Future()
        futures.append(future)
        return future
    node.parameter_client = SimpleNamespace(service_is_ready=lambda: True, call_async=request)
    refresh = lambda: module.ObservationNode._refresh_goal_tolerances(node)
    refresh()
    assert calls == [['goal_checker_plugins']] and node.goal_tolerances is None
    refresh()  # No blocking wait or duplicate call while first future runs.
    assert len(calls) == 1
    plugin = ParameterValue(type=9, string_array_value=['custom_checker'])
    futures[0].set_result(SimpleNamespace(values=[plugin]))
    refresh()
    assert calls[-1] == ['goal_checker_plugins', 'custom_checker.xy_goal_tolerance', 'custom_checker.yaw_goal_tolerance']
    assert node.goal_tolerances is None
    futures[1].set_result(SimpleNamespace(values=[plugin, ParameterValue(type=3, double_value=.17),
                                                ParameterValue(type=3, double_value=.11)]))
    refresh()
    assert node.goal_tolerances == (.17, .11)
    assert node._camera_model(node.now_ns).xy_tolerance == .17
    wall[0] += 3.
    refresh()
    futures[-1].set_result(SimpleNamespace(values=[ParameterValue(type=9, string_array_value=['a', 'b'])]))
    refresh()
    assert node.goal_tolerances is None


def test_nonresponding_parameter_service_has_bounded_future_and_expires_cache(node, monkeypatch):
    wall = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: wall[0])
    node.parameter_request_wall = node.parameter_success_wall = 100.
    original = node.parameter_future = Future()
    node.parameter_client = SimpleNamespace(service_is_ready=lambda: False)
    wall[0] = 107.
    module.ObservationNode._refresh_goal_tolerances(node)
    assert original.cancelled() and node.parameter_future is None
    with pytest.raises(ValueError, match='tolerances'):
        node._camera_model(node.now_ns)


@pytest.mark.parametrize('terminal', ['canceled', 'aborted', 'timeout', 'paused'])
def test_uncertain_visibility_counts_only_unique_actual_dispatch_and_is_not_refunded(terminal):
    s = state()
    request = s.offer('candidate_0', (1., 2., .3), (1., 2.), 100*SECOND,
                      module.UNKNOWN_VISIBILITY, [False])
    assert s.uncertain_dispatched == {} and s.total_attempts == 0
    assert set(request) == {'request_id', 'track_id', 'frame_id', 'stamp_ns', 'x', 'y', 'yaw'}
    event(s, request, 'dispatched')
    event(s, request, 'dispatched')
    event(s, request, 'accepted')
    event(s, request, terminal, 101)
    assert s.uncertain_dispatched == {'candidate_0': 1} and s.total_attempts == 1
    assert s.offer('candidate_0', (2., 3., .3), (2., 3.), 102*SECOND,
                   module.UNKNOWN_VISIBILITY, [False]) is None
    clear = s.offer('candidate_0', (2., 3., .3), (2., 3.), 102*SECOND,
                     module.CLEAR_VISIBILITY, [True])
    assert clear is not None
    event(s, clear, 'dispatched', 102)
    assert s.total_attempts == 2 and s.uncertain_dispatched == {'candidate_0': 1}
    assert s.status(102*SECOND)['uncertain_dispatched_by_task'] == {'candidate_0': 1}


def test_unselected_uncertain_proposal_expiry_rejection_and_clock_reset_do_not_spend_budget():
    s = state()
    first = s.offer('candidate_0', (1., 2., .3), (1., 2.), 100*SECOND,
                    module.UNKNOWN_VISIBILITY, [False])
    assert s.current(102*SECOND) is None
    event(s, first, 'rejected', 102)
    second = s.offer('candidate_0', (2., 3., .3), (2., 3.), 102*SECOND,
                     module.UNKNOWN_VISIBILITY, [False])
    assert second is not None and s.uncertain_dispatched == {}
    event(s, second, 'dispatched', 102)
    s.clock(103*SECOND)
    assert s.clock(SECOND)
    assert s.uncertain_dispatched == {} and s.total_attempts == 0
    assert not event(s, second, 'dispatched', 2)
    s.reconcile(['candidate_0'], 2*SECOND)
    assert s.offer('candidate_0', (2., 3., .3), (2., 3.), 2*SECOND,
                    module.UNKNOWN_VISIBILITY, [False]) is not None


def test_node_uncertain_view_metadata_never_changes_memory_and_dispatch_is_logged_once(node, monkeypatch):
    before, logs, calls = copy.deepcopy(node.memory), [], []
    node.get_logger = lambda: SimpleNamespace(info=lambda text: logs.append(json.loads(text)))
    def plan(*args, **kwargs):
        calls.append(kwargs)
        allowed = kwargs['allow_uncertain_visibility']
        return dict(pose=(11., 22., .4),
                    visibility_reason=module.UNKNOWN_VISIBILITY if allowed else module.CLEAR_VISIBILITY,
                    map_sightline_groups=[not allowed])
    monkeypatch.setattr(module, 'choose_observation_candidate', plan)
    node._tick()
    request = node.request_pub.records[-1]
    assert set(request) == {'request_id', 'track_id', 'frame_id', 'stamp_ns', 'x', 'y', 'yaw'}
    assert node.state.records[request['request_id']]['visibility_reason'] == module.UNKNOWN_VISIBILITY
    assert node.status_pub.records[-1]['proposal_visibility_reason'] == module.UNKNOWN_VISIBILITY
    dispatch = String(data=json.dumps({**request, 'event': 'dispatched'}))
    node._event_cb(dispatch)
    node._event_cb(dispatch)
    node._event_cb(String(data=json.dumps({**request, 'event': 'aborted'})))
    node._tick()
    assert [item['allow_uncertain_visibility'] for item in calls] == [True, False]
    second = node.request_pub.records[-1]
    assert node.state.records[second['request_id']]['visibility_reason'] == module.CLEAR_VISIBILITY
    recorded = [item for item in logs if item.get('state') == 'observation_visibility_dispatch']
    assert len(recorded) == 1 and recorded[0]['uncertain_dispatched_for_task'] == 1
    assert recorded[0]['map_sightline_groups'] == [False]
    assert node.memory == before


def test_confirmed_conflict_remains_incomplete_without_becoming_observation_task(node):
    node.memory['pending_landmarks'] = []
    node.memory['observation_hints'] = []
    node.memory['landmarks'] = [{'object_id': 'chair_1', 'confirmed': True, 'conflicted': True}]
    before = copy.deepcopy(node.memory)
    node._tick()
    status = node.status_pub.records[-1]
    assert status['pending_tracks_total'] == 0 and status['conflicted_tracks_total'] == 1
    assert status['incomplete'] is True and status['total_dispatched'] == 0
    assert node._pending() == {} and not node.planner_calls and not node.busy_pub.records[-1]
    assert node.memory == before


def capture_success(s, request, second=102, epoch=7):
    return s.event({'request_id': request['request_id'], 'track_id': request['track_id'],
                    'event': 'succeeded'}, int(second*SECOND), epoch)


def test_only_actual_success_authorizes_fixed_epoch_five_second_capture_window():
    s = state()
    request = offer(s)
    assert s.capture_request(100*SECOND, 7) is None
    event(s, request, 'dispatched')
    event(s, request, 'accepted')
    assert s.capture_request(101*SECOND, 7) is None
    capture_success(s, request)
    expected = dict(epoch=7, request_id=request['request_id'], track_id='candidate_0',
                    succeeded_stamp_ns=102*SECOND, expires_stamp_ns=107*SECOND)
    assert s.capture_request(103*SECOND, 7) == expected
    capture_success(s, request, 104, epoch=9)  # Duplicate cannot extend time or rebind epoch.
    assert s.capture_request(106*SECOND, 7) == expected
    assert s.capture_request(101*SECOND, 7) is None
    assert s.capture_request(107*SECOND, 7) is None
    assert s.capture_request(103*SECOND, 9) is None
    assert s.total_attempts == 1 and s.pending_ids == {'candidate_0'}


@pytest.mark.parametrize('epoch', [None, True, -1, 2.5, '7'])
def test_missing_or_invalid_public_memory_epoch_cannot_authorize_capture(epoch):
    s = state(); request = offer(s); event(s, request, 'dispatched')
    capture_success(s, request, epoch=epoch)
    assert s.capture is None


@pytest.mark.parametrize('terminal', ['rejected', 'canceled', 'paused', 'aborted', 'timeout'])
def test_failed_and_never_dispatched_requests_cannot_authorize_capture(terminal):
    s = state(); request = offer(s)
    capture_success(s, request)
    assert s.capture_request(103*SECOND, 7) is None
    s = state(); request = offer(s); event(s, request, 'dispatched')
    event(s, request, terminal, 101)
    capture_success(s, request)
    assert s.capture_request(103*SECOND, 7) is None


@pytest.mark.parametrize('invalidate', ['pause', 'clock_reset', 'resolved'])
def test_capture_authorization_is_cleared_on_pause_clock_reset_or_resolved_task(invalidate):
    s = state(); request = offer(s); event(s, request, 'dispatched'); capture_success(s, request)
    if invalidate == 'pause':
        s.set_enabled(False)
        s.set_enabled(True)  # Resume cannot revive the old request.
    elif invalidate == 'clock_reset':
        s.clock(1*SECOND)
    else:
        s.reconcile([], 103*SECOND)
    assert s.capture_request(103*SECOND, 7) is None


def test_eighth_success_can_capture_despite_no_more_navigation_dispatch_budget():
    s = state(); request = offer(s); event(s, request, 'dispatched')
    s.attempts.update({'other_%s' % i: 1 for i in range(7)})
    capture_success(s, request)
    s.reconcile(['candidate_0'], 103*SECOND)
    assert s.total_attempts == 8 and s.capture_request(103*SECOND, 7)
    assert s.ready('candidate_0', 103*SECOND) is False


def test_node_sends_deduplicatable_capture_request_during_settling_only(node):
    node.memory['epoch'] = 7
    original = copy.deepcopy(node.memory)
    node._tick()
    assert node.planner_calls[-1][1]['dedicated_capture'] is True
    request = node.request_pub.records[-1]
    assert not node.capture_pub.records
    node._event_cb(String(data=json.dumps({**request, 'event': 'dispatched'})))
    node.now_ns = 102*SECOND
    node._event_cb(String(data=json.dumps({**request, 'event': 'succeeded'})))
    for second in (102, 103, 106):
        node.now_ns = second*SECOND
        node._tick()
    assert len(node.capture_pub.records) == 3
    assert node.capture_pub.records[0] == node.capture_pub.records[-1]
    payload = node.capture_pub.records[-1]
    assert set(payload) == {'epoch', 'request_id', 'track_id', 'succeeded_stamp_ns', 'expires_stamp_ns'}
    assert payload['epoch'] == 7 and payload['request_id'] == request['request_id']
    assert payload['expires_stamp_ns'] == 107*SECOND
    node.now_ns = 107*SECOND
    node._tick()
    assert len(node.capture_pub.records) == 3
    assert node.memory == original  # Authorization alone creates no evidence or votes.


def test_node_user_command_cancels_capture_authorization_immediately(node):
    node.memory['epoch'] = 7
    node._tick(); request = node.request_pub.records[-1]
    node._event_cb(String(data=json.dumps({**request, 'event': 'dispatched'})))
    node._event_cb(String(data=json.dumps({**request, 'event': 'succeeded'})))
    node._command_cb(String(data='go to chair 0'))
    node._tick()
    assert not node.capture_pub.records and not node.state.enabled


def test_final_eighth_success_keeps_busy_for_dwell_and_bounded_late_catalog(node):
    node.memory['epoch'] = 7
    node.state.attempts = {'earlier_%s' % i: 1 for i in range(7)}
    node._tick(); request = node.request_pub.records[-1]
    node._event_cb(String(data=json.dumps({**request, 'event': 'dispatched'})))
    node.now_ns = 102*SECOND
    node._event_cb(String(data=json.dumps({**request, 'event': 'succeeded'})))
    node.memory.update(catalog_pending=1, catalog_idle=False)
    node.now_ns = 104*SECOND; node._tick()
    assert node.status_pub.records[-1]['state'] == 'settling'
    assert node.status_pub.records[-1]['busy'] is True
    assert node.capture_pub.records[-1]['expires_stamp_ns'] == 107*SECOND
    node.now_ns = 108*SECOND; node._tick()
    assert node.status_pub.records[-1]['state'] == 'budget_exhausted'
    assert node.status_pub.records[-1]['busy'] is True
    assert node.status_pub.records[-1]['detail'] == 'waiting for successful-observation catalog to settle'
    assert len(node.planner_calls) == 1 and node.state.total_attempts == 8
    # Catalog idle/restart cannot extend the original success+35s deadline.
    node.memory.update(catalog_pending=0, catalog_idle=True)
    node.now_ns = 120*SECOND; node._tick()
    assert node.status_pub.records[-1]['busy'] is False
    node.memory.update(catalog_pending=1, catalog_idle=False)
    node.now_ns = 136*SECOND; node._tick()
    assert node.status_pub.records[-1]['busy'] is True
    node.now_ns = 137*SECOND; node._tick()
    assert node.status_pub.records[-1]['busy'] is False
    assert node.status_pub.records[-1]['incomplete'] is True
    assert 'fixed 30 second' in node.status_pub.records[-1]['detail']


def test_successful_capture_window_can_cross_navigation_time_limit_without_new_navigation(node):
    node.memory['epoch'] = 7
    node._tick(); request = node.request_pub.records[-1]
    node._event_cb(String(data=json.dumps({**request, 'event': 'dispatched'})))
    node.now_ns = 219*SECOND
    node._event_cb(String(data=json.dumps({**request, 'event': 'succeeded'})))
    node.memory.update(catalog_pending=1, catalog_idle=False)
    node.now_ns = 221*SECOND; node._tick()
    assert node.state.time_exhausted(node.now_ns)
    assert node.status_pub.records[-1]['state'] == 'settling'
    assert node.status_pub.records[-1]['busy'] is True
    assert node.capture_pub.records[-1]['expires_stamp_ns'] == 224*SECOND
    node.now_ns = 225*SECOND; node._tick()
    assert node.status_pub.records[-1]['state'] == 'budget_exhausted'
    assert node.status_pub.records[-1]['busy'] is True
    assert len(node.planner_calls) == 1
    node.now_ns = 254*SECOND; node._tick()
    assert node.status_pub.records[-1]['busy'] is False


def test_public_memory_epoch_change_permanently_clears_old_capture_window():
    s = state(); request = offer(s); event(s, request, 'dispatched'); capture_success(s, request)
    assert s.capture_request(103*SECOND, 8) is None
    assert s.capture_request(103*SECOND, 7) is None


# R10 scheduling policy: change only selection order, never task eligibility.
def test_localized_pending_retry_precedes_new_hint_by_structure_not_id_or_label(node):
    node.memory['pending_landmarks'][0]['object_id'] = 'zz_existing_track'
    node.memory['pending_landmarks'][0]['label'] = 'arbitrary unseen class'
    hint = depth_hint(label='different arbitrary class')
    hint['hint_id'] = 'aa_new_hint'
    node.memory['observation_hints'] = [hint]
    node.state.attempts['zz_existing_track'] = 1
    node._tick()
    assert node.request_pub.records[-1]['track_id'] == 'zz_existing_track'
    assert len(node.planner_calls) == 1
    assert 'hypothesis_groups' not in node.planner_calls[0][1]
    assert node.state.total_attempts == 1  # A proposal never consumes a dispatch.


@pytest.mark.parametrize('unavailable', ['unsafe_pose', 'malformed_geometry', 'attempt_limit'])
def test_unavailable_pending_falls_back_to_existing_unknown_category_hint(node, monkeypatch, unavailable):
    node.memory['observation_hints'] = [depth_hint(label='previously unseen category')]
    if unavailable == 'malformed_geometry':
        node.memory['pending_landmarks'][0]['view_ids'] = []
        node.memory['pending_landmarks'][0]['evidence']['independent_viewpoints'] = {}
    elif unavailable == 'attempt_limit':
        node.state.attempts['candidate_0'] = 2
    calls = []
    def planner(*args, **kwargs):
        calls.append(kwargs)
        if 'hypothesis_groups' not in kwargs:
            return None
        return dict(pose=(11., 22., .4), visibility_reason=module.CLEAR_VISIBILITY,
                    map_sightline_groups=[True])
    monkeypatch.setattr(module, 'choose_observation_candidate', planner)
    node._tick()
    assert node.request_pub.records[-1]['track_id'] == 'depth_hint_0_0'
    assert 'hypothesis_groups' in calls[-1]
    # Invalid geometry and spent task budget skip the pending planner entirely;
    # a valid but unsafe pending pose must be tried before falling through.
    assert ['hypothesis_groups' in call for call in calls] == (
        [False, True] if unavailable == 'unsafe_pose' else [True])
    assert node.status_pub.records[-1]['observation_hints_unresolved'] == 1
    assert node.state.total_attempts == (2 if unavailable == 'attempt_limit' else 0)


def test_pending_layer_preserves_least_attempted_then_id_fairness(node):
    original = node.memory['pending_landmarks'][0]
    node.memory['pending_landmarks'] = [dict(copy.deepcopy(original), object_id=name)
                                        for name in ('track_c', 'track_b', 'track_a')]
    node.memory['observation_hints'] = [depth_hint()]
    node.state.attempts['track_c'] = 1
    selected = []
    for _ in range(3):
        node._tick()
        request = node.request_pub.records[-1]
        selected.append(request['track_id'])
        event(node.state, request, 'dispatched')
        event(node.state, request, 'aborted')
    assert selected == ['track_a', 'track_b', 'track_a']
    assert node.state.attempts == {'track_c': 1, 'track_a': 2, 'track_b': 1}
    node._tick()
    assert node.request_pub.records[-1]['track_id'] == 'track_b'


@pytest.mark.parametrize('dispatched', [False, True])
def test_priority_does_not_replace_existing_hint_proposal_or_active_goal(node, dispatched):
    original = node.memory['pending_landmarks']
    node.memory['pending_landmarks'] = []
    node.memory['observation_hints'] = [depth_hint()]
    node._tick()
    request = node.request_pub.records[-1]
    if dispatched:
        event(node.state, request, 'dispatched')
    count = len(node.planner_calls)
    node.memory['pending_landmarks'] = original
    node._tick()
    assert len(node.planner_calls) == count
    if dispatched:
        assert node.state.active == request['request_id'] and not node.state.cancel_requested
    else:
        assert node.request_pub.records[-1] == request
        assert node.state.proposal == request['request_id']


@pytest.mark.parametrize('limit', ['eight_global', '120_seconds'])
def test_priority_never_overrides_shared_global_or_sim_time_budget(node, limit):
    node.memory['observation_hints'] = [depth_hint()]
    if limit == 'eight_global':
        node.state.attempts = {'prior_%s' % i: 1 for i in range(8)}
    else:
        node.state.first_dispatch_ns = node.now_ns - 120*SECOND
    node._tick()
    assert not node.planner_calls
    assert not any(record.get('request_id') for record in node.request_pub.records)
    assert node.status_pub.records[-1]['state'] == 'budget_exhausted'


def test_prioritized_pending_keeps_one_unknown_visibility_attempt_gate(node):
    node.memory['observation_hints'] = [depth_hint()]
    node.state.attempts['candidate_0'] = 1
    node.state.uncertain_dispatched['candidate_0'] = 1
    node._tick()
    assert node.request_pub.records[-1]['track_id'] == 'candidate_0'
    assert node.planner_calls[-1][1]['allow_uncertain_visibility'] is False
    assert node.state.attempts['candidate_0'] == 1


def test_class_hint_one_surface_and_blank_label_are_not_object_identity(node):
    hint = depth_hint(label='')
    hint.update(hint_id='class_hint_0_0', hint_kind='class_ambiguity',
                label_candidates=['chair', 'table'])
    hint['surface_groups'] = hint['surface_groups'][:1]
    before = copy.deepcopy(hint)
    assert len(hint_geometry(hint, transform().transform)[3]) == 1
    invalid = copy.deepcopy(hint); invalid.pop('hint_kind')
    with pytest.raises(ValueError): hint_geometry(invalid, transform().transform)
    node.memory['pending_landmarks'] = []
    node.memory['observation_hints'] = [hint]
    node._tick()
    assert node.request_pub.records[-1]['track_id'] == 'class_hint_0_0'
    assert node.planner_calls[-1][1]['focus_hypotheses'] is True
    assert len(node.planner_calls[-1][1]['hypothesis_groups']) == 1
    assert hint == before and node.memory['pending_landmarks'] == []


def test_focus_accounting_is_actual_dispatch_only_and_idempotent():
    s = state('depth_hint_0_0')
    def focused(second=100, index=0):
        return s.offer('depth_hint_0_0', (1., 2., .3), (1., 2.), second*SECOND,
                       focus_group_index=index, hypothesis_group_count=2)
    rejected = focused()
    event(s, rejected, 'rejected')
    assert s.focus_dispatched == {} and s.total_attempts == 0
    request = focused(101)
    assert set(request) == {'request_id', 'track_id', 'frame_id', 'stamp_ns', 'x', 'y', 'yaw'}
    assert focused(101, 1) == request
    event(s, request, 'dispatched', 101); event(s, request, 'dispatched', 101.1)
    assert s.focus_dispatched == {'depth_hint_0_0': [0]} and s.total_attempts == 1
    assert focused(102, 1) is None  # no replacement while active
    event(s, request, 'canceled', 102)
    assert s.focus_dispatched == {'depth_hint_0_0': [0]}
    second = focused(103, 1)
    event(s, second, 'dispatched', 103); event(s, second, 'aborted', 104)
    assert s.focus_dispatched == {'depth_hint_0_0': [0, 1]}
    assert s.total_attempts == 2 and focused(105) is None
    s.clock(SECOND)
    assert s.focus_dispatched == {} and s.total_attempts == 0


def test_tick_passes_prior_focus_only_for_hint_and_preserves_protocols(node, monkeypatch):
    node.memory['pending_landmarks'] = []
    node.memory['epoch'] = 7
    node.memory['observation_hints'] = [depth_hint()]
    before = copy.deepcopy(node.memory)
    calls = []
    def choose(*args, **kwargs):
        calls.append(kwargs)
        prior = kwargs['attempted_focus_groups']
        return dict(pose=(11., 22., .4), visibility_reason=module.CLEAR_VISIBILITY,
                    map_sightline_groups=[True], focus_group_index=1 if prior else 0,
                    hypothesis_group_count=2)
    monkeypatch.setattr(module, 'choose_observation_candidate', choose)
    node._tick()
    first = node.request_pub.records[-1]
    assert calls[-1]['focus_hypotheses'] and calls[-1]['attempted_focus_groups'] == []
    assert len(calls[-1]['hypothesis_groups']) == 2
    assert node.status_pub.records[-1]['proposal_focus_group_index'] == 0
    event(node.state, first, 'dispatched', 100)
    event(node.state, first, 'aborted', 100.2)
    node.now_ns = int(100.2*SECOND)
    node._tick()
    second = node.request_pub.records[-1]
    assert second['request_id'] != first['request_id']
    assert calls[-1]['attempted_focus_groups'] == [0]
    assert node.status_pub.records[-1]['proposal_focus_group_index'] == 1
    event(node.state, second, 'dispatched', 100.2)
    capture_success(node.state, second, second=102, epoch=7)
    payload = node.state.capture_request(103*SECOND, 7)
    assert set(payload) == {'epoch', 'request_id', 'track_id', 'succeeded_stamp_ns', 'expires_stamp_ns'}
    assert node.memory == before


def test_pending_normal_geometry_never_uses_hypothesis_focus(node):
    node._tick()
    kwargs = node.planner_calls[-1][1]
    assert 'focus_hypotheses' not in kwargs and 'hypothesis_groups' not in kwargs


def test_focus_unknown_once_still_allows_clear_second_attempt():
    s = state('depth_hint_0_0')
    request = s.offer('depth_hint_0_0', (1., 2., .3), (1., 2.), 100*SECOND,
                      module.UNKNOWN_VISIBILITY, [False], 0, 2)
    event(s, request, 'dispatched'); event(s, request, 'aborted', 101)
    assert s.offer('depth_hint_0_0', (3., 2., .3), (3., 2.), 102*SECOND,
                   module.UNKNOWN_VISIBILITY, [False], 1, 2) is None
    clear = s.offer('depth_hint_0_0', (3., 2., .3), (3., 2.), 102*SECOND,
                    module.CLEAR_VISIBILITY, [True], 1, 2)
    assert clear and s.uncertain_dispatched == {'depth_hint_0_0': 1}
