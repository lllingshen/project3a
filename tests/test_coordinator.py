"""Command isolation and DDS callback-order checks without starting ROS nodes."""
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("rclpy")
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/tb3_coordinator"))
from tb3_coordinator.coordinator_node import CoordinatorNode, Mode


def coordinator():
    forwarded, goals, exploration, status = [], [], [], []
    logger = SimpleNamespace(info=lambda _: None, warn=lambda _: None)
    clock = SimpleNamespace(ns=42_000_000_000)
    cancel_response = CancelGoal.Response()
    def timer(seconds, callback):
        result = SimpleNamespace(seconds=seconds, callback=callback, canceled=False)
        result.cancel = lambda: setattr(result, "canceled", True)
        return result
    obj = SimpleNamespace(
        _mode=Mode.EXPLORING, _active_command="", _command_started_ns=0,
        _early_goal=None, _pending_timer=None, _resume_timer=None,
        _command_forwarded=False, _handoff_timer=None, _handoff_cancel_future=None,
        _handoff_cancel_ack=False, _handoff_quiet_since_ns=None,
        _active_nav_goals=set(), _terminal_nav_goals=set(), _canceling_goal_ids=set(),
        _nav_goal_handle=None, _nav_timeout_timer=None, _sweep_timer=None,
        _auto_resume=True, _resume_delay=3.0,
        _query_cmd_pub=SimpleNamespace(publish=forwarded.append),
        _expl_en_pub=SimpleNamespace(publish=exploration.append),
        _status_pub=SimpleNamespace(publish=status.append),
        get_logger=lambda: logger,
        get_clock=lambda: SimpleNamespace(now=lambda: Time(nanoseconds=clock.ns)),
        clock=clock, cancel_response=cancel_response,
        _cancel_all_client=SimpleNamespace(service_is_ready=lambda: True,
            call_async=lambda _: SimpleNamespace(done=lambda: True, result=lambda: cancel_response)),
        create_timer=timer, _send_nav2_goal=goals.append,
    )
    for name in ("_set_mode", "_set_exploration", "_publish_status", "_cancel_nav_goal",
                 "_user_command_cb", "_arm_pending_timeout", "_pending_timeout",
                 "_nav_status_cb", "_handoff_tick",
                 "_query_result_cb", "_goal_pose_cb", "_schedule_resume", "_resume_exploration_cb"):
        setattr(obj, name, MethodType(getattr(CoordinatorNode, name), obj))
    return obj, forwarded, goals, exploration, status


def settle_handoff(obj):
    obj._handoff_tick()  # request cancellation
    obj._handoff_tick()  # acknowledge and start quiet interval
    obj.clock.ns += 300_000_000
    obj._handoff_tick()


def nav_status(code):
    row = GoalStatus()
    row.goal_info.goal_id.uuid = [7] * 16
    row.status = code
    return GoalStatusArray(status_list=[row])


def query(command="Move to the red chair.", success=True):
    return SimpleNamespace(query_text=command, success=success, object_id="selected_red",
        semantic_name="chair", position=SimpleNamespace(x=2.0, y=0.0), status_message="no target")


def pose(seconds=42):
    msg = PoseStamped()
    msg.header.stamp = Time(seconds=seconds).to_msg()
    msg.header.frame_id = "map"
    msg.pose.position.x = 1.5
    return msg


@pytest.mark.parametrize("mode", [Mode.SEMANTIC_QUERYING, Mode.SEMANTIC_NAV])
def test_busy_command_is_not_forwarded_or_substituted(mode):
    obj, forwarded, goals, _, status = coordinator()
    obj._user_command_cb(String(data="Move to the red chair."))
    settle_handoff(obj)
    obj._mode = mode
    obj._user_command_cb(String(data="Move to the blue chair."))
    assert [msg.data for msg in forwarded] == ["Move to the red chair."]
    assert obj._active_command == "Move to the red chair."
    assert goals == [] and "busy" in status[-1].data


@pytest.mark.parametrize("success", [True, False])
def test_mismatched_result_cannot_finish_active_query(success):
    obj, _, goals, _, _ = coordinator()
    obj._user_command_cb(String(data="Move to the red chair."))
    settle_handoff(obj)
    pending = obj._pending_timer
    obj._query_result_cb(query("Move to the blue chair.", success))
    assert obj._mode == Mode.SEMANTIC_QUERYING
    assert obj._pending_timer is pending and not pending.canceled
    assert goals == []


@pytest.mark.parametrize("success", [True, False])
def test_early_pose_waits_for_matching_successful_selection(success):
    obj, _, goals, _, _ = coordinator()
    obj._user_command_cb(String(data="Move to the red chair."))
    settle_handoff(obj)
    early = pose()
    obj._goal_pose_cb(early)
    assert goals == [] and obj._early_goal is early
    obj._query_result_cb(query("Move to the blue chair."))
    assert goals == []
    obj._query_result_cb(query(success=success))
    assert goals == ([early] if success else [])
    assert obj._mode == (Mode.SEMANTIC_NAV if success else Mode.TARGET_FAILED)
    assert obj._pending_timer is None


@pytest.mark.parametrize("mode", [Mode.SEMANTIC_QUERYING, Mode.SEMANTIC_NAV])
def test_pose_from_before_active_command_is_rejected(mode):
    obj, _, goals, _, _ = coordinator()
    obj._user_command_cb(String(data="Move to the red chair."))
    settle_handoff(obj)
    obj._mode = mode
    pending = obj._pending_timer
    obj._goal_pose_cb(pose(seconds=41))
    assert goals == [] and obj._early_goal is None
    assert obj._pending_timer is pending and not pending.canceled


@pytest.mark.parametrize("selection_received", [False, True])
def test_query_or_pose_timeout_fails_then_resumes_without_navigation(selection_received):
    obj, _, goals, exploration, status = coordinator()
    obj._user_command_cb(String(data="Move to the red chair."))
    settle_handoff(obj)
    if selection_received:
        obj._query_result_cb(query())
    pending = obj._pending_timer
    assert pending.seconds == (5.0 if selection_received else 120.0)
    pending.callback()
    assert pending.canceled and obj._pending_timer is None
    assert obj._mode == Mode.TARGET_FAILED and goals == []
    assert "no new goal sent" in status[-1].data
    resume = obj._resume_timer
    resume.callback()
    assert resume.canceled and obj._resume_timer is None
    assert obj._mode == Mode.EXPLORING and goals == []
    assert [msg.data for msg in exploration] == [False, True]


def test_fresh_idle_server_can_acknowledge_without_any_status_history():
    obj, forwarded, goals, _, _ = coordinator()
    obj._user_command_cb(String(data="Move to the red chair."))
    assert forwarded == []
    obj._query_result_cb(query())  # no selection is accepted before forwarding
    assert obj._mode == Mode.SEMANTIC_QUERYING
    settle_handoff(obj)
    assert [msg.data for msg in forwarded] == ["Move to the red chair."]
    obj._handoff_tick()
    assert len(forwarded) == 1 and goals == []


def test_canceling_action_blocks_forwarding_until_terminal_and_quiet():
    obj, forwarded, goals, _, _ = coordinator()
    active = nav_status(GoalStatus.STATUS_EXECUTING)
    obj._nav_status_cb(active)
    obj.cancel_response.goals_canceling = [active.status_list[0].goal_info]
    obj._user_command_cb(String(data="Move to the red chair."))
    settle_handoff(obj)
    assert forwarded == []
    obj._nav_status_cb(nav_status(GoalStatus.STATUS_CANCELING))
    obj.clock.ns += 5_000_000_000
    obj._handoff_tick()
    assert forwarded == []
    obj._nav_status_cb(nav_status(GoalStatus.STATUS_CANCELED))
    obj._handoff_tick()
    obj.clock.ns += 200_000_000
    obj._handoff_tick()
    assert forwarded == []
    obj.clock.ns += 100_000_000
    obj._handoff_tick()
    assert len(forwarded) == 1 and goals == []


def test_stuck_cancellation_times_out_without_forwarding_or_navigation():
    obj, forwarded, goals, exploration, _ = coordinator()
    obj._nav_status_cb(nav_status(GoalStatus.STATUS_CANCELING))
    obj._user_command_cb(String(data="Move to the red chair."))
    settle_handoff(obj)
    handoff = obj._handoff_timer
    obj._pending_timeout()
    assert handoff.canceled and obj._handoff_timer is None
    assert obj._mode == Mode.TARGET_FAILED and forwarded == goals == []
    obj._resume_exploration_cb()
    assert obj._mode == Mode.EXPLORING
    assert [msg.data for msg in exploration] == [False, True]
