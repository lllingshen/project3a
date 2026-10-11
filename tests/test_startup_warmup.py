"""Warmup timing under late ROS clock discovery and simulation resets."""
import importlib.util
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("rclpy")
from rclpy.time import Time
from std_msgs.msg import Bool


PATH = Path(__file__).resolve().parents[1] / "src/tb3_frontier_exploration/scripts/startup_map_warmup_node.py"
SPEC = importlib.util.spec_from_file_location("startup_map_warmup_for_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Warmup = MODULE.StartupMapWarmupNode


def warmup(delay=12.0, full_rotation=False):
    clock, commands, completed = SimpleNamespace(ns=0), [], []
    timer = SimpleNamespace(canceled=False)
    timer.cancel = lambda: setattr(timer, "canceled", True)
    node = SimpleNamespace(
        _phase=Warmup._PH_INIT_WAIT, _phase_start=None, _last_clock_ns=None,
        _startup_delay=delay, _pause=2.0, _turn_duration=3.0, _omega=0.4,
        _published_done=False, _complete_topic="exploration_warmup_complete",
        _full_rotation=full_rotation, _full_turned=0.0, _full_leg=0,
        _last_yaw=None, _last_odom_ns=None, _full_leg_timeout=12.0,
        _timer=timer, _pub_cmd=SimpleNamespace(publish=commands.append),
        _pub_complete=SimpleNamespace(publish=completed.append),
        get_clock=lambda: SimpleNamespace(now=lambda: Time(nanoseconds=clock.ns)),
        get_logger=lambda: SimpleNamespace(info=lambda _: None, error=lambda _: None),
    )
    for name in vars(Warmup):
        if name.startswith("_PH_"):
            setattr(node, name, getattr(Warmup, name))
    for name in ("_tick", "_next_phase", "_send_stop", "_send_turn", "_finish", "_exploration_cb",
                 "_odom_cb", "_odom_ready", "_full_tick", "_abort_full"):
        setattr(node, name, MethodType(getattr(Warmup, name), node))

    def tick(seconds):
        clock.ns = round(seconds * 1e9)
        node._tick()
    return node, tick, commands, completed


def test_first_clock_at_twenty_seconds_starts_a_full_startup_delay():
    node, tick, commands, completed = warmup()
    tick(0)
    tick(0)
    assert node._phase_start is None
    tick(20)
    tick(31.999)
    assert node._phase == Warmup._PH_INIT_WAIT
    assert all(msg.angular.z == 0.0 for msg in commands)
    tick(32)
    assert node._phase == Warmup._PH_TURN_LEFT
    tick(32.02)
    assert commands[-1].angular.z == 0.4
    assert completed == []


@pytest.mark.parametrize("reset_time", [0.0, 5.0])
def test_clock_reset_during_turn_stops_and_restarts_wait(reset_time):
    node, tick, commands, completed = warmup()
    tick(20)
    tick(32)
    tick(32.5)
    assert commands[-1].angular.z > 0.0
    tick(reset_time)
    assert node._phase == Warmup._PH_INIT_WAIT and commands[-1].angular.z == 0.0
    first_valid = reset_time if reset_time else 1.0
    if not reset_time:
        tick(first_valid)
    tick(first_valid + 11.99)
    assert node._phase == Warmup._PH_INIT_WAIT
    tick(first_valid + 12)
    tick(first_valid + 12.02)
    assert commands[-1].angular.z == 0.4 and completed == []


def test_pause_holds_zero_velocity_for_configured_duration():
    node, tick, commands, _ = warmup(delay=0.0)
    tick(20)
    tick(20.02)
    tick(23.02)
    assert node._phase == Warmup._PH_PAUSE_1 and commands[-1].angular.z == 0.0
    tick(25.019)
    assert node._phase == Warmup._PH_PAUSE_1 and commands[-1].angular.z == 0.0
    tick(25.02)
    assert node._phase == Warmup._PH_RETURN_1
    tick(25.04)
    assert commands[-1].angular.z == -0.4


def test_user_command_cancels_warmup_before_clock_and_reset_cannot_restart_it():
    node, tick, commands, completed = warmup()
    node._exploration_cb(Bool(data=False))
    assert node._timer.canceled and node._phase == Warmup._PH_DONE
    count = len(commands)
    tick(20)
    tick(0)
    tick(40)
    assert len(commands) == count and len(completed) == 1 and completed[0].data


def odometry(node, seconds, yaw):
    stamp = round(seconds*1e9)
    node._odom_cb(SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp//1_000_000_000,
                                                     nanosec=stamp % 1_000_000_000)),
        pose=SimpleNamespace(pose=SimpleNamespace(orientation=SimpleNamespace(
            x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2))))))


def test_full_rotation_measures_four_quarters_with_pause_and_no_backtracking():
    node, tick, commands, completed = warmup(delay=0., full_rotation=True)
    yaw = math.radians(170.)  # The scan must cross the +/-pi wrap.
    pauses = []
    previous_phase = node._phase
    for step in range(1500):
        seconds = 100. + step*.02
        if commands:
            yaw += commands[-1].angular.z*.02
        odometry(node, seconds, yaw)
        tick(seconds)
        if node._phase == node._PH_FULL_PAUSE and previous_phase != node._phase:
            pauses.append((seconds, node._full_turned, node._full_leg))
        if previous_phase == node._PH_FULL_PAUSE and node._phase != previous_phase:
            assert seconds-pauses[-1][0] >= node._pause-1e-9
        previous_phase = node._phase
        if completed:
            break
    assert node._phase == node._PH_DONE and node._timer.canceled
    assert len(completed) == 1 and completed[0].data
    assert [entry[2] for entry in pauses] == [1, 2, 3, 4]
    assert all(angle >= leg*math.pi/2 for _, angle, leg in pauses)
    assert node._full_turned == pytest.approx(2*math.pi, abs=.02)
    assert all(cmd.angular.z >= 0. for cmd in commands)
    assert commands[-1].angular.z == 0.


def test_full_rotation_waits_for_odometry_then_stops_if_it_becomes_stale():
    node, tick, commands, completed = warmup(delay=0., full_rotation=True)
    tick(20.)
    tick(30.)
    assert node._phase == node._PH_INIT_WAIT and commands[-1].angular.z == 0.
    odometry(node, 30., 0.)
    tick(30.)
    odometry(node, 30.02, 0.)
    tick(30.02)
    assert commands[-1].angular.z > 0.
    tick(30.6)
    assert commands[-1].angular.z == 0. and not completed
    tick(42.)
    assert node._phase == node._PH_FAILED and node._timer.canceled
    assert commands[-1].angular.z == 0.
    assert len(completed) == 1 and not completed[0].data


def test_full_rotation_cannot_complete_on_elapsed_time_without_actual_rotation():
    node, tick, commands, completed = warmup(delay=0., full_rotation=True)
    for step in range(650):
        seconds = 20.+step*.02
        odometry(node, seconds, 0.)
        tick(seconds)
        if node._phase == node._PH_FAILED:
            break
    assert node._full_turned == 0.
    assert node._phase == node._PH_FAILED and commands[-1].angular.z == 0.
    assert len(completed) == 1 and not completed[0].data


def test_full_rotation_reverse_motion_is_not_counted_as_positive_coverage():
    node, tick, _, completed = warmup(delay=0., full_rotation=True)
    odometry(node, 20., 0.)
    tick(20.)
    odometry(node, 20.02, 0.)
    tick(20.02)
    for step in range(1, 101):
        seconds = 20.02+step*.02
        odometry(node, seconds, -.01*step)
        tick(seconds)
    assert node._full_turned == pytest.approx(-1.)
    assert node._full_leg == 0 and not completed


def test_full_rotation_clock_reset_discards_accumulated_angle_and_waits_again():
    node, tick, commands, completed = warmup(delay=0., full_rotation=True)
    odometry(node, 20., 0.)
    tick(20.)
    odometry(node, 20.02, 0.)
    tick(20.02)
    odometry(node, 20.22, .1)
    tick(20.22)
    assert node._full_turned > 0.
    tick(0.)
    assert node._full_turned == 0. and node._last_odom_ns is None
    assert node._phase == node._PH_INIT_WAIT and commands[-1].angular.z == 0.
    assert not completed


def test_full_rotation_user_command_cancels_scan_and_cannot_restart_it():
    node, tick, commands, completed = warmup(delay=0., full_rotation=True)
    odometry(node, 20., 0.)
    tick(20.)
    odometry(node, 20.02, 0.)
    tick(20.02)
    node._exploration_cb(Bool(data=False))
    assert node._phase == node._PH_DONE and commands[-1].angular.z == 0.
    count = len(commands)
    odometry(node, 20.04, .1)
    tick(20.04)
    assert len(commands) == count and len(completed) == 1


def test_full_rotation_missing_initial_odometry_has_a_bounded_wait():
    node, tick, commands, completed = warmup(delay=3., full_rotation=True)
    tick(20.)
    tick(34.99)
    assert node._phase == node._PH_INIT_WAIT and not completed
    tick(35.)
    assert node._phase == node._PH_FAILED and node._timer.canceled
    assert commands[-1].angular.z == 0.
    assert len(completed) == 1 and not completed[0].data


def test_full_rotation_losing_odometry_during_pause_stops_incomplete_on_deadline():
    node, tick, commands, completed = warmup(delay=0., full_rotation=True)
    yaw = 0.
    for step in range(400):
        seconds = 20.+step*.02
        if commands:
            yaw += commands[-1].angular.z*.02
        odometry(node, seconds, yaw)
        tick(seconds)
        if node._phase == node._PH_FULL_PAUSE:
            break
    assert node._full_leg == 1 and node._phase == node._PH_FULL_PAUSE
    pause_started = seconds
    tick(pause_started + node._pause + node._full_leg_timeout - .01)
    assert node._phase == node._PH_FULL_PAUSE and not completed
    tick(pause_started + node._pause + node._full_leg_timeout)
    assert node._phase == node._PH_FAILED and node._timer.canceled
    assert commands[-1].angular.z == 0.
    assert len(completed) == 1 and not completed[0].data
