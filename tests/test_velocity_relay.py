"""Exercise the actual relay callbacks without a ROS graph or robot motion."""
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("rclpy")
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock

PATH = Path(__file__).resolve().parents[1] / "src/tb3_bringup/scripts/velocity_relay.py"
SPEC = importlib.util.spec_from_file_location("velocity_relay_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def relay(monkeypatch):
    wall = SimpleNamespace(now=10.)
    monkeypatch.setattr(MODULE.time, "monotonic", lambda: wall.now)
    sent = []
    node = SimpleNamespace(_latest=None, _received_at=None, _last_clock_ns=None,
                           _pub=SimpleNamespace(publish=sent.append))
    for method in ("_command", "_on_clock", "_tick", "stop"):
        setattr(node, method, MethodType(getattr(MODULE.VelocityRelay, method), node))
    return node, wall, sent


def command(x=.2, z=.4):
    message = Twist()
    message.linear.x, message.angular.z = x, z
    return message


def clock(seconds):
    message = Clock()
    message.clock.sec = seconds
    return message


def test_initial_zero_and_latest_complete_command_is_forwarded_unchanged(relay):
    node, wall, sent = relay
    node._tick()
    assert sent[-1] == Twist()
    node._command(command(.1, .2))
    latest = command(.2, .4)
    latest.linear.y, latest.linear.z, latest.angular.x, latest.angular.y = .01, .02, .03, .04
    node._command(latest)
    node._tick()
    assert sent[-1] == latest
    latest.angular.z = 9.  # Caller mutation cannot change cached evidence.
    node._tick()
    assert sent[-1].angular.z == .4


def test_monotonic_watchdog_stops_at_half_second_even_if_simulation_clock_pauses(relay):
    node, wall, sent = relay
    node._on_clock(clock(100))
    node._command(command())
    wall.now = 10.499
    node._tick()
    assert sent[-1].angular.z == .4
    wall.now = 10.5
    node._tick()
    assert sent[-1] == Twist()
    wall.now = 20.
    node._tick()
    assert sent[-1] == Twist()


def test_clock_rollback_clears_cache_immediately_and_requires_a_new_command(relay):
    node, wall, sent = relay
    node._on_clock(clock(100))
    node._command(command())
    node._on_clock(clock(0))
    assert sent[-1] == Twist() and node._latest is None and node._received_at is None
    node._on_clock(clock(1))
    node._tick()
    assert sent[-1] == Twist()
    node._command(command(.3, -.2))
    node._tick()
    assert sent[-1] == command(.3, -.2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_command_invalidates_previous_motion(relay, value):
    node, wall, sent = relay
    node._command(command())
    node._command(command(value, .4))
    assert sent[-1] == Twist() and node._latest is None
    node._tick()
    assert sent[-1] == Twist()


def test_graceful_shutdown_publishes_zero_before_context_shutdown(monkeypatch):
    events = []
    node = SimpleNamespace(stop=lambda: events.append("stop"),
        _pub=SimpleNamespace(wait_for_all_acked=lambda timeout: events.append("acked")),
        destroy_node=lambda: events.append("destroy"))
    monkeypatch.setattr(MODULE, "VelocityRelay", lambda: node)
    monkeypatch.setattr(MODULE.signal, "signal", lambda *_: None)
    def init(**kwargs):
        assert kwargs["signal_handler_options"] == MODULE.SignalHandlerOptions.NO
        events.append("init")
    def spin(_node):
        assert _node is node
        raise KeyboardInterrupt
    monkeypatch.setattr(MODULE.rclpy, "init", init)
    monkeypatch.setattr(MODULE.rclpy, "spin", spin)
    monkeypatch.setattr(MODULE.rclpy, "shutdown", lambda: events.append("shutdown"))
    MODULE.main()
    assert events == ["init", "stop", "acked", "destroy", "shutdown"]
