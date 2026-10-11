"""Opt-in DDS integration on isolated domain 149, with no Gazebo or robot."""
import importlib.util
import os
from pathlib import Path
import time

import pytest

pytest.importorskip("rclpy")
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


@pytest.mark.skipif(os.environ.get("RUN_ROS_RELAY_SMOKE") != "1",
                    reason="Enable RUN_ROS_RELAY_SMOKE=1 for the isolated DDS smoke test")
def test_real_node_construction_clock_callbacks_forwarding_and_watchdog(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "src/tb3_bringup/scripts/velocity_relay.py"
    spec = importlib.util.spec_from_file_location("relay_real_ros_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Explicit isolation is part of the test: never use the project's domain 49.
    monkeypatch.setenv("ROS_DOMAIN_ID", "149")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    rclpy.init()
    relay = module.VelocityRelay()
    client = Node("velocity_relay_test_client")
    executor = SingleThreadedExecutor()
    executor.add_node(relay)
    executor.add_node(client)
    commands = client.create_publisher(Twist, "/cmd_vel", 10)
    clock_pub = client.create_publisher(Clock, "/clock", 10)
    received = []
    client.create_subscription(Twist, "/tb3_drive/cmd_vel", received.append, 10)

    def until(condition, seconds=2.):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.01)
        assert condition(), "isolated relay DDS condition timed out"

    def send_clock(seconds):
        message = Clock()
        message.clock.sec = seconds
        clock_pub.publish(message)
        until(lambda: relay._last_clock_ns == seconds * 10**9)

    try:
        until(lambda: commands.get_subscription_count() == 1 and
              clock_pub.get_subscription_count() == 1 and bool(received))
        assert received[-1] == Twist()
        send_clock(100)  # Exercises the actual inherited Node callback path.
        message = Twist()
        message.linear.x, message.angular.z = .2, .4
        commands.publish(message)
        until(lambda: received[-1] == message)
        until(lambda: received[-1] == Twist(), seconds=1.)  # /clock remains frozen.
        commands.publish(message)
        until(lambda: received[-1] == message)
        send_clock(0)
        until(lambda: received[-1] == Twist())
        assert relay._latest is None and relay._received_at is None
        send_clock(1)
        assert relay._latest is None
        message.angular.z = -.2
        commands.publish(message)
        until(lambda: received[-1] == message)
        relay.stop()
        until(lambda: received[-1] == Twist())
    finally:
        relay.stop()
        executor.remove_node(relay)
        executor.remove_node(client)
        relay.destroy_node()
        client.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
