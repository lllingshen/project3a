#!/usr/bin/env python3
"""One simulator drive writer; preserve commands and stop after .5 s of silence."""
import copy
import math
import signal
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rosgraph_msgs.msg import Clock as ClockMessage


class VelocityRelay(Node):
    def __init__(self):
        super().__init__("velocity_relay")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self._pub = self.create_publisher(Twist, "/tb3_drive/cmd_vel", qos)
        self._latest, self._received_at, self._last_clock_ns = None, None, None
        self.create_subscription(Twist, "/cmd_vel", self._command, qos)
        self.create_subscription(ClockMessage, "/clock", self._on_clock, qos_profile_sensor_data)
        # The watchdog must continue while simulated /clock is paused.
        self._timer = self.create_timer(.05, self._tick,
                                        clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.stop()

    def _command(self, message):
        values = (message.linear.x, message.linear.y, message.linear.z,
                  message.angular.x, message.angular.y, message.angular.z)
        if not all(math.isfinite(value) for value in values):
            self.stop()
            return
        self._latest = copy.deepcopy(message)
        self._received_at = time.monotonic()

    def _on_clock(self, message):
        stamp = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        if self._last_clock_ns is not None and stamp < self._last_clock_ns:
            self.stop()
        self._last_clock_ns = stamp

    def _tick(self):
        fresh = self._received_at is not None and time.monotonic() - self._received_at < .5
        self._pub.publish(self._latest if fresh else Twist())

    def stop(self):
        self._latest, self._received_at = None, None
        self._pub.publish(Twist())


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main(args=None):
    # Keep the ROS context alive until the final reliable zero is published.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    node = VelocityRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node._pub.wait_for_all_acked(Duration(seconds=.2))
        node.destroy_node()
        rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
