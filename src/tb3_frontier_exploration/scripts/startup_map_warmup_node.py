#!/usr/bin/env python3
"""
One-shot startup scan: pure rotation on cmd_vel to seed SLAM / frontier quality before exploration.

Default: delay → turn left ~45° → pause → return → pause → turn right ~45° → pause → return.
Optional full_rotation: four odometry-measured +90° turns with a pause after each.
Then publishes latched std_msgs/Bool true on exploration_warmup_complete (or param topic).
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class StartupMapWarmupNode(Node):
    _PH_INIT_WAIT = 0
    _PH_TURN_LEFT = 1
    _PH_PAUSE_1 = 2
    _PH_RETURN_1 = 3
    _PH_PAUSE_2 = 4
    _PH_TURN_RIGHT = 5
    _PH_PAUSE_3 = 6
    _PH_RETURN_2 = 7
    _PH_DONE = 8
    _PH_FULL_TURN = 9
    _PH_FULL_PAUSE = 10
    _PH_FAILED = 11

    def __init__(self):
        super().__init__("startup_map_warmup_node")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("warmup_complete_topic", "exploration_warmup_complete")
        self.declare_parameter("startup_delay_sec", 3.0)
        self.declare_parameter("pause_between_sec", 0.5)
        self.declare_parameter("scan_angle_deg", 45.0)
        self.declare_parameter("scan_angular_speed", 0.4)
        self.declare_parameter("timer_rate_hz", 50.0)
        self.declare_parameter("full_rotation", False)
        self.declare_parameter("odom_topic", "odom")

        self._cmd_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self._complete_topic = (
            self.get_parameter("warmup_complete_topic").get_parameter_value().string_value
        )
        angle_rad = math.radians(
            float(self.get_parameter("scan_angle_deg").get_parameter_value().double_value)
        )
        self._omega = float(self.get_parameter("scan_angular_speed").get_parameter_value().double_value)
        if self._omega <= 0.0:
            self._omega = 0.4
        self._turn_duration = angle_rad / self._omega
        self._startup_delay = float(
            self.get_parameter("startup_delay_sec").get_parameter_value().double_value
        )
        self._pause = float(self.get_parameter("pause_between_sec").get_parameter_value().double_value)
        self._full_rotation = self.get_parameter("full_rotation").value
        if self._full_rotation and (not all(math.isfinite(v) for v in
                (self._omega, self._startup_delay, self._pause))
                or self._startup_delay < 0.0 or self._pause < 0.0):
            raise ValueError("Full rotation requires finite speed and nonnegative wait durations")
        self._full_turned = 0.0
        self._full_leg = 0
        self._last_yaw = None
        self._last_odom_ns = None
        # A blocked robot must not be declared scanned merely because the
        # nominal turn duration elapsed. Leave completion false on failure.
        self._full_leg_timeout = max(12.0, 3.0 * (math.pi/2) / self._omega)

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._pub_cmd = self.create_publisher(Twist, self._cmd_topic, 10)
        self._pub_complete = self.create_publisher(Bool, self._complete_topic, qos)

        self._phase = self._PH_INIT_WAIT
        # use_sim_time starts at zero until the first /clock message. Gazebo
        # may already be far past startup_delay when that message arrives.
        self._phase_start = None
        self._last_clock_ns = None
        self._published_done = False
        self._exploration_sub = self.create_subscription(
            Bool, "exploration_enabled", self._exploration_cb, 10
        )
        self._odom_sub = (self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value,
            self._odom_cb, qos_profile_sensor_data) if self._full_rotation else None)

        rate = float(self.get_parameter("timer_rate_hz").get_parameter_value().double_value)
        if rate <= 0.0:
            rate = 50.0
        period = 1.0 / rate
        self._timer = self.create_timer(period, self._tick)

        if self._full_rotation:
            self.get_logger().info(
                "startup_map_warmup_node: full rotation, 4 measured +90° turns "
                "@ %.2f rad/s, delay=%.1fs, pause=%.1fs after each turn"
                % (self._omega, self._startup_delay, self._pause))
        else:
            self.get_logger().info(
                "startup_map_warmup_node: delay=%.1fs, ±%.0f° @ %.2f rad/s (~%.2fs each leg), cmd_vel=%s"
                % (
                    self._startup_delay,
                    math.degrees(angle_rad),
                    self._omega,
                    self._turn_duration,
                    self._cmd_topic,
                )
            )

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if not math.isfinite(norm) or norm < 1e-12:
            return
        stamp = msg.header.stamp.sec*1_000_000_000 + msg.header.stamp.nanosec
        if self._last_odom_ns is not None and stamp <= self._last_odom_ns:
            return
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), norm-2*(q.y*q.y + q.z*q.z))
        if (self._last_yaw is not None and self._last_odom_ns is not None
                and stamp-self._last_odom_ns <= 500_000_000
                and self._phase in (self._PH_FULL_TURN, self._PH_FULL_PAUSE)):
            # Signed accumulation crosses +/-pi without losing a revolution;
            # reverse motion cannot masquerade as positive scan coverage.
            delta = yaw-self._last_yaw
            self._full_turned += math.atan2(math.sin(delta), math.cos(delta))
        self._last_yaw, self._last_odom_ns = yaw, stamp

    def _odom_ready(self, now_ns) -> bool:
        return (self._last_odom_ns is not None
                and -100_000_000 <= now_ns-self._last_odom_ns <= 500_000_000)

    def _full_tick(self, now, dt) -> None:
        if not self._odom_ready(now.nanoseconds):
            self._send_stop()
            # Include the planned wait in each phase's deadline. Lost odometry
            # must not leave a paused/initial scan silently waiting forever.
            planned_wait = (self._startup_delay if self._phase == self._PH_INIT_WAIT
                            else self._pause if self._phase == self._PH_FULL_PAUSE else 0.0)
            if dt >= planned_wait + self._full_leg_timeout:
                self._abort_full("odometry unavailable during startup scan")
            return
        if self._phase == self._PH_INIT_WAIT:
            self._send_stop()
            if dt >= self._startup_delay:
                self._full_turned, self._full_leg = 0.0, 0
                self.get_logger().info("Warmup: full rotation, turn 1/4")
                self._next_phase(self._PH_FULL_TURN)
        elif self._phase == self._PH_FULL_TURN:
            if self._full_turned >= (self._full_leg+1)*math.pi/2:
                self._send_stop()
                self._full_leg += 1
                self._next_phase(self._PH_FULL_PAUSE)
            elif dt >= self._full_leg_timeout:
                self._abort_full("robot did not complete the measured 90-degree turn")
            else:
                self._send_turn(1.0)
        elif self._phase == self._PH_FULL_PAUSE:
            self._send_stop()
            if dt >= self._pause:
                if self._full_leg == 4:
                    self._finish()
                else:
                    self.get_logger().info("Warmup: full rotation, turn %d/4" % (self._full_leg+1))
                    self._next_phase(self._PH_FULL_TURN)

    def _abort_full(self, reason) -> None:
        self._send_stop()
        self._phase = self._PH_FAILED
        self._timer.cancel()
        self._pub_complete.publish(Bool(data=False))
        self.get_logger().error("Full-rotation warmup stopped incomplete: " + reason)

    def _exploration_cb(self, msg: Bool) -> None:
        if not msg.data and self._phase != self._PH_DONE:
            # A user command takes priority over the optional startup scan.
            # End this one-shot warmup so it cannot resume during navigation.
            self._send_stop()
            self.get_logger().info("Warmup stopped early: exploration disabled by coordinator")
            self._finish()

    def _next_phase(self, new_phase: int) -> None:
        self._phase = new_phase
        self._phase_start = self.get_clock().now()

    def _send_turn(self, sign: float) -> None:
        t = Twist()
        t.angular.z = sign * self._omega
        self._pub_cmd.publish(t)

    def _send_stop(self) -> None:
        self._pub_cmd.publish(Twist())

    def _tick(self) -> None:
        if self._phase in (self._PH_DONE, self._PH_FAILED):
            return

        now = self.get_clock().now()
        if self._last_clock_ns is not None and now.nanoseconds < self._last_clock_ns:
            # A world reset invalidates the unfinished scan timing. Stop and
            # restart its startup wait; never continue a turn on a negative dt.
            self._phase = self._PH_INIT_WAIT
            self._phase_start = None
            if self._full_rotation:
                self._full_turned, self._full_leg = 0.0, 0
                self._last_yaw, self._last_odom_ns = None, None
            self.get_logger().info("Warmup clock reset; restarting startup delay")
        self._last_clock_ns = now.nanoseconds
        if now.nanoseconds == 0:
            self._send_stop()
            return
        if self._phase_start is None:
            self._phase_start = now
            self._send_stop()
            return
        dt = (now - self._phase_start).nanoseconds / 1e9

        if self._full_rotation:
            self._full_tick(now, dt)
            return

        if self._phase == self._PH_INIT_WAIT:
            self._send_stop()
            if dt >= self._startup_delay:
                self.get_logger().info("Warmup: turn left ~%.0f°" % math.degrees(self._turn_duration * self._omega))
                self._next_phase(self._PH_TURN_LEFT)
            return

        if self._phase == self._PH_TURN_LEFT:
            self._send_turn(1.0)
            if dt >= self._turn_duration:
                self._send_stop()
                self._next_phase(self._PH_PAUSE_1)
            return

        if self._phase == self._PH_PAUSE_1:
            self._send_stop()
            if dt >= self._pause:
                self.get_logger().info("Warmup: return to forward")
                self._next_phase(self._PH_RETURN_1)
            return

        if self._phase == self._PH_RETURN_1:
            self._send_turn(-1.0)
            if dt >= self._turn_duration:
                self._send_stop()
                self._next_phase(self._PH_PAUSE_2)
            return

        if self._phase == self._PH_PAUSE_2:
            self._send_stop()
            if dt >= self._pause:
                self.get_logger().info("Warmup: turn right ~%.0f°" % math.degrees(self._turn_duration * self._omega))
                self._next_phase(self._PH_TURN_RIGHT)
            return

        if self._phase == self._PH_TURN_RIGHT:
            self._send_turn(-1.0)
            if dt >= self._turn_duration:
                self._send_stop()
                self._next_phase(self._PH_PAUSE_3)
            return

        if self._phase == self._PH_PAUSE_3:
            self._send_stop()
            if dt >= self._pause:
                self.get_logger().info("Warmup: return to forward (complete)")
                self._next_phase(self._PH_RETURN_2)
            return

        if self._phase == self._PH_RETURN_2:
            self._send_turn(1.0)
            if dt >= self._turn_duration:
                self._send_stop()
                self._finish()

    def _finish(self) -> None:
        self._phase = self._PH_DONE
        self._timer.cancel()
        self._pub_complete.publish(Bool(data=True))
        self._published_done = True
        self.get_logger().info(
            "Published exploration warmup complete on '%s' (latched)." % self._complete_topic
        )


def main(args=None):
    rclpy.init(args=args)
    node = StartupMapWarmupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if not node._published_done and not node._full_rotation:
        node._pub_complete.publish(Bool(data=True))
        node.get_logger().warn("Interrupted; published warmup complete so exploration is not blocked.")
    elif not node._published_done:
        node._send_stop()
        node._pub_complete.publish(Bool(data=False))
        node.get_logger().warn("Interrupted full scan remains incomplete; exploration is not enabled.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
