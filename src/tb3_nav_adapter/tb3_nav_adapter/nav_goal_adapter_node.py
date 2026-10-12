#!/usr/bin/env python3
"""
nav_goal_adapter_node.py — Stage-5 nav goal adapter ROS 2 node.

Subscribes to SemanticQueryResult from Stage 4, computes a safe approach
pose relative to the robot in map coordinates via TF, and publishes
a PoseStamped suitable for Nav2.

Subscribed topics
-----------------
  /semantic_query_node/selected_target   tb3_interfaces/SemanticQueryResult

Published topics
----------------
  ~/goal_pose                            geometry_msgs/PoseStamped

Parameters  (see config/nav_goal_adapter.yaml)
----------
  approach_distance       float   Stand-off from target (m)
  min_standoff_distance   float   Reject targets closer than this (m)
  input_topic             str     SemanticQueryResult topic
  output_topic            str     PoseStamped goal topic
  target_frame            str     Desired output frame (default "map")
  tf_timeout              float   TF lookup timeout (s)
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  registers PointStamped transform

from tb3_nav_adapter.goal_adapter_core import yaw_to_quaternion
from tb3_nav_adapter.safe_approach import CostGrid, select_safe_approach

try:
    from tb3_interfaces.msg import SemanticQueryResult
except ImportError:
    SemanticQueryResult = None


class NavGoalAdapterNode(Node):

    def __init__(self) -> None:
        super().__init__("nav_goal_adapter_node")

        if SemanticQueryResult is None:
            self.get_logger().fatal(
                "tb3_interfaces/msg/SemanticQueryResult not available. "
                "Did you source install/setup.bash after building tb3_interfaces?"
            )
            raise RuntimeError("SemanticQueryResult message not found")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("approach_distance", 0.5)
        self.declare_parameter("min_standoff_distance", 0.3)
        self.declare_parameter("max_approach_distance", 1.0)
        self.declare_parameter("max_endpoint_cost", 60)
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("costmap_max_age_sec", 3.0)
        self.declare_parameter("robot_tf_max_age_sec", 0.75)
        self.declare_parameter("input_topic", "/semantic_query_node/selected_target")
        self.declare_parameter("output_topic", "~/goal_pose")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("tf_timeout", 0.5)

        self._approach_dist = self.get_parameter("approach_distance").value
        self._min_standoff  = self.get_parameter("min_standoff_distance").value
        self._max_approach = self.get_parameter("max_approach_distance").value
        self._max_cost = self.get_parameter("max_endpoint_cost").value
        self._cost_age = self.get_parameter("costmap_max_age_sec").value
        self._robot_tf_age = self.get_parameter("robot_tf_max_age_sec").value
        in_topic            = self.get_parameter("input_topic").value
        out_topic           = self.get_parameter("output_topic").value
        self._target_frame  = self.get_parameter("target_frame").value
        self._tf_timeout    = self.get_parameter("tf_timeout").value
        if (not 0 < self._min_standoff <= self._approach_dist <= self._max_approach
                or not 0 <= self._max_cost < 99
                or not 0 < self._cost_age <= 3.
                or not 0 < self._robot_tf_age <= .75):
            raise ValueError("invalid safe approach or freshness limits")
        self._cost_grid = None
        self._cost_stamp_ns = None
        self._last_clock_ns = None

        # ── TF ────────────────────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── QoS ───────────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Sub / Pub ─────────────────────────────────────────────────────
        self.create_subscription(
            SemanticQueryResult, in_topic, self._query_cb, reliable_qos
        )
        self._goal_pub = self.create_publisher(PoseStamped, out_topic, reliable_qos)
        self._status_pub = self.create_publisher(String, "~/status", reliable_qos)
        self.create_subscription(
            OccupancyGrid, self.get_parameter("costmap_topic").value, self._cost_cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(
            Clock, "/clock", self._on_clock,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))

        self.get_logger().info(
            "NavGoalAdapterNode ready  approach=%.2f..%.2fm  standoff=%.2fm  target_frame=%s"
            % (self._approach_dist, self._max_approach, self._min_standoff, self._target_frame)
        )

    # ── Callback ──────────────────────────────────────────────────────────

    @staticmethod
    def _stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _observe_clock(self, now):
        if self._last_clock_ns is not None and now < self._last_clock_ns:
            self._cost_grid = self._cost_stamp_ns = None
        self._last_clock_ns = now

    def _on_clock(self, msg):
        # Observe the same TimeSource clock as query/cost callbacks. An older
        # queued clock message must not look like rollback relative to a newer
        # now() already observed by those callbacks.
        self._observe_clock(self.get_clock().now().nanoseconds)

    def _cost_cb(self, msg):
        self._observe_clock(self.get_clock().now().nanoseconds)
        try:
            if msg.header.frame_id != self._target_frame:
                raise ValueError("global costmap frame differs from target_frame")
            q = msg.info.origin.orientation
            norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
            if not math.isfinite(norm) or abs(norm-1.) > 1e-3 or abs(q.x)+abs(q.y) > 1e-6:
                raise ValueError("global costmap origin is not a planar unit quaternion")
            p = msg.info.origin.position
            grid = CostGrid(msg.info.width, msg.info.height, msg.info.resolution,
                            p.x, p.y, 2*math.atan2(q.z, q.w), tuple(msg.data))
            grid.validate()
            self._cost_grid = grid
            self._cost_stamp_ns = self._stamp_ns(msg.header.stamp)
        except (ValueError, TypeError) as exc:
            self._cost_grid = self._cost_stamp_ns = None
            self.get_logger().warn("Invalid full global costmap: %s" % exc)

    def _audit(self, msg, state, reason, **details):
        payload = dict(state=state, object_id=msg.object_id, reason=reason,
                       approach_distance=self._approach_dist,
                       min_standoff_distance=self._min_standoff,
                       max_approach_distance=self._max_approach,
                       max_endpoint_cost=self._max_cost, frame_id=self._target_frame,
                       stamp_ns=self.get_clock().now().nanoseconds,
                       costmap_stamp_ns=self._cost_stamp_ns, **details)
        self._status_pub.publish(String(data=json.dumps(payload, allow_nan=False)))
        if state == "rejected":
            self.get_logger().warn("No goal for %s: %s" % (msg.object_id, reason))

    def _query_cb(self, msg: SemanticQueryResult) -> None:
        query_wall = time.monotonic()
        query_now = self.get_clock().now().nanoseconds
        self._observe_clock(query_now)
        if not msg.success:
            self.get_logger().debug(
                "Query unsuccessful, skipping: %s" % msg.status_message
            )
            return

        source_frame = msg.frame_id or msg.header.frame_id
        if not source_frame:
            self._audit(msg, "rejected", "target has no coordinate frame")
            return
        now = self.get_clock().now().nanoseconds
        # Full costmap publications use ROS time. A future-stamped or stale
        # cached map must not survive clock rollback or authorize a new goal.
        if (self._cost_grid is None or self._cost_stamp_ns is None
                or not 0 <= now-self._cost_stamp_ns <= self._cost_age*1e9):
            self._audit(msg, "rejected", "fresh full global costmap unavailable")
            return
        # Transform the target before computing its standoff. If it is a
        # body-frame observation, use the observation timestamp, never now.
        target = PointStamped()
        target.header = msg.header
        target.header.frame_id = source_frame
        target.point = msg.position
        try:
            if source_frame != self._target_frame:
                target = self._tf_buffer.transform(
                    target, self._target_frame,
                    timeout=Duration(seconds=self._tf_timeout))
            # The robot pose only needs a fresh measured TF, not an exact
            # interpolation at a clock tick newer than the last odometry sample.
            # Waiting here would block the same executor that receives /tf.
            robot_tf = self._tf_buffer.lookup_transform(
                self._target_frame, "base_link", Time(), timeout=Duration())
            robot_lookup_mode = "latest_cached"
            snapshot_time = self.get_clock().now()
            if self._stamp_ns(robot_tf.header.stamp) > snapshot_time.nanoseconds:
                # A future latest sample can bracket now, but never relabel
                # that future sample as current or relax the freshness bound.
                robot_tf = self._tf_buffer.lookup_transform(
                    self._target_frame, "base_link", snapshot_time,
                    timeout=Duration())
                robot_lookup_mode = "current_cached_interpolation"
        except tf2_ros.TransformException as exc:
            self._audit(msg, "rejected", "target/robot TF unavailable: %s" % exc)
            return
        robot_stamp = self._stamp_ns(robot_tf.header.stamp)
        now = self.get_clock().now().nanoseconds
        self._observe_clock(now)
        if self._cost_grid is None:
            self._audit(msg, "rejected", "navigation context invalidated by clock rollback")
            return
        # This node spins on one thread, so /clock callbacks may wait during a
        # TF lookup or grid search. Charge monotonic elapsed time as well as ROS
        # time, conservatively for the real-time/slow Gazebo execution profile.
        effective_now = max(now, query_now + int((time.monotonic()-query_wall)*1e9))
        if (now < robot_stamp
                or effective_now-robot_stamp > self._robot_tf_age*1e9):
            self._audit(msg, "rejected", "fresh robot TF unavailable",
                        robot_tf_stamp_ns=robot_stamp)
            return

        tx, ty = target.point.x, target.point.y
        rx = robot_tf.transform.translation.x
        ry = robot_tf.transform.translation.y

        try:
            result = select_safe_approach(
                self._cost_grid, (tx, ty), (rx, ry), self._approach_dist,
                self._min_standoff, self._max_approach, self._max_cost)
        except ValueError as exc:
            self._audit(msg, "rejected", str(exc))
            return
        # Reject if the synchronous search or TF wait outlived its sensor data.
        now = self.get_clock().now().nanoseconds
        self._observe_clock(now)
        effective_now = max(now, query_now + int((time.monotonic()-query_wall)*1e9))
        if (self._cost_grid is None or self._cost_stamp_ns is None
                or now < self._cost_stamp_ns or now < robot_stamp
                or effective_now-self._cost_stamp_ns > self._cost_age*1e9
                or effective_now-robot_stamp > self._robot_tf_age*1e9):
            self._audit(msg, "rejected", "navigation context expired during approach search")
            return
        gx, gy, yaw = result.x, result.y, result.yaw
        qx, qy, qz, qw = yaw_to_quaternion(yaw)

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self._target_frame
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.get_logger().info(
            "[%s] %s → approach (%.2f, %.2f) in %s  yaw=%.1f°"
            % (msg.semantic_name, msg.object_id, gx, gy,
               self._target_frame, yaw * 57.2958))

        self._goal_pub.publish(goal)
        self._audit(msg, "published", "reachable costmap-checked approach",
                    target_xy=[tx, ty], robot_xy=[rx, ry],
                    robot_tf_stamp_ns=robot_stamp, robot_tf_lookup_mode=robot_lookup_mode,
                    **asdict(result))


def main(args=None):
    rclpy.init(args=args)
    node = NavGoalAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
