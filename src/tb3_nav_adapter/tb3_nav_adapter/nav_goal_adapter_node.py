#!/usr/bin/env python3
"""
nav_goal_adapter_node.py — Stage-5 nav goal adapter ROS 2 node.

Subscribes to SemanticQueryResult from Stage 4, computes a safe approach
pose relative to the robot in map coordinates via TF, and publishes
a PoseStamped suitable for Nav2.

Subscribed topics
-----------------
  /semantic_query_node/selected_target   tb3_query/SemanticQueryResult

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

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, PointStamped

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  registers PointStamped transform

from tb3_nav_adapter.goal_adapter_core import compute_approach_pose, yaw_to_quaternion

try:
    from tb3_query.msg import SemanticQueryResult
except ImportError:
    SemanticQueryResult = None


class NavGoalAdapterNode(Node):

    def __init__(self) -> None:
        super().__init__("nav_goal_adapter_node")

        if SemanticQueryResult is None:
            self.get_logger().fatal(
                "tb3_query/msg/SemanticQueryResult not available. "
                "Did you source install/setup.bash after building tb3_query?"
            )
            raise RuntimeError("SemanticQueryResult message not found")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("approach_distance", 0.5)
        self.declare_parameter("min_standoff_distance", 0.3)
        self.declare_parameter("input_topic", "/semantic_query_node/selected_target")
        self.declare_parameter("output_topic", "~/goal_pose")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("tf_timeout", 0.5)

        self._approach_dist = self.get_parameter("approach_distance").value
        self._min_standoff  = self.get_parameter("min_standoff_distance").value
        in_topic            = self.get_parameter("input_topic").value
        out_topic           = self.get_parameter("output_topic").value
        self._target_frame  = self.get_parameter("target_frame").value
        self._tf_timeout    = self.get_parameter("tf_timeout").value

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

        self.get_logger().info(
            "NavGoalAdapterNode ready  approach=%.2fm  standoff=%.2fm  target_frame=%s"
            % (self._approach_dist, self._min_standoff, self._target_frame)
        )

    # ── Callback ──────────────────────────────────────────────────────────

    def _query_cb(self, msg: SemanticQueryResult) -> None:
        if not msg.success:
            self.get_logger().debug(
                "Query unsuccessful, skipping: %s" % msg.status_message
            )
            return

        source_frame = msg.frame_id or msg.header.frame_id
        if not source_frame:
            self.get_logger().warn("Target has no coordinate frame; skipping goal")
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
            robot_tf = self._tf_buffer.lookup_transform(
                self._target_frame, "base_link", Time(),
                timeout=Duration(seconds=self._tf_timeout))
        except tf2_ros.TransformException as exc:
            self.get_logger().warn("Target/robot TF unavailable; no goal: %s" % exc)
            return

        tx, ty = target.point.x, target.point.y
        rx = robot_tf.transform.translation.x
        ry = robot_tf.transform.translation.y

        result = compute_approach_pose(
            tx, ty, self._approach_dist, self._min_standoff, rx, ry
        )

        if result is None:
            self.get_logger().warn(
                "Target %s too close (%.2fm), skipping goal"
                % (msg.object_id, ((tx-rx)**2 + (ty-ry)**2)**0.5)
            )
            return

        gx, gy, yaw = result
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
