#!/usr/bin/env python3
"""Evaluate one live command. Ground truth is used only after goal selection.

Source scripts/project3_env.sh first. This script never launches or stops the
stack. Results append to {"attempts": [...]} in --output, including failures.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


def gazebo_pose(name):
    try:
        result = subprocess.run(["gz", "model", "-m", name, "-p"],
                                text=True, capture_output=True, timeout=2.0)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) >= 6 and all(math.isfinite(x) for x in values[:6]):
                        return {"x": values[0], "y": values[1], "yaw": values[5]}
                except ValueError:
                    pass
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True)
    parser.add_argument("--truth", nargs="+", required=True, help="Actual Gazebo entity names")
    parser.add_argument("--expected-entity", help="Requested physical instance for a descriptive demo")
    parser.add_argument("--expected-id", help="Optional observed baseline memory ID")
    parser.add_argument("--mode", choices=("baseline", "locateanything"), default="baseline")
    parser.add_argument("--note", default="", help="Optional context, such as operator-assisted viewing")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-topic", help="Defaults to the selected mode's debug image")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--resume-timeout", type=float, default=15.0)
    parser.add_argument("--max-distance", type=float, default=1.2)
    parser.add_argument("--localization-tolerance", type=float, default=0.5)
    args = parser.parse_args()
    if args.expected_entity and args.expected_entity not in args.truth:
        parser.error("--expected-entity must occur in --truth")
    if args.output.exists():
        data = json.loads(args.output.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("attempts"), list):
            parser.error("--output must contain an object with an attempts list")
    return args


def main():
    args = parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.qos import qos_profile_sensor_data
    import tf2_ros
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection3DArray
    from tb3_query.msg import SemanticQueryResult
    from cv_bridge import CvBridge
    import cv2

    rclpy.init()

    class Observer(Node):
        def __init__(self):
            super().__init__("project3_check")
            self.buffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.buffer, self)
            self.pub = self.create_publisher(String, "/user_command", 10)
            self.statuses, self.landmarks = [], []
            self.selected = self.terminal = self.image = self.selection_image = None
            self.sent = False
            self.resumed = False
            self.create_subscription(String, "/coordinator_node/status", self.status_cb, 10)
            self.create_subscription(SemanticQueryResult, "/semantic_query_node/selected_target",
                                     self.target_cb, 10)
            self.create_subscription(Detection3DArray,
                                     "/semantic_map_memory_node/landmark_objects", self.memory_cb, 10)
            topic = args.image_topic or ("/locateanything/debug_image" if
                    args.mode == "locateanything" else "/detector_node/debug_image")
            self.create_subscription(Image, topic, self.image_cb, qos_profile_sensor_data)
            self.image_topic = topic

        def robot_map(self):
            tf = self.buffer.lookup_transform("map", "base_link", Time())
            return {"x": tf.transform.translation.x, "y": tf.transform.translation.y,
                    "yaw": yaw(tf.transform.rotation),
                    "stamp_sec": tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9}

        def status_cb(self, msg):
            if not self.sent:
                return
            self.statuses.append({"elapsed_s": time.monotonic() - started, "text": msg.data})
            print(msg.data, flush=True)
            if self.terminal is None and msg.data.startswith(("[TARGET_REACHED]", "[TARGET_FAILED]")):
                self.terminal = msg.data
            if self.terminal and "exploration resumed" in msg.data:
                self.resumed = True

        def target_cb(self, msg):
            if not self.sent or msg.query_text.strip() != args.command.strip():
                return
            self.selected = {
                "success": msg.success, "query_text": msg.query_text,
                "object_id": msg.object_id, "semantic_name": msg.semantic_name,
                "detector_label": msg.detector_label,
                "frame": msg.frame_id or msg.header.frame_id,
                "image_or_query_stamp_sec": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "x": msg.position.x, "y": msg.position.y,
                "status": msg.status_message,
            }
            if self.image is not None:
                self.capture_selection_image(self.image)
            if args.mode == "baseline":
                try:
                    robot = self.robot_map()
                    candidates = [x for x in self.landmarks if x["label"] == msg.detector_label]
                    if candidates:
                        self.selected["nearest_id_at_selection"] = min(
                            candidates, key=lambda x: distance(x, robot))["id"]
                except tf2_ros.TransformException:
                    pass

        def memory_cb(self, msg):
            self.landmarks = [{"id": d.id, "label": d.results[0].hypothesis.class_id,
                               "x": d.bbox.center.position.x, "y": d.bbox.center.position.y,
                               "frame": msg.header.frame_id}
                              for d in msg.detections if d.results]

        def image_cb(self, msg):
            self.image = msg
            self.capture_selection_image(msg)

        def capture_selection_image(self, msg):
            if not self.selected or not self.selected["success"]:
                return
            if self.selection_image is not None and args.mode == "baseline":
                return
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            query_stamp = self.selected["image_or_query_stamp_sec"]
            if ((args.mode == "locateanything" and abs(stamp - query_stamp) < 0.001)
                    or (args.mode == "baseline" and stamp >= query_stamp - 0.5)):
                # LocateAnything publishes raw input before inference and its
                # annotated result later with the same original timestamp.
                # Let that later box image replace the raw input even if DDS
                # delivered selected_target before the annotated callback.
                self.selection_image = msg

    node = Observer()
    started = time.monotonic()
    record = {"utc": datetime.now(timezone.utc).isoformat(), "command": args.command,
              "mode": args.mode, "expected_entity": args.expected_entity,
              "note": args.note,
              "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
              "gazebo_master_uri": os.environ.get("GAZEBO_MASTER_URI"),
              "expected_id": args.expected_id, "errors": [],
              "criterion_m": args.max_distance,
              "localization_tolerance_m": args.localization_tolerance}
    try:
        # Allow discovery, TF, landmarks, and debug images to arrive.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if (node.pub.get_subscription_count() and time.monotonic() - started >= 2.0
                    and (args.mode != "baseline" or node.landmarks)):
                break
        if not node.pub.get_subscription_count():
            raise RuntimeError("No coordinator subscribes to /user_command")
        record["landmarks_before"] = list(node.landmarks)
        node.sent = True
        started = time.monotonic()
        node.pub.publish(String(data=args.command))
        while node.terminal is None and time.monotonic() - started < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
        record["terminal_elapsed_s"] = time.monotonic() - started
        record["terminal"] = node.terminal or "TIMEOUT"

        # Measure immediately, before the coordinator's automatic resumption.
        sample_start = time.monotonic()
        try:
            record["robot_map"] = node.robot_map()
        except tf2_ros.TransformException as exc:
            record["errors"].append("robot map TF unavailable: " + str(exc))
        record["robot_world"] = gazebo_pose("waffle_pi")
        record["robot_sample_duration_s"] = time.monotonic() - sample_start
        # Parallel gz clients intermittently lose replies. Query static props
        # sequentially and retry once; the robot pose is already frozen above.
        record["truth_world"] = {}
        record["truth_requeries"] = []
        for name in args.truth:
            pose = gazebo_pose(name)
            if pose is None:
                record["truth_requeries"].append(name)
                pose = gazebo_pose(name)
            record["truth_world"][name] = pose
        if record["robot_world"] is None or any(p is None for p in record["truth_world"].values()):
            record["errors"].append("Gazebo pose query failed; no fallback used")

        # Keep the selected frame/box, before later exploration changes the view.
        saved_image = node.selection_image or node.image
        if args.image and saved_image:
            args.image.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(args.image), CvBridge().imgmsg_to_cv2(saved_image, "bgr8"))
            if not ok:
                raise RuntimeError("Could not save debug image")
            record["image"] = str(args.image)
            record["image_topic"] = node.image_topic
            record["image_stamp_sec"] = saved_image.header.stamp.sec + saved_image.header.stamp.nanosec * 1e-9
            record["image_at_selection"] = node.selection_image is not None
        elif args.image:
            record["errors"].append("No debug image received")
        resume_deadline = time.monotonic() + args.resume_timeout
        while node.terminal and not node.resumed and time.monotonic() < resume_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception as exc:
        record["errors"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        record.update(selected=node.selected, statuses=node.statuses,
                      exploration_resumed=node.resumed)
        node.destroy_node()
        rclpy.shutdown()

    selected, robot_world, robot_map = (record.get(k) for k in ("selected", "robot_world", "robot_map"))
    truth = record.get("truth_world", {})
    if selected and selected["success"] and selected["frame"] == "map" and robot_world and robot_map and all(truth.values()):
        # SE(2) alignment from paired robot poses accounts for SLAM's map origin
        # and yaw. This is evaluation only; it never changes the navigation goal.
        angle = robot_world["yaw"] - robot_map["yaw"]
        dx, dy = selected["x"] - robot_map["x"], selected["y"] - robot_map["y"]
        target_world = {"x": robot_world["x"] + math.cos(angle)*dx - math.sin(angle)*dy,
                        "y": robot_world["y"] + math.sin(angle)*dx + math.cos(angle)*dy}
        record["map_world_alignment"] = "SE2 from terminal robot TF and immediately queried Gazebo pose"
        if not record.get("terminal", "").startswith("[TARGET_REACHED]"):
            record["alignment_caveat"] = (
                "Failed/timeout navigation can leave the robot moving during the gz query; "
                "localization error from this sequential pose alignment is approximate.")
        record["selected_world"] = target_world
        nearest = min(truth, key=lambda name: distance(target_world, truth[name]))
        expected = args.expected_entity or nearest
        record["matched_entity"] = nearest
        record["selection_correct"] = nearest == expected
        record["localization_error_m"] = distance(target_world, truth[expected])
        record["final_distance_m"] = distance(robot_world, truth[expected])
        record["all_final_distances_m"] = {name: distance(robot_world, p) for name, p in truth.items()}
        record["localization_pass"] = record["localization_error_m"] <= args.localization_tolerance
        record["arrival_pass"] = record["final_distance_m"] <= args.max_distance
        if args.expected_id:
            record["selection_correct"] &= selected["object_id"] == args.expected_id
        if args.mode == "baseline" and not re.search(r"\d", args.command):
            record["nearest_selection_pass"] = selected["object_id"] == selected.get("nearest_id_at_selection")
            record["selection_correct"] &= record["nearest_selection_pass"]
    record["pass"] = bool(record.get("terminal", "").startswith("[TARGET_REACHED]")
                          and record.get("selection_correct") and record.get("localization_pass")
                          and record.get("arrival_pass") and record["exploration_resumed"]
                          and not record["errors"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.output.read_text()) if args.output.exists() else {"attempts": []}
    data["attempts"].append(record)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
