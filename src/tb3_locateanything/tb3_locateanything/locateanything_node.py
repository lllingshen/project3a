#!/usr/bin/env python3
"""One command, one frozen sensor snapshot, one selected map target.

The coordinator alone handles /user_command and pauses/cancels navigation.
This node replaces semantic_query_node in LocateAnything mode. It never reads
semantic memory or Gazebo object poses and never publishes a navigation goal.
"""
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import tf2_ros
from tb3_query.msg import SemanticQueryResult

from .geometry import CameraCalibration, Quaternion, Transform3D, Vector3, validate_camera_calibration
from .grounding_core import check_snapshot_times, localize_box, one_box, referring_expression
from .worker_client import WorkerClient


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def tf_geometry(msg):
    t, q = msg.transform.translation, msg.transform.rotation
    return Transform3D(Vector3(t.x, t.y, t.z), Quaternion(q.x, q.y, q.z, q.w))


def odom_pose(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    return p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y*q.y + q.z*q.z))


class LocateAnythingNode(Node):
    def __init__(self):
        super().__init__("locateanything_node")
        defaults = {
            "worker_python": "/home/lingshen/miniforge3/envs/locateanything3b/bin/python",
            "model_path": "/home/lingshen/research/locateanything_standalone/models/LocateAnything-3B",
            "eagle_path": "/home/lingshen/research/locateanything_standalone/Eagle/Embodied",
            "command_topic": "/semantic_query_node/command",
            "output_topic": "/semantic_query_node/selected_target",
            "image_topic": "/camera/image_raw", "camera_info_topic": "/camera/camera_info",
            "scan_topic": "/scan", "odom_topic": "/odom", "output_frame": "map",
            "optical_frame": "", "sensor_tolerance_sec": 0.15, "sensor_max_age_sec": 0.75,
            "stationary_seconds": 0.7, "capture_timeout_sec": 12.0,
            "worker_startup_timeout_sec": 180.0, "inference_timeout_sec": 90.0,
            "max_result_age_sec": 120.0, "evidence_dir": "",
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        self.bridge = CvBridge()
        self.images, self.scans, self.infos, self.odoms = (deque(maxlen=n) for n in (8, 80, 80, 150))
        for msg_type, key, callback in (
            (Image, "image_topic", self.images.append),
            (LaserScan, "scan_topic", self.scans.append),
            (CameraInfo, "camera_info_topic", self.infos.append),
            (Odometry, "odom_topic", self._odom_cb),
        ):
            self.create_subscription(msg_type, self.p[key], callback, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, "/locateanything/status", latched)
        self.debug_pub = self.create_publisher(Image, "/locateanything/debug_image", latched)
        self.marker_pub = self.create_publisher(Marker, "/locateanything/target_marker", latched)
        self.result_pub = self.create_publisher(SemanticQueryResult, self.p["output_topic"], 5)
        self.create_subscription(String, self.p["command_topic"], self._command_cb, 5)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.stationary_since = None
        self.pending = self.snapshot = None
        self.generation = 0
        self.worker = None
        self.worker_ready = False
        self.worker_error = ""
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(self._start_worker)
        self.future_kind = "startup"
        self.debug_image = None
        self.create_timer(0.1, self._tick)
        self.create_timer(1.0, self._republish_debug)
        self._status("loading_model")

    def _start_worker(self):
        self.worker = WorkerClient(self.p["worker_python"], self.p["model_path"], self.p["eagle_path"])
        return self.worker.read(self.p["worker_startup_timeout_sec"])

    def _status(self, state, **fields):
        row = {"state": state, "mode": "locateanything", **fields}
        msg = String()
        msg.data = json.dumps(row, allow_nan=False)
        self.status_pub.publish(msg)
        self.get_logger().info(msg.data)
        if self.p["evidence_dir"]:
            folder = Path(self.p["evidence_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "locateanything.jsonl").open("a", encoding="utf-8") as out:
                out.write(msg.data + "\n")

    def _odom_cb(self, msg):
        self.odoms.append(msg)
        t = msg.twist.twist
        if math.hypot(t.linear.x, t.linear.y) > 0.02 or abs(t.angular.z) > 0.03:
            self.stationary_since = None
            if self.snapshot is not None:
                self.snapshot["robot_moved"] = True
        elif self.stationary_since is None:
            self.stationary_since = stamp_ns(msg)
        if self.snapshot is not None:
            x, y, yaw = odom_pose(msg)
            ox, oy, oyaw = self.snapshot["odom_pose"]
            if math.hypot(x-ox, y-oy) > 0.05 or abs(math.atan2(math.sin(yaw-oyaw), math.cos(yaw-oyaw))) > 0.05:
                self.snapshot["robot_moved"] = True

    def _command_cb(self, msg):
        raw = msg.data.strip()
        if not raw:
            return
        self.generation += 1  # invalidate any result for a superseded command
        self.pending = {"id": self.generation, "command": raw,
                        "wall_start": time.monotonic(), "stamp_ns": self.get_clock().now().nanoseconds}
        self.debug_image = None
        marker = Marker()
        marker.header.frame_id = self.p["output_frame"]
        marker.ns, marker.id, marker.action = "locateanything", 0, Marker.DELETE
        self.marker_pub.publish(marker)
        if not self.worker_ready:
            self._fail(self.worker_error or "model loading; wait for ready before sending a command")
            return
        self._status("waiting_for_stationary_snapshot", request_id=self.generation, command=raw)

    def _fail(self, reason, **fields):
        request = self.pending
        if request is None:
            return
        out = SemanticQueryResult()
        out.success = False
        out.query_text = request["command"]
        out.header.frame_id = out.frame_id = self.p["output_frame"]
        metadata = {}
        if self.snapshot is not None and self.snapshot["request"]["id"] == request["id"]:
            out.header.stamp = self.snapshot["image"].header.stamp
            metadata = {k:v for k,v in self.snapshot["metadata"].items()
                        if k not in ("request_id", "command")}
        out.status_message = reason
        self.result_pub.publish(out)
        self._status("failed", request_id=request["id"], command=request["command"], reason=reason, **metadata, **fields)
        self.pending = None

    def _tick(self):
        if self.future is not None and self.future.done():
            future, kind = self.future, self.future_kind
            self.future = None
            response = {}
            try:
                response = future.result()
                if kind == "startup":
                    self.worker_ready = True
                    self._status("ready", **{k:v for k,v in response.items() if k != "status"})
                elif self.pending and self.snapshot["request"]["id"] == self.pending["id"]:
                    self._finish(response)
                else:
                    self._status("superseded_result_discarded", request_id=self.snapshot["request"]["id"])
            except Exception as exc:
                if kind == "startup":
                    self.worker_error = str(exc)
                    self._status("worker_error", reason=str(exc))
                elif self.pending and self.snapshot["request"]["id"] == self.pending["id"]:
                    self._fail(str(exc), **{k: response[k] for k in
                        ("answer", "prompt", "inference_seconds") if k in response})
            if kind == "inference":
                self.snapshot = None
        if self.pending is None:
            return
        if time.monotonic() - self.pending["wall_start"] > self.p["capture_timeout_sec"] and self.snapshot is None:
            self._fail("stationary synchronized sensor/TF snapshot unavailable before timeout")
            return
        if self.future is not None:
            return
        try:
            snapshot = self._capture()
        except (ValueError, tf2_ros.TransformException):
            return  # wait briefly for the coherent snapshot rather than using stale/latest TF
        self.snapshot = snapshot
        try:
            image = self.bridge.imgmsg_to_cv2(snapshot["image"], "bgr8")
            self.debug_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("PNG encoding failed")
        except (CvBridgeError, cv2.error, ValueError, TypeError) as exc:
            self._fail("invalid camera image: " + str(exc))
            self.snapshot = None
            return
        self.debug_image.header = snapshot["image"].header
        self.debug_pub.publish(self.debug_image)
        if self.p["evidence_dir"]:
            folder = Path(self.p["evidence_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / ("command_%03d_source.png" % self.pending["id"])), image)
        request = {"request_id": self.pending["id"], "command": self.pending["command"],
                   "expression": referring_expression(self.pending["command"]),
                   "image_png": base64.b64encode(encoded.tobytes()).decode("ascii")}
        self._status("inference", **snapshot["metadata"])
        self.future = self.pool.submit(self.worker.infer, request, self.p["inference_timeout_sec"])
        self.future_kind = "inference"

    def _capture(self):
        if not all((self.images, self.scans, self.infos, self.odoms)) or self.stationary_since is None:
            raise ValueError("waiting for stationary sensors")
        # Let dynamic TF catch up to the camera; never request latest TF as a
        # substitute for the image time. These images are still age-checked.
        now_ns = self.get_clock().now().nanoseconds
        eligible = [item for item in self.images if now_ns - stamp_ns(item) >= 100_000_000]
        if not eligible:
            raise ValueError("waiting for image-time transforms")
        image = eligible[-1]
        ns = stamp_ns(image)
        if ns < max(self.pending["stamp_ns"], self.stationary_since) + int(self.p["stationary_seconds"] * 1e9):
            raise ValueError("waiting for stationary settling interval")
        scan, info, odom = [min(buf, key=lambda item: abs(stamp_ns(item)-ns))
                            for buf in (self.scans, self.infos, self.odoms)]
        check_snapshot_times(self.get_clock().now().nanoseconds, ns, stamp_ns(scan), stamp_ns(info),
                             stamp_ns(odom), self.p["sensor_max_age_sec"], self.p["sensor_tolerance_sec"])
        if image.width != info.width or image.height != info.height or image.header.frame_id != info.header.frame_id:
            raise ValueError("camera image and calibration differ")
        optical = self.p["optical_frame"] or info.header.frame_id
        calibration = CameraCalibration(info.width, info.height, info.k[0], info.k[4],
                                        info.k[2], info.k[5], optical)
        if validate_camera_calibration(calibration):
            raise ValueError("invalid camera calibration")
        camera_to_scan = tf_geometry(self.tf_buffer.lookup_transform(
            scan.header.frame_id, optical, Time.from_msg(image.header.stamp)))
        scan_to_map = tf_geometry(self.tf_buffer.lookup_transform(
            self.p["output_frame"], scan.header.frame_id, Time.from_msg(scan.header.stamp)))
        return {"request": dict(self.pending), "image": image, "scan": scan, "calibration": calibration,
                "camera_to_scan": camera_to_scan, "scan_to_map": scan_to_map,
                "odom_pose": odom_pose(odom), "robot_moved": False,
                "metadata": {"request_id": self.pending["id"], "command": self.pending["command"],
                    "image_stamp_ns": ns, "image_frame": image.header.frame_id,
                    "scan_stamp_ns": stamp_ns(scan), "scan_frame": scan.header.frame_id,
                    "camera_info_stamp_ns": stamp_ns(info), "optical_frame": optical,
                    "scan_to_map_translation": scan_to_map.translation.as_dict(),
                    "scan_to_map_quaternion": vars(scan_to_map.rotation)}}

    def _finish(self, response):
        snapshot, request = self.snapshot, self.pending
        if response.get("request_id") != request["id"]:
            raise ValueError("worker request ID mismatch")
        age = (self.get_clock().now().nanoseconds - stamp_ns(snapshot["image"])) / 1e9
        if age < 0 or age > self.p["max_result_age_sec"]:
            raise ValueError("inference result expired")
        if snapshot["robot_moved"] or not self.odoms or abs(self.get_clock().now().nanoseconds - stamp_ns(self.odoms[-1])) / 1e9 > self.p["sensor_max_age_sec"]:
            raise ValueError("robot moved or odometry became stale during inference")
        image = snapshot["image"]
        box = one_box(response["answer"], image.width, image.height)
        self._show_box(image, box, request)
        scan = snapshot["scan"]
        point, geometry = localize_box(box, snapshot["calibration"], snapshot["camera_to_scan"],
            snapshot["scan_to_map"], scan.ranges, scan.angle_min, scan.angle_increment,
            max(0.12, scan.range_min), min(3.5, scan.range_max))
        out = SemanticQueryResult()
        out.header.stamp = image.header.stamp  # original observation time, never inference completion time
        out.header.frame_id = out.frame_id = self.p["output_frame"]
        out.success, out.query_text = True, request["command"]
        out.semantic_name = out.detector_label = "locateanything_target"
        out.object_id = "locateanything_%d" % request["id"]
        out.position.x, out.position.y, out.position.z = point.x, point.y, 0.0
        out.confidence = 1.0  # message transport placeholder, NOT a model confidence
        out.status_message = "selected exact visual instance; confidence=1.0 is transport-only"
        marker = Marker()
        marker.header = out.header
        marker.ns, marker.id, marker.type, marker.action = "locateanything", 0, Marker.SPHERE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y = point.x, point.y
        # Raise the display above baseline landmark spheres; the navigation
        # target itself stays on the ground plane in out.position.
        marker.pose.position.z = 0.35
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.25
        marker.color.r, marker.color.g, marker.color.a = 1.0, 1.0, 1.0
        self.marker_pub.publish(marker)
        row = {**snapshot["metadata"], **geometry, "answer": response["answer"], "prompt": response["prompt"],
               "box_xyxy": box, "target_map": point.as_dict(), "object_id": out.object_id,
               "inference_seconds": response["inference_seconds"], "result_age_sec": age,
               "confidence_kind": "transport_placeholder_not_model_confidence"}
        self._status("selected", **row)
        self.result_pub.publish(out)
        self.pending = None

    def _show_box(self, image, box, request):
        annotated = self.bridge.imgmsg_to_cv2(image, "bgr8").copy()
        x1, y1, x2, y2 = [round(value) for value in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(annotated, request["command"], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        self.debug_image = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        self.debug_image.header = image.header
        self.debug_pub.publish(self.debug_image)
        if self.p["evidence_dir"]:
            prefix = Path(self.p["evidence_dir"]) / ("command_%03d_box.png" % request["id"])
            cv2.imwrite(str(prefix), annotated)

    def _republish_debug(self):
        if self.debug_image is not None:
            self.debug_pub.publish(self.debug_image)

    def destroy_node(self):
        if self.worker is not None:
            self.worker.close()
        self.pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocateAnythingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
