#!/usr/bin/env python3
"""LocateAnything perception, visual memory and command lookup.

Frozen camera/laser observations provide the evidence for both cataloguing and
later description queries. The coordinator and Nav2 remain the motion owners.
"""
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tb3_interfaces.msg import SemanticQueryResult

from .geometry import (CameraCalibration, Quaternion, Transform3D, Vector3,
                       transform_point, validate_camera_calibration)
from .grounding_core import (allows_any_instance, catalog_boxes, check_snapshot_times, described_class, localize_box, memory_relation,
                             one_box, referring_expression, validate_localizable_box,
                             validate_surface_visibility)
from .pending_framing import retained_framing_source
from .visual_memory import Memory, class_box_complete
from .depth_core import AmbiguousDepthError, depth_in_metres, localize_depth_box
from .observation_hints import HintMemory
from .inventory_geometry import filter_ambiguous_inventory
from .inventory_parser import parse_inventory_response
from .output_parser import parse_locateanything_output
from .gallery import (build_gallery, has_spatial_relation, select_gallery_box,
                      select_gallery_part, validate_gallery_part, validate_gallery_part_groups)
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
            "localization_mode": "laser", "depth_topic": "/camera/depth/image_raw",
            "depth_info_topic": "/camera/depth/camera_info",
            "scan_topic": "/scan", "odom_topic": "/odom", "output_frame": "map",
            "imu_topic": "/imu", "max_robot_tilt_deg": 15.0,
            "optical_frame": "", "sensor_tolerance_sec": 0.15, "sensor_max_age_sec": 0.75,
            "stationary_seconds": 0.7, "capture_timeout_sec": 12.0,
            "worker_startup_timeout_sec": 180.0, "inference_timeout_sec": 25.0,
            "query_timeout_sec": 90.0, "evidence_dir": "",
            "memory_classes": ["chair", "person", "trash can"],
            "memory_max_views": 32, "memory_max_age_sec": 1800.0,
            "memory_capture_interval_sec": 0.25, "memory_empty_retry_sec": 20.0,
            "memory_view_angle_deg": 15.0,
            "query_max_views": 4,
            "enable_explicit_id_lookup": True,
            "relation_distance_margin_m": 0.15,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        if self.p["localization_mode"] not in ("laser", "depth"):
            raise ValueError("localization_mode must be laser or depth")
        self.bridge = CvBridge()
        self.images, self.scans, self.infos, self.odoms = (deque(maxlen=n) for n in (8, 80, 80, 150))
        self.imus = deque(maxlen=200)
        self.depths, self.depth_infos = deque(maxlen=12), deque(maxlen=40)
        for msg_type, key, callback in (
            (Image, "image_topic", self.images.append),
            (LaserScan, "scan_topic", self.scans.append),
            (CameraInfo, "camera_info_topic", self.infos.append),
            (Odometry, "odom_topic", self._odom_cb),
            (Imu, "imu_topic", self.imus.append),
        ):
            self.create_subscription(msg_type, self.p[key], callback, qos_profile_sensor_data)
        if self.p["localization_mode"] == "depth":
            self.create_subscription(Image, self.p["depth_topic"], self.depths.append, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, self.p["depth_info_topic"], self.depth_infos.append, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, "/locateanything/status", latched)
        self.memory_pub = self.create_publisher(String, "/locateanything/memory", latched)
        self.memory_markers_pub = self.create_publisher(MarkerArray, "/locateanything/memory_markers", latched)
        self.debug_pub = self.create_publisher(Image, "/locateanything/debug_image", latched)
        self.marker_pub = self.create_publisher(Marker, "/locateanything/target_marker", latched)
        self.result_pub = self.create_publisher(SemanticQueryResult, self.p["output_topic"], 5)
        self.create_subscription(SemanticQueryResult, "/semantic_navigation/reached_target",
                                 self._reached_target_cb, latched)
        self.create_subscription(String, self.p["command_topic"], self._command_cb, 5)
        self.create_subscription(String, "/locateanything/observation_capture", self._observation_capture_cb, 10)
        self.create_subscription(String, "/user_command", self._capture_user_command_cb, 10)
        self.create_subscription(Bool, "/exploration_enabled", self._capture_enabled_cb, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.memory = Memory(max_views=self.p["memory_max_views"],
                             max_age_sec=self.p["memory_max_age_sec"],
                             class_labels=self.p["memory_classes"],
                             confirm_classes=self.p["localization_mode"] == "depth",
                             view_angle_rad=math.radians(self.p["memory_view_angle_deg"]))
        self.observation_hints = HintMemory(max_age_sec=self.p["memory_max_age_sec"])
        self.stationary_since = None
        self.pending = None
        self.generation = self.epoch = 0
        self._reset_target_history()
        self.last_clock_ns = 0
        self._reset_observation_capture()
        self.last_capture_wall = 0.0
        self.catalog_queue = deque()
        self.worker = None
        self.worker_ready = False
        self.worker_error = ""
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(self._start_worker)
        self.job = {"kind": "startup", "epoch": self.epoch}
        self.debug_image = None
        self.create_timer(0.1, self._tick)
        self.create_timer(1.0, self._republish_debug)
        self.create_timer(1.0, self._memory_heartbeat)
        self._status("loading_model")

    def _start_worker(self):
        self.worker = WorkerClient(self.p["worker_python"], self.p["model_path"], self.p["eagle_path"])
        return self.worker.read(self.p["worker_startup_timeout_sec"])

    def _status(self, state, **fields):
        msg = String()
        msg.data = json.dumps({"state": state, "mode": "locateanything", **fields}, allow_nan=False)
        self.status_pub.publish(msg)
        self.get_logger().info(msg.data)
        if self.p["evidence_dir"]:
            folder = Path(self.p["evidence_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "locateanything.jsonl").open("a", encoding="utf-8") as out:
                out.write(msg.data + "\n")

    def _publish_memory(self):
        projection = self._refresh_positions()
        msg = String()
        summary = self.memory.serialize()
        summary["epoch"] = self.epoch
        summary["publish_stamp_ns"] = self.get_clock().now().nanoseconds
        summary["map_projection"] = projection
        for track in summary.get("pending_landmarks", []):
            source = retained_framing_source(self.memory, track, self.epoch, summary["publish_stamp_ns"])
            if source is not None:
                track["framing_source"] = source
        pending = sum(1 for view, _ in self.catalog_queue if view in self.memory.views)
        busy = bool(self.future is not None and self.job and self.job.get("kind") == "catalog")
        summary["catalog_pending"] = pending + int(busy)
        summary["catalog_idle"] = not pending and not busy
        hints = getattr(self, "observation_hints", None)
        summary["observation_hints"] = hints.serialize(self.get_clock().now().nanoseconds) if hints else []
        summary["observation_hints_capacity_rejections"] = hints.rejected_capacity if hints else 0
        msg.data = json.dumps(summary, allow_nan=False)
        self.memory_pub.publish(msg)
        if self.p["evidence_dir"]:
            (Path(self.p["evidence_dir"]) / "memory_latest.json").write_text(msg.data)
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for index, item in enumerate(self.memory.landmarks.values()):
            if item.conflicted or not item.confirmed:
                continue
            marker = Marker()
            marker.header.frame_id = self.p["output_frame"]
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns, marker.id = "locateanything_memory", index
            marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = item.x, item.y, 0.6
            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.18
            marker.color.r, marker.color.g, marker.color.a = 1.0, 0.8, 1.0
            marker.text = item.object_id
            markers.markers.append(marker)
        self.memory_markers_pub.publish(markers)

    def _memory_heartbeat(self):
        # Static observations still need new map coordinates after a SLAM
        # correction. This only projects cached evidence; it never adds votes.
        if self.get_clock().now().nanoseconds >= self.last_clock_ns:
            self._publish_memory()

    def _current_projection(self, source_frame):
        now = self.get_clock().now().nanoseconds
        current = self.tf_buffer.lookup_transform(self.p["output_frame"], source_frame, Time())
        if stamp_ns(current) > now:
            current = self.tf_buffer.lookup_transform(
                self.p["output_frame"], source_frame, Time(nanoseconds=now))
            if stamp_ns(current) != now:
                raise ValueError("current map-to-odom transform did not match requested clock time")
        if not 0 < stamp_ns(current) <= now or (now-stamp_ns(current))/1e9 > self.p["sensor_max_age_sec"]:
            raise ValueError("current map-to-odom transform is stale or future-dated")
        if current.header.frame_id != self.p["output_frame"] or current.child_frame_id != source_frame:
            raise ValueError("current map-to-odom transform frames do not match")
        geometry = tf_geometry(current)
        if not all(math.isfinite(v) for v in geometry.translation.as_dict().values()):
            raise ValueError("current map-to-odom translation is not finite")
        transform_point(geometry, Vector3(0., 0., 0.))  # Validate its quaternion.
        return current

    def _refresh_positions(self):
        """Project every retained object with one recorded map<-odom transform."""
        result = {"valid": False, "transform": None, "object_ids": [], "errors": []}
        frames = {view.payload["odom_frame"] for view in self.memory.views if view.observations}
        if not frames and not self.memory.landmarks:
            result["valid"] = True
            return result
        if len(frames) != 1:
            result["errors"].append("retained object observations must share one odometry frame")
            return result
        try:
            current = self._current_projection(next(iter(frames)))
        except (ValueError, tf2_ros.TransformException) as exc:
            result["errors"].append(str(exc))
            return result
        t, q = current.transform.translation, current.transform.rotation
        result["transform"] = {"parent": current.header.frame_id, "child": current.child_frame_id,
                               "stamp_ns": stamp_ns(current), "xyz": [t.x, t.y, t.z],
                               "quaternion": [q.x, q.y, q.z, q.w]}
        points = {}
        for view in self.memory.views:
            for observation in view.observations:
                try:
                    point, _ = self._localize(view, observation["box"], current)
                    points.setdefault(observation["object_id"], []).append(point)
                except (ValueError, tf2_ros.TransformException) as exc:
                    result["errors"].append(f'{observation["object_id"]}/{view.view_id}: {exc}')
        for oid, observations in points.items():
            obj = self.memory.landmarks.get(oid)
            if obj is not None:
                obj.x = sum(p.x for p in observations) / len(observations)
                obj.y = sum(p.y for p in observations) / len(observations)
                result["object_ids"].append(oid)
        missing = set(self.memory.landmarks)-set(result["object_ids"])
        if missing:
            result["errors"].append("objects without reprojectable observations: " + ", ".join(sorted(missing)))
        result["valid"] = not result["errors"]
        return result

    def _odom_cb(self, msg):
        self.odoms.append(msg)
        t = msg.twist.twist
        moving = math.hypot(t.linear.x, t.linear.y) > 0.02 or abs(t.angular.z) > 0.03
        if moving:
            self.stationary_since = None
        elif self.stationary_since is None:
            self.stationary_since = stamp_ns(msg)
        # A historical camera pose is unrelated to the robot's current pose.
        # Only motion since THIS query started invalidates a paused query.
        if self.pending and self.pending.get("query_pose") is not None:
            x, y, yaw = odom_pose(msg)
            ox, oy, oyaw = self.pending["query_pose"]
            if moving or math.hypot(x-ox, y-oy) > 0.05 or abs(math.atan2(math.sin(yaw-oyaw), math.cos(yaw-oyaw))) > 0.05:
                self.pending["robot_moved"] = True

    def _reset_target_history(self):
        # A restarted node or reset simulation must not accept a latched receipt
        # for a different lifetime of identically named memory objects.
        self.selection_session = uuid4().hex
        self.issued_selections = {}
        self.last_reached = None
        self.last_reached_request = 0

    def _reached_target_cb(self, msg):
        issued = self.issued_selections.get(msg.status_message)
        if not msg.success or issued is None:
            return
        request_id, epoch, object_id, command = issued
        if (epoch != self.epoch or request_id <= self.last_reached_request
                or msg.object_id != object_id or msg.query_text != command):
            return
        self.last_reached = object_id
        self.last_reached_request = request_id
        self._status("navigation_context_updated", object_id=object_id, request_id=request_id,
                     epoch=epoch, reason="Nav2 succeeded for this selection")

    def _command_cb(self, msg):
        raw = msg.data.strip()
        if not raw:
            return
        self._capture_user_command_cb(msg)
        self.generation += 1
        self.pending = {"id": self.generation, "command": raw, "wall_start": time.monotonic(),
                        "stamp_ns": self.get_clock().now().nanoseconds, "query_pose": None,
                        "robot_moved": False, "views": None, "fallback_views": deque(),
                        "gallery_attempted": False, "candidates": [], "errors": []}
        self.debug_image = None
        marker = Marker()
        marker.header.frame_id = self.p["output_frame"]
        marker.ns, marker.id, marker.action = "locateanything", 0, Marker.DELETE
        self.marker_pub.publish(marker)
        if not self.worker_ready:
            self._fail(self.worker_error or "model loading; wait for ready before sending a command")
            return
        self._status("waiting_for_stationary_query", request_id=self.generation, command=raw)

    def _fail(self, reason):
        request = self.pending
        if request is None:
            return
        out = SemanticQueryResult()
        out.success, out.query_text = False, request["command"]
        out.header.frame_id = out.frame_id = self.p["output_frame"]
        out.status_message = reason
        self.result_pub.publish(out)
        self._status("failed", request_id=request["id"], command=request["command"], reason=reason)
        self.pending = None

    def _check_query_deadline(self):
        if self.pending is None:
            raise ValueError("no active visual-memory query")
        if time.monotonic() - self.pending["wall_start"] >= self.p["query_timeout_sec"]:
            raise ValueError("visual-memory query exceeded total time budget")

    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if now < self.last_clock_ns:
            self._fail("simulation clock reset; discarded old visual memory")
            self.memory.clear()
            self._reset_target_history()
            self.catalog_queue.clear()
            for buf in (self.images, self.scans, self.infos, self.odoms, self.imus):
                buf.clear()
            for buf in (getattr(self, "depths", ()), getattr(self, "depth_infos", ())):
                if hasattr(buf, "clear"):
                    buf.clear()
            self.stationary_since = None
            self.epoch += 1
            self._reset_observation_capture()
            if getattr(self, "observation_hints", None) is not None:
                self.observation_hints.clear(self.epoch)
            self._publish_memory()
        self.last_clock_ns = now
        # A completed future may already be late. Reject the request before
        # processing its response, especially gallery responses that select
        # directly rather than accumulating another per-view candidate.
        if self.pending:
            try:
                self._check_query_deadline()
            except ValueError as exc:
                self._fail(str(exc))
        if self.future is not None and self.future.done():
            future, job = self.future, self.job
            self.future = None
            response = {}
            try:
                response = future.result()
                if job["kind"] == "startup":
                    self.worker_ready = True
                    self._status("ready", **{k:v for k,v in response.items() if k != "status"})
                elif job["epoch"] == self.epoch:
                    if job["kind"] == "catalog":
                        self._catalog_result(job, response)
                    elif self.pending and job["request_id"] == self.pending["id"]:
                        if job["kind"] == "gallery":
                            self._gallery_result(job, response)
                        else:
                            self._query_result(job, response)
            except Exception as exc:
                if self.worker is not None and self.worker.process.poll() is not None:
                    self.worker_ready = False
                    self.worker_error = "model worker stopped: " + str(exc)
                    self._fail(self.worker_error)
                if job["kind"] == "startup":
                    self.worker_error = str(exc)
                    self._status("worker_error", reason=str(exc))
                elif job["kind"] == "gallery" and self.pending and job["request_id"] == self.pending["id"]:
                    self._status("gallery_rejected", request_id=job["request_id"],
                                 reason=str(exc), answer=response.get("answer", ""))
                    self._fail("remembered-instance gallery did not yield one reliable target: " + str(exc))
                elif job["kind"] == "query" and self.pending and job["request_id"] == self.pending["id"]:
                    self.pending["errors"].append(str(exc))
                    self._status("query_view_rejected", request_id=job["request_id"],
                                 view_id=job["view"].view_id if job.get("view") else "gallery", reason=str(exc),
                                 answer=response.get("answer", ""))
                else:
                    self._status("observation_rejected", reason=str(exc),
                                 answer=response.get("answer", ""),
                                 supplemental_inventory=response.get("supplemental_inventory", []),
                                 view_id=job.get("view").view_id if job.get("view") else "")
            finally:
                if job["kind"] == "catalog":
                    # A rejected/evicted final observation still drains the
                    # queue; do not leave the latched summary stuck at busy.
                    self._publish_memory()
        if self.future is not None or not self.worker_ready:
            return
        if self.pending:
            try:
                self._step_query()
            except (ValueError, tf2_ros.TransformException) as exc:
                self._fail(str(exc))
            return
        # Process every admitted image before capturing another. Otherwise a
        # full evidence pool can evict its unprocessed view while inference
        # is running, silently discarding both the image and its result.
        if not self.catalog_queue:
            self._observe()
        while self.catalog_queue:
            view, label = self.catalog_queue.popleft()
            if view in self.memory.views:
                self._infer(view, "catalog", label)
                return
            if not self.catalog_queue:
                self._publish_memory()

    def _catalog_slot_available(self):
        return self.future is None and not self.catalog_queue

    def _observe(self):
        if self.pending is not None or not self._catalog_slot_available():
            return
        if self._observe_dedicated():
            return
        if time.monotonic() - self.last_capture_wall < self.p["memory_capture_interval_sec"]:
            return
        self.last_capture_wall = time.monotonic()
        try:
            snapshot = self._capture()
            x, y, yaw = snapshot["odom_pose"]
            image_ns = stamp_ns(snapshot["image"])
            retry_empty = bool(self.memory.views) and not self.memory.landmarks and (
                image_ns - max(v.stamp_ns for v in self.memory.views) > self.p["memory_empty_retry_sec"] * 1e9)
            if self.memory.should_capture(x, y, yaw, image_ns) or retry_empty:
                view = self.memory.add_view(snapshot, stamp_ns(snapshot["image"]), x, y, yaw)
                self._save_view(view)
                self.catalog_queue.append((view, None))
                self._publish_memory()
                self._status("view_stored", view_id=view.view_id,
                             objects=sum(o.confirmed for o in self.memory.landmarks.values()),
                             pending_objects=sum(not o.confirmed for o in self.memory.landmarks.values()),
                             odom_pose=list(view.payload["odom_pose"]),
                             image_stamp_ns=stamp_ns(view.payload["image"]))
        except (ValueError, tf2_ros.TransformException, CvBridgeError, cv2.error) as exc:
            # A missing/misaligned sensor should be visible to the operator,
            # while retaining valid historical memory during sensor startup.
            if time.monotonic() - getattr(self, "last_capture_error_wall", 0.) > 10.:
                self.last_capture_error_wall = time.monotonic()
                self._status("observation_waiting", reason=str(exc))

    def _reset_observation_capture(self):
        self.capture_request = None
        self.capture_requests_seen = {}
        self.capture_requests_consumed = set()
        self.capture_enabled = True
        self.capture_command_cutoff_ns = -1

    def _capture_user_command_cb(self, msg):
        if msg.data.strip():
            self.capture_request = None
            self.capture_enabled = False
            self.capture_command_cutoff_ns = self.get_clock().now().nanoseconds

    def _capture_enabled_cb(self, msg):
        self.capture_enabled = bool(msg.data)
        if not msg.data:
            self.capture_request = None
            self.capture_command_cutoff_ns = self.get_clock().now().nanoseconds

    def _observation_capture_cb(self, msg):
        """Accept only a short-lived request from a completed active look."""
        now = self.get_clock().now().nanoseconds
        if (self.p.get("localization_mode") != "depth" or self.pending is not None
                or not getattr(self, "capture_enabled", True) or now < self.last_clock_ns):
            return
        try:
            request = json.loads(msg.data)
            fields = {"epoch", "request_id", "track_id", "succeeded_stamp_ns", "expires_stamp_ns"}
            if not isinstance(request, dict) or set(request) != fields:
                return
            for key in ("epoch", "succeeded_stamp_ns", "expires_stamp_ns"):
                if type(request[key]) is not int:
                    return
            for key in ("request_id", "track_id"):
                if not isinstance(request[key], str) or not 1 <= len(request[key]) <= 160 or not request[key].strip():
                    return
            succeeded, expires = request["succeeded_stamp_ns"], request["expires_stamp_ns"]
            if (request["epoch"] != self.epoch or not 0 < succeeded <= now < expires
                    or expires-succeeded != 5_000_000_000
                    or succeeded <= getattr(self, "capture_command_cutoff_ns", -1)):
                return
            seen = self.capture_requests_seen
            key = request["request_id"]
            if key in self.capture_requests_consumed or (key in seen and seen[key] != request):
                return
            if key not in seen and len(seen) >= 8:
                return
            seen[key] = dict(request)
            self.capture_request = request
        except (TypeError, ValueError):
            return

    def _capture_source_positions(self, track_id, now):
        obj = self.memory.landmarks.get(track_id)
        if obj is not None and not obj.confirmed and not obj.conflicted:
            return [(view.x, view.y) for view in self.memory.views if view.view_id in obj.view_ids]
        hints = getattr(self, "observation_hints", None)
        if hints is not None:
            for hint in hints.serialize(now):
                if hint["hint_id"] == track_id and hint["status"] == "unresolved":
                    return [tuple(hint["source_pose"][:2])]
        return []

    def _observe_dedicated(self):
        request = getattr(self, "capture_request", None)
        if request is None:
            return False
        now = self.get_clock().now().nanoseconds
        if (request["epoch"] != self.epoch or now < self.last_clock_ns
                or now >= request["expires_stamp_ns"] or not self.capture_enabled):
            self.capture_request = None
            return False
        sources = self._capture_source_positions(request["track_id"], now)
        if not sources:
            self.capture_request = None
            return False
        stationary = self.stationary_since
        earliest = max(now if stationary is None else stationary,
                       request["succeeded_stamp_ns"]) + int(self.p["stationary_seconds"]*1e9)
        if stationary is None or now < earliest:
            return True
        try:
            snapshot = self._capture()
            after_capture = self.get_clock().now().nanoseconds
            if after_capture < now or after_capture >= request["expires_stamp_ns"]:
                self.capture_request = None
                return True
            ns = stamp_ns(snapshot["image"])
            x, y, yaw = snapshot["odom_pose"]
            if (ns < earliest or ns <= max((view.stamp_ns for view in self.memory.views), default=-1)
                    or any(math.dist((x, y), source) < .25 for source in sources)):
                return True
            if "depth_m" not in snapshot:
                raise ValueError("active observation requires synchronized depth")
            snapshot["metadata"]["observation_capture"] = dict(request)
            view = self.memory.add_view(snapshot, ns, x, y, yaw, dedicated_capture=True)
            self.capture_requests_consumed.add(request["request_id"])
            self.capture_request = None
            self._save_view(view)
            self.catalog_queue.append((view, None))
            self._publish_memory()
            self._status("observation_capture_stored", **request, view_id=view.view_id,
                         image_stamp_ns=ns, odom_pose=list(snapshot["odom_pose"]))
        except (ValueError, tf2_ros.TransformException, CvBridgeError, cv2.error):
            # Retry sensor acquisition only within the same five-second window.
            # A successfully stored exposure is consumed before any publication.
            pass
        return True

    def _save_view(self, view):
        if self.p["evidence_dir"]:
            folder = Path(self.p["evidence_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / (view.view_id + "_source.png")).write_bytes(
                base64.b64decode(view.payload["image_png"]))
            snapshot = view.payload
            scan = snapshot["scan"]
            record = {"stamp_ns": view.stamp_ns, "odom_pose": list(snapshot["odom_pose"]),
                      "robot_tilt_deg": snapshot["metadata"].get("robot_tilt_deg"),
                      "calibration": vars(snapshot["calibration"]),
                      "ranges": [float(v) if math.isfinite(v) else None for v in scan.ranges],
                      "angle_min": scan.angle_min, "angle_increment": scan.angle_increment,
                      "range_min": max(.12, scan.range_min), "range_max": min(3.5, scan.range_max)}
            for name in ("camera_to_scan", "scan_to_odom"):
                tf = snapshot[name]
                record[name] = {"translation": vars(tf.translation), "rotation": vars(tf.rotation)}
            if "depth_m" in snapshot:
                np.save(folder / (view.view_id + "_depth.npy"), snapshot["depth_m"], allow_pickle=False)
                tf = snapshot["camera_to_odom"]
                record["camera_to_odom"] = {"translation": vars(tf.translation), "rotation": vars(tf.rotation)}
                record["depth_stamp_ns"] = snapshot["metadata"]["depth_stamp_ns"]
                record["localization_source"] = "aligned_depth"
            (folder / (view.view_id + "_sensors.json")).write_text(json.dumps(record, allow_nan=False))

    def _validate_attitude(self, image_ns):
        # Wheel odometry and its TF stay planar even after the physical robot
        # climbs a low obstacle. Use the actual IMU before attaching any new
        # image/laser observation to a 2-D map.
        if not self.imus:
            raise ValueError("waiting for IMU orientation")
        imu = min(self.imus, key=lambda item: abs(stamp_ns(item)-image_ns))
        if abs(stamp_ns(imu)-image_ns) > self.p["sensor_tolerance_sec"] * 1e9:
            raise ValueError("camera and IMU orientation are not synchronized")
        q = imu.orientation
        norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if imu.orientation_covariance[0] < 0 or not math.isfinite(norm) or norm < 1e-6:
            raise ValueError("IMU orientation unavailable")
        tilt = math.acos(max(-1., min(1., 1 - 2*(q.x*q.x + q.y*q.y)/norm)))
        if tilt > math.radians(self.p["max_robot_tilt_deg"]):
            raise ValueError("robot tilted; cannot localize objects with a planar laser model")
        return math.degrees(tilt)

    def _capture(self):
        if not all((self.images, self.scans, self.infos, self.odoms)):
            raise ValueError("waiting for camera, scan, calibration and odometry")
        now_ns = self.get_clock().now().nanoseconds
        eligible = [item for item in self.images if now_ns - stamp_ns(item) >= 100_000_000]
        if self.p.get("localization_mode", "laser") == "depth":
            # RGB/depth callbacks can arrive in either order. Select the
            # newest complete exposure, instead of pairing the newest RGB
            # with another frame merely because its depth has not arrived.
            eligible = [item for item in eligible if all(
                any(abs(stamp_ns(other)-stamp_ns(item)) <= 5_000_000 for other in buf)
                for buf in (self.depths, self.depth_infos))]
        if not eligible:
            raise ValueError("waiting for synchronized RGB/depth exposure and image-time transforms"
                             if self.p.get("localization_mode", "laser") == "depth"
                             else "waiting for image-time transforms")
        image = eligible[-1]
        ns = stamp_ns(image)
        scan, info, odom = [min(buf, key=lambda item: abs(stamp_ns(item)-ns))
                            for buf in (self.scans, self.infos, self.odoms)]
        check_snapshot_times(now_ns, ns, stamp_ns(scan), stamp_ns(info), stamp_ns(odom),
                             self.p["sensor_max_age_sec"], self.p["sensor_tolerance_sec"])
        tilt_deg = self._validate_attitude(ns)
        if image.width != info.width or image.height != info.height or image.header.frame_id != info.header.frame_id:
            raise ValueError("camera image and calibration differ")
        optical = self.p["optical_frame"] or info.header.frame_id
        calibration = CameraCalibration(info.width, info.height, info.k[0], info.k[4],
                                        info.k[2], info.k[5], optical)
        if validate_camera_calibration(calibration):
            raise ValueError("invalid camera calibration")
        # The nearest laser message can precede or follow the camera image.
        # Express its camera ray in the laser frame at the LASER timestamp;
        # a same-time rigid transform loses rotation between these samples.
        odom_frame = odom.header.frame_id
        camera_to_scan = tf_geometry(self.tf_buffer.lookup_transform_full(
            target_frame=scan.header.frame_id, target_time=Time.from_msg(scan.header.stamp),
            source_frame=optical, source_time=Time.from_msg(image.header.stamp),
            fixed_frame=odom_frame))
        scan_to_map = tf_geometry(self.tf_buffer.lookup_transform(
            self.p["output_frame"], scan.header.frame_id, Time.from_msg(scan.header.stamp)))
        # Store odom coordinates too. Current map->odom corrects old views after
        # SLAM adjusts the map, without attaching today's laser to an old image.
        scan_to_odom = tf_geometry(self.tf_buffer.lookup_transform(
            odom_frame, scan.header.frame_id, Time.from_msg(scan.header.stamp)))
        bgr = self.bridge.imgmsg_to_cv2(image, "bgr8")
        ok, encoded = cv2.imencode(".png", bgr)
        if not ok:
            raise ValueError("PNG encoding failed")
        snapshot = {"image": image, "scan": scan, "calibration": calibration,
                "camera_to_scan": camera_to_scan, "scan_to_map": scan_to_map,
                "scan_to_odom": scan_to_odom, "odom_frame": odom_frame,
                "image_png": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "odom_pose": odom_pose(odom),
                "metadata": {"image_stamp_ns": ns, "image_frame": image.header.frame_id,
                    "robot_tilt_deg": tilt_deg,
                    "scan_stamp_ns": stamp_ns(scan), "scan_frame": scan.header.frame_id,
                    "camera_info_stamp_ns": stamp_ns(info), "optical_frame": optical,
                    "scan_to_map_translation": scan_to_map.translation.as_dict(),
                    "scan_to_map_quaternion": vars(scan_to_map.rotation)}}
        if self.p.get("localization_mode", "laser") == "depth":
            if optical != image.header.frame_id:
                raise ValueError("depth optical frame must match the RGB image frame")
            if not self.depths or not self.depth_infos:
                raise ValueError("waiting for aligned depth and depth calibration")
            depth_msg, depth_info = [min(buf, key=lambda item: abs(stamp_ns(item)-ns))
                                     for buf in (self.depths, self.depth_infos)]
            if any(abs(stamp_ns(item)-ns) > 5_000_000 for item in (depth_msg, depth_info)):
                raise ValueError("RGB and depth are not synchronized to the same exposure")
            if any(item.width != image.width or item.height != image.height or
                   item.header.frame_id != image.header.frame_id for item in (depth_msg, depth_info)):
                raise ValueError("RGB and depth are not pixel aligned")
            if not np.allclose(depth_info.k, info.k, atol=1e-6, rtol=0.):
                raise ValueError("RGB and depth intrinsics differ")
            if any(abs(value) > 1e-8 for item in (info, depth_info) for value in item.d):
                raise ValueError("aligned depth localization requires rectified pinhole images")
            snapshot["depth_m"] = depth_in_metres(
                self.bridge.imgmsg_to_cv2(depth_msg, "passthrough"), depth_msg.encoding)
            snapshot["camera_to_odom"] = tf_geometry(self.tf_buffer.lookup_transform(
                odom_frame, optical, Time.from_msg(image.header.stamp)))
            snapshot["metadata"].update(depth_stamp_ns=stamp_ns(depth_msg),
                                          localization_source="aligned_depth")
        return snapshot

    def _localize(self, view, box, projection=None):
        snapshot = view.payload
        if "depth_m" in snapshot:
            cache = snapshot.setdefault("depth_geometry", {})
            key = tuple(float(v) for v in box)
            if key not in cache:
                cache[key] = localize_depth_box(box, snapshot["calibration"], snapshot["depth_m"],
                                                snapshot["camera_to_odom"])
            odom_point, saved_geometry = cache[key]
            geometry = dict(saved_geometry)
        else:
            if self.p.get("localization_mode", "laser") == "depth":
                raise ValueError("depth-mode observation has no aligned depth; no laser fallback")
            scan = snapshot["scan"]
            odom_point, geometry = localize_box(box, snapshot["calibration"], snapshot["camera_to_scan"],
                snapshot["scan_to_odom"], scan.ranges, scan.angle_min, scan.angle_increment,
                max(0.12, scan.range_min), min(3.5, scan.range_max))
            if "laser_pixel" in geometry:
                validate_surface_visibility(box, geometry["laser_pixel"], snapshot.get("inventory_boxes", []),
                                            geometry.get("support_pixels"))
        current = projection if projection is not None else self._current_projection(snapshot["odom_frame"])
        point = transform_point(tf_geometry(current), odom_point)
        geometry["map_to_odom_stamp_ns"] = stamp_ns(current)
        return point, geometry

    def _infer(self, view, kind, expression):
        request_id = self.pending["id"] if kind == "query" else f"catalog_{view.view_id}_{expression}"
        request = {"request_id": request_id, "operation": "ground" if kind == "query" else "catalog",
                   "command": self.pending["command"] if kind == "query" else "",
                   "expression": expression, "image_png": view.payload["image_png"]}
        if kind == "catalog" and expression is None:
            request["operation"] = "inventory"
            request["categories"] = list(self.p["memory_classes"])
        self.job = {"kind": kind, "epoch": self.epoch, "view": view,
                    "label": expression, "request_id": request_id,
                    "categories": request.get("categories")}
        timeout = self.p["inference_timeout_sec"]
        if kind == "query":
            timeout = min(timeout, self.p["query_timeout_sec"] - (time.monotonic() - self.pending["wall_start"]))
            self._status("inference", request_id=request_id, command=request["command"],
                         view_id=view.view_id, source="visual_memory", **view.payload["metadata"])
        self.future = self.pool.submit(self.worker.infer, request, max(0.1, timeout))

    def _identity_depth_proof(self, view, box, geometry):
        """Bind optional identity correction to this retained exact RGB-D cache.

        It never supplies a class, point or missing depth. Ordinary admission
        remains unchanged when this additional provenance is unavailable.
        """
        try:
            payload, image = view.payload, view.payload['image']
            metadata = payload['metadata']
            cached = payload['depth_geometry'][tuple(box)][1]
            if (self.p.get('localization_mode', 'laser') != 'depth'
                    or not self.memory.confirm_classes or view not in self.memory.views
                    or 'depth_m' not in payload or 'camera_to_odom' not in payload
                    or geometry.get('localization_source') != 'aligned_depth'
                    or metadata.get('localization_source') != 'aligned_depth'
                    or metadata['image_stamp_ns'] != view.stamp_ns
                    or stamp_ns(image) != view.stamp_ns
                    or image.header.frame_id != payload['calibration'].frame_id
                    or box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height
                    or cached['surface_group'] != geometry['surface_group']
                    or cached['identity_connectivity'] != geometry['identity_connectivity']):
                return None
            return dict(epoch=self.epoch, view_id=view.view_id, stamp_ns=view.stamp_ns,
                        depth_stamp_ns=metadata['depth_stamp_ns'], odom_frame=payload['odom_frame'],
                        box=list(box), surface_group=geometry['surface_group'],
                        connectivity=geometry['identity_connectivity'])
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            return None

    def _catalog_result(self, job, response):
        if response.get("request_id") != job["request_id"]:
            raise ValueError("worker request ID mismatch")
        view = job["view"]
        # Capacity backpressure preserves the admitted image, but its normal
        # lifetime still applies if inference or a clock jump makes it old.
        self.memory.views_for_query(self.get_clock().now().nanoseconds)
        if view not in self.memory.views:
            return
        image = view.payload["image"]
        inventory = (parse_inventory_response(response, image.width, image.height, job["categories"])
                     if job.get("categories") else
                     {"observations": [{"label": job["label"], "box": box}
                       for box in catalog_boxes(response["answer"], image.width, image.height)]})
        view.payload["inventory_boxes"] = [item["box"] for item in inventory["observations"]]
        accepted, ambiguous = filter_ambiguous_inventory(inventory["observations"])
        self._refresh_positions()
        localized = {}
        ambiguous_depth = {}
        initially_ambiguous = {item["index"] for item in ambiguous}
        if initially_ambiguous and self.p.get("localization_mode", "laser") == "depth":
            geometries = {}
            for index in initially_ambiguous:
                box = inventory["observations"][index]["box"]
                try:
                    validate_localizable_box(box, image.width, image.height)
                    located = self._localize(view, box)
                    localized[tuple(box)] = located
                    geometries[index] = located[1]
                except AmbiguousDepthError as exc:
                    # Keep all measured alternatives for a nonconfirming joint
                    # class/depth task. They cannot remove a conflict edge.
                    if exc.hint_usable:
                        ambiguous_depth[index] = exc
                except (ValueError, tf2_ros.TransformException):
                    # An unresolved surface cannot disambiguate nested labels.
                    pass
            accepted, ambiguous = filter_ambiguous_inventory(
                inventory["observations"], geometries=geometries)
        depth_resolved = sorted(initially_ambiguous - {item["index"] for item in ambiguous})
        added, rejected, observations = [], [item["reason"] for item in ambiguous], []
        observation_hints = []
        class_hints = self._record_class_hints(view, ambiguous, localized, ambiguous_depth)
        for detection in accepted:
            box, label = detection["box"], detection["label"]
            try:
                validate_localizable_box(box, image.width, image.height)
                cached = localized.get(tuple(box))
                point, geometry = cached if cached is not None else self._localize(view, box)
                class_complete = (class_box_complete(box, image.width, image.height)
                                  if self.p.get("localization_mode", "laser") == "depth" else True)
                obj = self.memory.remember(view, label, box, point.x, point.y,
                                           support_points=geometry.get("support_points"),
                                           class_evidence_complete=class_complete,
                                           identity_depth=self._identity_depth_proof(view, box, geometry))
                added.append(obj.object_id)
                observations.append({"label": label, "box": box,
                                     "map_point": point.as_dict(), **geometry,
                                     "class_evidence_complete": class_complete})
            except AmbiguousDepthError as exc:
                # Preserve the measurement ambiguity as an observation task,
                # never as a guessed target point or a vote in object Memory.
                rejected.append(str(exc))
                hint_id = self._record_depth_hint(view, label, box, exc)
                if hint_id is not None:
                    observation_hints.append(hint_id)
            except (ValueError, tf2_ros.TransformException) as exc:
                rejected.append(str(exc))
        hints = getattr(self, "observation_hints", None)
        if hints is not None:
            hints.reconcile(list(self.memory.landmarks.values()), view.view_id,
                            view.payload["odom_pose"], view.stamp_ns)
        self._status("memory_updated", view_id=view.view_id, label=job["label"],
                     categories=job.get("categories"),
                     unreported_labels=inventory.get("unreported_labels", []),
                     answer=response["answer"], object_ids=added, rejected=rejected,
                     supplemental_inventory=response.get("supplemental_inventory", []),
                     observations=observations,
                     identity_bridge_observations=list(view.bridge_observations),
                     ambiguous_detections=ambiguous,
                     depth_observation_hints=observation_hints,
                     class_observation_hints=class_hints,
                     depth_resolved_indices=depth_resolved,
                     objects=sum(o.confirmed for o in self.memory.landmarks.values()),
                     pending_objects=sum(not o.confirmed for o in self.memory.landmarks.values()),
                     inference_seconds=response["inference_seconds"],
                     generation_mode=response.get("generation_mode", ""))

    def _record_class_hints(self, view, ambiguous, localized, ambiguous_depth=None):
        """Retain all unresolved labels and measured surfaces, never class votes.

        Every member needs either an independently localized surface or all of
        its usable ambiguous depth groups. Ordinary failures provide no proof;
        missing/clipped evidence and overflow reject the whole task. Nothing
        here removes a conflict edge or makes a surface a selectable object.
        """
        if self.p.get("localization_mode", "laser") != "depth":
            return []
        ambiguous_depth = ambiguous_depth or {}
        by_index = {item["index"]: item for item in ambiguous}
        components = sorted({tuple(item["ambiguous_indices"]) for item in ambiguous})
        recorded = []
        for indices in components:
            items = [by_index[index] for index in indices]
            groups = []
            for item in items:
                error = ambiguous_depth.get(item["index"])
                if error is not None:
                    if (not error.hint_usable or error.box != tuple(item["box"])
                            or error.group_count != len(error.groups)
                            or not 2 <= len(error.groups) <= 8):
                        break
                    measured = error.groups
                else:
                    location = localized.get(tuple(item["box"]))
                    geometry = location[1] if location is not None else {}
                    group = geometry.get("surface_group")
                    if geometry.get("localization_source") != "aligned_depth" or group is None:
                        break
                    measured = [group]
                # Deduplicate only identical measurements, without choosing a
                # class or merging nearby but distinct surface hypotheses.
                for group in measured:
                    if group not in groups:
                        groups.append(group)
                if len(groups) > 8:
                    break
            else:
                context = self._hint_context(view)
                if context is None:
                    continue
                box = [min(item["box"][0] for item in items),
                       min(item["box"][1] for item in items),
                       max(item["box"][2] for item in items),
                       max(item["box"][3] for item in items)]
                try:
                    hint_id = self.observation_hints.add_class(
                        [item["label"] for item in items], view.view_id, view.stamp_ns,
                        view.payload["odom_pose"], context, box, groups,
                        list(self.memory.landmarks.values()))
                except (KeyError, TypeError, ValueError):
                    # Bounded provenance is mandatory; reject the entire hint
                    # rather than dropping a competing class or surface.
                    continue
                if hint_id is not None:
                    recorded.append(hint_id)
        return recorded

    def _hint_context(self, view):
        snapshot = view.payload
        if snapshot.get("odom_frame", "odom") != "odom" or "camera_to_odom" not in snapshot:
            return None
        if getattr(self, "observation_hints", None) is None:
            self.observation_hints = HintMemory(
                max_age_sec=self.p.get("memory_max_age_sec", 1800.), epoch=self.epoch)
        origin = transform_point(snapshot["camera_to_odom"], Vector3(0., 0., 0.))
        forward = transform_point(snapshot["camera_to_odom"], Vector3(0., 0., 1.))
        return [origin.x, origin.y, math.atan2(forward.y-origin.y, forward.x-origin.x)]

    def _record_depth_hint(self, view, label, box, error):
        if not error.hint_usable:
            return None
        source_camera = self._hint_context(view)
        if source_camera is None:
            return None
        return self.observation_hints.add(label, view.view_id, view.stamp_ns,
                                         view.payload["odom_pose"], source_camera, box,
                                         error.groups, list(self.memory.landmarks.values()))

    def _infer_gallery(self, gallery):
        self._check_query_deadline()
        self._check_query_motion()
        request = self.pending
        request["gallery_attempted"] = True
        request["views"] = deque()
        request["fallback_views"] = deque()
        # Joint candidates let the model compare colors/appearance across past
        # viewpoints. Multi-instance decoding permits None or ambiguity rather
        # than forcing a target independently in every partial view.
        message = {"request_id": request["id"], "operation": "catalog",
                   "command": request["command"],
                   "expression": referring_expression(request["command"]),
                   "image_png": gallery.image_png}
        self.job = {"kind": "gallery", "epoch": self.epoch,
                    "request_id": request["id"], "gallery": gallery}
        self._status("inference", request_id=request["id"], command=request["command"],
                     source="visual_memory_gallery", object_ids=[e.object_id for e in gallery.entries],
                     excluded_objects=gallery.excluded_objects)
        if self.p["evidence_dir"]:
            folder = Path(self.p["evidence_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / ("command_%03d_gallery_source.png" % request["id"])).write_bytes(
                base64.b64decode(gallery.image_png))
        timeout = min(self.p["inference_timeout_sec"], self.p["query_timeout_sec"] -
                      (time.monotonic() - request["wall_start"]))
        self.future = self.pool.submit(self.worker.infer, message, max(.1, timeout))

    def _gallery_result(self, job, response):
        self._check_query_deadline()
        if response.get("request_id") != self.pending["id"]:
            raise ValueError("worker request ID mismatch")
        self._check_query_motion()
        gallery = job["gallery"]
        height, width = gallery.image_bgr.shape[:2]
        any_instance = allows_any_instance(self.pending["command"])
        parsed = parse_locateanything_output(response["answer"], width, height)
        if not parsed["parse_success"]:
            raise ValueError("malformed model output: " + ",".join(parsed["errors"]))
        boxes = parsed["pixel_boxes"]
        if not boxes:
            raise ValueError("LocateAnything found no matching instance")
        candidates = {}
        for box in boxes:
            candidate = self._gallery_candidate(job, box)
            candidates.setdefault(candidate[4].object_id, candidate)
        choices = list(candidates.values())
        if any_instance:
            view, box, point, geometry, obj = random.choice(choices)
            geometry = dict(geometry, selection_policy="random_model_matches",
                            random_candidate_ids=list(candidates))
        elif len(choices) > 1:
            view, box, point, geometry, obj = self._nearest_gallery_candidate(choices)
        else:
            view, box, point, geometry, obj = choices[0]
        obj.descriptions.add(self.pending["command"])
        self._select(view, box, point, geometry, obj, response, "gallery_description")

    def _nearest_gallery_candidate(self, candidates):
        """Break a nonspatial model-match tie by current straight-line distance.

        Class/affordance matching belongs to LA. Distance only chooses among
        its validated matches; neither a nearer nonmatch nor a pending track
        may enter this set. Reproject all evidence and the robot together so
        a SLAM correction cannot change the coordinate system mid-comparison.
        """
        return self._distance_candidate(candidates)

    def _distance_candidate(self, candidates, *, farthest=False,
                            policy="nearest_model_matches", minimum_gap=0.0):
        """Compare every candidate using one fresh map/odom projection."""
        self._check_query_deadline()
        self._check_query_motion()
        odometry = self.odoms[-1]
        frame = odometry.header.frame_id
        if not frame or any(view.payload["odom_frame"] != frame
                            for view, *_ in candidates):
            raise ValueError("model matches and robot must share one odometry frame")
        projection = self._current_projection(frame)
        position = odometry.pose.pose.position
        robot = transform_point(tf_geometry(projection), Vector3(position.x, position.y, position.z))
        if not all(math.isfinite(value) for value in robot.as_dict().values()):
            raise ValueError("current robot position is not finite")
        current_candidates, distances = [], {}
        for view, box, _point, geometry, obj in candidates:
            self._check_query_deadline()
            self._check_query_motion()
            point, current_geometry = self._localize(view, box, projection)
            distance = math.hypot(point.x-robot.x, point.y-robot.y)
            if not math.isfinite(distance):
                raise ValueError("model match has no finite navigation distance")
            distances[obj.object_id] = distance
            current_candidates.append((view, box, point, {**geometry, **current_geometry}, obj))
        ordered = sorted(current_candidates,
                         key=lambda item: (distances[item[4].object_id], item[4].object_id),
                         reverse=farthest)
        if len(ordered) > 1 and minimum_gap > 0:
            gap = abs(distances[ordered[0][4].object_id] - distances[ordered[1][4].object_id])
            if gap <= minimum_gap:
                raise ValueError("candidate distances are too similar for a reliable near/far comparison")
        chosen = ordered[0]
        view, box, point, geometry, obj = chosen
        geometry = dict(geometry, selection_policy=policy,
                        model_candidate_ids=sorted(distances), candidate_distances_m=distances,
                        selection_robot_map=robot.as_dict(), selection_distance_kind="straight_line_xy")
        return view, box, point, geometry, obj

    def _select_memory_relation(self, views, relation):
        kind, label = relation
        objects = [obj for obj in self.memory.landmarks.values()
                   if obj.label == label or label in obj.seen_known_labels]
        if any(not obj.confirmed or obj.conflicted or obj.label != label for obj in objects):
            raise ValueError("relational candidates have unconfirmed or conflicting class evidence; "
                             "continue exploration")
        excluded = None
        if kind == "another":
            previous = self.memory.landmarks.get(self.last_reached)
            if previous is None or not previous.confirmed or previous.conflicted or previous.label != label:
                raise ValueError("another requires a previous successfully reached object of the same "
                                 "class in the current memory; reach an explicit object first")
            excluded = previous.object_id
            objects = [obj for obj in objects if obj.object_id != excluded]
        if not objects or (kind != "another" and len(objects) < 2):
            raise ValueError("not enough confirmed instances for this relation; continue exploration")
        candidates = self._memory_candidates(views, objects)
        if len(candidates) != len(objects):
            raise ValueError("not all relational candidates have usable retained geometry; "
                             "continue exploration")
        view, box, point, geometry, obj = self._distance_candidate(
            candidates, farthest=kind == "farthest", policy=kind + "_confirmed_class",
            minimum_gap=(0.0 if kind == "another" else self.p.get("relation_distance_margin_m", .15)))
        geometry["relation_candidate_ids"] = geometry.pop("model_candidate_ids")
        geometry.update(relation=kind, requested_class=label, excluded_object_id=excluded,
                        comparison_scope="confirmed instances in current visual memory")
        obj.descriptions.add(self.pending["command"])
        self._select(view, box, point, geometry, obj,
                     {"answer": "measured relation over confirmed visual-memory instances",
                      "prompt": "", "inference_seconds": 0.0}, "memory_relation")

    def _gallery_candidate(self, job, box):
        """Resolve and validate one model box before any instance is chosen."""
        self._check_query_deadline()
        self._check_query_motion()
        gallery = job["gallery"]
        part_evidence = None
        try:
            entry = select_gallery_box(gallery, box)
        except ValueError:
            entry, part_box = select_gallery_part(gallery, box)
            if job["epoch"] != self.epoch:
                raise ValueError("gallery part belongs to a superseded memory epoch")
            if "depth_m" not in entry.view.payload:
                raise ValueError("gallery part requires original aligned depth; no laser fallback")
            try:
                _, part_geometry = self._localize(entry.view, part_box)
            except AmbiguousDepthError as exc:
                # All measured groups must prove the same retained identity.
                # They are not alternative target coordinates to choose from.
                part_evidence = validate_gallery_part_groups(
                    self.memory, entry, exc, part_box, self.get_clock().now().nanoseconds)
                part_geometry = {"localization_source": "aligned_depth", "depth_ambiguity": True,
                                 "group_count": exc.group_count, "surface_groups": list(exc.groups)}
            try:
                if part_evidence is None:
                    part_evidence = validate_gallery_part(
                        self.memory, entry, part_geometry, self.get_clock().now().nanoseconds)
            except ValueError as exc:
                self._status("gallery_part_rejected", request_id=self.pending["id"],
                             view_id=entry.view.view_id, image_stamp_ns=entry.view.stamp_ns,
                             source_part_box_xyxy=part_box, gallery_box_xyxy=box,
                             depth_geometry=part_geometry, reason=str(exc))
                raise
            part_evidence.update(source_part_box_xyxy=part_box, gallery_box_xyxy=box,
                                 source_object_box_xyxy=list(entry.source_box),
                                 source_crop_xyxy=list(entry.source_crop),
                                 content_box_xyxy=list(entry.content_box), depth_geometry=part_geometry)
            self._check_query_deadline()
            self._check_query_motion()
        obj = self.memory.landmarks.get(entry.object_id)
        if obj is None or not obj.confirmed or obj.conflicted:
            raise ValueError("gallery instance expired, unconfirmed or conflicted before selection")
        point, geometry = self._localize(entry.view, entry.source_box)
        geometry["gallery_box_xyxy"] = box
        geometry["gallery_object_ids"] = [e.object_id for e in gallery.entries]
        geometry["gallery_excluded_objects"] = gallery.excluded_objects
        geometry["gallery_tile_box_xyxy"] = entry.tile_box
        if part_evidence is not None:
            # Part pixels prove identity only. Preserve the original full-box
            # navigation target and its frozen sensor provenance.
            geometry["gallery_part_evidence"] = part_evidence
        return entry.view, entry.source_box, point, geometry, obj

    def _memory_candidates(self, views, objects):
        """One usable observation per confirmed instance, regardless of view count."""
        remaining = {obj.object_id: obj for obj in objects if obj.confirmed and not obj.conflicted}
        candidates = []
        for view in views:
            for observation in view.observations:
                obj = remaining.get(observation["object_id"])
                if obj is None:
                    continue
                if (self.memory.confirm_classes and obj.label not in
                        observation.get("labels", [obj.label])):
                    continue
                self._check_query_deadline()
                self._check_query_motion()
                box = observation["box"]
                try:
                    point, geometry = self._localize(view, box)
                except (ValueError, tf2_ros.TransformException):
                    # One bad historical view must not hide a usable later one.
                    continue
                candidates.append((view, box, point, geometry, obj))
                del remaining[obj.object_id]
        return candidates

    def _step_query(self):
        self._check_query_deadline()
        request = self.pending
        if request.get("gallery_attempted"):
            raise ValueError("gallery query ended without a reliable target")
        now = self.get_clock().now().nanoseconds
        if request["views"] is None:
            settled = self.stationary_since is not None and now - max(self.stationary_since, request["stamp_ns"]) >= self.p["stationary_seconds"] * 1e9
            if not settled or not self.odoms or abs(now - stamp_ns(self.odoms[-1])) > self.p["sensor_max_age_sec"] * 1e9:
                if time.monotonic() - request["wall_start"] > self.p["capture_timeout_sec"]:
                    raise ValueError("robot did not settle with fresh odometry before query timeout")
                return
            request["query_pose"] = odom_pose(self.odoms[-1])
            views = self.memory.views_for_query(now)
            self._refresh_positions()
            # A query may retain one new image for later cataloguing, but must
            # not evict another query's still-unprocessed image.
            try:
                if self._catalog_slot_available():
                    snapshot = self._capture()
                    x, y, yaw = snapshot["odom_pose"]
                    if self.memory.should_capture(x, y, yaw, stamp_ns(snapshot["image"])):
                        view = self.memory.add_view(snapshot, stamp_ns(snapshot["image"]), x, y, yaw)
                        self._save_view(view)
                        self.catalog_queue.append((view, None))
                        self._publish_memory()
                        views = self.memory.views_for_query(now)
            except (ValueError, tf2_ros.TransformException, CvBridgeError, cv2.error):
                pass
            # Explicit IDs identify a stored instance; ordinary descriptions
            # keep their wording for LocateAnything visual grounding.
            obj = (self.memory.lookup_id(request["command"])
                   if self.p.get("enable_explicit_id_lookup", True) else None)
            if obj is not None:
                candidates = self._memory_candidates(views, [obj])
                if not candidates:
                    raise ValueError("requested object has no usable retained visual observation")
                self._select(*candidates[0], {"answer": "memory ID lookup",
                             "prompt": "", "inference_seconds": 0.0}, "object_id")
                return
            if not views:
                raise ValueError("no synchronized visual observation is available")
            requested = described_class(request["command"], self.memory.class_labels)
            relation = memory_relation(request["command"], self.memory.class_labels)
            if relation is not None:
                self._select_memory_relation(views, relation)
                return
            if not has_spatial_relation(request["command"]):
                try:
                    # Let LA assess usable confirmed objects even if another
                    # track is pending. LA supplies the matching subset; an
                    # incomplete inventory must not block semantic inference.
                    gallery = build_gallery(self.memory, requested, allow_partial=True)
                except ValueError as exc:
                    raise ValueError("cannot build remembered-instance gallery: " + str(exc)) from exc
                self._infer_gallery(gallery)
                return
            if self.p.get("localization_mode", "laser") == "depth":
                # Pending-only views otherwise become fallback and are skipped
                # when a confirmed view yields a box. They can still change a
                # relation's answer. Unlike a nonspatial gallery query, this
                # path still requires complete relevant class evidence.
                relevant = [obj for obj in self.memory.landmarks.values()
                            if requested is None or requested == obj.label
                            or requested in obj.seen_known_labels]
                pending = sorted(obj.object_id for obj in relevant if not obj.confirmed)
                if pending:
                    raise ValueError("unconfirmed class evidence in relational candidates: "
                                     + ", ".join(pending) + "; continue exploration from different positions")
                conflicts = sorted(obj.object_id for obj in relevant if obj.conflicted)
                if conflicts:
                    raise ValueError("conflicting class evidence in relational candidates: "
                                     + ", ".join(conflicts)
                                     + "; clear memory and re-observe with more reliable localization")
            primary, fallback = self.memory.representative_views(now, self.p["query_max_views"])
            request["views"] = deque(primary)
            request["fallback_views"] = deque(fallback)
        self._check_query_motion()
        if not request["views"] and not request["candidates"] and request.get("fallback_views"):
            request["views"] = request["fallback_views"]
            request["fallback_views"] = deque()
        if request["views"]:
            self._infer(request["views"].popleft(), "query", referring_expression(request["command"]))
            return
        candidates = request["candidates"]
        if not candidates:
            raise ValueError("target not reliably grounded in retained views: " + "; ".join(request["errors"][-3:]))
        # SLAM may have corrected map->odom while several model calls ran.
        # Compare/release targets in the current map, not cached map positions
        # computed at different inference completion times.
        candidates = [(view, box, *self._localize(view, box), response)
                      for view, box, _point, _geometry, response in candidates]
        if allows_any_instance(request["command"]):
            matches = {}
            for candidate in candidates:
                obj = self.memory.match_box(candidate[0], candidate[1])
                if obj is not None and obj.confirmed and not obj.conflicted:
                    matches.setdefault(obj.object_id, candidate)
            if not matches:
                raise ValueError("no model-matched confirmed instance in retained views")
            view, box, point, geometry, response = random.choice(list(matches.values()))
            geometry = dict(geometry, selection_policy="random_model_matches",
                            random_candidate_ids=list(matches))
        else:
            first = candidates[0]
            if any(math.hypot(item[2].x-first[2].x, item[2].y-first[2].y) > 0.6 for item in candidates[1:]):
                raise ValueError("retained views gave conflicting targets; use a more specific description or object ID")
            view, box, point, geometry, response = first
        obj = self.memory.match_box(view, box)
        if self.p.get("localization_mode", "laser") == "depth":
            if obj is None or not obj.confirmed:
                raise ValueError("object class is not confirmed from independent viewpoints; continue exploration")
            if obj.conflicted:
                raise ValueError("object has conflicting category evidence; clear memory and re-observe with more reliable localization")
            # A referring-expression result selects an existing identity. It
            # supplies no new class evidence: calling remember with obj.label
            # would backfill that class into a differently labelled old view.
            obj.descriptions.add(request["command"])
        else:
            obj = self.memory.remember(view, obj.label if obj else "object", box,
                                       point.x, point.y, description=request["command"],
                                       support_points=geometry.get("support_points"))
        self._select(view, box, point, geometry, obj, response, "description")

    def _check_query_motion(self):
        now = self.get_clock().now().nanoseconds
        if self.pending["robot_moved"] or not self.odoms or abs(now - stamp_ns(self.odoms[-1])) > self.p["sensor_max_age_sec"] * 1e9:
            raise ValueError("robot moved or odometry became stale during query")

    def _query_result(self, job, response):
        if response.get("request_id") != self.pending["id"]:
            raise ValueError("worker request ID mismatch")
        self._check_query_motion()
        view = job["view"]
        image = view.payload["image"]
        box = one_box(response["answer"], image.width, image.height)
        requested = described_class(self.pending["command"], self.memory.class_labels)
        observed = self.memory.match_box(view, box)
        if requested and observed and observed.label not in (requested, "object"):
            raise ValueError(f"grounded box matches remembered {observed.label}, not requested {requested}")
        point, geometry = self._localize(view, box)
        self.pending["candidates"].append((view, box, point, geometry, response))
        self._status("query_candidate", request_id=self.pending["id"], view_id=view.view_id,
                     answer=response["answer"], box_xyxy=box, target_map=point.as_dict())

    def _select(self, view, box, point, geometry, obj, response, query_kind):
        self._check_query_deadline()
        if not obj.confirmed:
            raise ValueError("object class is not confirmed from independent viewpoints; continue exploration")
        if obj.conflicted:
            raise ValueError("object has conflicting category evidence; clear memory and re-observe with more reliable localization")
        request, image = self.pending, view.payload["image"]
        self._check_query_motion()
        self._show_box(image, box, request)
        out = SemanticQueryResult()
        out.header.stamp = image.header.stamp
        out.header.frame_id = out.frame_id = self.p["output_frame"]
        out.success, out.query_text = True, request["command"]
        out.semantic_name = out.detector_label = obj.label
        out.object_id = obj.object_id
        out.position.x, out.position.y, out.position.z = point.x, point.y, 0.0
        out.confidence = 1.0  # transport placeholder, NOT a model confidence
        out.status_message = (f"selected from visual memory {view.view_id}; confidence is transport-only; "
                              f"selection_token={self.selection_session}:{self.epoch}:{request['id']}")
        marker = Marker()
        marker.header = out.header
        marker.ns, marker.id, marker.type, marker.action = "locateanything", 0, Marker.SPHERE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = point.x, point.y, 0.35
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.25
        marker.color.r, marker.color.g, marker.color.a = 1.0, 1.0, 1.0
        age = (self.get_clock().now().nanoseconds - stamp_ns(image)) / 1e9
        self._publish_memory()
        # Debug image encoding and refreshing stored map positions can take
        # time. Recheck before publishing a successful target or marker.
        self._check_query_deadline()
        self._check_query_motion()
        self.marker_pub.publish(marker)
        # Capture and localization both describe the sensor source. Merge their
        # fields first so shared keys cannot abort a successful depth query.
        sensor_details = {**view.payload["metadata"], **geometry}
        self._status("selected", request_id=request["id"], command=request["command"],
            **sensor_details, answer=response["answer"], prompt=response["prompt"],
            box_xyxy=box, target_map=point.as_dict(), object_id=obj.object_id,
            inference_seconds=response["inference_seconds"], observation_age_sec=age,
            view_id=view.view_id,
            source="visual_memory_gallery" if query_kind == "gallery_description" else "visual_memory",
            query_kind=query_kind,
            historical_view=stamp_ns(image) < request["stamp_ns"],
            query_robot_odom=list(request["query_pose"]), observation_robot_odom=list(view.payload["odom_pose"]),
            confidence_kind="transport_placeholder_not_model_confidence")
        self.issued_selections[out.status_message] = (request["id"], self.epoch, obj.object_id, request["command"])
        while len(self.issued_selections) > 32:
            del self.issued_selections[next(iter(self.issued_selections))]
        self.result_pub.publish(out)
        self.pending = None

    def _show_box(self, image, box, request):
        source = self.bridge.imgmsg_to_cv2(image, "bgr8")
        annotated = source.copy()
        x1, y1, x2, y2 = [round(value) for value in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(annotated, request["command"], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        self.debug_image = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        self.debug_image.header = image.header
        self.debug_pub.publish(self.debug_image)
        if self.p["evidence_dir"]:
            folder = Path(self.p["evidence_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / ("command_%03d_source.png" % request["id"])), source)
            cv2.imwrite(str(folder / ("command_%03d_box.png" % request["id"])), annotated)

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
