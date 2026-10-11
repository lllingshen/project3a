"""Bounded proposals for a new view of pending visual-memory tracks.

This node does not own a Nav2 action client, publish velocity, or change Memory.
The frontier explorer decides when geometric exploration has no useful goal
and may then dispatch one proposal. All budgets count its dispatched events.
"""
from collections import OrderedDict
import json
import math
from statistics import median
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                      qos_profile_sensor_data, qos_profile_system_default)
from rclpy.time import Time
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PolygonStamped
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, String

from .vertical_framing import VerticalFraming
from .pending_framing import framing_xyz
from .observation_planner import (CameraFraming, Grid, choose_observation_candidate,
                                  CLEAR_VISIBILITY, UNKNOWN_VISIBILITY, convex_footprint,
                                  _inside_footprint, traversable_cost)

from .observation_novelty import SearchProgress, ranked_hint_groups

SECOND = 1_000_000_000
TERMINAL = {"rejected", "succeeded", "aborted", "canceled", "timeout", "paused"}


class RevisitState:
    """Pure bookkeeping: proposals are not attempts until actually dispatched."""

    def __init__(self, enabled=True, namespace=None):
        self.namespace = namespace or uuid.uuid4().hex[:12]
        self.epoch, self.sequence, self.last_ns = 0, 0, None
        self.enabled = enabled
        self._reset()

    def _reset(self):
        self.records = OrderedDict()
        self.proposal = self.active = None
        self.attempts, self.attempted_odom = {}, {}
        self.uncertain_dispatched = {}
        self.focus_dispatched = {}
        self.first_dispatch_ns, self.dwell_until_ns = None, 0
        self.pending_ids = set()
        self.cancel_requested = False
        self.capture = None

    def clock(self, now):
        rolled = self.last_ns is not None and now < self.last_ns
        if rolled:
            self.epoch += 1
            self._reset()
        self.last_ns = now
        return rolled

    @property
    def total_attempts(self):
        return sum(self.attempts.values())

    def budget_exhausted(self, now):
        return self.total_attempts >= 8 or self.time_exhausted(now)

    def time_exhausted(self, now):
        return self.first_dispatch_ns is not None and now-self.first_dispatch_ns >= 120*SECOND

    def withdraw(self):
        self.proposal = None
        self.capture = None
        if self.active:
            self.cancel_requested = True

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if not self.enabled:
            self.withdraw()

    def reconcile(self, pending_ids, now):
        """Return whether a previously proposed/active request needs clearing."""
        self.pending_ids = set(pending_ids)
        if self.capture and (not self.enabled or
                             self.capture["track_id"] not in self.pending_ids):
            self.capture = None
        invalid = (not self.enabled or self.budget_exhausted(now))
        if self.proposal and (invalid or self.records[self.proposal]["track_id"] not in self.pending_ids):
            self.proposal = None
            return True
        # Reaching the dispatch-count limit prevents another goal; it must not
        # cancel the valid eighth goal immediately after it was dispatched.
        if self.active and (not self.enabled or self.time_exhausted(now)
                            or self.records[self.active]["track_id"] not in self.pending_ids):
            if not self.cancel_requested:
                self.cancel_requested = True
                return True
        return False

    def ready(self, track_id, now):
        return (self.enabled and not self.active and now >= self.dwell_until_ns
                and not self.budget_exhausted(now) and track_id in self.pending_ids
                and self.attempts.get(track_id, 0) < 2)

    def current(self, now):
        if self.proposal:
            record = self.records[self.proposal]
            if now-record["stamp_ns"] < 2*SECOND:
                return self.payload(record)
            self.proposal = None
        return None

    @staticmethod
    def payload(record):
        return {key: record[key] for key in
                ("request_id", "track_id", "frame_id", "stamp_ns", "x", "y", "yaw")}

    def offer(self, track_id, pose, odom_xy, now, visibility_reason=CLEAR_VISIBILITY,
              map_sightline_groups=(), focus_group_index=None, hypothesis_group_count=0):
        if ((focus_group_index is None and hypothesis_group_count != 0)
                or (focus_group_index is not None and
                    (type(focus_group_index) is not int or type(hypothesis_group_count) is not int
                     or not 1 <= hypothesis_group_count <= 8
                     or not 0 <= focus_group_index < hypothesis_group_count))):
            return None
        if (not self.ready(track_id, now)
                or visibility_reason not in (CLEAR_VISIBILITY, UNKNOWN_VISIBILITY)
                or (visibility_reason == UNKNOWN_VISIBILITY
                    and self.uncertain_dispatched.get(track_id, 0) >= 1)):
            return None
        current = self.current(now)
        if current:
            return current
        self.sequence += 1
        request_id = f"observation-{self.namespace}-{self.epoch}-{self.sequence}"
        record = dict(request_id=request_id, track_id=track_id, frame_id="map",
                      stamp_ns=now, x=float(pose[0]), y=float(pose[1]), yaw=float(pose[2]),
                      odom_xy=tuple(odom_xy), dispatched=False, done=False,
                      visibility_reason=visibility_reason,
                      focus_group_index=focus_group_index,
                      hypothesis_group_count=hypothesis_group_count,
                      map_sightline_groups=list(map_sightline_groups))
        self.records[request_id] = record
        self.proposal = request_id
        # C++ rejects requests older than three seconds; this bounded history
        # additionally retains many expired proposals for delayed event delivery.
        while len(self.records) > 256:
            oldest = next(key for key in self.records if key not in (self.active, self.proposal))
            del self.records[oldest]
        return self.payload(record)

    def cached_positions(self, track_id):
        """Recent unattempted sensor-planned positions, not reusable goals.

        Keep odom coordinates so a SLAM correction cannot move the physical
        candidate. The caller must replan its heading and recheck every guard.
        Expiry/protocol rejection consumes neither an attempt nor a position.
        """
        points = []
        excluded = self.attempted_odom.get(track_id, [])
        for record in reversed(self.records.values()):
            if record["track_id"] != track_id or record["dispatched"]:
                continue
            point = record["odom_xy"]
            if (any(math.dist(point, old) < .65 for old in excluded)
                    or any(math.dist(point, old) < .05 for old in points)):
                continue
            points.append(point)
            if len(points) == 32:
                break
        return points

    def capture_request(self, now, memory_epoch):
        """Repeat one successful request only within its fixed settling window."""
        if self.capture and (type(memory_epoch) is not int or memory_epoch != self.capture["epoch"]):
            self.capture = None  # An old success never adopts a new Memory epoch.
        if (self.capture is None or not self.enabled or
                not self.capture["succeeded_stamp_ns"] <= now < self.capture["expires_stamp_ns"]):
            return None
        return dict(self.capture)

    def event(self, message, now, memory_epoch=None):
        request_id = message.get("request_id")
        record = self.records.get(request_id)
        kind = message.get("event")
        if (record is None or message.get("track_id") != record["track_id"]
                or kind not in TERMINAL | {"dispatched", "accepted"}):
            return False
        track = record["track_id"]
        if kind == "dispatched":
            if not record["dispatched"]:
                self.capture = None
                record["dispatched"] = True
                self.attempts[track] = self.attempts.get(track, 0)+1
                focus = record.get("focus_group_index")
                if focus is not None:
                    self.focus_dispatched.setdefault(track, []).append(focus)
                if record["visibility_reason"] == UNKNOWN_VISIBILITY:
                    self.uncertain_dispatched[track] = self.uncertain_dispatched.get(track, 0)+1
                self.attempted_odom.setdefault(track, []).append(record["odom_xy"])
                if self.first_dispatch_ns is None:
                    sent = message.get("stamp_ns", now)
                    self.first_dispatch_ns = (sent if isinstance(sent, int)
                        and record["stamp_ns"] <= sent <= now else now)
            if not record["done"]:
                self.active, self.proposal = request_id, None
                self.cancel_requested = False
        elif kind == "accepted" and not record["done"]:
            # Normally follows dispatched; pin it even if callbacks arrive late.
            self.active, self.proposal = request_id, None
        elif kind in TERMINAL and not record["done"]:
            record["done"] = True
            if kind == "rejected" and message.get("reason_code") == "geometry":
                points = self.attempted_odom.setdefault(track, [])
                if record["odom_xy"] not in points:
                    points.append(record["odom_xy"])
            if kind == "succeeded" and record["dispatched"]:
                self.dwell_until_ns = max(self.dwell_until_ns, now+5*SECOND)
                if self.enabled and type(memory_epoch) is int and memory_epoch >= 0:
                    self.capture = {"epoch": memory_epoch, "request_id": request_id,
                                    "track_id": track, "succeeded_stamp_ns": now,
                                    "expires_stamp_ns": now+5*SECOND}
            if self.active == request_id:
                self.active, self.cancel_requested = None, False
            if self.proposal == request_id:
                self.proposal = None
        return True

    def status(self, now):
        if not self.enabled:
            state = "paused"
        elif self.active:
            state = "canceling" if self.cancel_requested else "observing"
        elif now < self.dwell_until_ns:
            state = "settling"
        elif self.budget_exhausted(now):
            state = "budget_exhausted"
        elif not self.pending_ids:
            state = "no_pending_tracks"
        elif not any(self.attempts.get(track, 0) < 2 for track in self.pending_ids):
            state = "track_budgets_exhausted"
        else:
            state = "ready"
        return {"state": state, "epoch": self.epoch, "pending_tracks": len(self.pending_ids),
                "incomplete": bool(self.pending_ids), "attempts": dict(self.attempts),
                "total_dispatched": self.total_attempts, "global_attempt_limit": 8,
                "per_track_attempt_limit": 2, "time_limit_sec": 120,
                "uncertain_visibility_limit_per_task": 1,
                "uncertain_dispatched_by_task": dict(self.uncertain_dispatched),
                "focus_dispatched_by_task": {key: list(value) for key, value in self.focus_dispatched.items()},
                "active_focus_group_index": self.records[self.active].get("focus_group_index") if self.active else None,
                "proposal_focus_group_index": self.records[self.proposal].get("focus_group_index") if self.proposal else None,
                "hypothesis_framing_scope": "Only the dispatched focus group must fit; other groups remain unresolved measurements.",
                "active_visibility_reason": self.records[self.active]["visibility_reason"] if self.active else "",
                "proposal_visibility_reason": self.records[self.proposal]["visibility_reason"] if self.proposal else "",
                "elapsed_since_first_dispatch_sec": None if self.first_dispatch_ns is None
                    else max(0., (now-self.first_dispatch_ns)/SECOND),
                "active_request_id": self.active or "", "proposal_request_id": self.proposal or "",
                "scope": "Shared pending-track and depth-hypothesis observation budget; no claim that the full scene is inventoried."}


def yaw_of(q):
    values = (q.x, q.y, q.z, q.w)
    norm = math.sqrt(sum(v*v for v in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid transform rotation")
    x, y, z, w = (v/norm for v in values)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def transform_xy(point, transform, inverse=False):
    x, y = map(float, point)
    t, angle = transform.translation, yaw_of(transform.rotation)
    if not all(math.isfinite(v) for v in (x, y, t.x, t.y)):
        raise ValueError("non-finite observation coordinate")
    c, s = math.cos(angle), math.sin(angle)
    if inverse:
        x, y = x-t.x, y-t.y
        return c*x+s*y, -s*x+c*y
    return c*x-s*y+t.x, s*x+c*y+t.y


def footprint_stamp_ns(stamp):
    if stamp.sec < 0 or not 0 <= stamp.nanosec < SECOND:
        return -1
    return stamp.sec*SECOND+stamp.nanosec


def transform_xyz(point, transform, inverse=False):
    """Full rigid transform for measured footprint vertices, including tilt."""
    p = tuple(float(v) for v in point)
    t = tuple(float(getattr(transform.translation, axis)) for axis in ("x", "y", "z"))
    q = tuple(float(getattr(transform.rotation, axis)) for axis in ("x", "y", "z", "w"))
    norm = math.sqrt(sum(v*v for v in q))
    if (len(p) != 3 or not all(math.isfinite(v) for v in p+t+q)
            or not math.isfinite(norm) or abs(norm*norm-1.) > 1e-3):
        raise ValueError("invalid footprint rigid transform")
    qx, qy, qz, qw = (v/norm for v in q)
    if inverse:
        p = tuple(a-b for a, b in zip(p, t))
        qx, qy, qz = -qx, -qy, -qz
    x, y, z = p
    ux, uy, uz = 2*(qy*z-qz*y), 2*(qz*x-qx*z), 2*(qx*y-qy*x)
    rotated = (x+qw*ux+qy*uz-qz*uy,
               y+qw*uy+qz*ux-qx*uz,
               z+qw*uz+qx*uy-qy*ux)
    return rotated if inverse else tuple(a+b for a, b in zip(rotated, t))


def track_geometry(track, views, transform, include_xyz=False, memory_epoch=None, odom_frame="odom"):
    """All support and saved camera positions are odom; record x/y is map."""
    support = [transform_xy(point, transform) for point in track.get("support_points", [])]
    target = ((median(p[0] for p in support), median(p[1] for p in support)) if support
              else (float(track["x"]), float(track["y"])))
    previous = [transform_xy((v["x"], v["y"]), transform)
                for entries in track.get("evidence", {}).get("independent_viewpoints", {}).values()
                for v in entries]
    previous.extend(transform_xy((v["x"], v["y"]), transform) for v in views
                    if v.get("view_id") in track.get("view_ids", []))
    if not all(math.isfinite(v) for v in target) or not previous:
        raise ValueError("pending track lacks finite target or retained camera viewpoints")
    result = (target, previous, support)
    if not include_xyz:
        return result
    xyz = framing_xyz(track, views, memory_epoch, odom_frame)
    try:
        measured = [transform_xyz(point, transform) for point in xyz]
    except (ValueError, TypeError, OverflowError):
        measured = []  # Optional height cannot disable the original XY task.
    return (*result, measured)


def hint_geometry(hint, transform, include_xyz=False):
    """Transform every measured hypothesis; the anchor is not an object pose."""
    raw_groups = hint["surface_groups"]
    minimum = 1 if hint.get("hint_kind") == "class_ambiguity" else 2
    if not isinstance(raw_groups, list) or not minimum <= len(raw_groups) <= 8:
        raise ValueError("hint lacks the required complete measured surface groups")
    groups, groups_xyz = [], []
    for group in raw_groups:
        points = group["odom_xyz"]
        if (not isinstance(points, list) or not 1 <= len(points) <= 64
                or any(len(p) != 3 or not all(math.isfinite(float(v)) for v in p) for p in points)):
            raise ValueError("invalid measured surface in depth hint")
        measured = [transform_xyz(point, transform) for point in points]
        groups_xyz.append(measured)
        groups.append([point[:2] for point in measured])
    support = [point for group in groups for point in group]
    target = (median(p[0] for p in support), median(p[1] for p in support))
    poses = [hint[key] for key in ("source_pose", "source_camera")]
    if any(len(pose) != 3 or not all(math.isfinite(float(v)) for v in pose) for pose in poses):
        raise ValueError("invalid depth hint source viewpoint")
    previous = [transform_xy(poses[0][:2], transform)]
    camera_xy = transform_xy(poses[1][:2], transform)
    angle = float(poses[1][2]) + yaw_of(transform.rotation)
    source = (*camera_xy, math.atan2(math.sin(angle), math.cos(angle)))
    result = (target, previous, support, groups, source)
    return (*result, groups_xyz) if include_xyz else result


def capture_viewpoints(views, transform):
    """Memory's capture deduplication covers every saved camera, in odom."""
    offset = yaw_of(transform.rotation)
    poses = []
    for view in views:
        x, y = transform_xy((view["x"], view["y"]), transform)
        yaw = float(view["yaw"])+offset
        if not math.isfinite(yaw):
            raise ValueError("non-finite retained camera orientation")
        poses.append((x, y, math.atan2(math.sin(yaw), math.cos(yaw))))
    return poses


def camera_framing(info, transform, tolerances):
    """Level optical camera only: XY memory cannot model pitched/rolled FOV."""
    q = transform.rotation
    norm = math.sqrt(sum(v*v for v in (q.x, q.y, q.z, q.w)))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid base-to-camera rotation")
    x, y, z, w = (v/norm for v in (q.x, q.y, q.z, q.w))
    right = (1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y))
    down = (2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x))
    forward = (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
    # Accommodate sub-degree rounded calibration (e.g. a serialized pi/2),
    # but do not treat a deliberately pitched/rolled mount as planar.
    if (max(abs(right[2]), abs(forward[2])) > math.sin(math.radians(.5)) or
            right[0]*forward[1]-right[1]*forward[0] < .999):
        raise ValueError("horizontal framing requires a level upright camera; pitch/roll is unsupported")
    if (any(not math.isfinite(v) or abs(v) > 1e-8 for v in info.d)
            or not math.isfinite(info.k[1]) or abs(info.k[1]) > 1e-8):
        raise ValueError("horizontal framing requires rectified pinhole CameraInfo")
    camera = CameraFraming(info.width, info.k[0], info.k[2],
                           transform.translation.x, transform.translation.y,
                           math.atan2(forward[1], forward[0]), *tolerances)
    ideal_right, ideal_left = camera.limits()
    if (info.height < 8 or not math.isfinite(info.k[4]) or info.k[4] <= 0 or
            not math.isfinite(info.k[5])):
        raise ValueError("horizontal framing needs valid vertical CameraInfo for projection bounds")
    edge = max(2., info.width*.005)
    bounds = []
    for u in (edge, info.width-edge):
        bearings = []
        for v in (0., float(info.height)):
            dx, dy = (u-camera.cx)/camera.fx, (v-info.k[5])/info.k[4]
            px = right[0]*dx+down[0]*dy+forward[0]
            py = right[1]*dx+down[1]*dy+forward[1]
            # On either vertical image edge, XY is affine in v; its bearing
            # is monotone while it remains in the front half-plane. Thus the
            # two endpoint rays bound all valid image rows, without assuming
            # an object height. Horizontal projection is also monotone in u.
            front = px*math.cos(camera.yaw_offset)+py*math.sin(camera.yaw_offset)
            if front <= 0 or right[0]*py-right[1]*px <= 0:
                raise ValueError("camera edge projection cannot be bounded in the horizontal plane")
            bearings.append(math.atan2(math.sin(math.atan2(py, px)-camera.yaw_offset),
                                       math.cos(math.atan2(py, px)-camera.yaw_offset)))
        bounds.append(bearings)
    margin = max(0., ideal_left-min(bounds[0]), ideal_right+max(bounds[1]))+1e-12
    # This only bounds horizontal framing conditional on the surface's vertical
    # ray lying in the image. Neither object height nor vertical visibility is
    # inferred from the planar support points.
    camera = CameraFraming(camera.width, camera.fx, camera.cx, camera.offset_x,
                           camera.offset_y, camera.yaw_offset, *tolerances, margin,
                           (info.height, info.k[4], info.k[5], (x, y, z, w),
                            (transform.translation.x, transform.translation.y,
                             transform.translation.z)))
    camera.limits()
    return camera


class ObservationNode(Node):
    def __init__(self):
        super().__init__("observation_node")
        self.declare_parameter("initial_exploration_enabled", True)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("footprint_topic", "/global_costmap/published_footprint")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.state = RevisitState(self.get_parameter("initial_exploration_enabled").value)
        self.memory = self.map_msg = self.cost_msg = None
        self.footprint_msgs = []
        self.camera_info = None
        self.goal_checker = self.goal_tolerances = self.parameter_future = None
        self.parameter_request_wall = self.parameter_success_wall = -math.inf
        self.parameter_names = []
        self.last_status = self.catalog_wait_ns = None
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.request_pub = self.create_publisher(String, "/locateanything/observation_request", reliable)
        self.capture_pub = self.create_publisher(String, "/locateanything/observation_capture", reliable)
        self.busy_pub = self.create_publisher(Bool, "/locateanything/observation_busy", reliable)
        self.status_pub = self.create_publisher(String, "/locateanything/observation_status", latched)
        self.create_subscription(String, "/locateanything/memory", self._memory_cb, latched)
        self.create_subscription(OccupancyGrid, self.get_parameter("map_topic").value,
                                 self._map_cb, latched)
        self.create_subscription(OccupancyGrid, self.get_parameter("costmap_topic").value,
                                 self._cost_cb, latched)
        # Nav2 uses SystemDefaultsQoS here, not the latched map QoS.
        self.create_subscription(PolygonStamped, self.get_parameter("footprint_topic").value,
                                 self._footprint_cb, qos_profile_system_default)
        self.create_subscription(Bool, "/exploration_enabled", self._enabled_cb, reliable)
        self.create_subscription(String, "/user_command", self._command_cb, reliable)
        self.create_subscription(String, "/locateanything/observation_event", self._event_cb, reliable)
        self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value,
                                 self._camera_cb, qos_profile_sensor_data)
        self.parameter_client = self.create_client(GetParameters, "/controller_server/get_parameters")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(1., self._tick)

    def _now(self):
        now = self.get_clock().now().nanoseconds
        if self.state.clock(now):
            self.memory = self.map_msg = self.cost_msg = None
            self.footprint_msgs = []
            self.camera_info = None
            self.catalog_wait_ns = None
            self._clear()
        return now

    def _clear(self):
        self.request_pub.publish(String(data=json.dumps({"request_id": ""})))

    def _memory_cb(self, message):
        now = self._now()
        try:
            memory = json.loads(message.data)
            if (not isinstance(memory, dict) or not isinstance(memory.get("pending_landmarks", []), list)
                    or not isinstance(memory.get("observation_hints", []), list)):
                raise ValueError("invalid pending memory")
            self.memory = memory
            if self.state.reconcile(self._pending(), now):
                self._clear()
        except (ValueError, TypeError):
            self.memory = None
            self.state.withdraw()
            self._clear()

    def _pending(self):
        tasks = {item["object_id"]: item for item in (self.memory or {}).get("pending_landmarks", [])
                 if isinstance(item, dict) and isinstance(item.get("object_id"), str)
                 and not item.get("confirmed") and not item.get("conflicted")}
        for hint in (self.memory or {}).get("observation_hints", []):
            if (isinstance(hint, dict) and isinstance(hint.get("hint_id"), str)
                    and hint["hint_id"] and hint.get("status") == "unresolved"
                    and hint["hint_id"] not in tasks):
                tasks[hint["hint_id"]] = hint
        return tasks

    def _map_cb(self, message):
        self._now()
        self.map_msg = message

    def _cost_cb(self, message):
        self._now()
        self.cost_msg = message

    def _camera_cb(self, message):
        self._now()
        self.camera_info = message

    def _footprint_cb(self, message):
        self._now()
        messages = {footprint_stamp_ns(m.header.stamp): m for m in self.footprint_msgs}
        messages[footprint_stamp_ns(message.header.stamp)] = message
        self.footprint_msgs = [messages[stamp] for stamp in sorted(messages)[-3:]]

    def _start_footprint(self, now, current_robot):
        """Reproject the actual Nav2 shape, never reuse an old world polygon.

        Its 1 Hz update can be older than the current robot TF. Recover the
        rigid base-local shape with EXACT cached TF at the footprint stamp,
        then place it at this tick's already validated fresh robot transform.
        No timeout, spin, latest-TF fallback, or radius assumption is used.
        """
        messages = getattr(self, "footprint_msgs", [])
        detail = "waiting for map-frame footprint for unknown robot start"
        for message in reversed(messages):
            try:
                if message.header.frame_id.lstrip("/") != "map":
                    raise ValueError("footprint is not in map frame")
                stamp = footprint_stamp_ns(message.header.stamp)
                if stamp <= 0 or stamp > now or now-stamp > 3*SECOND:
                    raise ValueError("waiting for non-future footprint no older than 3 seconds")
                if not 3 <= len(message.polygon.points) <= 64:
                    raise ValueError("footprint requires 3..64 vertices")
                if any(not math.isfinite(p.z) for p in message.polygon.points):
                    raise ValueError("non-finite footprint vertex")
                convex_footprint([(p.x, p.y) for p in message.polygon.points])
                base = self.get_parameter("robot_base_frame").value
                stamped = self.tf_buffer.lookup_transform("map", base, Time(nanoseconds=stamp))
                returned_stamp = footprint_stamp_ns(stamped.header.stamp)
                if (returned_stamp != stamp or stamped.header.frame_id.lstrip("/") != "map"
                        or stamped.child_frame_id.lstrip("/") != base.lstrip("/")):
                    raise ValueError("exact footprint-stamp robot transform unavailable")
                local = [transform_xyz((p.x, p.y, p.z), stamped.transform, inverse=True)
                         for p in message.polygon.points]
                local_xy = convex_footprint([p[:2] for p in local])
                if not _inside_footprint((0., 0.), local_xy):
                    raise ValueError("footprint does not contain the robot origin")
                projected = [transform_xyz(p, current_robot) for p in local]
                result = convex_footprint([p[:2] for p in projected])
                if not _inside_footprint((current_robot.translation.x, current_robot.translation.y), result):
                    raise ValueError("current footprint does not contain the robot")
                return result
            except (ValueError, TypeError, tf2_ros.TransformException) as exc:
                detail = str(exc)
        raise ValueError(detail)

    def _refresh_goal_tolerances(self):
        """Nonblocking discovery/cache of the actual Nav2 goal checker values."""
        wall = time.monotonic()
        if self.parameter_future is not None:
            if self.parameter_future.done():
                try:
                    values = dict(zip(self.parameter_names, self.parameter_future.result().values))
                    plugins = values['goal_checker_plugins']
                    if plugins.type != 9 or len(plugins.string_array_value) != 1:
                        raise ValueError("a single active Nav2 goal checker is required")
                    checker = plugins.string_array_value[0]
                    if checker != self.goal_checker:
                        self.goal_checker, self.goal_tolerances = checker, None
                        self.parameter_request_wall = -math.inf
                    else:
                        pair = [values[checker+'.'+name] for name in ('xy_goal_tolerance', 'yaw_goal_tolerance')]
                        if any(v.type != 3 or not math.isfinite(v.double_value) or v.double_value < 0 for v in pair):
                            raise ValueError("invalid Nav2 goal checker tolerances")
                        self.goal_tolerances = tuple(v.double_value for v in pair)
                        self.parameter_success_wall = wall
                except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                    self.goal_tolerances = None
                self.parameter_future = None
            elif wall-self.parameter_request_wall >= 2.:
                self.parameter_future.cancel()
                self.parameter_future = None
        if self.parameter_future is None and wall-self.parameter_request_wall >= 2. and self.parameter_client.service_is_ready():
            self.parameter_names = ['goal_checker_plugins']
            if self.goal_checker:
                self.parameter_names += [self.goal_checker+'.'+name for name in ('xy_goal_tolerance', 'yaw_goal_tolerance')]
            request = GetParameters.Request(names=self.parameter_names)
            self.parameter_future = self.parameter_client.call_async(request)
            self.parameter_request_wall = wall

    def _camera_model(self, now):
        if self.goal_tolerances is None or time.monotonic()-self.parameter_success_wall > 6.:
            raise ValueError("waiting for current Nav2 goal checker tolerances")
        info = self.camera_info
        if info is None or not info.header.frame_id:
            raise ValueError("waiting for CameraInfo for horizontal framing")
        stamp = info.header.stamp.sec*SECOND+info.header.stamp.nanosec
        if abs(now-stamp) > 3*SECOND:
            raise ValueError("waiting for fresh CameraInfo for horizontal framing")
        base = self.get_parameter("robot_base_frame").value
        # This is a rigid base-to-sensor calibration, so static TF stamp zero
        # is valid. Dynamic map/robot transforms retain their freshness guard.
        extrinsic = self.tf_buffer.lookup_transform(base, info.header.frame_id, Time()).transform
        return camera_framing(info, extrinsic, self.goal_tolerances)

    def _enabled_cb(self, message):
        self._now()
        self.state.set_enabled(message.data)
        if not message.data:
            if hasattr(self, "_search_progress"):
                self._search_progress.reset()
            self.catalog_wait_ns = None
            self._clear()
            self.busy_pub.publish(Bool(data=False))

    def _command_cb(self, message):
        self._now()
        if message.data.strip():
            if hasattr(self, "_search_progress"):
                self._search_progress.reset()
            self.state.set_enabled(False)
            self.catalog_wait_ns = None
            self._clear()
            self.busy_pub.publish(Bool(data=False))

    def _event_cb(self, message):
        now = self._now()
        try:
            event = json.loads(message.data)
            record = self.state.records.get(event.get("request_id")) if isinstance(event, dict) else None
            was_dispatched = bool(record and record["dispatched"])
            if isinstance(event, dict) and self.state.event(event, now, (self.memory or {}).get("epoch")):
                if event.get("event") == "dispatched" and not was_dispatched:
                    self.get_logger().info(json.dumps({
                        "state": "observation_visibility_dispatch", "request_id": event["request_id"],
                        "track_id": record["track_id"], "visibility_reason": record["visibility_reason"],
                        "focus_group_index": record.get("focus_group_index"),
                        "hypothesis_group_count": record.get("hypothesis_group_count", 0),
                        "focus_novel_fraction": record.get("focus_novel_fraction"),
                        "vertical_framing": record.get("vertical_framing"),
                        "planning_goal_tolerances": record.get("planning_goal_tolerances"),
                        "framing_scope": "one_measured_hypothesis_group" if record.get("focus_group_index") is not None else "localized_target",
                        "map_sightline_groups": record["map_sightline_groups"],
                        "uncertain_dispatched_for_task": self.state.uncertain_dispatched.get(record["track_id"], 0)}))
                if self.state.reconcile(self._pending(), now):
                    self._clear()
        except (ValueError, TypeError):
            pass

    @staticmethod
    def _grid(message):
        if message is None or message.header.frame_id.lstrip("/") != "map":
            raise ValueError("waiting for map-frame occupancy grids")
        info = message.info
        grid = Grid(info.width, info.height, info.resolution, info.origin.position.x,
                    info.origin.position.y, yaw_of(info.origin.orientation), message.data)
        if not grid.valid():
            raise ValueError("invalid occupancy grid")
        return grid

    def _transform(self, source, now):
        stamped = self.tf_buffer.lookup_transform("map", source, Time())
        stamp = stamped.header.stamp.sec*SECOND+stamped.header.stamp.nanosec
        if hasattr(self, "_planning_tf_stamps"):
            self._planning_tf_stamps[source] = stamp
        if abs(now-stamp) > .75*SECOND:
            raise ValueError("waiting for fresh map transform")
        return stamped.transform

    def _bridge_robot_transform(self, now):
        """Unknown-start proof needs one non-future, nonblocking current pose."""
        base = self.get_parameter("robot_base_frame").value
        stamped = self.tf_buffer.lookup_transform("map", base, Time())
        stamp = footprint_stamp_ns(stamped.header.stamp)
        if stamp > now:
            stamped = self.tf_buffer.lookup_transform("map", base, Time(nanoseconds=now))
            stamp = footprint_stamp_ns(stamped.header.stamp)
            if stamp != now:
                raise ValueError("exact current robot transform unavailable for footprint proof")
        if (stamp <= 0 or stamp > now or now-stamp > .75*SECOND
                or stamped.header.frame_id.lstrip("/") != "map"
                or stamped.child_frame_id.lstrip("/") != base.lstrip("/")):
            raise ValueError("waiting for non-future fresh robot transform for footprint proof")
        transform_xyz((0., 0., 0.), stamped.transform)  # Validate the full rigid transform.
        if hasattr(self, "_planning_tf_stamps"):
            self._planning_tf_stamps[base] = stamp
        return stamped.transform

    def _tick(self):
        now = self._now()
        self._refresh_goal_tolerances()
        pending = self._pending()
        if not pending:
            self.catalog_wait_ns = None
        if self.state.reconcile(pending, now):
            self._clear()
        capture = self.state.capture_request(now, (self.memory or {}).get("epoch"))
        if capture is not None:
            self.capture_pub.publish(String(data=json.dumps(capture, allow_nan=False)))
        detail = ""
        state = self.state.status(now)["state"]
        busy = bool(pending and self.state.enabled and state in ("observing", "settling"))
        # Navigation budgets do not discard the last successful observation.
        # Its fixed five-second capture window may cross the navigation time
        # limit; already-running catalog work gets at most thirty more seconds.
        # This deadline never resets when catalog activity or state changes.
        if (self.state.capture and pending and self.state.enabled and
                state in ("budget_exhausted", "track_budgets_exhausted") and
                (self.memory.get("catalog_pending", 0) or not self.memory.get("catalog_idle", True))):
            busy = now < self.state.capture["expires_stamp_ns"]+30*SECOND
            detail = ("waiting for successful-observation catalog to settle" if busy else
                      "successful-observation catalog exceeded its fixed 30 second finish window")
        if state != "ready" and hasattr(self, "_search_progress"):
            self._search_progress.reset()
        if state == "ready":
            try:
                if self.memory.get("catalog_pending", 0) or not self.memory.get("catalog_idle", True):
                    if self.catalog_wait_ns is None:
                        self.catalog_wait_ns = now
                    busy = now-self.catalog_wait_ns < 30*SECOND
                    raise ValueError("waiting for catalog to settle" if busy else
                                     "catalog did not settle within 30 seconds; memory remains incomplete")
                self.catalog_wait_ns = None
                maps, costs = self._grid(self.map_msg), self._grid(self.cost_msg)
                stamp = self.cost_msg.header.stamp.sec*SECOND+self.cost_msg.header.stamp.nanosec
                if abs(now-stamp) > 3*SECOND:
                    raise ValueError("waiting for fresh global costmap")
                planning_started = time.monotonic()
                self._planning_tf_stamps = {}
                transform = self._transform(self.get_parameter("odom_frame").value, now)
                robot_transform = self._transform(self.get_parameter("robot_base_frame").value, now)
                robot = robot_transform.translation
                start_footprint = ()
                if maps.value(robot.x, robot.y) == -1 and traversable_cost(costs.value(robot.x, robot.y)):
                    robot_transform = self._bridge_robot_transform(now)
                    robot = robot_transform.translation
                    if maps.value(robot.x, robot.y) == -1 and traversable_cost(costs.value(robot.x, robot.y)):
                        start_footprint = self._start_footprint(now, robot_transform)
                camera = self._camera_model(now)
                retained_cameras = capture_viewpoints(self.memory.get("views", []), transform)
                planning_deferred = False
                had_proposal = self.state.proposal is not None
                request = self.state.current(now)
                if had_proposal and request is None:
                    self._clear()  # Retire the queued old ID before publishing its replacement.
                if request is None:
                    # Localized pending tracks retain their original priority.
                    # Hint focus groups then compete globally by measured novelty.
                    eligible = {key: value for key, value in pending.items() if self.state.ready(key, now)}
                    localized = sorted((key for key in eligible if "hint_id" not in eligible[key]),
                                       key=lambda key: (self.state.attempts.get(key, 0), key))
                    hints = {key: value for key, value in eligible.items() if "hint_id" in value}
                    ranked = ranked_hint_groups(hints, self.memory.get("landmarks", []),
                                                self.state.attempts, self.state.focus_dispatched)
                    options = [(key, None, None, False) for key in localized] + ranked
                    reachability_cache = {}
                    if not hasattr(self, "_search_progress"):
                        self._search_progress = SearchProgress()
                    progress = self._search_progress
                    progress.begin(options, (self.state.epoch, self.memory.get("epoch"), self.state.total_attempts))
                    chosen = None
                    planning_deferred = False
                    planning_calls = 0
                    while True:
                        work = progress.next(options)
                        if work is None:
                            progress.reset()
                            break
                        if planning_calls >= 4 or time.monotonic()-planning_started >= .2:
                            planning_deferred = True
                            break
                        row, recheck_uncertain = work
                        track_id, focus_index, novelty, tried = row
                        track = eligible[track_id]
                        try:
                            kwargs = {}
                            if focus_index is not None:
                                target, previous, support, groups, source, xyz_groups = hint_geometry(track, transform, include_xyz=True)
                                vertical, vertical_detail = None, "optional vertical calibration unavailable"
                                if camera.vertical_calibration:
                                    t, q = robot_transform.translation, robot_transform.rotation
                                    try:
                                        vertical = VerticalFraming(*camera.vertical_calibration,
                                                                   (t.x, t.y, t.z), (q.x, q.y, q.z, q.w),
                                                                   camera.xy_tolerance, camera.yaw_tolerance)
                                    except ValueError as exc:
                                        # Soft framing metadata cannot disable an
                                        # otherwise eligible legacy partial view.
                                        vertical_detail = str(exc)
                                kwargs = {"hypothesis_groups": groups, "source_camera": source,
                                          "hypothesis_xyz_groups": xyz_groups, "vertical_framing": vertical,
                                          "focus_hypotheses": True, "forced_focus_index": focus_index,
                                          "attempted_focus_groups": self.state.focus_dispatched.get(track_id, [])}
                            else:
                                target, previous, support, xyz = track_geometry(
                                    track, self.memory.get("views", []), transform, include_xyz=True,
                                    memory_epoch=self.memory.get("epoch"),
                                    odom_frame=self.get_parameter("odom_frame").value)
                                if xyz and camera.vertical_calibration:
                                    t, q = robot_transform.translation, robot_transform.rotation
                                    try:
                                        vertical = VerticalFraming(*camera.vertical_calibration,
                                                                   (t.x, t.y, t.z), (q.x, q.y, q.z, q.w),
                                                                   camera.xy_tolerance, camera.yaw_tolerance)
                                        kwargs = {"vertical_framing": vertical, "vertical_support_xyz": xyz}
                                    except ValueError:
                                        pass  # Unavailable optional calibration keeps original XY behavior.
                        except (ValueError, TypeError, KeyError):
                            progress.tested(row)
                            continue
                        attempted = [transform_xy(point, transform)
                                     for point in self.state.attempted_odom.get(track_id, [])]
                        cached = [transform_xy(point, transform)
                                  for point in self.state.cached_positions(track_id)]
                        candidate = choose_observation_candidate(maps, costs, (robot.x, robot.y), target,
                                                       previous, attempted, support_xy=support,
                                                       capture_viewpoints=retained_cameras, camera=camera,
                                                       cached_viewpoints=cached,
                                                       start_footprint=start_footprint,
                                                       allow_uncertain_visibility=self.state.uncertain_dispatched.get(track_id, 0) < 1,
                                                       dedicated_capture=True,
                                                       prefer_viewpoint_diversity=focus_index is None,
                                                       _reachability_cache=reachability_cache, **kwargs)
                        planning_calls += 1
                        if candidate is None:
                            progress.tested(row)
                            continue
                        if focus_index is not None and vertical is None:
                            candidate["vertical_framing"] = {
                                "available": False, "reason": vertical_detail,
                                "scope": "No vertical preference; original partial-view selection retained."}
                        if focus_index is None and kwargs.get("vertical_support_xyz") and candidate.get("vertical_framing"):
                            source = track["framing_source"]
                            candidate["vertical_framing"]["source"] = {
                                key: source[key] for key in ("epoch", "object_id", "view_id", "stamp_ns",
                                                           "depth_stamp_ns", "odom_frame", "box")}
                            candidate["vertical_framing"]["source_scope"] = (
                                "One retained accepted observation's measured XYZ; unknown extent is not inferred.")
                        candidate["focus_novel_fraction"] = novelty
                        if (focus_index is not None and not recheck_uncertain
                                and candidate["visibility_reason"] == UNKNOWN_VISIBILITY):
                            progress.tested(row, uncertain=True)
                            continue
                        chosen = (track_id, candidate)
                        break
                    if chosen is not None:
                        elapsed = time.monotonic()-planning_started
                        actual_now = self.get_clock().now().nanoseconds
                        # The clock callback may not run during CPU planning.
                        # Add elapsed wall time conservatively to captured input
                        # ages; never reinterpret this pose using a newer TF.
                        age_extra = max(elapsed, (actual_now-now)/SECOND)
                        fresh = (actual_now >= now and elapsed < .5
                                 and abs(now-stamp)/SECOND+age_extra <= 3.
                                 and all(abs(now-t)/SECOND+age_extra <= .75
                                         for t in self._planning_tf_stamps.values()))
                        if fresh:
                            track_id, candidate = chosen
                            pose = candidate["pose"]
                            request = self.state.offer(track_id, pose, transform_xy(pose[:2], transform, True), actual_now,
                                                       candidate["visibility_reason"], candidate["map_sightline_groups"],
                                                       candidate.get("focus_group_index"),
                                                       candidate.get("hypothesis_group_count", 0))
                            if request:
                                self.state.records[request["request_id"]]["focus_novel_fraction"] = candidate["focus_novel_fraction"]
                                self.state.records[request["request_id"]]["vertical_framing"] = candidate.get("vertical_framing")
                                read_age = time.monotonic()-getattr(self, "parameter_success_wall", -math.inf)
                                self.state.records[request["request_id"]]["planning_goal_tolerances"] = {
                                    "xy_tolerance_m": camera.xy_tolerance,
                                    "yaw_tolerance_rad": camera.yaw_tolerance,
                                    "goal_checker": getattr(self, "goal_checker", None),
                                    "parameter_read_age_wall_sec": read_age if math.isfinite(read_age) else None,
                                    "planning_stamp_ns": now}
                                progress.reset()
                        else:
                            # A consistently slow/stale first group must not
                            # starve the tail or hold completion forever.
                            progress.tested(row)
                            planning_deferred = progress.next(options) is not None
                            if not planning_deferred:
                                progress.reset()
                if request:
                    self.request_pub.publish(String(data=json.dumps(request, allow_nan=False)))
                    busy = True
                elif request is None and planning_deferred:
                    detail = "continuing bounded observation candidate search with fresh inputs"
                    busy = True
                    self._clear()
                else:
                    detail = "no safe reachable independent observation pose"
                    self._clear()
            except (ValueError, TypeError, KeyError, tf2_ros.TransformException) as exc:
                detail = str(exc)
                self.state.proposal = None
                self._clear()
        status = self.state.status(now)
        raw_pending = (self.memory or {}).get("pending_landmarks", [])
        hints = [hint for hint in (self.memory or {}).get("observation_hints", []) if isinstance(hint, dict)]
        unresolved = [hint for hint in hints if hint.get("status") == "unresolved"]
        expired = [hint for hint in hints if hint.get("status") == "expired"]
        conflicted = [track for track in (self.memory or {}).get("landmarks", [])
                      if isinstance(track, dict) and track.get("conflicted")]
        status["pending_tracks_total"] = len(raw_pending)
        status["conflicted_tracks_total"] = len(conflicted)
        status["observation_hints_unresolved"] = len(unresolved)
        status["observation_hints_expired"] = len(expired)
        status["observation_hints_exhausted"] = sum(self.state.attempts.get(hint.get("hint_id"), 0) >= 2
                                                    for hint in unresolved)
        status["incomplete"] = bool(raw_pending or unresolved or expired or conflicted)
        status["busy"] = busy
        status["detail"] = detail
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
        self.busy_pub.publish(Bool(data=busy))
        key = (status["state"], detail, status["total_dispatched"], status["pending_tracks"])
        if key != self.last_status:
            self.get_logger().info(json.dumps(status, allow_nan=False))
            self.last_status = key


def main(args=None):
    rclpy.init(args=args)
    node = ObservationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._clear()
            node.busy_pub.publish(Bool(data=False))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
