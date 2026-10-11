"""Exercise production callbacks with ROS messages, without spinning ROS or ML.

The fake publishers record actual SemanticQueryResult messages. Sensor/TF and
worker boundaries are controlled, while query state transitions remain real.
"""
import base64
from collections import deque
from concurrent.futures import Future
import math
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan
from std_msgs.msg import String

from tb3_locateanything import locateanything_node as module
from tb3_locateanything.geometry import (CameraCalibration, Quaternion, Transform3D, Vector3,
                                        quaternion_rotate_vector)
from tb3_locateanything.visual_memory import Memory


SECOND = 1_000_000_000
BOX = [100.0, 100.0, 250.0, 400.0]
IDENTITY = Transform3D(Vector3(0, 0, 0), Quaternion(0, 0, 0, 1))
IMAGE_PNG = base64.b64encode(module.cv2.imencode('.png', np.zeros((480, 640, 3), np.uint8))[1]).decode('ascii')


def stamped(message, seconds, frame):
    message.header.stamp = Time(nanoseconds=int(seconds * SECOND)).to_msg()
    message.header.frame_id = frame
    return message


def odom(seconds=100, x=8.0, y=9.0, yaw=0.0, speed=0.0):
    message = stamped(Odometry(), seconds, "odom")
    message.pose.pose.position.x, message.pose.pose.position.y = float(x), float(y)
    message.pose.pose.orientation.z = math.sin(yaw / 2)
    message.pose.pose.orientation.w = math.cos(yaw / 2)
    message.twist.twist.linear.x = float(speed)
    return message


def transform(seconds=100, translation=(0.0, 0.0, 0.0)):
    message = stamped(TransformStamped(), seconds, "map")
    message.child_frame_id = "odom"
    t = message.transform.translation
    t.x, t.y, t.z = map(float, translation)
    message.transform.rotation.w = 1.0
    return message


def imu(seconds=99.8, tilt=0.0):
    message = stamped(Imu(), seconds, "imu_link")
    message.orientation.y = math.sin(tilt / 2)
    message.orientation.w = math.cos(tilt / 2)
    return message


def publisher():
    messages = []
    return SimpleNamespace(publish=messages.append, messages=messages)


@pytest.fixture
def node():
    # Bind real methods to a plain object: no rclpy.init(), process, timer or DDS.
    result = SimpleNamespace()
    for name, value in vars(module.LocateAnythingNode).items():
        if callable(value) and not name.startswith("__") and name != "destroy_node":
            setattr(result, name, MethodType(value, result))
    result.now_ns = 100 * SECOND
    result.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=result.now_ns))
    result.p = {"output_frame": "map", "optical_frame": "", "sensor_max_age_sec": .75,
                "sensor_tolerance_sec": .15, "stationary_seconds": .7,
                "max_robot_tilt_deg": 15.,
                "capture_timeout_sec": 12., "query_timeout_sec": 90.,
                "inference_timeout_sec": 25., "query_max_views": 4,
                "memory_capture_interval_sec": 3., "memory_empty_retry_sec": 20.,
                "memory_classes": ["chair", "person", "trash can"], "evidence_dir": ""}
    result.memory = Memory()
    result.images, result.scans, result.infos = deque(), deque(), deque()
    result.odoms = deque([odom()])
    result.imus = deque([imu()])
    result.stationary_since = 90 * SECOND
    result.pending, result.future, result.job = None, None, None
    result.worker_ready, result.worker_error = True, ""
    result.worker = SimpleNamespace(infer=lambda *_: None, process=SimpleNamespace(poll=lambda: None))
    result.inference_calls = []
    def submit(*args):
        result.inference_calls.append(args)
        return Future()
    result.pool = SimpleNamespace(submit=submit)
    result.generation, result.epoch, result.last_clock_ns = 0, 0, result.now_ns
    result._reset_target_history()
    result.last_capture_wall = module.time.monotonic()
    result.catalog_queue, result.debug_image = deque(), None
    result.tf_buffer = SimpleNamespace(lookup_transform=lambda *args: transform())
    for name in ("result_pub", "marker_pub", "memory_pub", "memory_markers_pub", "debug_pub"):
        setattr(result, name, publisher())
    result.events = []
    result._status = lambda state, **fields: result.events.append({"state": state, **fields})
    result.shown_images = []
    result._show_box = lambda image, box, request: result.shown_images.append(image)
    return result


def stored_view(node, seconds=10, pose=(0.0, 0.0, 0.0)):
    image = stamped(Image(), seconds, "camera_optical")
    image.width, image.height = 640, 480
    scan = stamped(LaserScan(), seconds, "base_scan")
    scan.ranges = [1.0, 2.0, 3.0]
    scan.angle_min, scan.angle_increment = -1.0, .5
    scan.range_min, scan.range_max = .1, 5.0
    payload = {"image": image, "scan": scan, "image_png": IMAGE_PNG,
               "calibration": CameraCalibration(640, 480, 320, 320, 320, 240, "camera_optical"),
               "camera_to_scan": IDENTITY, "scan_to_odom": IDENTITY,
               "scan_to_map": IDENTITY, "odom_frame": "odom", "odom_pose": pose,
               "metadata": {"image_stamp_ns": module.stamp_ns(image)}}
    return node.memory.add_view(payload, module.stamp_ns(image), *pose)


def active_query(node, text="move to the chair", views=()):
    node._command_cb(String(data=text))
    node.pending["stamp_ns"] = 99 * SECOND
    node.pending["query_pose"] = module.odom_pose(node.odoms[-1])
    node.pending["views"] = None if views is None else deque(views)
    return node.pending


def test_localization_uses_historical_scan_and_tf_with_current_map_alignment(node, monkeypatch):
    view = stored_view(node)
    node.scans.append(stamped(LaserScan(), 100, "different_current_scan"))
    calls = []

    def localize(*args):
        calls.append(args)
        return Vector3(2, 3, 0), {"range_m": 2.0}

    monkeypatch.setattr(module, "localize_box", localize)
    tf_calls = []
    current = transform(100, (20, 30, 0))
    node.tf_buffer = SimpleNamespace(lookup_transform=lambda *args: tf_calls.append(args) or current)
    point, geometry = node._localize(view, BOX)
    assert calls[0][1] is view.payload["calibration"]
    assert calls[0][2] is view.payload["camera_to_scan"]
    assert calls[0][3] is view.payload["scan_to_odom"]
    assert calls[0][4] is view.payload["scan"].ranges
    assert point == Vector3(22, 33, 0)
    assert tf_calls[0][:2] == ("map", "odom")
    assert tf_calls[0][2].nanoseconds == 0  # latest map alignment, not old sensor substitution
    assert geometry["map_to_odom_stamp_ns"] == 100 * SECOND


def test_localization_rejects_stale_current_map_alignment(node, monkeypatch):
    view = stored_view(node)
    monkeypatch.setattr(module, "localize_box", lambda *args: (Vector3(1, 2, 0), {}))
    node.tf_buffer = SimpleNamespace(lookup_transform=lambda *args: transform(90))
    with pytest.raises(ValueError, match="map-to-odom transform is stale"):
        node._localize(view, BOX)


def test_capture_freezes_synchronized_messages_and_observation_time_transforms(node):
    image = stamped(Image(), 99.8, "camera_optical")
    image.width, image.height = 640, 480
    scan = stamped(LaserScan(), 99.81, "base_scan")
    info = stamped(CameraInfo(), 99.8, "camera_optical")
    info.width, info.height = 640, 480
    info.k = [320., 0., 320., 0., 320., 240., 0., 0., 1.]
    node.images.append(image)
    node.scans.append(scan)
    node.infos.append(info)
    node.odoms.append(odom(99.8))
    calls = []

    def lookup(*args):
        calls.append(args)
        return transform(99.8, (float(len(calls)), 0, 0))

    full_calls = []
    node.tf_buffer = SimpleNamespace(lookup_transform=lookup,
        lookup_transform_full=lambda **kwargs: full_calls.append(kwargs) or transform(99.8, (7., 0., 0.)))
    node.bridge = SimpleNamespace(imgmsg_to_cv2=lambda *args: np.zeros((480, 640, 3), dtype=np.uint8))
    snapshot = node._capture()
    assert snapshot["image"] is image and snapshot["scan"] is scan
    assert snapshot["image_png"]
    assert [(target, source, when.nanoseconds) for target, source, when in calls] == [
        ("map", "base_scan", module.stamp_ns(scan)),
        ("odom", "base_scan", module.stamp_ns(scan))]
    assert len(full_calls) == 1
    assert full_calls[0]["target_frame"] == "base_scan"
    assert full_calls[0]["source_frame"] == "camera_optical"
    assert full_calls[0]["fixed_frame"] == "odom"
    assert full_calls[0]["target_time"].nanoseconds == module.stamp_ns(scan)
    assert full_calls[0]["source_time"].nanoseconds == module.stamp_ns(image)
    assert snapshot["camera_to_scan"].translation.x == 7
    assert snapshot["scan_to_odom"].translation.x == 2


def depth_capture_inputs(node):
    node.p["localization_mode"] = "depth"
    image = stamped(Image(), 99.8, "camera_optical")
    image.width, image.height = 640, 480
    info = stamped(CameraInfo(), 99.8, "camera_optical")
    info.width, info.height = 640, 480
    info.k = [320., 0., 320., 0., 320., 240., 0., 0., 1.]
    depth = stamped(Image(), 99.8, "camera_optical")
    depth.width, depth.height, depth.encoding = 640, 480, "32FC1"
    node.images.append(image)
    node.infos.append(info)
    node.scans.append(stamped(LaserScan(), 99.81, "base_scan"))
    node.odoms.append(odom(99.8))
    node.depths, node.depth_infos = deque([depth]), deque([info])
    node.tf_buffer = SimpleNamespace(lookup_transform=lambda *args: transform(99.8),
                                     lookup_transform_full=lambda **kwargs: transform(99.8))
    node.bridge = SimpleNamespace(imgmsg_to_cv2=lambda msg, encoding:
        np.full((480, 640), 3., np.float32) if msg is depth else np.zeros((480, 640, 3), np.uint8))
    return depth, info


def test_capture_keeps_synchronized_depth_separate_from_laser(node):
    depth, _ = depth_capture_inputs(node)
    snapshot = node._capture()
    assert snapshot["depth_m"].shape == (480, 640)
    assert snapshot["depth_m"][200, 300] == 3.
    assert snapshot["metadata"]["depth_stamp_ns"] == module.stamp_ns(depth)
    assert snapshot["metadata"]["localization_source"] == "aligned_depth"
    assert snapshot["camera_to_odom"] == IDENTITY


def test_capture_uses_latest_complete_exposure_when_new_rgb_arrives_before_depth(node):
    from copy import deepcopy
    depth_capture_inputs(node)
    newer = deepcopy(node.images[-1])
    newer.header.stamp = Time(nanoseconds=99_850_000_000).to_msg()
    node.images.append(newer)
    assert node._capture()["metadata"]["image_stamp_ns"] == 99_800_000_000


@pytest.mark.parametrize("failure", ["time", "frame", "intrinsics", "missing", "override", "distortion"])
def test_depth_capture_rejects_unaligned_or_missing_depth(node, failure):
    from copy import deepcopy
    depth, info = depth_capture_inputs(node)
    if failure == "time":
        depth.header.stamp = Time(nanoseconds=99_700_000_000).to_msg()
    elif failure == "frame":
        depth.header.frame_id = "other_camera"
    elif failure == "intrinsics":
        changed = deepcopy(info)
        changed.k[0] += 1.
        node.depth_infos = deque([changed])
    elif failure == "override":
        node.p["optical_frame"] = "unrelated_camera"
    elif failure == "distortion":
        info.d = [0.1, 0., 0., 0., 0.]
    else:
        node.depths.clear()
    with pytest.raises(ValueError, match="depth"):
        node._capture()


def test_depth_mode_never_silently_uses_old_laser_only_view(node):
    node.p["localization_mode"] = "depth"
    with pytest.raises(ValueError, match="no aligned depth"):
        node._localize(stored_view(node), BOX)


def test_historical_depth_is_frozen_but_current_map_transform_is_refreshed(node):
    node.p["localization_mode"] = "depth"
    view = stored_view(node)
    depth = np.full((480, 640), 6., np.float32)
    depth[120:380, 120:230] = 3.
    view.payload.update(depth_m=depth, camera_to_odom=Transform3D(
        Vector3(0., 0., 2.), Quaternion(-.5, .5, -.5, .5)))
    first, _ = node._localize(view, BOX)
    node.depths = deque(["unrelated current depth must not be read"])
    node.tf_buffer.lookup_transform = lambda *args: transform(100, (5., 7., 0.))
    second, meta = node._localize(view, BOX)
    assert second.x-first.x == pytest.approx(5.)
    assert second.y-first.y == pytest.approx(7.)
    assert len(view.payload["depth_geometry"]) == 1
    assert meta["localization_source"] == "aligned_depth"


@pytest.mark.parametrize("scan_seconds", [99.7, 99.9])
def test_capture_compensates_rotation_for_earlier_and_later_laser_samples(node, scan_seconds):
    # Real tf2 interpolation: the robot rotates at 0.4 rad/s. The optical
    # forward ray captured at 99.8 must select the same world direction even
    # when the nearest laser message is 100 ms earlier or later.
    node.images.append(stamped(Image(), 99.8, "camera_optical"))
    node.images[-1].width, node.images[-1].height = 640, 480
    scan = stamped(LaserScan(), scan_seconds, "base_scan")
    node.scans.append(scan)
    info = stamped(CameraInfo(), 99.8, "camera_optical")
    info.width, info.height = 640, 480
    info.k = [320., 0., 320., 0., 320., 240., 0., 0., 1.]
    node.infos.append(info)
    node.odoms.append(odom(99.8, yaw=.08))
    node.bridge = SimpleNamespace(imgmsg_to_cv2=lambda *args: np.zeros((480, 640, 3), np.uint8))
    buffer = module.tf2_ros.Buffer()
    optical = transform()
    optical.header.frame_id, optical.child_frame_id = "base_scan", "camera_optical"
    optical.transform.rotation.x, optical.transform.rotation.y = -.5, .5
    optical.transform.rotation.z, optical.transform.rotation.w = -.5, .5
    buffer.set_transform_static(optical, "test")
    map_odom = transform()
    map_odom.header.frame_id, map_odom.child_frame_id = "map", "odom"
    buffer.set_transform_static(map_odom, "test")
    for when in (99.6, 100.):
        motion = transform(when)
        motion.header.frame_id, motion.child_frame_id = "odom", "base_scan"
        yaw = .4 * (when - 99.6)
        motion.transform.rotation.z = math.sin(yaw / 2)
        motion.transform.rotation.w = math.cos(yaw / 2)
        buffer.set_transform(motion, "test")
    node.tf_buffer = buffer
    snapshot = node._capture()
    ray = quaternion_rotate_vector(snapshot["camera_to_scan"].rotation, Vector3(0, 0, 1))
    expected_bearing = .4 * (99.8 - scan_seconds)
    assert math.atan2(ray.y, ray.x) == pytest.approx(expected_bearing, abs=1e-7)
    world_ray = quaternion_rotate_vector(snapshot["scan_to_odom"].rotation, ray)
    assert math.atan2(world_ray.y, world_ray.x) == pytest.approx(.08, abs=1e-7)


def test_capture_does_not_fall_back_to_same_time_tf_when_full_transform_is_missing(node):
    view = stored_view(node, 99.8)
    node.images.append(view.payload["image"])
    node.scans.append(view.payload["scan"])
    info = stamped(CameraInfo(), 99.8, "camera_optical")
    info.width, info.height = 640, 480
    info.k = [320., 0., 320., 0., 320., 240., 0., 0., 1.]
    node.infos.append(info)
    node.odoms.append(odom(99.8))
    def missing(**kwargs):
        raise module.tf2_ros.ExtrapolationException("missing cross-time odometry")
    node.tf_buffer = SimpleNamespace(lookup_transform_full=missing,
                                    lookup_transform=lambda *args: transform())
    with pytest.raises(module.tf2_ros.ExtrapolationException, match="cross-time"):
        node._capture()


def test_inference_uses_selected_old_view_image_not_live_camera(node):
    view = stored_view(node)
    request = active_query(node)
    calls = []
    node.worker = SimpleNamespace(infer=object())
    future = Future()
    node.pool = SimpleNamespace(submit=lambda *args: calls.append(args) or future)
    node._infer(view, "query", "the chair")
    assert calls[0][1]["image_png"] == view.payload["image_png"]
    assert calls[0][1]["request_id"] == request["id"]
    assert node.job["view"] is view and node.future is future


def test_motion_guard_compares_query_start_pose_not_historical_view(node):
    view = stored_view(node, pose=(-20, -30, 1.5))
    request = active_query(node, views=[view])
    node._odom_cb(odom(x=8, y=9))
    assert not request["robot_moved"]
    node._check_query_motion()
    node._odom_cb(odom(x=8.1, y=9))
    with pytest.raises(ValueError, match="robot moved"):
        node._check_query_motion()


def test_conflicting_map_targets_publish_one_failure_and_no_success(node):
    first = stored_view(node)
    second = stored_view(node, 20, pose=(2, 0, 0))
    request = active_query(node)
    request["candidates"] = [(first, BOX, Vector3(1, 2, 0), {}, {}),
                             (second, BOX, Vector3(4, 2, 0), {}, {})]
    node._localize = lambda view, box, *projection: (Vector3(1. if view is first else 4., 2., 0.), {})
    node._tick()
    assert node.pending is None
    assert len(node.result_pub.messages) == 1
    assert not node.result_pub.messages[0].success
    assert "conflicting targets" in node.result_pub.messages[0].status_message
    assert node.memory.landmarks == {}


def test_candidates_are_reprojected_before_final_comparison_and_selection(node):
    first = stored_view(node)
    second = stored_view(node, 20, pose=(2, 0, 0))
    request = active_query(node)
    response = {"answer": "valid prior result", "prompt": "the chair", "inference_seconds": 1.0}
    # These points were computed at different times during slow inference.
    # A subsequent SLAM alignment makes their original evidence agree again.
    request["candidates"] = [(first, BOX, Vector3(1., 2., 0.), {}, response),
                             (second, BOX, Vector3(4., 2., 0.), {}, response)]
    node._localize = lambda view, box, *projection: (Vector3(10., 20., 0.), {"map_to_odom_stamp_ns": 100 * SECOND})
    node._tick()
    assert node.pending is None and len(node.result_pub.messages) == 1
    selected = node.result_pub.messages[0]
    assert selected.success
    assert (selected.position.x, selected.position.y) == (10., 20.)


@pytest.mark.parametrize("answer", ["<box>None</box>", "<box><0><0><998><998></box>",
    "<box><100><200><350><900></box><box><550><250><850><850></box>"])
def test_invalid_model_target_never_reaches_localization_or_success(node, answer):
    view = stored_view(node)
    request = active_query(node)
    node.future = Future()
    node.future.set_result({"request_id": request["id"], "answer": answer})
    node.job = {"kind": "query", "epoch": 0, "request_id": request["id"], "view": view}
    calls = []
    node._localize = lambda *args: calls.append(args)
    node._tick()
    assert not calls and node.pending is None
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success


def test_query_deadline_prevents_success_even_when_late_worker_finished(node):
    view = stored_view(node)
    request = active_query(node)
    request["wall_start"] -= 91
    node.future = Future()
    node.future.set_result({"request_id": request["id"],
        "answer": "<box><100><200><350><900></box>", "prompt": "the chair", "inference_seconds": 1.0})
    node.job = {"kind": "query", "epoch": 0, "request_id": request["id"], "view": view}
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    node._tick()
    assert node.pending is None
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert "time budget" in node.result_pub.messages[0].status_message


@pytest.mark.parametrize("command", ["go to chair 99", "do not go to chair0; go to the blue chair"])
def test_missing_or_qualified_id_fails_without_inference_or_fallback(node, command):
    node.p["enable_explicit_id_lookup"] = True
    view = stored_view(node)
    node.memory.remember(view, "chair", BOX, 1, 2)
    active_query(node, command, views=None)
    node._tick()
    assert node.pending is None
    assert len(node.result_pub.messages) == 1
    assert not node.result_pub.messages[0].success
    assert node.future is None
    assert not any(event["state"] == "selected" for event in node.events)


@pytest.mark.parametrize("command,kind", [
    ("go to a forklift_1", "gallery"),
    ("go to the red forklift_1", "gallery"),
    ("go to forklift_99", "gallery"),
    ("do not go to forklift_1; go to the blue forklift", "gallery"),
    ("go to forklift_1 or forklift_0", "gallery"),
    ("go to the forklift right of forklift_0", "query"),
])
def test_disabled_id_lookup_sends_wording_to_model(node, command, kind):
    node.p["enable_explicit_id_lookup"] = False
    node.memory = Memory(class_labels=["forklift"])
    view = stored_view(node)
    node.memory.remember(view, "forklift", BOX, 1, 2)
    node.memory.remember(view, "forklift", [350, 100, 520, 400], 4, 2)

    def forbidden_lookup(_):
        raise AssertionError("disabled ID parser was called")

    node.memory.lookup_id = forbidden_lookup
    active_query(node, command, views=None)
    node._tick()
    assert node.pending is not None and node.job["kind"] == kind
    assert len(node.inference_calls) == 1
    request = node.inference_calls[0][1]
    assert request["command"] == command
    assert request["expression"] == module.referring_expression(command)
    assert not node.result_pub.messages  # No navigation target before model output.
    assert not any(event["state"] == "selected" for event in node.events)


def test_clock_rollback_invalidates_memory_sensors_pending_and_completed_old_job(node):
    view = stored_view(node)
    node.memory.remember(view, "chair", BOX, 1, 2)
    request = active_query(node)
    node.images.append(view.payload["image"])
    node.scans.append(view.payload["scan"])
    node.catalog_queue.append((view, "person"))
    node.future = Future()
    node.future.set_result({"request_id": request["id"], "answer": "unused"})
    node.job = {"kind": "query", "epoch": 0, "request_id": request["id"], "view": view}
    node.now_ns = SECOND
    node._tick()
    assert node.epoch == 1 and node.pending is None
    assert node.memory.views == [] and node.memory.landmarks == {}
    assert not node.images and not node.scans and not node.odoms and not node.catalog_queue
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert "clock reset" in node.result_pub.messages[0].status_message


@pytest.mark.parametrize("worker_error", [False, True])
@pytest.mark.parametrize("job_kind", ["query", "gallery"])
def test_superseded_worker_result_never_enters_new_query(node, worker_error, job_kind):
    view = stored_view(node)
    old = active_query(node, "move to the first chair")
    node.future = Future()
    if worker_error:
        node.future.set_exception(ValueError("old request failed"))
    else:
        node.future.set_result({"request_id": old["id"], "answer": "must not be parsed"})
    node.job = {"kind": job_kind, "epoch": 0, "request_id": old["id"], "view": view}
    node._command_cb(String(data="move to the second chair"))
    current = node.pending
    node._tick()
    assert node.pending is current
    assert current["id"] != old["id"] and current["candidates"] == [] and current["errors"] == []
    assert node.result_pub.messages == []


@pytest.mark.parametrize('kind', ['description', 'gallery_description', 'object_id'])
def test_historical_target_keeps_evidence_stamp_and_current_query_pose(node, kind):
    view = stored_view(node, 10, pose=(-3, -4, 1))
    view.payload['metadata'].update(localization_source='aligned_depth', depth_stamp_ns=10 * SECOND)
    obj = node.memory.remember(view, "chair", BOX, 1, 2)
    active_query(node)
    response = {"answer": "a valid box", "prompt": "the chair", "inference_seconds": .4}
    geometry = {'localization_source': 'aligned_depth', 'map_to_odom_stamp_ns': 100 * SECOND}
    node._select(view, BOX, Vector3(1., 2., 0.), geometry, obj, response, kind)
    assert node.pending is None and len(node.result_pub.messages) == 1
    selected = node.result_pub.messages[0]
    assert selected.success and selected.object_id == "chair_0"
    assert module.stamp_ns(selected) == 10 * SECOND
    assert selected.frame_id == "map" and selected.position.x == 1
    assert node.shown_images == [view.payload["image"]]
    event = next(item for item in node.events if item["state"] == "selected")
    assert event["historical_view"] and event["observation_age_sec"] == 90
    assert event["query_robot_odom"] == [8., 9., 0.]
    assert event["observation_robot_odom"] == [-3, -4, 1]
    assert event['localization_source'] == 'aligned_depth'
    assert event['depth_stamp_ns'] == 10 * SECOND
    assert event['map_to_odom_stamp_ns'] == 100 * SECOND
    assert event['query_kind'] == kind


@pytest.mark.parametrize('kind', ['description', 'gallery_description', 'object_id'])
def test_unconfirmed_track_cannot_publish_any_navigation_target(node, kind):
    node.memory = Memory(confirm_classes=True)
    view = stored_view(node)
    obj = node.memory.remember(view, 'chair', BOX, 1., 2.)
    active_query(node)
    markers_before = list(node.marker_pub.messages)
    with pytest.raises(ValueError, match='not confirmed from independent viewpoints'):
        node._select(view, BOX, Vector3(1., 2., 0.), {}, obj, {}, kind)
    assert not node.result_pub.messages and node.marker_pub.messages == markers_before


def test_pending_track_published_separately_without_navigation_marker(node):
    import json
    node.memory = Memory(confirm_classes=True)
    view = stored_view(node)
    obj = node.memory.remember(view, 'chair', BOX, 1., 2.)
    node._publish_memory()
    payload = json.loads(node.memory_pub.messages[-1].data)
    assert payload['landmarks'] == []
    assert payload['pending_landmarks'][0]['object_id'] == obj.object_id
    assert len(node.memory_markers_pub.messages[-1].markers) == 1  # DELETEALL only


def test_depth_description_cannot_backfill_class_and_promote_pending_track(node):
    node.p["localization_mode"] = "depth"
    node.memory = Memory(confirm_classes=True)
    first = stored_view(node)
    obj = node.memory.remember(first, "chair", BOX, 1., 2.)
    second = stored_view(node, 20, pose=(2., 0., 0.))
    assert node.memory.remember(second, "object", BOX, 1., 2.) is obj
    assert not obj.confirmed and obj.label_votes == {"chair": 1}
    before = node.memory.serialize()
    request = active_query(node, "move to the chair beside the table")
    request["candidates"] = [(second, BOX, Vector3(1., 2., 0.), {}, {})]
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    node._tick()
    assert node.memory.serialize() == before
    assert not obj.confirmed and obj.object_id == "candidate_0"
    assert second.observations[0]["labels"] == ["object"]
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert "not confirmed from independent viewpoints" in node.result_pub.messages[0].status_message


def test_depth_description_retained_without_changing_original_class_evidence(node):
    node.p["localization_mode"] = "depth"
    node.memory = Memory(confirm_classes=True)
    first = stored_view(node)
    obj = node.memory.remember(first, "chair", BOX, 1., 2., support_points=[(1., 2.)])
    second = stored_view(node, 20, pose=(2., 0., 0.))
    node.memory.remember(second, "chair", BOX, 1., 2., support_points=[(1., 2.)])
    third = stored_view(node, 30, pose=(4., 0., 0.))
    assert node.memory.remember(third, "person", BOX, 1., 2., support_points=[(1., 2.)]) is obj
    # A description cannot turn this old person call into another chair vote.
    assert obj.confirmed and not obj.conflicted
    votes_before = dict(obj.label_votes)
    observations_before = [dict(o, labels=list(o["labels"])) for o in third.observations]
    request = active_query(node, "move to the chair beside the table")
    response = {"answer": "valid box", "prompt": "chair beside the table", "inference_seconds": .5}
    request["candidates"] = [(third, BOX, Vector3(1., 2., 0.), {}, response)]
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    node._tick()
    assert len(node.result_pub.messages) == 1 and node.result_pub.messages[0].success
    assert obj.descriptions == {request["command"]}
    assert obj.label_votes == votes_before
    assert third.observations == observations_before


def test_depth_description_without_catalog_identity_cannot_create_track(node):
    node.p["localization_mode"] = "depth"
    node.memory = Memory(confirm_classes=True)
    view = stored_view(node)
    request = active_query(node, "move to the chair beside the table")
    request["candidates"] = [(view, BOX, Vector3(1., 2., 0.), {}, {})]
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    node._tick()
    assert not node.memory.landmarks and not view.observations
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success


def test_robot_motion_prevents_historical_selection(node):
    view = stored_view(node)
    obj = node.memory.remember(view, "chair", BOX, 1, 2)
    active_query(node)
    node._odom_cb(odom(speed=.1))
    with pytest.raises(ValueError, match="robot moved"):
        node._select(view, BOX, Vector3(1, 2, 0), {}, obj, {}, "description")
    assert not node.result_pub.messages and not node.shown_images


@pytest.mark.parametrize('depth', [False, True])
def test_known_id_relocalizes_historical_evidence_without_model_inference(node, depth):
    node.p["enable_explicit_id_lookup"] = True
    view = stored_view(node)
    geometry = {}
    if depth:
        node.p['localization_mode'] = 'depth'
        view.payload['metadata'].update(localization_source='aligned_depth', depth_stamp_ns=10 * SECOND)
        geometry = {'localization_source': 'aligned_depth', 'depth_m': 2.}
    node.memory.remember(view, "chair", BOX, 1, 2)
    active_query(node, "go to chair0", views=None)
    calls = []
    node._localize = lambda selected_view, box, *projection: calls.append((selected_view, box)) or (Vector3(5., 6., 0.), geometry)
    node._tick()
    assert calls and all(item == (view, BOX) for item in calls)
    assert node.future is None and node.pending is None
    assert len(node.result_pub.messages) == 1
    assert node.result_pub.messages[0].success
    assert node.result_pub.messages[0].position.x == 5
    assert node.result_pub.messages[0].object_id == "chair_0"


def test_confirmed_id_selects_class_consistent_source_after_one_contradictory_view(node):
    node.p["enable_explicit_id_lookup"] = True
    node.p['localization_mode'] = 'depth'
    node.memory = Memory(class_labels=['chair', 'person'], confirm_classes=True)
    first = stored_view(node, 10, pose=(0., 0., 0.))
    obj = node.memory.remember(first, 'chair', BOX, 1., 2., support_points=[(1., 2.)])
    second = stored_view(node, 20, pose=(2., 0., 0.))
    node.memory.remember(second, 'chair', BOX, 1., 2., support_points=[(1., 2.)])
    wrong = stored_view(node, 30, pose=(4., 0., 0.))
    node.memory.remember(wrong, 'person', BOX, 1., 2., support_points=[(1., 2.)])
    assert obj.confirmed and not obj.conflicted and obj.object_id == 'chair_0'
    active_query(node, 'go to chair_0', views=None)
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    node._tick()
    assert node.pending is None and node.result_pub.messages[0].success
    assert node.shown_images == [second.payload['image']]
    assert node.events[-1]['view_id'] == second.view_id


def depth_relation_scene(node, other_label='chair'):
    node.p['localization_mode'] = 'depth'
    node.memory = Memory(max_views=32, confirm_classes=True)
    first = stored_view(node, 10, pose=(0., 0., 0.))
    obj = node.memory.remember(first, 'chair', BOX, 1., 0., support_points=[(1., 0.)])
    second = stored_view(node, 20, pose=(.7, 0., 0.))
    node.memory.remember(second, 'chair', BOX, 1., 0., support_points=[(1., 0.)])
    separate = stored_view(node, 30, pose=(3., 0., 0.))
    other = node.memory.remember(separate, other_label, BOX, -1., 0., support_points=[(-1., 0.)])
    node._localize = lambda *args: (Vector3(1., 0., 0.), {})
    node._refresh_positions = lambda: None
    return obj, other, second, separate


def test_depth_relation_cannot_ignore_pending_target_in_another_view(node):
    confirmed, pending, primary_view, pending_view = depth_relation_scene(node)
    primary, fallback = node.memory.representative_views(node.now_ns)
    assert primary == [primary_view] and fallback == [pending_view]
    before = node.memory.serialize()
    active_query(node, 'Move to the leftmost chair.', views=None)
    node._tick()
    assert not node.inference_calls and node.pending is None
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert pending.object_id in node.result_pub.messages[0].status_message
    assert 'unconfirmed class evidence in relational candidates' in node.result_pub.messages[0].status_message
    assert node.memory.serialize() == before and confirmed.confirmed


def test_depth_relation_with_unknown_target_class_checks_all_pending_classes(node):
    _, pending, _, _ = depth_relation_scene(node, 'person')
    active_query(node, 'Move to the leftmost object.', views=None)
    node._tick()
    assert not node.inference_calls and not node.result_pub.messages[0].success
    assert pending.object_id in node.result_pub.messages[0].status_message


def test_depth_relation_allows_unrelated_pending_class(node):
    confirmed, pending, primary, _ = depth_relation_scene(node, 'person')
    request = active_query(node, 'Move to the leftmost chair.', views=None)
    node._tick()
    assert node.job['kind'] == 'query' and node.job['view'] is primary
    node.future.set_result({'request_id': request['id'],
                           'answer': '<box><156><208><391><833></box>',
                           'prompt': 'the leftmost chair', 'inference_seconds': .1})
    node._tick()
    assert node.pending is None and node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == confirmed.object_id
    assert not pending.confirmed


def test_explicit_confirmed_id_remains_available_with_pending_same_class(node):
    node.p["enable_explicit_id_lookup"] = True
    confirmed, pending, _, _ = depth_relation_scene(node)
    active_query(node, 'go to ' + confirmed.object_id, views=None)
    node._tick()
    assert not node.inference_calls and node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == confirmed.object_id
    assert not pending.confirmed


def test_disabled_id_lookup_queries_confirmed_gallery_without_pending_track(node):
    node.p["enable_explicit_id_lookup"] = False
    confirmed, pending, _, _ = depth_relation_scene(node)
    active_query(node, "go to " + confirmed.object_id, views=None)
    node._tick()
    assert len(node.inference_calls) == 1 and node.pending is not None
    assert not node.result_pub.messages
    assert [entry.object_id for entry in node.job["gallery"].entries] == [confirmed.object_id]
    assert node.job["gallery"].excluded_objects == {pending.object_id: "unconfirmed"}
    assert not pending.confirmed


def test_depth_relation_rejects_relevant_conflicted_class(node):
    _, other, _, _ = depth_relation_scene(node)
    # The evidence/conflict rules themselves are tested in visual_memory;
    # here exercise the relation-retrieval boundary with a quarantined ID.
    other.confirmed, other.conflicted = True, True
    active_query(node, 'Move to the leftmost chair.', views=None)
    node._tick()
    assert not node.inference_calls and not node.result_pub.messages[0].success
    assert 'conflicting class evidence in relational candidates' in node.result_pub.messages[0].status_message


def test_query_retrieval_uses_rich_primary_and_keeps_ungrounded_view_for_fallback(node):
    rich = stored_view(node)
    node.memory.remember(rich, "chair", BOX, 1, 2)
    other_box = [300, 100, 500, 400]
    node.memory.remember(rich, "chair", other_box, 4, 2)
    partial = stored_view(node, 20, pose=(2, 0, 0))
    node.memory.remember(partial, "chair", BOX, 1, 2)
    empty = stored_view(node, 30, pose=(4, 0, 0))
    active_query(node, "move to the chair beside the other chair", views=None)
    node._localize = lambda view, box, *projection: (Vector3(1. if box == BOX else 4., 2., 0.), {})
    inferred = []
    node._infer = lambda view, kind, expression: inferred.append(view)
    node._step_query()
    assert inferred == [rich]
    assert list(node.pending["views"]) == []
    assert list(node.pending["fallback_views"]) == [empty]
    assert partial not in inferred and partial not in node.pending["fallback_views"]


def test_valid_primary_candidate_skips_ungrounded_fallback(node):
    primary = stored_view(node)
    fallback = stored_view(node, 20, pose=(2, 0, 0))
    request = active_query(node)
    request["fallback_views"] = deque([fallback])
    response = {"answer": "valid primary", "prompt": "chair", "inference_seconds": 1.0}
    request["candidates"] = [(primary, BOX, Vector3(1., 2., 0.), {}, response)]
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    inferred = []
    node._infer = lambda view, kind, expression: inferred.append(view)
    node._tick()
    assert not inferred
    assert node.pending is None and len(node.result_pub.messages) == 1
    assert node.result_pub.messages[0].success


def test_failed_primary_releases_ungrounded_fallback_once(node):
    fallback = stored_view(node)
    request = active_query(node)
    request["errors"] = ["primary returned no target"]
    request["fallback_views"] = deque([fallback])
    inferred = []
    node._infer = lambda view, kind, expression: inferred.append(view)
    node._tick()
    assert inferred == [fallback]
    assert not request["fallback_views"] and not request["views"]
    assert not node.result_pub.messages
    node._tick()  # fallback also had no usable result
    assert inferred == [fallback]
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success


def test_requested_table_rejects_a_known_chair_before_localization(node):
    node.memory = Memory(class_labels=["table", "chair"])
    view = stored_view(node)
    box = [64., 96., 224., 432.]
    node.memory.remember(view, "chair", box, 1, 2)
    request = active_query(node, "Move to the green table.")
    localized = []
    node._localize = lambda *args: localized.append(args) or (Vector3(1., 2., 0.), {})
    response = {"request_id": request["id"], "answer": "<box><100><200><350><900></box>"}
    with pytest.raises(ValueError, match="remembered chair, not requested table"):
        node._query_result({"view": view}, response)
    assert not localized and not request["candidates"]
    assert not node.result_pub.messages


def test_class_conflict_does_not_discard_an_existing_correct_candidate(node):
    node.memory = Memory(class_labels=["table", "chair"])
    table_view = stored_view(node)
    chair_view = stored_view(node, 20, pose=(2., 0., 0.))
    box = [64., 96., 224., 432.]
    node.memory.remember(table_view, "table", box, 1, 2)
    node.memory.remember(chair_view, "chair", box, 4, 2)
    request = active_query(node, "Move to the green table.")
    node._localize = lambda view, box, *projection: (Vector3(1. if view is table_view else 4., 2., 0.), {})
    response = {"request_id": request["id"], "answer": "<box><100><200><350><900></box>",
                "prompt": "the green table", "inference_seconds": .2}
    node._query_result({"view": table_view}, response)
    with pytest.raises(ValueError, match="remembered chair"):
        node._query_result({"view": chair_view}, response)
    assert len(request["candidates"]) == 1
    node._step_query()
    assert node.pending is None and len(node.result_pub.messages) == 1
    assert node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == "table_0"


def start_gallery(node, two_objects=False):
    view = stored_view(node)
    node.memory.remember(view, "chair", BOX, 1, 2)
    if two_objects:
        node.memory.remember(view, "chair", [350, 100, 520, 400], 4, 2)
    active_query(node, "Move to the blue chair.", views=None)
    localized = []
    node._localize = lambda source, box, *projection: localized.append((source, list(box))) or (Vector3(5., 6., 0.), {})
    node._tick()
    assert node.job["kind"] == "gallery" and node.pending["gallery_attempted"]
    return node.job["gallery"], localized


def gallery_answer(gallery, box):
    height, width = gallery.image_bgr.shape[:2]
    values = [round(value / (width if index % 2 == 0 else height) * 1000)
              for index, value in enumerate(box)]
    return "<box>" + "".join("<%d>" % value for value in values) + "</box>"


def complete_gallery(node, answer):
    node.future.set_result({"request_id": node.pending["id"], "answer": answer,
                           "prompt": "the blue chair", "inference_seconds": .2})


def test_gallery_uses_catalog_prompt_and_selects_original_sensor_box(node):
    gallery, localized = start_gallery(node)
    request = node.inference_calls[0][1]
    assert request["operation"] == "catalog"
    assert request["command"] == "Move to the blue chair."
    assert request["expression"] == "the blue chair"
    assert request["image_png"] == gallery.image_png
    entry = gallery.entries[0]
    assert entry.object_box != entry.source_box
    complete_gallery(node, gallery_answer(gallery, entry.object_box))
    node._tick()
    assert node.pending is None and len(node.result_pub.messages) == 1
    out = node.result_pub.messages[0]
    assert out.success and out.object_id == entry.object_id
    assert module.stamp_ns(out) == entry.view.stamp_ns
    assert (out.position.x, out.position.y) == (5., 6.)
    assert localized and all(source is entry.view and box == entry.source_box for source, box in localized)
    selected = next(event for event in node.events if event["state"] == "selected")
    assert selected["source"] == "visual_memory_gallery"
    assert selected["box_xyxy"] == entry.source_box
    assert selected["gallery_object_ids"] == [entry.object_id]


@pytest.mark.parametrize("rejection", ["absent", "mixed_invalid", "span", "malformed", "worker_error"])
def test_gallery_failure_is_terminal_and_never_falls_back_to_single_view(node, rejection):
    gallery, _ = start_gallery(node, two_objects=True)
    # Even if a stale caller left fallback views, the gallery decision must
    # finish this request instead of forcing a target out of a partial view.
    node.pending["views"] = deque([gallery.entries[0].view])
    node.pending["fallback_views"] = deque([gallery.entries[0].view])
    single_view_calls = []
    node._infer = lambda *args: single_view_calls.append(args)
    if rejection == "worker_error":
        node.future.set_exception(ValueError("worker could not decode this query"))
    else:
        first, second = [entry.object_box for entry in gallery.entries]
        answers = {
            "absent": "<box>None</box>",
            "mixed_invalid": gallery_answer(gallery, first) + "<box><0><0><1001><900></box>",
            "span": gallery_answer(gallery, [first[0], min(first[1], second[1]),
                                             second[2], max(first[3], second[3])]),
            "malformed": "not a grounding result",
        }
        complete_gallery(node, answers[rejection])
    node._tick()
    assert node.pending is None and not single_view_calls
    assert len(node.inference_calls) == 1
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert "gallery did not yield one reliable target" in node.result_pub.messages[0].status_message
    assert any(event["state"] == "gallery_rejected" for event in node.events)
    assert not any(event["state"] == "selected" for event in node.events)


@pytest.mark.parametrize("case", ["no_views", "uncatalogued", "missing_class", "bad_image"])
def test_nonspatial_query_without_usable_gallery_fails_without_raw_view_inference(node, case):
    node.memory = Memory(class_labels=["chair", "table"])
    if case != "no_views":
        view = stored_view(node)
        if case != "uncatalogued":
            node.memory.remember(view, "chair", BOX, 1, 2)
        if case == "bad_image":
            view.payload["image_png"] = "not PNG"
    command = "Move to the green table." if case == "missing_class" else "Move to the blue chair."
    active_query(node, command, views=None)
    node._tick()
    assert node.pending is None and not node.inference_calls
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert not any(event["state"] == "selected" for event in node.events)


def test_late_gallery_future_is_rejected_before_localization_or_selection(node):
    gallery, localized = start_gallery(node)
    initial_calls = len(localized)
    complete_gallery(node, gallery_answer(gallery, gallery.entries[0].object_box))
    node.pending["wall_start"] -= 91
    node._tick()
    assert len(localized) == initial_calls
    assert node.pending is None and len(node.result_pub.messages) == 1
    assert not node.result_pub.messages[0].success
    assert "time budget" in node.result_pub.messages[0].status_message


@pytest.mark.parametrize("kind", ["description", "object_id", "gallery_description"])
def test_common_selection_deadline_guards_all_query_paths(node, kind):
    view = stored_view(node)
    obj = node.memory.remember(view, "chair", BOX, 1, 2)
    request = active_query(node)
    request["wall_start"] -= 91
    with pytest.raises(ValueError, match="time budget"):
        node._select(view, BOX, Vector3(1., 2., 0.), {}, obj, {}, kind)
    assert not node.result_pub.messages and not node.shown_images


def test_selection_rechecks_deadline_after_expensive_memory_refresh(node, monkeypatch):
    wall = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: wall[0])
    view = stored_view(node)
    obj = node.memory.remember(view, "chair", BOX, 1, 2)
    active_query(node)
    node._publish_memory = lambda: wall.__setitem__(0, 91.)
    with pytest.raises(ValueError, match="time budget"):
        node._select(view, BOX, Vector3(1., 2., 0.), {}, obj, {}, "description")
    assert not node.result_pub.messages
    assert not any(marker.action == module.Marker.ADD for marker in node.marker_pub.messages)
    assert not any(event["state"] == "selected" for event in node.events)


def capture_snapshot(view, seconds=99.9, pose=(80., 0., .8)):
    payload = dict(view.payload)
    image = stamped(Image(), seconds, "camera_optical")
    image.width, image.height = 640, 480
    payload.update(image=image, odom_pose=pose,
                   metadata={"image_stamp_ns": module.stamp_ns(image)})
    return payload


def test_busy_catalog_survives_full_evidence_pool_and_commits_result(node):
    node.memory = Memory(max_views=32)
    for index in range(32):
        view = stored_view(node, 10 + index, pose=(2. * index, 0., 0.))
        node.memory.remember(view, "chair", BOX, 2. * index, 0.)
    busy_view = stored_view(node, 99.8, pose=(70., 0., 0.))
    assert len(node.memory.views) == 32
    assert sum(bool(view.observations) for view in node.memory.views) == 31
    node._refresh_positions = lambda: None
    node._localize = lambda *args: (Vector3(1000., 2., 0.), {})
    node._infer(busy_view, "catalog", None)
    future, job, retained = node.future, node.job, list(node.memory.views)
    captures = []
    node._capture = lambda: captures.append(True) or capture_snapshot(busy_view)
    node.last_capture_wall -= 4
    for _ in range(40):
        node._tick()
        node._observe()  # Direct capture callers obey the same admission rule.
    assert not captures and node.memory.views == retained
    assert node.future is future and node.job is job and not node.catalog_queue
    assert len(node.inference_calls) == 1

    future.set_result({"request_id": job["request_id"], "inference_seconds": .3,
        "answer": "<ref>chair</ref><box>None</box>"
                  "<ref>person</ref><box><100><200><350><900></box>"
                  "<ref>trash can</ref><box>None</box>"})
    node.last_capture_wall = module.time.monotonic()
    node._tick()
    assert node.future is None and not node.catalog_queue
    assert busy_view in node.memory.views
    assert len(busy_view.observations) == 1
    obj = node.memory.landmarks[busy_view.observations[0]["object_id"]]
    assert obj.label == "person" and obj.x == 1000. and obj.y == 2.
    update = next(event for event in node.events if event["state"] == "memory_updated")
    assert update["view_id"] == busy_view.view_id and update["object_ids"] == [obj.object_id]


def test_waiting_catalog_is_dispatched_before_any_new_capture(node):
    node.memory = Memory(max_views=1)
    queued = stored_view(node, 99.8)
    node.catalog_queue.append((queued, None))
    captures = []
    node._capture = lambda: captures.append(True) or capture_snapshot(queued)
    node.last_capture_wall -= 4
    node._observe()
    node._tick()
    assert not captures and node.memory.views == [queued]
    assert node.job["view"] is queued and node.job["kind"] == "catalog"
    assert node.future is not None and not node.catalog_queue
    assert len(node.inference_calls) == 1


def test_expired_waiting_catalog_clears_latched_busy_summary(node):
    node.memory = Memory(max_age_sec=1.)
    queued = stored_view(node, 98.)
    node.catalog_queue.append((queued, None))
    node._publish_memory()
    assert module.json.loads(node.memory_pub.messages[-1].data)["catalog_pending"] == 1
    node.memory.views_for_query(node.now_ns)  # A query legitimately expires it.
    node._tick()
    summary = module.json.loads(node.memory_pub.messages[-1].data)
    assert summary["catalog_pending"] == 0 and summary["catalog_idle"] is True
    assert not node.catalog_queue and not node.inference_calls


def test_repeated_queries_keep_priority_without_adding_more_catalog_views(node):
    node.p["enable_explicit_id_lookup"] = True
    old = stored_view(node, 99., pose=(0., 0., 0.))
    obj = node.memory.remember(old, "chair", BOX, 1., 2.)
    node._refresh_positions = lambda: None
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    captures = []
    node._capture = lambda: captures.append(True) or capture_snapshot(old)
    for _ in range(3):
        active_query(node, "go to " + obj.object_id, views=None)
        node._tick()
        assert node.pending is None and node.result_pub.messages[-1].success
        assert len(node.catalog_queue) == 1
        assert not node.inference_calls  # User query runs before queued catalog.
    assert len(captures) == 1 and len(node.memory.views) == 2
    queued, = node.catalog_queue
    assert queued[0] is not old and not queued[0].observations
    node._tick()
    assert node.job["view"] is queued[0] and len(node.inference_calls) == 1


def test_catalog_failure_releases_slot_and_resumes_capture(node):
    old = stored_view(node, 99.)
    node._infer(old, "catalog", None)
    node.future.set_exception(ValueError("model inference failed"))
    incoming = capture_snapshot(old)
    node._capture = lambda: incoming
    node.last_capture_wall -= 4
    node._tick()
    assert any(event["state"] == "observation_rejected" and
               "model inference failed" in event["reason"] for event in node.events)
    assert len(node.inference_calls) == 2 and node.future is not None
    assert node.job["view"].payload is incoming and node.job["view"] is not old
    assert not node.catalog_queue


@pytest.mark.parametrize("now, retained", [(103., True), (110., False)])
def test_catalog_completion_applies_normal_evidence_lifetime(node, now, retained):
    node.memory = Memory(max_age_sec=5.)
    view = stored_view(node, 99.)
    node._refresh_positions = lambda: None
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    node._infer(view, "catalog", "chair")
    node.future.set_result({"request_id": node.job["request_id"],
        "answer": "<box><100><200><350><900></box>", "inference_seconds": .3})
    node.now_ns = int(now * SECOND)
    node._tick()
    assert (view in node.memory.views) is retained
    assert bool(node.memory.landmarks) is retained
    assert bool(view.observations) is retained
    assert node.future is None and not node.catalog_queue
    summary = module.json.loads(node.memory_pub.messages[-1].data)
    assert summary["catalog_pending"] == 0 and summary["catalog_idle"] is True


def test_catalog_clock_reset_discards_old_result_before_accepting_new_capture(node):
    old = stored_view(node, 99.)
    node._infer(old, "catalog", None)
    future, request_id = node.future, node.job["request_id"]
    captures = []
    incoming = capture_snapshot(old, seconds=.8, pose=(0., 0., 0.))
    node._capture = lambda: captures.append(True) or incoming
    node.last_capture_wall -= 4
    node.now_ns = SECOND
    node._tick()
    assert node.epoch == 1 and node.memory.views == [] and not captures
    assert node.future is future and not node.catalog_queue
    future.set_result({"request_id": request_id, "answer": "invalid old result"})
    node._tick()
    assert len(captures) == 1 and len(node.memory.views) == 1
    assert node.memory.views[0].payload is incoming
    assert node.job["epoch"] == 1 and node.job["view"] is node.memory.views[0]
    assert not node.memory.landmarks
    assert not any(event["state"] in ("memory_updated", "observation_rejected") for event in node.events)


def test_active_gallery_query_keeps_snapshots_frozen_even_while_model_busy(node):
    gallery, _ = start_gallery(node)
    retained = list(node.memory.views)
    node.last_capture_wall -= 4
    captures = []
    node._capture = lambda: captures.append(True)
    node._tick()
    assert not captures and node.memory.views == retained
    assert node.pending is not None and not node.future.done()
    assert len(node.inference_calls) == 1


def test_generic_affordance_keeps_a_known_chair_candidate(node):
    node.memory = Memory(class_labels=["table", "chair", "object"])
    view = stored_view(node)
    node.memory.remember(view, "chair", [64., 96., 224., 432.], 1, 2)
    request = active_query(node, "Move to the red object I can sit on.")
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    response = {"request_id": request["id"], "answer": "<box><100><200><350><900></box>"}
    node._query_result({"view": view}, response)
    assert len(request["candidates"]) == 1
    assert request["candidates"][0][0] is view


def test_joint_inventory_pairs_returned_labels_with_surface_evidence(node):
    node.p["memory_classes"] = ["forklift", "storage rack"]
    view = stored_view(node)
    node._infer(view, "catalog", None)
    request = node.inference_calls[-1][1]
    assert request["operation"] == "inventory"
    assert request["categories"] == node.p["memory_classes"]
    node._localize = lambda *args: (Vector3(1., 2., 0.), {"support_points": [[7., 8.], [7.1, 8.]]})
    node._catalog_result(node.job, {"request_id": request["request_id"], "inference_seconds": .3,
        "answer": "<ref>forklift</ref><box>None</box><ref>storage rack</ref><box><100><200><350><900></box>"})
    obj, = node.memory.landmarks.values()
    assert obj.label == "storage_rack" and obj.support_points
    assert not obj.conflicted


@pytest.mark.parametrize("reverse", [False, True])
def test_nested_cross_class_inventory_is_rejected_before_localizing_either_box(node, reverse):
    node.p["memory_classes"] = ["forklift", "storage rack", "fire extinguisher"]
    view = stored_view(node)
    job = {"view": view, "label": None, "categories": node.p["memory_classes"], "request_id": "nested"}
    # Frozen view178: the purported rack is the forklift cab, fully inside
    # the forklift box but with IoU .444; the unrelated extinguisher is valid.
    groups = ["<ref>forklift truck</ref><box><622><38><915><525></box>",
              "<ref>storage shelving unit</ref><box><644><40><877><312></box>"]
    if reverse:
        groups.reverse()
    groups.append("<ref>fire extinguisher</ref><box><336><238><401><531></box>")
    localized = []
    node._localize = lambda source, box, *projection: localized.append(list(box)) or (
        Vector3(1., 2., 0.), {"support_points": [[7., 8.], [7.1, 8.]]})
    node._catalog_result(job, {"request_id": "nested", "answer": "".join(groups), "inference_seconds": .3})
    assert localized == [[215.04, 114.24, 256.64, 254.88]]
    obj, = node.memory.landmarks.values()
    assert obj.object_id == "fire_extinguisher_0" and not obj.conflicted
    assert len(view.observations) == 1
    assert len(view.payload["inventory_boxes"]) == 3  # Preserve all boxes for occlusion evidence.
    event = node.events[-1]
    assert event["object_ids"] == ["fire_extinguisher_0"]
    assert len(event["ambiguous_detections"]) == len(event["rejected"]) == 2
    assert {item["label"] for item in event["ambiguous_detections"]} == {"forklift", "storage_rack"}


def test_ambiguous_inventory_cannot_poison_an_existing_correct_track(node):
    node.p["memory_classes"] = ["forklift", "storage rack"]
    old = stored_view(node)
    obj = node.memory.remember(old, "forklift", BOX, 1., 2., support_points=[[7., 8.]])
    new = stored_view(node, 20, pose=(2., 0., 0.))
    calls = []
    node._localize = lambda source, box, *projection: calls.append(source) or (
        Vector3(1., 2., 0.), {"support_points": [[7., 8.]]})
    job = {"view": new, "label": None, "categories": node.p["memory_classes"], "request_id": "conflict"}
    node._catalog_result(job, {"request_id": "conflict", "inference_seconds": .3,
        "answer": "<ref>forklift</ref><box><622><38><915><525></box>"
                  "<ref>storage rack</ref><box><644><40><877><312></box>"})
    assert calls == [old]  # Existing map alignment refresh is still allowed.
    assert list(node.memory.landmarks.values()) == [obj]
    assert obj.label == "forklift" and not obj.conflicted
    assert obj.seen_known_labels == {"forklift"} and obj.view_ids == {old.view_id}
    assert new.observations == []


@pytest.mark.parametrize('geometry_case', ['separated', 'overlapping', 'failed'])
def test_supplemental_nested_inventory_needs_independent_separated_depth(node, geometry_case):
    node.p['localization_mode'] = 'depth'
    view = stored_view(node)
    job = {'view': view, 'label': None, 'categories': ['fire extinguisher', 'storage rack'],
           'request_id': 'supplement'}
    supplement = {'category': 'storage rack', 'prompt': 'single class storage rack',
                  'answer': '<box><100><100><900><800></box>', 'inference_seconds': .2}
    response = {'request_id': 'supplement', 'inference_seconds': .5,
                'answer': '<ref>fire extinguisher</ref><box><300><300><400><500></box>'
                          '<ref>storage shelving unit</ref><box>None</box>',
                'supplemental_inventory': [supplement]}
    calls = []

    def localize(source, box, *projection):
        calls.append(tuple(box))
        rack = box[0] == 64.
        if rack and geometry_case == 'failed':
            raise ValueError('multiple separated depth surfaces')
        x = 0. if rack or geometry_case == 'overlapping' else 2.
        return Vector3(x, 0., 0.), {
            'localization_source': 'aligned_depth', 'depth_m': 1. if rack else 3.,
            'support_points': [[x, 0.], [x + .1, 0.], [x, .1]]}

    node._localize = localize
    node._catalog_result(job, response)
    assert len(calls) == 2  # Reuse the frozen surface checks when committing.
    assert node.events[-1]['supplemental_inventory'] == [supplement]
    if geometry_case == 'separated':
        assert {obj.label for obj in node.memory.landmarks.values()} == {'fire_extinguisher', 'storage_rack'}
        assert node.events[-1]['depth_resolved_indices'] == [0, 1]
        assert not node.events[-1]['ambiguous_detections']
    else:
        assert not node.memory.landmarks and not view.observations
        assert len(node.events[-1]['ambiguous_detections']) == 2
        assert not node.events[-1]['depth_resolved_indices']


def test_unambiguous_same_class_instances_keep_separate_ids_and_cannot_link(node):
    view = stored_view(node)
    # Even nearby/same sampled support must not merge separate boxes seen
    # together. The ambiguity prefilter leaves same-class identity to Memory.
    node._localize = lambda *args: (Vector3(1., 2., 0.), {"support_points": [[7., 8.]]})
    node._catalog_result({"view": view, "label": None, "categories": ["forklift"], "request_id": "two"},
        {"request_id": "two", "inference_seconds": .3,
         "answer": "<ref>forklift truck</ref><box><100><100><300><500></box>"
                   "<box><600><100><800><500></box>"})
    first, second = node.memory.landmarks.values()
    assert [first.object_id, second.object_id] == ["forklift_0", "forklift_1"]
    assert first.cannot_link == {second.object_id} and second.cannot_link == {first.object_id}
    assert not first.conflicted and not second.conflicted
    assert node.events[-1]["ambiguous_detections"] == []


def test_expired_inventory_jobs_do_not_delay_next_retained_view(node):
    expired = stored_view(node)
    kept = stored_view(node, 20, pose=(2., 0., 0.))
    node.memory.views.remove(expired)
    node.catalog_queue.extend([(expired, None)] * 20 + [(kept, None)])
    node._tick()
    assert len(node.inference_calls) == 1
    assert node.job["view"] is kept and not node.catalog_queue


def test_failed_last_inventory_clears_latched_busy_status(node):
    import json
    view = stored_view(node)
    node._infer(view, "catalog", None)
    node._publish_memory()
    assert json.loads(node.memory_pub.messages[-1].data)["catalog_pending"] == 1
    node.future.set_exception(ValueError("malformed model output"))
    node._tick()
    summary = json.loads(node.memory_pub.messages[-1].data)
    assert summary["catalog_idle"] and summary["catalog_pending"] == 0


def test_query_capture_publishes_pending_inventory_before_gallery_finishes(node):
    import json
    old = stored_view(node)
    node.memory.remember(old, "chair", BOX, 1., 2.)
    node._localize = lambda *args: (Vector3(1., 2., 0.), {})
    node._publish_memory()
    assert json.loads(node.memory_pub.messages[-1].data)["catalog_idle"]

    image = stamped(Image(), 99.8, "camera_optical")
    image.width, image.height = 640, 480
    snapshot = {**old.payload, "image": image, "odom_pose": (2., 0., 0.)}
    node._capture = lambda: snapshot
    active_query(node, "move to the chair", views=None)
    node._step_query()

    assert node.job["kind"] == "gallery" and not node.future.done()
    assert len(node.catalog_queue) == 1
    summary = json.loads(node.memory_pub.messages[-1].data)
    assert summary["catalog_pending"] == 1
    assert not summary["catalog_idle"]


def test_attitude_rejects_climbed_obstacle_even_with_planar_wheel_odom(node):
    node.imus = deque([imu(99.8, math.radians(36))])
    assert module.odom_pose(node.odoms[-1])[2] == 0
    with pytest.raises(ValueError, match='robot tilted'):
        node._validate_attitude(int(99.8 * SECOND))


def test_attitude_accepts_yaw_and_small_floor_tilt(node):
    message = imu(99.8, math.radians(5))
    node.imus = deque([message])
    assert node._validate_attitude(int(99.8 * SECOND)) == pytest.approx(5)
    message.orientation.y = 0.
    message.orientation.z = 1.
    message.orientation.w = 0.
    assert node._validate_attitude(int(99.8 * SECOND)) == pytest.approx(0)


@pytest.mark.parametrize('kind', ['missing', 'stale', 'invalid', 'unavailable'])
def test_attitude_requires_valid_synchronized_imu(node, kind):
    if kind == 'missing':
        node.imus.clear()
    elif kind == 'stale':
        node.imus = deque([imu(98)])
    elif kind == 'invalid':
        node.imus[-1].orientation.w = 0.
    else:
        node.imus[-1].orientation_covariance[0] = -1.
    with pytest.raises(ValueError):
        node._validate_attitude(int(99.8 * SECOND))


def depth_hint_catalog(node, answer, *, categories=('chair', 'person'), seconds=99.):
    """Real catalog parser/Memory/hint store; only the depth measurement is controlled."""
    node.p['localization_mode'] = 'depth'
    node.memory = Memory(confirm_classes=True)
    view = stored_view(node, seconds, pose=(4., 5., .3))
    view.payload['camera_to_odom'] = Transform3D(
        Vector3(4.1, 5.2, .4), Quaternion(-.5, .5, -.5, .5))
    node._refresh_positions = lambda: None
    job = {'view': view, 'label': None, 'categories': list(categories), 'request_id': 'hint_catalog'}
    response = {'request_id': 'hint_catalog', 'answer': answer, 'inference_seconds': .2}
    return view, job, response


def measured_depth_ambiguity(box, *, usable=True):
    groups = [
        {'odom_xyz': [[2., 1., .1], [2.1, 1., .2], [2., 1.1, .3]],
         'pixels': [[120, 180], [124, 184], [128, 188]], 'count': 30},
        {'odom_xyz': [[4., 1., .1], [4.1, 1., .2], [4., 1.1, .3]],
         'pixels': [[180, 220], [184, 224], [188, 228]], 'count': 40},
    ]
    return module.AmbiguousDepthError(box, groups if usable else (), 2 if usable else 9,
                                      None if usable else 'too_many_surface_groups')


def test_ambiguous_depth_creates_only_observation_hint_and_other_detection_is_remembered(node, monkeypatch):
    view, job, response = depth_hint_catalog(node,
        '<ref>chair</ref><box><100><200><350><900></box>'
        '<ref>person</ref><box><600><100><800><700></box>')
    calls = []
    remember = node.memory.remember
    def record(*args, **kwargs):
        calls.append(args[1])
        return remember(*args, **kwargs)
    monkeypatch.setattr(node.memory, 'remember', record)
    errors = []
    def localize(source, box, *projection):
        assert source is view
        if box[0] == 64.:
            error = measured_depth_ambiguity(box);errors.append(error)
            raise error
        return Vector3(7., 3., 0.), {'localization_source': 'aligned_depth',
                                    'support_points': [[7., 3.], [7.1, 3.], [7., 3.1]]}
    node._localize = localize
    node._catalog_result(job, response)
    assert calls == ['person']  # Never guess a chair position or add a class vote.
    person, = node.memory.landmarks.values()
    assert person.label == 'person'
    assert len(view.observations) == 1 and view.observations[0]['object_id'] == person.object_id
    hint, = node.observation_hints.serialize(node.now_ns)
    assert hint['label'] == 'chair' and hint['status'] == 'unresolved'
    assert hint['source_view_id'] == view.view_id and hint['stamp_ns'] == view.stamp_ns
    assert hint['source_pose'] == [4., 5., .3]
    assert hint['source_camera'][:2] == [4.1, 5.2]
    assert hint['surface_groups'] == list(errors[0].groups)
    assert not any(key in hint for key in ('x', 'y', 'object_id', 'confirmed'))
    assert node.events[-1]['depth_observation_hints'] == [hint['hint_id']]
    assert node.events[-1]['object_ids'] == [person.object_id]
    assert not node.result_pub.messages
    node._publish_memory()
    payload = module.json.loads(node.memory_pub.messages[-1].data)
    assert payload['observation_hints'] == [hint]
    assert not payload['landmarks']  # A single valid depth view still needs normal class confirmation.
    assert {obj['object_id'] for obj in payload['pending_landmarks']} == {person.object_id}
    assert payload['observation_hints_capacity_rejections'] == 0


def test_hint_cannot_satisfy_an_indexed_navigation_query(node):
    view, job, response = depth_hint_catalog(node,
        '<ref>chair</ref><box><100><200><350><900></box>', categories=('chair',))
    def localize(source, box, *projection):
        raise measured_depth_ambiguity(box)
    node._localize = localize
    node._catalog_result(job, response)
    assert not node.memory.landmarks and not view.observations
    assert len(node.observation_hints.serialize(node.now_ns)) == 1
    active_query(node, 'go to chair 0', views=None)
    node._tick()
    assert node.pending is None
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert not any(event['state'] == 'selected' for event in node.events)
    assert not node.inference_calls


@pytest.mark.parametrize('kind', ['left_clipped', 'right_clipped', 'whole_image', 'ordinary_error', 'unusable_hint'])
def test_rejected_detection_never_becomes_observation_hint(node, kind):
    encoded = {'left_clipped': '<0><200><350><900>',
               'right_clipped': '<600><200><1000><900>',
               'whole_image': '<0><0><1000><1000>'}.get(kind, '<100><200><350><900>')
    view, job, response = depth_hint_catalog(node, '<ref>chair</ref><box>'+encoded+'</box>',
                                            categories=('chair',))
    localized = []
    def localize(source, box, *projection):
        localized.append(box)
        if kind == 'ordinary_error':
            raise ValueError('insufficient valid depth pixels')
        raise measured_depth_ambiguity(box, usable=kind != 'unusable_hint')
    node._localize = localize
    node._catalog_result(job, response)
    if kind in ('left_clipped', 'right_clipped', 'whole_image'):
        assert localized == []  # Existing visibility rejection precedes localization.
    else:
        assert len(localized) == 1
    assert not node.memory.landmarks and not view.observations
    assert not getattr(node, 'observation_hints', None)
    assert node.events[-1]['depth_observation_hints'] == []
    assert node.events[-1]['rejected']
    assert not node.result_pub.messages


def test_cross_class_ambiguous_boxes_do_not_escape_via_depth_hints(node):
    view, job, response = depth_hint_catalog(node,
        '<ref>forklift</ref><box><100><100><800><900></box>'
        '<ref>storage rack</ref><box><200><200><600><700></box>',
        categories=('forklift', 'storage rack'))
    localized = []
    def localize(source, box, *projection):
        localized.append(box)
        raise measured_depth_ambiguity(box)
    node._localize = localize
    node._catalog_result(job, response)
    assert len(localized) == 2  # Independent depth checks cannot resolve the cross-class pair.
    assert len(node.events[-1]['ambiguous_detections']) == 2
    assert not node.events[-1]['depth_resolved_indices']
    assert node.events[-1]['depth_observation_hints'] == []
    hint, = node.observation_hints.serialize(node.now_ns)
    assert hint['hint_kind'] == 'class_ambiguity' and hint['label'] == ''
    assert hint['label_candidates'] == ['forklift', 'storage_rack']
    assert hint['surface_groups'] == list(measured_depth_ambiguity(localized[0]).groups)
    assert not node.memory.landmarks and not view.observations and not node.result_pub.messages


def test_clock_rollback_clears_hint_payload_and_discards_old_catalog_future(node):
    view, job, response = depth_hint_catalog(node,
        '<ref>chair</ref><box><100><200><350><900></box>', categories=('chair',))
    def localize(source, box, *projection):
        raise measured_depth_ambiguity(box)
    node._localize = localize
    node._catalog_result(job, response)
    hint, = node.observation_hints.serialize(node.now_ns)
    assert hint['epoch'] == 0
    node.future = Future();node.future.set_result(response)
    node.job = {'kind': 'catalog', 'epoch': 0, **job}
    event_count = len(node.events)
    node.now_ns = SECOND
    node._tick()
    assert node.epoch == node.observation_hints.epoch == 1
    assert node.observation_hints.serialize(node.now_ns) == []
    assert not node.observation_hints.known_surfaces and not node.memory.landmarks
    payload = module.json.loads(node.memory_pub.messages[-1].data)
    assert payload['observation_hints'] == [] and payload['observation_hints_capacity_rejections'] == 0
    assert not any(e['state'] == 'memory_updated' for e in node.events[event_count:])
    assert not node.result_pub.messages


@pytest.mark.parametrize("mode,top,expected", [("depth", 0, False), ("depth", 10, True), ("laser", 0, True)])
def test_catalog_records_class_box_completeness_without_discarding_geometry(node, mode, top, expected):
    node.p['localization_mode'] = mode
    node.memory = Memory(confirm_classes=(mode == 'depth'))
    view = stored_view(node, 99, (0., 0., 0.))
    node._refresh_positions = lambda: None
    node._localize = lambda *_: (Vector3(1., 2., 0.), {
        'localization_source': 'aligned_depth' if mode == 'depth' else 'laser',
        'support_points': [[1., 2.], [1.1, 2.]]})
    job = {'view': view, 'label': None, 'categories': ['chair'], 'request_id': 'quality'}
    response = {'request_id': 'quality', 'inference_seconds': .2,
                'answer': '<ref>chair</ref><box><200><%d><600><700></box>' % top}
    node._catalog_result(job, response)
    obj, = node.memory.landmarks.values()
    assert obj.support_points and obj.label_votes == {'chair': 1}
    assert view.observations[0]['label_completeness'] == {'chair': expected}
    event = next(e for e in node.events if e['state'] == 'memory_updated')
    assert not event['rejected'] and len(event['observations']) == 1
    assert event['observations'][0]['class_evidence_complete'] is expected


def remembered_cached_depth(node, seconds=10., point=Vector3(2., 3., 1.), pose=(0., 0., 0.)):
    view = stored_view(node, seconds, pose)
    view.payload.update(depth_m=object(), camera_to_odom=IDENTITY,
                        depth_geometry={tuple(BOX): (point, {'localization_source':'aligned_depth'})})
    obj = node.memory.remember(view, 'chair', BOX, 2., 3.)
    return view, obj


def test_memory_publication_uses_one_transform_for_all_cached_observations(node, monkeypatch):
    _, obj = remembered_cached_depth(node, 10., Vector3(2., 3., 1.))
    _, same = remembered_cached_depth(node, 20., Vector3(4., 5., 1.), (1., 0., 0.))
    assert obj.object_id == same.object_id
    calls = []
    def lookup(*args):
        calls.append(args)
        # Repeated latest lookups would expose inconsistent frames.
        return transform(100., (10.*len(calls), 20.*len(calls), 0.))
    node.tf_buffer.lookup_transform = lookup
    monkeypatch.setattr(module, 'localize_depth_box', lambda *_: pytest.fail('cached depth must not rerun'))
    node._publish_memory()
    payload = module.json.loads(node.memory_pub.messages[-1].data)
    assert len(calls) == 1
    assert (obj.x, obj.y) == pytest.approx((13., 24.))
    assert payload['epoch'] == 0 and payload['publish_stamp_ns'] == 100*SECOND
    projection = payload['map_projection']
    assert projection['valid'] and projection['object_ids'] == [obj.object_id]
    assert projection['transform'] == {'parent':'map', 'child':'odom', 'stamp_ns':100*SECOND,
                                       'xyz':[10.,20.,0.], 'quaternion':[0.,0.,0.,1.]}


def test_stationary_memory_heartbeat_updates_map_xy_without_votes_capture_or_inference(node):
    _, obj = remembered_cached_depth(node)
    before = module.json.dumps(node.memory.serialize(), sort_keys=True)
    calls = []
    node._observe = lambda: pytest.fail('heartbeat cannot capture an image')
    node.memory.remember = lambda *_a, **_k: pytest.fail('heartbeat cannot add a vote')
    node.tf_buffer.lookup_transform = lambda *args: calls.append(args) or transform(100., (.02, .03, 0.))
    node._memory_heartbeat()
    assert (obj.x, obj.y) == pytest.approx((2.02, 3.03))
    after = node.memory.serialize()
    for row in after['landmarks']:
        row['x'], row['y'] = 2., 3.
    assert module.json.dumps(after, sort_keys=True) == before
    assert len(calls) == 1 and not node.inference_calls
    assert len(node.memory_pub.messages) == 1


def test_future_projection_is_interpolated_at_one_captured_clock(node):
    _, obj = remembered_cached_depth(node)
    calls = []
    def lookup(target, source, stamp):
        calls.append(stamp.nanoseconds)
        return transform(100.2 if not stamp.nanoseconds else 100.,
                         (99., 99., 0.) if not stamp.nanoseconds else (.1, .2, 0.))
    node.tf_buffer.lookup_transform = lookup
    node._publish_memory()
    payload = module.json.loads(node.memory_pub.messages[-1].data)
    assert calls == [0,100*SECOND]
    assert payload['map_projection']['valid']
    assert payload['map_projection']['transform']['stamp_ns'] == 100*SECOND
    assert (obj.x, obj.y) == pytest.approx((2.1,3.2))


@pytest.mark.parametrize('failure', ['stale', 'future_wrong_stamp', 'frames', 'nonfinite', 'zero_quaternion', 'missing'])
def test_bad_projection_metadata_never_claims_old_coordinates_are_current(node, failure):
    _, obj = remembered_cached_depth(node)
    def lookup(target, source, stamp):
        if failure == 'missing':
            raise module.tf2_ros.TransformException('missing transform')
        value = transform(99. if failure == 'stale' else 100.)
        if failure == 'future_wrong_stamp':
            value = transform(100.2 if not stamp.nanoseconds else 99.9)
        elif failure == 'frames':
            value.child_frame_id = 'wrong_odom'
        elif failure == 'nonfinite':
            value.transform.translation.x = float('nan')
        elif failure == 'zero_quaternion':
            value.transform.rotation.w = 0.
        return value
    node.tf_buffer.lookup_transform = lookup
    node._publish_memory()
    payload = module.json.loads(node.memory_pub.messages[-1].data)
    assert not payload['map_projection']['valid']
    assert payload['map_projection']['errors'] and payload['map_projection']['transform'] is None
    assert (obj.x, obj.y) == (2.,3.)


def test_incomplete_reprojection_is_explicit_and_not_snapshot_valid(node, monkeypatch):
    good, obj = remembered_cached_depth(node)
    bad, _ = remembered_cached_depth(node, 20., pose=(1.,0.,0.))
    bad.payload['depth_geometry'].clear()
    monkeypatch.setattr(module, 'localize_depth_box', lambda *_: (_ for _ in ()).throw(ValueError('no depth')))
    node._publish_memory()
    projection = module.json.loads(node.memory_pub.messages[-1].data)['map_projection']
    assert not projection['valid'] and projection['errors']
    assert obj.object_id in projection['object_ids']
    assert projection['transform']['child'] == 'odom'
    assert f'{obj.object_id}/{bad.view_id}' in projection['errors'][0]


def test_memory_projection_rejects_mixed_odom_frames(node):
    remembered_cached_depth(node)
    other, _ = remembered_cached_depth(node, 20., pose=(1.,0.,0.))
    other.payload['odom_frame'] = 'other_odom'
    node._publish_memory()
    projection = module.json.loads(node.memory_pub.messages[-1].data)['map_projection']
    assert not projection['valid'] and projection['transform'] is None
    assert 'share one odometry frame' in projection['errors'][0]


def test_heartbeat_does_not_publish_previous_epoch_after_clock_rollback(node):
    remembered_cached_depth(node)
    node.now_ns = SECOND
    node._memory_heartbeat()
    assert not node.memory_pub.messages
    node._observe = lambda: None
    node._tick()
    payload = module.json.loads(node.memory_pub.messages[-1].data)
    assert payload['epoch'] == 1 and payload['publish_stamp_ns'] == SECOND
    assert payload['landmarks'] == [] and payload['map_projection']['transform'] is None
