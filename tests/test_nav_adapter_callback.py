"""ROS message checks without launching nodes, Gazebo, or navigation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
import tf2_ros
from rclpy.time import Time
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PointStamped, TransformStamped

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/tb3_nav_adapter"))
from tb3_nav_adapter.nav_goal_adapter_node import NavGoalAdapterNode


def adapter(buffer):
    goals, warnings = [], []
    logger = SimpleNamespace(info=lambda _: None, debug=lambda _: None,
                             warn=warnings.append)
    obj = SimpleNamespace(
        _target_frame="map", _tf_timeout=0.0, _tf_buffer=buffer,
        _approach_dist=0.5, _min_standoff=0.3,
        _goal_pub=SimpleNamespace(publish=goals.append),
        get_logger=lambda: logger,
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=42)),
    )
    return obj, goals, warnings


def target(frame="map"):
    return SimpleNamespace(
        success=True, frame_id=frame,
        header=Header(frame_id=frame, stamp=Time(seconds=10).to_msg()),
        position=Point(x=0.0, y=0.0),
        object_id="selected_7", semantic_name="the red chair",
    )


def robot_transform(*args, **kwargs):
    transform = TransformStamped()
    transform.transform.translation.x = -2.0
    return transform


def test_map_origin_target_produces_robot_relative_goal():
    obj, goals, _ = adapter(SimpleNamespace(lookup_transform=robot_transform))
    NavGoalAdapterNode._query_cb(obj, target())
    assert len(goals) == 1
    assert goals[0].header.frame_id == "map"
    assert goals[0].pose.position.x == pytest.approx(-0.5)


def test_missing_robot_transform_never_publishes_fallback():
    def unavailable(*args, **kwargs):
        raise tf2_ros.LookupException("no map to robot transform")
    obj, goals, warnings = adapter(SimpleNamespace(lookup_transform=unavailable))
    NavGoalAdapterNode._query_cb(obj, target())
    assert goals == []
    assert "no goal" in warnings[0]


def test_body_frame_target_transform_preserves_observation_timestamp():
    observed_stamps = []
    def transform(point, frame, **kwargs):
        observed_stamps.append(point.header.stamp.sec)
        result = PointStamped()
        result.header.frame_id = frame
        result.point = Point(x=0.0, y=0.0)
        return result
    obj, goals, _ = adapter(SimpleNamespace(
        transform=transform, lookup_transform=robot_transform))
    NavGoalAdapterNode._query_cb(obj, target("base_link"))
    assert observed_stamps == [10]
    assert len(goals) == 1
    assert goals[0].header.stamp.sec == 42


def test_missing_frame_never_publishes_goal():
    obj, goals, warnings = adapter(SimpleNamespace())
    NavGoalAdapterNode._query_cb(obj, target(""))
    assert goals == []
    assert "no coordinate frame" in warnings[0]
