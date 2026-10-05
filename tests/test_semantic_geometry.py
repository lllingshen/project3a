"""Regression checks for map-origin versus robot-relative target geometry."""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" / name)
                for name in ("tb3_query", "tb3_nav_adapter")]

from tb3_query.query_core import MemoryObject, select_target
from tb3_nav_adapter.goal_adapter_core import compute_approach_pose


def objects():
    return [MemoryObject("person_0", "person", 0.0, 0.0, 0.9),
            MemoryObject("person_1", "person", 4.0, 0.0, 0.8)]


def test_nearest_is_relative_to_robot_not_map_origin():
    result = select_target(objects(), "person", "person", "go to person",
                           robot_x=3.0, robot_y=0.0)
    assert result.object_id == "person_1"
    assert (result.x, result.y) == (4.0, 0.0)


def test_index_keeps_requested_instance_even_when_farther():
    result = select_target(objects(), "person", "person", "go to person 0",
                           desired_index=0, robot_x=3.0, robot_y=0.0)
    assert result.success and result.object_id == "person_0"


def test_unobserved_index_fails_without_substitution():
    result = select_target(objects(), "person", "person", "go to person 2",
                           desired_index=2, robot_x=3.0)
    assert not result.success and not result.object_id


def test_target_at_map_origin_has_valid_approach():
    # Robot west of a central target: stop half a metre to its west.
    goal = compute_approach_pose(0.0, 0.0, robot_x=-2.0, robot_y=0.0)
    assert goal == pytest.approx((-0.5, 0.0, 0.0))


def test_standoff_faces_target_from_current_robot_side():
    goal = compute_approach_pose(2.0, 3.0, robot_x=2.0, robot_y=5.0)
    assert goal == pytest.approx((2.0, 3.5, -math.pi / 2))


def test_geometry_is_invariant_to_map_translation():
    a = compute_approach_pose(2.0, 1.0, robot_x=-1.0, robot_y=4.0)
    b = compute_approach_pose(12.0, -6.0, robot_x=9.0, robot_y=-3.0)
    assert b == pytest.approx((a[0] + 10.0, a[1] - 7.0, a[2]))


def test_close_target_is_measured_from_robot():
    assert compute_approach_pose(10.1, 10.0, robot_x=10.0, robot_y=10.0) is None


def test_body_frame_caller_keeps_original_geometry():
    assert compute_approach_pose(2.0, 0.0) == pytest.approx((1.5, 0.0, 0.0))
