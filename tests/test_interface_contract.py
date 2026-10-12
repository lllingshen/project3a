"""Check the generated ROS transport after extracting the shared interface."""
import pytest

pytest.importorskip("rclpy")


def test_perception_adapter_and_coordinator_share_the_serializable_message():
    from rclpy.serialization import deserialize_message, serialize_message
    from tb3_interfaces.msg import SemanticQueryResult
    from tb3_locateanything.locateanything_node import SemanticQueryResult as PerceptionResult
    from tb3_nav_adapter.nav_goal_adapter_node import SemanticQueryResult as AdapterResult
    from tb3_coordinator.coordinator_node import SemanticQueryResult as CoordinatorResult

    assert PerceptionResult is AdapterResult is CoordinatorResult is SemanticQueryResult
    selected = PerceptionResult()
    selected.success = True
    selected.query_text = "go to the white chair"
    selected.object_id = "chair_0"
    selected.semantic_name = selected.detector_label = "chair"
    selected.frame_id = selected.header.frame_id = "map"
    selected.position.x = 1.5
    selected.position.y = -0.25
    received = deserialize_message(serialize_message(selected), CoordinatorResult)
    assert received == selected


def test_installed_workspace_has_no_removed_pipeline_packages():
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

    assert get_package_share_directory("tb3_interfaces")
    for package in ("tb3_detector", "tb3_localizer", "tb3_memory", "tb3_query"):
        with pytest.raises(PackageNotFoundError):
            get_package_share_directory(package)
