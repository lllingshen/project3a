"""Command routing changes only the drive plugin ROS settings, in both modes."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("drive_robot_test", ROOT / "src/tb3_bringup/launch/drive_robot.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SOURCE = '''<sdf version="1.6"><model name="robot"><link name="base">
<inertial><mass>1.4</mass></inertial><collision name="body"><geometry><box><size>.3 .2 .1</size></box></geometry></collision>
<sensor name="camera" type="camera"><camera><horizontal_fov>1.08</horizontal_fov></camera></sensor></link>
<joint name="left" type="revolute"><axis><xyz>0 1 0</xyz></axis></joint>
<plugin name="drive" filename="libgazebo_ros_diff_drive.so"><wheel_separation>.287</wheel_separation>
<max_wheel_torque>20</max_wheel_torque><max_wheel_acceleration>1</max_wheel_acceleration>
<ros><remapping>odom:=/odom</remapping><remapping>cmd_vel:=/old_command</remapping></ros></plugin>
</model></sdf>'''


def canonical(element):
    return element.tag, sorted(element.attrib.items()), (element.text or "").strip(), [canonical(c) for c in element]


@pytest.mark.parametrize("camera_type", ["camera", "depth"])
def test_drive_remap_qos_and_every_physical_camera_parameter_unchanged(camera_type):
    source = SOURCE.replace('type="camera"', 'type="' + camera_type + '"')
    before, after = ET.fromstring(source), ET.fromstring(MODULE.drive_sdf_text(source))
    plugin = after.find("model/plugin")
    assert [m.text for m in plugin.findall("ros/remapping")] == [
        "cmd_vel:=/tb3_drive/cmd_vel", "odom:=/odom"]
    subscription = plugin.find("ros/qos/topic[@name='/tb3_drive/cmd_vel']/subscription")
    assert subscription.findtext("history") == "keep_last"
    assert subscription.find("history").get("depth") == "10"
    assert subscription.findtext("reliability") == "reliable"
    assert subscription.findtext("durability") == "volatile"
    for root in (before, after):
        drive = root.find("model/plugin")
        drive.remove(drive.find("ros"))
    assert canonical(after) == canonical(before)


def test_routing_is_idempotent_and_generated_files_are_unique_without_changing_source(tmp_path):
    source = tmp_path / "robot.sdf"
    source.write_text(SOURCE)
    first = Path(MODULE.make_drive_sdf(source, tmp_path))
    second = Path(MODULE.make_drive_sdf(source, tmp_path))
    assert first != second and first.read_text() == second.read_text()
    assert source.read_text() == SOURCE
    assert MODULE.drive_sdf_text(first.read_text()) == first.read_text()
    from lxml import etree
    assert etree.fromstring(first.read_text()).tag == "sdf"


@pytest.mark.parametrize("source", ["<sdf/>", SOURCE.replace("libgazebo_ros_diff_drive.so", "unknown.so"),
                                   SOURCE.replace("</model>", '<plugin filename="libgazebo_ros_diff_drive.so"/></model>')])
def test_missing_or_ambiguous_drive_plugin_fails_explicitly(source):
    with pytest.raises(ValueError):
        MODULE.drive_sdf_text(source)


def test_actual_upstream_robot_only_changes_drive_ros_configuration():
    source = Path("/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf")
    if not source.exists():
        pytest.skip("TurtleBot3 Gazebo unavailable")
    before = ET.fromstring(source.read_text())
    after = ET.fromstring(MODULE.drive_sdf_text(source.read_text()))
    for root in (before, after):
        drive = root.find("model/plugin[@filename='libgazebo_ros_diff_drive.so']")
        drive.remove(drive.find("ros"))
    assert canonical(after) == canonical(before)
