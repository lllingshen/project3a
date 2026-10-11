"""The RGB-D model must preserve physics and use one shared optical camera."""

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rgbd_robot", ROOT / "src/tb3_bringup/launch/rgbd_robot.py"
)
rgbd_robot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rgbd_robot)

SOURCE = """<sdf version="1.6"><model name="turtlebot3_waffle_pi">
<link name="base"><collision name="body"><geometry><box>
<size>.3 .3 .1</size></box></geometry></collision></link>
<link name="base_scan"><sensor name="scan" type="ray"><update_rate>5</update_rate></sensor></link>
<link name="camera_rgb_optical_frame"><pose>.076 0 .093 0 0 0</pose>
<sensor name="camera" type="camera"><update_rate>30</update_rate>
<camera name="picam"><horizontal_fov>1.085595</horizontal_fov>
<image><width>640</width><height>480</height><format>R8G8B8</format></image>
<clip><near>.03</near><far>100</far></clip></camera>
<plugin name="camera_driver" filename="libgazebo_ros_camera.so"><ros/></plugin>
</sensor></link>
<joint name="fixed" type="fixed"><parent>base</parent><child>base_scan</child></joint>
<plugin name="drive" filename="libgazebo_ros_diff_drive.so"><max_wheel_torque>20</max_wheel_torque></plugin>
</model></sdf>"""


def canonical(element):
    return element.tag, sorted(element.attrib.items()), (element.text or "").strip(), [
        canonical(child) for child in element
    ]


def test_single_sensor_aligned_topics_and_unchanged_robot_physics():
    before = ET.fromstring(SOURCE)
    after = ET.fromstring(rgbd_robot.rgbd_sdf_text(SOURCE))
    old_sensor = before.find(".//sensor[@name='camera']")
    sensor = after.find(".//sensor[@name='camera']")
    assert sensor.get("type") == "depth"
    assert len(after.findall(".//sensor[@type='depth']")) == 1
    assert not after.findall(".//sensor[@type='camera']")
    for path in ("camera/image", "camera/horizontal_fov", "update_rate"):
        assert canonical(sensor.find(path)) == canonical(old_sensor.find(path))
    plugin = sensor.find("plugin")
    assert plugin.findtext("frame_name") == "camera_rgb_optical_frame"
    assert plugin.findtext("camera_name") == "camera"
    assert float(plugin.findtext("cx")) == 319.5
    assert float(plugin.findtext("cy")) == 239.5
    assert {remap.text for remap in plugin.findall("ros/remapping")} == {
        "camera/image_raw:=/camera/image_raw", "camera/camera_info:=/camera/camera_info",
        "camera/depth/image_raw:=/camera/depth/image_raw",
        "camera/depth/camera_info:=/camera/depth/camera_info",
    }
    assert float(sensor.findtext("camera/clip/near")) == float(plugin.findtext("min_depth"))
    assert float(sensor.findtext("camera/clip/far")) == float(plugin.findtext("max_depth"))
    before.find(".//link[@name='camera_rgb_optical_frame']").remove(old_sensor)
    after.find(".//link[@name='camera_rgb_optical_frame']").remove(sensor)
    assert canonical(after) == canonical(before)


def test_generated_files_are_unique_and_original_is_untouched(tmp_path):
    source = tmp_path / "upstream.sdf"
    source.write_text(SOURCE)
    first = Path(rgbd_robot.make_rgbd_sdf(source, tmp_path / "runtime"))
    second = Path(rgbd_robot.make_rgbd_sdf(source, tmp_path / "runtime"))
    assert first != second
    assert first.read_text() == second.read_text()
    assert source.read_text() == SOURCE


def test_generated_unicode_is_accepted_by_ros_spawn_entity_parser():
    from lxml import etree
    assert etree.fromstring(rgbd_robot.rgbd_sdf_text(SOURCE)).tag == "sdf"


@pytest.mark.parametrize("source", [
    SOURCE.replace("camera_rgb_optical_frame", "wrong_frame"),
    SOURCE.replace('type="camera"', 'type="multicamera"'),
    SOURCE.replace("libgazebo_ros_camera.so", "unsupported.so"),
    SOURCE.replace("<width>640</width>", "<width>0</width>"),
    SOURCE.replace("<height>480</height>", "<height>bad</height>"),
    SOURCE.replace('<link name="base">', '<link name="extra"><sensor name="extra" type="camera"/></link><link name="base">'),
])
def test_rejects_unverified_camera_layout(source):
    with pytest.raises(ValueError):
        rgbd_robot.rgbd_sdf_text(source)


def test_installed_upstream_robot_can_be_derived_without_physics_changes():
    path = Path("/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf")
    if not path.exists():
        pytest.skip("TurtleBot3 Gazebo is not installed on this machine")
    before = ET.fromstring(path.read_text())
    after = ET.fromstring(rgbd_robot.rgbd_sdf_text(path.read_text()))
    for tree in (before, after):
        link = tree.find(".//link[@name='camera_rgb_optical_frame']")
        link.remove(link.find("sensor"))
    assert canonical(after) == canonical(before)
