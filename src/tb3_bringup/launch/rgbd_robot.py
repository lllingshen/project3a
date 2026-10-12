"""Derive the LocateAnything RGB-D robot from the installed TurtleBot3 SDF.

Only the existing camera sensor changes. One Gazebo depth camera produces both
RGB and depth, so they share a renderer, viewpoint, calibration and optical
frame. Meshes, collisions, drive parameters and the laser remain upstream data.
The optional laser-localization configuration can use the upstream RGB camera.
A separate drive routing helper derives the command topic for either camera mode.

The ROS 2 gazebo_ros_camera plugin supports both camera and depth sensors:
https://github.com/ros-simulation/gazebo_ros_pkgs/blob/ros2/gazebo_plugins/src/gazebo_ros_camera.cpp
Depth images use 32FC1, in metres along the optical Z axis (not radial range).
"""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


OPTICAL_FRAME = "camera_rgb_optical_frame"
CAMERA_NAME = "camera"
MIN_DEPTH = 0.05
MAX_DEPTH = 8.0


def _set_text(parent: ET.Element, name: str, value: str) -> None:
    child = parent.find(name)
    if child is None:
        child = ET.SubElement(parent, name)
    child.text = value


def rgbd_sdf_text(source: str) -> str:
    """Change the known waffle_pi camera to a single aligned RGB-D sensor.

    Fail on a different camera layout instead of silently assigning an optical
    frame that disagrees with the upstream robot_state_publisher URDF.
    """
    root = ET.fromstring(source)
    model = root.find("model")
    if model is None:
        raise ValueError("RGB-D robot source must contain one SDF model")
    cameras = [
        sensor
        for sensor in model.findall("link/sensor")
        if sensor.get("type") in {"camera", "depth", "multicamera"}
    ]
    sensor = model.find(f"link[@name='{OPTICAL_FRAME}']/sensor[@name='camera']")
    if len(cameras) != 1 or sensor is None or cameras[0] is not sensor:
        raise ValueError("RGB-D mode requires the upstream waffle_pi camera layout")
    if sensor.get("type") not in {"camera", "depth"}:
        raise ValueError("RGB-D mode does not support a multicamera source")
    camera = sensor.find("camera")
    if camera is None or camera.find("image") is None:
        raise ValueError("The upstream camera has no image calibration")
    try:
        width = int(camera.findtext("image/width", "0"))
        height = int(camera.findtext("image/height", "0"))
    except ValueError as exc:
        raise ValueError("The upstream camera has invalid image dimensions") from exc
    if width <= 0 or height <= 0:
        raise ValueError("The upstream camera has invalid image dimensions")
    plugins = sensor.findall("plugin")
    if len(plugins) != 1 or plugins[0].get("filename") != "libgazebo_ros_camera.so":
        raise ValueError("RGB-D mode requires libgazebo_ros_camera.so")

    # Gazebo's depth sensor renders RGB and per-pixel depth together. Adding a
    # second camera would create a separate image stream with potential skew.
    sensor.set("type", "depth")
    _set_text(sensor, "always_on", "true")
    clip = camera.find("clip")
    if clip is None:
        clip = ET.SubElement(camera, "clip")
    _set_text(clip, "near", str(MIN_DEPTH))
    _set_text(clip, "far", str(MAX_DEPTH))
    plugin = plugins[0]
    plugin.set("name", "camera_driver")
    _set_text(plugin, "camera_name", CAMERA_NAME)
    _set_text(plugin, "frame_name", OPTICAL_FRAME)
    # Match the pixel centres used by this plugin's depth-to-point-cloud
    # projection; its historical CameraInfo defaults are offset by one pixel.
    _set_text(plugin, "cx", str((width - 1) / 2.0))
    _set_text(plugin, "cy", str((height - 1) / 2.0))
    _set_text(plugin, "min_depth", str(MIN_DEPTH))
    _set_text(plugin, "max_depth", str(MAX_DEPTH))
    ros = plugin.find("ros")
    if ros is None:
        ros = ET.SubElement(plugin, "ros")
    # Absolute destinations make this contract independent of a parent launch
    # namespace. The original sensor is replaced, so there is one RGB publisher.
    _set_text(ros, "namespace", "/")
    for child in list(ros):
        if child.tag == "remapping":
            ros.remove(child)
    for topic in ("image_raw", "camera_info", "depth/image_raw", "depth/camera_info"):
        ET.SubElement(ros, "remapping").text = f"{CAMERA_NAME}/{topic}:=/{CAMERA_NAME}/{topic}"

    ET.indent(root, space="  ")
    # ROS Humble spawn_entity passes decoded text to lxml.fromstring, which
    # rejects a Unicode string containing an XML encoding declaration.
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def make_rgbd_sdf(source_sdf, output_dir=None) -> str:
    """Write a unique derived model and return its path for spawn_entity.py.

    The source in /opt remains untouched. Keep the generated file for the
    lifetime of the launch; Gazebo's spawn process opens it asynchronously.
    """
    source_path = Path(source_sdf)
    text = rgbd_sdf_text(source_path.read_text(encoding="utf-8"))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="tb3_rgbd_", suffix=".sdf",
        dir=output_dir, delete=False,
    ) as generated:
        generated.write(text)
        return generated.name
