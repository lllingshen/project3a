"""Derive command routing for any upstream TurtleBot3 without changing physics."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


DRIVE_TOPIC = "/tb3_drive/cmd_vel"


def drive_sdf_text(source):
    root = ET.fromstring(source)
    models = root.findall("model")
    if len(models) != 1:
        raise ValueError("Drive source must contain one SDF model")
    plugins = models[0].findall("plugin[@filename='libgazebo_ros_diff_drive.so']")
    if len(plugins) != 1:
        raise ValueError("Drive routing requires exactly one Gazebo ROS diff-drive plugin")
    plugin = plugins[0]
    ros = plugin.find("ros")
    if ros is None:
        ros = ET.SubElement(plugin, "ros")
    for remap in list(ros.findall("remapping")):
        if (remap.text or "").split(":=", 1)[0].strip() in ("cmd_vel", "/cmd_vel"):
            ros.remove(remap)
    remap = ET.Element("remapping")
    remap.text = "cmd_vel:=" + DRIVE_TOPIC
    ros.insert(0, remap)
    qos = ros.find("qos")
    if qos is None:
        qos = ET.SubElement(ros, "qos")
    for topic in list(qos.findall("topic")):
        if topic.get("name") == DRIVE_TOPIC:
            qos.remove(topic)
    # gazebo_ros resolves QoS keys after ROS remapping, so use the final FQN.
    topic = ET.SubElement(qos, "topic", name=DRIVE_TOPIC)
    subscription = ET.SubElement(topic, "subscription")
    ET.SubElement(subscription, "history", depth="10").text = "keep_last"
    ET.SubElement(subscription, "reliability").text = "reliable"
    ET.SubElement(subscription, "durability").text = "volatile"
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def make_drive_sdf(source_sdf, output_dir=None):
    """Write a unique launch-lifetime SDF; leave the installed source untouched."""
    text = drive_sdf_text(Path(source_sdf).read_text(encoding="utf-8"))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="tb3_drive_",
                                     suffix=".sdf", dir=output_dir, delete=False) as generated:
        generated.write(text)
        return generated.name
