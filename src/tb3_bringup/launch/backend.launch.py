"""LocateAnything perception, visual memory, query, and navigation backend.

Starts alongside sim.launch.py and nav.launch.py. Exploration waits for
active Nav2, valid transforms, and a usable costmap. The optional startup
rotation is disabled by default; enable it with startup_scan:=true.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    pkg_fe = get_package_share_directory("tb3_frontier_exploration")
    fe_config = os.path.join(pkg_fe, "config", "params.yaml")

    map_topic = LaunchConfiguration("map_topic", default="/map")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    costmap_topic = LaunchConfiguration(
        "costmap_topic", default="/global_costmap/costmap")
    odom_topic = LaunchConfiguration("odom_topic", default="/odometry/filtered")

    return LaunchDescription([
        # This backend only makes sense next to the Gazebo sim, so sim
        # time defaults to true here (unlike the per-package launches).
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("mode", default_value="locateanything",
                              choices=["locateanything"]),
        DeclareLaunchArgument("worker_python", default_value=os.environ.get(
            "LOCATEANYTHING_PYTHON", "/home/lingshen/miniforge3/envs/locateanything3b/bin/python")),
        DeclareLaunchArgument("model_path", default_value=os.environ.get(
            "LOCATEANYTHING_MODEL", "/home/lingshen/research/locateanything_standalone/models/LocateAnything-3B")),
        DeclareLaunchArgument("eagle_path", default_value=os.environ.get(
            "LOCATEANYTHING_EAGLE", "/home/lingshen/research/locateanything_standalone/Eagle")),
        DeclareLaunchArgument("explore", default_value="true"),
        DeclareLaunchArgument("startup_scan", default_value="false",
                              description="Perform a stationary warmup scan before exploring"),
        DeclareLaunchArgument("auto_resume_exploration", default_value="true",
                              description="Resume exploration after a semantic command finishes"),
        DeclareLaunchArgument("warmup_scan_angle_deg", default_value="45.0"),
        DeclareLaunchArgument("warmup_full_rotation", default_value="false"),
        DeclareLaunchArgument("warmup_startup_delay_sec", default_value="3.0"),
        DeclareLaunchArgument("warmup_pause_sec", default_value="0.5"),
        DeclareLaunchArgument("evidence_dir", default_value=""),
        DeclareLaunchArgument("localization_mode", default_value="depth", choices=["laser", "depth"]),
        DeclareLaunchArgument("memory_classes", default_value="['chair', 'person', 'trash can']",
                              description="LocateAnything catalog categories as a YAML string list"),
        DeclareLaunchArgument("map_topic", default_value="/map",
                              description="OccupancyGrid topic for the map"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
        DeclareLaunchArgument("costmap_topic", default_value="/global_costmap/costmap",
                              description="Costmap topic for frontier cost sampling"),
        DeclareLaunchArgument("odom_topic", default_value="/odometry/filtered",
                              description="Odometry topic for goal assignment"),

        # ── LocateAnything and navigation ─────────────────────────────
        Node(
            package="tb3_locateanything",
            executable="locateanything_node",
            name="locateanything_node",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time,
                         "worker_python": LaunchConfiguration("worker_python"),
                         "model_path": LaunchConfiguration("model_path"),
                         "evidence_dir": LaunchConfiguration("evidence_dir"),
                         "localization_mode": LaunchConfiguration("localization_mode"),
                         "camera_info_topic": camera_info_topic,
                         "memory_classes": LaunchConfiguration("memory_classes"),
                         "eagle_path": LaunchConfiguration("eagle_path")}],
        ),
        Node(
            package="tb3_locateanything",
            executable="observation_node",
            name="observation_node",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time,
                         "initial_exploration_enabled": ParameterValue(
                             LaunchConfiguration("explore"), value_type=bool),
                         "map_topic": map_topic, "costmap_topic": costmap_topic,
                         "camera_info_topic": camera_info_topic}],
        ),
        Node(
            package="tb3_nav_adapter",
            executable="nav_goal_adapter_node",
            name="nav_goal_adapter_node",
            output="screen",
            parameters=[
                PathJoinSubstitution([FindPackageShare("tb3_nav_adapter"),
                                      "config", "nav_goal_adapter.yaml"]),
                {"use_sim_time": use_sim_time},
            ],
        ),
        Node(
            package="tb3_coordinator",
            executable="coordinator_node",
            name="coordinator_node",
            output="screen",
            parameters=[
                PathJoinSubstitution([FindPackageShare("tb3_coordinator"),
                                      "config", "coordinator.yaml"]),
                {"use_sim_time": use_sim_time,
                 "auto_resume_exploration": ParameterValue(
                     LaunchConfiguration("auto_resume_exploration"), value_type=bool)},
            ],
        ),

        # ── Frontier exploration ─────────────────────────────────────────
        # Optional one-shot cmd_vel rotation scan; signals
        # exploration_warmup_complete when done.
        Node(
            package="tb3_frontier_exploration",
            executable="startup_map_warmup_node.py",
            name="startup_map_warmup_node",
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("explore"), "'.lower() == 'true' and '",
                LaunchConfiguration("startup_scan"), "'.lower() == 'true'"])),
            parameters=[{"use_sim_time": use_sim_time,
                         "scan_angle_deg": ParameterValue(
                             LaunchConfiguration("warmup_scan_angle_deg"), value_type=float),
                         "full_rotation": ParameterValue(
                             LaunchConfiguration("warmup_full_rotation"), value_type=bool),
                         "startup_delay_sec": ParameterValue(
                             LaunchConfiguration("warmup_startup_delay_sec"), value_type=float),
                         "pause_between_sec": ParameterValue(
                             LaunchConfiguration("warmup_pause_sec"), value_type=float)}],
            output="screen",
        ),
        Node(
            package="tb3_frontier_exploration",
            executable="frontier_detection_node",
            name="frontier_detection_node",
            condition=IfCondition(LaunchConfiguration("explore")),
            parameters=[
                fe_config,
                {"use_sim_time": use_sim_time},
                # Flat {name: value}. A dict entry in `parameters` is a literal
                # override list, not a YAML document: launch_ros flattens any
                # nesting into dotted names, so the old
                # {"frontier_detection_node": {"ros__parameters": {...}}} form
                # produced undeclared parameters named
                # "frontier_detection_node.ros__parameters.map_topic" and the
                # two launch arguments silently did nothing. The node-name /
                # ros__parameters wrapper belongs only in the YAML file above.
                {"map_topic": map_topic,
                 "costmap_topic": costmap_topic},
            ],
            output="screen",
        ),
        Node(
            package="tb3_frontier_exploration",
            executable="goal_assignment_node",
            name="goal_assignment_node",
            condition=IfCondition(LaunchConfiguration("explore")),
            parameters=[
                fe_config,
                {"use_sim_time": use_sim_time},
                # Same flattening bug as above. Note the subscription this feeds
                # is an intentional no-op hook (goal_assignment_node.cpp
                # odomCallback); the argument is kept working so the topic name
                # stays overridable, not because pose comes from it.
                {"odom_topic": odom_topic,
                 "require_startup_warmup": ParameterValue(
                     LaunchConfiguration("startup_scan"), value_type=bool),
                 "require_nav2_active": True},
            ],
            output="screen",
        ),

    ])
