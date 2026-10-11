"""Run one independent LocateAnything demo with its world and catalog categories.

Usage:
    ros2 launch tb3_bringup locateanything_demo.launch.py scenario:=warehouse_objects

The profile sets only simulator setup and category names. Object positions and
Gazebo entity names are not passed to perception or memory. Exploration is on
by default; set explore:=false for a stationary observation demonstration.
SLAM and exploration start without scheduled delays or a mandatory rotation;
set startup_scan:=true to restore the optional initial scan.
After a command finishes, the robot waits for the next command by default;
set auto_resume_exploration:=true to continue exploring automatically.
Wait for Nav2 and LocateAnything memory readiness before sending /user_command.
"""
import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


SCENARIOS = {
    "chairs": {
        "world": "locateanything_chairs.world",
        "classes": ["chair", "person", "trash can"],
        "warmup_angle": "70.0", "warmup_delay": "12.0", "warmup_pause": "2.0",
    },
    "warehouse_objects": {
        "world": "locateanything_warehouse_objects.world",
        "classes": ["forklift", "fire extinguisher", "storage rack"],
        "warmup_angle": "70.0", "warmup_delay": "12.0", "warmup_pause": "2.0",
    },
    "warehouse_colors": {
        "world": "locateanything_warehouse_colors.world",
        "classes": ["person", "table", "chair"],
        "warmup_angle": "70.0", "warmup_delay": "12.0", "warmup_pause": "2.0",
    },
}


def generate_launch_description():
    bringup = get_package_share_directory("tb3_bringup")
    actions = [
        DeclareLaunchArgument("scenario", default_value="chairs", choices=list(SCENARIOS)),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_gazebo_gui", default_value="true"),
        DeclareLaunchArgument("slam_profile", default_value="simulation_world_odom",
                              choices=["simulation_world_odom", "default"],
                              description="SLAM odometry profile forwarded to nav.launch.py"),
        DeclareLaunchArgument("explore", default_value="true"),
        DeclareLaunchArgument("startup_scan", default_value="false",
                              description="Perform the optional initial rotation before exploring"),
        DeclareLaunchArgument("auto_resume_exploration", default_value="false",
                              description="Resume exploration after a semantic command finishes"),
        DeclareLaunchArgument("evidence_dir", default_value=""),
        DeclareLaunchArgument("x_pose", default_value="AUTO"),
        DeclareLaunchArgument("y_pose", default_value="AUTO"),
        DeclareLaunchArgument("yaw_pose", default_value="AUTO"),
        DeclareLaunchArgument("spawn_seed", default_value=""),
        DeclareLaunchArgument("spawn_margin", default_value="0.15"),
        DeclareLaunchArgument("spawn_bounds", default_value=""),
        DeclareLaunchArgument("rgbd_camera", default_value="true"),
        DeclareLaunchArgument("localization_mode", default_value="depth", choices=["laser", "depth"]),
        DeclareLaunchArgument("warmup_full_rotation", default_value="true"),
    ]
    for name, profile in SCENARIOS.items():
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "full_stack.launch.py")),
            condition=LaunchConfigurationEquals("scenario", name),
            launch_arguments={
                "mode": "locateanything", "explore": LaunchConfiguration("explore"),
                "startup_scan": LaunchConfiguration("startup_scan"),
                "auto_resume_exploration": LaunchConfiguration("auto_resume_exploration"),
                "use_sim_time": "true",
                "world": os.path.join(bringup, "worlds", profile["world"]),
                "x_pose": LaunchConfiguration("x_pose"), "y_pose": LaunchConfiguration("y_pose"),
                "yaw_pose": LaunchConfiguration("yaw_pose"),
                "spawn_seed": LaunchConfiguration("spawn_seed"),
                "spawn_margin": LaunchConfiguration("spawn_margin"),
                "spawn_bounds": LaunchConfiguration("spawn_bounds"),
                "rgbd_camera": LaunchConfiguration("rgbd_camera"),
                "localization_mode": LaunchConfiguration("localization_mode"),
                "memory_classes": json.dumps(profile["classes"]),
                "warmup_scan_angle_deg": profile["warmup_angle"],
                "warmup_full_rotation": LaunchConfiguration("warmup_full_rotation"),
                "warmup_startup_delay_sec": profile["warmup_delay"],
                "warmup_pause_sec": profile["warmup_pause"],
                "use_rviz": LaunchConfiguration("use_rviz"),
                "slam_profile": LaunchConfiguration("slam_profile"),
                "use_gazebo_gui": LaunchConfiguration("use_gazebo_gui"),
                "evidence_dir": LaunchConfiguration("evidence_dir"),
            }.items(),
        ))
    return LaunchDescription(actions)
