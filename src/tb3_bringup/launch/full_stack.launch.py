"""Start simulation, SLAM/Nav2, and LocateAnything together.

Use locateanything_demo.launch.py to choose one of the three demo scenes.
Exploration waits for actual map, transform, and Nav2 readiness inside the nodes.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    bringup_launch = os.path.join(
        get_package_share_directory("tb3_bringup"), "launch")

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch, "sim.launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "world": LaunchConfiguration("world"),
            "use_gazebo_gui": LaunchConfiguration("use_gazebo_gui"),
            "x_pose": LaunchConfiguration("x_pose"),
            "y_pose": LaunchConfiguration("y_pose"),
            "yaw_pose": LaunchConfiguration("yaw_pose"),
            "spawn_seed": LaunchConfiguration("spawn_seed"),
            "spawn_margin": LaunchConfiguration("spawn_margin"),
            "spawn_bounds": LaunchConfiguration("spawn_bounds"),
            "rgbd_camera": PythonExpression([
                "'", LaunchConfiguration("rgbd_camera"), "'.lower() == 'true' or '",
                LaunchConfiguration("localization_mode"), "' == 'depth'"]),
        }.items(),
    )

    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch, "nav.launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "use_rviz": LaunchConfiguration("use_rviz"),
            "slam_profile": LaunchConfiguration("slam_profile"),
            "mode": LaunchConfiguration("mode"),
        }.items(),
    )

    backend = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch, "backend.launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "mode": LaunchConfiguration("mode"),
            "localization_mode": LaunchConfiguration("localization_mode"),
            "explore": LaunchConfiguration("explore"),
            "startup_scan": LaunchConfiguration("startup_scan"),
            "auto_resume_exploration": LaunchConfiguration("auto_resume_exploration"),
            "warmup_scan_angle_deg": LaunchConfiguration("warmup_scan_angle_deg"),
            "warmup_full_rotation": LaunchConfiguration("warmup_full_rotation"),
            "warmup_startup_delay_sec": LaunchConfiguration("warmup_startup_delay_sec"),
            "warmup_pause_sec": LaunchConfiguration("warmup_pause_sec"),
            "evidence_dir": LaunchConfiguration("evidence_dir"),
            "memory_classes": LaunchConfiguration("memory_classes"),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_gazebo_gui", default_value="true"),
        DeclareLaunchArgument("slam_profile", default_value="simulation_world_odom",
                              choices=["simulation_world_odom", "default"],
                              description="SLAM odometry profile forwarded to nav.launch.py"),
        DeclareLaunchArgument("mode", default_value="locateanything",
                              choices=["locateanything"]),
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
        DeclareLaunchArgument("memory_classes", default_value="['chair', 'person', 'trash can']",
                              description="LocateAnything catalog categories as a YAML string list"),
        DeclareLaunchArgument("use_rviz", default_value="true",
                              description="Launch RViz with semantic nav config"),
        # Forwarded to sim.launch.py, which resolves aliases and the
        # collision-verified AUTO random spawn (see safe_spawn.py).
        DeclareLaunchArgument("world", default_value="warehouse_models_person",
                              description="Gazebo world: alias or absolute path"),
        DeclareLaunchArgument("x_pose", default_value="AUTO"),
        DeclareLaunchArgument("y_pose", default_value="AUTO"),
        DeclareLaunchArgument("yaw_pose", default_value="AUTO"),
        DeclareLaunchArgument("spawn_seed", default_value=""),
        DeclareLaunchArgument("spawn_margin", default_value="0.15"),
        DeclareLaunchArgument("spawn_bounds", default_value=""),
        DeclareLaunchArgument("rgbd_camera", default_value="true"),
        DeclareLaunchArgument("localization_mode", default_value="depth", choices=["laser", "depth"]),

        sim,
        nav,
        backend,
    ])
