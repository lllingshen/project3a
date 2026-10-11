"""
full_stack.launch.py — Terminals 1-5 in one process (convenience shell).

The RECOMMENDED way to run the course is the six-terminal flow in the
README Quick start (sim / nav / backend / localizer / detector as
separate `ros2 launch` commands, so each subsystem has its own logs and
can be restarted alone). This file exists for the cases where one command
is preferable (demos, quick smoke tests, the grader's batch run).

It is a thin shell: it includes the standalone sub-launches for the selected mode.
Baseline retains the course's fixed launch delays. LocateAnything starts
simulation, SLAM/navigation, and its backend together; exploration waits for
actual navigation, transform, and map readiness inside the nodes.

Usage:
    export TURTLEBOT3_MODEL=waffle_pi
    ros2 launch tb3_bringup full_stack.launch.py
    ros2 launch tb3_bringup full_stack.launch.py mode:=locateanything

    # Then in another terminal:
    ros2 topic pub --once /user_command std_msgs/String "data: 'go to person 0'"

All arguments are forwarded to the relevant sub-launch: `world`,
`x_pose`, `y_pose` (sim), `use_rviz`, `slam_profile` (nav), `mode`, `explore`,
`auto_resume_exploration`, `evidence_dir`, `memory_classes`, `use_runtime_debug`
(backend), `use_sim_time` (all).
LocateAnything mode omits the YOLO detector and the baseline localizer.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import LaunchConfigurationEquals
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
                "'", LaunchConfiguration("rgbd_camera"), "'.lower() == 'true' or ('",
                LaunchConfiguration("mode"), "' == 'locateanything' and '",
                LaunchConfiguration("localization_mode"), "' == 'depth')"]),
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

    localizer = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch, "localizer.launch.py")),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=LaunchConfigurationEquals("mode", "baseline"),
    )

    detector = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_launch, "detector.launch.py")),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=LaunchConfigurationEquals("mode", "baseline"),
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
            "use_runtime_debug": LaunchConfiguration("use_runtime_debug"),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_gazebo_gui", default_value="true"),
        DeclareLaunchArgument("slam_profile", default_value="simulation_world_odom",
                              choices=["simulation_world_odom", "default"],
                              description="SLAM odometry profile forwarded to nav.launch.py"),
        DeclareLaunchArgument("mode", default_value="baseline",
                              choices=["baseline", "locateanything"]),
        DeclareLaunchArgument("explore", default_value="true"),
        DeclareLaunchArgument("startup_scan", default_value=PythonExpression([
            "'true' if '", LaunchConfiguration("mode"), "' == 'baseline' else 'false'"]),
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
        DeclareLaunchArgument("use_runtime_debug", default_value="false",
                              description="Launch semantic runtime debug node"),
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
        DeclareLaunchArgument("rgbd_camera", default_value="false"),
        DeclareLaunchArgument("localization_mode", default_value="laser", choices=["laser", "depth"]),

        # Baseline fixed delays replace the human "wait until ready" of the
        # six-terminal flow. LocateAnything starts all three subsystems now;
        # the exploration nodes retain their readiness checks before moving.
        #   t=0   sim      — Gazebo needs no one.
        #   t=10  nav      — Nav2 (autostart) needs Gazebo up so SLAM can
        #                    publish /map before planner_server configures;
        #                    10 s absorbs CPU jitter (5 s made
        #                    lifecycle_manager hang on "Waiting for service
        #                    planner_server/get_state...").
        #   t=15  baseline detector + localizer — camera/scan topics exist.
        #   t=20  backend  — coordinator + exploration want Nav2's
        #                    navigate_to_pose action to be answering.
        sim,
        GroupAction(actions=[nav, backend],
                    condition=LaunchConfigurationEquals("mode", "locateanything")),
        TimerAction(period=10.0, actions=[nav],
                    condition=LaunchConfigurationEquals("mode", "baseline")),
        TimerAction(period=15.0, actions=[detector],
                    condition=LaunchConfigurationEquals("mode", "baseline")),
        TimerAction(period=15.0, actions=[localizer],
                    condition=LaunchConfigurationEquals("mode", "baseline")),
        TimerAction(period=20.0, actions=[backend],
                    condition=LaunchConfigurationEquals("mode", "baseline")),
    ])
