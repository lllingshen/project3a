"""
nav.launch.py — Terminal 2: SLAM Toolbox + Nav2 + RViz.

Starts the Nav2 bringup with slam:=True (which launches slam_toolbox
internally, so no pre-built map file is needed) plus RViz with the course
config. Start this AFTER the simulation (sim.launch.py) is up: SLAM needs
/scan and /clock from Gazebo before it can publish /map, and Nav2's
lifecycle manager waits on /map to finish configuring.

Usage:
    ros2 launch tb3_bringup nav.launch.py
    ros2 launch tb3_bringup nav.launch.py use_rviz:=false
    ros2 launch tb3_bringup nav.launch.py slam_profile:=default

Runs standalone — without the simulation it simply waits (lifecycle
manager logs "Waiting for service ..."), it does not crash.
"""
import os
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_navigation(context):
    profile = LaunchConfiguration("slam_profile").perform(context)
    if profile not in ("simulation_world_odom", "default"):
        raise ValueError("Unknown slam_profile: " + profile)
    mode = LaunchConfiguration("mode").perform(context)
    if mode not in ("baseline", "locateanything"):
        raise ValueError("Unknown navigation mode: " + mode)
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    pkg_nav2 = get_package_share_directory("nav2_bringup")

    # Nav2 bringup: slam=True makes it launch slam_toolbox internally.
    # map arg is required even with slam=True; provide a dummy path that
    # won't be loaded. params_file must be explicit to avoid empty-path
    # errors from ParameterFile.
    dummy_map = os.path.join(pkg_nav2, "maps", "turtlebot3_world.yaml")
    nav2_params = os.path.join(pkg_nav2, "params", "nav2_params.yaml")
    # Use one controller for initial rotation and curved-path tracking. The
    # shim/DWB combination repeatedly chose opposite turn directions on a
    # recorded curved path, producing rotation without forward progress.
    with open(nav2_params) as source:
        nav2_config = yaml.safe_load(source)
    controller = nav2_config["controller_server"]["ros__parameters"]
    if mode == "locateanything":
        # Camera framing uses the actual goal-checker tolerance. Keep the
        # assignment baseline and XY tolerance unchanged.
        if controller["goal_checker_plugins"] != ["general_goal_checker"]:
            raise ValueError("LocateAnything requires one general_goal_checker")
        controller["general_goal_checker"]["yaw_goal_tolerance"] = 0.10
    local_costmap = nav2_config["local_costmap"]["local_costmap"]["ros__parameters"]
    controller["FollowPath"] = {
        "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
        "desired_linear_vel": 0.22,
        "lookahead_dist": 0.45,
        "use_velocity_scaled_lookahead_dist": False,
        "use_rotate_to_heading": True,
        "rotate_to_heading_angular_vel": 0.6,
        "rotate_to_heading_min_angle": 0.785,
        "max_angular_accel": 1.5,
        "transform_tolerance": controller["FollowPath"]["transform_tolerance"],
        "min_approach_linear_velocity": 0.05,
        "approach_velocity_scaling_dist": 0.6,
        "use_regulated_linear_velocity_scaling": True,
        "regulated_linear_scaling_min_radius": 0.6,
        "regulated_linear_scaling_min_speed": 0.08,
        "use_collision_detection": True,
        "max_allowed_time_to_collision_up_to_carrot": 1.0,
        "use_cost_regulated_linear_velocity_scaling": True,
        "cost_scaling_dist": 0.3,
        "cost_scaling_gain": 1.0,
        "inflation_cost_scaling_factor": local_costmap["inflation_layer"]["cost_scaling_factor"],
        "use_interpolation": True,
        "allow_reversing": False,
    }
    # Frontier goals lie in known free space; do not shortcut through unknown
    # global-costmap cells on the way to them.
    nav2_config["planner_server"]["ros__parameters"]["GridBased"]["allow_unknown"] = False
    # The adapter explicitly chooses a costmap-checked parking point. Do not
    # silently replace it with a distant endpoint under NavFn's default .5 m.
    nav2_config["planner_server"]["ros__parameters"]["GridBased"]["tolerance"] = 0.05
    # Keep the complete installed SLAM configuration. Its default 0.5 m scan
    # acceptance distance can exceed a short frontier move's actual travel
    # before Nav2's goal tolerance is reached, leaving /map unchanged.
    slam_params = os.path.join(
        get_package_share_directory("slam_toolbox"),
        "config", "mapper_params_online_sync.yaml",
    )
    with open(slam_params) as source:
        slam_config = yaml.safe_load(source)
    slam_config["slam_toolbox"]["ros__parameters"].update({
        "minimum_travel_distance": 0.10,
        "minimum_travel_heading": 0.10,
        "map_update_interval": 1.0,
    })
    if profile == "simulation_world_odom":
        # The course Gazebo diff-drive plugin already publishes world odometry.
        # Keep local laser matching, but do not add broad loop-closure corrections
        # to this low-drift input. This is not a real encoder-odometry profile.
        slam_config["slam_toolbox"]["ros__parameters"].update({
            "do_loop_closing": False,
            "use_scan_matching": True,
        })
    nav2_config["slam_toolbox"] = slam_config["slam_toolbox"]
    # Humble's RewrittenYaml only replaces existing keys. Generate a local
    # merged file so the controller and SLAM parameters are present at startup.
    with tempfile.NamedTemporaryFile(mode="w", prefix="tb3_nav_", suffix=".yaml", delete=False) as merged:
        yaml.safe_dump(nav2_config, merged)
        nav2_params = merged.name
    # Mute the very noisy `worldToMap failed: mx,my: ...` ERROR that
    # planner_server prints whenever the costmap inflation samples one cell
    # past the static-map boundary. It is benign in Nav2 Humble (planning
    # still succeeds and the goal is reached), but it floods the terminal
    # at planning rate.
    #
    # nav2_bringup forwards this single string verbatim as
    # `--ros-args --log-level <value>` to every Nav2 node. ROS 2's
    # `--log-level` supports a per-logger form `<logger>:=<level>`; when
    # passed a logger name that does not exist on the receiving node the
    # rcl logging machinery silently ignores it and the process default
    # stays at INFO. Therefore `planner_server:=fatal`:
    #   • on planner_server  → matches its node logger → silenced.
    #   • on every other Nav2 node (bt_navigator, controller_server, …)
    #     → no such logger → INFO defaults preserved, goal lifecycle and
    #     recoveries still print.
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam": "True",
            "map": dummy_map,
            "params_file": nav2_params,
            "autostart": "True",
            "use_composition": "False",
            "log_level": "planner_server:=fatal",
        }.items(),
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare("tb3_bringup"), "rviz", "semantic_nav.rviz"
    ])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        output="screen",
    )

    return [
        LogInfo(msg=["SLAM profile: ", profile, "; params_file: ", nav2_params]),
        nav2,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("mode", default_value="baseline",
                              choices=["baseline", "locateanything"],
                              description="LocateAnything camera-facing precision or baseline goal tolerance"),
        DeclareLaunchArgument("slam_profile", default_value="simulation_world_odom",
                              choices=["simulation_world_odom", "default"],
                              description="Simulation world odometry or upstream loop-closure settings"),
        DeclareLaunchArgument("use_rviz", default_value="true",
                              description="Launch RViz with semantic nav config"),
        OpaqueFunction(function=_launch_navigation),
    ])
