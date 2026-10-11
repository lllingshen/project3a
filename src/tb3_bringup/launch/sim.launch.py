"""
sim.launch.py — Terminal 1: Gazebo + TurtleBot3 only.

Brings up gzserver + gzclient with a course world, robot_state_publisher,
and the TurtleBot3 spawn. Worlds and the vendored Gazebo models they
reference both ship in this package (tb3_bringup/worlds, tb3_bringup/models);
GAZEBO_MODEL_PATH is pointed at the installed models so they resolve offline.

Usage:
    export TURTLEBOT3_MODEL=waffle_pi
    ros2 launch tb3_bringup sim.launch.py
    ros2 launch tb3_bringup sim.launch.py world:=warehouse_models

Runs standalone — no SLAM/Nav2/course nodes are started here.
"""

import importlib.util
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


# Presets resolve world files only; safe poses are derived from collision geometry.
WORLD_PRESETS = {
    "warehouse_models_person": {"file": "warehouse_models_person.world"},
    "warehouse_models": {"file": "warehouse_semantic_models.world"},
}


def _helper(name):
    spec = importlib.util.spec_from_file_location(
        "tb3_bringup_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    pkg_bringup = get_package_share_directory("tb3_bringup")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    launch_tb3 = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "launch"
    )

    def make_gazebo_actions(context, *_args, **_kwargs):
        """Verify a random/fixed collision-free pose before starting Gazebo."""
        raw = LaunchConfiguration("world").perform(context).strip()
        if not raw:
            raw = "warehouse_models_person"

        if raw in WORLD_PRESETS:
            preset = WORLD_PRESETS[raw]
            world_file = os.path.join(pkg_bringup, "worlds", preset["file"])
            source = "alias"
        else:
            # Treat as a direct path. We deliberately do not silently
            # fall back to the default if the file is missing — fail
            # loudly so users notice typos.
            world_file = os.path.abspath(os.path.expanduser(raw))
            source = "path"

        if not os.path.isfile(world_file):
            valid_aliases = ", ".join(sorted(WORLD_PRESETS.keys()))
            raise FileNotFoundError(
                f"[sim] world={raw!r} could not be resolved. "
                f"Tried as {source}: {world_file}. "
                f"Pass one of the built-in aliases ({valid_aliases}) or an "
                f"absolute path to a .world file."
            )

        def argument(name):
            return LaunchConfiguration(name).perform(context).strip()

        def coordinate(name):
            value = argument(name)
            return None if value.upper() == "AUTO" else float(value)

        model = os.environ.get("TURTLEBOT3_MODEL", "waffle_pi")
        robot_sdf = os.path.join(get_package_share_directory("turtlebot3_gazebo"),
                                 "models", "turtlebot3_" + model, "model.sdf")
        rgbd = argument("rgbd_camera").lower()
        if rgbd not in ("true", "false"):
            raise ValueError("rgbd_camera must be true or false")
        if rgbd == "true":
            robot_sdf = _helper("rgbd_robot").make_rgbd_sdf(robot_sdf)
        robot_sdf = _helper("drive_robot").make_drive_sdf(robot_sdf)
        geometry = _helper("safe_spawn")
        model_paths = [os.path.join(pkg_bringup, "models")]
        model_paths += [p for p in os.environ.get("GAZEBO_MODEL_PATH", "").split(":") if p]
        model_paths += [os.path.join(get_package_share_directory("turtlebot3_gazebo"), "models"),
                        "/usr/share/gazebo-11/models", os.path.expanduser("~/.gazebo/models")]
        bounds = argument("spawn_bounds")
        bounds = tuple(float(v) for v in bounds.split(",")) if bounds else None
        scene = geometry.load_scene(world_file, model_paths, bounds)
        selected = geometry.choose_spawn(
            scene, geometry.robot_radius(robot_sdf, model_paths),
            seed=int(argument("spawn_seed")) if argument("spawn_seed") else None,
            x=coordinate("x_pose"), y=coordinate("y_pose"), yaw=coordinate("yaw_pose"),
            margin=float(argument("spawn_margin")))
        x_final, y_final, yaw_final = (str(selected[k]) for k in ("x", "y", "yaw"))

        gzserver = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
            ),
            launch_arguments={"world": world_file}.items(),
        )
        gzclient = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
            ),
            condition=IfCondition(LaunchConfiguration("use_gazebo_gui")),
        )
        rsp = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_tb3, "robot_state_publisher.launch.py")
            ),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        )
        spawn = Node(
            package="gazebo_ros", executable="spawn_entity.py", output="screen",
            arguments=["-entity", model, "-file", robot_sdf, "-x", x_final,
                       "-y", y_final, "-z", "0.01", "-Y", yaw_final],
        )
        velocity_relay = Node(package="tb3_bringup", executable="velocity_relay.py",
                              output="screen")

        return [
            LogInfo(msg=(
                f"[sim] world={raw!r} → {world_file} ;"
                f" spawn=({x_final}, {y_final}, yaw={yaw_final}) ;"
                f" mode={selected['mode']} seed={selected['seed']} ;"
                f" collision_radius={selected['robot_radius']:.4f}m"
                f" margin={selected['margin']:.3f}m clearance={selected['clearance']:.4f}m ;"
                f" reproduce: x_pose:={x_final} y_pose:={y_final} yaw_pose:={yaw_final}"
            )),
            gzserver, gzclient, rsp, spawn, velocity_relay,
        ]

    # The .world files reference vendored Gazebo models via model:// URIs.
    # Prepend the package-shipped models directory so a fresh machine never
    # has to hit the (deprecated, slow) online Gazebo model database.
    set_gazebo_model_path = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=[
            os.path.join(pkg_bringup, "models"),
            ":",
            EnvironmentVariable("GAZEBO_MODEL_PATH", default_value=""),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_gazebo_gui", default_value="true"),

        DeclareLaunchArgument(
            "x_pose", default_value="AUTO",
            description=(
                "Robot Gazebo-world x, or AUTO for a verified random safe pose."
            ),
        ),
        DeclareLaunchArgument(
            "y_pose", default_value="AUTO",
            description=(
                "Robot Gazebo-world y; fixed coordinates must supply both x and y."
            ),
        ),
        DeclareLaunchArgument("yaw_pose", default_value="AUTO",
                              description="Spawn yaw radians, or AUTO for seeded random heading"),
        DeclareLaunchArgument("spawn_seed", default_value="",
                              description="Random seed; empty generates and logs a fresh seed"),
        DeclareLaunchArgument("spawn_margin", default_value="0.15",
                              description="Extra metres around the robot collision envelope"),
        DeclareLaunchArgument("spawn_bounds", default_value="",
                              description="Optional xmin,xmax,ymin,ymax; otherwise infer enclosing walls"),
        DeclareLaunchArgument("rgbd_camera", default_value="false",
                              description="Generate aligned RGB-D camera from the upstream robot SDF"),
        DeclareLaunchArgument(
            "world",
            default_value="warehouse_models_person",
            description=(
                "Gazebo world: alias (warehouse_models_person | "
                "warehouse_models) or absolute path to a .world file."
            ),
        ),

        # Must come before the Gazebo actions below.
        set_gazebo_model_path,

        OpaqueFunction(function=make_gazebo_actions),
    ])
