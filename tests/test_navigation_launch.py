"""Exercise the actual generated Nav2 YAML, including new Humble parameters."""
import importlib.util
from pathlib import Path

import pytest
import yaml


def _launch_module(filename="nav.launch.py"):
    pytest.importorskip("launch")
    path = Path(__file__).parents[1] / "src/tb3_bringup/launch" / filename
    spec = importlib.util.spec_from_file_location("tested_" + filename.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context_and_description(filename="nav.launch.py", profile=None, mode=None):
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    context = LaunchContext()
    if profile is not None:
        context.launch_configurations["slam_profile"] = profile
    if mode is not None:
        context.launch_configurations["mode"] = mode
    description = _launch_module(filename).generate_launch_description()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    return context, description


def _generated_config(profile=None, mode=None):
    from launch.actions import OpaqueFunction
    from nav2_common.launch import HasNodeParams, RewrittenYaml

    context, description = _context_and_description(profile=profile, mode=mode)
    opaque = next(a for a in description.entities if isinstance(a, OpaqueFunction))
    actions = opaque.execute(context)  # Build the real include; never start nodes.
    include = next(a for a in actions if a.__class__.__name__ == "IncludeLaunchDescription")
    generated = Path(dict(include.launch_arguments)["params_file"])
    rewritten = None
    try:
        config = yaml.safe_load(generated.read_text())
        # Exercise Nav2's actual switch between its default and supplied SLAM
        # files: a partial or incorrectly nested node must not be ignored.
        assert HasNodeParams(source_file=str(generated), node_name="slam_toolbox").perform(context) == "True"
        rewritten = Path(RewrittenYaml(
            source_file=str(generated), root_key="",
            param_rewrites={"use_sim_time": "true"}, convert_types=True,
        ).perform(context))
        applied = yaml.safe_load(rewritten.read_text())
        assert applied["slam_toolbox"]["ros__parameters"] == config["slam_toolbox"]["ros__parameters"]
        assert isinstance(applied["slam_toolbox"]["ros__parameters"]["do_loop_closing"], bool)
        assert isinstance(applied["slam_toolbox"]["ros__parameters"]["use_scan_matching"], bool)
        assert applied["controller_server"]["ros__parameters"]["general_goal_checker"] == config["controller_server"]["ros__parameters"]["general_goal_checker"]
        return config
    finally:
        generated.unlink()
        if rewritten is not None:
            rewritten.unlink()


@pytest.mark.parametrize("profile", ["simulation_world_odom", "default"])
@pytest.mark.parametrize("mode", ["locateanything"])
def test_generated_navigation_config_uses_one_collision_checked_path_controller(profile, mode):
    pytest.importorskip("launch")
    from ament_index_python.packages import get_package_share_directory

    config = _generated_config(profile, mode)
    upstream_path = Path(get_package_share_directory("nav2_bringup")) / "params/nav2_params.yaml"
    upstream = yaml.safe_load(upstream_path.read_text())
    follow = config["controller_server"]["ros__parameters"]["FollowPath"]
    original = upstream["controller_server"]["ros__parameters"]["FollowPath"]
    assert follow["plugin"] == "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
    assert not ({"primary_controller", "critics", "max_vel_x", "angular_dist_threshold"} & follow.keys())
    assert follow["use_rotate_to_heading"] is True  # Includes final goal orientation.
    assert follow["allow_reversing"] is False
    assert 0. < follow["desired_linear_vel"] <= original["max_vel_x"]
    assert 0. < follow["rotate_to_heading_angular_vel"] <= original["max_vel_theta"]
    assert 0. < follow["max_angular_accel"] <= original["acc_lim_theta"]
    assert follow["use_collision_detection"] is True
    assert follow["max_allowed_time_to_collision_up_to_carrot"] >= 1.
    assert follow["lookahead_dist"] >= follow["desired_linear_vel"]
    assert follow["use_velocity_scaled_lookahead_dist"] is False
    assert follow["use_regulated_linear_velocity_scaling"] is True
    assert follow["use_cost_regulated_linear_velocity_scaling"] is True
    assert 0. < follow["min_approach_linear_velocity"] <= follow["regulated_linear_scaling_min_speed"] < follow["desired_linear_vel"]
    costmap = config["local_costmap"]["local_costmap"]["ros__parameters"]
    inflation = costmap["inflation_layer"]
    assert follow["inflation_cost_scaling_factor"] == inflation["cost_scaling_factor"]
    assert 0. < follow["cost_scaling_dist"] <= inflation["inflation_radius"]
    assert 0. < follow["cost_scaling_gain"] <= 1.
    assert 0. < follow["approach_velocity_scaling_dist"] < costmap["width"] / 2
    assert follow["transform_tolerance"] == original["transform_tolerance"]
    slam_path = Path(get_package_share_directory("slam_toolbox")) / "config/mapper_params_online_sync.yaml"
    original_slam = yaml.safe_load(slam_path.read_text())
    slam = config.pop("slam_toolbox")
    expected_overrides = {
        "minimum_travel_distance": 0.10,
        "minimum_travel_heading": 0.10,
        "map_update_interval": 1.0,
    }
    if profile == "simulation_world_odom":
        expected_overrides.update({"do_loop_closing": False, "use_scan_matching": True})
    else:
        # Upstream loop and local matching policies must survive unchanged.
        for key in ("do_loop_closing", "use_scan_matching"):
            assert slam["ros__parameters"][key] == original_slam["slam_toolbox"]["ros__parameters"][key]
    # A shortest accepted exploration goal must require translation beyond
    # SLAM's outer XY gate even with an arbitrary last-accepted scan anchor.
    # slam_toolbox 2.6.10 compares squared distance to 0.8 * travel**2.
    import math
    frontier_path = Path(__file__).parents[1] / "src/tb3_frontier_exploration/config/params.yaml"
    frontier = yaml.safe_load(frontier_path.read_text())["goal_assignment_node"]["ros__parameters"]
    goal_checker = config["controller_server"]["ros__parameters"]["general_goal_checker"]
    original_checker = upstream["controller_server"]["ros__parameters"]["general_goal_checker"]
    expected_checker = dict(original_checker)
    if mode == "locateanything":
        expected_checker["yaw_goal_tolerance"] = 0.10
    assert goal_checker == expected_checker
    assert goal_checker["xy_goal_tolerance"] == 0.25
    assert config["controller_server"]["ros__parameters"]["goal_checker_plugins"] == ["general_goal_checker"]
    minimum_translation = frontier["min_frontier_distance"] - goal_checker["xy_goal_tolerance"]
    outer_scan_gate = math.sqrt(0.8) * slam["ros__parameters"]["minimum_travel_distance"]
    assert minimum_translation > 2 * outer_scan_gate
    for key, value in expected_overrides.items():
        assert slam["ros__parameters"][key] == value
        slam["ros__parameters"][key] = original_slam["slam_toolbox"]["ros__parameters"][key]
    assert {"slam_toolbox": slam} == original_slam
    planner = config["planner_server"]["ros__parameters"]["GridBased"]
    assert planner["allow_unknown"] is False
    assert planner["tolerance"] == 0.05
    planner["allow_unknown"] = upstream["planner_server"]["ros__parameters"]["GridBased"]["allow_unknown"]
    planner["tolerance"] = upstream["planner_server"]["ros__parameters"]["GridBased"]["tolerance"]
    # Apart from the declared controller, planner and LA yaw overrides, watchdogs, collision
    # geometry, costmaps, smoother limits and all other Nav2 parameters stay intact.
    config["controller_server"]["ros__parameters"]["FollowPath"] = original
    config["controller_server"]["ros__parameters"]["general_goal_checker"] = original_checker
    assert config == upstream


def test_default_slam_profile_is_simulation_world_odometry():
    pytest.importorskip("launch")
    assert _generated_config() == _generated_config("simulation_world_odom")


@pytest.mark.parametrize("filename", ["nav.launch.py", "full_stack.launch.py", "locateanything_demo.launch.py"])
def test_slam_profile_choices_reject_typos(filename):
    pytest.importorskip("launch")
    with pytest.raises(RuntimeError, match="slam_profile"):
        _context_and_description(filename, "simulation_typo")


@pytest.mark.parametrize("filename", ["full_stack.launch.py", "locateanything_demo.launch.py"])
@pytest.mark.parametrize("profile", [None, "simulation_world_odom", "default"])
def test_slam_profile_is_explicitly_forwarded_through_wrappers(filename, profile):
    pytest.importorskip("launch")
    from launch.actions import GroupAction, IncludeLaunchDescription, TimerAction
    from launch.utilities import perform_substitutions
    from launch.utilities import normalize_to_list_of_substitutions

    context, description = _context_and_description(filename, profile)
    expected = "simulation_world_odom" if profile is None else profile
    assert context.launch_configurations["slam_profile"] == expected
    actions = list(description.entities)
    includes = []
    while actions:
        action = actions.pop()
        if action.condition is not None and not action.condition.evaluate(context):
            continue
        if isinstance(action, TimerAction):
            actions.extend(action.actions)
        elif isinstance(action, GroupAction):
            actions.extend(action.get_sub_entities())
        elif isinstance(action, IncludeLaunchDescription):
            args = dict(action.launch_arguments)
            if "slam_profile" in args:
                includes.append(action)
                assert perform_substitutions(context, normalize_to_list_of_substitutions(args["slam_profile"])) == expected
    assert len(includes) == 1


def test_navigation_default_mode_uses_locateanything_goal_tolerances():
    assert _generated_config() == _generated_config(mode="locateanything")
    assert _generated_config()["controller_server"]["ros__parameters"]["general_goal_checker"]["yaw_goal_tolerance"] == 0.10


@pytest.mark.parametrize("filename", ["nav.launch.py", "full_stack.launch.py"])
def test_navigation_mode_rejects_typos(filename):
    pytest.importorskip("launch")
    with pytest.raises(RuntimeError, match="mode"):
        _context_and_description(filename, mode="locateanything_typo")


@pytest.mark.parametrize("mode", [None, "locateanything"])
def test_full_stack_forwards_mode_to_nav(mode):
    from launch.actions import GroupAction, IncludeLaunchDescription, TimerAction
    from launch.utilities import perform_substitutions, normalize_to_list_of_substitutions

    context, description = _context_and_description("full_stack.launch.py", mode=mode)
    expected = "locateanything" if mode is None else mode
    actions, forwarded = list(description.entities), []
    while actions:
        action = actions.pop()
        if action.condition is not None and not action.condition.evaluate(context):
            continue
        if isinstance(action, TimerAction):
            actions.extend(action.actions)
        elif isinstance(action, GroupAction):
            actions.extend(action.get_sub_entities())
        elif isinstance(action, IncludeLaunchDescription):
            args = dict(action.launch_arguments)
            if "slam_profile" in args:
                forwarded.append(perform_substitutions(context, normalize_to_list_of_substitutions(args["mode"])))
    assert forwarded == [expected]


def test_demo_forwards_locateanything_mode_for_every_scenario():
    from launch.actions import IncludeLaunchDescription
    from launch.utilities import perform_substitutions, normalize_to_list_of_substitutions

    context, description = _context_and_description("locateanything_demo.launch.py")
    includes = [action for action in description.entities if isinstance(action, IncludeLaunchDescription)]
    assert len(includes) == 3
    for action in includes:
        args = dict(action.launch_arguments)
        assert perform_substitutions(context, normalize_to_list_of_substitutions(args["mode"])) == "locateanything"
