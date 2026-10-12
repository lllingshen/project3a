"""Evaluate ROS launch conditions without starting processes or loading models."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("launch_ros")
import launch.logging
from launch import LaunchContext, LaunchDescription, LaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters, normalize_parameters


ROOT = Path(__file__).resolve().parents[1]
SHARED = {"nav_goal_adapter_node", "coordinator_node"}
EXPLORATION = {"frontier_detection_node", "goal_assignment_node"}
WARMUP = "startup_map_warmup_node.py"


@pytest.fixture(autouse=True)
def isolate_launch_logs(monkeypatch, tmp_path):
    # Replacing the property avoids reading its lazy getter, which would first
    # create ~/.ros/log before an instance-level monkeypatch could replace it.
    monkeypatch.setattr(type(launch.logging.launch_config), "log_dir",
                        property(lambda self: str(tmp_path)))


def launch_file(name, monkeypatch):
    path = ROOT / "src/tb3_bringup/launch" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only package lookup and sub-file loading are isolated. Actual launch
    # arguments, substitutions, conditions, timers and node actions are tested.
    monkeypatch.setattr(module, "get_package_share_directory",
                        lambda package: str(ROOT / "src" / package))
    if hasattr(module, "Node"):
        class InspectableNode(Node):
            def __init__(self, **kwargs):
                self.test_parameters = kwargs.get("parameters", [])
                self.test_name = kwargs.get("name")
                super().__init__(**kwargs)
        monkeypatch.setattr(module, "Node", InspectableNode)
    if hasattr(module, "PythonLaunchDescriptionSource"):
        monkeypatch.setattr(module, "PythonLaunchDescriptionSource", lambda location:
                            LaunchDescriptionSource(LaunchDescription(), location=location))
    return module.generate_launch_description()


def context_for(description, **arguments):
    context = LaunchContext()
    context.launch_configurations.update(arguments)
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    return context


def text(context, value):
    return perform_substitutions(context, normalize_to_list_of_substitutions(value))


def scheduled_actions(actions, context, delay=0.0):
    """Resolve active launch branches and their earliest scheduled start time."""
    for action in actions:
        if action.condition is not None and not action.condition.evaluate(context):
            continue
        if isinstance(action, TimerAction):
            period = (action.period if isinstance(action.period, (int, float))
                      else float(text(context, action.period)))
            yield from scheduled_actions(action.actions, context,
                                         delay + period)
        elif isinstance(action, GroupAction):
            yield from scheduled_actions(action.get_sub_entities(), context, delay)
        else:
            yield delay, action


def selected_actions(actions, context):
    for _, action in scheduled_actions(actions, context):
        yield action


def node_parameters(node, context):
    """Read effective parameters, honoring inline overrides after YAML files."""
    evaluated = evaluate_parameters(context, normalize_parameters(node.test_parameters))
    result = {}
    for block in evaluated:
        if isinstance(block, Path):
            result.update(yaml.safe_load(block.read_text())[
                text(context, node.test_name)]["ros__parameters"])
        else:
            result.update(block)
    return result


@pytest.mark.parametrize("mode", ["locateanything"])
@pytest.mark.parametrize("explore", ["true", "false"])
def test_backend_runs_exactly_one_semantic_pipeline(monkeypatch, mode, explore):
    description = launch_file("backend.launch.py", monkeypatch)
    context = context_for(description, mode=mode, explore=explore)
    executables = [
        text(context, action.node_executable)
        for action in selected_actions(description.entities, context)
        if isinstance(action, Node)
    ]
    perception = {"locateanything_node", "observation_node"}
    exploration = set(EXPLORATION) if explore == "true" else set()
    assert set(executables) == perception | SHARED | exploration
    assert len(executables) == len(set(executables))
    if mode == "locateanything":
        observer = next(action for action in selected_actions(description.entities, context)
                        if isinstance(action, Node)
                        and text(context, action.node_executable) == "observation_node")
        parameters = evaluate_parameters(context, normalize_parameters(observer.test_parameters))
        params = {key: value for block in parameters for key, value in block.items()}
        assert params["initial_exploration_enabled"] is (explore == "true")


def test_backend_shares_custom_camera_calibration_topic_between_perception_and_observer(monkeypatch):
    description = launch_file('backend.launch.py', monkeypatch)
    context = context_for(description, mode='locateanything', camera_info_topic='/custom/camera_info')
    checked = set()
    for action in selected_actions(description.entities, context):
        if not isinstance(action, Node):
            continue
        executable = text(context, action.node_executable)
        if executable in {'locateanything_node', 'observation_node'}:
            blocks = evaluate_parameters(context, normalize_parameters(action.test_parameters))
            params = {key: value for block in blocks for key, value in block.items()}
            assert params['camera_info_topic'] == '/custom/camera_info'
            checked.add(executable)
    assert len(checked) == 2


@pytest.mark.parametrize("mode", ["locateanything"])
def test_full_stack_starts_only_the_locateanything_stack(monkeypatch, mode):
    description = launch_file("full_stack.launch.py", monkeypatch)
    context = context_for(description, mode=mode, explore="false", evidence_dir="/tmp/la-evidence")
    includes = {}
    for action in selected_actions(description.entities, context):
        if isinstance(action, IncludeLaunchDescription):
            source = action.launch_description_source
            source.get_launch_description(context)
            assert Path(source.location).name not in includes
            includes[Path(source.location).name] = action
    expected = {"sim.launch.py", "nav.launch.py", "backend.launch.py"}
    assert set(includes) == expected
    backend_arguments = {
        text(context, name): text(context, value)
        for name, value in includes["backend.launch.py"].launch_arguments
    }
    assert backend_arguments["mode"] == mode
    assert backend_arguments["explore"] == "false"
    assert backend_arguments["evidence_dir"] == "/tmp/la-evidence"


@pytest.mark.parametrize("mode,expected", [
    ("locateanything", {"sim.launch.py": 0.0, "nav.launch.py": 0.0,
                        "backend.launch.py": 0.0}),
])
def test_stack_starts_locateanything_and_slam_without_fixed_delays(monkeypatch, mode, expected):
    description = launch_file("full_stack.launch.py", monkeypatch)
    context = context_for(description, mode=mode)
    scheduled = []
    for delay, action in scheduled_actions(description.entities, context):
        if isinstance(action, IncludeLaunchDescription):
            source = action.launch_description_source
            source.get_launch_description(context)
            scheduled.append((Path(source.location).name, delay))
    assert len(scheduled) == len(expected)
    assert dict(scheduled) == expected


@pytest.mark.parametrize("mode", ["locateanything"])
@pytest.mark.parametrize("explore", ["true", "false"])
@pytest.mark.parametrize("startup_scan", [None, "true", "false"])
def test_scan_node_and_exploration_gate_always_agree(monkeypatch, mode, explore, startup_scan):
    arguments = {"mode": mode, "explore": explore}
    if startup_scan is not None:
        arguments["startup_scan"] = startup_scan
    description = launch_file("backend.launch.py", monkeypatch)
    context = context_for(description, **arguments)
    nodes = {text(context, action.node_executable): action
             for action in selected_actions(description.entities, context)
             if isinstance(action, Node)}
    requires_scan = False if startup_scan is None else startup_scan == "true"
    assert (WARMUP in nodes) is (explore == "true" and requires_scan)
    assert ("goal_assignment_node" in nodes) is (explore == "true")
    if explore == "true":
        params = node_parameters(nodes["goal_assignment_node"], context)
        assert type(params["require_startup_warmup"]) is bool
        assert params["require_startup_warmup"] is requires_scan
        # Starting immediately must still wait for active Nav2 lifecycle nodes;
        # an advertised action server alone can precede activation.
        assert type(params["require_nav2_active"]) is bool
        assert params["require_nav2_active"] is (mode == "locateanything")


@pytest.mark.parametrize("filename", ["backend.launch.py", "full_stack.launch.py"])
def test_launch_defaults_use_locateanything_and_depth(monkeypatch, filename):
    description = launch_file(filename, monkeypatch)
    context = context_for(description)
    assert context.launch_configurations["mode"] == "locateanything"
    assert context.launch_configurations["explore"] == "true"
    assert context.launch_configurations["auto_resume_exploration"] == "true"
    assert context.launch_configurations["warmup_full_rotation"] == "false"
    assert context.launch_configurations["startup_scan"].lower() == "false"
    assert context.launch_configurations["localization_mode"] == "depth"


@pytest.mark.parametrize("categories", [
    ["chair", "person", "trash can"],
    ["forklift", "fire extinguisher", "storage rack"],
    ["person", "table", "chair"],
])
def test_catalog_class_argument_reaches_ros_as_a_string_array(monkeypatch, categories):
    stack = launch_file("full_stack.launch.py", monkeypatch)
    stack_context = context_for(stack, mode="locateanything", memory_classes=json.dumps(categories))
    includes = [a for a in selected_actions(stack.entities, stack_context)
                if isinstance(a, IncludeLaunchDescription)]
    # Only backend receives the category list; simulator world coordinates and
    # target entity names are never injected into the perception parameters.
    forwarded = [dict((text(stack_context, k), text(stack_context, v))
                      for k, v in action.launch_arguments) for action in includes]
    backend_args = next(args for args in forwarded if "memory_classes" in args)
    backend = launch_file("backend.launch.py", monkeypatch)
    context = context_for(backend, **backend_args)
    node = next(a for a in selected_actions(backend.entities, context)
                if isinstance(a, Node) and text(context, a.node_executable) == "locateanything_node")
    evaluated = evaluate_parameters(context, normalize_parameters(node.test_parameters))
    params = {key: value for block in evaluated for key, value in block.items()}
    assert isinstance(params["memory_classes"], tuple)
    assert list(params["memory_classes"]) == categories
    assert all(isinstance(category, str) for category in params["memory_classes"])
    assert "world" not in params and "x_pose" not in params and "y_pose" not in params


@pytest.mark.parametrize("scenario,spawn,classes", [
    ("chairs", ("AUTO", "AUTO"), ["chair", "person", "trash can"]),
    ("warehouse_objects", ("AUTO", "AUTO"), ["forklift", "fire extinguisher", "storage rack"]),
    ("warehouse_colors", ("AUTO", "AUTO"), ["person", "table", "chair"]),
])
def test_demo_profile_selects_its_world_and_catalog(monkeypatch, scenario, spawn, classes):
    description = launch_file("locateanything_demo.launch.py", monkeypatch)
    context = context_for(description, scenario=scenario, explore="false",
                          use_rviz="false", evidence_dir="/tmp/demo-check")
    includes = [a for a in selected_actions(description.entities, context)
                if isinstance(a, IncludeLaunchDescription)]
    assert len(includes) == 1
    action = includes[0]
    action.launch_description_source.get_launch_description(context)
    assert Path(action.launch_description_source.location).name == "full_stack.launch.py"
    args = {text(context, k): text(context, v) for k, v in action.launch_arguments}
    assert args["mode"] == "locateanything" and args["explore"] == "false"
    assert args["auto_resume_exploration"] == "false"
    assert Path(args["world"]).name == "locateanything_" + scenario + ".world"
    assert (args["x_pose"], args["y_pose"]) == spawn
    assert json.loads(args["memory_classes"]) == classes
    assert args["use_rviz"] == "false" and args["evidence_dir"] == "/tmp/demo-check"
    assert args["startup_scan"] == "false"
    backend = launch_file("backend.launch.py", monkeypatch)
    backend_context = context_for(backend, **{**args, "explore": "true"})
    nodes = {text(backend_context, a.node_executable): a
             for a in selected_actions(backend.entities, backend_context) if isinstance(a, Node)}
    assert WARMUP not in nodes
    assert node_parameters(nodes["goal_assignment_node"], backend_context)["require_startup_warmup"] is False


@pytest.mark.parametrize("scenario", ["chairs", "warehouse_objects", "warehouse_colors"])
def test_demo_scan_can_be_explicitly_enabled_with_original_parameters(monkeypatch, scenario):
    description = launch_file("locateanything_demo.launch.py", monkeypatch)
    context = context_for(description, scenario=scenario, startup_scan="true")
    action = next(a for a in selected_actions(description.entities, context)
                  if isinstance(a, IncludeLaunchDescription))
    args = {text(context, k): text(context, v) for k, v in action.launch_arguments}
    stack = launch_file("full_stack.launch.py", monkeypatch)
    stack_context = context_for(stack, **args)
    forwarded = [{text(stack_context, k): text(stack_context, v) for k, v in a.launch_arguments}
                 for a in selected_actions(stack.entities, stack_context)
                 if isinstance(a, IncludeLaunchDescription)]
    args = next(item for item in forwarded if "startup_scan" in item)
    assert args["startup_scan"] == "true"
    expected_warmup = (70.0, 12.0, 2.0)
    backend = launch_file("backend.launch.py", monkeypatch)
    backend_context = context_for(backend, **{**args, "explore": "true"})
    warmup = next(a for a in selected_actions(backend.entities, backend_context)
                  if isinstance(a, Node)
                  and text(backend_context, a.node_executable) == "startup_map_warmup_node.py")
    evaluated = evaluate_parameters(backend_context, normalize_parameters(warmup.test_parameters))
    params = {key: value for block in evaluated for key, value in block.items()}
    assert (params["scan_angle_deg"], params["startup_delay_sec"], params["pause_between_sec"]) == expected_warmup
    assert params["full_rotation"] is True
    assert type(params["scan_angle_deg"]) is float and type(params["startup_delay_sec"]) is float
    assert type(params["pause_between_sec"]) is float


def test_demo_rejects_unknown_scenario(monkeypatch):
    description = launch_file("locateanything_demo.launch.py", monkeypatch)
    with pytest.raises(RuntimeError, match="scenario"):
        context_for(description, scenario="unknown")


def test_demo_explores_by_default_to_observe_objects_outside_initial_view(monkeypatch):
    description = launch_file("locateanything_demo.launch.py", monkeypatch)
    context = context_for(description, scenario="warehouse_objects")
    action = next(a for a in selected_actions(description.entities, context)
                  if isinstance(a, IncludeLaunchDescription))
    args = {text(context, k): text(context, v) for k, v in action.launch_arguments}
    assert args["explore"] == "true"


def test_full_stack_warmup_overrides_reach_ros_as_floats(monkeypatch):
    description = launch_file("full_stack.launch.py", monkeypatch)
    context = context_for(description, mode="locateanything",
                          startup_scan="true",
                          warmup_scan_angle_deg="80", warmup_startup_delay_sec="9",
                          warmup_pause_sec="4")
    includes = [a for a in selected_actions(description.entities, context)
                if isinstance(a, IncludeLaunchDescription)]
    forwarded = [{text(context, k): text(context, v) for k, v in action.launch_arguments}
                 for action in includes]
    args = next(item for item in forwarded if "warmup_scan_angle_deg" in item)
    backend = launch_file("backend.launch.py", monkeypatch)
    backend_context = context_for(backend, **args)
    warmup = next(a for a in selected_actions(backend.entities, backend_context)
                  if isinstance(a, Node)
                  and text(backend_context, a.node_executable) == "startup_map_warmup_node.py")
    evaluated = evaluate_parameters(backend_context, normalize_parameters(warmup.test_parameters))
    params = {key: value for block in evaluated for key, value in block.items()}
    assert params["scan_angle_deg"] == 80.0 and type(params["scan_angle_deg"]) is float
    assert params["startup_delay_sec"] == 9.0 and type(params["startup_delay_sec"]) is float
    assert params["pause_between_sec"] == 4.0 and type(params["pause_between_sec"]) is float


@pytest.mark.parametrize("filename,arguments,expected", [
    ("backend.launch.py", {}, True),
    ("backend.launch.py", {"auto_resume_exploration": "false"}, False),
    ("full_stack.launch.py", {}, True),
    ("full_stack.launch.py", {"auto_resume_exploration": "false"}, False),
    ("locateanything_demo.launch.py", {}, False),
    ("locateanything_demo.launch.py", {"auto_resume_exploration": "true"}, True),
])
def test_resume_policy_reaches_coordinator_and_overrides_yaml(monkeypatch, filename, arguments, expected):
    # Follow the selected includes down to the backend, evaluating each launch
    # boundary. No processes are started and the actual coordinator YAML is read.
    while True:
        description = launch_file(filename, monkeypatch)
        context = context_for(description, **arguments)
        if filename == "backend.launch.py":
            break
        filename = ("full_stack.launch.py" if filename == "locateanything_demo.launch.py"
                    else "backend.launch.py")
        matches = []
        for action in selected_actions(description.entities, context):
            if isinstance(action, IncludeLaunchDescription):
                source = action.launch_description_source
                source.get_launch_description(context)
                if Path(source.location).name == filename:
                    matches.append(action)
        assert len(matches) == 1
        arguments = {text(context, k): text(context, v) for k, v in matches[0].launch_arguments}

    coordinator = next(a for a in selected_actions(description.entities, context)
                       if isinstance(a, Node)
                       and text(context, a.node_executable) == "coordinator_node")
    evaluated = evaluate_parameters(context, normalize_parameters(coordinator.test_parameters))
    effective = {}
    for block in evaluated:
        if isinstance(block, Path):
            effective.update(yaml.safe_load(block.read_text())["coordinator_node"]["ros__parameters"])
        else:
            effective.update(block)
    assert type(effective["auto_resume_exploration"]) is bool
    assert effective["auto_resume_exploration"] is expected


@pytest.mark.parametrize("mode,localization,rgbd,expected", [
    ("locateanything","laser","false","False"),
    ("locateanything","depth","false","True"),
    ("locateanything","laser","true","True"),
])
def test_rgbd_camera_enabled_for_depth_mode(monkeypatch,mode,localization,rgbd,expected):
    description=launch_file("full_stack.launch.py",monkeypatch)
    context=context_for(description,mode=mode,localization_mode=localization,rgbd_camera=rgbd,
                        spawn_seed="498",yaw_pose="1.2",spawn_margin="0.2")
    arguments={}
    for action in selected_actions(description.entities,context):
        if isinstance(action,IncludeLaunchDescription):
            source=action.launch_description_source; source.get_launch_description(context)
            arguments[Path(source.location).name]={text(context,k):text(context,v) for k,v in action.launch_arguments}
    assert arguments["sim.launch.py"]["rgbd_camera"] == expected
    assert arguments["sim.launch.py"]["spawn_seed"] == "498"
    assert arguments["sim.launch.py"]["yaw_pose"] == "1.2"
    assert arguments["sim.launch.py"]["spawn_margin"] == "0.2"
    assert arguments["backend.launch.py"]["localization_mode"] == localization
    assert "spawn_seed" not in arguments["backend.launch.py"]
    assert "spawn_bounds" not in arguments["backend.launch.py"]


def test_demo_random_defaults_and_fixed_pose_overrides_forwarded(monkeypatch):
    description=launch_file("locateanything_demo.launch.py",monkeypatch)
    context=context_for(description)
    assert context.launch_configurations["localization_mode"] == "depth"
    assert context.launch_configurations["rgbd_camera"] == "true"
    assert context.launch_configurations["x_pose"] == context.launch_configurations["y_pose"] == "AUTO"
    context=context_for(description,x_pose="-1.5",y_pose="0",yaw_pose="0",spawn_seed="99")
    action=next(a for a in selected_actions(description.entities,context) if isinstance(a,IncludeLaunchDescription))
    args={text(context,k):text(context,v) for k,v in action.launch_arguments}
    assert [args[k] for k in ("x_pose","y_pose","yaw_pose","spawn_seed")] == ["-1.5","0","0","99"]


def sim_actions(monkeypatch, **arguments):
    from launch.actions import OpaqueFunction
    from ament_index_python.packages import get_package_share_directory as actual_share
    path=ROOT/"src/tb3_bringup/launch/sim.launch.py"
    spec=importlib.util.spec_from_file_location("test_sim_actions",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,"get_package_share_directory",lambda p:
                        str(ROOT/"src/tb3_bringup") if p=="tb3_bringup" else actual_share(p))
    class InspectSpawn(Node):
        def __init__(self, **kwargs):
            self.test_arguments=kwargs.get("arguments",[])
            super().__init__(**kwargs)
    monkeypatch.setattr(module,"Node",InspectSpawn)
    description=module.generate_launch_description()
    context=context_for(description,**arguments)
    opaque=next(a for a in description.entities if isinstance(a,OpaqueFunction))
    return opaque.execute(context)


def test_sim_seed_produces_one_verified_spawner_without_running_it(monkeypatch):
    arguments={"world":str(ROOT/"src/tb3_bringup/worlds/locateanything_warehouse_objects.world"),
               "spawn_seed":"498", "rgbd_camera":"false"}
    actions=sim_actions(monkeypatch,**arguments)
    spawner=next(a for a in actions if isinstance(a,Node))
    args=dict(zip(spawner.test_arguments[::2],spawner.test_arguments[1::2]))
    assert args["-x"] == "-0.10770457613571693"
    assert args["-y"] == "0.6656073692069402"
    assert args["-Y"] == "-1.5846674468921569"
    assert Path(args["-file"]).name.startswith("tb3_drive_")
    import xml.etree.ElementTree as ET
    tree = ET.parse(args["-file"])
    assert tree.find(".//sensor[@name='camera']").get("type") == "camera"
    assert tree.findtext(".//plugin[@filename='libgazebo_ros_diff_drive.so']/ros/remapping") == "cmd_vel:=/tb3_drive/cmd_vel"
    assert [text(LaunchContext(), a.node_executable) for a in actions if isinstance(a,Node)] == [
        "spawn_entity.py", "velocity_relay.py"]


def test_sim_fixed_overlap_raises_before_producing_any_spawn_action(monkeypatch):
    with pytest.raises(ValueError,match="collision clearance"):
        sim_actions(monkeypatch,world=str(ROOT/"src/tb3_bringup/worlds/locateanything_warehouse_objects.world"),
                    x_pose="0",y_pose="0",yaw_pose="0")


def test_sim_rgbd_keeps_verified_fixed_pose_and_uses_derived_robot(monkeypatch):
    actions=sim_actions(monkeypatch,world=str(ROOT/"src/tb3_bringup/worlds/locateanything_warehouse_objects.world"),
                        x_pose="-1.5",y_pose="0",yaw_pose=".5",rgbd_camera="true",spawn_seed="498")
    spawner=next(a for a in actions if isinstance(a,Node))
    args=dict(zip(spawner.test_arguments[::2],spawner.test_arguments[1::2]))
    assert (args["-x"],args["-y"],args["-Y"]) == ("-1.5","0.0","0.5")
    import xml.etree.ElementTree as ET
    sensors=ET.parse(args["-file"]).getroot().findall(".//sensor")
    assert any(sensor.get("type")=="depth" for sensor in sensors)


@pytest.mark.parametrize("filename", ["backend.launch.py", "full_stack.launch.py", "nav.launch.py"])
def test_removed_pipeline_cannot_be_selected(monkeypatch, filename):
    description = launch_file(filename, monkeypatch)
    with pytest.raises(RuntimeError, match="mode"):
        context_for(description, mode="baseline")
