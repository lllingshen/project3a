# Running reference

Start with the [two-terminal workflow](../README.md#run-locateanything).
Use `scenario:=chairs`, `scenario:=warehouse_objects`, or `scenario:=warehouse_colors`.

## Launch options

| Option | Effect |
|---|---|
| `explore:=false` | Keep the robot stationary for observation |
| `auto_resume_exploration:=true` | Resume exploration after a command finishes |
| `startup_scan:=true` | Enable the optional initial rotation |
| `spawn_seed:=498` | Repeat a safe starting pose for the same scene |
| `use_rviz:=false use_gazebo_gui:=false` | Run without graphical windows |
| `localization_mode:=laser rgbd_camera:=false` | Use LocateAnything's optional camera/LiDAR localization; objects must intersect the scan plane |
| `slam_profile:=default` | Use upstream SLAM loop-closure settings instead of the simulation world-odometry profile |
| `evidence_dir:=/tmp/locateanything-evidence` | Save query evidence outside Git |

For a custom world, use `full_stack.launch.py world:=/absolute/path/to/scene.world`
and supply its `memory_classes` as a YAML string list. This starts simulation,
navigation, and LocateAnything immediately; exploration waits for readiness.
The three demo profiles are the recommended entry points.

## Files to change

| Task | Location |
|---|---|
| Add a scene or its catalog classes | [demo launch](../src/tb3_bringup/launch/locateanything_demo.launch.py) and [worlds](../src/tb3_bringup/worlds/) |
| Perception, memory, and natural-language selection | [LocateAnything package](../src/tb3_locateanything/tb3_locateanything/) |
| Shared ROS message | [SemanticQueryResult](../src/tb3_interfaces/msg/SemanticQueryResult.msg) |
| Exploration thresholds | [frontier parameters](../src/tb3_frontier_exploration/config/params.yaml) |
| Approach pose and object-facing geometry | [navigation adapter](../src/tb3_nav_adapter/tb3_nav_adapter/) |
| Command lifecycle and exploration control | [coordinator](../src/tb3_coordinator/tb3_coordinator/coordinator_node.py) |

Changing a message definition requires rebuilding and sourcing the workspace
in every terminal. After changing launch files or configuration, rebuild with
`colcon build --symlink-install` and restart the affected launch.
