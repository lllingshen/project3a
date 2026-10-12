# Launch files

Use `locateanything_demo.launch.py scenario:=warehouse_objects` to start a demo.
All semantic launch entry points use LocateAnything.

| Launch | Starts |
|---|---|
| `locateanything_demo.launch.py` | One of three scenes with the corresponding catalog classes |
| `full_stack.launch.py` | Simulator, navigation, and backend together |
| `sim.launch.py` | Gazebo, TurtleBot3, safe spawn, and velocity relay |
| `nav.launch.py` | SLAM Toolbox, Nav2, and RViz |
| `backend.launch.py` | LocateAnything, observation planning, navigation adapter, coordinator, and frontier exploration |

The demo defaults to aligned-depth localization and waits after each command.
`startup_scan:=true` enables the optional initial rotation. Exploration waits
for active navigation, transforms, and map/costmap readiness before moving.

For separate launches, start `sim.launch.py rgbd_camera:=true`, `nav.launch.py`,
and `backend.launch.py localization_mode:=depth` in terminals that sourced the
workspace. Use the demo wrapper when you want its scene-specific catalog.

See the [running reference](../../../docs/running-reference.md) for options and
the [main README](../../../README.md) for build and command examples.
