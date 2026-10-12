# `tb3_nav_adapter`

Converts a selected object into a safe Nav2 approach pose. It checks the
costmap and object geometry, leaves room between the robot and the object,
and faces the selected target. The target arrives through
`tb3_interfaces/msg/SemanticQueryResult`.

- Node: `nav_goal_adapter_node`, started by `backend.launch.py`.
- [Configuration](config/nav_goal_adapter.yaml) · [Main README](../../README.md).
