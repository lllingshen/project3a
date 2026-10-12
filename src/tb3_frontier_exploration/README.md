# `tb3_frontier_exploration`

Autonomous exploration in C++. `frontier_detection_node` finds boundaries
between mapped and unknown space. `goal_assignment_node` chooses goals and
sends them to Nav2, with map, transform, lifecycle, and collision checks.
The optional `startup_map_warmup_node.py` performs an initial rotation.

The coordinator pauses exploration while serving an object-navigation command.
LocateAnything's observation planner can request viewpoints that help build memory.

- [Configuration](config/params.yaml) · [Main README](../../README.md).
