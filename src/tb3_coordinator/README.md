# `tb3_coordinator`

Coordinates exploration and object navigation. A `/user_command` pauses
exploration, forwards the command to LocateAnything, and waits for a selected
target and safe approach pose before asking Nav2 to drive there.

It reports `TARGET_REACHED` or `TARGET_FAILED`. The demo then waits for the
next command; `auto_resume_exploration:=true` enables automatic resumption.
Visual memory belongs to `tb3_locateanything`.

- Node: `coordinator_node`, started by `backend.launch.py`.
- [Configuration](config/coordinator.yaml) · [Main README](../../README.md).
