# Troubleshooting LocateAnything

## The launch or a Python import fails

Build all six packages and source `scripts/project3_env.sh` in each new terminal.
Check `LOCATEANYTHING_PYTHON`, `LOCATEANYTHING_MODEL`, and `LOCATEANYTHING_EAGLE`.
The model worker needs its own compatible environment and GPU; ROS uses system Python.
If updating an existing checkout, rebuild in a fresh workspace to avoid loading
stale generated message types or old installed packages.

## The robot waits at startup

Exploration waits for Nav2 lifecycle activation, a map, valid TF, and a fresh
costmap. Inspect terminal A for the first error. The default demo does not
perform a stationary startup rotation.

## A requested object is missing

Let the robot explore and check its confirmed memory:

```bash
ros2 topic echo --once /locateanything/memory --qos-durability transient_local
```

An unseen or unconfirmed object cannot be selected by ID. IDs are assigned in
the current run. Depth localization requires aligned depth, camera calibration,
and valid transforms; it does not silently fall back to LiDAR.

## The robot reached a different object

Compare the command, selected box, selected ID, and physical object. A Nav2
success only establishes arrival at the chosen goal. Start with explicit
category/color commands, then try descriptions of purpose or appearance.

## Switching scenes

Stop the running launch with `Ctrl+C`, wait for its processes to exit, and start
the next scene from a terminal that sourced this workspace. Run one simulation
at a time. A new launch starts a new map and memory.

See the [warehouse guide](warehouse_examples.md) for detailed examples.
