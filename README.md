# TurtleBot3 Semantic Navigation — Project 3

A TurtleBot3 explores a room, builds a map, and remembers the objects it sees.
You then give it a command, and it navigates to a remembered object.
This project builds on [Yiyuchen Hu's ROS 2 course workspace](https://github.com/YiyuchenHu/turtlebot3-gazebo-navigation-hw).

## Two ways to run

| Mode | How it finds objects | Example command |
|---|---|---|
| **YOLO baseline** | Detects people, chairs, and trash cans; uses LiDAR to estimate their positions | `go to person 0` |
| **LocateAnything** | Builds its own visual memory and uses RGB-D to locate objects; accepts descriptions of color, appearance, or purpose | `go to the red chair` |

LocateAnything runs without YOLO. The recent updates keep the YOLO detection,
localization, memory, and query code unchanged. Both modes use the updated
shared SLAM, exploration, and navigation components.

**Map first, query later:** let the robot observe the room before asking it to
visit an object. A command selects from remembered objects; it does not guarantee
that an unseen object will be found.

<a id="setup-by-platform"></a>

## Setup and build

Use **Ubuntu 22.04, ROS 2 Humble, and Gazebo Classic 11**.
The YOLO weights are included in this repository. LocateAnything additionally
needs its external model, Python environment, Eagle source, and a working CUDA GPU.
These are already configured on the development machine.

<details>
<summary>First-time setup on another machine</summary>

Follow the [course setup](https://github.com/YiyuchenHu/turtlebot3-gazebo-navigation-hw#setup-by-platform)
for ROS, Gazebo, and the YOLO Python dependencies, then clone this project:

```bash
git clone https://github.com/lllingshen/project3a.git ~/project3a
```

The updated workspace also uses these packages:

```bash
sudo apt install nlohmann-json3-dev python3-numpy python3-yaml \
  ros-humble-nav2-regulated-pure-pursuit-controller
```

For LocateAnything, set the paths to your existing installation **before**
sourcing the environment helper:

```bash
export LOCATEANYTHING_PYTHON=/path/to/model-environment/bin/python
export LOCATEANYTHING_MODEL=/path/to/LocateAnything-3B
export LOCATEANYTHING_EAGLE=/path/to/Eagle
```

Setting paths does not install the model or its dependencies. For YOLO alone,
PyTorch and Ultralytics must be importable by the ROS Python environment;
the LocateAnything model and GPU are not required.

</details>

Build from the repository root. The examples use the current local checkout;
replace the `cd` path if you cloned elsewhere.

```bash
cd ~/Desktop/CSE498/LAB3
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PATH=/usr/bin:/bin bash --noprofile --norc -c \
  'source /opt/ros/humble/setup.bash; colcon build --symlink-install --executor sequential'
```

This builds with ROS's system Python. Keep `--symlink-install` on later builds.

<a id="running"></a>

## Prepare every terminal

Before running either mode, execute this in **each terminal**:

```bash
cd ~/Desktop/CSE498/LAB3
source scripts/project3_env.sh
```

The helper loads ROS, the built packages, and the local model environment.
Run one simulation at a time. Stop the running launch with `Ctrl+C` before
switching modes or scenes; a fresh launch rebuilds the map and object memory.

## Run LocateAnything

Use **two terminals**.

**Terminal A — start a scene:**

```bash
ros2 launch tb3_bringup locateanything_demo.launch.py scenario:=chairs
```

Choose a scene by changing `scenario`:

| Scene | Objects | Commands to try |
|---|---|---|
| `chairs` | One red chair and one blue chair | `go to the red chair`, `go to the blue chair` |
| `warehouse_objects` | Two forklifts, three fire extinguishers, three storage racks | `go to a random forklift`, `go to a fire extinguisher` |
| `warehouse_colors` | Two people, green/blue tables, white/black chairs | `go to the green table`, `go to the white chair` |

The robot starts from a random safe pose and begins exploring when navigation
is ready. There is no fixed startup countdown or mandatory initial spin.
Let it explore until the target appears in its confirmed memory:

```bash
ros2 topic echo --once /locateanything/memory --qos-durability transient_local
```

**Terminal B — send a command:**

```bash
ros2 topic pub --once /user_command std_msgs/String "data: 'go to the red chair'"
```

Replace the text inside the single quotes to try another command. You can also
use a confirmed ID shown in memory, such as `go to chair_0`. IDs depend on the
current run; `chair_0` does not always mean the red chair.

**Wait for `TARGET_REACHED` or `TARGET_FAILED` before sending the next command.**
Watch status in terminal A, or use this in terminal B and press `Ctrl+C` when
you want to type another command:

```bash
ros2 topic echo /coordinator_node/status
```

After a command succeeds or fails, this demo stays stopped and accepts the next
command. Add `auto_resume_exploration:=true` to the launch command if you want
it to continue exploring afterward. Add `spawn_seed:=498` to repeat a starting
pose with the same scene.

Start with explicit colors and object names. Descriptions such as
`go to the red object I can sit on` also go to the model, but purpose and spatial
relations can be misunderstood. **Reaching a target does not prove that the
correct object was selected.**

## Run the YOLO baseline

Use the same terminal preparation above. The course detection, localization,
memory, and query chain is available through `mode:=baseline`.

**Terminal A — start the original person scene:**

```bash
ros2 launch tb3_bringup full_stack.launch.py mode:=baseline world:=warehouse_models_person
```

Wait for `Managed nodes are active`, then let the robot explore until people
appear as object markers in RViz. Unlike the LocateAnything launcher, this
launcher retains the course startup delays and warmup scan.

**Terminal B — send a command:**

```bash
ros2 topic pub --once /user_command std_msgs/String "data: 'go to person'"
```

To choose a particular person, use an observed ID, for example `go to person 0`.
Wait for the command to finish before sending another. Exploration resumes
automatically by default.

For the chair and trash-can scene, restart terminal A with:

```bash
ros2 launch tb3_bringup full_stack.launch.py mode:=baseline world:=warehouse_models
```

After those objects appear in memory, try `go to chair` or `go to trash can`.
The YOLO model supports the three course classes; use LocateAnything for the
forklift, fire-extinguisher, and storage-rack scene.

<details>
<summary>Original six-terminal workflow</summary>

Use this **instead of** the one-command launch above. Prepare each terminal,
then start them in order, waiting for each ready signal.

| Terminal | Command | Ready when |
|---|---|---|
| T1 | `ros2 launch tb3_bringup sim.launch.py` | Gazebo shows the robot |
| T2 | `ros2 launch tb3_bringup nav.launch.py` | `Managed nodes are active` |
| T3 | `ros2 launch tb3_bringup backend.launch.py mode:=baseline` | Coordinator is ready and exploration starts |
| T4 | `ros2 launch tb3_bringup localizer.launch.py` | `LocalizerNode ready` |
| T5 | `ros2 launch tb3_bringup detector.launch.py` | Model is loaded and detection boxes appear |
| T6 | Send `/user_command` using the examples above | The target is already in memory |

</details>

## Repository layout

| Package | Role |
|---|---|
| `tb3_bringup` | Launch files, scenes, robot setup, and RViz configuration |
| `tb3_detector` | YOLO object detection |
| `tb3_localizer` | YOLO detection boxes + LiDAR → object positions |
| `tb3_memory` | YOLO object memory |
| `tb3_query` | YOLO command parsing and target lookup |
| `tb3_locateanything` | LocateAnything inference, visual memory, RGB-D localization, and target selection |
| `tb3_coordinator` | YOLO map landmarks and shared exploration/navigation coordination |
| `tb3_nav_adapter` | Choose a safe position near the selected object |
| `tb3_frontier_exploration` | Explore unmapped parts of the room |

## More

- **LocateAnything details and warehouse examples:** [warehouse guide](docs/warehouse_examples.md).
- **Course requirements:** [INSTRUCTIONS.md](INSTRUCTIONS.md).
- **Troubleshooting:** [course troubleshooting](docs/troubleshooting.md).
- **Course background:** [NOTES.md](NOTES.md).
- **Earlier project report and measurements:** [report](report/report.pdf),
  [initial results](report/results.json), [visual-memory results](report/visual_memory_results.json),
  and [warehouse results](report/warehouse_results.json).

The report and saved measurements describe earlier versions and conditions;
they do not validate every behavior of the current RGB-D pipeline.

## Credits and license

Based on [Yiyuchen Hu's course project](https://github.com/YiyuchenHu/turtlebot3-gazebo-navigation-hw).
Detector and geometry helpers also draw on
[Ling Shen's earlier semantic-navigation work](https://github.com/lllingshen/turtlebot3-semantic-research).
Code is MIT-licensed; see [LICENSE](LICENSE). Third-party Gazebo models retain
their own licenses and attribution in [NOTICE](NOTICE).
