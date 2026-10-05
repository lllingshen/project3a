# TurtleBot3 Gazebo Navigation Course

## Project 3 — Ling Shen

This private assignment repository implements the detector and adds command-triggered
LocateAnything navigation. The original course instructions and reference demos remain
below; their measurements are not this project's results. See
[the English report](report/report.pdf), [LaTeX source](report/report.tex), and
[measured attempts](report/results.json).

### Build and local environment

Validated on this machine: Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11,
Python 3.10, NumPy 1.26.4, Ultralytics 8.4.60, PyTorch 2.7.1+cu128,
and an RTX 5090. No global dependency changes are needed on this machine.
The shipped `tb3det_yolo26n.pt` and whitelist (`person`, `trash_can`, `chair`)
are unchanged. All downloaded models, runtime evidence, and build outputs stay
outside Git.

```bash
cd /home/lingshen/Documents/ChatGPT/CSE498/project3a
# Build with ROS's system Python, without the model environment's setuptools.
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PATH=/usr/bin:/bin bash --noprofile --norc -c \
  'source /opt/ros/humble/setup.bash; colcon build --symlink-install --executor sequential'
```

In **every runtime terminal**, first run:

```bash
cd /home/lingshen/Documents/ChatGPT/CSE498/project3a
source scripts/project3_env.sh
```

The helper selects ROS domain **49**, localhost communication, and Gazebo port
**11349**. It adds the existing Python 3.10 model packages for the YOLO node;
LocateAnything runs in its own persistent subprocess with ROS Python paths removed.
These existing external resources are required:

```bash
export LOCATEANYTHING_PYTHON=/home/lingshen/miniforge3/envs/locateanything3b/bin/python
export LOCATEANYTHING_MODEL=/home/lingshen/research/locateanything_standalone/models/LocateAnything-3B
export LOCATEANYTHING_EAGLE=/home/lingshen/research/locateanything_standalone/Eagle
# Set overrides BEFORE sourcing scripts/project3_env.sh on another installation.
```

### Recording setup: two colored chairs

Start the following in separate prepared terminals, in order. This compact demo
uses a stationary start and a SLAM map of the visible room. `explore:=false`
omits frontier motion for repeatable recording; the baseline acceptance worlds
use the normal `explore:=true` default.

| Terminal | Command | Wait for |
|---|---|---|
| T1 | `ros2 launch tb3_bringup sim.launch.py world:="$PWD/src/tb3_bringup/worlds/locateanything_chairs.world" x_pose:=-1.2 y_pose:=0.0` | `Successfully spawned entity [waffle_pi]`; Gazebo shows both chairs |
| T2 | `ros2 launch tb3_bringup nav.launch.py` | `lifecycle_manager_navigation: Managed nodes are active`; RViz has a map |
| T3 | `ros2 launch tb3_bringup backend.launch.py mode:=locateanything explore:=false evidence_dir:="$PWD/.runtime/recording"` | `/locateanything/status` says `"state": "ready"`; coordinator ready |
| T4 | `ros2 launch tb3_bringup localizer.launch.py` | `LocalizerNode ready` |
| T5 | `ros2 launch tb3_bringup detector.launch.py` | `detector_node ready` and live image |

T4/T5 retain baseline perception for comparison. They cannot replace the target
selected by LocateAnything: only the selected command handler publishes the
query result, which contains the exact grounded map point and a unique instance ID.
Do not run another backend or query handler in parallel on this ROS domain.

RViz opens with **LocateAnything Selected Box**, **LocateAnything Target**,
**SLAM Map**, **Nav2 Plan**, **Detector Debug Image**, and **Semantic Memory Markers**.
The LocateAnything image is the frozen inference input with its original timestamp;
the detector image is live. Yellow marks the selected instance. The green plan is
`/plan`; the selected map marker is `/locateanything/target_marker`.
The standard detector may label these recolored chairs incorrectly; that is part
of the recorded comparison, and its landmarks do not control LocateAnything mode.

In T6, use the commands below. The checker sends the full command to
`/user_command`, prints coordinator status, measures the robot and **actual**
Gazebo object poses at arrival, and appends the result. Entity names are used
only by this evaluator, never by the grounding or navigation code.

```bash
python3 scripts/project3_check.py \
  --command 'Move to the red chair.' --mode locateanything \
  --truth demo_red_chair demo_blue_chair --expected-entity demo_red_chair \
  --output .runtime/recording_attempts.json --image .runtime/recording-red.png
```

Wait for the checker to finish. Confirm the selected box encloses the **red**
chair, `TARGET_REACHED` appears, and `final_distance_m <= 1.2` with
`selection_correct: true`. The status alone is insufficient. Then reset the
robot to the same visible starting view (after navigation has finished):

```bash
timeout 15 ros2 service call /gazebo/set_entity_state gazebo_msgs/srv/SetEntityState \
  "{state: {name: waffle_pi, pose: {position: {x: -1.2, y: 0.0, z: 0.01}, orientation: {w: 1.0}}, reference_frame: world}}"
```

Wait for `success=True`, a stationary robot, and both chairs in the camera view.
This resets the simulated robot only; it is not a target-selection shortcut.
One later reset applied but lost its service response during validation. If the
15-second timeout expires, stop T1–T5 and restart the demo in order; do not
assume the displayed status establishes the robot's position.

```bash
python3 scripts/project3_check.py \
  --command 'Move to the blue chair.' --mode locateanything \
  --truth demo_red_chair demo_blue_chair --expected-entity demo_blue_chair \
  --output .runtime/recording_attempts.json --image .runtime/recording-blue.png
```

To send a command without measuring, the original interface still works:

```bash
ros2 topic pub --once /user_command std_msgs/String "data: 'Move to the red chair.'"
ros2 topic echo /coordinator_node/status
```

Send one command at a time. A busy command is rejected explicitly. Missing or
ambiguous model boxes, stale sensors, robot motion during inference, failed
localization, or unavailable transforms produce failure without a new navigation
goal. A numeric confidence of `1.0` in the LocateAnything result is only the
required message-field placeholder, not calibrated model confidence.

### Baseline mode and course acceptance

Stop T3 with Ctrl-C, then use `mode:=baseline` in the same backend command to
compare the same scene/descriptions. Restarting T3 resets semantic memory; wait
for landmarks to reappear. Never run both backends together.

For the original course tests, stop T1–T5, then follow the original six-terminal
flow below, with `world:=warehouse_models_person` first and
`world:=warehouse_models` second. Use `source scripts/project3_env.sh` in each
terminal, `mode:=baseline`, and leave exploration enabled. Observe IDs from
`/semantic_map_memory_node/landmark_objects`; they depend on observation order.
Test generic `go to person`, at least two distinct observed `go to person N`
commands, then `go to trash can` and `go to chair` in the second world.
`scripts/project3_check.py --help` describes the same live measurement helper.
The inherited acceptance script remains available, but its global process
cleanup and map-origin skip were unsuitable for this concurrent local session;
the project checks explicitly covered generic selection, distinct IDs, resumption,
and measured arrival instead.

Measured baseline arrivals passed for generic person, two distinct person IDs,
and the trash can. The chair command passed at 0.973 m after an operator-guided
view and a 22-second stationary observation allowed the unchanged semantic memory
to relabel it correctly. The preceding unaided 600-second chair check failed;
this is not a fully unattended acceptance pass. The report preserves these and
the earlier failed trash-can attempts. The final LocateAnything red/blue runs
reached the correct chairs at 1.080 m and 1.170 m, respectively.

### Video checklist (about 2–4 minutes)

1. Show Gazebo, RViz, and T6; identify **LocateAnything mode** and the red/blue scene.
2. Show the red command being entered, its selected box, and the yellow map target.
3. Show the navigation path, motion, arrival, and measured final distance.
4. Label the scene reset, then repeat with the blue command and blue instance.
5. Label any speed-up. Save the completed manual recording in
   [demo_video/](demo_video/), for example as `project3_demo.mp4`.

The validated examples and limitations are in the report. They are individual
demonstrations, not a general success-rate benchmark. Targets must be visible,
stationary, intersect the LiDAR plane, and lie in the mapped scene.

Recompile the report with the project-local compiler used in this session:

```bash
XDG_CACHE_HOME="$PWD/.tools/cache" .tools/tectonic/tectonic \
  --only-cached --keep-logs --outdir report report/report.tex
```

Focused regression tests (run after building):

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PATH=/usr/bin:/bin bash --noprofile --norc -c \
  'source /opt/ros/humble/setup.bash; source install/setup.bash; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests src/tb3_locateanything/test'
```

### Attribution

- Course framework: Yiyuchen Hu's
  [turtlebot3-gazebo-navigation-hw](https://github.com/YiyuchenHu/turtlebot3-gazebo-navigation-hw),
  commit `a048d1e50e185d62cd32151e2437fab394186e06`; history, LICENSE and NOTICE preserved.
- Reused earlier work: Ling Shen's
  [turtlebot3-semantic-research](https://github.com/lllingshen/turtlebot3-semantic-research),
  commit `694fff69d87852d592be99f39f0ca77c9b04b610`: detector methods,
  strict LocateAnything output parsing, and camera/TF geometry helpers.
- External Eagle installation: commit `8442db3b79f7fd2357e468e6eecdd9b6a82049ff`;
  model and runtime remain external. New work is the online snapshot/worker bridge,
  exact-instance navigation, mode selection, colored-chair scene, and focused
  coordinate/cancellation fixes documented in the report.

`origin` is `lllingshen/project3a`; `upstream` remains the course repository with
push URL `DISABLED`. Both source checkouts were used only as references.

---

## Original course README

A ROS 2 course workspace for **object-based semantic navigation** on a
simulated TurtleBot3. The robot explores an unknown room on its own, builds a
SLAM map, recognises the objects it passes and remembers where they are — so
`go to chair` makes it stop exploring and drive to the real chair. Gazebo,
SLAM, Nav2, frontier exploration, semantic memory, command parsing and the
coordinator state machine are all provided and working; the course asks you to
write the object detector in `tb3_detector`. The system is designed *map first,
query later*: the robot builds a semantic map while it explores, and `go to X`
is a lookup in that map.

As shipped, that detector is a stub: `detector_core.py` publishes empty
detections and the rest of the pipeline waits for it. Implementing its `load()`
and `infer()` is the graded task. The assignment and acceptance criteria are in
**[INSTRUCTIONS.md](INSTRUCTIONS.md)**; rationale and measured results are in
**[NOTES.md](NOTES.md)**.

## Demo

![Map first](docs/media/demo_map.gif)

Map first: the robot explores the room on its own, and what it detects
becomes landmarks on the map (person, trash can, chair).

![Query later](docs/media/demo_query.gif)

Query later: `go to person`, `go to trash can`, `go to chair`; each command
drives to the landmark, then exploration resumes.

<sub>Sped up; the speed factor is shown in the corner.</sub>

## Repository layout

| Package (`src/`) | Role | You edit it? |
|---|---|---|
| `tb3_detector` | YOLO detection on the camera image (`tb3det_yolo26n`, fine-tuned; weights included) | **Yes — `detector_core.py` is the assignment** |
| `tb3_bringup` | every launch file, the RViz config, the Gazebo worlds, the vendored models and `semantic_targets.yaml` | No |
| `tb3_localizer` | bbox centre → bearing + LiDAR range → `(x, y)` in `base_link` | No (optional bonus, NOTES.md §9) |
| `tb3_memory` | short-term memory, stable IDs `person_0…` | No |
| `tb3_coordinator` | persistent map landmarks + the coordinator state machine | No |
| `tb3_query` | rule-based command parsing (`SemanticQueryResult` msg) | No |
| `tb3_nav_adapter` | approach-pose computation for Nav2 | No |
| `tb3_frontier_exploration` | frontier detection + goal assignment (C++) | No |

Outside `src/`: [`docker/`](docker/) is the image behind the macOS and Windows
setups, [`docs/`](docs/) the setup pages and demo media, [`scripts/`](scripts/)
the acceptance run.

![System overview](docs/media/system_overview.svg)

<sub>Full-resolution PNG: [docs/media/system_overview.png](docs/media/system_overview.png)</sub>

## Setup by platform

| Platform | Start here | What you get |
|---|---|---|
| **Ubuntu 22.04** *(default — the validated setup)* | the steps just below | native ROS 2 Humble + Gazebo Classic 11 |
| **macOS** (Apple Silicon or Intel) | [docs/setup-macos.md](docs/setup-macos.md) | a Docker container with this checkout mounted, Gazebo/RViz in a browser tab, then the steps below |
| **Windows 10/11** | [docs/setup-windows.md](docs/setup-windows.md) | WSL 2 + Ubuntu 22.04 running the same native stack, then the steps below (Docker is a documented fallback) |

You need Ubuntu 22.04 with ROS 2 Humble and Gazebo Classic 11, 4+ CPU cores,
and `tmux` (for `scripts/acceptance_run.sh` only).

**1. apt packages**

```bash
sudo apt install \
  ros-humble-desktop \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-turtlebot3-gazebo \
  ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-slam-toolbox \
  ros-humble-vision-msgs ros-humble-cv-bridge \
  ros-humble-tf2-geometry-msgs \
  ros-humble-rqt-image-view \
  python3-colcon-common-extensions \
  python3-pip \
  tmux
```

**2. pip packages.** Order and pins matter: `ultralytics` pulls in `torch` and
would fetch the ~2 GB CUDA build, so install the CPU build first; `numpy` stays
below 2.0 because Humble's `cv_bridge` is built against NumPy 1.x.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install 'numpy==1.26.4' 'opencv-python==4.9.0.80' 'ultralytics==8.4.31'
```

**3. Detector weights** — nothing to do. `tb3det_yolo26n.pt` (5.4 MB) ships
with the repository and is picked up by the build.

**Clone the repository** into your home directory (the macOS and Windows pages
already do this — skip it if you came from one of them):

```bash
cd ~
git clone https://github.com/YiyuchenHu/turtlebot3-gazebo-navigation-hw.git turtlebot3-gazebo-navigation-course
```

**4. Build.** If you have conda, run `conda deactivate` first.

```bash
cd ~/turtlebot3-gazebo-navigation-course
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Keep `--symlink-install` on **every** build — dropping it later makes edits to
configs and launch files silently stop taking effect.

## Running

In **every** terminal, prepare the environment first:

```bash
cd ~/turtlebot3-gazebo-navigation-course
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
```

Then open the terminals **in order**, waiting for each ready signal before
starting the next:

| # | Command | What it starts | Ready when … |
|---|---|---|---|
| T1 | `ros2 launch tb3_bringup sim.launch.py` | Gazebo (server + GUI) + TurtleBot3 spawn, vendored `GAZEBO_MODEL_PATH` | Console prints `Successfully spawned entity [waffle_pi]` and the Gazebo window shows the room + robot (~5 s; first-ever start can take longer) |
| T2 | `ros2 launch tb3_bringup nav.launch.py` | SLAM Toolbox + Nav2 + RViz | Console prints `[lifecycle_manager_navigation]: Managed nodes are active` (~5–10 s); RViz shows a first gray map patch |
| T3 | `ros2 launch tb3_bringup backend.launch.py` | Course backend: memory, semantic map memory, query, nav adapter, coordinator, warmup + frontier exploration | `CoordinatorNode ready — mode=EXPLORING` immediately; after a ±45° warm-up scan, `frontier exploration enabled` (~10 s) and the robot starts exploring |
| T4 | `ros2 launch tb3_bringup localizer.launch.py` | Localizer (bbox + LiDAR → object position) | `LocalizerNode ready` + `Image width learned: 640 px` (~1 s), then quiet until detections arrive |
| T5 | `ros2 launch tb3_bringup detector.launch.py` | Detector: YOLO inference on the camera image | `Model loaded. Classes: [...]` then `detector_node ready` (~3 s, first inference warms up torch); `ros2 topic echo /detector_node/detections` streams non-empty `detections` once an object is in view; RViz **Detector Debug Image** shows green boxes |
| T6 | *(no launch — the command console)* | Send user commands, watch status | — |

```bash
# T6
ros2 topic pub --once /user_command std_msgs/String "data: 'go to person 0'"
ros2 topic echo /coordinator_node/status
```

**Map first, query later.** Exploration takes 2–8 minutes, and only objects that
are already on the semantic map can be navigated to: `go to X` is a lookup in
that map, not a search for X. Commands are not queued — a reply of
`no active <target> in memory` means the robot has not seen that object yet;
wait until its marker appears in `/semantic_memory_markers` in RViz, then send
the command again. Later commands reuse the same map, so once a landmark is
there it stays reachable. You can restart any single terminal without touching
the others.

## More

- **Something is broken** → [docs/troubleshooting.md](docs/troubleshooting.md)
  (clean restart, the Nav2 spin-in-place wedge, chair/trash-can mix-ups).
- **Build details, other worlds, the one-command launch, and where to change
  things** → [docs/running-reference.md](docs/running-reference.md).
- **Why it is built this way, and the measured results** → [NOTES.md](NOTES.md).

## License

Course material is MIT-licensed (see [LICENSE](LICENSE)). The vendored Gazebo
models are third-party content — see [NOTICE](NOTICE) for provenance and
licensing status.
