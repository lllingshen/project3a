#!/usr/bin/env bash
# Source this in each terminal. Reuses the existing Python 3.10 model packages.
PROJECT3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
if [[ -f "$PROJECT3_ROOT/install/setup.bash" ]]; then
    source "$PROJECT3_ROOT/install/setup.bash"
fi
export LOCATEANYTHING_PYTHON="${LOCATEANYTHING_PYTHON:-/home/lingshen/miniforge3/envs/locateanything3b/bin/python}"
export LOCATEANYTHING_MODEL="${LOCATEANYTHING_MODEL:-/home/lingshen/research/locateanything_standalone/models/LocateAnything-3B}"
export LOCATEANYTHING_EAGLE="${LOCATEANYTHING_EAGLE:-/home/lingshen/research/locateanything_standalone/Eagle}"
PROJECT3_MODEL_PACKAGES="$(dirname "$(dirname "$LOCATEANYTHING_PYTHON")")/lib/python3.10/site-packages"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT3_MODEL_PACKAGES"
export PATH="/usr/bin:/bin:$PATH"
export TURTLEBOT3_MODEL=waffle_pi
# Separate this assignment's ROS graph from other local projects.
export ROS_DOMAIN_ID="${PROJECT3_ROS_DOMAIN_ID:-49}"
export ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI="http://127.0.0.1:${PROJECT3_GAZEBO_PORT:-11349}"
export ROS_LOG_DIR="$PROJECT3_ROOT/.runtime/ros_logs"
export GAZEBO_MODEL_DATABASE_URI=""
mkdir -p "$ROS_LOG_DIR"
cd "$PROJECT3_ROOT"
