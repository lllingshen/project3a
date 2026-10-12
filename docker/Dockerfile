# syntax=docker/dockerfile:1
#
# Course image for macOS / Windows hosts: ROS 2 Humble + Gazebo Classic 11 +
# Nav2 + simulation dependencies, plus a virtual desktop you open in a
# browser (noVNC on port 6080). The repository itself is NOT baked in — the
# compose file bind-mounts your checkout at ~/turtlebot3-gazebo-navigation-course
# inside the container, so the README's Build and Running sections apply
# verbatim.
#
# This image is linux/amd64 only. Gazebo Classic 11 has no arm64 packages
# for Ubuntu 22.04 (neither Ubuntu ports nor OSRF ship them), so ROS Humble's
# ros-humble-gazebo-ros-pkgs / ros-humble-turtlebot3-gazebo exist for amd64
# only. On Apple Silicon the container runs through Docker Desktop's Rosetta
# emulation (docker-compose.yml pins the platform); on Intel Macs and on
# Windows it runs natively.

FROM ros:humble
# = ros:humble-ros-base-jammy — Ubuntu 22.04, Python 3.10, ROS apt repo
#   already configured. Multi-arch upstream; we refuse anything but amd64
#   below so the failure is explicit instead of a mysterious apt error.

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN [ "$(dpkg --print-architecture)" = "amd64" ] || { \
      echo >&2 "ERROR: this image must be built for linux/amd64" \
               "(Gazebo Classic 11 has no arm64 packages on Ubuntu 22.04)." \
               "Use docker compose (platform is pinned there) or pass" \
               "--platform linux/amd64 to docker build."; exit 1; }

# `retry <cmd>`: runs a step up to three times. Needed because Rosetta
# (Apple Silicon) occasionally kills a process with SIGILL mid-install;
# see retry.sh for the details. On a native amd64 host it never triggers.
COPY retry.sh /usr/local/bin/retry
RUN chmod 755 /usr/local/bin/retry

# Keep apt's downloaded .debs and package lists in BuildKit cache mounts (the
# --mount lines on the apt steps below) instead of the image: a rebuild after
# a failed or edited step then skips the ~2.5 GB download. The base image's
# docker-clean hook would delete the .debs again, so it goes.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# ---------------------------------------------------------------------------
# 1. Course dependencies — the exact list validated on Ubuntu 22.04
#    (README → Installation → step 1). Recommends are installed, as a plain
#    `sudo apt install` does on the native setup. Do not add or remove
#    packages here without re-validating on the Ubuntu side.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && retry apt-get install -y \
      ros-humble-desktop \
      ros-humble-gazebo-ros-pkgs \
      ros-humble-turtlebot3-gazebo \
      ros-humble-navigation2 ros-humble-nav2-bringup \
      ros-humble-slam-toolbox \
      ros-humble-vision-msgs ros-humble-cv-bridge \
      ros-humble-tf2-geometry-msgs \
      ros-humble-rqt-image-view \
      ros-humble-nav2-regulated-pure-pursuit-controller \
      nlohmann-json3-dev python3-numpy python3-yaml python3-opencv \
      python3-colcon-common-extensions \
      python3-pip \
      tmux git

# LocateAnything model inference uses a separately provisioned CUDA environment,
# model checkpoint, and Eagle code. This image supplies ROS/simulation only;
# it does not install a detector or claim to provide the model runtime.

# ---------------------------------------------------------------------------
# 3. Browser GUI plumbing — NOT course dependencies; this is what replaces
#    the host's X server on macOS / Windows:
#      tigervnc-standalone-server  Xvnc: an X server with a built-in VNC server
#      novnc + websockify          serve the VNC session to a browser tab
#      openbox                     a minimal window manager (Gazebo/RViz need one)
#      xfonts-base                 the core X fonts Xvnc refuses to start without
#    Rendering is Mesa's llvmpipe (software OpenGL) — there is no GPU here.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && retry apt-get install -y \
      tigervnc-standalone-server \
      novnc websockify \
      openbox \
      xfonts-base

COPY novnc-index.html /usr/share/novnc/index.html
COPY entrypoint.sh /usr/local/bin/tb3-entrypoint.sh
RUN chmod 755 /usr/local/bin/tb3-entrypoint.sh

# Every process in the container renders to the virtual display.
ENV DISPLAY=:1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    QT_X11_NO_MITSHM=1 \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    TURTLEBOT3_MODEL=waffle_pi

# Interactive shells (`docker compose exec tb3 bash`) come up with ROS and,
# once it exists, the workspace overlay sourced — the same lines the README
# puts at the top of every terminal; repeating them by hand is harmless.
RUN cat >> /root/.bashrc <<'BASHRC'

# --- tb3 course (docker/Dockerfile) ---
source /opt/ros/humble/setup.bash
if [ -f "$HOME/turtlebot3-gazebo-navigation-course/install/setup.bash" ]; then
    source "$HOME/turtlebot3-gazebo-navigation-course/install/setup.bash"
fi
export TURTLEBOT3_MODEL=waffle_pi
BASHRC

WORKDIR /root/turtlebot3-gazebo-navigation-course
EXPOSE 6080

# Healthy once noVNC answers, i.e. the virtual desktop is up.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:6080/vnc.html', timeout=2)"

ENTRYPOINT ["/usr/local/bin/tb3-entrypoint.sh"]
CMD ["sleep", "infinity"]
