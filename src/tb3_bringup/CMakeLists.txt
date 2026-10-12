cmake_minimum_required(VERSION 3.8)
project(tb3_bringup)

find_package(ament_cmake REQUIRED)

# Launch/configuration assets. Every directory listed here must
# exist in the source tree — CMake's install(DIRECTORY) fails the build when a
# listed directory is missing, and git does not track empty directories.
#
#   launch  every launch file in the course
#   worlds  the Gazebo .world files
#   models  vendored Gazebo models the .world files reference via model:// URIs
#   rviz    the RViz config nav.launch.py opens
install(DIRECTORY
  launch
  worlds
  models
  rviz
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS scripts/velocity_relay.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
