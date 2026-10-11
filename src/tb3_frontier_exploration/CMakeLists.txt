cmake_minimum_required(VERSION 3.8)
project(tb3_frontier_exploration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(nlohmann_json REQUIRED)

# Frontier detection node
add_executable(frontier_detection_node src/frontier_detection_node.cpp)
target_compile_features(frontier_detection_node PRIVATE cxx_std_17)
ament_target_dependencies(frontier_detection_node
  rclcpp nav_msgs geometry_msgs visualization_msgs std_msgs tf2 tf2_ros)
target_link_libraries(frontier_detection_node nlohmann_json::nlohmann_json)

# Goal assignment node
add_executable(goal_assignment_node src/goal_assignment_node.cpp)
target_compile_features(goal_assignment_node PRIVATE cxx_std_17)
ament_target_dependencies(goal_assignment_node
  rclcpp nav_msgs sensor_msgs geometry_msgs nav2_msgs tf2 tf2_ros std_msgs std_srvs)
target_link_libraries(goal_assignment_node nlohmann_json::nlohmann_json)

# Install
install(TARGETS
  frontier_detection_node
  goal_assignment_node
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/startup_map_warmup_node.py
  DESTINATION lib/${PROJECT_NAME})

# Only params.yaml lives here now. The Gazebo worlds and the vendored models
# they reference moved to tb3_bringup, next to the launch files that load them.
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  add_executable(test_footprint_bridge test/test_footprint_bridge.cpp)
  target_compile_features(test_footprint_bridge PRIVATE cxx_std_17)
  add_test(NAME footprint_bridge COMMAND test_footprint_bridge)
  add_executable(test_footprint_proof test/test_footprint_proof.cpp)
  target_compile_features(test_footprint_proof PRIVATE cxx_std_17)
  ament_target_dependencies(test_footprint_proof geometry_msgs tf2)
  add_test(NAME footprint_proof COMMAND test_footprint_proof)
  add_executable(test_goal_selection_policy test/test_goal_selection_policy.cpp)
  target_compile_features(test_goal_selection_policy PRIVATE cxx_std_17)
  add_test(NAME goal_selection_policy COMMAND test_goal_selection_policy)
  add_executable(test_exploration_policy test/test_exploration_policy.cpp)
  target_compile_features(test_exploration_policy PRIVATE cxx_std_17)
  add_test(NAME exploration_policy COMMAND test_exploration_policy)
  add_executable(test_safe_frontier test/test_safe_frontier.cpp)
  target_compile_features(test_safe_frontier PRIVATE cxx_std_17)
  add_test(NAME safe_frontier COMMAND test_safe_frontier)
  add_executable(test_observation_protocol test/test_observation_protocol.cpp)
  target_compile_features(test_observation_protocol PRIVATE cxx_std_17)
  target_link_libraries(test_observation_protocol nlohmann_json::nlohmann_json)
  add_test(NAME observation_protocol COMMAND test_observation_protocol)
  add_executable(test_observation_selection test/test_observation_selection.cpp)
  target_compile_features(test_observation_selection PRIVATE cxx_std_17)
  ament_target_dependencies(test_observation_selection
    rclcpp nav_msgs sensor_msgs geometry_msgs nav2_msgs tf2 tf2_ros std_msgs std_srvs)
  target_link_libraries(test_observation_selection nlohmann_json::nlohmann_json)
  add_test(NAME observation_selection COMMAND test_observation_selection)
  set_tests_properties(observation_selection PROPERTIES
    ENVIRONMENT "ROS_DOMAIN_ID=179;ROS_LOCALHOST_ONLY=1" TIMEOUT 20)
  add_executable(test_navigation_readiness test/test_navigation_readiness.cpp)
  target_compile_features(test_navigation_readiness PRIVATE cxx_std_17)
  ament_target_dependencies(test_navigation_readiness
    rclcpp nav_msgs sensor_msgs geometry_msgs nav2_msgs tf2 tf2_ros std_msgs std_srvs)
  target_link_libraries(test_navigation_readiness nlohmann_json::nlohmann_json)
  add_test(NAME navigation_readiness COMMAND test_navigation_readiness)
  set_tests_properties(navigation_readiness PROPERTIES
    ENVIRONMENT "ROS_DOMAIN_ID=180;ROS_LOCALHOST_ONLY=1" TIMEOUT 20)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
