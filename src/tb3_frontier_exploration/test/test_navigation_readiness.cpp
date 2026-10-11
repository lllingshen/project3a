// Real ROS callbacks on an isolated domain; no simulator or physical robot.
#define main goal_assignment_program_main
#include "../src/goal_assignment_node.cpp"
#undef main

#include <iostream>
#include <stdexcept>
#include <thread>

namespace tb3_frontier_exploration
{
class GoalAssignmentNodeTestAccess
{
public:
  static void run()
  {
    const auto require = [](bool ok, const char * message) {
        if (!ok) {throw std::runtime_error(message);}
      };
    rclcpp::NodeOptions options;
    options.parameter_overrides({
      rclcpp::Parameter("require_nav2_active", true),
      rclcpp::Parameter("navigation_lifecycle_service", "/test_navigation/is_active"),
      rclcpp::Parameter("navigate_to_pose_action", "/test_navigation/go")});
    auto node = std::make_shared<GoalAssignmentNode>(options);
    node->timer_->cancel();
    node->nav_active_timer_->cancel();
    require(!node->navigationLifecycleReady(), "startup has no active evidence");
    const auto before = std::chrono::steady_clock::now();
    node->pollNavigationActive();
    require(std::chrono::steady_clock::now() - before < std::chrono::milliseconds(100),
      "missing service poll must not block the executor");
    require(!node->nav_active_request_, "missing service must not allocate a pending request");

    auto mock = std::make_shared<rclcpp::Node>("navigation_readiness_mock");
    bool active = false;
    auto service = mock->create_service<std_srvs::srv::Trigger>("/test_navigation/is_active",
      [&active](std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response) {response->success = active;});
    int received_goals = 0;
    using Nav = GoalAssignmentNode::NavigateToPose;
    using ServerHandle = rclcpp_action::ServerGoalHandle<Nav>;
    auto action = rclcpp_action::create_server<Nav>(mock, "/test_navigation/go",
      [&received_goals](const rclcpp_action::GoalUUID &, std::shared_ptr<const Nav::Goal>) {
        ++received_goals;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [](std::shared_ptr<ServerHandle>) {return rclcpp_action::CancelResponse::ACCEPT;},
      [](std::shared_ptr<ServerHandle> handle) {handle->succeed(std::make_shared<Nav::Result>());});
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.add_node(mock);
    const auto spin_until = [&](auto condition) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!condition() && std::chrono::steady_clock::now() < deadline) {
          executor.spin_some();
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        require(condition(), "mock ROS callback/discovery timed out");
      };
    spin_until([&]() {return node->nav_active_client_->service_is_ready() &&
        node->nav_client_->action_server_is_ready();});
    node->pollNavigationActive();
    spin_until([&]() {return !node->nav_active_request_;});
    require(!node->navigationLifecycleReady(), "inactive manager must block advertised action");
    node->dispatchNavGoal(1, 0, "map", "test");
    executor.spin_some();
    require(received_goals == 0 && node->failed_goals_.empty() && node->state_ == State::IDLE,
      "inactive startup must neither dispatch nor blacklist a valid frontier");

    active = true;
    node->pollNavigationActive();
    spin_until([&]() {return !node->nav_active_request_;});
    require(node->navigationLifecycleReady(), "fresh active callback enables navigation");
    node->dispatchNavGoal(1, 0, "map", "test");
    spin_until([&]() {return received_goals == 1 && node->state_ == State::IDLE;});
    require(node->failed_goals_.empty(), "successful ready dispatch does not create a failed region");
    node->nav_active_stamp_ = std::chrono::steady_clock::now() - std::chrono::seconds(4);
    require(!node->navigationLifecycleReady(), "stale active evidence must expire");
    node->dispatchNavGoal(2, 0, "map", "test");
    executor.spin_some();
    require(received_goals == 1 && node->state_ == State::IDLE,
      "stale active evidence cannot dispatch another goal");

    // Leave the mock executor idle so this request receives no response.
    node->pollNavigationActive();
    require(bool(node->nav_active_request_), "pending request setup");
    const auto old_id = node->nav_active_request_->id;
    const auto old_generation = node->nav_active_generation_;
    node->pollNavigationActive();
    require(node->nav_active_request_->id == old_id,
      "poll must retain at most one in-flight request");
    node->nav_active_request_->sent -= std::chrono::seconds(3);
    node->pollNavigationActive();
    require(node->nav_active_generation_ > old_generation && node->nav_active_request_ &&
      node->nav_active_request_->id != old_id && !node->navigationLifecycleReady(),
      "timed-out request is invalidated and replaced without preserving active status");
    require(!node->nav_active_client_->remove_pending_request(old_id),
      "timed-out request must already be removed from rclcpp's pending-request map");
    require(node->nav_active_client_->prune_pending_requests() == 1,
      "only the replacement request remains; timeouts cannot leak requests");
    node->nav_active_request_.reset();
    ++node->nav_active_generation_;
    executor.spin_some();
    require(!node->navigationLifecycleReady(), "retired request replies cannot restore active state");

    auto baseline = std::make_shared<GoalAssignmentNode>();
    baseline->timer_->cancel();
    require(!baseline->nav_active_client_ && baseline->navigationLifecycleReady(),
      "default baseline retains existing behavior without a new service dependency");
    std::cout << "Navigation lifecycle callbacks, expiry, nonblocking timeout and dispatch checks passed\n";
  }
};
}  // namespace tb3_frontier_exploration

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    tb3_frontier_exploration::GoalAssignmentNodeTestAccess::run();
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
