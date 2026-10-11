// Exercise the actual selector and pre-dispatch map checks without an action
// server, executor spin, simulator, or navigation command.
#define main goal_assignment_program_main
#include "../src/goal_assignment_node.cpp"
#undef main

#include <iostream>
#include <stdexcept>

namespace tb3_frontier_exploration
{
class GoalAssignmentNodeTestAccess
{
public:
  static void run()
  {
    auto node = std::make_shared<GoalAssignmentNode>();
    const auto require = [](bool condition, const char * message) {
        if (!condition) {throw std::runtime_error(message);}
      };
    auto grid = std::make_shared<nav_msgs::msg::OccupancyGrid>();
    grid->header.frame_id = "map";
    grid->header.stamp = node->now();
    grid->info.width = 20; grid->info.height = 20;
    grid->info.resolution = .1;
    grid->info.origin.position.x = -1; grid->info.origin.position.y = -1;
    grid->info.origin.orientation.w = 1;
    grid->data.assign(400, 0);
    node->latest_map_ = grid;
    node->current_costmap_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(*grid);
    geometry_msgs::msg::PoseArray candidates;
    candidates.header.frame_id = "map";
    candidates.poses.resize(1);
    candidates.poses[0].position.x = .25;
    candidates.poses[0].position.y = .05;
    candidates.poses[0].orientation.w = 1;
    std::string frame = "map";
    const auto select = [&](bool observation) {
        return observation ? node->selectBestFrontier(candidates, .05, .05, frame,
          GoalSelectionPurpose::Observation) :
          node->selectBestFrontier(candidates, .05, .05, frame);
      };

    require(!select(false), "ordinary frontier .2m away remains below the .5m travel gate");
    require(select(true) == 0, "safe observation .2m away is eligible for a new source-relative view");
    candidates.poses[0].position.x = .05;
    require(select(true) == 0 && !select(false),
      "safe same-position observation may turn; same-position ordinary frontier stays rejected");
    candidates.poses[0].position.x = .55;
    require(select(false) == 0, "ordinary frontier exact .5m boundary is unchanged");
    candidates.poses[0].position.x = .25;

    for (const int cost : {-1, 61, 100}) {
      node->current_costmap_->data.assign(400, cost);
      require(!select(true), "observation distance exemption cannot bypass unknown/high/lethal costs");
    }
    node->current_costmap_->data.assign(400, 60);
    require(select(true) == 0, "unchanged maximum allowed cost remains 60");
    node->current_costmap_->data.assign(400, 0);
    node->current_costmap_->header.stamp = node->now() - rclcpp::Duration::from_seconds(4);
    require(!select(true), "observation still requires a fresh dynamic costmap");
    node->current_costmap_->header.stamp = node->now();
    node->current_costmap_->header.frame_id = "odom";
    require(!select(true), "observation still rejects a costmap in the wrong frame");
    node->current_costmap_->header.frame_id = "map";
    node->recordFailedGoal(.25, .05);
    require(!select(true), "nearby observation still respects remembered failed regions");
    node->failed_goals_.clear();
    node->recordSuccessfulVisit(.25, .05);
    require(select(true) == 0, "one visit remains below the existing repeated-visit limit");
    node->recordSuccessfulVisit(.25, .05);
    require(select(true) == 0, "observation uses per-task evidence, not global frontier visits");
    // Move the robot back so ordinary selection is rejected specifically by
    // repeated visits, independently of its minimum-travel gate.
    GoalSelectionRejection reason = GoalSelectionRejection::None;
    require(!node->selectBestFrontier(candidates, -.55, .05, frame,
      GoalSelectionPurpose::Frontier, &reason) && reason == GoalSelectionRejection::Visited,
      "twice-visited safe region is still rejected for ordinary frontier coverage");
    require(node->selectBestFrontier(candidates, -.55, .05, frame,
      GoalSelectionPurpose::Observation, &reason) == 0,
      "the same region remains usable for a bounded targeted observation");
    node->current_costmap_->data.assign(400, 61);
    require(!node->selectBestFrontier(candidates, -.55, .05, frame,
      GoalSelectionPurpose::Observation, &reason) && reason == GoalSelectionRejection::Cost,
      "visited exemption cannot bypass cost safety; exact filter is reported");
    node->current_costmap_->data.assign(400, 0);
    node->recordFailedGoal(.25, .05);
    require(!node->selectBestFrontier(candidates, -.55, .05, frame,
      GoalSelectionPurpose::Observation, &reason) && reason == GoalSelectionRejection::FailedRegion,
      "visited exemption cannot bypass failed navigation regions");
    node->failed_goals_.clear();
    node->visit_history_.clear();

    geometry_msgs::msg::TransformStamped robot;
    robot.header.frame_id = "map"; robot.child_frame_id = "base_link";
    robot.header.stamp = node->now();
    robot.transform.translation.x = .05; robot.transform.translation.y = .05;
    robot.transform.rotation.w = 1;
    require(node->tf_buffer_->setTransform(robot, "unit_test"), "local TF setup");
    int serial = 0;
    const auto rejected_before_dispatch = [&]() {
        ObservationRequest request;
        request.request_id = "selection_test:" + std::to_string(serial++);
        request.track_id = "candidate_0";
        request.frame_id = "map";
        request.stamp_ns = node->now().nanoseconds();
        request.x = .25; request.y = .05; request.yaw = 1.0;
        std::string reason;
        require(node->observations_.offer(request, request.stamp_ns, reason), "proposal setup");
        require(!node->dispatchObservation(.05, .05, "map"), "unsafe map rejects actual observation branch");
        require(!node->observations_.pending && !node->observations_.active &&
          node->state_ == State::IDLE, "rejected proposal never reaches action dispatch");
      };
    grid->data[10*20+12] = -1;
    rejected_before_dispatch();
    grid->data.assign(400, 0);
    for (int row = 0; row < 20; ++row) {grid->data[row*20+11] = 100;}
    rejected_before_dispatch();
    // Unknown start plus missing proof is a readiness wait, not a permanent
    // geometry rejection. The candidate and its dispatch budget remain intact.
    grid->data.assign(400,0);
    grid->data[10*20+10]=-1;
    node->current_costmap_->data[10*20+10]=43;
    ObservationRequest waiting;
    waiting.request_id="footprint_wait"; waiting.track_id="candidate_0";
    waiting.frame_id="map"; waiting.stamp_ns=node->now().nanoseconds();
    waiting.x=.25; waiting.y=.05; waiting.yaw=1;
    std::string wait_reason;
    require(node->observations_.offer(waiting,waiting.stamp_ns,wait_reason),"footprint wait proposal setup");
    require(!node->dispatchObservation(.05,.05,"map") && node->observations_.pending &&
      !node->observations_.active && node->state_==State::IDLE,
      "missing exact footprint proof must wait without retiring proposal or sending an action");
    node->observations_.retire_pending();
    auto footprint=std::make_shared<geometry_msgs::msg::PolygonStamped>();
    footprint->header=robot.header;
    for(const auto & [x,y]:FootprintPolygon{{-.15,-.15},{.25,-.15},{.25,.25},{-.15,.25}}) {
      geometry_msgs::msg::Point32 p;p.x=x;p.y=y;footprint->polygon.points.push_back(p);
    }
    node->footprint_proof_.add(footprint,node->now().nanoseconds());
    // Now connectivity succeeds, but the original failed-region policy still
    // rejects before dispatch; this test never needs an action server/motion.
    node->recordFailedGoal(.25,.05);
    rejected_before_dispatch();
    std::cout << "Actual observation/ordinary selection and map-safety regression passed\n";
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
