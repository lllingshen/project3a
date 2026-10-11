#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/time.hpp>

#include <mutex>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "exploration_policy.hpp"
#include "safe_frontier.hpp"
#include "footprint_proof.hpp"
#include "observation_protocol.hpp"
#include "goal_selection_policy.hpp"

/**
 * @file goal_assignment_node.cpp
 * @brief ROS 2 node: assign Nav2 navigation goals from frontier centroids for autonomous exploration.
 *
 * This node is the bridge between **candidate frontier generation** (frontier_detection_node) and **actual
 * autonomous navigation** (Nav2):
 *
 * - Subscribes to the frontier centroid topic (default `/frontiers`) as `geometry_msgs/PoseArray`, produced by
 *   the frontier detection node.
 * - Obtains the robot pose in the **map** frame using **TF** (`frame_id` → `robot_base_frame`, e.g. map → base_link).
 *   Odometry is subscribed for topic compatibility but pose selection uses TF only in the current implementation.
 * - **Filters** frontier candidates: drops goals too close to the robot and optionally avoids a disk around the
 *   last failed/canceled/aborted goal.
 * - **Selects** the best valid frontier using a **nearest-to-robot** policy among remaining candidates.
 * - Sends **`navigate_to_pose`** goals to Nav2 **asynchronously** via `rclcpp_action`.
 * - Handles **goal timeout** (cancel if navigation stalls), **success** (records a visited goal), **cancellation**,
 *   **rejection**, and **abort** (records position for avoidance on the next selection).
 * - **Records failed goals** so `selectBestFrontier` can skip centroids near recent failures.
 *
 * Together with frontier_detection_node, this closes the exploration loop: map → frontiers → Nav2 → motion → updated map.
 */

namespace tb3_frontier_exploration
{

enum class State { IDLE, NAVIGATING };

class GoalAssignmentNode : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandle = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  /**
   * @brief Construct the goal assignment node: parameters, TF, subscriptions, Nav2 action client, and control timer.
   *
   * @param none No constructor arguments; all behavior is configured via ROS parameters.
   * @return N/A (constructs the node in place).
   *
   * Pipeline role:
   * - Connects the exploration executive to `/frontiers` and Nav2’s `navigate_to_pose` action; without this setup,
   *   frontier centroids would never become motion commands.
   *
   * Implementation summary:
   * 1. Declares parameters for frontier and odom topic names, Nav2 action name, map/base frames, timeouts,
   *    selection distances, timer rate, and exploration-complete log throttling.
   * 2. Creates `tf2_ros::Buffer` and `TransformListener` for map-frame robot position.
   * 3. Subscribes to `PoseArray` frontiers and `Odometry` (odom callback is currently a no-op).
   * 4. Creates the `NavigateToPose` action client and a wall timer at `rate` Hz to run `timerCallback`.
   * 5. Logs startup configuration (topics, distances, timeout).
   *
   * Notes:
   * - Initial FSM state is `IDLE`; the timer drives both idle goal dispatch and navigating timeout checks.
   * - Action server must be available (Nav2 bt_navigator) before goals are sent.
   */
  explicit GoalAssignmentNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("goal_assignment_node", options),
    state_(State::IDLE)
  {
    declare_parameter<std::string>("frontiers_topic", "/frontiers");
    declare_parameter<std::string>("odom_topic", "/odometry/filtered");
    declare_parameter<std::string>("navigate_to_pose_action", "navigate_to_pose");
    declare_parameter<bool>("require_nav2_active", false);
    declare_parameter<std::string>("navigation_lifecycle_service",
      "/lifecycle_manager_navigation/is_active");
    declare_parameter<std::string>("frame_id", "map");
    declare_parameter<std::string>("robot_base_frame", "base_link");
    declare_parameter<double>("goal_timeout", 60.0);
    declare_parameter<double>("goal_max_duration_sec", 180.0);
    declare_parameter<double>("goal_progress_distance", 0.10);
    declare_parameter<std::string>("map_topic", "/map");
    declare_parameter<std::string>("footprint_topic", "/global_costmap/published_footprint");
    declare_parameter<std::string>("imu_topic", "/imu");
    declare_parameter<double>("blocked_tilt_deg", 15.0);
    declare_parameter<double>("blocked_tilt_duration_sec", 0.5);
    declare_parameter<double>("rate", 1.0);
    declare_parameter<double>("min_frontier_distance", 0.5);
    declare_parameter<double>("failed_goal_avoidance_radius", 1.0);
    // Failed goals are remembered as a set with local map evidence, not one slot.
    // The single slot was the direct cause of the 40 s limit cycle: a frontier
    // goal failed and was recorded, the fallback goal then failed and
    // OVERWROTE that record, which re-armed the very frontier that had just
    // failed. See NOTES.md 7.8 for the full cycle and the two-part fix.
    declare_parameter<double>("failed_goal_ttl_sec", 90.0);
    declare_parameter<int>("failed_goal_max_entries", 64);
    declare_parameter<double>("exploration_complete_log_interval", 5.0);
    declare_parameter<bool>("require_startup_warmup", false);
    declare_parameter<std::string>("startup_warmup_complete_topic", "exploration_warmup_complete");
    declare_parameter<double>("startup_warmup_timeout_sec", 90.0);
    declare_parameter<bool>("enable_repeated_centroid_filter", true);
    declare_parameter<double>("repeated_centroid_reject_radius", 0.45);
    declare_parameter<int>("repeated_centroid_max_visits", 2);

    // ── Bootstrap-drive fallback parameters ───────────────────────────────
    // When exploration is enabled but no valid frontier passes the filters,
    // sample N points on a circle of radius R around the robot, query the
    // global costmap, and send the lowest-cost reachable one to Nav2 as a
    // regular navigate_to_pose goal. This replaces the old in-place 360°
    // sweep (which provided almost no SLAM benefit because the laser only
    // sees current line-of-sight geometry).
    declare_parameter<bool>("enable_fallback_drive", true);
    declare_parameter<int>("fallback_bootstrap_max_goals", 2);
    declare_parameter<std::string>("fallback_costmap_topic", "/global_costmap/costmap");
    declare_parameter<double>("fallback_radius_m", 1.5);
    declare_parameter<int>("fallback_num_samples", 16);
    // OccupancyGrid cost is 0–100; we reject candidates above this. 60 is
    // well below the inflation lethal threshold but above free space.
    declare_parameter<int>("fallback_max_cost", 60);
    // Minimum gap between consecutive fallback dispatches; avoids re-sending
    // an unreachable goal every timer tick while Nav2 is rejecting it.
    declare_parameter<double>("fallback_cooldown_sec", 4.0);

    // ── Exploration-complete detection ────────────────────────────────────
    // When frontier_detection has reported 0 safe centroids for K consecutive
    // messages AND the robot has already completed at least one navigation
    // goal (i.e. exploration actually started), declare the SLAM scan
    // "complete". This stops the fallback path from generating endless
    // "wander inside the known map" goals (the symptom the operator hit:
    // the robot bouncing between (1.00, 1.44) ↔ (0.94, -0.28) forever
    // because anti-loop's 0.45 m radius is smaller than the fallback radius).
    declare_parameter<bool>("enable_exploration_complete_detection", true);
    declare_parameter<int>("empty_frontier_streak_threshold", 3);
    // Stop when no usable frontier persists this long. Fallback motion does
    // not reset the interval, nor does republishing the same blocked list.
    declare_parameter<double>("stalemate_idle_sec", 20.0);
    // Exploration-complete detection used to be gated on "the robot has
    // reached at least one goal". That gate is closed exactly when the node
    // is deadlocked from the start (visit_history_ empty), so it disabled the
    // escape hatch in the one case that needed it. It is replaced by a
    // minimum-uptime guard: refuse to declare completion until exploration has
    // been ENABLED for this long. Measured from the moment exploration became
    // enabled, not from node construction, so a slow startup warm-up cannot
    // eat the guard interval.
    declare_parameter<double>("min_exploration_time_sec", 60.0);
    // Policy when complete: "stop" = stay idle; "return_home" = drive back
    // to the pose recorded at first successful TF lookup.
    declare_parameter<std::string>("on_exploration_complete", "stop");

    std::string frontiers_topic = get_parameter("frontiers_topic").as_string();
    std::string odom_topic = get_parameter("odom_topic").as_string();
    std::string action_name = get_parameter("navigate_to_pose_action").as_string();
    std::string fallback_costmap_topic =
      get_parameter("fallback_costmap_topic").as_string();

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    frontiers_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      frontiers_topic, 10, std::bind(&GoalAssignmentNode::frontiersCallback, this, std::placeholders::_1));
    observation_event_pub_ = create_publisher<std_msgs::msg::String>(
      "/locateanything/observation_event", rclcpp::QoS(10).reliable());
    observation_sub_ = create_subscription<std_msgs::msg::String>(
      "/locateanything/observation_request", rclcpp::QoS(10).reliable(),
      [this](const std_msgs::msg::String::SharedPtr msg) {observationCallback(msg->data);});
    observation_busy_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/locateanything/observation_busy", rclcpp::QoS(10).reliable(),
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        observation_busy_ = msg->data;
        observation_busy_stamp_ns_ = now().nanoseconds();
      });
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, 10, std::bind(&GoalAssignmentNode::odomCallback, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("imu_topic").as_string(), rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
        if (msg->orientation_covariance[0] < 0.0) {
          inclination_.discard_pending();
          return;
        }
        const auto & q = msg->orientation;
        const double stamp = rclcpp::Time(msg->header.stamp).seconds();
        if (std::abs(now().seconds() - stamp) > 0.5) {
          inclination_.discard_pending();
          return;
        }
        if (!inclination_.observe(stamp, q.x, q.y, q.z, q.w,
          get_parameter("blocked_tilt_deg").as_double() * M_PI / 180.0,
          get_parameter("blocked_tilt_duration_sec").as_double())) {return;}
        // Cancel without recording the destination as failed: the chassis is
        // tilted, so this does not establish anything about that destination.
        cancelObservation("paused", "robot blocked by sustained IMU tilt");
        ++goal_generation_;
        auto handle = current_goal_handle_;
        current_goal_handle_.reset();
        goal_sent_time_.reset();
        no_frontier_since_.reset();
        state_ = State::IDLE;
        if (handle) {nav_client_->async_cancel_goal(handle);}
        RCLCPP_ERROR(get_logger(),
          "[robot_blocked] IMU tilt %.1f deg persisted for %.2fs; exploration locked. "
          "Canceled owned goal without marking its target failed. "
          "Physically level the robot and restart this node before exploring again.",
          inclination_.tilt_rad * 180.0 / M_PI,
          get_parameter("blocked_tilt_duration_sec").as_double());
      });

    // Global costmap is published with TRANSIENT_LOCAL durability by Nav2
    // (so late joiners receive the latest grid). Match that on the
    // subscription to get the first frame immediately on connect.
    auto costmap_qos = rclcpp::QoS(1).transient_local().reliable();
    costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      fallback_costmap_topic, costmap_qos,
      std::bind(&GoalAssignmentNode::costmapCallback, this, std::placeholders::_1));
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      get_parameter("map_topic").as_string(), costmap_qos,
      [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        if (msg->info.width && msg->info.height && msg->info.resolution > 0.0 &&
          msg->data.size() == static_cast<size_t>(msg->info.width) * msg->info.height)
        {latest_map_ = msg;}
      });

    // Nav2 publishes a volatile SystemDefaultsQoS PolygonStamped. Keep a
    // bounded history; its exact-stamp TF may arrive after the polygon.
    footprint_sub_ = create_subscription<geometry_msgs::msg::PolygonStamped>(
      get_parameter("footprint_topic").as_string(), rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::PolygonStamped::ConstSharedPtr msg) {
        footprint_proof_.add(msg, now().nanoseconds());
      });

    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, action_name);
    require_nav2_active_ = get_parameter("require_nav2_active").as_bool();
    if (require_nav2_active_) {
      nav_active_client_ = create_client<std_srvs::srv::Trigger>(
        get_parameter("navigation_lifecycle_service").as_string());
      nav_active_timer_ = create_wall_timer(std::chrono::seconds(1),
        std::bind(&GoalAssignmentNode::pollNavigationActive, this));
    }

    require_startup_warmup_ = get_parameter("require_startup_warmup").as_bool();
    startup_warmup_done_ = !require_startup_warmup_;
    warmup_watch_start_ = now();
    std::string warmup_topic = get_parameter("startup_warmup_complete_topic").as_string();
    warmup_sub_ = create_subscription<std_msgs::msg::Bool>(
      warmup_topic, rclcpp::QoS(10),
      std::bind(&GoalAssignmentNode::warmupCompleteCallback, this, std::placeholders::_1));

    exploration_enabled_sub_ = create_subscription<std_msgs::msg::Bool>(
      "exploration_enabled", rclcpp::QoS(10),
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        exploration_enabled_ = msg->data;
        if (!exploration_enabled_) {
          // The coordinator cannot cancel this client's action handle.
          // Invalidate pending acceptance/results as well as an active goal.
          cancelObservation("paused", "exploration disabled");
          ++goal_generation_;
          auto handle = current_goal_handle_;
          current_goal_handle_.reset();
          goal_sent_time_.reset();
          state_ = State::IDLE;
          no_frontier_since_.reset();
          if (handle) {
            nav_client_->async_cancel_goal(handle);
            RCLCPP_INFO(get_logger(), "[exploration] Canceling owned goal on pause");
          }
        }
        RCLCPP_INFO(get_logger(), "[exploration] %s via /exploration_enabled",
          exploration_enabled_ ? "ENABLED" : "DISABLED");
      });

    double rate_hz = get_parameter("rate").as_double();
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_hz),
      std::bind(&GoalAssignmentNode::timerCallback, this));

    RCLCPP_INFO(get_logger(),
      "[goal_assignment] Started: frontiers=%s, action=%s, min_frontier_dist=%.2f, failed_avoid_radius=%.2f, goal_timeout=%.1fs",
      frontiers_topic.c_str(), action_name.c_str(),
      get_parameter("min_frontier_distance").as_double(),
      get_parameter("failed_goal_avoidance_radius").as_double(),
      get_parameter("goal_timeout").as_double());
    if (get_parameter("enable_fallback_drive").as_bool()) {
      RCLCPP_INFO(get_logger(),
        "[goal_assignment] Fallback drive ENABLED: costmap=%s, radius=%.2fm, samples=%d, max_cost=%d, cooldown=%.1fs",
        fallback_costmap_topic.c_str(),
        get_parameter("fallback_radius_m").as_double(),
        static_cast<int>(get_parameter("fallback_num_samples").as_int()),
        static_cast<int>(get_parameter("fallback_max_cost").as_int()),
        get_parameter("fallback_cooldown_sec").as_double());
    }
    if (get_parameter("enable_exploration_complete_detection").as_bool()) {
      RCLCPP_INFO(get_logger(),
        "[goal_assignment] Exploration-complete detection ENABLED: streak_threshold=%d, stalemate=%.1fs, on_complete=%s",
        static_cast<int>(get_parameter("empty_frontier_streak_threshold").as_int()),
        get_parameter("stalemate_idle_sec").as_double(),
        get_parameter("on_exploration_complete").as_string().c_str());
    }
    if (require_startup_warmup_) {
      RCLCPP_INFO(get_logger(),
        "[goal_assignment] Startup warmup required; waiting for Bool true on '%s' (or timeout %.1fs)",
        warmup_topic.c_str(), get_parameter("startup_warmup_timeout_sec").as_double());
    }
  }

private:
  friend class GoalAssignmentNodeTestAccess;

  // An advertised action can still belong to an inactive lifecycle node.
  // Poll the manager asynchronously and age evidence using wall time, so a
  // paused/missing simulation clock cannot keep an old active result valid.
  using SteadyClock = std::chrono::steady_clock;
  bool navigationLifecycleReady() const
  {
    return !require_nav2_active_ || (nav_active_ && nav_active_stamp_ &&
      SteadyClock::now() - *nav_active_stamp_ < std::chrono::seconds(3) &&
      nav_active_client_->service_is_ready());
  }

  void pollNavigationActive()
  {
    const auto stamp = SteadyClock::now();
    if (nav_active_request_) {
      if (stamp - nav_active_request_->sent < std::chrono::seconds(2)) {return;}
      nav_active_client_->remove_pending_request(nav_active_request_->id);
      nav_active_request_.reset();
      ++nav_active_generation_;
      nav_active_ = false;
      nav_active_stamp_.reset();
    }
    if (!nav_active_client_->service_is_ready()) {
      nav_active_ = false;
      nav_active_stamp_.reset();
      return;
    }
    const auto generation = ++nav_active_generation_;
    try {
      const auto request = nav_active_client_->async_send_request(
        std::make_shared<std_srvs::srv::Trigger::Request>(),
        [this, generation, stamp](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture result) {
          if (generation != nav_active_generation_) {return;}
          nav_active_request_.reset();
          // A late answer must not refresh an expired readiness observation.
          nav_active_ = SteadyClock::now() - stamp < std::chrono::seconds(2) &&
            result.get()->success;
          nav_active_stamp_ = nav_active_ ? std::optional<SteadyClock::time_point>(stamp) :
            std::nullopt;
        });
      nav_active_request_ = NavigationActiveRequest{request.request_id, stamp};
    } catch (const std::exception & error) {
      nav_active_ = false;
      nav_active_stamp_.reset();
      RCLCPP_WARN(get_logger(), "[idle] Nav2 lifecycle check failed: %s", error.what());
    }
  }
  // Forward-declared early so member functions whose signatures use Point
  // (selectFallbackGoal, countSuccessfulVisitsNear, recordFailedGoal, etc.)
  // can reference the type. The actual storage members of type Point live
  // farther down with the rest of the data members.
  struct Point { double x, y; };

  void observationEvent(const ObservationRequest & request, const char * event,
    const std::string & reason = "", const std::string & reason_code = "")
  {
    nlohmann::json payload{{"request_id", request.request_id}, {"track_id", request.track_id},
      {"event", event}};
    // Partial identities recovered from malformed requests have no validated
    // frame/pose. Do not log their default zero coordinates as real evidence.
    payload["stamp_ns"] = std::string(event) == "dispatched" && observation_sent_ns_ ?
      *observation_sent_ns_ : now().nanoseconds();
    if (!request.frame_id.empty()) {
      payload["frame_id"] = request.frame_id;
      payload["request_x"] = request.x;
      payload["request_y"] = request.y;
      payload["request_yaw"] = request.yaw;
      payload["request_stamp_ns"] = request.stamp_ns;
    }
    if (!reason.empty()) {payload["reason"] = reason;}
    if (!reason_code.empty()) {payload["reason_code"] = reason_code;}
    std_msgs::msg::String message;
    message.data = payload.dump();
    observation_event_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "[observation] %s", message.data.c_str());
  }

  void cancelObservation(const char * event, const std::string & reason, bool clear_dwell = true)
  {
    const bool had_active = observations_.active.has_value();
    for (const auto & request : observations_.cancel_all()) {observationEvent(request, event, reason);}
    if (clear_dwell) {observation_dwell_until_ns_.reset();}
    observation_sent_ns_.reset();
    if (had_active) {
      ++goal_generation_;
      auto handle = current_goal_handle_;
      current_goal_handle_.reset();
      goal_sent_time_.reset();
      state_ = State::IDLE;
      if (handle) {nav_client_->async_cancel_goal(handle);}
    }
  }

  void checkObservationClock()
  {
    footprint_proof_.observe_clock(now().nanoseconds());
    if (observations_.clock_rolled_back(now().nanoseconds())) {
      cancelObservation("canceled", "simulation clock rolled back; old observation invalidated");
      observation_busy_stamp_ns_.reset();
      observation_phase_.clock_rollback();
    }
  }

  void observationCallback(const std::string & data)
  {
    checkObservationClock();
    std::string reason;
    const auto parsed = parse_observation_request(data, reason, get_parameter("frame_id").as_string());
    if (!parsed) {
      // A malformed payload cannot navigate; recover safe IDs only to let the
      // sender correlate the rejection. Never use its unvalidated coordinates.
      ObservationRequest identity;
      try {
        const auto j = data.size() <= 4096 ? nlohmann::json::parse(data) : nlohmann::json();
        if (j.is_object()) {
          for (const char * key : {"request_id", "track_id"}) {
            if (j.contains(key) && j[key].is_string()) {
              const auto value = j[key].get<std::string>();
              if (observation_id_valid(value)) {
                (std::string(key) == "request_id" ? identity.request_id : identity.track_id) = value;
              }
            }
          }
        }
      } catch (const std::exception &) {}
      observationEvent(identity, "rejected", "invalid request: " + reason, "protocol");
      return;
    }
    if (parsed->clear()) {cancelObservation("canceled", "proposal withdrawn", false); return;}
    if (!exploration_enabled_ || inclination_.blocked || start_motion_.blocked) {
      observationEvent(*parsed, "paused", "exploration disabled or robot blocked");
      return;
    }
    if (!observations_.offer(*parsed, now().nanoseconds(), reason)) {
      const auto code = reason == "request is expired or future-dated" ? "expired" :
        reason == "another observation is active" ? "busy" : "protocol";
      observationEvent(*parsed, "rejected", reason, code);
    }
  }

  // First dispatch follows geometric exhaustion; an established bounded phase
  // may prioritize later observations. The raw map adds a
  // known-free connectivity check to the shared cost/failed-region filters.
  bool dispatchObservation(double rx, double ry, const std::string & frame_id)
  {
    if (!observations_.pending) {return false;}
    const auto request = *observations_.pending;
    const auto evaluation_time = now();
    const auto reject = [&](const std::string & reason, const char * code = "geometry") {
        observations_.retire_pending();
        observationEvent(request, "rejected", reason, code);
        return false;
      };
    if (observation_phase_.expired(evaluation_time.nanoseconds())) {
      return reject("observation phase reached its fixed 120 second deadline", "budget");
    }
    if (!observation_fresh(request, evaluation_time.nanoseconds())) {return reject("request expired before dispatch", "expired");}
    geometry_msgs::msg::TransformStamped robot;
    try {
      robot = tf_buffer_->lookupTransform(frame_id,
        get_parameter("robot_base_frame").as_string(), tf2::TimePointZero);
      const auto latest_ns = rclcpp::Time(robot.header.stamp).nanoseconds();
      const auto lookup_ns = robot_safety_lookup_stamp(evaluation_time.nanoseconds(), latest_ns);
      if (!lookup_ns) {return false;}
      if (*lookup_ns < latest_ns) {
        robot = tf_buffer_->lookupTransform(frame_id, get_parameter("robot_base_frame").as_string(),
          rclcpp::Time(*lookup_ns, evaluation_time.get_clock_type()));
      }
      const double age = (evaluation_time - rclcpp::Time(robot.header.stamp)).seconds();
      if (age < 0.0 || age > .75 || !std::isfinite(robot.transform.translation.x) ||
        !std::isfinite(robot.transform.translation.y)) {return false;}
      rx = robot.transform.translation.x; ry = robot.transform.translation.y;
    } catch (const std::exception &) {return false;}
    nav_msgs::msg::OccupancyGrid::SharedPtr costs;
    {
      std::lock_guard<std::mutex> lock(costmap_mutex_);
      costs = current_costmap_;
    }
    const auto fresh = [&](const nav_msgs::msg::OccupancyGrid::SharedPtr & grid) {
        if (!grid || grid->header.frame_id != frame_id) {return false;}
        const double age = (evaluation_time - rclcpp::Time(grid->header.stamp)).seconds();
        return age >= 0.0 && age <= 3.0;
      };
    // A static SLAM map remains useful after frontier exhaustion. Require a
    // current dynamic costmap, but do not wait for the robot to move in order
    // to refresh the raw map that is needed to authorize that very movement.
    if (!latest_map_ || latest_map_->header.frame_id != frame_id ||
      rclcpp::Time(latest_map_->header.stamp) > evaluation_time || !fresh(costs))
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "[observation] Waiting for a valid map and fresh costmap; proposal not rejected as unsafe");
      return false;
    }
    const auto grid = [](const nav_msgs::msg::OccupancyGrid & map) {
        const auto & q = map.info.origin.orientation;
        const double yaw = std::atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z));
        return GridView{static_cast<int>(map.info.width), static_cast<int>(map.info.height),
          map.info.resolution, map.info.origin.position.x, map.info.origin.position.y, yaw, map.data};
      };
    const auto map_grid = grid(*latest_map_), cost_grid = grid(*costs);
    if (!map_grid.valid() || !cost_grid.valid() || !std::isfinite(map_grid.x) ||
      !std::isfinite(map_grid.y) || !std::isfinite(map_grid.yaw) || !std::isfinite(map_grid.resolution) ||
      !std::isfinite(cost_grid.x) || !std::isfinite(cost_grid.y) || !std::isfinite(cost_grid.yaw) ||
      !std::isfinite(cost_grid.resolution))
    {return false;}
    if (!observation_grid_contains(map_grid, request.x, request.y) ||
      !observation_grid_contains(cost_grid, request.x, request.y) ||
      map_grid.at(request.x, request.y) != 0)
    {return reject("observation endpoint is not in a known-free cell within map/costmap bounds");}
    std::optional<VerifiedFootprint> footprint;
    if (map_grid.at(rx,ry)==-1 && traversable_cost(cost_grid.at(rx,ry))) {
      footprint=footprint_proof_.current(frame_id,
        get_parameter("robot_base_frame").as_string(), evaluation_time.nanoseconds(), robot,
        [&](int64_t stamp) {
          return tf_buffer_->lookupTransform(frame_id, get_parameter("robot_base_frame").as_string(),
            rclcpp::Time(stamp,evaluation_time.get_clock_type()), rclcpp::Duration::from_seconds(0));
        });
    }
    if (map_grid.at(rx,ry)==-1 && traversable_cost(cost_grid.at(rx,ry)) && !footprint) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "[observation] Waiting for verified current footprint at unknown start; proposal not rejected as unsafe");
      return false;
    }
    const auto reachable = reachable_free_cells(map_grid, cost_grid, rx, ry,
      footprint ? &footprint->polygon : nullptr);
    const auto [col, row] = map_grid.cell(request.x, request.y);
    if (reachable.size() != map_grid.data.size() || !reachable[map_grid.index(col, row)]) {
      return reject("observation endpoint has no known-free connection to robot");
    }
    if (footprint) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
        "[observation] Verified footprint start bridge used: footprint_stamp_ns=%ld robot=(%.4f,%.4f)",
        static_cast<long>(footprint->source_stamp_ns), rx, ry);
    }
    geometry_msgs::msg::PoseArray candidates;
    candidates.header.frame_id = frame_id;
    geometry_msgs::msg::Pose pose;
    pose.position.x = request.x; pose.position.y = request.y;
    pose.orientation.z = std::sin(request.yaw/2); pose.orientation.w = std::cos(request.yaw/2);
    candidates.poses.push_back(pose);
    std::string selected_frame = frame_id;
    // Observation independence is relative to the stored source view. The
    // current robot may already be close to that safe new viewpoint and only
    // need to turn. Frontier minimum travel/global visits measure geometric
    // coverage; observation retries are bounded by the separate per-task
    // attempted positions and dispatch budgets in the observation proposer.
    GoalSelectionRejection selection_rejection = GoalSelectionRejection::None;
    if (!selectBestFrontier(candidates, rx, ry, selected_frame,
      GoalSelectionPurpose::Observation, &selection_rejection))
    {
      return reject(std::string("observation endpoint rejected: ") +
        goal_selection_rejection_reason(selection_rejection));
    }
    if (observation_phase_.expired(now().nanoseconds())) {
      return reject("observation phase expired during safety evaluation", "budget");
    }
    const auto started = observations_.start(now().nanoseconds());
    if (!started) {return reject("request became stale before send", "expired");}
    // Do not erase geometric progress/stop bookkeeping; these visits improve
    // semantic evidence, not unknown-space coverage.
    dispatchNavGoal(request.x, request.y, frame_id, "observation", request.yaw, started);
    return true;
  }

  void warmupCompleteCallback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (msg->data) {
      startup_warmup_done_ = true;
      RCLCPP_INFO(get_logger(), "[warmup] Received complete signal — frontier exploration enabled.");
    }
  }

  /**
   * @brief Store the latest frontier centroid message from frontier_detection_node under a mutex.
   *
   * @param msg Shared pointer to `geometry_msgs/PoseArray` (each pose’s position is a frontier centroid in map frame).
   * @return void; updates member `latest_frontiers_`.
   *
   * Pipeline role:
   * - Provides the **input queue** of exploration candidates for the timer-driven executive; without fresh messages,
   *   the node has nothing to navigate toward.
   *
   * Implementation summary:
   * 1. Lock `frontiers_mutex_`.
   * 2. Assign `latest_frontiers_ = msg` (shared pointer copy, replaces previous snapshot).
   * 3. Optionally log centroid count at DEBUG level.
   *
   * Notes:
   * - Each callback **replaces** the entire cached array when frontier_detection republishes on map update.
   * - Thread-safe with `timerCallback`, which reads the same pointer under the same mutex.
   */
  void frontiersCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(frontiers_mutex_);
      latest_frontiers_ = msg;
      frontiers_received_ = now();
    }
    RCLCPP_DEBUG(get_logger(), "[frontiers] Received %zu centroids", msg->poses.size());

    // Update the streak counter used by exploration-complete detection.
    // frontier_detection publishes its safe centroids on this topic; an
    // empty array means "no safe frontier in the current map". After K
    // consecutive empty messages we assume the SLAM scan has converged.
    if (msg->poses.empty()) {
      ++empty_frontier_streak_;
    } else {
      empty_frontier_streak_ = 0;
      // The timer resumes only for a selectable frontier. A republished
      // nonempty list of the same blocked points must not re-arm exploration.
    }
  }

  /**
   * @brief Odometry subscription callback (currently unused for pose or control).
   *
   * @param msg Latest `nav_msgs/Odometry` on `odom_topic` (ignored in the body).
   * @return void.
   *
   * Pipeline role:
   * - Reserved hook for future use (e.g. velocity checks or fallback pose); **does not** participate in frontier
   *   selection today—robot position comes from TF in `getRobotPoseInMap`.
   *
   * Implementation summary:
   * 1. Cast `msg` to void to silence unused-parameter warnings.
   *
   * Notes:
   * - Keeping the subscription allows the node to match launch files that expect an odom topic without extra nodes.
   * - Do not assume pose is updated here when reading or extending the code.
   */
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    (void)msg;
  }

  /**
   * @brief Look up the robot base frame position projected into the map plane using TF.
   *
   * @param rx Output: robot x in `frame_id` (map) frame, meters.
   * @param ry Output: robot y in `frame_id` (map) frame, meters.
   * @return `true` if `lookupTransform(frame_id, robot_base_frame, ...)` succeeds; `false` on TF failure (outputs undefined).
   *
   * Pipeline role:
   * - Enables **distance-based** frontier filtering and **nearest-neighbor** selection; without a valid transform,
   *   the timer callback aborts before sending a goal.
   *
   * Implementation summary:
   * 1. Read `frame_id` and `robot_base_frame` from parameters.
   * 2. Call `tf_buffer_->lookupTransform` with latest time (`TimePointZero`) and 0.5 s timeout.
   * 3. On success, set `rx`, `ry` from `transform.translation` (x, y); on exception, log at DEBUG and return false.
   *
   * Notes:
   * - Only **translation** is used; yaw is not needed for Euclidean distance in the plane.
   * - Requires TF tree from `robot_base_frame` to `frame_id` to be published (e.g. map → odom → base_link).
   */
  bool getRobotPoseInMap(double & rx, double & ry)
  {
    std::string frame_id = get_parameter("frame_id").as_string();
    std::string robot_frame = get_parameter("robot_base_frame").as_string();
    try {
      geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform(
        frame_id, robot_frame, tf2::TimePointZero, tf2::durationFromSec(0.5));
      rx = t.transform.translation.x;
      ry = t.transform.translation.y;
      // Memorize the very first successful pose as "home", used by the
      // return_home policy when exploration is complete. We use the first
      // valid TF rather than the launch-arg x_pose/y_pose because the
      // start frame coincides with map origin only after SLAM stabilizes.
      if (!start_pose_) {
        start_pose_ = Point{rx, ry};
        RCLCPP_INFO(get_logger(),
          "[start] Recorded start pose (%.2f, %.2f) — used as 'home' on exploration complete",
          rx, ry);
      }
      return true;
    } catch (const std::exception & e) {
      // Throttled to 1/5s so a stalled TF tree does not flood the log,
      // but the operator still sees the silent failure that previously
      // caused goal_assignment to look "frozen" (timer keeps running, no
      // [goal] Selected ever appears).
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "[TF] Lookup %s -> %s failed (no goal will be sent until TF recovers): %s",
        frame_id.c_str(), robot_frame.c_str(), e.what());
      return false;
    }
  }

  /**
   * @brief Choose the nearest valid frontier index for the robot, applying distance and failure-avoidance filters.
   *
   * @param frontiers `PoseArray` of candidate centroids (positions in map frame; header may supply `frame_id`).
   * @param rx Robot x in map frame (meters), from TF.
   * @param ry Robot y in map frame (meters), from TF.
   * @param frame_id In/out: goal frame string; if empty on input, set from `frontiers.header.frame_id` when a candidate exists.
   * @param purpose Ordinary frontier coverage or a bounded targeted observation.
   * @param rejection Optional exact selection filter for a single-candidate request.
   * @return Index of the selected pose in `frontiers.poses`, or `std::nullopt` if every candidate is filtered out.
   *
   * Pipeline role:
   * - Implements the exploration **policy** (greedy nearest valid frontier); bridges perception output to a single
   *   actionable goal per timer cycle.
   *
   * Implementation summary:
   * 1. Load `min_frontier_distance` and `failed_goal_avoidance_radius` from parameters (squared for comparisons).
   * 2. For each frontier pose, compute squared distance to `(rx, ry)`; skip if closer than `min_frontier_distance`.
   * 3. Skip centroids within `failed_goal_avoidance_radius` of ANY unexpired entry in `failed_goals_`.
   * 4. Among remaining, track the index with minimum squared distance.
   * 5. If none remain, return nullopt; else if `frame_id` was empty, copy `frontiers.header.frame_id`, then return index.
   *
   * Notes:
   * - **Nearest** is Euclidean in the map **xy** plane; z is ignored.
   * - Ties are broken by **first** minimum encountered (order is array order).
   */
  std::optional<size_t> selectBestFrontier(
    const geometry_msgs::msg::PoseArray & frontiers,
    double rx, double ry,
    std::string & frame_id,
    GoalSelectionPurpose purpose = GoalSelectionPurpose::Frontier,
    GoalSelectionRejection * rejection = nullptr)
  {
    const double min_dist = get_parameter("min_frontier_distance").as_double();
    const double min_dist_sq = min_dist * min_dist;
    const double avoid_radius = get_parameter("failed_goal_avoidance_radius").as_double();
    const double avoid_radius_sq = avoid_radius * avoid_radius;
    if (rejection) {*rejection = GoalSelectionRejection::None;}

    size_t best = 0;
    double best_dist_sq = 1e30;
    bool found = false;

    for (size_t i = 0; i < frontiers.poses.size(); i++) {
      double gx = frontiers.poses[i].position.x;
      double gy = frontiers.poses[i].position.y;
      double dx = gx - rx;
      double dy = gy - ry;
      double d2 = dx * dx + dy * dy;

      // Revalidate against the latest costmap; a formerly safe observation
      // position can become obstructed after its frontier message was sent.
      const int current_cost = static_cast<int>(getCostAtWorld(gx, gy));
      // Preserve the ordinary selector's short-circuit order, including when
      // its failed-region bookkeeping is consulted/pruned.
      const bool reaches_safety = current_cost >= 0 && current_cost <= 60 &&
        !(purpose == GoalSelectionPurpose::Frontier && d2 < min_dist_sq);
      const bool failed_region = reaches_safety && isNearFailedGoal(gx, gy, avoid_radius_sq);
      bool repeated_visit = false;
      if (reaches_safety && !failed_region &&
        get_parameter("enable_repeated_centroid_filter").as_bool())
      {
        const double loop_r = get_parameter("repeated_centroid_reject_radius").as_double();
        const int max_vis = get_parameter("repeated_centroid_max_visits").as_int();
        if (loop_r > 0.0 && max_vis > 0) {
          repeated_visit = static_cast<int>(countSuccessfulVisitsNear(gx, gy, loop_r * loop_r)) >= max_vis;
        }
      }
      const auto filtered = goal_selection_rejection(
        purpose, current_cost, d2 < min_dist_sq, failed_region, repeated_visit);
      if (filtered != GoalSelectionRejection::None) {
        if (rejection) {*rejection = filtered;}
        RCLCPP_DEBUG(get_logger(), "[select] Candidate [%zu] (%.2f, %.2f) skipped: %s",
          i, gx, gy, goal_selection_rejection_reason(filtered));
        continue;
      }

      if (d2 < best_dist_sq) {
        best_dist_sq = d2;
        best = i;
        found = true;
      }
    }

    if (!found) return std::nullopt;
    if (frame_id.empty()) frame_id = frontiers.header.frame_id;
    return best;
  }

  /**
   * @brief Periodic executive: enforce Nav2 goal timeout when navigating; when idle, select and send the next frontier goal.
   *
   * @param none.
   * @return void; may call `async_cancel_goal` or `async_send_goal` as side effects.
   *
   * Pipeline role:
   * - **Central loop** for closed-loop exploration: supervises the current Nav2 goal and dispatches the next one
   *   when safe; without this timer, no goals would be issued after startup.
   *
   * Implementation summary:
   * 1. While navigating, observe progress and cancel only for a stall or the absolute duration limit.
   * 2. Wait for startup warmup, Nav2, and a valid robot pose before choosing another goal.
   * 3. Prefer a usable frontier; this is the only dispatch that clears the no-frontier interval.
   * 4. If none exists, evaluate the stop condition before considering a finite startup fallback.
   *
   * Notes:
   * - State is set to `NAVIGATING` **before** acceptance; if the server **rejects**, the response callback resets to `IDLE`.
   * - Timeout uses time **after** acceptance (`goal_sent_time_` set in response callback), not wall time from send call.
   * - Cancel path copies handle before reset so `async_cancel_goal` still receives a valid handle.
   */
  void timerCallback()
  {
    checkObservationClock();
    if (observations_.pending && !observation_fresh(*observations_.pending, now().nanoseconds())) {
      const auto expired = observations_.retire_pending();
      observationEvent(*expired, "rejected", "request expired before dispatch", "expired");
    }
    if (inclination_.blocked || start_motion_.blocked) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
        "[robot_blocked] Exploration remains locked after a robot-state failure; this is not exploration complete.");
      return;
    }
    if (!exploration_enabled_) {
      return;
    }
    if (observation_dwell_until_ns_) {
      if (now().nanoseconds() < *observation_dwell_until_ns_) {return;}
      observation_dwell_until_ns_.reset();
    }
    const double goal_timeout = get_parameter("goal_timeout").as_double();
    double robot_x, robot_y;
    if (getRobotPoseInMap(robot_x, robot_y)) {
      const int start_cost = static_cast<int>(getCostAtWorld(robot_x, robot_y));
      if (start_motion_.observe_cost(now().seconds(), start_cost)) {
        lockRobotBlocked("robot start remains inscribed/lethal in the fresh costmap for 2 seconds");
        return;
      }
      // Hold dispatch/completion during the debounce rather than entering
      // another destination while the robot's own footprint is in collision.
      if (start_cost >= 99) {no_frontier_since_.reset(); return;}
      if (state_ == State::IDLE && start_cost < 0) {
        no_frontier_since_.reset();
        empty_frontier_streak_ = 0;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "[idle] Waiting for known robot start in a fresh costmap; not exploration complete");
        return;
      }
    } else {
      start_motion_.observe_cost(now().seconds(), -1);
      no_frontier_since_.reset();
      empty_frontier_streak_ = 0;
    }


    if (state_ == State::NAVIGATING) {
      // Bound a send whose acceptance never returns; a late acceptance is
      // invalidated by generation and canceled using its own immutable ID.
      if (observations_.active && !current_goal_handle_ && observation_sent_ns_ &&
        now().nanoseconds() - *observation_sent_ns_ >= 5000000000LL)
      {
        cancelObservation("timeout", "Nav2 observation acceptance did not arrive within 5 seconds");
        return;
      }
      if (current_goal_handle_ && goal_sent_time_) {
        double rx, ry;
        const double seconds = now().seconds();
        if (getRobotPoseInMap(rx, ry)) {
          start_motion_.observe(rx, ry);
          progress_.observe(seconds, rx, ry, std::numeric_limits<double>::quiet_NaN(),
            get_parameter("goal_progress_distance").as_double());
        }
        const bool total_limit = progress_.total_expired(seconds,
          get_parameter("goal_max_duration_sec").as_double());
        if (total_limit || progress_.stalled(seconds, goal_timeout)) {
          RCLCPP_WARN(get_logger(),
            "[timeout] %s: elapsed=%.1fs, no_progress=%.1fs; canceling",
            total_limit ? "absolute navigation limit" : "navigation stalled",
            seconds - progress_.started, seconds - progress_.last_progress);
          auto handle = current_goal_handle_;
          if (observations_.active) {
            const auto request = *observations_.active;
            observationEvent(request, "timeout", total_limit ? "absolute navigation limit" : "navigation stalled");
            observations_.finish(request.request_id);
            observation_sent_ns_.reset();
          }
          ++goal_generation_;
          current_goal_handle_.reset();
          goal_sent_time_.reset();
          recordNavigationFailure(seconds - progress_.started);
          state_ = State::IDLE;
          if (handle) {
            nav_client_->async_cancel_goal(handle);
          }
        }
      }
      return;
    }

    if (require_startup_warmup_ && !startup_warmup_done_) {
      const double tout = get_parameter("startup_warmup_timeout_sec").as_double();
      if ((now() - warmup_watch_start_).seconds() >= tout) {
        if (!warmup_timeout_announced_) {
          RCLCPP_WARN(get_logger(),
            "[warmup] No complete signal after %.1fs — enabling exploration anyway.", tout);
          warmup_timeout_announced_ = true;
        }
        startup_warmup_done_ = true;
      } else {
        if (!last_warmup_waiting_log_ ||
          (now() - *last_warmup_waiting_log_).seconds() >= 5.0)
        {
          RCLCPP_INFO(get_logger(), "[warmup] Waiting for startup map warmup…");
          last_warmup_waiting_log_ = now();
        }
        return;
      }
    }

    if (!nav_client_->action_server_is_ready()) {
      // Throttled WARN (was DEBUG): if Nav2 lifecycle never finishes
      // bringup or bt_navigator deactivates, goal_assignment would silently
      // skip every cycle. Make this visible so the operator can correlate
      // with lifecycle_manager logs.
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "[idle] Nav2 navigate_to_pose action server not ready — no goal will be sent");
      return;
    }
    if (!navigationLifecycleReady()) {
      no_frontier_since_.reset();
      empty_frontier_streak_ = 0;
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
        "[idle] Waiting for Nav2 lifecycle manager to confirm all navigation nodes active");
      return;
    }

    // Exploration is genuinely running from here on: enabled, warm-up done,
    // Nav2 answering. This is the zero point for min_exploration_time_sec.
    if (!exploration_start_time_) {
      exploration_start_time_ = now();
    }

    double rx, ry;
    if (!getRobotPoseInMap(rx, ry)) return;

    std::string frame_id = get_parameter("frame_id").as_string();

    // ── Path 1: prefer a real frontier centroid ─────────────────────────
    geometry_msgs::msg::PoseArray::SharedPtr frontiers;
    {
      std::lock_guard<std::mutex> lock(frontiers_mutex_);
      frontiers = latest_frontiers_;
    }

    if (!frontiers || !frontiers_received_ || !fresh_frontier_input(now().seconds(),
      frontiers_received_->seconds(), rclcpp::Time(frontiers->header.stamp).seconds(),
      frontiers->header.frame_id == frame_id))
    {
      no_frontier_since_.reset();
      empty_frontier_streak_ = 0;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "[idle] Waiting for fresh frontier safety evaluation in map frame; not exploration complete");
      return;
    }

    // Once an actual observation starts, its existing continuous time budget
    // also covers proposal/catalog gaps. Do not spend that bounded window on
    // ordinary frontier travel. No goal is preempted: NAVIGATING, warmup,
    // safety, fresh frontier input and the success dwell were checked above.
    // Missing/false/stale busy or the fixed deadline restores normal priority.
    if (observation_phase_.reserve(exploration_enabled_, observation_busy_,
      observation_busy_stamp_ns_, now().nanoseconds()))
    {
      if (dispatchObservation(rx, ry, frame_id)) {return;}
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
        "[observation] Reserving bounded semantic phase while waiting for a safe proposal");
      return;
    }

    if (frontiers && !frontiers->poses.empty()) {
      std::optional<size_t> best_opt = selectBestFrontier(*frontiers, rx, ry, frame_id);
      if (best_opt) {
        if (exploration_complete_) {
          RCLCPP_INFO(get_logger(), "[exploration] A usable frontier appeared — resuming.");
          exploration_complete_ = false;
        }
        no_frontier_since_.reset();
        bootstrap_.frontier_started = true;
        size_t best = *best_opt;
        double gx = frontiers->poses[best].position.x;
        double gy = frontiers->poses[best].position.y;
        double dist = std::hypot(gx - rx, gy - ry);

        RCLCPP_INFO(get_logger(), "[goal] Selected [%zu] (%.2f, %.2f) dist=%.2f m",
          best, gx, gy, dist);

        const auto & q = frontiers->poses[best].orientation;
        const double norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
        const double yaw = std::isfinite(norm) && norm > 1e-12 ?
          std::atan2(2*(q.w*q.z + q.x*q.y), norm-2*(q.y*q.y+q.z*q.z)) : 0.0;

        // Remove the chosen centroid from the cached array so a stale
        // republish does not immediately re-select the same point.
        {
          std::lock_guard<std::mutex> lock(frontiers_mutex_);
          if (latest_frontiers_ && best < latest_frontiers_->poses.size()) {
            latest_frontiers_->poses.erase(latest_frontiers_->poses.begin() + best);
          }
        }

        dispatchNavGoal(gx, gy, frame_id, "frontier", yaw);
        return;
      }
      RCLCPP_DEBUG(get_logger(),
        "[select] No valid frontier after filters (min_dist=%.2f, avoid_radius=%.2f)",
        get_parameter("min_frontier_distance").as_double(),
        get_parameter("failed_goal_avoidance_radius").as_double());
    }

    if (!no_frontier_since_) {no_frontier_since_ = now();}
    const double no_frontier_sec = (now() - *no_frontier_since_).seconds();
    const double min_expl = get_parameter("min_exploration_time_sec").as_double();
    const double exploration_age = exploration_start_time_ ?
      (now() - *exploration_start_time_).seconds() : 0.0;
    const bool empty_done = empty_frontier_streak_ >=
      get_parameter("empty_frontier_streak_threshold").as_int();
    const bool geometry_done = no_frontier_stop(
      get_parameter("enable_exploration_complete_detection").as_bool(),
      exploration_age, min_expl, empty_frontier_streak_,
      get_parameter("empty_frontier_streak_threshold").as_int(), no_frontier_sec,
      get_parameter("stalemate_idle_sec").as_double());
    // A transient empty frontier list during startup is not permission to
    // spend the semantic verification budget. First meet the original
    // geometric-completion conditions, preserving their unchanged limits.
    // The ordinary eligible branch above already dispatched and returned.
    if (observation_dispatch_allowed(observations_.pending.has_value(), false,
      geometry_done, exploration_complete_) && dispatchObservation(rx, ry, frame_id)) {return;}
    const bool semantic_busy = observations_.pending.has_value() || observation_busy_fresh(
      observation_busy_, observation_busy_stamp_ns_, now().nanoseconds());
    if (semantic_busy) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
        "[observation] Semantic verification busy; geometric frontier exhaustion is not final completion");
    }

    if (exploration_complete_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000,
        "[complete] Idle — no usable frontier. Resume requires a usable new frontier.");
      return;
    }
    // Evaluate BEFORE fallback. Moving through known free space is not evidence
    // of exploration progress, and cannot reset the no-frontier interval.
    if (!semantic_busy && geometry_done)
    {
      if (start_motion_.stationary_failures > 0) {
        lockRobotBlocked("no usable frontier after an unresolved failure to leave the start pose");
        return;
      }
      RCLCPP_INFO(get_logger(),
        "[complete-trigger] %s — empty_streak=%d, no_usable_frontier=%.1fs",
        empty_done ? "empty /frontiers streak" : "all frontier candidates blocked or already visited",
        empty_frontier_streak_, no_frontier_sec);
      declareExplorationComplete(frame_id);
      return;
    }

    // ── Path 2: finite bootstrap fallback before normal frontiers start ──
    // Replaces the legacy in-place sweep. The point is for the robot to
    // actually translate so SLAM grows the map, after which Path 1 will
    // naturally take over again.
    if (get_parameter("enable_fallback_drive").as_bool() &&
      bootstrap_.available(get_parameter("fallback_bootstrap_max_goals").as_int()))
    {
      const double cooldown = get_parameter("fallback_cooldown_sec").as_double();
      const bool cooldown_ok = !last_fallback_send_time_ ||
        (now() - *last_fallback_send_time_).seconds() >= cooldown;
      if (cooldown_ok) {
        if (auto fb = selectFallbackGoal(rx, ry); fb) {
          last_fallback_send_time_ = now();
          ++bootstrap_.used;
          dispatchNavGoal(fb->x, fb->y, frame_id, "fallback");
          return;
        }
      }
    }

    logExplorationCompleteIfDue();
  }

  /**
   * @brief One-shot transition into "exploration complete" mode.
   *
   * Sets `exploration_complete_` so the timerCallback short-circuit fires
   * on subsequent ticks, and executes the policy from
   * `on_exploration_complete`:
   *  - "stop"        → just log a clear banner; the robot stays idle.
   *  - "return_home" → dispatch one Nav2 goal back to the recorded start
   *                    pose. After it finishes (success or fail) the robot
   *                    stays idle until new frontiers appear.
   *
   * Idempotent: subsequent calls do nothing while `exploration_complete_`
   * is already true.
   */
  void declareExplorationComplete(const std::string & frame_id)
  {
    if (exploration_complete_ || start_motion_.blocked || inclination_.blocked) return;
    exploration_complete_ = true;

    const std::string policy =
      get_parameter("on_exploration_complete").as_string();

    RCLCPP_INFO(get_logger(),
      "★ EXPLORATION COMPLETE — %d consecutive empty /frontiers messages, "
      "%zu goals reached. Policy=%s.",
      empty_frontier_streak_, visit_history_.size(), policy.c_str());

    if (policy == "return_home") {
      if (start_pose_) {
        RCLCPP_INFO(get_logger(),
          "[home] Returning to start pose (%.2f, %.2f)",
          start_pose_->x, start_pose_->y);
        dispatchNavGoal(start_pose_->x, start_pose_->y, frame_id, "home");
      } else {
        RCLCPP_WARN(get_logger(),
          "[home] return_home requested but no start pose was ever recorded "
          "(TF lookup never succeeded?) — staying idle here.");
      }
    } else {
      RCLCPP_INFO(get_logger(),
        "[stop] Robot will stay idle. To return to start automatically, set "
        "on_exploration_complete: \"return_home\" in params.yaml. To force "
        "re-exploration publish Bool true on /exploration_enabled when "
        "frontiers are available again.");
    }
  }

  /**
   * @brief Build a NavigateToPose goal at (gx, gy) with the requested planar orientation
   *        and dispatch it asynchronously, wiring response/result callbacks.
   *
   * Used by both the frontier path and the fallback-drive path so the FSM
   * transitions, timeout bookkeeping, and failure recording stay identical
   * regardless of where the goal came from. `source` is a short tag
   * ("frontier" / "fallback") for log readability.
   */
  void dispatchNavGoal(double gx, double gy, const std::string & frame_id, const char * source,
    double goal_yaw = 0.0, std::optional<ObservationRequest> observation = std::nullopt)
  {
    if (!exploration_enabled_ || inclination_.blocked || start_motion_.blocked) {
      if (observation) {cancelObservation("paused", "exploration disabled or robot blocked before send");}
      return;
    }
    // Recheck at the actual send boundary, after potentially expensive map
    // selection. Do not blacklist a valid target while Nav2 is starting.
    if (require_nav2_active_ &&
      (!navigationLifecycleReady() || !nav_client_->action_server_is_ready()))
    {
      if (observation) {cancelObservation("paused", "Nav2 lifecycle not ready before send");}
      return;
    }

    NavigateToPose::Goal goal;
    goal.pose.header.stamp = now();
    goal.pose.header.frame_id = frame_id;
    goal.pose.pose.position.x = gx;
    goal.pose.pose.position.y = gy;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation.w = std::cos(goal_yaw/2.0);
    goal.pose.pose.orientation.x = 0.0;
    goal.pose.pose.orientation.y = 0.0;
    goal.pose.pose.orientation.z = std::sin(goal_yaw/2.0);

    current_goal_x_ = gx;
    current_goal_y_ = gy;

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    const std::string source_tag(source);
    const auto generation = ++goal_generation_;
    send_opts.goal_response_callback = [this, source_tag, generation, observation](typename GoalHandle::SharedPtr gh) {
      if (generation != goal_generation_ || !exploration_enabled_) {
        if (gh) {
          nav_client_->async_cancel_goal(gh);
          RCLCPP_INFO(get_logger(), "[exploration] Canceled late goal acceptance after pause");
          if (observation) {observationEvent(*observation, "canceled", "late acceptance for invalidated observation");}
        }
        return;
      }
      if (gh) {
        RCLCPP_INFO(get_logger(), "[action] Goal accepted (%s)", source_tag.c_str());
        current_goal_handle_ = gh;
        goal_sent_time_ = now();
        progress_.start(now().seconds());
        double rx = std::numeric_limits<double>::quiet_NaN(), ry = rx;
        getRobotPoseInMap(rx, ry);
        start_motion_.start(rx, ry);
        state_ = State::NAVIGATING;
        if (observation) {observationEvent(*observation, "accepted");}
      } else {
        RCLCPP_WARN(get_logger(), "[action] Goal rejected (%s)", source_tag.c_str());
        recordFailedGoal(current_goal_x_, current_goal_y_);
        state_ = State::IDLE;
        if (observation) {
          observations_.finish(observation->request_id);
          observation_sent_ns_.reset();
          observationEvent(*observation, "rejected", "Nav2 rejected the observation goal", "nav2");
        }
      }
    };
    send_opts.feedback_callback = [this, generation](typename GoalHandle::SharedPtr,
      const std::shared_ptr<const NavigateToPose::Feedback> feedback) {
        if (generation != goal_generation_ || !exploration_enabled_ ||
          state_ != State::NAVIGATING || !goal_sent_time_) {return;}
        const auto & p = feedback->current_pose.pose.position;
        progress_.observe(now().seconds(), p.x, p.y, feedback->distance_remaining,
          get_parameter("goal_progress_distance").as_double());
      };
    send_opts.result_callback = [this, source_tag, generation, observation](const GoalHandle::WrappedResult & result) {
      if (generation != goal_generation_) return;
      current_goal_handle_.reset();
      goal_sent_time_.reset();
      state_ = State::IDLE;
      if (observation) {
        observations_.finish(observation->request_id);
        observation_sent_ns_.reset();
      }

      switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
          RCLCPP_INFO(get_logger(), "[result] SUCCEEDED (%s)", source_tag.c_str());
          // Deliberately NOT clearing failed_goals_ here -- see recordFailedGoal.
          start_motion_.stationary_failures = 0;
          recordSuccessfulVisit(current_goal_x_, current_goal_y_);
          if (observation) {
            observation_dwell_until_ns_ = now().nanoseconds() + 5000000000LL;
            observationEvent(*observation, "succeeded");
          }
          break;
        case rclcpp_action::ResultCode::ABORTED:
          RCLCPP_WARN(get_logger(), "[result] ABORTED (%s)", source_tag.c_str());
          recordNavigationFailure(now().seconds() - progress_.started);
          if (observation) {observationEvent(*observation, "aborted");}
          break;
        case rclcpp_action::ResultCode::CANCELED:
          RCLCPP_WARN(get_logger(), "[result] CANCELED (%s)", source_tag.c_str());
          recordNavigationFailure(now().seconds() - progress_.started);
          if (observation) {observationEvent(*observation, "canceled");}
          break;
        default:
          RCLCPP_WARN(get_logger(), "[result] Unknown code (%s)", source_tag.c_str());
          recordFailedGoal(current_goal_x_, current_goal_y_);
          if (observation) {observationEvent(*observation, "aborted", "unknown Nav2 result code");}
      }
    };

    state_ = State::NAVIGATING;
    try {
      nav_client_->async_send_goal(goal, send_opts);
      if (observation) {
        observation_sent_ns_ = now().nanoseconds();
        observation_phase_.dispatched(*observation_sent_ns_);
        observationEvent(*observation, "dispatched");
      }
    } catch (const std::exception & e) {
      if (!observation) {throw;}
      ++goal_generation_;
      observations_.finish(observation->request_id);
      observation_sent_ns_.reset();
      state_ = State::IDLE;
      observationEvent(*observation, "rejected", std::string("Nav2 send failed: ") + e.what(), "nav2");
    }
  }

  /**
   * @brief Cache the latest global costmap snapshot for fallback sampling.
   *
   * Subscribes with TRANSIENT_LOCAL durability (matching Nav2's default) so
   * the very first frame received already reflects the current obstacle
   * picture. Drops zero-sized grids defensively.
   */
  void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    if (msg->info.width == 0 || msg->info.height == 0) return;
    std::lock_guard<std::mutex> lock(costmap_mutex_);
    current_costmap_ = msg;
  }

  /**
   * @brief Look up the OccupancyGrid cost (0–100) at a world-frame point.
   *
   * @return -1 if costmap is missing or the point falls outside the grid;
   *         otherwise the 8-bit signed cell value (-1 unknown, 0 free, 100 lethal).
   *
   * Used by `selectFallbackGoal` to pick the lowest-cost direction without
   * having to subscribe to the local planner's internal data.
   */
  int8_t getCostAtWorld(double wx, double wy)
  {
    std::lock_guard<std::mutex> lock(costmap_mutex_);
    if (!current_costmap_) {return -1;}
    const auto & cm = *current_costmap_;
    if (cm.header.frame_id != get_parameter("frame_id").as_string() ||
      std::abs((now() - rclcpp::Time(cm.header.stamp)).seconds()) > 3.0) {return -1;}
    const auto & q = cm.info.origin.orientation;
    const double yaw = std::atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z));
    const GridView grid{static_cast<int>(cm.info.width), static_cast<int>(cm.info.height),
      cm.info.resolution, cm.info.origin.position.x, cm.info.origin.position.y, yaw, cm.data};
    return static_cast<int8_t>(grid.at(wx, wy));
  }

  void lockRobotBlocked(const char * reason)
  {
    start_motion_.blocked = true;
    cancelObservation("paused", std::string("robot blocked: ") + reason);
    ++goal_generation_;
    auto handle = current_goal_handle_;
    current_goal_handle_.reset();
    goal_sent_time_.reset();
    no_frontier_since_.reset();
    state_ = State::IDLE;
    if (handle) {nav_client_->async_cancel_goal(handle);}
    RCLCPP_ERROR(get_logger(),
      "[robot_blocked] %s; exploration locked, not COMPLETE. "
      "No remote target failure recorded. Check robot clearance before restarting this node.", reason);
  }

  void recordNavigationFailure(double elapsed)
  {
    if (start_motion_.no_initial_motion(elapsed)) {
      if (start_motion_.failed(elapsed)) {
        lockRobotBlocked("two consecutive navigation attempts had less than 0.10 m initial translation");
      } else {
        RCLCPP_WARN(get_logger(),
          "[start_stall] No initial translation; allow one retry without blacklisting the remote target");
      }
      return;
    }
    start_motion_.failed(elapsed);
    recordFailedGoal(current_goal_x_, current_goal_y_);
  }

  /**
   * @brief Pick a low-cost point on a circle of radius R around the robot.
   *
   * @return The chosen point in map frame, or `nullopt` if no sample qualifies.
   *
   * Strategy:
   *  1. Sample N evenly spaced angles (parameter `fallback_num_samples`).
   *  2. For each, compute the world-frame point at `(rx + R cos θ, ry + R sin θ)`
   *     (`R = fallback_radius_m`).
   *  3. Read its cost from the cached global costmap.
   *  4. Skip cells outside the costmap (cost == -1) and unknown / lethal cells
   *     (cost > `fallback_max_cost`).
   *  5. Honor the same anti-loop and last-failed-goal filters used for
   *     frontier centroids so the bot does not bounce back to a known-bad
   *     spot.
   *  6. Return the candidate with the smallest cost; ties are broken by
   *     iteration order, which is fine because the next tick will resample.
   *
   * If the global costmap has not arrived yet (e.g. very early after launch)
   * this function returns `nullopt`, letting the caller log
   * "exploration complete" instead of busy-spinning.
   */
  std::optional<Point> selectFallbackGoal(double rx, double ry)
  {
    {
      std::lock_guard<std::mutex> lock(costmap_mutex_);
      if (!current_costmap_) {
        // No costmap yet — caller will throttle the "no valid frontier" log.
        return std::nullopt;
      }
    }

    const double radius = get_parameter("fallback_radius_m").as_double();
    const int n_samples = static_cast<int>(get_parameter("fallback_num_samples").as_int());
    const int max_cost = static_cast<int>(get_parameter("fallback_max_cost").as_int());
    if (radius <= 0.0 || n_samples <= 0) return std::nullopt;

    const double avoid_r = get_parameter("failed_goal_avoidance_radius").as_double();
    const double avoid_r_sq = avoid_r * avoid_r;
    const bool antiloop_on = get_parameter("enable_repeated_centroid_filter").as_bool();
    const double loop_r = get_parameter("repeated_centroid_reject_radius").as_double();
    const int max_visits = static_cast<int>(get_parameter("repeated_centroid_max_visits").as_int());
    const double loop_r_sq = loop_r * loop_r;

    std::optional<Point> best;
    int best_cost = std::numeric_limits<int>::max();

    for (int i = 0; i < n_samples; ++i) {
      const double theta = (2.0 * M_PI * i) / static_cast<double>(n_samples);
      const double gx = rx + radius * std::cos(theta);
      const double gy = ry + radius * std::sin(theta);

      const int cost = static_cast<int>(getCostAtWorld(gx, gy));
      // -1 = unknown / out-of-bounds → skip.
      if (cost < 0 || cost > max_cost) continue;

      // Avoid the discs around all unexpired failed goals (frontier AND
      // fallback), so a failing fallback point is not re-offered either.
      if (isNearFailedGoal(gx, gy, avoid_r_sq)) continue;
      // Anti-loop: skip points whose neighborhood we have already visited too often.
      if (antiloop_on && loop_r > 0.0 && max_visits > 0 &&
          static_cast<int>(countSuccessfulVisitsNear(gx, gy, loop_r_sq)) >= max_visits)
      {
        continue;
      }

      if (cost < best_cost) {
        best_cost = cost;
        best = Point{gx, gy};
      }
    }

    if (best) {
      RCLCPP_INFO(get_logger(),
        "[fallback] Drive to (%.2f, %.2f)  cost=%d  radius=%.2fm  samples=%d",
        best->x, best->y, best_cost, radius, n_samples);
    } else {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
        "[fallback] No low-cost sample found around robot (radius=%.2fm, max_cost=%d)",
        radius, max_cost);
    }
    return best;
  }

  /**
   * @brief Remember a world-frame goal position that failed, timed out, or was rejected so selection can avoid it.
   *
   * @param gx Goal x in map frame (meters), same as sent to Nav2.
   * @param gy Goal y in map frame (meters).
   * @return void; appends to (or refreshes an entry in) `failed_goals_`.
   *
   * Pipeline role:
   * - Provides **recovery**: stops selection from handing back a goal that has
   *   just failed, until the retry cooldown and local map-change requirements are met.
   *
   * Implementation summary:
   * 1. Drop entries whose cooldown elapsed AND whose local map changed.
   * 2. If an entry already covers (gx, gy), refresh its timestamp instead of
   *    appending, so one repeatedly-failing spot cannot fill the set.
   * 3. Otherwise append; evict oldest beyond `failed_goal_max_entries`.
   *
   * Notes:
   * - frontier and fallback goals share this set and CANNOT overwrite each
   *   other. That mutual overwrite was the deadlock.
   * - Entries are NOT cleared on SUCCEEDED. Clearing on success would re-arm a
   *   chronically failing frontier as soon as any fallback goal happened to
   *   succeed, which is the same bug through a different door. A retry requires
   *   both the cooldown and changed local occupancy evidence.
   */
  void recordFailedGoal(double gx, double gy)
  {
    pruneFailedGoals();
    const double radius = get_parameter("failed_goal_avoidance_radius").as_double();
    const double r_sq = radius * radius;
    for (auto & f : failed_goals_) {
      const double dx = gx - f.x, dy = gy - f.y;
      if (dx * dx + dy * dy < r_sq) {
        f.stamp = now();
        f.map_patch = mapPatch(f.x, f.y);
        RCLCPP_DEBUG(get_logger(),
          "[failed] Refreshed (%.2f, %.2f); set=%zu", f.x, f.y, failed_goals_.size());
        return;
      }
    }
    failed_goals_.push_back({gx, gy, now(), mapPatch(gx, gy)});
    const size_t cap = static_cast<size_t>(
      std::max<int64_t>(1, get_parameter("failed_goal_max_entries").as_int()));
    while (failed_goals_.size() > cap) {
      failed_goals_.erase(failed_goals_.begin());
    }
    RCLCPP_DEBUG(get_logger(), "[failed] Recorded (%.2f, %.2f) for avoidance; set=%zu",
      gx, gy, failed_goals_.size());
  }

  /** Sample a 1.6 m square at fixed world coordinates around a failed goal. */
  std::vector<int8_t> mapPatch(double gx, double gy) const
  {
    if (!latest_map_) {return {};}
    const auto & map = *latest_map_;
    const auto & q = map.info.origin.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    const double c = std::cos(yaw), s = std::sin(yaw);
    std::vector<int8_t> patch;
    patch.reserve(81);
    for (int y = -4; y <= 4; ++y) {
      for (int x = -4; x <= 4; ++x) {
        const double dx = gx + 0.2 * x - map.info.origin.position.x;
        const double dy = gy + 0.2 * y - map.info.origin.position.y;
        const int mx = static_cast<int>(std::floor((c * dx + s * dy) / map.info.resolution));
        const int my = static_cast<int>(std::floor((-s * dx + c * dy) / map.info.resolution));
        if (mx < 0 || my < 0 || mx >= static_cast<int>(map.info.width) ||
          my >= static_cast<int>(map.info.height))
        {patch.push_back(-1);} else {
          patch.push_back(occupancy_class(map.data[static_cast<size_t>(my) * map.info.width + mx]));
        }
      }
    }
    return patch;
  }

  /** Retry only after cooldown AND substantial local occupancy change. */
  void pruneFailedGoals()
  {
    const double cooldown = std::max(0.0, get_parameter("failed_goal_ttl_sec").as_double());
    const auto t = now();
    failed_goals_.erase(
      std::remove_if(failed_goals_.begin(), failed_goals_.end(),
        [&](const FailedGoal & f) {
          const double age = (t - f.stamp).seconds();
          return retry_failed_region(age, cooldown, f.map_patch, mapPatch(f.x, f.y));
        }),
      failed_goals_.end());
  }

  /** True when (gx, gy) lies within `radius_sq` of any unexpired failure. */
  bool isNearFailedGoal(double gx, double gy, double radius_sq)
  {
    pruneFailedGoals();
    for (const auto & f : failed_goals_) {
      const double dx = gx - f.x, dy = gy - f.y;
      if (dx * dx + dy * dy < radius_sq) return true;
    }
    return false;
  }

  /** Count how many successful frontier goals landed within radius (map plane) of (mx, my). */
  size_t countSuccessfulVisitsNear(double mx, double my, double radius_sq) const
  {
    size_t n = 0;
    for (const auto & p : visit_history_) {
      const double dx = mx - p.x;
      const double dy = my - p.y;
      if (dx * dx + dy * dy <= radius_sq) {
        ++n;
      }
    }
    return n;
  }

  void recordSuccessfulVisit(double gx, double gy)
  {
    const size_t k_max_history = 256;
    visit_history_.push_back({gx, gy});
    while (visit_history_.size() > k_max_history) {
      visit_history_.erase(visit_history_.begin());
    }
    RCLCPP_DEBUG(get_logger(), "[antiloop] Recorded reached centroid (%.2f, %.2f), history=%zu",
      gx, gy, visit_history_.size());
  }

  /**
   * @brief Log a throttled INFO message when no valid frontiers are available (exploration appears complete).
   *
   * @param none Uses `exploration_complete_log_interval` parameter and `last_exploration_complete_log_time_`.
   * @return void.
   *
   * Pipeline role:
   * - Gives operators a clear **status** when the robot is idle due to empty or filtered frontier lists, without
   *   flooding logs on every timer tick.
   *
   * Implementation summary:
   * 1. Read `exploration_complete_log_interval` (seconds).
   * 2. If no prior log time, or `now` minus last log ≥ interval, emit INFO and store `now` as last log time.
   *
   * Notes:
   * - Exploration may **resume** automatically when frontier_detection publishes new centroids; the message states this.
   */
  void logExplorationCompleteIfDue()
  {
    double interval = get_parameter("exploration_complete_log_interval").as_double();
    auto now_t = now();
    if (!last_exploration_complete_log_time_ ||
        (now_t - *last_exploration_complete_log_time_).seconds() >= interval) {
      RCLCPP_INFO(get_logger(), "[exploration] No usable frontier yet — waiting for fresh candidates or the stop condition");
      last_exploration_complete_log_time_ = now_t;
    }
  }

  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr frontiers_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr observation_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr observation_busy_sub_;
  ObservationPhase observation_phase_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr observation_event_pub_;
  ObservationProtocol observations_;
  std::optional<int64_t> observation_sent_ns_, observation_dwell_until_ns_, observation_busy_stamp_ns_;
  bool observation_busy_{false};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr warmup_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr exploration_enabled_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr footprint_sub_;
  FootprintProofCache footprint_proof_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  nav_msgs::msg::OccupancyGrid::SharedPtr latest_map_;
  bool exploration_enabled_{true};
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  bool require_nav2_active_{false}, nav_active_{false};
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr nav_active_client_;
  rclcpp::TimerBase::SharedPtr nav_active_timer_;
  struct NavigationActiveRequest {int64_t id; SteadyClock::time_point sent;};
  std::optional<NavigationActiveRequest> nav_active_request_;
  std::optional<SteadyClock::time_point> nav_active_stamp_;
  uint64_t nav_active_generation_{0};
  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  geometry_msgs::msg::PoseArray::SharedPtr latest_frontiers_;
  std::optional<rclcpp::Time> frontiers_received_;
  std::mutex frontiers_mutex_;

  // Cached global costmap snapshot used by the fallback drive sampler.
  // Updated in costmapCallback; read in getCostAtWorld / selectFallbackGoal.
  nav_msgs::msg::OccupancyGrid::SharedPtr current_costmap_;
  std::mutex costmap_mutex_;

  State state_;
  typename GoalHandle::SharedPtr current_goal_handle_;
  uint64_t goal_generation_{0};
  std::optional<rclcpp::Time> goal_sent_time_;
  ProgressWatchdog progress_;
  StartMotionGuard start_motion_;
  InclinationGuard inclination_;
  BootstrapBudget bootstrap_;
  double current_goal_x_{0.0};
  double current_goal_y_{0.0};

  // Point is forward-declared at the top of the private section.
  // Time passing alone cannot make the same unchanged blocked region useful.
  struct FailedGoal {
    double x, y;
    rclcpp::Time stamp;
    std::vector<int8_t> map_patch;
  };
  std::vector<FailedGoal> failed_goals_;
  // Set when exploration_enabled_ first turns true; the min-uptime guard for
  // exploration-complete detection is measured from here.
  std::optional<rclcpp::Time> exploration_start_time_;
  std::optional<rclcpp::Time> last_exploration_complete_log_time_;
  // Throttle for fallback dispatches; prevents re-sending the same point
  // every timer tick while Nav2 is rejecting/aborting it.
  std::optional<rclcpp::Time> last_fallback_send_time_;

  // First successful map-frame robot pose; used as the "home" target by
  // the return_home policy on exploration complete.
  std::optional<Point> start_pose_;
  // Counter of consecutive `/frontiers` messages with zero poses; updated
  // in frontiersCallback. When this crosses
  // `empty_frontier_streak_threshold` after the minimum exploration period,
  // timerCallback can declare that no more usable goals are available.
  int empty_frontier_streak_{0};
  // Fallback motion does not renew this interval. Only a usable frontier does.
  std::optional<rclcpp::Time> no_frontier_since_;
  // Resumed only when selection finds a usable frontier, not just a nonempty list.
  bool exploration_complete_{false};

  std::vector<Point> visit_history_;

  bool require_startup_warmup_{true};
  bool startup_warmup_done_{false};
  rclcpp::Time warmup_watch_start_;
  bool warmup_timeout_announced_{false};
  std::optional<rclcpp::Time> last_warmup_waiting_log_;
};

}  // namespace tb3_frontier_exploration

/**
 * @brief Program entry: initialize ROS 2 and spin one `GoalAssignmentNode` until shutdown.
 *
 * @param argc Standard C argument count.
 * @param argv Standard C argument vector (ROS 2 remapping may apply).
 * @return 0 after `rclcpp::shutdown`.
 *
 * Pipeline role:
 * - Starts the goal-assignment executable so the exploration stack can receive Nav2 action results and frontier updates.
 *
 * Implementation summary:
 * 1. `rclcpp::init(argc, argv)`.
 * 2. `rclcpp::spin` on a shared `GoalAssignmentNode`.
 * 3. `rclcpp::shutdown()`; return 0.
 *
 * Notes:
 * - Single-threaded executor: timer, subscriptions, and action callbacks are serialized unless a multi-threaded spinner is used elsewhere.
 */
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<tb3_frontier_exploration::GoalAssignmentNode>());
  rclcpp::shutdown();
  return 0;
}
