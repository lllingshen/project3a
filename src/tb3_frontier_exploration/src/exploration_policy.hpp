#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace tb3_frontier_exploration
{
// A fallback is an initial attempt to expose frontiers, not a second patrol
// policy after ordinary frontier exploration has run out of candidates.
struct BootstrapBudget
{
  int used{0};
  bool frontier_started{false};
  bool available(int limit) const {return !frontier_started && used < limit;}
};

// A tilted chassis invalidates planar navigation. This is a robot-state fault,
// not evidence that each subsequently selected remote frontier is unreachable.
struct InclinationGuard
{
  bool blocked{false};
  double tilt_rad{0.0};
  std::optional<double> unsafe_since, last_stamp;

  void discard_pending() {unsafe_since.reset(); last_stamp.reset();}

  bool observe(double stamp, double x, double y, double z, double w,
    double limit_rad, double duration, double maximum_sample_gap = 0.25)
  {
    if (blocked) {return false;}  // Latched until the node is restarted.
    const double norm = x*x + y*y + z*z + w*w;
    if (!std::isfinite(stamp) || !std::isfinite(norm) || norm < 1e-12) {
      discard_pending();
      return false;
    }
    if (last_stamp && (stamp < *last_stamp || stamp - *last_stamp > maximum_sample_gap)) {
      unsafe_since.reset();
    }
    last_stamp = stamp;
    tilt_rad = std::acos(std::clamp(1.0 - 2.0 * (x*x + y*y) / norm, -1.0, 1.0));
    if (tilt_rad < limit_rad) {
      unsafe_since.reset();
      return false;
    }
    if (!unsafe_since) {unsafe_since = stamp;}
    if (stamp - *unsafe_since + 1e-9 < duration) {return false;}
    blocked = true;
    return true;
  }
};

struct ProgressWatchdog
{
  double started{0.0}, last_progress{0.0};
  double anchor_x{0.0}, anchor_y{0.0};
  double best_distance{std::numeric_limits<double>::infinity()};
  bool have_pose{false};

  void start(double now)
  {
    started = last_progress = now;
    best_distance = std::numeric_limits<double>::infinity();
    have_pose = false;
  }

  void observe(double now, double x, double y, double distance, double minimum_progress)
  {
    if (now < started) {start(now);}
    if (std::isfinite(x) && std::isfinite(y)) {
      if (!have_pose || std::hypot(x - anchor_x, y - anchor_y) >= minimum_progress) {
        anchor_x = x;
        anchor_y = y;
        have_pose = true;
        last_progress = now;
      }
    }
    if (std::isfinite(distance) && distance >= 0.0 &&
      distance <= best_distance - minimum_progress)
    {
      best_distance = distance;
      last_progress = now;
    }
  }

  bool stalled(double now, double timeout) const
  {return timeout > 0.0 && now >= last_progress && now - last_progress >= timeout;}
  bool total_expired(double now, double timeout) const
  {return timeout > 0.0 && now >= started && now - started >= timeout;}
};

// Remaining-path replans are not physical translation. Keep this independent
// of ProgressWatchdog so repeated failures to leave one start pose become a
// robot-state fault rather than evidence against several distant frontiers.
struct StartMotionGuard
{
  bool have_start{false}, blocked{false};
  double start_x{0}, start_y{0}, maximum_translation{0};
  double failed_x{0}, failed_y{0};
  int stationary_failures{0};
  std::optional<double> inscribed_since;

  void start(double x, double y)
  {
    have_start = std::isfinite(x) && std::isfinite(y);
    start_x = x; start_y = y; maximum_translation = 0;
  }
  void observe(double x, double y)
  {
    if (!std::isfinite(x) || !std::isfinite(y)) {return;}
    if (!have_start) {start(x, y); return;}
    maximum_translation = std::max(maximum_translation, std::hypot(x-start_x, y-start_y));
    if (maximum_translation >= .10) {stationary_failures = 0;}
  }
  bool no_initial_motion(double elapsed) const
  {return have_start && elapsed >= 5.0 && maximum_translation < .10;}
  bool failed(double elapsed)
  {
    if (!no_initial_motion(elapsed)) {stationary_failures = 0; return false;}
    if (stationary_failures && std::hypot(start_x-failed_x, start_y-failed_y) >= .15) {
      stationary_failures = 0;
    }
    failed_x = start_x; failed_y = start_y;
    blocked = ++stationary_failures >= 2;
    return blocked;
  }
  bool observe_cost(double now, int cost)
  {
    if (blocked) {return true;}
    if (!std::isfinite(now) || cost < 99 || cost > 100) {
      inscribed_since.reset(); return false;
    }
    if (!inscribed_since || now < *inscribed_since) {inscribed_since = now;}
    if (now - *inscribed_since >= 2.0) {blocked = true;}
    return blocked;
  }
};

inline bool fresh_frontier_input(double now, double received, double stamped,
  bool matching_frame, double maximum_age = 10.0)
{
  return matching_frame && std::isfinite(now) && std::isfinite(received) &&
         std::isfinite(stamped) && now >= received && now >= stamped &&
         now - received <= maximum_age && now - stamped <= maximum_age;
}

// Compare categorical occupancy, not noisy costmap inflation. Samples are at
// fixed world coordinates around the failed goal, so map origin/size changes
// do not by themselves count as newly available evidence.
inline int8_t occupancy_class(int value)
{
  if (value < 0 || value > 100) {return -1;}
  if (value <= 25) {return 0;}
  if (value >= 65) {return 1;}
  return -1;
}

inline bool local_map_changed(const std::vector<int8_t> & before,
  const std::vector<int8_t> & after, size_t minimum_changed = 8)
{
  if (before.empty() || before.size() != after.size()) {return false;}
  size_t changed = 0;
  for (size_t i = 0; i < before.size(); ++i) {
    // Losing knowledge is not evidence that an inaccessible goal became usable.
    if (after[i] >= 0 && before[i] != after[i]) {++changed;}
  }
  return changed >= minimum_changed;
}

inline bool retry_failed_region(double age, double cooldown,
  const std::vector<int8_t> & before, const std::vector<int8_t> & after)
{
  return age >= std::max(0.0, cooldown) && local_map_changed(before, after);
}

inline bool no_frontier_stop(bool enabled, double exploration_age, double minimum_age,
  int empty_messages, int empty_threshold, double unusable_age, double blocked_timeout)
{
  return enabled && exploration_age >= minimum_age &&
         (empty_messages >= empty_threshold || unusable_age >= blocked_timeout);
}
}  // namespace tb3_frontier_exploration
