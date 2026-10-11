#include "../src/exploration_policy.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace tb3_frontier_exploration;

void require(bool condition, const char * message)
{
  if (!condition) {throw std::runtime_error(message);}
}

int main()
{
  const double limit = 15.0 * std::acos(-1.0) / 180.0;
  const double pitch = 36.0 * std::acos(-1.0) / 180.0;
  InclinationGuard tilt;
  require(!tilt.observe(1.0, 0, 0, 1, 0, limit, .5) && !tilt.blocked,
    "arbitrary yaw is not chassis tilt");
  require(!tilt.observe(1.1, 0, std::sin(pitch/2), 0, std::cos(pitch/2), limit, .5),
    "one pitched sample cannot lock exploration");
  require(!tilt.observe(1.2, 0, 0, 0, 1, limit, .5), "level sample clears pending tilt evidence");
  for (int i = 0; i < 5; ++i) {
    require(!tilt.observe(2.0 + .1*i, 0, std::sin(pitch/2), 0, std::cos(pitch/2), limit, .5),
      "sustained tilt must wait the full debounce period");
  }
  require(tilt.observe(2.5, 0, std::sin(pitch/2), 0, std::cos(pitch/2), limit, .5),
    "36 degree pitched robot locks after half a second");
  require(tilt.blocked && !tilt.observe(3.0, 0, 0, 0, 1, limit, .5) && tilt.blocked,
    "leveling alone does not silently re-enable a physically failed run");

  InclinationGuard gap;
  gap.observe(1, std::sin(pitch/2), 0, 0, std::cos(pitch/2), limit, .5);
  require(!gap.observe(2, std::sin(pitch/2), 0, 0, std::cos(pitch/2), limit, .5),
    "missing samples cannot establish continuous tilt");
  gap.observe(2.1, 0, 0, 0, 0, limit, .5);
  require(!gap.unsafe_since && !gap.blocked, "invalid orientation clears pending evidence");
  gap.observe(2.2, 0, std::sin(pitch/2), 0, std::cos(pitch/2), limit, .5);
  gap.discard_pending();
  require(!gap.unsafe_since, "unavailable or stale IMU must break the debounce interval");
  gap.observe(10, 0, std::sin(pitch/2), 0, std::cos(pitch/2), limit, .5);
  require(!gap.observe(1, 0, std::sin(pitch/2), 0, std::cos(pitch/2), limit, .5) &&
    gap.unsafe_since == 1.0, "simulation clock rollback restarts pending tilt evidence");
  InclinationGuard normalized;
  normalized.observe(1, 0, 2*std::sin(pitch/2), 0, 2*std::cos(pitch/2), limit, .5);
  require(std::abs(normalized.tilt_rad - pitch) < 1e-10,
    "orientation normalization preserves physical tilt");

  StartMotionGuard motion;
  motion.start(1, 2);
  motion.observe(1.01, 2);
  require(motion.no_initial_motion(20) && !motion.failed(20),
    "one stationary failure permits one retry");
  motion.start(1.01, 2);
  motion.observe(1.03, 2);
  require(motion.failed(20) && motion.blocked,
    "second stationary failure locks robot rather than blacklisting another target");
  StartMotionGuard moved;
  moved.start(0,0); moved.failed(20);
  moved.start(0,0); moved.observe(.2,0);
  require(!moved.failed(20) && moved.stationary_failures==0,
    "real translation resets stationary-failure suspicion");
  moved.start(1,1);
  require(!moved.no_initial_motion(1) && !moved.failed(1),
    "immediate planner rejection alone is not physical blockage");
  moved.start(0,0); moved.failed(20);
  moved.start(1,0);
  require(!moved.failed(20), "independent start positions do not accumulate a false blockage");
  StartMotionGuard inscribed;
  require(!inscribed.observe_cost(0,99) && !inscribed.observe_cost(1,99),
    "inscribed start requires sustained fresh evidence");
  require(!inscribed.observe_cost(1.5,-1),"missing cost resets pending blockage");
  require(!inscribed.observe_cost(2,100) && inscribed.observe_cost(4,99),
    "persistent inscribed/lethal start locks before dispatch or completion");
  require(inscribed.observe_cost(5,0),"costmap clearing cannot silently unlock a failed run");
  StartMotionGuard rollback;
  rollback.observe_cost(10,99);
  require(!rollback.observe_cost(1,99),"clock rollback restarts inscribed debounce");
  StartMotionGuard late_tf;
  late_tf.start(std::numeric_limits<double>::quiet_NaN(), 0);
  late_tf.observe(1,2);
  require(late_tf.have_start && late_tf.no_initial_motion(20),
    "recovering TF establishes a missing start anchor");
  require(fresh_frontier_input(10,9,5,true), "fresh matching frontier input is usable");
  require(!fresh_frontier_input(20,9,5,true), "stale frontiers cannot dispatch or imply completion");
  require(!fresh_frontier_input(20,19,5,true), "republishing old map evidence is not fresh");
  require(!fresh_frontier_input(10,9,5,false), "mismatched frontier frame fails closed");
  require(!fresh_frontier_input(1,9,5,true), "clock rollback invalidates older-run inputs");

  BootstrapBudget budget;
  require(budget.available(2), "startup must allow bootstrap");
  ++budget.used;
  require(budget.available(2), "second bootstrap must remain available");
  ++budget.used;
  require(!budget.available(2), "known-map fallback cannot run indefinitely");
  budget.used = 0;
  budget.frontier_started = true;
  require(!budget.available(2), "normal frontier exploration cannot return to fallback patrol");
  require(!BootstrapBudget{}.available(0), "zero bootstrap budget disables it");

  // Completion does not depend on a low-cost fallback point being available.
  require(no_frontier_stop(true, 70, 60, 3, 3, 0, 20), "empty frontiers must stop");
  require(no_frontier_stop(true, 70, 60, 0, 3, 20, 20), "filtered frontiers must stop");
  require(!no_frontier_stop(true, 59, 60, 3, 3, 30, 20), "protect startup mapping time");
  require(!no_frontier_stop(true, 70, 60, 0, 3, 0, 20), "usable frontier resets idle interval");
  require(!no_frontier_stop(false, 70, 60, 3, 3, 30, 20), "respect disabled stop policy");

  ProgressWatchdog progress;
  progress.start(0);
  for (int t = 0; t <= 40; ++t) {
    progress.observe(t, .15 * t, 0, 8.0 - .15 * t, .10);
    require(!progress.stalled(t, 20), "long moving route must survive old 20s hard cap");
  }
  require(progress.stalled(60, 20), "stationary route still times out");
  require(!progress.total_expired(40, 180), "long useful route stays within total cap");

  progress.start(0);
  progress.observe(0, 0, 0, 5, .1);
  for (int t = 1; t <= 20; ++t) {
    progress.observe(t, .01 * (t % 2), 0, 5.0 - .01 * (t % 2), .1);
  }
  require(progress.stalled(20, 20), "odometry and remaining-distance noise is not progress");

  const double nan = std::numeric_limits<double>::quiet_NaN();
  progress.start(0);
  progress.observe(0, nan, nan, 8.0, .1);
  progress.observe(15, nan, nan, 7.8, .1);
  require(!progress.stalled(21, 20), "valid Nav2 remaining-path improvement refreshes watchdog");
  progress.observe(34, nan, nan, -1.0, .1);
  progress.observe(34, nan, nan, nan, .1);
  require(progress.stalled(35, 20), "invalid feedback cannot fake progress");

  progress.start(0);
  for (int t = 0; t <= 180; ++t) {
    progress.observe(t, (t % 2) ? .2 : 0, 0, 10.0, .1);
  }
  require(!progress.stalled(180, 20), "movement is visible to stall monitor");
  require(progress.total_expired(180, 180), "oscillation cannot bypass total duration cap");
  progress.start(100);
  progress.observe(120, 1, 0, 9, .1);
  progress.observe(1, 0, 0, 10, .1);
  require(progress.started == 1 && progress.last_progress == 1 &&
    !progress.stalled(1, 20) && !progress.total_expired(1, 180),
    "clock rollback resets progress clock");

  require(occupancy_class(-1) == -1 && occupancy_class(50) == -1,
    "unknown and uncertain map samples remain unknown");
  require(occupancy_class(0) == occupancy_class(20) && occupancy_class(70) == occupancy_class(100),
    "occupancy confidence jitter is not topology change");
  std::vector<int8_t> before(81, 0), after = before;
  require(!retry_failed_region(10000, 90, before, after), "time alone must never retry failed corner");
  for (int i = 0; i < 7; ++i) {after[i] = 1;}
  require(!retry_failed_region(10000, 90, before, after), "minor map jitter cannot rearm failures");
  after[7] = 1;
  require(!retry_failed_region(89, 90, before, after), "local changes still respect retry cooldown");
  require(retry_failed_region(90, 90, before, after), "substantial local occupancy changes permit retry");
  before.assign(81, 1);
  after = before;
  for (int i = 0; i < 8; ++i) {after[i] = 0;}
  require(retry_failed_region(100, 90, before, after), "opened local obstacle can be reconsidered");
  before.assign(81, -1);
  require(retry_failed_region(100, 90, before, after), "newly observed local map can permit retry");
  after.assign(81, -1);
  before.assign(81, 0);
  require(!retry_failed_region(100, 90, before, after), "losing map knowledge is not new reachability");
  require(!retry_failed_region(100, 90, {}, after), "missing evidence cannot rearm failures");
  require(!retry_failed_region(-1, 90, before, after), "negative failure age does not expire record");
  std::cout << "exploration policy regressions passed\n";
}
