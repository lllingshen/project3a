#pragma once

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "safe_frontier.hpp"

namespace tb3_frontier_exploration
{
// A proposal is a sensor-derived observation position, never an object goal.
// Repeated publications may renew its timestamp, but cannot change its pose.
struct ObservationRequest
{
  std::string request_id, track_id, frame_id;
  int64_t stamp_ns{0};
  double x{0}, y{0}, yaw{0};
  bool clear() const {return request_id.empty();}
};

inline bool observation_id_valid(const std::string & id)
{
  if (id.empty() || id.size() > 128) {return false;}
  for (const char c : id) {
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
      (c >= '0' && c <= '9') || c == '_' || c == '-' || c == ':' || c == '.'))
    {return false;}
  }
  return true;
}

inline std::optional<ObservationRequest> parse_observation_request(
  const std::string & text, std::string & reason, const std::string & frame = "map")
{
  reason.clear();
  try {
    if (text.size() > 4096) {throw std::runtime_error("request exceeds 4096 bytes");}
    const auto j = nlohmann::json::parse(text);
    if (!j.is_object() || !j.contains("request_id") || !j["request_id"].is_string()) {
      throw std::runtime_error("request_id must be a string in a JSON object");
    }
    ObservationRequest r;
    r.request_id = j["request_id"].get<std::string>();
    if (r.clear()) {
      if (j.size() != 1) {throw std::runtime_error("clear request may contain only request_id");}
      return r;
    }
    const std::set<std::string> fields{"request_id", "track_id", "frame_id", "stamp_ns", "x", "y", "yaw"};
    if (j.size() != fields.size()) {throw std::runtime_error("request fields do not match schema");}
    for (const auto & key : fields) {
      if (!j.contains(key)) {throw std::runtime_error("missing request field: " + key);}
    }
    if (!j["track_id"].is_string() || !j["frame_id"].is_string()) {
      throw std::runtime_error("track_id/frame_id must be strings");
    }
    r.track_id = j["track_id"].get<std::string>();
    r.frame_id = j["frame_id"].get<std::string>();
    if (!observation_id_valid(r.request_id) || !observation_id_valid(r.track_id)) {
      throw std::runtime_error("invalid or overlong request/track ID");
    }
    if (r.frame_id != frame) {throw std::runtime_error("request frame must exactly match exploration frame");}
    const auto & stamp = j["stamp_ns"];
    if (!stamp.is_number_integer() ||
      (stamp.is_number_unsigned() && stamp.get<uint64_t>() > static_cast<uint64_t>(INT64_MAX)))
    {throw std::runtime_error("stamp_ns must be a nonnegative int64");}
    r.stamp_ns = stamp.get<int64_t>();
    if (r.stamp_ns < 0) {throw std::runtime_error("stamp_ns must be nonnegative");}
    for (const char * key : {"x", "y", "yaw"}) {
      if (!j[key].is_number() || !std::isfinite(j[key].get<double>())) {
        throw std::runtime_error(std::string(key) + " must be finite numeric data");
      }
    }
    r.x = j["x"].get<double>(); r.y = j["y"].get<double>(); r.yaw = j["yaw"].get<double>();
    return r;
  } catch (const std::exception & e) {
    reason = e.what();
    return std::nullopt;
  }
}

inline bool observation_fresh(const ObservationRequest & request, int64_t now_ns)
{
  return now_ns >= request.stamp_ns && now_ns - request.stamp_ns <= 3000000000LL;
}

inline bool observation_busy_fresh(bool busy, const std::optional<int64_t> & stamp_ns, int64_t now_ns)
{
  return busy && stamp_ns && now_ns >= *stamp_ns && now_ns - *stamp_ns <= 3000000000LL;
}

// Fixed protocol bound, matching RevisitState.time_exhausted/time_limit_sec.
// This is not a separately configurable duration; stage validation checks the
// current Python policy as well as this constant.
inline constexpr int64_t OBSERVATION_PHASE_LIMIT_NS = 120000000000LL;

// One bounded semantic phase per simulation-clock epoch. Only an actual send
// anchors it. A withdrawal, result, duplicate proposal, or pause/resume must
// never renew the Python proposer's continuous 120-second budget.
class ObservationPhase
{
public:
  void dispatched(int64_t now_ns)
  {
    if (!first_dispatch_ns_ && now_ns >= 0) {first_dispatch_ns_ = now_ns;}
  }

  void clock_rollback() {first_dispatch_ns_.reset();}

  bool expired(int64_t now_ns) const
  {
    return first_dispatch_ns_ && now_ns >= *first_dispatch_ns_ &&
           now_ns - *first_dispatch_ns_ >= OBSERVATION_PHASE_LIMIT_NS;
  }

  bool reserve(bool enabled, bool busy, const std::optional<int64_t> & busy_stamp_ns,
    int64_t now_ns) const
  {
    return enabled && first_dispatch_ns_ && now_ns >= *first_dispatch_ns_ &&
           !expired(now_ns) && observation_busy_fresh(busy, busy_stamp_ns, now_ns);
  }

private:
  std::optional<int64_t> first_dispatch_ns_;
};

// Static occupancy does not expire while a robot waits. Dynamic obstacles and
// robot pose do expire. Publishing frontiers certifies this evaluation time,
// not that the underlying SLAM geometry was recently changed.
inline bool safety_evaluation_fresh(int64_t now_ns, int64_t map_ns, int64_t cost_ns, int64_t robot_ns)
{
  return map_ns >= 0 && cost_ns >= 0 && robot_ns >= 0 && now_ns >= map_ns &&
         now_ns >= cost_ns && now_ns - cost_ns <= 3000000000LL &&
         now_ns >= robot_ns && now_ns - robot_ns <= 750000000LL;
}

// TF delivery can precede the node's /clock callback. In that case request an
// interpolated transform at the captured clock time, never accept future pose
// data directly. A recent past transform remains usable without extrapolating.
inline std::optional<int64_t> robot_safety_lookup_stamp(int64_t now_ns, int64_t latest_ns)
{
  if (now_ns < 0 || latest_ns < 0) {return std::nullopt;}
  if (latest_ns > now_ns) {return now_ns;}
  if (now_ns - latest_ns > 750000000LL) {return std::nullopt;}
  return latest_ns;
}

inline bool observation_dispatch_allowed(bool have_proposal, bool ordinary_eligible,
  bool geometry_done, bool exploration_complete)
{
  return have_proposal && !ordinary_eligible && (geometry_done || exploration_complete);
}

// Check continuous cell coordinates before GridView::cell casts to int. A
// finite but absurd JSON coordinate must fail closed, not overflow that cast.
inline bool observation_grid_contains(const GridView & grid, double wx, double wy)
{
  if (!grid.valid() || !std::isfinite(wx) || !std::isfinite(wy)) {return false;}
  const double dx = wx - grid.x, dy = wy - grid.y;
  const double col = (std::cos(grid.yaw)*dx + std::sin(grid.yaw)*dy)/grid.resolution;
  const double row = (-std::sin(grid.yaw)*dx + std::cos(grid.yaw)*dy)/grid.resolution;
  return std::isfinite(col) && std::isfinite(row) &&
         col >= 0 && row >= 0 && col < grid.width && row < grid.height;
}

inline bool same_observation(const ObservationRequest & a, const ObservationRequest & b)
{
  return a.request_id == b.request_id && a.track_id == b.track_id && a.frame_id == b.frame_id &&
         a.x == b.x && a.y == b.y && a.yaw == b.yaw;
}

// One proposal in flight. Retired IDs are never evicted: at the bounded cap
// new proposals fail closed rather than silently making an old ID replayable.
class ObservationProtocol
{
public:
  std::optional<ObservationRequest> pending, active;

  bool offer(const ObservationRequest & request, int64_t now_ns, std::string & reason)
  {
    reason.clear();
    if (request.clear()) {reason = "clear must cancel owned state explicitly"; return false;}
    if (active && same_observation(*active, request)) {return true;}
    if (active) {reason = "another observation is active"; return false;}
    if (retired_.count(request.request_id)) {reason = "request_id was already handled"; return false;}
    if (retired_.size() >= 4096) {reason = "request replay ledger is full"; return false;}
    if (!observation_fresh(request, now_ns)) {reason = "request is expired or future-dated"; return false;}
    if (pending && !same_observation(*pending, request)) {
      reason = "another immutable proposal is pending; clear before replacing"; return false;
    }
    pending = request;
    return true;
  }

  std::optional<ObservationRequest> retire_pending()
  {
    const auto previous = pending;
    if (previous) {retired_.insert(previous->request_id);}
    pending.reset();
    return previous;
  }

  std::optional<ObservationRequest> start(int64_t now_ns)
  {
    if (active || !pending || !observation_fresh(*pending, now_ns)) {return std::nullopt;}
    active = retire_pending();
    return active;
  }

  void finish(const std::string & request_id)
  {
    if (active && active->request_id == request_id) {active.reset();}
  }

  std::vector<ObservationRequest> cancel_all()
  {
    std::vector<ObservationRequest> result;
    if (pending) {result.push_back(*retire_pending());}
    if (active) {result.push_back(*active); active.reset();}
    return result;
  }

  // Caller cancels the actual action and emits events when this returns true.
  bool clock_rolled_back(int64_t now_ns)
  {
    const bool rollback = last_clock_ns_ && now_ns < *last_clock_ns_;
    last_clock_ns_ = now_ns;
    return rollback;
  }

private:
  std::set<std::string> retired_;
  std::optional<int64_t> last_clock_ns_;
};
}  // namespace tb3_frontier_exploration
