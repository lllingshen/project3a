#pragma once

#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Transform.h>
#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include "footprint_geometry.hpp"

namespace tb3_frontier_exploration
{
inline int64_t footprint_stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec<0 || stamp.nanosec>=1000000000U) {return -1;}
  return static_cast<int64_t>(stamp.sec)*1000000000LL+stamp.nanosec;
}

inline std::optional<tf2::Transform> footprint_transform(
  const geometry_msgs::msg::TransformStamped & value,
  const std::string & frame, const std::string & base)
{
  if (value.header.frame_id != frame || value.child_frame_id != base) {return std::nullopt;}
  const auto & t=value.transform.translation;
  const auto & q=value.transform.rotation;
  for (const double v : {t.x,t.y,t.z,q.x,q.y,q.z,q.w}) {
    if (!std::isfinite(v)) {return std::nullopt;}
  }
  tf2::Quaternion rotation(q.x,q.y,q.z,q.w);
  if (std::abs(rotation.length2()-1.0) > 1e-3) {return std::nullopt;}
  rotation.normalize();
  return tf2::Transform(rotation,tf2::Vector3(t.x,t.y,t.z));
}

struct VerifiedFootprint
{
  FootprintPolygon polygon;
  int64_t source_stamp_ns;
};

// A tiny provenance cache avoids waiting forever when the most recent
// footprint precedes its TF delivery. Each candidate still requires its own
// exact historical transform; an older available candidate must be <=3s old.
class FootprintProofCache
{
public:
  void observe_clock(int64_t now_ns)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    check_clock(now_ns);
  }

  void add(const geometry_msgs::msg::PolygonStamped::ConstSharedPtr & polygon, int64_t now_ns)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    check_clock(now_ns);
    if (!polygon) {return;}
    const auto stamp=footprint_stamp_ns(polygon->header.stamp);
    samples_.erase(std::remove_if(samples_.begin(),samples_.end(),[stamp](const auto & p) {
      return footprint_stamp_ns(p->header.stamp)==stamp;
    }),samples_.end());
    samples_.push_front(polygon);
    std::sort(samples_.begin(),samples_.end(),[](const auto & a,const auto & b) {
      return footprint_stamp_ns(a->header.stamp)>footprint_stamp_ns(b->header.stamp);
    });
    while (samples_.size()>3) {samples_.pop_back();}
  }

  // lookup(stamp_ns) MUST perform a nonblocking, exact-time map<-base lookup.
  // Its returned stamp is checked, so returning latest instead fails closed.
  template<typename Lookup>
  std::optional<VerifiedFootprint> current(const std::string & frame,
    const std::string & base, int64_t now_ns,
    const geometry_msgs::msg::TransformStamped & robot, Lookup lookup)
  {
    std::deque<geometry_msgs::msg::PolygonStamped::ConstSharedPtr> samples;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      check_clock(now_ns);
      samples=samples_;
    }
    const auto robot_stamp=footprint_stamp_ns(robot.header.stamp);
    const auto current_tf=footprint_transform(robot,frame,base);
    if (now_ns<=0 || robot_stamp<=0 || robot_stamp>now_ns ||
      now_ns-robot_stamp>750000000LL || !current_tf) {return std::nullopt;}
    for (const auto & sample : samples) {
      const auto stamp=footprint_stamp_ns(sample->header.stamp);
      if (sample->header.frame_id!=frame || stamp<=0 || stamp>now_ns ||
        now_ns-stamp>3000000000LL) {continue;}
      if (sample->polygon.points.size()<3 || sample->polygon.points.size()>64) {continue;}
      FootprintPolygon original;
      bool finite=true;
      for (const auto & p : sample->polygon.points) {
        finite=finite && std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        original.emplace_back(p.x,p.y);
      }
      if (!finite || !valid_footprint(original)) {continue;}
      try {
        const auto old_robot=lookup(stamp);
        if (footprint_stamp_ns(old_robot.header.stamp)!=stamp) {continue;}
        const auto old_tf=footprint_transform(old_robot,frame,base);
        if (!old_tf) {continue;}
        FootprintPolygon local, projected;
        for (const auto & p : sample->polygon.points) {
          const auto body=old_tf->inverse()*tf2::Vector3(p.x,p.y,p.z);
          const auto world=(*current_tf)*body;
          local.emplace_back(body.x(),body.y());
          projected.emplace_back(world.x(),world.y());
        }
        if (!valid_footprint(local) || !footprint_contains(local,{0,0}) ||
          !valid_footprint(projected) || !footprint_contains(projected,
          {robot.transform.translation.x,robot.transform.translation.y})) {continue;}
        return VerifiedFootprint{projected,stamp};
      } catch (const std::exception &) {
        // TF has not reached this stamp yet, or its history was evicted.
        // The next cached sample is independently checked; no latest fallback.
      }
    }
    return std::nullopt;
  }

private:
  void check_clock(int64_t now_ns)
  {
    if (last_now_ && now_ns<*last_now_) {samples_.clear();}
    last_now_=now_ns;
  }
  std::mutex mutex_;
  std::optional<int64_t> last_now_;
  std::deque<geometry_msgs::msg::PolygonStamped::ConstSharedPtr> samples_;
};
}  // namespace tb3_frontier_exploration
