#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace tb3_frontier_exploration
{
using FootprintPoint = std::pair<double, double>;
using FootprintPolygon = std::vector<FootprintPoint>;
constexpr double footprint_epsilon = 1e-9;

inline double footprint_cross(FootprintPoint a, FootprintPoint b, FootprintPoint p)
{return (b.first-a.first)*(p.second-a.second)-(b.second-a.second)*(p.first-a.first);}

// Accept either winding, but no duplicate vertices, concavity, self crossing,
// or degenerate edges. Checking every vertex against every edge also rejects
// star polygons whose consecutive turns happen to share the same sign.
inline bool valid_footprint(const FootprintPolygon & polygon)
{
  if (polygon.size() < 3 || polygon.size() > 64) {return false;}
  for (const auto & p : polygon) {
    if (!std::isfinite(p.first) || !std::isfinite(p.second)) {return false;}
  }
  double winding = 0;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const auto a = polygon[i], b = polygon[(i+1)%polygon.size()];
    const auto c = polygon[(i+2)%polygon.size()];
    const double turn = footprint_cross(a, b, c);
    if (!std::isfinite(turn) || std::abs(turn) <= footprint_epsilon) {return false;}
    if (winding == 0) {winding = turn > 0 ? 1 : -1;}
    if (turn*winding <= footprint_epsilon) {return false;}
    for (const auto & p : polygon) {
      const double side=footprint_cross(a,b,p)*winding;
      if (!std::isfinite(side) || side < -footprint_epsilon) {return false;}
    }
  }
  return true;
}

// Caller has validated the convex polygon once.
inline bool footprint_contains(const FootprintPolygon & polygon, FootprintPoint p)
{
  if (polygon.size() < 3 || !std::isfinite(p.first) || !std::isfinite(p.second)) {return false;}
  const double winding = footprint_cross(polygon[0], polygon[1], polygon[2]) > 0 ? 1 : -1;
  for (size_t i = 0; i < polygon.size(); ++i) {
    if (winding*footprint_cross(polygon[i], polygon[(i+1)%polygon.size()], p) <
      -footprint_epsilon) {return false;}
  }
  return true;
}

// Separating-axis test in cost-grid cell units. Pure edge/point contact has
// zero area and is not an overlapping cost cell. Normals are normalized so
// the tolerance is independent of the map/cost resolution ratio.
inline bool footprint_positive_overlap(const FootprintPolygon & a, const FootprintPolygon & b)
{
  for (const auto * polygon : {&a, &b}) {
    for (size_t i = 0; i < polygon->size(); ++i) {
      const auto p = (*polygon)[i], q = (*polygon)[(i+1)%polygon->size()];
      const double length = std::hypot(q.first-p.first, q.second-p.second);
      if (!(length > 0)) {return false;}
      const double nx = -(q.second-p.second)/length, ny = (q.first-p.first)/length;
      double lo_a = INFINITY, hi_a = -INFINITY, lo_b = INFINITY, hi_b = -INFINITY;
      for (const auto & v : a) {
        const double d = nx*v.first+ny*v.second; lo_a = std::min(lo_a,d); hi_a = std::max(hi_a,d);
      }
      for (const auto & v : b) {
        const double d = nx*v.first+ny*v.second; lo_b = std::min(lo_b,d); hi_b = std::max(hi_b,d);
      }
      if (std::min(hi_a,hi_b)-std::max(lo_a,lo_b) <= footprint_epsilon) {return false;}
    }
  }
  return true;
}
}  // namespace tb3_frontier_exploration
