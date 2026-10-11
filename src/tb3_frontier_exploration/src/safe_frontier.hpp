#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <utility>
#include <vector>
#include "footprint_geometry.hpp"

namespace tb3_frontier_exploration
{
using GridCell = std::pair<int, int>;

// Non-owning grid view: map and costmap may have different origins/resolutions.
struct GridView
{
  int width, height;
  double resolution, x, y, yaw;
  const std::vector<int8_t> & data;

  bool valid() const
  {return width > 0 && height > 0 && resolution > 0 &&
      data.size() == static_cast<size_t>(width) * height;}
  bool inside(int col, int row) const
  {return col >= 0 && row >= 0 && col < width && row < height;}
  int index(int col, int row) const {return row * width + col;}
  GridCell cell(double wx, double wy) const
  {
    const double dx = wx - x, dy = wy - y;
    return {static_cast<int>(std::floor((std::cos(yaw)*dx + std::sin(yaw)*dy)/resolution)),
      static_cast<int>(std::floor((-std::sin(yaw)*dx + std::cos(yaw)*dy)/resolution))};
  }
  std::pair<double, double> center(int col, int row) const
  {
    const double dx = (col + .5)*resolution, dy = (row + .5)*resolution;
    return {x + std::cos(yaw)*dx - std::sin(yaw)*dy,
      y + std::sin(yaw)*dx + std::cos(yaw)*dy};
  }
  int at(double wx, double wy) const
  {
    if (!valid() || !std::isfinite(wx) || !std::isfinite(wy)) {return -1;}
    const auto [col, row] = cell(wx, wy);
    return inside(col, row) ? data[index(col, row)] : -1;
  }
};

inline FootprintPolygon grid_cell_corners(const GridView & grid, int col, int row)
{
  FootprintPolygon corners;
  for (const auto & [dx,dy] : std::vector<GridCell>{{0,0},{1,0},{1,1},{0,1}}) {
    const double x = (col+dx)*grid.resolution, y = (row+dy)*grid.resolution;
    corners.emplace_back(grid.x+std::cos(grid.yaw)*x-std::sin(grid.yaw)*y,
      grid.y+std::sin(grid.yaw)*x+std::cos(grid.yaw)*y);
  }
  return corners;
}

// Nav2 OccupancyGrid: 1..98 is graded inflation, 99 inscribed, 100 lethal.
// Inflation may refill the footprint after obstacle-layer clearing. A bridge
// uses the same traversability bound as a known-free route, never an unknown
// cost or a collision cell. Goal cost limits are checked separately.
inline bool traversable_cost(int cost) {return cost >= 0 && cost < 99;}

// Require the entire unknown map cell to be inside the verified footprint,
// and every cost cell overlapping its area to be traversable. This is not
// a centre sample or an axis-aligned bounding-box approximation.
inline bool footprint_bridge_cell(const GridView & map, const GridView & costs,
  int col, int row, const FootprintPolygon & footprint)
{
  const auto corners = grid_cell_corners(map,col,row);
  FootprintPolygon cost_corners;
  double min_x=INFINITY, min_y=INFINITY, max_x=-INFINITY, max_y=-INFINITY;
  for (const auto & p : corners) {
    if (!footprint_contains(footprint,p)) {return false;}
    const double dx=p.first-costs.x, dy=p.second-costs.y;
    const double x=(std::cos(costs.yaw)*dx+std::sin(costs.yaw)*dy)/costs.resolution;
    const double y=(-std::sin(costs.yaw)*dx+std::cos(costs.yaw)*dy)/costs.resolution;
    if (!std::isfinite(x) || !std::isfinite(y)) {return false;}
    min_x=std::min(min_x,x); max_x=std::max(max_x,x);
    min_y=std::min(min_y,y); max_y=std::max(max_y,y);
    cost_corners.emplace_back(x,y);
  }
  // Never clip an uncovered portion to the available costmap.
  if (min_x < -footprint_epsilon || min_y < -footprint_epsilon ||
    max_x > costs.width+footprint_epsilon || max_y > costs.height+footprint_epsilon)
  {return false;}
  bool covered=false;
  const int x0=std::max(0,static_cast<int>(std::floor(min_x)));
  const int y0=std::max(0,static_cast<int>(std::floor(min_y)));
  const int x1=std::min(costs.width-1,static_cast<int>(std::floor(max_x)));
  const int y1=std::min(costs.height-1,static_cast<int>(std::floor(max_y)));
  for (int y=y0; y<=y1; ++y) for (int x=x0; x<=x1; ++x) {
    const FootprintPolygon cell{{double(x),double(y)},{double(x+1),double(y)},
      {double(x+1),double(y+1)},{double(x),double(y+1)}};
    if (!footprint_positive_overlap(cost_corners,cell)) {continue;}
    covered=true;
    if (!traversable_cost(costs.data[costs.index(x,y)])) {return false;}
  }
  return covered;
}

// Four-connected traversal cannot cut diagonally across occupied corners.
// This establishes known-free connectivity, not a substitute for Nav2 planning.
inline std::vector<uint8_t> reachable_free_cells(const GridView & map,
  const GridView & costs, double rx, double ry,
  const FootprintPolygon * verified_current_footprint = nullptr)
{
  if (!map.valid() || !costs.valid() || !std::isfinite(rx) || !std::isfinite(ry)) {return {};}
  std::vector<uint8_t> reached(map.data.size(), 0);
  const auto [sx, sy] = map.cell(rx, ry);
  // This exception is start-only. A known-free start always follows the
  // original traversal, even when a footprint is available. No map is edited.
  const bool bridge = map.inside(sx,sy) && map.data[map.index(sx,sy)] == -1 &&
    traversable_cost(costs.at(rx,ry)) && verified_current_footprint &&
    valid_footprint(*verified_current_footprint) &&
    footprint_contains(*verified_current_footprint,{rx,ry});
  const auto usable = [&](int x, int y) {
      if (!map.inside(x, y)) {return false;}
      const int raw = map.data[map.index(x,y)];
      if (raw == -1 && bridge) {
        return footprint_bridge_cell(map,costs,x,y,*verified_current_footprint);
      }
      if (raw != 0) {return false;}
      const auto [wx, wy] = map.center(x, y);
      const int cost = costs.at(wx, wy);
      return traversable_cost(cost);
    };
  if (!usable(sx, sy)) {return reached;}
  std::queue<GridCell> pending;
  pending.emplace(sx, sy);
  reached[map.index(sx, sy)] = 1;
  const int dx[] = {1, -1, 0, 0}, dy[] = {0, 0, 1, -1};
  while (!pending.empty()) {
    const auto [x, y] = pending.front(); pending.pop();
    for (int k = 0; k < 4; ++k) {
      const int nx = x + dx[k], ny = y + dy[k];
      if (!usable(nx, ny) || reached[map.index(nx, ny)]) {continue;}
      reached[map.index(nx, ny)] = 1;
      pending.emplace(nx, ny);
    }
  }
  // The temporary bridge is an internal traversal aid, never a known-free
  // endpoint or evidence that an otherwise entirely unknown region is mapped.
  if (bridge) {
    for (size_t i=0; i<reached.size(); ++i) {
      if (map.data[i]!=0) {reached[i]=0;}
    }
  }
  return reached;
}

// Search a bounded known-free band from every actual frontier cell. The band
// is candidate discovery, not the robot's route: costs along the retreat may
// be high, while the selected endpoint must have a safe robot connection.
// Prefer the minimum necessary retreat, then proximity to the cluster centre.
inline std::optional<std::pair<double, double>> safe_frontier_approach(
  const GridView & map, const GridView & costs, const std::vector<uint8_t> & reachable,
  const std::vector<GridCell> & frontier, double cx, double cy,
  double radius = .5, int maximum_cost = 60, int edge_margin = 0,
  GridCell * selected_frontier = nullptr)
{
  if (!map.valid() || !costs.valid() || reachable.size() != map.data.size() ||
    !std::isfinite(radius) || radius < 0 || !std::isfinite(cx) || !std::isfinite(cy))
  {return std::nullopt;}
  std::vector<int> steps(map.data.size(), -1);
  std::vector<GridCell> seeds(map.data.size());
  std::queue<GridCell> pending;
  const auto free = [&](int x, int y) {
      return map.inside(x, y) && map.data[map.index(x, y)] == 0;
    };
  for (const auto & [x, y] : frontier) {
    if (free(x, y) && steps[map.index(x, y)] < 0) {
      steps[map.index(x, y)] = 0;
      seeds[map.index(x, y)] = {x, y}; pending.emplace(x, y);
    }
  }
  std::optional<std::pair<double, double>> best;
  int best_steps = std::numeric_limits<int>::max();
  double best_distance = std::numeric_limits<double>::infinity();
  const int dx[] = {1, -1, 0, 0}, dy[] = {0, 0, 1, -1};
  while (!pending.empty()) {
    const auto [x, y] = pending.front(); pending.pop();
    const int retreat = steps[map.index(x, y)];
    if (retreat > best_steps) {break;}  // BFS has exhausted the best band depth.
    const auto [wx, wy] = map.center(x, y);
    const auto [mx, my] = costs.cell(wx, wy);
    const int cost = costs.at(wx, wy);
    const double distance = std::hypot(wx - cx, wy - cy);
    if (reachable[map.index(x, y)] && cost >= 0 && cost <= maximum_cost &&
      mx >= edge_margin && my >= edge_margin &&
      mx < costs.width - edge_margin && my < costs.height - edge_margin &&
      (retreat < best_steps || distance < best_distance - 1e-9))
    {
      best = {wx, wy}; best_steps = retreat; best_distance = distance;
      if (selected_frontier) {*selected_frontier = seeds[map.index(x, y)];}
    }
    if ((retreat + 1) * map.resolution > radius + 1e-9) {continue;}
    for (int k = 0; k < 4; ++k) {
      const int nx = x + dx[k], ny = y + dy[k];
      if (!free(nx, ny) || steps[map.index(nx, ny)] >= 0) {continue;}
      steps[map.index(nx, ny)] = retreat + 1;
      seeds[map.index(nx, ny)] = seeds[map.index(x, y)]; pending.emplace(nx, ny);
    }
  }
  return best;
}

// Face the unknown side of the actual selected boundary, not the cluster's
// arithmetic centre (which can lie inside already explored space for a ring).
inline std::optional<double> frontier_outward_yaw(
  const GridView & map, GridCell seed, double approach_x, double approach_y)
{
  const auto [x, y] = seed;
  if (!map.valid() || !map.inside(x, y) || map.data[map.index(x, y)] != 0 ||
    !std::isfinite(approach_x) || !std::isfinite(approach_y)) {return std::nullopt;}
  const auto [sx, sy] = map.center(x, y);
  double sum_x = 0, sum_y = 0;
  std::optional<std::pair<double, double>> first_unknown;
  for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
    const int nx = x + dx, ny = y + dy;
    if (!map.inside(nx, ny) || map.data[map.index(nx, ny)] >= 0) {continue;}
    const auto unknown = map.center(nx, ny);
    if (!first_unknown) {first_unknown = unknown;}
    sum_x += unknown.first - sx; sum_y += unknown.second - sy;
  }
  if (!first_unknown) {return std::nullopt;}
  const double length = std::hypot(sum_x, sum_y);
  // Opposite neighbours may cancel, or their average may point at an occupied
  // gap between them. Use an actual unknown cell in either case.
  auto target = *first_unknown;
  if (length > 1e-9) {
    const auto average = std::make_pair(sx + map.resolution * sum_x / length,
                                        sy + map.resolution * sum_y / length);
    if (map.at(average.first, average.second) < 0) {target = average;}
  }
  return std::atan2(target.second - approach_y, target.first - approach_x);
}
}  // namespace tb3_frontier_exploration
