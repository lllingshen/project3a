#pragma once

namespace tb3_frontier_exploration
{
// Frontier visits measure geometric coverage. A bounded, source-independent
// semantic observation may need fresh visual evidence from the same region.
// Traversability and remembered navigation failures remain shared safety gates.
enum class GoalSelectionPurpose { Frontier, Observation };
enum class GoalSelectionRejection { None, Cost, MinimumDistance, FailedRegion, Visited };

inline GoalSelectionRejection goal_selection_rejection(
  GoalSelectionPurpose purpose, int cost, bool below_frontier_distance,
  bool near_failed_region, bool repeated_frontier_visit)
{
  if (cost < 0 || cost > 60) {return GoalSelectionRejection::Cost;}
  if (purpose == GoalSelectionPurpose::Frontier && below_frontier_distance) {
    return GoalSelectionRejection::MinimumDistance;
  }
  if (near_failed_region) {return GoalSelectionRejection::FailedRegion;}
  if (purpose == GoalSelectionPurpose::Frontier && repeated_frontier_visit) {
    return GoalSelectionRejection::Visited;
  }
  return GoalSelectionRejection::None;
}

inline const char * goal_selection_rejection_reason(GoalSelectionRejection reason)
{
  switch (reason) {
    case GoalSelectionRejection::Cost: return "unknown or unsafe current cost";
    case GoalSelectionRejection::MinimumDistance: return "below frontier minimum distance";
    case GoalSelectionRejection::FailedRegion: return "within a remembered failed navigation region";
    case GoalSelectionRejection::Visited: return "frontier global successful-visit limit";
    case GoalSelectionRejection::None: return "no eligible candidate";
  }
  return "no eligible candidate";
}
}  // namespace tb3_frontier_exploration
