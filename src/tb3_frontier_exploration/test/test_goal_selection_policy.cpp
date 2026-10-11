#include "../src/goal_selection_policy.hpp"

#include <iostream>
#include <stdexcept>

using namespace tb3_frontier_exploration;

void require(bool condition, const char * message)
{
  if (!condition) {throw std::runtime_error(message);}
}

int main()
{
  const auto frontier = GoalSelectionPurpose::Frontier;
  const auto observation = GoalSelectionPurpose::Observation;
  using Rejection = GoalSelectionRejection;
  // Real late-r8 contexts selected cost43/cost55 endpoints in twice-visited
  // regions. These are geometry-progress exclusions, not driving hazards.
  for (const int cost : {0, 43, 55, 60}) {
    require(goal_selection_rejection(frontier, cost, false, false, true) == Rejection::Visited,
      "ordinary visited-region filter must remain unchanged");
    require(goal_selection_rejection(observation, cost, false, false, true) == Rejection::None,
      "targeted observation may revisit a traversable region");
    require(goal_selection_rejection(observation, cost, true, false, true) == Rejection::None,
      "observation may turn at a new source-independent viewpoint near the current robot");
    require(goal_selection_rejection(frontier, cost, true, false, false) == Rejection::MinimumDistance,
      "ordinary minimum travel must remain unchanged");
  }
  for (const auto purpose : {frontier, observation}) {
    for (const int cost : {-1, 61, 100}) {
      require(goal_selection_rejection(purpose, cost, false, false, true) == Rejection::Cost,
        "unknown, inflated and lethal endpoint costs remain shared exclusions");
    }
    require(goal_selection_rejection(purpose, 43, false, true, true) == Rejection::FailedRegion,
      "a traversable but previously failed region must remain excluded in both paths");
  }
  require(goal_selection_rejection(frontier, 0, false, false, false) == Rejection::None,
    "an ordinary unvisited safe frontier remains eligible");
  std::cout << "Goal selection purpose/safety policy regression passed\n";
}
