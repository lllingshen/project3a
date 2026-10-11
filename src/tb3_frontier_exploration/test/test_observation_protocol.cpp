#include "../src/observation_protocol.hpp"
#include "../src/safe_frontier.hpp"
#include "../src/exploration_policy.hpp"

#include <iostream>
#include <stdexcept>

using namespace tb3_frontier_exploration;

void require(bool condition, const char * message)
{
  if (!condition) {throw std::runtime_error(message);}
}

int main()
{
  std::string reason;
  const std::string valid = R"({"request_id":"epoch1:request_0","track_id":"candidate_7","frame_id":"map","stamp_ns":10000000000,"x":1.2,"y":-0.7,"yaw":2.1})";
  auto request = parse_observation_request(valid, reason);
  require(request && reason.empty() && request->track_id == "candidate_7" && request->x == 1.2,
    "valid sensor-derived map proposal preserves its identity and geometry");
  require(parse_observation_request(R"({"request_id":""})", reason)->clear(),
    "explicit empty request ID is a withdrawal, not a zero-coordinate goal");
  for (const auto & invalid : std::vector<std::string>{
      "null", "[]", "{", "{}", R"({"request_id":12})", R"({"request_id":"","x":0})",
      std::string(4097, 'x')})
  {
    require(!parse_observation_request(invalid, reason) && !reason.empty(),
      "malformed, missing or oversized request fails closed");
  }
  const auto json = nlohmann::json::parse(valid);
  const auto invalid_field = [&](const char * key, const nlohmann::json & value) {
      auto changed = json; changed[key] = value;
      require(!parse_observation_request(changed.dump(), reason), "invalid schema field must reject");
    };
  invalid_field("track_id", "candidate 7");
  invalid_field("request_id", std::string(129, 'a'));
  invalid_field("frame_id", "/map");
  invalid_field("frame_id", "odom");
  invalid_field("x", nullptr);
  invalid_field("x", "1.2");
  invalid_field("y", true);
  invalid_field("yaw", nullptr);
  invalid_field("stamp_ns", -1);
  invalid_field("stamp_ns", 10000000000.5);
  invalid_field("stamp_ns", std::numeric_limits<uint64_t>::max());
  auto missing = json; missing.erase("track_id");
  require(!parse_observation_request(missing.dump(), reason), "missing identity cannot be inferred");
  auto extra = json; extra["source"] = "truth";
  require(!parse_observation_request(extra.dump(), reason), "unknown request fields cannot alter protocol");
  auto nonfinite = valid;
  nonfinite.replace(nonfinite.find("1.2"), 3, "1e999");
  require(!parse_observation_request(nonfinite, reason), "overflow/nonfinite coordinates fail closed");

  require(observation_fresh(*request, 10000000000LL), "current simulation timestamp accepted");
  require(observation_fresh(*request, 13000000000LL), "exact three-second boundary accepted");
  require(!observation_fresh(*request, 13000000001LL), "expired request cannot dispatch");
  require(!observation_fresh(*request, 9999999999LL), "future/previous-run timestamp cannot dispatch");
  require(!observation_busy_fresh(false,10000000000LL,10000000000LL), "false heartbeat never blocks baseline completion");
  require(!observation_busy_fresh(true,std::nullopt,10000000000LL), "missing publisher preserves baseline behavior");
  require(observation_busy_fresh(true,10000000000LL,13000000000LL), "fresh true heartbeat prevents premature semantic completion");
  require(!observation_busy_fresh(true,10000000000LL,13000000001LL), "lost busy publisher cannot block completion forever");
  require(!observation_busy_fresh(true,10000000000LL,1000000000LL), "clock reset invalidates old busy heartbeat");
  require(safety_evaluation_fresh(100000000000LL,1000000000LL,100000000000LL,100000000000LL),
    "static old map plus fresh dynamic safety inputs produces a current frontier evaluation");
  require(!safety_evaluation_fresh(100000000000LL,1000000000LL,96000000000LL,100000000000LL),
    "stale dynamic obstacles cannot be certified by an old or new map");
  require(!safety_evaluation_fresh(100000000000LL,1000000000LL,100000000000LL,99000000000LL),
    "stale robot TF cannot authorize observation or ordinary frontier evaluation");
  require(!safety_evaluation_fresh(1000000000LL,10000000000LL,1000000000LL,1000000000LL),
    "clock reset invalidates previous-run map despite fresh dynamic data");
  require(!safety_evaluation_fresh(1000000000LL,0,1000000001LL,1000000000LL),
    "future dynamic timestamps cannot authorize goals");
  require(robot_safety_lookup_stamp(10000000000LL,10099000000LL) == 10000000000LL,
    "TF arriving 99ms ahead of the node clock requires interpolation at captured now");
  require(robot_safety_lookup_stamp(10000000000LL,9967000000LL) == 9967000000LL,
    "fresh past 33ms TF remains usable without impossible forward extrapolation");
  require(robot_safety_lookup_stamp(10000000000LL,9983000000LL) == 9983000000LL,
    "fresh past 17ms TF remains usable without widening future tolerance");
  require(robot_safety_lookup_stamp(10000000000LL,9250000000LL).has_value() &&
    !robot_safety_lookup_stamp(10000000000LL,9249999999LL),
    "the existing .75-second stale limit is unchanged");
  require(!robot_safety_lookup_stamp(-1,0) && !robot_safety_lookup_stamp(1,-1),
    "invalid simulation timestamps cannot select a lookup time");
  require(!safety_evaluation_fresh(10000000000LL,0,10000000000LL,10099000000LL) &&
    safety_evaluation_fresh(10000000000LL,0,10000000000LL,
      *robot_safety_lookup_stamp(10000000000LL,10099000000LL)),
    "future latest pose itself still fails; only successfully interpolated current pose is eligible");
  require(!observation_dispatch_allowed(false,false,true,false),
    "no semantic request preserves ordinary completion rather than injecting a goal");
  require(!observation_dispatch_allowed(true,true,true,false),
    "before the first actual observation, ordinary frontier priority is unchanged");
  require(observation_dispatch_allowed(true,false,true,false),
    "verified proposal can dispatch when ordinary geometry is complete");
  require(observation_dispatch_allowed(true,false,false,true),
    "new evidence can request bounded observation after an earlier geometric completion");
  require(!observation_dispatch_allowed(true,false,
    no_frontier_stop(true,30,60,10,3,25,20),false),
    "temporary startup emptiness cannot consume observation budget before minimum exploration age");
  require(!observation_dispatch_allowed(true,false,
    no_frontier_stop(true,70,60,0,3,3,20),false),
    "one transient filtered frontier tick cannot launch semantic verification");
  require(observation_dispatch_allowed(true,false,
    no_frontier_stop(true,70,60,0,3,20,20),false),
    "unchanged geometric stalemate condition authorizes the semantic phase");

  // Reproduce both observed gaps without a ROS action server. A queued or
  // withdrawn offer is not a send; only the local successful send call anchors
  // priority, and delayed acceptance/results cannot renew it.
  constexpr int64_t sec = 1000000000LL;
  ObservationPhase phase;
  ObservationProtocol phase_queue;
  auto phase_request = *request;
  phase_request.stamp_ns = 200*sec;
  require(phase_queue.offer(phase_request, 200*sec, reason), "phase proposal setup");
  require(!phase.reserve(true,true,200*sec,200*sec),
    "proposal and fresh busy alone never steal startup/ordinary frontier priority");
  require(phase_queue.start(200*sec).has_value(), "phase accepted queue setup");
  require(!phase.reserve(true,true,200*sec,200*sec),
    "queue start before actual send still cannot anchor a semantic phase");
  phase.dispatched(200*sec);
  phase_queue.finish(phase_request.request_id);
  require(phase.reserve(true,true,224*sec,224*sec),
    "gap after first five-second dwell reserves next tick before delayed proposal exists");
  phase_request.request_id = "epoch1:phase_second";
  phase_request.stamp_ns = 225*sec;
  require(phase_queue.offer(phase_request,225*sec,reason), "second proposal arrives after dwell gap");
  phase_queue.cancel_all();
  require(phase.reserve(true,true,236*sec,236*sec),
    "withdrawal while catalog settles does not hand the established phase to a frontier");
  require(phase.reserve(true,true,319*sec,319*sec),
    "planning, stale-sensor waits and catalog work consume the same continuous window");
  require(!phase.reserve(true,true,320*sec,320*sec) && phase.expired(320*sec),
    "exact 120-second deadline restores frontier opportunity even if busy remains true");
  phase.dispatched(310*sec);  // Simulates another actual send / duplicate send notification.
  phase.dispatched(500*sec);
  require(phase.expired(320*sec) && !phase.reserve(true,true,501*sec,501*sec),
    "subsequent send, delayed acceptance/result or new offer cannot replenish the deadline");
  require(!phase.reserve(false,true,250*sec,250*sec),
    "disable/user pause releases priority immediately while preserving the first anchor");
  require(phase.reserve(true,true,250*sec,250*sec) && phase.expired(320*sec),
    "resume before deadline uses the remaining original window, never a new 120 seconds");
  require(!phase.reserve(true,false,250*sec,250*sec) &&
    !phase.reserve(true,true,std::nullopt,250*sec),
    "false or absent semantic heartbeat preserves normal navigation opportunity");
  require(phase.reserve(true,true,247*sec,250*sec) &&
    !phase.reserve(true,true,247*sec,250*sec+1),
    "unchanged three-second busy freshness boundary limits a lost heartbeat");
  require(!phase.reserve(true,true,251*sec,250*sec),
    "future busy evidence cannot reserve priority");
  require(!phase.reserve(true,true,10*sec,10*sec),
    "an old phase cannot reserve after clock rollback even before reset is handled");
  phase.clock_rollback();
  require(!phase.reserve(true,true,10*sec,10*sec) && !phase.expired(10*sec),
    "explicit simulation epoch reset discards the old first-send anchor");
  phase.dispatched(-1);
  require(!phase.reserve(true,true,10*sec,10*sec), "invalid send timestamp does not open a phase");
  phase.dispatched(10*sec);
  require(phase.reserve(true,true,129*sec,129*sec) && phase.expired(130*sec),
    "a genuinely new clock epoch receives one new fixed window");

  // R17 measured dispatch/capture boundaries: first send at 216.3 sim s,
  // dwell finishes about 240.3, next offer at 241; later catalog withdrawal
  // precedes a frontier send about 254.3 and the next offer at 257. Both
  // gaps were covered by observed true busy heartbeats less than 1.4 s old.
  ObservationPhase measured_phase;
  measured_phase.dispatched(216300000000LL);
  require(measured_phase.reserve(true,true,239000000000LL,240300000000LL),
    "recorded first frontier boundary is reserved even before offer241 exists");
  require(measured_phase.reserve(true,true,253000000000LL,254300000000LL),
    "recorded second frontier boundary is reserved during catalog's no-offer gap");
  require(!measured_phase.reserve(true,true,336300000000LL,336300000000LL),
    "recorded first send has an immutable336.3sim deadline despite both gaps");

  ObservationProtocol queue;
  require(queue.offer(*request, 11000000000LL, reason) && queue.pending && !queue.active,
    "receipt queues a request without charging an actual dispatch");
  auto refresh = *request; refresh.stamp_ns = 12000000000LL;
  require(queue.offer(refresh, 12000000000LL, reason) && queue.pending->stamp_ns == refresh.stamp_ns,
    "heartbeat renews the same immutable proposal while geometric frontiers have priority");
  auto moved = refresh; moved.x += .1;
  require(!queue.offer(moved, 12000000000LL, reason) && queue.pending->x == request->x,
    "same ID cannot silently change destination");
  auto other = refresh; other.request_id = "epoch1:request_1";
  require(!queue.offer(other, 12000000000LL, reason), "pending proposal needs explicit withdrawal before replacement");
  auto dispatched = queue.start(13000000000LL);
  require(dispatched && queue.active && !queue.pending, "dispatch moves exactly one proposal to active");
  require(!queue.start(13000000000LL), "the same pending request cannot send twice");
  require(queue.offer(refresh, 20000000000LL, reason) && queue.active && !queue.pending,
    "late duplicate publication of active request cannot enqueue another goal");
  require(!queue.offer(other, 13000000000LL, reason), "only one observation action may be active");
  queue.finish(other.request_id);
  require(queue.active.has_value(), "late result for another ID cannot finish active request");
  queue.finish(dispatched->request_id);
  refresh.stamp_ns = 14000000000LL;
  require(!queue.offer(refresh, 14000000000LL, reason), "completed ID remains non-replayable even with fresh timestamp");
  require(dispatched->x == 1.2 && dispatched->request_id == request->request_id,
    "callback copy preserves original coordinates and IDs after queue changes");

  ObservationProtocol expired;
  require(expired.offer(*request, 10000000000LL, reason), "queue expiry setup");
  require(!expired.start(14000000000LL), "expiry is rechecked at send, not only at receipt");
  require(expired.retire_pending().has_value() && !expired.pending,
    "stale proposal is removed with identity for a correlated rejection event");
  require(!expired.offer(refresh, 14000000000LL, reason), "expired ID cannot replay as a new attempt");

  ObservationProtocol cancel;
  require(cancel.offer(*request, 10000000000LL, reason) && cancel.start(10000000000LL), "cancel setup");
  require(!cancel.clock_rolled_back(10000000000LL) && !cancel.clock_rolled_back(11000000000LL),
    "normal clock progress does not invalidate requests");
  require(cancel.clock_rolled_back(1000000000LL), "simulation rollback is explicit");
  const auto invalidated = cancel.cancel_all();
  require(invalidated.size() == 1 && invalidated[0].request_id == request->request_id &&
    !cancel.pending && !cancel.active, "clock reset/pause/withdraw returns the owned ID and clears all work");
  require(cancel.cancel_all().empty(), "repeated pause cannot re-emit an active request");
  auto new_epoch = *request; new_epoch.request_id = "epoch2:request_0"; new_epoch.stamp_ns = 1000000000LL;
  require(cancel.offer(new_epoch, 1000000000LL, reason), "new epoch uses a new request identity");
  cancel.finish(request->request_id);
  require(cancel.pending && cancel.pending->request_id == new_epoch.request_id,
    "old immutable callback cannot remove a new epoch proposal");

  // The dispatcher's extra raw-map safety check must reject disconnected free
  // islands and unknown endpoints even when the global costmap says zero.
  std::vector<int8_t> map_data(49, 0), costs(49, 0);
  for (int y = 0; y < 7; ++y) {map_data[y*7+3] = 100;}
  GridView map{7,7,.2,-.7,-.7,0,map_data}, cost{7,7,.2,-.7,-.7,0,costs};
  require(observation_grid_contains(map,-.4,0) && !observation_grid_contains(map,1e300,0) &&
    !observation_grid_contains(map,0,-1e300), "finite huge coordinates fail before integer cell conversion");
  const auto reachable = reachable_free_cells(map, cost, -.4, 0);
  require(reachable[3*7+1] && !reachable[3*7+5],
    "cheap isolated destination is not reachable through occupied map wall");
  map_data[3*7+1] = -1;
  require(map.at(-.4,0) == -1, "unknown raw map cell is not certified by a free costmap");
  std::cout << "observation protocol and map-safety checks passed\n";
}
