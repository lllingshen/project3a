import json
import math

import pytest

from tb3_locateanything.visual_memory import Landmark, Memory, UnknownObjectId, InvalidObjectReference


SECOND = 1_000_000_000
BOX = [10, 20, 100, 150]


def view(memory, stamp=1, x=0, y=0, yaw=0, payload=None):
    return memory.add_view({} if payload is None else payload, int(stamp * SECOND), x, y, yaw)


def test_historical_sensor_payload_and_tf_survive_new_robot_pose():
    memory = Memory()
    payload = {"rgb": object(), "scan": [1., 2.], "camera_info": object(),
               "scan_to_map": {"translation": [10, 20, 0]}, "camera_to_scan": object()}
    initial = view(memory, payload=payload)
    obj = memory.remember(initial, "chair", BOX, 12, 20, "a seat")
    view(memory, 5, x=8)
    assert memory.views_for_query(10 * SECOND)[0] is initial
    assert initial.payload is payload
    assert initial.payload["scan_to_map"]["translation"] == [10, 20, 0]
    assert (obj.x, obj.y) == (12, 20)
    assert "payload" not in memory.serialize()["views"][0]
    json.dumps(memory.serialize(), allow_nan=False)


def test_map_coordinates_fuse_independent_views_but_not_repeated_descriptions():
    memory = Memory()
    first = view(memory)
    obj = memory.remember(first, "chair", BOX, 10, 20, "seat beside the doorway")
    same = memory.remember(first, "chair", BOX, 10.1, 20.1, "the chair")
    assert same is obj
    assert obj.observations == 1
    assert (obj.x, obj.y) == (10, 20)
    assert len(first.observations) == 1
    second = view(memory, 2, x=1)
    assert memory.remember(second, "chair", BOX, 10.2, 20.2) is obj
    assert obj.observations == 2
    assert (obj.x, obj.y) == pytest.approx((10.1, 20.1))
    assert obj.view_ids == {first.view_id, second.view_id}
    assert obj.descriptions == {"seat beside the doorway", "the chair"}


def test_same_view_identity_survives_large_map_coordinate_correction():
    memory = Memory()
    first = view(memory)
    original = memory.remember(first, "chair", BOX, 1, 2)
    recovered = memory.remember(first, "object", [12, 22, 99, 149], 4, 6, "the seat")
    assert recovered is original
    assert original.object_id == "chair_0" and original.observations == 1
    assert len(memory.landmarks) == 1 and len(first.observations) == 1


def test_unknown_label_reuses_known_instance_and_known_classes_remain_distinct():
    memory = Memory()
    first = view(memory)
    chair = memory.remember(first, "chair", BOX, 1, 2)
    assert memory.remember(first, "object", BOX, 1.05, 2.05) is chair
    assert chair.label == "chair"
    person = memory.remember(first, "person", [110, 20, 200, 150], 1, 2)
    assert person is not chair
    distant = memory.remember(first, "chair", [220, 20, 310, 150], 3, 2)
    assert distant.object_id == "chair_1"


def test_unknown_instance_can_gain_label_without_changing_its_id():
    memory = Memory()
    first = view(memory)
    obj = memory.remember(first, "object", BOX, 1, 2)
    assert memory.remember(first, "chair", BOX, 1, 2) is obj
    assert obj.label == "chair"
    assert memory.lookup_id("go to object 0") is obj
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("chair_0")


def test_separate_same_view_instances_do_not_merge_on_map_proximity():
    memory = Memory()
    first = view(memory)
    left = memory.remember(first, "chair", BOX, 1, 2)
    right = memory.remember(first, "chair", [110, 20, 200, 150], 1.1, 2)
    assert left.object_id == "chair_0"
    assert right.object_id == "chair_1"
    assert memory.match_box(first, [115, 25, 195, 145]) is right


@pytest.mark.parametrize("text", ["go to chair 1", "chair_1", "chair1", "chair number 1",
                                  "Please move to the chair_1.", "drive to chair 1!",
                                  "go to a chair_1", "a chair_1", "Navigate to a chair 1!"])
def test_explicit_ids_use_exact_instance(text):
    memory = Memory()
    first = view(memory)
    memory.remember(first, "chair", BOX, 0, 0)
    second = memory.remember(first, "chair", [110, 20, 200, 150], 3, 0)
    assert memory.lookup_id(text) is second
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("go to chair 99")
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("go to a chair 99")
    assert memory.lookup_id("go to the chair beside a table") is None


@pytest.mark.parametrize("label,text", [("person", "person0"), ("trash_can", "trash can 0"),
                                        ("trash_can", "trash_can_0"), ("object", "object_0"),
                                        ("object", "Please move to an object_0.")])
def test_all_supported_id_categories(label, text):
    memory = Memory()
    obj = memory.remember(view(memory), label, BOX, 1, 2)
    assert memory.lookup_id(text) is obj


def test_indefinite_article_forklift_command_selects_exact_id():
    memory = Memory(class_labels=["forklift"])
    source = view(memory)
    memory.remember(source, "forklift", BOX, 0, 0)
    target = memory.remember(source, "forklift", [110, 20, 200, 150], 3, 0)
    assert target.object_id == "forklift_1"
    assert memory.lookup_id("go to a forklift_1") is target
    assert memory.lookup_id("go to a forklift") is None


@pytest.mark.parametrize("text", ["chair -1", "chair 1.2", "chair 0 or person 0",
    "do not go to chair0, go to the blue chair", "go to the chair right of chair0",
    "chair0 beside the door", "go to red chair0", "not chair0",
    "do not go to a chair_0", "go to a chair_0 or person_0", "go to a red chair_0",
    "go to an object_0 beside the door"])
def test_invalid_id_is_distinguishable_from_normal_description(text):
    with pytest.raises(InvalidObjectReference):
        Memory().lookup_id(text)


def test_view_pose_dedup_wraps_yaw_and_preserves_grounded_old_payload():
    memory = Memory()
    assert memory.should_capture(0, 0, math.pi - .05, SECOND)
    first = view(memory, yaw=math.pi - .05, payload={"frame": "initial"})
    assert not memory.should_capture(.1, 0, -math.pi + .05, 2 * SECOND)
    memory.remember(first, "chair", BOX, 2, 0)
    returned = view(memory, 3, x=.1, yaw=-math.pi + .05, payload={"frame": "occluded"})
    assert returned is first and first.payload["frame"] == "initial"
    assert memory.should_capture(1, 0, 0, 4 * SECOND)


def test_ungrounded_duplicate_can_be_replaced_and_views_are_bounded():
    memory = Memory(max_views=2)
    empty = view(memory)
    clear = view(memory, 2, x=.1)
    assert empty not in memory.views
    memory.remember(clear, "chair", BOX, 2, 0)
    discardable = view(memory, 3, x=2)
    newest = view(memory, 4, x=4)
    assert memory.views == [clear, newest]
    assert discardable not in memory.views
    assert len(memory.views) == 2
    with pytest.raises(ValueError, match="evicted"):
        memory.remember(discardable, "object", BOX, 9, 9)


def test_expiration_clears_views_and_instances_without_reusing_old_id():
    memory = Memory(max_age_sec=3)
    first = view(memory)
    memory.remember(first, "chair", BOX, 1, 1)
    assert memory.views_for_query(4 * SECOND) == [first]
    assert memory.views_for_query(4 * SECOND + 1) == []
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("chair0")
    obj = memory.remember(view(memory, 5), "chair", BOX, 1, 1)
    assert obj.object_id == "chair_1"


def test_clock_rollback_clears_map_session_and_rejects_old_view_reference():
    memory = Memory()
    old = view(memory, 50)
    memory.remember(old, "chair", BOX, 1, 2)
    memory.views_for_query(52 * SECOND)
    assert memory.should_capture(0, 0, 0, SECOND)
    assert memory.landmarks == {} and memory.views == []
    new = view(memory, 1)
    with pytest.raises(ValueError, match="another map session"):
        memory.remember(old, "chair", BOX, 1, 2)
    assert memory.remember(new, "chair", BOX, 8, 9).object_id == "chair_0"


def test_query_clock_rollback_clears_memory_but_normal_sensor_delay_does_not():
    memory = Memory()
    first = view(memory, 10)
    memory.remember(first, "chair", BOX, 1, 1)
    memory.views_for_query(11 * SECOND)
    memory.should_capture(2, 0, 0, int(10.9 * SECOND))
    assert first in memory.views
    assert memory.views_for_query(SECOND) == []
    assert memory.landmarks == {}


def test_query_priority_is_object_count_then_recency():
    memory = Memory()
    rich = view(memory)
    memory.remember(rich, "chair", BOX, 1, 1)
    memory.remember(rich, "person", [110, 20, 200, 150], 4, 1)
    newer = view(memory, 2, x=2)
    latest = view(memory, 3, x=4)
    assert memory.views_for_query(4 * SECOND) == [rich, latest, newer]


def test_same_pixels_with_conflicting_classes_keep_one_unusable_physical_track():
    memory = Memory()
    first = view(memory)
    obj = memory.remember(first, "chair", BOX, 1, 1)
    assert memory.match_box(first, [15, 25, 95, 145]) is obj
    assert memory.match_box(first, [200, 200, 250, 250]) is None
    assert memory.remember(first, "person", BOX, 1, 1) is obj
    assert memory.match_box(first, BOX) is obj
    assert len(memory.landmarks) == 1
    assert obj.conflicted and obj.label_votes == {"chair": 1, "person": 1}
    with pytest.raises(InvalidObjectReference, match="conflicting"):
        memory.lookup_id("chair_0")
    with pytest.raises(InvalidObjectReference, match="conflicting"):
        memory.lookup_id("go to a chair_0")


def test_surface_overlap_associates_large_object_despite_center_shift():
    memory = Memory()
    first = view(memory)
    obj = memory.remember(first, "forklift", BOX, 1, 1,
                          support_points=[(1., 1.), (1.5, 1.)])
    second = view(memory, 2, x=1)
    same = memory.remember(second, "forklift", BOX, 1.9, 1.2,
                           support_points=[(1.6, 1.), (1.9, 1.2)])
    assert same is obj and obj.observations == 2
    assert len(memory.landmarks) == 1
    assert (1., 1.) in obj.support_points and (1.9, 1.2) in obj.support_points


def test_disjoint_surfaces_do_not_fall_back_to_nearby_map_centers():
    memory = Memory()
    first = memory.remember(view(memory), "forklift", BOX, 1, 1,
                            support_points=[(10., 10.)])
    second = memory.remember(view(memory, 2, x=1), "forklift", BOX, 1.01, 1,
                             support_points=[(11., 10.)])
    assert first is not second and len(memory.landmarks) == 2


def test_missing_surface_evidence_preserves_legacy_point_association():
    memory = Memory()
    first = memory.remember(view(memory), "chair", BOX, 1, 1,
                            support_points=[(10., 10.)])
    assert memory.remember(view(memory, 2, x=1), "chair", BOX, 1.1, 1) is first


def test_same_frame_separate_boxes_cannot_link_even_with_adjacent_support():
    memory = Memory()
    first = view(memory)
    left = memory.remember(first, "forklift", BOX, 1, 1, support_points=[(1., 1.)])
    right = memory.remember(first, "forklift", [110, 20, 200, 150], 1.1, 1,
                            support_points=[(1.1, 1.)])
    assert left is not right
    later = view(memory, 2, x=1)
    with pytest.raises(ValueError, match="multiple physical tracks"):
        memory.remember(later, "forklift", BOX, 1.05, 1, support_points=[(1.05, 1.)])
    assert len(memory.landmarks) == 2 and later.observations == []


def test_cross_class_surface_match_keeps_original_class_despite_new_majority():
    memory = Memory(class_labels=["forklift", "storage rack"])
    first = view(memory)
    obj = memory.remember(first, "forklift", BOX, 1, 1, support_points=[(1., 1.)])
    assert memory.remember(first, "storage rack", BOX, 1, 1,
                           support_points=[(1., 1.)]) is obj
    for _ in range(5):
        memory.remember(first, "storage rack", BOX, 1, 1, support_points=[(1., 1.)])
    assert obj.conflicted and obj.label_votes == {"forklift": 1, "storage_rack": 1}
    second = view(memory, 2, x=1)
    assert memory.remember(second, "storage rack", BOX, 2, 1,
                           support_points=[(1.1, 1.)]) is obj
    assert obj.conflicted and obj.label == "forklift"
    assert obj.object_id == "forklift_0"  # Track references remain stable.
    with pytest.raises(InvalidObjectReference, match="clear memory and re-observe"):
        memory.lookup_id("forklift_0")
    assert obj.observations == 2
    record = memory.serialize()["landmarks"][0]
    assert record["evidence"] == {"label_votes": {"forklift": 1, "storage_rack": 2},
                                   "seen_known_labels": ["forklift", "storage_rack"],
                                   "retained_views": 2, "cannot_link_ids": []}
    assert record["conflicted"] is True and len(record["support_points"]) == 2
    json.dumps(memory.serialize(), allow_nan=False)


def test_repeated_views_cannot_vote_away_a_known_class_conflict():
    memory = Memory()
    first = view(memory)
    obj = memory.remember(first, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    memory.remember(first, "person", BOX, 1, 1, support_points=[(1., 1.)])
    for stamp, label in ((2, "chair"), (3, "person"), (4, "chair")):
        memory.remember(view(memory, stamp, x=stamp), label, BOX, 1, 1,
                        support_points=[(1., 1.)])
    assert obj.label_votes == {"chair": 3, "person": 2} and obj.conflicted
    memory.remember(view(memory, 5, x=5), "chair", BOX, 1, 1, support_points=[(1., 1.)])
    assert obj.label_votes == {"chair": 4, "person": 2} and obj.conflicted
    assert obj.label == "chair" and obj.seen_known_labels == {"chair", "person"}


def test_overlapping_different_objects_do_not_merge_without_high_overlap():
    memory = Memory()
    first = view(memory)
    chair = memory.remember(first, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    person = memory.remember(first, "person", [70, 20, 160, 150], 1, 1,
                             support_points=[(1., 1.)])
    assert chair is not person and not chair.conflicted and not person.conflicted


def test_surface_history_is_bounded_spatially_sampled_and_deduplicated():
    memory = Memory()
    first = view(memory)
    points = [(i / 100, 0.) for i in range(201)]
    obj = memory.remember(first, "forklift", BOX, 1, 1, support_points=points)
    assert len(obj.support_points) == 64
    assert min(x for x, y in obj.support_points) == 0.
    assert max(x for x, y in obj.support_points) == pytest.approx(2., abs=.02)
    before = list(obj.support_points)
    memory.remember(first, "forklift", BOX, 1, 1, support_points=points)
    assert obj.support_points == before and obj.observations == 1


@pytest.mark.parametrize("points", [[(1, 2, 3)], [(float("nan"), 2)], [(1, float("inf"))]])
def test_invalid_surface_evidence_does_not_create_an_instance(points):
    memory = Memory()
    with pytest.raises(ValueError, match="finite odom XY"):
        memory.remember(view(memory), "chair", BOX, 1, 1, support_points=points)
    assert memory.landmarks == {}


def test_complete_surface_view_merges_fragments_without_covisible_cannot_link():
    memory = Memory(class_labels=["forklift"])
    first = view(memory)
    original = memory.remember(first, "forklift", BOX, 1, 1,
                               support_points=[(1., 1.)], description="first face")
    second = view(memory, 2, x=1)
    fragment = memory.remember(second, "forklift", BOX, 1.8, 1,
                               support_points=[(1.8, 1.)], description="second face")
    assert fragment is not original
    complete = view(memory, 3, x=2)
    joined = memory.remember(complete, "forklift", BOX, 1.4, 1,
                             support_points=[(1.1, 1.), (1.4, 1.), (1.7, 1.)])
    assert joined is original and list(memory.landmarks) == ["forklift_0"]
    assert joined.observations == 3
    assert joined.label_votes == {"forklift": 3}
    assert joined.descriptions == {"first face", "second face"}
    assert all(o["object_id"] == "forklift_0" for v in memory.views for o in v.observations)
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("forklift_1")
    distinct = memory.remember(complete, "forklift", [110, 20, 200, 150], 5, 5,
                               support_points=[(5., 5.)])
    assert distinct.object_id == "forklift_2"


def test_surface_fragment_merge_preserves_cross_class_votes_and_blocks_navigation():
    memory = Memory(class_labels=["forklift", "storage rack"])
    original = memory.remember(view(memory), "forklift", BOX, 1, 1,
                               support_points=[(1., 1.)])
    memory.remember(view(memory, 2, x=1), "storage rack", BOX, 1.8, 1,
                    support_points=[(1.8, 1.)])
    joined = memory.remember(view(memory, 3, x=2), "object", BOX, 1.4, 1,
                             support_points=[(1.1, 1.), (1.7, 1.)])
    assert joined is original and len(memory.landmarks) == 1
    assert joined.label_votes == {"forklift": 1, "storage_rack": 1}
    assert joined.conflicted
    with pytest.raises(InvalidObjectReference, match="conflicting"):
        memory.lookup_id("forklift_0")


@pytest.mark.parametrize("second_label,second_box", [
    ("chair", [50, 20, 140, 150]), ("person", BOX),
])
def test_same_view_pixel_overlap_cannot_override_disjoint_surface_geometry(second_label, second_box):
    memory = Memory()
    first = view(memory)
    a = memory.remember(first, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    b = memory.remember(first, second_label, second_box, 1, 1, support_points=[(3., 1.)])
    assert a is not b
    assert a.support_points == [(1., 1.)] and b.support_points == [(3., 1.)]
    assert a.cannot_link == {b.object_id} and b.cannot_link == {a.object_id}


def test_cannot_link_survives_eviction_of_original_covisible_frame():
    memory = Memory(max_age_sec=3)
    initial = view(memory)
    a = memory.remember(initial, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    b = memory.remember(initial, "chair", [110, 20, 200, 150], 2, 1,
                        support_points=[(2., 1.)])
    memory.remember(view(memory, 3, x=1), "chair", BOX, 1, 1, support_points=[(1., 1.)])
    memory.remember(view(memory, 4, x=2), "chair", BOX, 2, 1, support_points=[(2., 1.)])
    broad = view(memory, 5, x=3)
    assert initial not in memory.views and len(memory.landmarks) == 2
    with pytest.raises(ValueError, match="multiple physical tracks"):
        memory.remember(broad, "chair", BOX, 1.5, 1, support_points=[(1.1, 1.), (1.9, 1.)])
    assert set(memory.landmarks) == {a.object_id, b.object_id}


def test_pruning_dead_track_removes_its_cannot_link_references():
    memory = Memory(max_age_sec=3)
    initial = view(memory)
    a = memory.remember(initial, "chair", BOX, 1, 1)
    memory.remember(initial, "chair", [110, 20, 200, 150], 2, 1)
    memory.remember(view(memory, 3, x=1), "chair", BOX, 1, 1)
    memory.views_for_query(5 * SECOND)
    assert len(memory.landmarks) == 1 and a.cannot_link == set()


def test_evicted_last_observation_removes_ghost_without_reusing_its_id():
    memory = Memory(max_views=1)
    first = view(memory)
    memory.remember(first, "chair", BOX, 1, 1)
    replacement = view(memory, 2, x=1)
    assert memory.serialize()["landmarks"] == []
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("chair_0")
    assert memory.remember(replacement, "chair", BOX, 1, 1).object_id == "chair_1"


def test_eviction_preserves_unique_instance_before_redundant_rich_view():
    memory = Memory(max_views=3)
    unique = view(memory)
    extinguisher = memory.remember(unique, "fire extinguisher", BOX, 8, 8)
    redundant = []
    for stamp in (2, 3):
        rich = view(memory, stamp, x=stamp)
        memory.remember(rich, "chair", BOX, 1, 1)
        memory.remember(rich, "person", [110, 20, 200, 150], 3, 1)
        redundant.append(rich)
    view(memory, 4, x=4)
    assert unique in memory.views and redundant[0] not in memory.views
    assert extinguisher.view_ids == {unique.view_id}
    assert all(obj.view_ids for obj in memory.landmarks.values())


def test_label_votes_expire_but_class_conflict_survives_source_expiration():
    memory = Memory(max_age_sec=3)
    first = view(memory)
    obj = memory.remember(first, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    memory.remember(first, "person", BOX, 1, 1, support_points=[(1., 1.)])
    later = view(memory, 3, x=3)
    memory.remember(later, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    memory.views_for_query(5 * SECOND)
    assert obj.view_ids == {later.view_id}
    assert obj.label_votes == {"chair": 1} and obj.conflicted
    assert obj.seen_known_labels == {"chair", "person"}


def test_eviction_protects_one_source_for_each_conflicting_class():
    memory = Memory(max_views=4)
    original = view(memory)
    obj = memory.remember(original, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    others = []
    for stamp in (2, 3, 4):
        source = view(memory, stamp, x=stamp)
        others.append(source)
        memory.remember(source, "person", BOX, 1, 1, support_points=[(1., 1.)])
    view(memory, 5, x=5)
    assert original in memory.views and others[0] not in memory.views
    assert obj.conflicted and obj.label_votes == {"chair": 1, "person": 2}


def test_capacity_loss_of_original_class_image_does_not_clear_conflict():
    memory = Memory(max_views=2)
    original = view(memory)
    obj = memory.remember(original, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    memory.remember(view(memory, 2, x=2), "person", BOX, 1, 1, support_points=[(1., 1.)])
    newer = view(memory, 3, x=3)
    assert original not in memory.views
    memory.remember(newer, "person", BOX, 1, 1, support_points=[(1., 1.)])
    assert obj.label == "chair" and obj.object_id == "chair_0"
    assert obj.label_votes == {"person": 2}
    assert obj.seen_known_labels == {"chair", "person"} and obj.conflicted
    with pytest.raises(InvalidObjectReference, match="clear memory"):
        memory.lookup_id("chair_0")


def test_merge_preserves_conflict_whose_original_source_already_expired():
    memory = Memory(max_age_sec=3)
    original_view = view(memory)
    original = memory.remember(original_view, "chair", BOX, 1, 1, support_points=[(1., 1.)])
    memory.remember(original_view, "person", BOX, 1, 1, support_points=[(1., 1.)])
    memory.remember(view(memory, 3, x=3), "chair", BOX, 1, 1, support_points=[(1., 1.)])
    fragment = memory.remember(view(memory, 4, x=4), "chair", BOX, 1.8, 1,
                               support_points=[(1.8, 1.)])
    assert fragment is not original
    joined = memory.remember(view(memory, 5, x=5), "chair", BOX, 1.4, 1,
                             support_points=[(1.1, 1.), (1.7, 1.)])
    assert original_view not in memory.views and joined is original
    assert joined.label_votes == {"chair": 3}
    assert joined.seen_known_labels == {"chair", "person"} and joined.conflicted


def test_clear_or_complete_expiration_removes_class_conflict_with_track():
    memory = Memory(max_age_sec=2)
    first = view(memory)
    obj = memory.remember(first, "chair", BOX, 1, 1)
    memory.remember(first, "person", BOX, 1, 1)
    memory.views_for_query(4 * SECOND)
    assert not memory.landmarks and not memory.views
    fresh = memory.remember(view(memory, 5, x=5), "chair", BOX, 1, 1)
    assert fresh.object_id == "chair_1" and not fresh.conflicted
    assert fresh.seen_known_labels == {"chair"}
    memory.clear()
    newest = memory.remember(view(memory, 6, x=6), "person", BOX, 1, 1)
    assert not newest.conflicted and newest.seen_known_labels == {"person"}


def test_representative_views_keep_rich_context_and_drop_grounded_subsets():
    memory = Memory()
    rich = view(memory)
    memory.remember(rich, "chair", BOX, 1, 1)
    memory.remember(rich, "chair", [110, 20, 200, 150], 4, 1)
    closeup = view(memory, 2, x=2)
    memory.remember(closeup, "chair", BOX, 1, 1)
    primary, fallback = memory.representative_views(3 * SECOND)
    assert primary == [rich] and fallback == []


def test_representative_views_keep_complementary_object_coverage():
    memory = Memory()
    first = view(memory)
    memory.remember(first, "chair", BOX, 1, 1)
    memory.remember(first, "person", [110, 20, 200, 150], 3, 1)
    second = view(memory, 2, x=2)
    memory.remember(second, "chair", BOX, 1, 1)
    memory.remember(second, "trash_can", [110, 20, 200, 150], 5, 1)
    primary, fallback = memory.representative_views(3 * SECOND)
    assert primary == [second, first] and fallback == []


def test_representative_views_separate_new_ungrounded_fallback():
    memory = Memory()
    grounded = view(memory)
    memory.remember(grounded, "chair", BOX, 1, 1)
    empty = view(memory, 2, x=2)
    assert memory.representative_views(3 * SECOND) == ([grounded], [empty])


def test_representative_views_use_newest_fallback_when_none_grounded():
    memory = Memory()
    first = view(memory)
    newest = view(memory, 2, x=2)
    assert memory.representative_views(3 * SECOND) == ([], [newest, first])


def test_representative_views_share_one_budget_between_primary_and_fallback():
    memory = Memory()
    first = view(memory)
    memory.remember(first, "chair", BOX, 1, 1)
    second = view(memory, 2, x=2)
    memory.remember(second, "person", BOX, 4, 1)
    empty = view(memory, 3, x=4)
    assert memory.representative_views(4 * SECOND, max_views=2) == ([second, first], [])
    assert memory.representative_views(4 * SECOND, max_views=3) == ([second, first], [empty])
    with pytest.raises(ValueError, match="positive integer"):
        memory.representative_views(4 * SECOND, max_views=0)


def test_representative_views_expiration_and_clock_reset_drop_stale_coverage():
    memory = Memory(max_age_sec=3)
    old = view(memory, 10)
    memory.remember(old, "chair", BOX, 1, 1)
    recent = view(memory, 12, x=2)
    assert memory.representative_views(14 * SECOND) == ([], [recent])
    assert memory.representative_views(SECOND) == ([], [])
    assert memory.landmarks == {}


def test_configured_warehouse_class_ids_are_not_limited_to_course_classes():
    memory = Memory(class_labels=['forklift', 'fire extinguisher', 'shelf', 'table'])
    first = view(memory)
    extinguisher = memory.remember(first, 'fire extinguisher', BOX, 1, 2)
    assert memory.lookup_id('go to fire extinguisher 0') is extinguisher
    assert memory.lookup_id('fire_extinguisher_0') is extinguisher
    with pytest.raises(UnknownObjectId):
        memory.lookup_id('go to forklift 0')
    with pytest.raises(InvalidObjectReference):
        memory.lookup_id('do not go to fire extinguisher 0')
    memory.clear()
    with pytest.raises(UnknownObjectId):
        memory.lookup_id('go to table 0')


def confirming_memory(**kwargs):
    return Memory(confirm_classes=True, view_distance=.05,
                  class_labels=["forklift", "storage rack"], **kwargs)


def observed(memory, stamp, x, label="chair", yaw=0., support=(1., 1.)):
    source = view(memory, stamp, x=x, yaw=yaw)
    return memory.remember(source, label, BOX, 1., 1., support_points=[support])


def test_confirmation_is_opt_in_and_pending_tracks_have_no_public_class_id():
    legacy = Landmark("chair_0", "chair", 1., 1., 1, SECOND)
    assert legacy.confirmed is True
    memory = confirming_memory()
    obj = observed(memory, 1, 0.)
    assert obj.object_id == "candidate_0" and not obj.confirmed
    record = memory.serialize()
    assert record["landmarks"] == []
    assert record["pending_landmarks"][0]["confirmed"] is False
    assert record["pending_landmarks"][0]["evidence"]["independent_viewpoints"] == {
        "chair": [{"view_id": "view_0", "x": 0., "y": 0.}]}
    with pytest.raises(InvalidObjectReference, match="pending"):
        memory.lookup_id("go to candidate_0")
    with pytest.raises(InvalidObjectReference, match="pending"):
        memory.lookup_id("go to a candidate_0")
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("chair_0")
    assert memory.representative_views(SECOND) == ([], memory.views)
    assert "pending_landmarks" not in Memory().serialize()


def test_pure_rotation_repeated_frames_and_close_translation_cannot_confirm():
    memory = confirming_memory()
    obj = observed(memory, 1, 0.)
    for stamp, x, yaw in ((2, 0., .7), (3, 0., 1.4), (4, .1, 2.1), (5, .2, 2.8)):
        assert observed(memory, stamp, x, yaw=yaw) is obj
    for _ in range(5):
        memory.remember(memory.views[-1], "chair", BOX, 1., 1., support_points=[(1., 1.)])
    assert not obj.confirmed and obj.label_votes == {"chair": 1}
    assert obj.object_id == "candidate_0"


def test_two_translated_agreeing_views_promote_and_rename_all_original_evidence():
    memory = confirming_memory()
    obj = observed(memory, 1, 0.)
    assert observed(memory, 2, .25) is obj
    assert obj.confirmed and obj.object_id == "chair_0" and obj.label_votes == {"chair": 2}
    assert set(memory.landmarks) == {"chair_0"}
    assert all(o["object_id"] == "chair_0" for v in memory.views for o in v.observations)
    assert memory.serialize()["pending_landmarks"] == []
    assert memory.lookup_id("chair_0") is obj
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("candidate_0")


def test_one_initial_wrong_class_then_three_independent_correct_views_promote_correct_id():
    memory = confirming_memory()
    obj = observed(memory, 1, 0., "storage rack")
    assert observed(memory, 2, .4, "forklift") is obj
    assert observed(memory, 3, .8, "forklift") is obj
    assert not obj.confirmed and not obj.conflicted
    assert obj.label_votes == {"storage_rack": 1, "forklift": 2}
    assert observed(memory, 4, 1.2, "forklift") is obj
    assert obj.confirmed and obj.label == "forklift" and obj.object_id == "forklift_0"
    assert obj.seen_known_labels == {"storage_rack", "forklift"} and not obj.conflicted
    assert memory.lookup_id("forklift_0") is obj
    with pytest.raises(UnknownObjectId):
        memory.lookup_id("storage_rack_0")
    assert all(o["object_id"] == "forklift_0" for v in memory.views for o in v.observations)


def test_confirmed_track_audits_one_bad_view_and_quarantines_two_independent_bad_views():
    memory = confirming_memory()
    obj = observed(memory, 1, 0.)
    observed(memory, 2, .4)
    assert obj.confirmed
    observed(memory, 3, .8, "person")
    observed(memory, 4, .8, "person", yaw=1.)
    assert obj.label_votes == {"chair": 2, "person": 1} and not obj.conflicted
    assert memory.lookup_id("chair_0") is obj
    observed(memory, 5, 1.2, "person")
    assert obj.conflicted and obj.label == "chair" and obj.object_id == "chair_0"
    with pytest.raises(InvalidObjectReference, match="conflicting"):
        memory.lookup_id("chair_0")


def test_promotion_with_two_contradictory_views_is_quarantined_before_any_refresh():
    memory = confirming_memory()
    labels = ["storage rack", "forklift", "storage rack", "forklift", "forklift"]
    for stamp, label in enumerate(labels, 1):
        obj = observed(memory, stamp, stamp * .4, label)
        assert not obj.confirmed
    assert observed(memory, 6, 2.4, "forklift") is obj
    assert obj.confirmed and obj.object_id == "forklift_0" and obj.conflicted
    assert obj.label_votes == {"storage_rack": 2, "forklift": 4}
    with pytest.raises(InvalidObjectReference, match="conflicting"):
        memory.lookup_id("forklift_0")
    before = memory.serialize()
    memory.views_for_query(7 * SECOND)
    assert memory.serialize() == before
    with pytest.raises(InvalidObjectReference, match="conflicting"):
        memory.lookup_id("forklift_0")


def test_pending_track_cannot_be_promoted_by_expiring_runner_up_or_repeating_old_evidence():
    memory = confirming_memory(max_age_sec=3)
    obj = observed(memory, 1, 0., "person")
    observed(memory, 2, 1., "chair")
    observed(memory, 3, 2., "chair")
    assert obj.label_votes == {"person": 1, "chair": 2} and not obj.confirmed
    memory.views_for_query(5*SECOND)
    assert obj.label_votes == {"chair": 2} and not obj.confirmed
    source = memory.views[-1]
    memory.remember(source, "chair", BOX, 1., 1., support_points=[(1., 1.)])
    assert not obj.confirmed and obj.object_id == "candidate_0"
    assert memory.serialize()["landmarks"] == []


def test_expiration_cannot_clear_a_confirmed_class_conflict():
    memory = confirming_memory(max_age_sec=3)
    obj = observed(memory, 1, 0.)
    observed(memory, 2, 1.)
    observed(memory, 3, 2., "person")
    observed(memory, 4, 3., "person")
    assert obj.conflicted
    observed(memory, 5, 4., "chair")
    memory.views_for_query(8*SECOND)
    assert obj.label_votes == {"chair": 1} and obj.conflicted
    assert obj.confirmed and obj.seen_known_labels == {"chair", "person"}


def test_promotion_preserves_covisible_cannot_links_and_rewrites_peer_ids():
    memory = confirming_memory()
    first = view(memory)
    a = memory.remember(first, "chair", BOX, 1., 1., support_points=[(1., 1.)])
    b = memory.remember(first, "chair", [110, 20, 200, 150], 2., 1., support_points=[(2., 1.)])
    assert a.cannot_link == {"candidate_1"} and b.cannot_link == {"candidate_0"}
    second = view(memory, 2, x=1.)
    assert memory.remember(second, "chair", BOX, 1., 1., support_points=[(1., 1.)]) is a
    assert a.object_id == "chair_0" and b.cannot_link == {"chair_0"}
    memory.remember(second, "chair", [110, 20, 200, 150], 2., 1., support_points=[(2., 1.)])
    assert b.object_id == "chair_1" and a.cannot_link == {"chair_1"}
    assert b.cannot_link == {"chair_0"}
    later = view(memory, 3, x=2.)
    with pytest.raises(ValueError, match="multiple physical tracks"):
        memory.remember(later, "chair", BOX, 1.5, 1., support_points=[(1.1, 1.), (1.9, 1.)])


def test_merging_an_older_pending_fragment_preserves_the_confirmed_public_id():
    memory = confirming_memory()
    old = observed(memory, 1, 0., support=(1., 1.))
    public = observed(memory, 2, 1., support=(2., 1.))
    assert old is not public
    assert observed(memory, 3, 2., support=(2., 1.)) is public
    assert public.object_id == "chair_0" and public.confirmed
    joined = memory.remember(view(memory, 4, x=3.), "chair", BOX, 1.5, 1.,
                             support_points=[(1.1, 1.), (1.9, 1.)])
    assert joined is public and set(memory.landmarks) == {"chair_0"}
    assert all(o["object_id"] == "chair_0" for v in memory.views for o in v.observations)


def test_candidate_counter_is_global_and_canonical_counters_only_advance_on_promotion():
    memory = confirming_memory(max_views=1)
    a = observed(memory, 1, 0., "chair")
    b = observed(memory, 2, 1., "person")
    assert a.object_id == "candidate_0" and b.object_id == "candidate_1"
    assert memory._next_object == {}
    memory.clear()
    c = observed(memory, 3, 0., "person")
    assert c.object_id == "candidate_0"


def quality_observation(memory, stamp, x, label="chair", complete=True, yaw=0.):
    source = view(memory, stamp, x=x, yaw=yaw)
    box = BOX if complete else [10, 0, 100, 150]
    return source, memory.remember(source, label, box, 1., 2.,
        support_points=[[1., 2.]], class_evidence_complete=complete)


def test_complete_confirmed_class_ignores_only_truncated_competitors_but_keeps_raw_evidence():
    memory = Memory(confirm_classes=True)
    quality_observation(memory, 1, 0.)
    _, obj = quality_observation(memory, 2, .8)
    assert obj.confirmed and obj.confirmed_class_complete
    quality_observation(memory, 3, 1.6, "table", False)
    quality_observation(memory, 4, 2.4, "table", False)
    assert obj.label_votes == {"chair": 2, "table": 2}
    assert obj.conflict_votes == {} and not obj.conflicted and obj.object_id == "chair_0"
    record = memory.serialize()["landmarks"][0]
    assert record["evidence"]["seen_known_labels"] == ["chair", "table"]
    assert record["evidence"]["confirmed_class_evidence"]["view_id"] == "view_0"
    assert memory.serialize()["views"][-1]["observations"][0]["label_completeness"] == {"table": False}


def test_truncated_initial_class_still_confirms_with_two_independent_views():
    memory = Memory(confirm_classes=True, class_labels=["storage rack"])
    quality_observation(memory, 1, 0., "storage rack", False)
    _, obj = quality_observation(memory, 2, .8, "storage rack", False)
    assert obj.confirmed and obj.object_id == "storage_rack_0"
    assert not obj.confirmed_class_complete and not obj.conflicted
    quality_observation(memory, 3, 1.6, "forklift", False)
    quality_observation(memory, 4, 2.4, "forklift", False)
    assert obj.conflict_votes == {"forklift": 2} and obj.conflicted


def test_two_complete_competitors_keep_existing_sticky_conflict_rule():
    memory = Memory(confirm_classes=True)
    quality_observation(memory, 1, 0.)
    _, obj = quality_observation(memory, 2, .8)
    quality_observation(memory, 3, 1.6, "table", True)
    quality_observation(memory, 4, 2.4, "table", False)
    assert obj.conflict_votes == {"table": 1} and not obj.conflicted
    quality_observation(memory, 5, 3.2, "table", True)
    assert obj.conflict_votes == {"table": 2} and obj.conflicted
    quality_observation(memory, 6, 4., "chair", True)
    assert obj.conflicted and obj.label == "chair"


def test_quality_filter_precedes_independent_viewpoint_selection():
    memory = Memory(confirm_classes=True)
    quality_observation(memory, 1, 0.)
    _, obj = quality_observation(memory, 2, .8)
    quality_observation(memory, 3, 1.6, "table", False)
    quality_observation(memory, 4, 1.7, "table", True, yaw=1.)
    quality_observation(memory, 5, 2.4, "table", True)
    assert obj.conflict_votes == {"table": 2} and obj.conflicted


def test_expiration_cannot_downgrade_complete_class_protection():
    # TTL forces original strong pictures out regardless of the retention
    # policy; later weak pictures keep the physical track alive.
    memory = Memory(confirm_classes=True, max_views=32, max_age_sec=2)
    first, _ = quality_observation(memory, 1, 0.)
    second, obj = quality_observation(memory, 2, .8)
    audit = dict(obj.confirmed_class_evidence)
    for stamp in range(3, 7):
        quality_observation(memory, stamp, .8*(stamp-1), "table", False)
    assert first not in memory.views and second not in memory.views
    assert obj.confirmed_class_complete and obj.confirmed_class_evidence == audit
    assert obj.label_votes == {"table": 3} and not obj.conflicted
    quality_observation(memory, 7, 4.8, "table", True)
    quality_observation(memory, 8, 5.6, "table", True)
    assert obj.conflicted


def test_confirmed_weak_class_can_gain_quality_without_reconfirming():
    memory = Memory(confirm_classes=True)
    quality_observation(memory, 1, 0., complete=False)
    _, obj = quality_observation(memory, 2, .8, complete=False)
    assert obj.confirmed and not obj.confirmed_class_complete
    quality_observation(memory, 3, 1.6, complete=True)
    assert obj.confirmed_class_complete
    quality_observation(memory, 4, 2.4, "table", False)
    quality_observation(memory, 5, 3.2, "table", False)
    assert not obj.conflicted


def test_quality_upgrade_in_same_frame_does_not_add_vote_or_promote():
    memory = Memory(confirm_classes=True)
    source, obj = quality_observation(memory, 1, 0., complete=False)
    for complete in [True, False, True]:
        same = memory.remember(source, "chair", BOX, 1., 2., support_points=[[1., 2.]],
                               class_evidence_complete=complete)
        assert same is obj and obj.label_votes == {"chair": 1} and not obj.confirmed
    quality_observation(memory, 2, .8, complete=False)
    assert obj.confirmed_class_complete
    assert obj.confirmed_class_evidence["box"] == BOX


def test_quality_audit_is_cleared_by_clock_rollback():
    memory = Memory(confirm_classes=True)
    quality_observation(memory, 10, 0.)
    _, old = quality_observation(memory, 11, .8)
    assert old.confirmed_class_complete
    quality_observation(memory, 1, 0., complete=False)
    _, fresh = quality_observation(memory, 2, .8, complete=False)
    assert fresh is not old and fresh.confirmed and not fresh.confirmed_class_complete
    assert len(memory.landmarks) == 1


def test_legacy_memory_preserves_conflict_rules_even_for_weak_evidence():
    memory = Memory(confirm_classes=False)
    _, obj = quality_observation(memory, 1, 0., complete=True)
    quality_observation(memory, 2, .8, "table", False)
    assert obj.conflicted and obj.object_id == "chair_0"


def test_merged_confirmed_fragments_preserve_evicted_complete_quality():
    memory = Memory(confirm_classes=True)
    first, a = quality_observation(memory, 1, 0., complete=False)
    quality_observation(memory, 2, .8, complete=False)
    s = view(memory, 3, x=1.6)
    b = memory.remember(s, "chair", BOX, 4., 2., support_points=[[4., 2.]])
    s2 = view(memory, 4, x=2.4)
    memory.remember(s2, "chair", BOX, 4., 2., support_points=[[4., 2.]])
    assert a.confirmed and b.confirmed_class_complete
    audit = dict(b.confirmed_class_evidence)
    # Remove only the original stronger frame; another view still owns b.
    memory.views.remove(s)
    joined = memory._merge_tracks([a, b])
    assert joined is a and joined.confirmed_class_complete
    assert joined.confirmed_class_evidence == audit


@pytest.mark.parametrize("box,expected", [
    ([4, 3, 636, 477], True), ([4, 0, 636, 477], False),
    ([4, 3, 636, 480], False), ([3.2, 3, 636, 477], False),
    ([4, 2.4, 636, 477], False), ([4, 3, 636.8, 477], False),
    ([4, 3, 636, 477.6], False),
])
def test_class_box_completeness_uses_geometry_edge_margin(box, expected):
    from tb3_locateanything.visual_memory import class_box_complete
    assert class_box_complete(box, 640, 480) is expected


@pytest.mark.parametrize("width,height", [(0, 480), (640, float('nan')), (float('inf'), 480)])
def test_class_completeness_rejects_invalid_calibration_dimensions(width, height):
    from tb3_locateanything.visual_memory import class_box_complete
    with pytest.raises(ValueError, match="dimensions"):
        class_box_complete(BOX, width, height)


def test_complete_evidence_for_another_label_cannot_protect_weak_winning_class():
    memory = Memory(confirm_classes=True)
    quality_observation(memory, 1, 0., "table", True)
    quality_observation(memory, 2, .8, "chair", False)
    quality_observation(memory, 3, 1.6, "chair", False)
    _, obj = quality_observation(memory, 4, 2.4, "chair", False)
    assert obj.confirmed and obj.label == "chair" and not obj.confirmed_class_complete
    quality_observation(memory, 5, 3.2, "table", True)
    assert obj.conflicted and obj.conflict_votes == {"table": 2}


def test_same_frame_different_labels_keep_distinct_quality_and_source_boxes():
    memory = Memory(confirm_classes=True)
    source, obj = quality_observation(memory, 1, 0., "chair", True)
    truncated = [10, 0, 100, 150]
    same = memory.remember(source, "table", truncated, 1., 2.,
                           support_points=[[1., 2.]], class_evidence_complete=False)
    assert same is obj
    observation = memory.serialize()["views"][0]["observations"][0]
    assert observation["label_completeness"] == {"chair": True, "table": False}
    assert observation["label_boxes"] == {"chair": BOX, "table": truncated}
    assert obj.label_votes == {"chair": 1, "table": 1} and not obj.confirmed
