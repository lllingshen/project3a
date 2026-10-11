"""Multiple semantic matches must reach deterministic, model-bounded selection.

These exercise real query callbacks, gallery identity checks, and TF geometry.
The model response and historical range measurement are controlled; no model,
ROS graph, simulator, or navigation action runs in this test module.
"""
import math
from types import SimpleNamespace

import pytest

from tb3_locateanything.geometry import Vector3
from tb3_locateanything.visual_memory import Memory
from test_visual_memory_node import (
    active_query, complete_gallery, gallery_answer, module, node, odom,
    stored_view, transform,
)


SOURCE_BOX = [150., 100., 450., 400.]
COMMAND = "go to the object that can store cargo"


def scene(node, monkeypatch, objects):
    """Retain independently identifiable boxes with known odometry geometry."""
    node.memory = Memory(max_views=32, class_labels=sorted({label for label, _, _ in objects}))
    points, tracks = {}, []
    for index, (label, x, y) in enumerate(objects):
        view = stored_view(node, 10 + index, pose=(float(index), 0., 0.))
        obj = node.memory.remember(view, label, SOURCE_BOX, x, y)
        tracks.append(obj)
        points[id(view.payload["scan"].ranges)] = Vector3(x, y, 0.)
    assert len(node.memory.landmarks) == len(objects)
    node._refresh_positions = lambda: None
    monkeypatch.setattr(module, "localize_box",
                        lambda *args: (points[id(args[4])], {"range_m": 2.}))
    return tracks


def begin(node, command=COMMAND):
    active_query(node, command, views=None)
    node._tick()
    assert node.job is not None and node.job["kind"] == "gallery"
    assert not node.result_pub.messages
    request = node.inference_calls[-1][1]
    assert request["command"] == command
    assert request["expression"] == module.referring_expression(command)
    return node.job["gallery"]


def finish(node, gallery, object_ids):
    by_id = {entry.object_id: entry for entry in gallery.entries}
    answer = "".join(gallery_answer(gallery, by_id[oid].object_box) for oid in object_ids)
    complete_gallery(node, answer)
    node._tick()
    assert node.pending is None and len(node.result_pub.messages) == 1
    return node.result_pub.messages[0]


def test_nearest_choice_is_limited_to_model_matches_and_current_robot_pose(node, monkeypatch):
    closest, matched_far, matched_near = scene(node, monkeypatch, [
        ("fire_extinguisher", 8., 9.), ("storage_rack", 1., 1.),
        ("storage_rack", 9., 9.),
    ])
    gallery = begin(node)
    # Historical observation poses are near the distant rack. Current robot
    # pose is (8, 9), so only the newly computed distances can choose correctly.
    out = finish(node, gallery, [matched_far.object_id, matched_near.object_id])
    assert out.success and out.object_id == matched_near.object_id
    event = node.events[-1]
    assert event["selection_policy"] == "nearest_model_matches"
    assert set(event["model_candidate_ids"]) == {matched_far.object_id, matched_near.object_id}
    assert closest.object_id not in event["candidate_distances_m"]
    assert event["candidate_distances_m"][matched_near.object_id] == pytest.approx(1.)
    assert event["candidate_distances_m"][matched_far.object_id] == pytest.approx(math.hypot(7., 8.))


def test_model_matches_share_current_rotated_translated_map_alignment(node, monkeypatch):
    near, far = scene(node, monkeypatch, [
        ("storage_rack", 9., 9.), ("storage_rack", 8., 15.),
    ])
    current = transform(100, (20., 30., 0.))
    current.transform.rotation.z = math.sin(math.pi / 4)
    current.transform.rotation.w = math.cos(math.pi / 4)
    node.tf_buffer = SimpleNamespace(lookup_transform=lambda *args: current)
    original_localize, projections = node._localize, []

    def localize(view, box, projection=None):
        if projection is not None:
            projections.append(projection)
        return original_localize(view, box, projection)

    node._localize = localize
    gallery = begin(node)
    out = finish(node, gallery, [far.object_id, near.object_id])
    assert out.success and out.object_id == near.object_id
    assert (out.position.x, out.position.y) == pytest.approx((11., 39.))
    event = node.events[-1]
    assert event["candidate_distances_m"] == pytest.approx({near.object_id: 1., far.object_id: 6.})
    assert len(projections) >= 2 and all(projection is current for projection in projections)


def test_repeated_boxes_are_one_candidate_and_ties_use_stable_id(node, monkeypatch):
    first, second = scene(node, monkeypatch, [
        ("storage_rack", 7., 9.), ("storage_rack", 9., 9.),
    ])
    gallery = begin(node)
    out = finish(node, gallery, [second.object_id, second.object_id, first.object_id])
    assert out.success and out.object_id == min(first.object_id, second.object_id)
    event = node.events[-1]
    assert len(event["model_candidate_ids"]) == 2
    assert event["candidate_distances_m"] == pytest.approx({first.object_id: 1., second.object_id: 1.})


def test_single_model_match_is_not_replaced_by_nearest_memory_object(node, monkeypatch):
    close, chosen = scene(node, monkeypatch, [
        ("storage_rack", 8., 9.), ("storage_rack", 20., 20.),
    ])
    gallery = begin(node)
    out = finish(node, gallery, [chosen.object_id])
    assert out.success and out.object_id == chosen.object_id != close.object_id
    assert "nearest_model_matches" != node.events[-1].get("selection_policy")


@pytest.mark.parametrize("failure", ["stale", "missing", "wrong_frame", "invalid_rotation"])
def test_multiple_matches_never_publish_navigation_without_valid_current_tf(node, monkeypatch, failure):
    tracks = scene(node, monkeypatch, [("storage_rack", 7., 9.), ("storage_rack", 9., 9.)])
    gallery = begin(node)
    current = transform(90 if failure == "stale" else 100)
    if failure == "wrong_frame":
        current.child_frame_id = "different_odom"
    if failure == "invalid_rotation":
        current.transform.rotation.w = 0.

    def lookup(*args):
        if failure == "missing":
            raise module.tf2_ros.TransformException("no current map transform")
        return current

    node.tf_buffer = SimpleNamespace(lookup_transform=lookup)
    out = finish(node, gallery, [obj.object_id for obj in tracks])
    assert not out.success
    assert all(marker.action != marker.ADD for marker in node.marker_pub.messages)
    assert not any(event["state"] == "selected" for event in node.events)


@pytest.mark.parametrize("invalidation", ["timeout", "motion", "stale_odom"])
def test_nearest_selection_preserves_final_invalidation_checks(node, monkeypatch, invalidation):
    tracks = scene(node, monkeypatch, [("storage_rack", 7., 9.), ("storage_rack", 9., 9.)])
    gallery = begin(node)
    nearest = node._nearest_gallery_candidate

    def choose(choices):
        chosen = nearest(choices)
        if invalidation == "timeout":
            node.pending["wall_start"] -= 91
        elif invalidation == "motion":
            node.pending["robot_moved"] = True
        else:
            node.odoms[-1] = odom(90)
        return chosen

    node._nearest_gallery_candidate = choose
    out = finish(node, gallery, [obj.object_id for obj in tracks])
    assert not out.success
    assert all(marker.action != marker.ADD for marker in node.marker_pub.messages)
    assert not any(event["state"] == "selected" for event in node.events)


@pytest.mark.parametrize("suffix", [
    "<box><0><0><1001><900></box>", "<box><0><0><998><998></box>",
])
def test_multiple_match_selection_does_not_discard_invalid_model_boxes(node, monkeypatch, suffix):
    scene(node, monkeypatch, [("storage_rack", 7., 9.), ("storage_rack", 9., 9.)])
    gallery = begin(node)
    answer = "".join(gallery_answer(gallery, entry.object_box) for entry in gallery.entries) + suffix
    complete_gallery(node, answer)
    node._tick()
    assert node.pending is None and not node.result_pub.messages[0].success
    assert not any(event["state"] == "selected" for event in node.events)


@pytest.mark.parametrize("relation", [
    "near", "far", "nearer", "further", "closer", "nearest", "farthest", "closest", "furthest",
])
def test_distance_relation_uses_measured_current_robot_distance(node, monkeypatch, relation):
    near, far = scene(node, monkeypatch, [("storage_rack", 7., 9.), ("storage_rack", 12., 9.)])
    command = f"go to the {relation} storage rack"
    active_query(node, command, views=None)
    node._tick()
    out = node.result_pub.messages[-1]
    expected = far if relation in {"far", "further", "farthest", "furthest"} else near
    assert out.success and out.object_id == expected.object_id
    assert not node.inference_calls
    assert node.events[-1]["query_kind"] == "memory_relation"
    assert node.events[-1]["candidate_distances_m"] == {near.object_id: 1., far.object_id: 4.}


def test_replay_five_box_cargo_answer_reaches_target_selection(node, monkeypatch):
    # Exact output from the reported live failure. Images/geometry below are
    # synthetic fixtures, so this proves response handling, not model accuracy.
    tracks = scene(node, monkeypatch, [
        ("fire_extinguisher", 8., 9.), ("fire_extinguisher", 10., 9.),
        ("fire_extinguisher", 12., 9.), ("forklift", 7., 12.),
        ("forklift", 8., 12.), ("storage_rack", 20., 20.),
        ("storage_rack", 13., 9.), ("storage_rack", 8., 11.),
    ])
    begin(node, "go to the obejct that can store cargo")
    answer = (
        "<ref>the obejct that can store cargo</ref>"
        "<box><694><375><962><645></box>"
        "<box><355><740><646><936></box>"
        "<box><44><703><299><972></box>"
        "<box><27><401><317><617></box>"
        "<box><355><404><645><616></box><|im_end|>"
    )
    complete_gallery(node, answer)
    node._tick()
    assert node.pending is None and node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == tracks[-1].object_id
    event = node.events[-1]
    assert event["selection_policy"] == "nearest_model_matches"
    assert set(event["model_candidate_ids"]) == {track.object_id for track in tracks[3:]}
    assert len(event["candidate_distances_m"]) == 5
