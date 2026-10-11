"""Measured relation and successful-arrival history, without ROS/GPU execution."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tb3_locateanything.grounding_core import memory_relation
from test_multiple_gallery_matches import scene, SOURCE_BOX
from test_visual_memory_node import active_query, node, odom, transform, SECOND


def run(node, command):
    active_query(node, command, views=None)
    node._tick()
    assert node.pending is None
    return node.result_pub.messages[-1]


def test_farther_person_changes_after_robot_moves_with_same_old_views(node, monkeypatch):
    center, corner, table = scene(node, monkeypatch, [
        ("person", 0., 0.), ("person", -2.1, -3.1), ("table", -20., -20.),
    ])
    node.odoms.append(odom(x=1.5, y=2.3))
    out = run(node, "go to the farther person")
    assert out.success and out.object_id == corner.object_id
    assert set(node.events[-1]["candidate_distances_m"]) == {center.object_id, corner.object_id}
    node.odoms.append(odom(x=-2., y=-3.))
    out = run(node, "go to the farther person")
    assert out.success and out.object_id == center.object_id
    assert not node.inference_calls


def test_another_excludes_successful_arrival_not_most_recent_selection(node, monkeypatch):
    first, second = scene(node, monkeypatch, [("chair", 7., 9.), ("chair", 12., 9.)])
    reached = run(node, "go to " + first.object_id)
    assert reached.success and node.last_reached is None
    node._reached_target_cb(reached)
    assert node.last_reached == first.object_id
    # This selection has no successful Nav2 receipt (e.g. canceled or failed).
    selected_only = run(node, "go to " + second.object_id)
    assert selected_only.success and node.last_reached == first.object_id
    out = run(node, "go to another chair")
    assert out.success and out.object_id == second.object_id
    assert node.events[-1]["excluded_object_id"] == first.object_id
    node._reached_target_cb(out)
    assert node.last_reached == second.object_id
    out = run(node, "go to the other chair")
    assert out.success and out.object_id == first.object_id


@pytest.mark.parametrize("history", ["none", "other_class", "selected_only"])
def test_another_without_same_class_arrival_is_not_guessed(node, monkeypatch, history):
    chair, other, person = scene(node, monkeypatch, [
        ("chair", 7., 9.), ("chair", 12., 9.), ("person", 0., 0.),
    ])
    if history != "none":
        receipt = run(node, "go to " + (person if history == "other_class" else chair).object_id)
        if history == "other_class":
            node._reached_target_cb(receipt)
    out = run(node, "go to another chair")
    assert not out.success and "previous successfully reached" in out.status_message


def test_arrival_receipts_reject_wrong_identity_old_results_and_previous_session(node, monkeypatch):
    a, b = scene(node, monkeypatch, [("chair", 7., 9.), ("chair", 12., 9.)])
    older = run(node, "go to " + a.object_id)
    newer = run(node, "go to " + b.object_id)
    wrong = deepcopy(newer)
    wrong.object_id = a.object_id
    node._reached_target_cb(wrong)
    assert node.last_reached is None
    node._reached_target_cb(newer)
    node._reached_target_cb(older)
    assert node.last_reached == b.object_id
    node._reset_target_history()  # also called on node restart
    node._reached_target_cb(newer)
    assert node.last_reached is None


def test_clock_reset_clears_arrival_history_and_rejects_latched_receipt(node, monkeypatch):
    a, b = scene(node, monkeypatch, [("chair", 7., 9.), ("chair", 12., 9.)])
    receipt = run(node, "go to " + a.object_id)
    node._reached_target_cb(receipt)
    node.now_ns = 2 * SECOND
    node._tick()
    node._reached_target_cb(receipt)
    assert node.epoch == 1 and node.last_reached is None
    assert not node.memory.landmarks


@pytest.mark.parametrize("problem", ["tie", "single", "pending", "conflict", "missing_geometry"])
def test_incomplete_or_ambiguous_comparison_fails(node, monkeypatch, problem):
    coords = [("chair", 7., 9.)]
    if problem != "single":
        coords.append(("chair", 9.05 if problem == "tie" else 12., 9.))
    tracks = scene(node, monkeypatch, coords)
    if problem == "pending":
        tracks[-1].confirmed = False
    elif problem == "conflict":
        tracks[-1].conflicted = True
    elif problem == "missing_geometry":
        broken_view = node.memory.views[-1]
        localize = node._localize
        def with_missing(view, box, projection=None):
            if view is broken_view:
                raise ValueError("no valid depth")
            return localize(view, box, projection)
        node._localize = with_missing
    out = run(node, "go to the farther chair")
    assert not out.success
    assert not any(event["state"] == "selected" for event in node.events)


def test_comparison_uses_one_fresh_projection_for_robot_and_all_candidates(node, monkeypatch):
    a, b = scene(node, monkeypatch, [("chair", 7., 9.), ("chair", 12., 9.)])
    current = transform(100, (20., 30., 0.))
    node.tf_buffer = SimpleNamespace(lookup_transform=lambda *args: current)
    out = run(node, "go to the farther chair")
    assert out.success and out.object_id == b.object_id
    event = node.events[-1]
    assert event["selection_robot_map"] == {"x": 28., "y": 39., "z": 0.}
    assert event["candidate_distances_m"] == {a.object_id: 1., b.object_id: 4.}
    node.tf_buffer = SimpleNamespace(lookup_transform=lambda *args: transform(90))
    assert not run(node, "go to the farther chair").success


@pytest.mark.parametrize("command", ["another red chair", "farther person from the table",
                                     "nearest object I can sit on", "nearer chair beside the table"])
def test_extra_predicates_are_not_silently_discarded(command):
    with pytest.raises(ValueError, match="additional attributes or reference objects"):
        memory_relation(command, ["chair", "person", "table", "object"])


def test_corner_preserves_full_scene_query(node, monkeypatch):
    scene(node, monkeypatch, [("person", 0., 0.), ("person", -2., -3.)])
    active_query(node, "go to the person at the corner", views=None)
    node._tick()
    assert node.job["kind"] == "query"
    assert node.inference_calls[-1][1]["expression"] == "the person at the corner"


def test_left_chair_rejects_known_table_box(node, monkeypatch):
    chair, table = scene(node, monkeypatch, [("chair", 7., 9.), ("table", 12., 9.)])
    table_view = node.memory.views[-1]
    active_query(node, "go to the left chair")
    # The exact table box is supplied by a controlled wrong-category model reply.
    answer = "<box><234.375><208.333333333><703.125><833.333333333></box>"
    with pytest.raises(ValueError, match="remembered table, not requested chair"):
        node._query_result({"view": table_view}, {"request_id": node.pending["id"], "answer": answer})
