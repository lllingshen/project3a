"""Pending tracks must not veto model queries over usable confirmed objects."""
from copy import deepcopy

import pytest

from tb3_locateanything.geometry import Vector3
from tb3_locateanything.visual_memory import Memory
from test_visual_memory_node import (
    BOX, active_query, gallery_answer, module, node, stored_view,
)


COMMANDS = [
    "go to the object that can store cargo",
    "go to the storage rack",
]


def warehouse_memory(node, *, confirm=True):
    """Independent views confirm two racks and a forklift; one rack stays pending."""
    node.p["localization_mode"] = "depth"
    node.memory = Memory(max_views=32, confirm_classes=True,
                         class_labels=["storage_rack", "forklift", "fire_extinguisher"])
    tracks = []
    for index, label in enumerate(["storage_rack", "storage_rack", "forklift"]):
        x = float(2 + 3 * index)
        first = stored_view(node, 10 + index * 10, pose=(0., 0., 0.))
        obj = node.memory.remember(first, label, BOX, x, 1., support_points=[(x, 1.)])
        if confirm:
            second = stored_view(node, 11 + index * 10, pose=(.7, 0., 0.))
            assert node.memory.remember(second, label, BOX, x, 1.,
                                        support_points=[(x, 1.)]) is obj
        tracks.append(obj)
    view = stored_view(node, 50, pose=(3., 0., 0.))
    pending = node.memory.remember(view, "storage_rack", BOX, -2., 1.,
                                   support_points=[(-2., 1.)])
    assert not pending.confirmed
    assert all(obj.confirmed == confirm for obj in tracks)
    # Keep query routing/selection real while freezing sensor geometry at the
    # worker boundary. There is no ROS spin, model process or navigation here.
    node._refresh_positions = lambda: None
    localized = []
    node._localize = lambda source, box, *projection: (
        localized.append((source, list(box))) or (Vector3(5., 6., 0.), {}))
    return tracks, pending, localized


def begin_query(node, command):
    active_query(node, command, views=None)
    node._tick()
    assert node.job is not None and node.job["kind"] == "gallery"
    assert len(node.inference_calls) == 1
    assert not node.result_pub.messages
    assert not any(event["state"] == "selected" for event in node.events)
    return node.job["gallery"]


def finish_query(node, answer):
    node.future.set_result({"request_id": node.pending["id"], "answer": answer,
                           "prompt": node.inference_calls[0][1]["expression"],
                           "inference_seconds": .2})
    node._tick()


@pytest.mark.parametrize("command", COMMANDS)
def test_pending_rack_does_not_block_model_and_selected_source_is_preserved(node, command):
    tracks, pending, localized = warehouse_memory(node)
    pending_before = deepcopy(vars(pending))
    gallery = begin_query(node, command)
    assert not localized
    eligible_ids = {entry.object_id for entry in gallery.entries}
    expected = tracks if "object" in command else tracks[:2]
    assert eligible_ids == {obj.object_id for obj in expected}
    assert pending.object_id not in eligible_ids
    assert gallery.excluded_objects == {pending.object_id: "unconfirmed"}
    inference = next(event for event in node.events if event["state"] == "inference")
    assert inference["excluded_objects"] == gallery.excluded_objects
    request = node.inference_calls[0][1]
    assert request["operation"] == "catalog"
    assert request["command"] == command
    assert request["expression"] == command.removeprefix("go to ")
    assert request["image_png"] == gallery.image_png

    # Select the second rack, never an automatic first category/ID match.
    chosen = next(entry for entry in gallery.entries if entry.object_id == tracks[1].object_id)
    assert chosen.object_box != chosen.source_box
    finish_query(node, gallery_answer(gallery, chosen.object_box))
    assert node.pending is None and len(node.result_pub.messages) == 1
    out = node.result_pub.messages[0]
    assert out.success and out.object_id == tracks[1].object_id
    assert node.events[-1]["gallery_excluded_objects"] == gallery.excluded_objects
    assert module.stamp_ns(out) == chosen.view.stamp_ns
    assert (out.position.x, out.position.y) == (5., 6.)
    assert localized and all(source is chosen.view and box == chosen.source_box
                             for source, box in localized)
    assert vars(pending) == pending_before
    assert node.memory.landmarks[pending.object_id] is pending
    assert not pending.confirmed


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("response", ["none", "mixed_invalid", "outside", "background", "malformed"])
def test_partial_gallery_still_requires_valid_model_matches(node, command, response):
    tracks, pending, _ = warehouse_memory(node)
    pending_before = deepcopy(vars(pending))
    gallery = begin_query(node, command)
    rack_boxes = [next(entry.object_box for entry in gallery.entries
                       if entry.object_id == track.object_id) for track in tracks[:2]]
    answers = {
        "none": "<box>None</box>",
        "mixed_invalid": gallery_answer(gallery, rack_boxes[0]) + "<box><0><0><1001><900></box>",
        "outside": "<box><-100><100><1100><900></box>",
        "background": gallery_answer(gallery, [1, 1, 8, 8]),
        "malformed": "not a grounding result",
    }
    finish_query(node, answers[response])
    assert node.pending is None and len(node.result_pub.messages) == 1
    assert not node.result_pub.messages[0].success
    assert len(node.inference_calls) == 1
    assert not any(event["state"] == "selected" for event in node.events)
    assert vars(pending) == pending_before


@pytest.mark.parametrize("command", COMMANDS)
def test_multiple_model_matches_select_nearest_without_promoting_pending(node, command):
    tracks, pending, _ = warehouse_memory(node)
    pending_before = deepcopy(vars(pending))
    gallery = begin_query(node, command)
    racks = [entry for entry in gallery.entries if entry.object_id in
             {obj.object_id for obj in tracks[:2]}]
    finish_query(node, "".join(gallery_answer(gallery, entry.object_box) for entry in racks))
    assert node.pending is None and node.result_pub.messages[0].success
    # Fixture positions tie; stable ID ordering chooses the first model match.
    assert node.result_pub.messages[0].object_id == tracks[0].object_id
    assert node.events[-1]["selection_policy"] == "nearest_model_matches"
    assert node.events[-1]["model_candidate_ids"] == [obj.object_id for obj in tracks[:2]]
    assert vars(pending) == pending_before


def test_no_eligible_requested_category_never_falls_back_to_another_category(node):
    _, pending, localized = warehouse_memory(node)
    pending_before = deepcopy(vars(pending))
    active_query(node, "go to the fire extinguisher", views=None)
    node._tick()
    assert node.pending is None and not node.inference_calls and not localized
    assert len(node.result_pub.messages) == 1
    assert not node.result_pub.messages[0].success
    assert "no usable retained visual instance" in node.result_pub.messages[0].status_message
    assert vars(pending) == pending_before


@pytest.mark.parametrize("reason", ["conflicted", "unusable_image_box"])
def test_gallery_omits_ineligible_confirmed_objects_without_unquarantining_them(node, reason):
    tracks, pending, _ = warehouse_memory(node)
    excluded = tracks[2]
    if reason == "conflicted":
        excluded.conflicted = True
    else:
        for view in node.memory.views:
            for observation in view.observations:
                if observation["object_id"] == excluded.object_id:
                    observation["box"] = [0, 100, 250, 400]  # Horizontally clipped.
    excluded_before = deepcopy(vars(excluded))
    pending_before = deepcopy(vars(pending))
    gallery = begin_query(node, COMMANDS[0])
    assert {entry.object_id for entry in gallery.entries} == {
        obj.object_id for obj in tracks[:2]}
    finish_query(node, "<box>None</box>")
    assert not node.result_pub.messages[0].success
    assert vars(excluded) == excluded_before
    assert vars(pending) == pending_before


@pytest.mark.parametrize("command", COMMANDS)
def test_all_pending_memory_cannot_supply_a_query_target(node, command):
    _, pending, localized = warehouse_memory(node, confirm=False)
    before = deepcopy(node.memory.serialize())
    active_query(node, command, views=None)
    node._tick()
    assert node.pending is None and not node.inference_calls and not localized
    assert len(node.result_pub.messages) == 1
    assert not node.result_pub.messages[0].success
    assert "no usable retained visual instance" in node.result_pub.messages[0].status_message
    assert node.memory.serialize() == before
    assert not pending.confirmed
