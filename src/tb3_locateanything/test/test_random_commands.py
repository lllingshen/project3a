"""Natural-language targets remain model-grounded, including random requests."""
import pytest

from tb3_locateanything.geometry import Vector3
from tb3_locateanything.grounding_core import allows_any_instance
from tb3_locateanything.visual_memory import Memory
from test_visual_memory_node import (
    BOX, active_query, complete_gallery, gallery_answer, module, node, stored_view,
)


@pytest.mark.parametrize("command", [
    "go to a random forklift", "Please MOVE to one random forklift!",
    "navigate to random forklift.", "drive to any forklift", "go to a forklift",
    "go to an extinguisher", "any fire extinguisher", "a random FIRE_EXTINGUISHER",
    "go to a random red forklift", "move to any red object I can sit on",
    "go to a random unfamiliar machine",
])
def test_any_permission_is_generic_and_preserves_descriptive_meaning(command):
    assert allows_any_instance(command)


@pytest.mark.parametrize("command", [
    "go to forklift_1", "do not go to a random forklift",
    "go to any forklift except the red one", "go to any forklift or chair",
    "go to any forklift and then a chair", "go to another forklift",
    "go to the nearest forklift", "go to the forklift", "", "a random",
    "go to the chair beside any forklift", "not a random forklift",
])
def test_any_permission_does_not_override_negation_or_another_target(command):
    assert not allows_any_instance(command)


def two_forklifts(node):
    node.memory = Memory(class_labels=["forklift", "chair"])
    view = stored_view(node)
    first = node.memory.remember(view, "forklift", BOX, 1., 2.)
    second = node.memory.remember(view, "forklift", [350., 100., 520., 400.], 4., 2.)
    node._refresh_positions = lambda: None
    node._localize = lambda _view, box, projection=None: (Vector3(box[0] / 100., 2., 0.), {})
    return first, second, view


def start_request(node, command="go to a random forklift"):
    active_query(node, command, views=None)
    node._tick()
    assert not node.result_pub.messages  # Target requires an actual model response.
    assert node.job["kind"] == "gallery"
    request = node.inference_calls[0][1]
    assert request["operation"] == "catalog"
    assert request["command"] == command
    assert request["expression"] == module.referring_expression(command)
    return node.job["gallery"]


@pytest.mark.parametrize("command", [
    "go to forklift_1", "go to a forklift_1", "go to an forklift_1",
    "please move to the forklift_1!",
])
def test_exact_instance_still_selects_requested_id_by_default(node, monkeypatch, command):
    first, second, _ = two_forklifts(node)

    def forbidden_choice(_):
        raise AssertionError("an exact ID must never choose a random instance")

    monkeypatch.setattr(module.random, "choice", forbidden_choice)
    active_query(node, command, views=None)
    node._tick()
    assert not node.inference_calls and node.pending is None
    assert node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == second.object_id != first.object_id
    assert node.events[-1]["query_kind"] == "object_id"


@pytest.mark.parametrize("selected_index", [0, 1])
@pytest.mark.parametrize("command", [
    "go to a random forklift", "go to any forklift", "go to a forklift",
    "go to a random red forklift", "move to any red object I can sit on",
])
def test_only_model_matches_are_eligible_and_full_semantics_reach_model(
        node, monkeypatch, selected_index, command):
    first, second, _ = two_forklifts(node)
    gallery = start_request(node, command)
    pools = []

    def choose(candidates):
        pools.append(list(candidates))
        return candidates[selected_index]

    monkeypatch.setattr(module.random, "choice", choose)
    complete_gallery(node, "".join(gallery_answer(gallery, entry.object_box)
                                    for entry in gallery.entries))
    node._tick()
    assert len(pools) == 1 and len(pools[0]) == 2
    assert node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == gallery.entries[selected_index].object_id
    event = node.events[-1]
    assert event["query_kind"] == "gallery_description"
    assert event["selection_policy"] == "random_model_matches"
    assert sorted(event["random_candidate_ids"]) == [first.object_id, second.object_id]


def test_one_model_match_cannot_be_replaced_by_another_memory_instance(node, monkeypatch):
    first, second, _ = two_forklifts(node)
    gallery = start_request(node, "go to a random red forklift")
    target = next(entry for entry in gallery.entries if entry.object_id == second.object_id)
    pools = []

    def choose(candidates):
        pools.append(list(candidates))
        return candidates[0]

    monkeypatch.setattr(module.random, "choice", choose)
    complete_gallery(node, gallery_answer(gallery, target.object_box))
    node._tick()
    assert all(len(pool) == 1 for pool in pools)
    assert node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == second.object_id != first.object_id
    assert node.events[-1]["random_candidate_ids"] == [second.object_id]


def test_duplicate_model_boxes_do_not_weight_one_instance_multiple_times(node, monkeypatch):
    first, second, _ = two_forklifts(node)
    gallery = start_request(node)
    pools = []

    def choose(candidates):
        pools.append(list(candidates))
        return candidates[-1]

    monkeypatch.setattr(module.random, "choice", choose)
    first_box, second_box = [gallery_answer(gallery, entry.object_box) for entry in gallery.entries]
    complete_gallery(node, first_box * 3 + second_box)
    node._tick()
    assert len(pools) == 1 and len(pools[0]) == 2
    assert node.result_pub.messages[0].success
    assert sorted(node.events[-1]["random_candidate_ids"]) == [first.object_id, second.object_id]


@pytest.mark.parametrize("answer_kind", ["none", "malformed", "invalid_box", "background", "mixed_invalid"])
def test_random_request_never_guesses_from_memory_when_model_output_is_invalid(node, monkeypatch, answer_kind):
    two_forklifts(node)
    gallery = start_request(node)
    valid = gallery_answer(gallery, gallery.entries[0].object_box)
    answers = {
        "none": "<box>None</box>",
        "malformed": "not a grounding result",
        "invalid_box": "<box><0><0><1001><900></box>",
        "background": "<box><0><0><998><998></box>",
        "mixed_invalid": valid + "<box><0><0><1001><900></box>",
    }

    def forbidden_choice(_):
        raise AssertionError("invalid model output must not become random memory selection")

    monkeypatch.setattr(module.random, "choice", forbidden_choice)
    complete_gallery(node, answers[answer_kind])
    node._tick()
    assert node.pending is None
    assert len(node.inference_calls) == 1
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert not any(event["state"] == "selected" for event in node.events)


def test_definite_target_chooses_nearest_model_match_without_random_choice(node, monkeypatch):
    _, second, _ = two_forklifts(node)
    gallery = start_request(node, "go to the red forklift")

    def forbidden_choice(_):
        raise AssertionError("ordinary descriptions use nearest, not random choice")

    monkeypatch.setattr(module.random, "choice", forbidden_choice)
    complete_gallery(node, "".join(gallery_answer(gallery, entry.object_box)
                                    for entry in gallery.entries))
    node._tick()
    assert node.pending is None and node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == second.object_id
    assert node.events[-1]["selection_policy"] == "nearest_model_matches"


@pytest.mark.parametrize("excluded_state", ["pending", "conflicted", "missing_image"])
def test_random_gallery_can_present_eligible_subset_but_model_still_selects(node, excluded_state):
    first, second, view = two_forklifts(node)
    if excluded_state == "pending":
        node.memory.confirm_classes = True
        second.confirmed = False
    elif excluded_state == "conflicted":
        second.conflicted = True
    else:
        view.observations[:] = [item for item in view.observations
                                if item["object_id"] != second.object_id]
    gallery = start_request(node)
    assert [entry.object_id for entry in gallery.entries] == [first.object_id]
    complete_gallery(node, gallery_answer(gallery, gallery.entries[0].object_box))
    node._tick()
    assert node.result_pub.messages[0].success
    assert node.result_pub.messages[0].object_id == first.object_id


def test_empty_eligible_subset_never_reaches_model_or_navigation(node):
    first, second, _ = two_forklifts(node)
    first.confirmed, second.conflicted = False, True
    active_query(node, "go to a random forklift", views=None)
    node._tick()
    assert node.pending is None and not node.inference_calls
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success


@pytest.mark.parametrize("failure", [ValueError, module.tf2_ros.TransformException])
def test_id_evidence_tries_another_retained_view_after_geometry_failure(node, failure):
    first, _, old = two_forklifts(node)
    latest = stored_view(node, 20, pose=(2., 0., 0.))
    assert node.memory.remember(latest, "forklift", BOX, 1., 2.) is first
    calls = []

    def localize(view, box):
        calls.append(view)
        if view is latest:
            raise failure("unusable latest geometry")
        return Vector3(1., 2., 0.), {"source": "older valid evidence"}

    node._localize = localize
    active_query(node, "go to forklift_0")
    candidates = node._memory_candidates([latest, old], [first])
    assert calls == [latest, old]
    assert len(candidates) == 1
    assert candidates[0][0] is old and candidates[0][-1] is first


def test_id_evidence_skips_source_that_did_not_support_confirmed_class(node):
    first, _, old = two_forklifts(node)
    latest = stored_view(node, 20, pose=(2., 0., 0.))
    assert node.memory.remember(latest, "forklift", BOX, 1., 2.) is first
    latest.observations[0]["labels"] = ["chair"]
    node.memory.confirm_classes = True
    calls = []
    node._localize = lambda view, _box: calls.append(view) or (Vector3(1., 2., 0.), {})
    active_query(node, "go to forklift_0")
    candidates = node._memory_candidates([latest, old], [first])
    assert calls == [old] and candidates[0][0] is old


@pytest.mark.parametrize("invalidated", ["timeout", "motion"])
def test_random_selection_retains_final_deadline_and_motion_checks(node, monkeypatch, invalidated):
    two_forklifts(node)
    gallery = start_request(node)

    def choose(candidates):
        if invalidated == "timeout":
            node.pending["wall_start"] -= 91
        else:
            node.pending["robot_moved"] = True
        return candidates[0]

    monkeypatch.setattr(module.random, "choice", choose)
    complete_gallery(node, "".join(gallery_answer(gallery, entry.object_box)
                                    for entry in gallery.entries))
    node._tick()
    assert node.pending is None
    assert len(node.result_pub.messages) == 1 and not node.result_pub.messages[0].success
    assert not any(event["state"] == "selected" for event in node.events)
