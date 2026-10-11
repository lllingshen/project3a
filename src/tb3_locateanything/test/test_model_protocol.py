"""Prompt protocol tests do not import the CUDA model or require ROS."""
import pytest

from tb3_locateanything.model_worker import infer_request, model_prompt


def test_grounding_keeps_the_full_referring_expression():
    expression = "the chair to the right of the red chair"
    assert model_prompt("ground", expression) == (
        "Locate a single instance that matches the following description: "
        "the chair to the right of the red chair.")


def test_catalog_uses_eagle_multi_instance_prompt_for_arbitrary_class():
    assert model_prompt("catalog", "trash can") == (
        "Locate all the instances that matches the following description: trash can.")


def test_inventory_uses_upstream_category_separator_without_a_class_count():
    assert model_prompt("inventory", categories=["forklift", "fire extinguisher", "storage rack"]) == (
        "Locate all the instances that matches the following description: "
        "forklift truck</c>fire extinguisher</c>storage shelving unit.")
    assert model_prompt("inventory", categories=["traffic light"]) == (
        "Locate all the instances that matches the following description: traffic light.")


def test_inventory_alias_inputs_produce_the_same_prompt_as_canonical_classes():
    canonical = ["forklift", "fire extinguisher", "storage_rack"]
    aliases = ["FORKLIFT_TRUCK", "fire extinguisher", "storage shelving unit"]
    assert model_prompt("inventory", categories=aliases) == model_prompt("inventory", categories=canonical)


def test_inventory_aliases_do_not_rewrite_user_grounding_or_catalog_expression():
    phrase = "the storage rack to the left of the forklift"
    assert model_prompt("ground", phrase).endswith(phrase + ".")
    assert model_prompt("catalog", "forklift").endswith("description: forklift.")


@pytest.mark.parametrize("categories", [None, [], "chair", [""], [3],
                                        ["chair", "CHAIR"], ["chair</c>person"],
                                        ["forklift", "forklift truck"],
                                        ["storage shelving unit", "storage_rack"]])
def test_inventory_does_not_accept_ambiguous_or_unstructured_categories(categories):
    with pytest.raises(ValueError):
        model_prompt("inventory", categories=categories)


@pytest.mark.parametrize("operation,expression", [
    ("unknown", "chair"), ("catalog", ""), ("ground", "   "),
])
def test_invalid_request_does_not_silently_run_another_task(operation, expression):
    with pytest.raises(ValueError):
        model_prompt(operation, expression)


BOX = "<box><100><100><300><500></box>"


def fake_generator(*answers):
    calls = []
    queued = iter(answers)

    def generate(prompt, multi_instance):
        calls.append((prompt, multi_instance))
        return next(queued)

    return generate, calls


@pytest.mark.parametrize("rack", ["", "<ref>storage shelving unit</ref><box>None</box>"])
def test_inventory_runs_one_canonical_rack_supplement_and_records_both_calls(rack):
    primary = "<ref>forklift truck</ref>" + BOX + rack
    supplement = "<ref>storage rack</ref>" + BOX + "<|im_end|>"
    generate, calls = fake_generator((primary, 1.25), (supplement, .75))
    result = infer_request({"request_id": "catalog_1", "operation": "inventory",
                           "categories": ["forklift", "storage_rack"]}, 640, 480, generate)
    assert calls == [(model_prompt("inventory", categories=["forklift", "storage rack"]), True),
                     ("Locate all the instances that matches the following description: storage rack.", True)]
    assert result["answer"] == primary
    assert result["supplemental_inventory"] == [{
        "category": "storage rack", "prompt": calls[1][0],
        "answer": supplement, "inference_seconds": .75}]
    assert result["primary_inference_seconds"] == 1.25
    assert result["total_inference_seconds"] == result["inference_seconds"] == 2.
    assert result["image_width"] == 640 and result["image_height"] == 480


def test_all_negative_inventory_is_bounded_by_one_call_per_configured_category():
    generate, calls = fake_generator(("<ref>forklift truck</ref><box>None</box>", .4),
        ("<ref>forklift</ref><box>None</box>", .2),
        ("<ref>fire extinguisher</ref><box>None</box>", .3),
        ("<ref>storage rack</ref><box>None</box>", .6))
    result = infer_request({"request_id": 1, "operation": "inventory",
                           "categories": ["forklift", "fire extinguisher", "storage rack"]},
                          640, 480, generate)
    assert len(calls) == 4 and len(result["supplemental_inventory"]) == 3
    assert [item["category"] for item in result["supplemental_inventory"]] == [
        "forklift", "fire extinguisher", "storage rack"]
    assert result["total_inference_seconds"] == pytest.approx(1.5)


@pytest.mark.parametrize("category", ["forklift", "storage rack", "safety helmet"])
def test_primary_positive_categories_never_request_supplements(category):
    primary = f"<ref>{category}</ref>" + BOX
    generate, calls = fake_generator((primary, .4))
    result = infer_request({"request_id": 1, "operation": "inventory",
                           "categories": [category]}, 640, 480, generate)
    assert len(calls) == 1 and result["supplemental_inventory"] == []
    assert result["total_inference_seconds"] == .4


def test_custom_missing_categories_are_retried_symmetrically_with_positive_classes_preserved():
    primary = ("<ref>traffic light</ref>" + BOX +
               "<ref>safety helmet</ref><box>None</box>")
    generate, calls = fake_generator((primary, .8),
        ("<ref>safety helmet</ref>" + BOX, .2),
        ("<ref>coat rack</ref><box>None</box>", .3))
    result = infer_request({"request_id": 8, "operation": "inventory",
                           "categories": ["traffic light", "safety helmet", "coat rack"]},
                          640, 480, generate)
    assert calls[1:] == [(model_prompt("catalog", "safety helmet"), True),
                        (model_prompt("catalog", "coat rack"), True)]
    assert result["answer"] == primary
    assert [item["category"] for item in result["supplemental_inventory"]] == [
        "safety helmet", "coat rack"]
    assert result["primary_inference_seconds"] == .8
    assert result["total_inference_seconds"] == pytest.approx(1.3)


def test_configured_aliases_use_canonical_single_class_prompts_and_primary_labels():
    generate, calls = fake_generator(
        ("<ref>storage shelving unit</ref>" + BOX, .4),
        ("<ref>forklift truck</ref>" + BOX, .2))
    result = infer_request({"request_id": 9, "operation": "inventory",
                           "categories": ["FORKLIFT_TRUCK", "storage shelving unit"]},
                          640, 480, generate)
    assert len(calls) == 2
    assert calls[0] == (model_prompt("inventory", categories=["forklift", "storage rack"]), True)
    assert calls[1] == (model_prompt("catalog", "forklift"), True)
    assert result["categories"] == ["forklift", "storage rack"]
    assert result["supplemental_inventory"][0]["category"] == "forklift"


@pytest.mark.parametrize("operation,multi", [("ground", False), ("catalog", True)])
def test_user_grounding_and_catalog_remain_one_unchanged_prompt(operation, multi):
    phrase = "the storage rack beside the forklift"
    generate, calls = fake_generator((BOX, .5))
    result = infer_request({"request_id": 2, "operation": operation, "expression": phrase},
                           640, 480, generate)
    assert calls == [(model_prompt(operation, phrase), multi)]
    assert result["expression"] == phrase and result["supplemental_inventory"] == []


def test_invalid_primary_inventory_stops_before_any_retry():
    generate, calls = fake_generator(("not labelled boxes", .5))
    with pytest.raises(ValueError, match="paired ref"):
        infer_request({"request_id": 3, "operation": "inventory",
                       "categories": ["storage rack"]}, 640, 480, generate)
    assert len(calls) == 1


def test_wrong_class_supplement_is_not_emitted_as_success():
    generate, calls = fake_generator(("<ref>storage shelving unit</ref><box>None</box>", .5),
                                     ("<ref>forklift truck</ref>" + BOX, .5))
    with pytest.raises(ValueError, match="unrequested label"):
        infer_request({"request_id": 4, "operation": "inventory",
                       "categories": ["storage rack"]}, 640, 480, generate)
    assert len(calls) == 2
