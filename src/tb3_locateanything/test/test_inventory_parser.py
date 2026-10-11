"""Label/box pairing and explicit absence for native LocateAnything inventory."""
import pytest

from tb3_locateanything.inventory_parser import parse_inventory_output, parse_inventory_response


BOX = "<box><100><200><300><800></box>"
OTHER = "<box><600><100><900><700></box>"
CLASSES = ["forklift", "fire extinguisher", "storage rack"]


def parse(answer, categories=CLASSES):
    return parse_inventory_output(answer, 640, 480, categories)


def test_multiple_boxes_stay_with_their_ref_and_omission_is_not_absence():
    result = parse("<ref>forklift</ref>" + BOX + OTHER +
                   "<ref>fire extinguisher</ref>" + BOX + "<|im_end|>")
    assert result["observations"] == [
        {"label": "forklift", "box": [64, 96, 192, 384]},
        {"label": "forklift", "box": [384, 48, 576, 336]},
        {"label": "fire_extinguisher", "box": [64, 96, 192, 384]}]
    assert result["absent_labels"] == []
    assert result["unreported_labels"] == ["storage_rack"]


def test_one_class_absent_and_another_present_are_not_a_contradiction():
    result = parse("<ref>forklift</ref><box>None</box>\n"
                   "<ref>fire_extinguisher</ref>" + BOX +
                   "<ref>storage rack</ref><box>None</box>")
    assert result["absent_labels"] == ["forklift", "storage_rack"]
    assert result["unreported_labels"] == []
    assert result["observations"] == [{"label": "fire_extinguisher", "box": [64, 96, 192, 384]}]


def test_same_pixels_with_different_labels_remain_a_conflict_for_memory():
    result = parse("<ref>forklift</ref>" + BOX + "<ref>storage rack</ref>" + BOX)
    assert len(result["observations"]) == 2
    assert result["conflicts"] == [{"box": [64, 96, 192, 384],
                                    "labels": ["forklift", "storage_rack"]}]


def test_repeated_refs_for_separate_instances_are_accepted():
    result = parse("<ref>forklift</ref>" + BOX + "<ref>forklift</ref>" + OTHER)
    assert [o["label"] for o in result["observations"]] == ["forklift", "forklift"]
    assert result["conflicts"] == []


@pytest.mark.parametrize("categories", [CLASSES, ["forklift truck", "fire extinguisher", "storage shelving unit"]])
def test_model_aliases_preserve_canonical_memory_labels_and_absence(categories):
    result = parse("<ref>forklift truck</ref>" + BOX +
                   "<ref>storage shelving unit</ref><box>None</box>", categories)
    assert result["observations"] == [{"label": "forklift", "box": [64, 96, 192, 384]}]
    assert result["absent_labels"] == ["storage_rack"]
    assert result["unreported_labels"] == ["fire_extinguisher"]


def test_canonical_response_is_accepted_when_requested_with_model_aliases():
    result = parse("<ref>storage_rack</ref>" + BOX,
                   ["forklift truck", "storage shelving unit"])
    assert result["observations"][0]["label"] == "storage_rack"
    assert result["unreported_labels"] == ["forklift"]


def test_alias_and_canonical_refs_share_one_class_without_creating_a_conflict():
    result = parse("<ref>forklift truck</ref>" + BOX + "<ref>forklift</ref>" + OTHER)
    assert [o["label"] for o in result["observations"]] == ["forklift", "forklift"]
    assert result["conflicts"] == []


def test_alias_cannot_hide_contradictory_absence_for_the_same_class():
    with pytest.raises(ValueError, match="invalid inventory boxes"):
        parse("<ref>storage rack</ref><box>None</box><ref>storage shelving unit</ref>" + BOX)


def test_alias_still_requires_the_class_to_have_been_requested():
    with pytest.raises(ValueError, match="unrequested label"):
        parse("<ref>storage shelving unit</ref>" + BOX, ["forklift"])


def test_frozen_repaired_cab_response_uses_existing_forklift_identity():
    # Real category-probe view86: the visible cab was formerly called a rack.
    result = parse("<ref>forklift truck</ref><box><333><90><508><346></box>"
                   "<ref>fire extinguisher</ref><box>None</box>"
                   "<ref>storage shelving unit</ref><box>None</box><|im_end|>")
    assert result["observations"] == [{"label": "forklift", "box": [213.12, 43.2, 325.12, 166.08]}]
    assert result["absent_labels"] == ["fire_extinguisher", "storage_rack"]
    assert result["unreported_labels"] == []


@pytest.mark.parametrize("answer", [
    "", "none", BOX, "<box>None</box>",
    "<ref>chair</ref>" + BOX,
    "<ref>forklift</ref>",
    "<ref>forklift" + BOX,
    "<ref>forklift</ref>" + BOX + "<ref>storage rack</ref>",
    "<ref>forklift</ref>" + BOX + "<box>",
    "<ref>forklift</ref>" + BOX + "</box>",
    "<ref>forklift</ref><box><10><10><1001><80></box>",
    "<ref>forklift</ref><box><300><200><100><800></box>",
    "<ref>forklift</ref><box><-1><200><300><800></box>",
    "<ref>forklift</ref><box><nan><200><300><800></box>",
    "<ref>forklift</ref><box>None</box>" + BOX,
    "<ref>forklift</ref><box>None</box><ref>forklift</ref>" + BOX,
    "<ref>forklift</ref>" + BOX + "unparsed prose",
    "<ref>forklift</ref>" + BOX + "<|im_end|><ref>storage rack</ref>" + OTHER,
])
def test_malformed_or_unrequested_groups_reject_the_complete_answer(answer):
    with pytest.raises(ValueError):
        parse(answer)


def test_arbitrary_requested_labels_case_and_whitespace_are_normalized():
    result = parse("  <ref>TRAFFIC_LIGHT</ref>\n" + BOX + "  <|endoftext|>", ["traffic light"])
    assert result["observations"][0]["label"] == "traffic_light"


def test_invalid_image_dimensions_are_not_silently_used():
    with pytest.raises(ValueError):
        parse_inventory_output("<ref>forklift</ref>" + BOX, 0, 480, CLASSES)


@pytest.mark.parametrize("answer,categories,labels,absent", [
    ("<ref>forklift</ref><box><0><2><502><548></box>"
     "<ref>fire extinguisher</ref><box><581><135><736><550></box>"
     "<ref>storage rack</ref><box><703><0><999><538></box><|im_end|>",
     CLASSES, ["forklift", "fire_extinguisher", "storage_rack"], []),
    ("<ref>chair</ref><box><62><17><372><546></box>"
     "<box><604><18><961><548></box><ref>person</ref><box>None</box>"
     "<ref>trash can</ref><box>None</box><|im_end|>",
     ["chair", "person", "trash can"], ["chair", "chair"], ["person", "trash_can"]),
    ("<ref>forklift</ref><box>None</box>"
     "<ref>fire extinguisher</ref><box>None</box>"
     "<ref>storage rack</ref><box>None</box><|im_end|>",
     CLASSES, [], ["forklift", "fire_extinguisher", "storage_rack"]),
])
def test_native_model_responses_from_offline_inventory_probe(answer, categories, labels, absent):
    # Frozen real responses cover full, mixed and all-absent inventory output.
    result = parse(answer, categories)
    assert [item["label"] for item in result["observations"]] == labels
    assert result["absent_labels"] == absent
    assert result["unreported_labels"] == []
    assert result["conflicts"] == []


def response_with_supplement(primary, answer, category="storage rack"):
    return {"answer": primary, "supplemental_inventory": [
        {"category": category, "answer": answer,
         "prompt": "the recorded catalog prompt", "inference_seconds": .5}]}


@pytest.mark.parametrize("primary_rack", ["", "<ref>storage shelving unit</ref><box>None</box>"])
@pytest.mark.parametrize("supplement", [OTHER, "<ref>storage rack</ref>" + OTHER + "<|im_end|>"])
def test_supplemented_inventory_fills_missing_boxes_without_losing_primary(primary_rack, supplement):
    response = response_with_supplement("<ref>forklift truck</ref>" + BOX + primary_rack, supplement)
    original = response["answer"]
    result = parse_inventory_response(response, 640, 480, CLASSES)
    assert result["observations"] == [
        {"label": "forklift", "box": [64, 96, 192, 384]},
        {"label": "storage_rack", "box": [384, 48, 576, 336]}]
    assert result["absent_labels"] == []
    assert result["unreported_labels"] == ["fire_extinguisher"]
    assert response["answer"] == original


def test_supplement_keeps_cross_class_overlap_for_downstream_ambiguity_filter():
    result = parse_inventory_response(response_with_supplement(
        "<ref>forklift truck</ref>" + BOX, "<ref>storage rack</ref>" + BOX), 640, 480, CLASSES)
    assert len(result["observations"]) == 2
    assert result["conflicts"] == [{"box": [64, 96, 192, 384],
                                    "labels": ["forklift", "storage_rack"]}]


def test_only_an_explicit_supplement_none_establishes_its_absence():
    primary = "<ref>forklift truck</ref>" + BOX
    unchanged = parse_inventory_response({"answer": primary}, 640, 480, CLASSES)
    assert unchanged == parse(primary)
    result = parse_inventory_response(response_with_supplement(
        primary, "<ref>storage rack</ref><box>None</box>"), 640, 480, CLASSES)
    assert result["absent_labels"] == ["storage_rack"]
    assert result["unreported_labels"] == ["fire_extinguisher"]


@pytest.mark.parametrize("answer", ["", "none", "unparsed prose" + OTHER,
    "<ref>forklift</ref>" + OTHER,
    "<ref>storage rack</ref>" + OTHER + "unparsed prose",
    "<ref>storage rack</ref><box><900><100><600><700></box>",
    "<ref>storage rack</ref><box>None</box>" + OTHER,
    "<ref>storage rack</ref>" + OTHER + "<ref>fire extinguisher</ref>" + BOX])
def test_malformed_or_wrong_class_supplement_rejects_response(answer):
    with pytest.raises(ValueError):
        parse_inventory_response(response_with_supplement(
            "<ref>forklift truck</ref>" + BOX, answer), 640, 480, CLASSES)


def test_supplement_must_be_configured_and_not_already_present_in_primary():
    with pytest.raises(ValueError, match="unrequested category"):
        parse_inventory_response(response_with_supplement(
            "<ref>forklift</ref>" + BOX, OTHER), 640, 480, ["forklift"])
    with pytest.raises(ValueError, match="already has boxes"):
        parse_inventory_response(response_with_supplement(
            "<ref>storage shelving unit</ref>" + BOX, OTHER), 640, 480, CLASSES)


def test_alias_cannot_hide_a_duplicate_supplement():
    response = response_with_supplement("<ref>forklift</ref>" + BOX, OTHER)
    response["supplemental_inventory"].append(
        {"category": "storage shelving unit", "answer": "<box>None</box>"})
    with pytest.raises(ValueError, match="duplicate inventory supplement"):
        parse_inventory_response(response, 640, 480, CLASSES)


@pytest.mark.parametrize("supplements", [None, {}, "box", [None], [{}],
    [{"category": "storage rack"}], [{"category": 3, "answer": OTHER}]])
def test_unstructured_supplements_fail_cleanly(supplements):
    with pytest.raises(ValueError):
        parse_inventory_response({"answer": "<ref>forklift</ref>" + BOX,
                                  "supplemental_inventory": supplements}, 640, 480, CLASSES)
