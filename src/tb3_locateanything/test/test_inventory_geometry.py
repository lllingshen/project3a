import copy

import pytest

from tb3_locateanything.inventory_geometry import depth_separated, filter_ambiguous_inventory
from tb3_locateanything.inventory_parser import parse_inventory_output


def detection(label, box):
    return {"label": label, "box": box}


def test_nested_cross_class_boxes_reject_both_even_with_low_iou():
    items = [detection("vehicle", [0, 0, 100, 100]),
             detection("furniture", [30, 30, 50, 50])]
    accepted, rejected = filter_ambiguous_inventory(items)
    assert accepted == []
    assert [r["index"] for r in rejected] == [0, 1]
    assert all(r["ambiguous_indices"] == [0, 1] for r in rejected)
    assert rejected[0]["overlaps"] == [{"index": 1, "intersection_over_smaller": 1.}]
    assert "another view" in rejected[0]["reason"]


def test_identical_cross_class_boxes_are_not_committed_under_either_label():
    accepted, rejected = filter_ambiguous_inventory([
        detection("class_a", [10, 10, 30, 30]), detection("class_b", [10, 10, 30, 30])])
    assert not accepted and len(rejected) == 2


def test_near_total_overlap_rejects_a_whole_connected_group():
    items = [detection("a", [0, 0, 10, 10]), detection("b", [1, 0, 11, 10]),
             detection("c", [2, 0, 12, 10]), detection("d", [50, 0, 60, 10])]
    accepted, rejected = filter_ambiguous_inventory(items)
    assert accepted == [items[3]]
    assert all(r["ambiguous_indices"] == [0, 1, 2] for r in rejected)
    assert [r["index"] for r in rejected] == [0, 1, 2]


@pytest.mark.parametrize("second", [[50, 0, 150, 100], [100, 0, 200, 100], [200, 0, 300, 100]])
def test_partial_occlusion_touching_and_disjoint_different_classes_survive(second):
    items = [detection("a", [0, 0, 100, 100]), detection("b", second)]
    accepted, rejected = filter_ambiguous_inventory(items)
    assert accepted == items and rejected == []


def test_same_class_boxes_are_left_to_memory_including_label_normalization():
    items = [detection("storage_rack", [0, 0, 100, 100]),
             detection(" Storage RACK ", [20, 20, 50, 50]),
             detection("storage_rack", [200, 200, 300, 300])]
    accepted, rejected = filter_ambiguous_inventory(items)
    assert accepted == items and not rejected


def test_filter_does_not_mutate_parser_output_and_preserves_unrelated_observations():
    parsed = parse_inventory_output(
        "<ref>vehicle</ref><box><100><100><700><700></box>"
        "<ref>furniture</ref><box><200><200><400><400></box>"
        "<ref>person</ref><box><800><100><900><700></box>",
        640, 480, ["vehicle", "furniture", "person"])
    original = copy.deepcopy(parsed)
    accepted, rejected = filter_ambiguous_inventory(parsed["observations"])
    assert parsed == original
    assert len(accepted) == 1 and accepted[0] is parsed["observations"][2]
    assert accepted[0]["label"] == "person" and len(rejected) == 2


def test_reversing_inventory_order_does_not_select_an_arbitrary_winner():
    items = [detection("a", [0, 0, 100, 100]), detection("b", [30, 30, 50, 50])]
    assert not filter_ambiguous_inventory(items)[0]
    assert not filter_ambiguous_inventory(reversed(items))[0]


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, float("nan"), 1], [0, 0, 1], None])
def test_invalid_box_fails_before_any_observation_is_accepted(box):
    with pytest.raises(ValueError, match="box"):
        filter_ambiguous_inventory([detection("a", [0, 0, 1, 1]), detection("b", box)])


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan")])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="min_containment"):
        filter_ambiguous_inventory([], threshold)


def test_empty_inventory():
    assert filter_ambiguous_inventory([]) == ([], [])


def depth_geometry(depth, x=0., y=0.):
    return {"localization_source": "aligned_depth", "depth_m": depth,
            "support_points": [[x, y], [x+.2, y], [x, y+.2]]}


def test_nested_objects_with_independently_separated_depth_survive_without_mutation():
    items = [detection("frame", [0, 0, 100, 100]), detection("object", [30, 30, 50, 50])]
    geometry = {0: depth_geometry(1.45), 1: depth_geometry(4.13, y=2.5)}
    original = copy.deepcopy((items, geometry))
    assert filter_ambiguous_inventory(items, geometries=geometry) == (items, [])
    assert (items, geometry) == original


def test_sparse_samples_do_not_exempt_nested_part_when_whole_xy_bounds_overlap():
    whole = {"localization_source": "aligned_depth", "depth_m": 2.,
             "support_points": [[0., 0.], [4., 0.], [0., 4.]]}
    part = depth_geometry(3., x=1., y=1.)
    # Individual sparse sample distances exceed .35 m; the whole object's
    # bounds still contain the part, so a median-depth gap cannot exempt it.
    assert not depth_separated(whole, part)


def test_lateral_support_separation_without_depth_gap_is_not_an_exception():
    assert not depth_separated(depth_geometry(2.), depth_geometry(2.49, x=3.))


def test_depth_gap_without_enough_xy_separation_is_not_an_exception():
    assert not depth_separated(depth_geometry(1.), depth_geometry(3., y=.54))


def test_both_thresholds_are_inclusive_and_symmetric():
    first = {"localization_source": "aligned_depth", "depth_m": 1.,
             "support_points": [[0., -.2], [.2, -.2], [0., 0.]]}
    second = depth_geometry(1.5, y=.35)
    assert depth_separated(first, second)
    assert depth_separated(second, first)


@pytest.mark.parametrize("invalid", [
    None, [], {}, {"localization_source": "laser", "depth_m": 4., "support_points": [[0, 2], [1, 2], [0, 3]]},
    {**depth_geometry(4., y=3.), "depth_m": 0},
    {**depth_geometry(4., y=3.), "depth_m": -1},
    {**depth_geometry(4., y=3.), "depth_m": float("nan")},
    {**depth_geometry(4., y=3.), "depth_m": float("inf")},
    {**depth_geometry(4., y=3.), "depth_m": "4"},
    {**depth_geometry(4., y=3.), "depth_m": True},
    {**depth_geometry(4., y=3.), "depth_m": 10**1000},
    {**depth_geometry(4., y=3.), "support_points": None},
    {**depth_geometry(4., y=3.), "support_points": [[0, 3], [1, 3]]},
    {**depth_geometry(4., y=3.), "support_points": [[0, 3], [0, 3], [1, 3]]},
    {**depth_geometry(4., y=3.), "support_points": [[0, 3], [1, 3], [0, float("nan")]]},
    {**depth_geometry(4., y=3.), "support_points": [[0, 3], [1, 3], [0, float("inf")]]},
    {**depth_geometry(4., y=3.), "support_points": [[0, 3], [1, 3], [0, 10**1000]]},
    {**depth_geometry(4., y=3.), "support_points": [[0, 3], [1, 3], [0, 3, 4]]},
    {**depth_geometry(4., y=3.), "support_points": [[0, 3], [1, 3], [0, "4"]]},
    {**depth_geometry(4., y=3.), "support_points": [[0, 3], [1, 3], [True, 4]]},
])
def test_missing_invalid_or_insufficient_support_fails_closed(invalid):
    valid = depth_geometry(1.)
    assert not depth_separated(valid, invalid)
    assert not depth_separated(invalid, valid)
    items = [detection("a", [0, 0, 10, 10]), detection("b", [1, 1, 9, 9])]
    assert not filter_ambiguous_inventory(items, geometries={0: valid, 1: invalid})[0]


def test_remaining_ambiguous_edges_still_reject_entire_connected_group():
    items = [detection(label, [0, 0, 100, 100]) for label in ("a", "b", "c")]
    # A and B have distinct depth support, but C has no valid geometry and
    # remains ambiguous with both. It must not inherit either exemption.
    geometry = {0: depth_geometry(1.), 1: depth_geometry(4., y=3.)}
    accepted, rejected = filter_ambiguous_inventory(items, geometries=geometry)
    assert not accepted
    assert all(r["ambiguous_indices"] == [0, 1, 2] for r in rejected)
    assert [x["index"] for x in rejected[0]["overlaps"]] == [2]


def test_geometry_must_be_indexed_metadata_not_an_unaligned_list():
    items = [detection("a", [0, 0, 10, 10]), detection("b", [1, 1, 9, 9])]
    assert not filter_ambiguous_inventory(items, geometries=[depth_geometry(1.), depth_geometry(4., y=3.)])[0]
