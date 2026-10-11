import math
import pytest

from tb3_locateanything.geometry import CameraCalibration, Quaternion, Transform3D, Vector3
from tb3_locateanything.grounding_core import (
    catalog_boxes, check_snapshot_times, described_class, localize_box, one_box,
    referring_expression, validate_localizable_box, validate_surface_visibility,
)


def test_descriptive_words_survive_navigation_verb():
    assert referring_expression("Move to the person wearing a green shirt.") == "the person wearing a green shirt"
    assert referring_expression("the red chair beside the table") == "the red chair beside the table"


@pytest.mark.parametrize("command,expected", [
    ("Move to the green table.", "table"),
    ("go to the left chair", "chair"),
    ("go to the right person", "person"),
    ("go to the object left of the table", None),
    ("Move to the chair to the right of the red table.", "chair"),
    ("Move to the red object I can sit on.", None),
    ("Move to the object beside the forklift.", None),
    ("Don't move to the table; move to the chair instead.", None),
    ("Move to the item near the table.", None),
    ("Move to the small FIRE_EXTINGUISHER.", "fire_extinguisher"),
    ("Move to the very unusually large remarkably shiny table.", None),
])
def test_explicit_class_hint_preserves_target_role_and_declines_ambiguous_prefixes(command, expected):
    # Visual memory also exposes a generic 'object' fallback class; it must
    # never turn an affordance query into an incompatible-class rejection.
    labels = ["chair", "person", "table", "forklift", "fire extinguisher", "object"]
    assert described_class(command, labels) == expected


def test_boxes_use_original_image_dimensions():
    assert one_box("<box><100><200><700><900></box>", 640, 480) == [64, 96, 448, 432]


def test_catalog_keeps_all_localizable_instances_and_accepts_absence():
    answer = "<box><100><200><350><900></box><box><550><250><850><850></box>"
    assert catalog_boxes(answer, 640, 480) == [
        [64, 96, 224, 432], [352, 120, 544, 408]]
    assert catalog_boxes("<box>None</box>", 640, 480) == []


def test_catalog_does_not_discard_complete_objects_beside_clipped_ones():
    answer = ("<box><0><0><999><998></box><box><0><200><37><534></box>"
              "<box><100><200><350><900></box>")
    assert catalog_boxes(answer, 640, 480) == [[64, 96, 224, 432]]


@pytest.mark.parametrize("answer", [
    "", "none", "<box><10><10><1001><100></box>",
    "<box>None</box><box><100><200><350><900></box>",
    "<box><100><200><350><900></box><box>",
])
def test_malformed_catalog_fails_instead_of_creating_partial_memory(answer):
    with pytest.raises(ValueError, match="malformed model output"):
        catalog_boxes(answer, 640, 480)


@pytest.mark.parametrize("answer", [
    "<box><0><0><998><998></box>",
    "<box><1><300><999><500></box>",
    "<box><300><1><500><999></box>",
])
def test_whole_frame_and_opposite_edge_targets_are_rejected(answer):
    with pytest.raises(ValueError, match="whole image|horizontal image edge|opposite image edges"):
        one_box(answer, 640, 480)
    assert catalog_boxes(answer, 640, 480) == []


@pytest.mark.parametrize("answer", [
    "<box><0><0><86><511></box>",  # left fragment seen during the warehouse sweep
    "<box><0><202><37><534></box>",  # narrow left fragment, not a complete chair
    "<box><900><200><1000><800></box>",  # right clipping has the same bearing error
])
def test_one_horizontal_edge_is_enough_to_make_a_target_unreliable(answer):
    with pytest.raises(ValueError, match="horizontal image edge"):
        one_box(answer, 640, 480)
    assert catalog_boxes(answer, 640, 480) == []


@pytest.mark.parametrize("answer,expected", [
    ("<box><100><0><350><800></box>", [64, 0, 224, 384]),
    ("<box><600><200><850><1000></box>", [384, 96, 544, 480]),
])
def test_one_vertical_edge_does_not_remove_a_valid_horizontal_bearing(answer, expected):
    assert one_box(answer, 640, 480) == expected


@pytest.mark.parametrize("answer,expected", [
    ("<box><66><21><372><548></box>", [42.24, 10.08, 238.08, 263.04]),
    ("<box><604><19><959><547></box>", [386.56, 9.12, 613.76, 262.56]),
])
def test_complete_red_and_blue_chair_observations_remain_usable(answer, expected):
    assert one_box(answer, 640, 480) == expected


@pytest.mark.parametrize("box", [
    [0, 10, float("nan"), 40], [-1, 10, 20, 30], [30, 10, 20, 30],
])
def test_geometry_guard_rejects_invalid_boxes_before_ray_computation(box):
    with pytest.raises(ValueError):
        validate_localizable_box(box, 640, 480)


@pytest.mark.parametrize("answer", [
    "", "none", "<box>None</box>", "<box><-1><0><50><80></box>",
    "<box><10><10><1001><100></box>", "<box><40><20><10><30></box>",
    "<box><0><0><20><30></box><box><40><40><50><50></box>",
    "<box>None</box><box><0><0><20><30></box>",
    "<box><0><0><20><30></box><box>",
])
def test_invalid_absent_or_ambiguous_target_does_not_localize(answer):
    with pytest.raises(ValueError):
        one_box(answer, 640, 480)


def test_capture_rejects_stale_and_mismatched_sensors():
    check_snapshot_times(2_000_000_000, 1_800_000_000, 1_810_000_000, 1_800_000_000, 1_850_000_000)
    with pytest.raises(ValueError, match="stale"):
        check_snapshot_times(2_000_000_000, 1_000_000_000, 1_000_000_000, 1_000_000_000, 1_000_000_000)
    with pytest.raises(ValueError, match="synchronized"):
        check_snapshot_times(2_000_000_000, 1_800_000_000, 1_300_000_000, 1_800_000_000, 1_800_000_000)


CALIBRATION = CameraCalibration(640, 480, 320, 320, 320, 240, "camera_optical")
# Optical z forward -> scan x forward; optical x right -> scan -y.
CAMERA_TO_SCAN = Transform3D(Vector3(0, 0, 0), Quaternion(-0.5, 0.5, -0.5, 0.5))
MAP_TF = Transform3D(Vector3(10, 20, 0), Quaternion(0, 0, math.sin(math.pi/4), math.cos(math.pi/4)))


def fuse(box, ranges):
    return localize_box(box, CALIBRATION, CAMERA_TO_SCAN, MAP_TF,
                        ranges, 0, 2*math.pi/360, 0.12, 3.5)


def test_optical_ray_scan_wrap_and_map_rotation():
    ranges = [float("inf")] * 360
    # Optical center points at scan ray 0; include both sides of its seam.
    for idx in (358, 359, 0, 1, 2):
        ranges[idx] = 2.0
    point, record = fuse([280, 100, 360, 400], ranges)
    assert point.x == pytest.approx(10)
    assert point.y == pytest.approx(22)
    assert record["valid_scan_rays"] == 5


def test_right_image_target_has_negative_scan_bearing():
    point, record = fuse([400, 100, 500, 400], [2.0] * 360)
    assert record["bearing_rad"] < 0
    assert point.x > 10  # rotated through +90 degrees in map


def test_no_range_and_background_discontinuity_rejected():
    with pytest.raises(ValueError, match="no usable"):
        fuse([280, 100, 360, 400], [float("inf")] * 360)
    ranges = [2.0] * 360
    ranges[0] = 3.3
    with pytest.raises(ValueError, match="mixed"):
        fuse([280, 100, 360, 400], ranges)


def test_surface_support_stops_at_foreground_gap_and_keeps_scan_wrap():
    ranges = [3.2] * 360
    for idx in list(range(354, 360)) + list(range(7)):
        ranges[idx] = 1.5
    _, record = fuse([180, 100, 460, 400], ranges)
    support = record["support_points"]
    assert len(support) == 13
    assert all(math.hypot(x-10, y-20) == pytest.approx(1.5) for x, y in support)
    assert min(x for x, _ in support) < 10 < max(x for x, _ in support)


def test_surface_support_is_bounded_and_stays_inside_box():
    _, record = fuse([80, 100, 560, 400], [2.0] * 360)
    assert 2 <= len(record["support_points"]) <= 32
    for x, y in record["support_points"]:
        assert abs(math.atan2(-(x-10), y-20)) <= math.atan(240/320) + 1e-6


def test_surface_support_does_not_bridge_missing_returns():
    ranges = [2.0] * 360
    ranges[4] = ranges[356] = float("inf")
    _, record = fuse([180, 100, 460, 400], ranges)
    assert len(record["support_points"]) == 7


def test_single_laser_speck_cannot_attach_an_occluded_target_to_a_shelf():
    ranges = [float("inf")] * 360
    ranges[0] = 2.95
    with pytest.raises(ValueError, match="fewer than two usable"):
        fuse([313, 160, 334, 250], ranges)
    ranges[359] = 2.96
    _, record = fuse([300, 160, 340, 250], ranges)
    assert len(record["support_points"]) == 2


def test_visible_top_behind_foreground_shelf_does_not_borrow_shelf_range():
    # This frame's extinguisher top is visible above the shelf. The laser
    # hits that shelf near the image horizon, outside the extinguisher box.
    with pytest.raises(ValueError, match="projects outside target box"):
        fuse([285, 147, 313, 180], [.88] * 360)


def test_laser_reprojection_includes_camera_translation():
    # A camera 0.1 m above the laser sees its surface below the horizon.
    camera_above = Transform3D(Vector3(0, 0, .1), CAMERA_TO_SCAN.rotation)
    _, record = localize_box([280, 250, 360, 280], CALIBRATION,
        camera_above, MAP_TF, [2.] * 360, 0, 2*math.pi/360, .12, 3.5)
    assert record["laser_pixel"] == pytest.approx([320, 256])
    with pytest.raises(ValueError, match="projects outside target box"):
        localize_box([280, 230, 360, 245], CALIBRATION,
            camera_above, MAP_TF, [2.] * 360, 0, 2*math.pi/360, .12, 3.5)


def test_clipped_foreground_box_is_still_an_occluder_for_background_target():
    forklift = [425., 49., 609., 252.]
    rack = [523., 0., 640., 269.]
    with pytest.raises(ValueError, match="overlaps another detected object"):
        validate_surface_visibility(forklift, [527., 244.], [forklift, rack])
    validate_surface_visibility(forklift, [480., 244.], [forklift, rack])
    validate_surface_visibility(forklift, [527., 244.], [forklift])


def test_support_catches_occlusion_that_median_center_misses():
    # Saved view68: the visible forklift is behind a rack. The synthetic
    # median center projects just left of the rack, but every actual ray
    # supporting it lands on that rack's low foreground beam.
    forklift = [43.52, 112.8, 188.16, 250.08]
    rack = [108.16, 0., 478.08, 269.76]
    support = [[u, 233.5] for u in [165.06, 154.58, 143.91, 133.15, 122.30, 111.27]]
    validate_surface_visibility(forklift, [104.23, 233.58], [forklift, rack])
    with pytest.raises(ValueError, match="laser support overlaps"):
        validate_surface_visibility(forklift, [104.23, 233.58], [forklift, rack], support)
    # Most of the foreground rack's own laser face is visible and clear.
    rack_support = [[u, 233.] for u in range(115, 476, 10)]
    validate_surface_visibility(rack, [291.64, 232.31], [forklift, rack], rack_support)


@pytest.mark.parametrize("inside_count, rejected", [(1, False), (2, True), (3, True)])
def test_surface_support_needs_at_least_half_covered(inside_count, rejected):
    box, other = [50, 50, 150, 150], [130, 50, 230, 150]
    support = [[140, 100]] * inside_count + [[100, 100]] * (4-inside_count)
    if rejected:
        with pytest.raises(ValueError, match="laser support overlaps"):
            validate_surface_visibility(box, [100, 100], [box, other], support)
    else:
        validate_surface_visibility(box, [100, 100], [box, other], support)


def test_same_instance_box_is_not_its_own_support_occluder():
    box, duplicate = [50, 50, 150, 150], [55, 55, 150, 150]
    validate_surface_visibility(box, [100, 100], [box, duplicate], [[100, 100]] * 4)


def test_support_pixels_project_actual_returns_with_camera_baseline():
    camera_above = Transform3D(Vector3(0, 0, .1), CAMERA_TO_SCAN.rotation)
    _, record = localize_box([280, 250, 360, 280], CALIBRATION,
        camera_above, MAP_TF, [2.] * 360, 0, 2*math.pi/360, .12, 3.5)
    assert len(record["support_pixels"]) == len(record["support_points"])
    for (x, y), (u, v) in zip(record["support_points"], record["support_pixels"]):
        # Undo the 90-degree frozen map transform and optical-frame rotation.
        scan_x, scan_y = y-20, -(x-10)
        assert u == pytest.approx(320-320*scan_y/scan_x)
        assert v == pytest.approx(240+32/scan_x)


def test_long_support_cannot_substitute_for_missing_center_returns():
    # A coherent surface at the edge of a wide box supplies many rays, but
    # only one return in its center window. Do not interpolate a target there.
    ranges = [math.inf] * 360
    for index in range(2, 10):
        ranges[index] = 2.
    with pytest.raises(ValueError, match="fewer than two usable"):
        fuse([180, 100, 460, 400], ranges)


def test_median_pixel_tolerance_does_not_hide_most_actual_support_outside_box():
    lateral_camera = Transform3D(Vector3(0., .268, 0.), CAMERA_TO_SCAN.rotation)
    # The synthetic center is u=362.88 (within the old 3 px padding), while
    # most measured rays project beyond this box's strict right edge u=360.
    with pytest.raises(ValueError, match="most laser support projects outside"):
        localize_box([280, 100, 360, 400], CALIBRATION, lateral_camera,
            MAP_TF, [2.] * 360, 0., 2*math.pi/360, .12, 3.5)
    visible_camera = Transform3D(Vector3(0., .1, 0.), CAMERA_TO_SCAN.rotation)
    _, record = localize_box([280, 100, 360, 400], CALIBRATION, visible_camera,
        MAP_TF, [2.] * 360, 0., 2*math.pi/360, .12, 3.5)
    assert sum(280 <= u <= 360 for u, _ in record['support_pixels']) > len(record['support_pixels'])/2


def plane_ranges(normal_angle=0., offset=2., count=360):
    """Intersections with normal dot XY = offset; other rays see nothing."""
    result = []
    for index in range(count):
        denominator = math.cos(2*math.pi*index/count - normal_angle)
        result.append(offset/denominator if denominator > 1e-8 else math.inf)
    return result


def test_wall_shadow_box_rejected_by_continuous_plane_both_sides():
    # RGB can propose a box around a shadow; the laser still sees one wall.
    with pytest.raises(ValueError, match="background plane beyond both sides"):
        fuse([280, 100, 360, 400], plane_ranges())


def test_flat_rack_face_with_depth_breaks_at_its_sides_is_retained():
    wall = plane_ranges(offset=3.)
    rack = plane_ranges(offset=1.5)
    for index in range(360):
        theta = math.atan2(math.sin(math.radians(index)), math.cos(math.radians(index)))
        if abs(theta) <= .145:
            wall[index] = rack[index]
    point, record = fuse([280, 100, 360, 400], wall)
    assert point.y == pytest.approx(21.5, abs=.002)
    assert len(record["support_points"]) >= 4


def test_nonplanar_rack_surface_is_not_classified_as_background_plane():
    # Two connected faces meeting at a corner: x = 1.5 + abs(y).
    ranges = []
    for index in range(360):
        theta = math.radians(index)
        denominator = math.cos(theta)-abs(math.sin(theta))
        ranges.append(1.5/denominator if denominator > 1e-8 else math.inf)
    _, record = fuse([180, 100, 460, 400], ranges)
    support = record["support_points"]
    assert len(support) >= 4
    assert max(y for _, y in support)-min(y for _, y in support) > .1


def test_only_one_visible_plane_extension_is_inconclusive():
    ranges = plane_ranges()
    for index in range(360):
        theta = math.atan2(math.sin(math.radians(index)), math.cos(math.radians(index)))
        if theta < -.14:
            ranges[index] = math.inf
    _, record = fuse([280, 100, 360, 400], ranges)
    assert len(record["support_points"]) >= 4


@pytest.mark.parametrize("count", [360, 720])
def test_oblique_wall_uses_predicted_ray_spacing_not_fixed_gap(count):
    ranges = plane_ranges(math.radians(65), .35, count)
    increment = 2*math.pi/count
    # These neighboring, valid wall returns are naturally >10 cm apart
    # under oblique viewing, despite lying on exactly the same plane.
    index = round(math.radians(344)/increment)
    separation = math.sqrt(ranges[index]**2 + ranges[index+1]**2
        - 2*ranges[index]*ranges[index+1]*math.cos(increment))
    if count == 360:
        assert separation > .1
    with pytest.raises(ValueError, match="background plane beyond both sides"):
        localize_box([300, 100, 380, 400], CALIBRATION, CAMERA_TO_SCAN,
            MAP_TF, ranges, 0., increment, .12, 3.5)
