import math
import pytest

from tb3_locateanything.geometry import CameraCalibration, Quaternion, Transform3D, Vector3
from tb3_locateanything.grounding_core import check_snapshot_times, localize_box, one_box, referring_expression


def test_descriptive_words_survive_navigation_verb():
    assert referring_expression("Move to the person wearing a green shirt.") == "the person wearing a green shirt"
    assert referring_expression("the red chair beside the table") == "the red chair beside the table"


def test_boxes_use_original_image_dimensions():
    assert one_box("<box><100><200><700><900></box>", 640, 480) == [64, 96, 448, 432]


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
