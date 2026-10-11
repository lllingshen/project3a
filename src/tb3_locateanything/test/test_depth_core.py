import numpy as np
import pytest

from tb3_locateanything.depth_core import AmbiguousDepthError, _ambiguity_error, depth_in_metres, localize_depth_box
from tb3_locateanything.geometry import CameraCalibration, Quaternion, Transform3D, Vector3


CAL = CameraCalibration(120, 100, 100., 100., 60., 50., "optical")
TF = Transform3D(Vector3(0., 0., 1.5), Quaternion(-.5, .5, -.5, .5))
BOX = (35., 15., 85., 85.)


def scene():
    depth = np.full((100, 120), 6., dtype=np.float32)
    depth[18:82, 38:82] = 3.
    return depth


def test_depth_units_and_invalid_values():
    assert depth_in_metres(np.array([[2500, 0]], np.uint16), "16UC1")[0, 0] == 2.5
    x = depth_in_metres(np.array([[1., np.inf, 0., -1.]], np.float32), "32FC1")
    assert x[0, 0] == 1. and np.isnan(x[0, 1:]).all()
    with pytest.raises(ValueError, match="32FC1"):
        depth_in_metres(np.ones((2, 2)), "rgb8")


def test_bounded_surface_uses_axial_depth_and_frozen_transform():
    point, meta = localize_depth_box(BOX, CAL, scene(), TF)
    assert point.x == pytest.approx(3.)
    assert abs(point.y) < .05
    assert meta["depth_m"] == 3.
    assert meta["localization_source"] == "aligned_depth"
    assert len(meta["support_points"]) <= 64


def test_foreground_pole_crossing_box_does_not_supply_target_depth():
    depth = scene()
    depth[:, 53:65] = 1.  # Extends outside box: foreground occluder.
    point, meta = localize_depth_box(BOX, CAL, depth, TF)
    assert point.x == pytest.approx(3.)
    assert all(p[0] == pytest.approx(3.) for p in meta["support_points"])
    assert meta["depth_outside_planes"] >= 1


def test_hollow_frame_does_not_use_background_through_its_hole():
    depth = np.full((100, 120), 6., dtype=np.float32)
    depth[18:82, 38:43] = depth[18:82, 77:82] = 3.
    depth[18:23, 38:82] = depth[77:82, 38:82] = 3.
    # Background connects to outside through an opening in the frame.
    depth[18:23, 52:66] = 6.
    point, _ = localize_depth_box(BOX, CAL, depth, TF)
    assert point.x == pytest.approx(3.)


@pytest.mark.parametrize("slanted", [False, True])
def test_closed_thin_frame_hole_cannot_be_selected_as_background_object(slanted):
    depth = np.full((100, 120), 6., dtype=np.float32)
    if slanted:
        # An actual slanted plane has inverse depth affine in pixel u.
        depth[:] = 6. / (1. + .002*np.arange(120))[None, :]
    depth[8:42, 38:39] = depth[8:42, 81:82] = 3.
    depth[8:9, 38:82] = depth[41:42, 38:82] = 3.
    point, _ = localize_depth_box((35., 5., 85., 45.), CAL, depth, TF)
    assert point.x == pytest.approx(3.)


def test_two_bounded_layers_are_rejected_instead_of_picking_nearer():
    depth = np.full((100, 120), 6., dtype=np.float32)
    depth[20:78, 38:57] = 2.
    depth[20:78, 62:81] = 4.
    with pytest.raises(ValueError, match="multiple separated"):
        localize_depth_box(BOX, CAL, depth, TF)


def test_wall_shadow_box_has_no_bounded_depth_surface():
    with pytest.raises(ValueError, match="no depth surface"):
        localize_depth_box(BOX, CAL, np.full((100, 120), 3., np.float32), TF)


def test_slanted_surface_is_not_split_into_depth_histogram_peaks():
    depth = scene()
    depth[18:82, 38:82] = np.linspace(2.5, 3.5, 44)[None, :]
    point, _ = localize_depth_box(BOX, CAL, depth, TF)
    assert 2.8 < point.x < 3.2


def test_no_valid_depth_or_wrong_image_shape():
    with pytest.raises(ValueError, match="insufficient"):
        localize_depth_box(BOX, CAL, np.full((100, 120), np.nan), TF)
    with pytest.raises(ValueError, match="dimensions"):
        localize_depth_box(BOX, CAL, np.ones((100, 100)), TF)


def test_deep_object_with_nearby_front_side_and_rear_surfaces_is_retained():
    # Front frame, receding side, and rear face form one measured physical
    # structure spanning 1.28 m. Small depth discontinuities and one missing
    # image column split its surfaces; the returned support must retain both
    # ends instead of treating different median depths as different objects.
    depth = np.full((100, 120), 6., dtype=np.float32)
    depth[18:82, 38:52] = 2.
    depth[18:82, 53:75] = np.linspace(2.08, 3.2, 22)[None, :]
    depth[18:82, 75:82] = 3.28
    point, meta = localize_depth_box(BOX, CAL, depth, TF)
    support = np.asarray(meta["support_points"])
    assert 2. <= point.x <= 3.28
    assert support[:, 0].min() == pytest.approx(2.)
    assert support[:, 0].max() == pytest.approx(3.28)


@pytest.mark.parametrize("has_frame", [False, True])
def test_room_corner_background_does_not_become_a_target_through_a_closed_hole(has_frame):
    # Two measured planes meet at a room corner. Neither fitting one average
    # plane nor relying on connected pixels outside the box is sufficient.
    u = np.arange(120)
    corner = np.minimum(6. / (1. + .006*(u-60)), 6. / (1. - .006*(u-60)))
    depth = np.broadcast_to(corner, (100, 120)).astype(np.float32).copy()
    box = (35., 5., 85., 45.)
    if not has_frame:
        with pytest.raises(ValueError, match="no depth surface"):
            localize_depth_box(box, CAL, depth, TF)
        return
    depth[8:42, 38:39] = depth[8:42, 81:82] = 3.
    depth[8:9, 38:82] = depth[41:42, 38:82] = 3.
    point, meta = localize_depth_box(box, CAL, depth, TF)
    assert point.x == pytest.approx(3.)
    assert all(p[0] == pytest.approx(3.) for p in meta["support_points"])
    assert meta["depth_outside_planes"] >= 2


def test_bounded_foreground_cannot_outvote_a_separate_small_distant_target():
    # Both visible surfaces are bounded by the model's box. The foreground
    # covers most of the image, but no measured connection identifies it as
    # the requested object. Discarding the smaller, distant surface based on
    # pixel majority would silently choose the foreground's 2 m position.
    depth = np.full((100, 120), 6., dtype=np.float32)
    depth[9:76, 24:96] = 2.
    depth[20:39, 55:74] = 4.
    with pytest.raises(ValueError, match="separated|ambiguous"):
        localize_depth_box((20., 5., 100., 80.), CAL, depth, TF)


def test_ambiguity_remains_a_rejection_with_all_measured_layers_and_pixel_pairs():
    depth = np.full((100, 120), 6., dtype=np.float32)
    depth[20:78, 38:57] = 2.
    depth[20:78, 62:81] = 4.
    with pytest.raises(AmbiguousDepthError) as caught:
        localize_depth_box(BOX, CAL, depth, TF)
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.box == BOX and error.hint_usable and error.hint_reason is None
    assert error.group_count == len(error.groups) == 2
    assert not hasattr(error, "point")
    assert sorted(round(np.median(g["odom_xyz"], axis=0)[0]) for g in error.groups) == [2, 4]
    for group in error.groups:
        assert 1 <= len(group["odom_xyz"]) == len(group["pixels"]) <= 64
        assert group["count"] > 64
        for point, (u, v) in zip(group["odom_xyz"], group["pixels"]):
            measured = float(depth[v, u])
            assert point == pytest.approx([measured, -(u-CAL.cx)*measured/CAL.fx,
                                           1.5-(v-CAL.cy)*measured/CAL.fy])
            assert BOX[0] <= u <= BOX[2] and BOX[1] <= v <= BOX[3]


def test_ambiguity_partition_keeps_nonmain_connected_surfaces_together():
    us, vs = np.meshgrid(np.arange(12), np.arange(2))
    components = [{"rows": np.zeros(3, int), "cols": np.arange(3)+offset}
                  for offset in (0, 3, 6)]
    clouds = [np.array([[2., 0., 1.], [2., .01, 1.], [2., .02, 1.]]),
              np.array([[4., 0., 1.], [4., .01, 1.], [4., .02, 1.]]),
              np.array([[4.1, 0., 1.], [4.1, .01, 1.], [4.1, .02, 1.]])]
    error = _ambiguity_error((0., 0., 12., 2.), components, clouds, us, vs)
    assert error.hint_usable and error.group_count == 2
    assert sorted(g["count"] for g in error.groups) == [3, 6]
    second = max(error.groups, key=lambda g:g["count"])
    assert set(p[0] for p in second["odom_xyz"]) == {4., 4.1}


def test_overfull_ambiguity_never_silently_drops_a_group():
    us, vs = np.meshgrid(np.arange(9), np.arange(1))
    components = [{"rows": np.array([0]), "cols": np.array([i])} for i in range(9)]
    clouds = [np.array([[float(i), 0., 1.]]) for i in range(9)]
    error = _ambiguity_error((0., 0., 9., 1.), components, clouds, us, vs)
    assert isinstance(error, ValueError) and not error.hint_usable
    assert error.groups == () and error.group_count == 9
    assert error.hint_reason == "too_many_surface_groups"


def test_ambiguity_bounded_sampling_preserves_all_xyz_and_pixel_extrema():
    us, vs = np.meshgrid(np.arange(100), np.arange(2))
    components = [{"rows": np.zeros(100, int), "cols": np.arange(100)},
                  {"rows": np.ones(100, int), "cols": np.arange(100)}]
    cloud = np.column_stack((np.full(100, 2.), np.linspace(-1., 1., 100), np.full(100, 1.)))
    cloud[31, 2] = 1.8  # Not necessarily included by uniform index sampling.
    other = cloud.copy(); other[:, 0] = 4.
    error = _ambiguity_error((0., 0., 100., 2.), components, [cloud, other], us, vs)
    assert error.hint_usable
    for original, group in zip((cloud, other), error.groups):
        sample = np.array(group["odom_xyz"])
        assert len(sample) <= 64
        assert sample.min(0) == pytest.approx(original.min(0))
        assert sample.max(0) == pytest.approx(original.max(0))
        assert min(p[0] for p in group["pixels"]) == 0
        assert max(p[0] for p in group["pixels"]) == 99
