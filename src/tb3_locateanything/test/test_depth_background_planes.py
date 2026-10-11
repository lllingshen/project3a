"""Background plane provenance must survive pixel removal before flood fill."""
import numpy as np
import pytest
from tb3_locateanything.depth_core import localize_depth_box
from tb3_locateanything.geometry import CameraCalibration, Quaternion, Transform3D, Vector3

CAL = CameraCalibration(120, 100, 100., 100., 60., 50., 'optical')
TF = Transform3D(Vector3(0., 0., 1.5), Quaternion(-.5, .5, -.5, .5))
BOX = (35., 15., 85., 85.)


def parallel_patch(offset):
    depth = np.full((100, 120), 3., dtype=np.float32)
    depth[18:82, 38:82] = 3.-offset
    return depth


@pytest.mark.parametrize('offset', [.024999, .025, .025001, .035, -.025001])
def test_near_coplanar_patch_does_not_become_object_after_outside_pixels_removed(offset):
    # Float32 encoding can put nominal 25 mm residuals slightly on either side
    # of the pixel threshold. A fitted inside plane is still coplanar under
    # the existing 40 mm component rule; that outside evidence must survive.
    with pytest.raises(ValueError, match='no depth surface'):
        localize_depth_box(BOX, CAL, parallel_patch(offset), TF)


@pytest.mark.parametrize('offset', [.041, .1, 1.])
def test_separate_parallel_object_retains_original_plane_distance_boundary(offset):
    point, geometry = localize_depth_box(BOX, CAL, parallel_patch(offset), TF)
    assert point.x == pytest.approx(3.-offset)
    assert geometry['depth_outside_planes'] >= 1


def test_nonplanar_object_near_background_is_not_rejected_as_one_plane():
    depth = parallel_patch(.06)
    # A broad depth curve is connected but does not satisfy the plane fit.
    u = np.linspace(-1., 1., 44)
    depth[18:82, 38:82] = 2.6 + .25*u*u
    point, geometry = localize_depth_box(BOX, CAL, depth, TF)
    assert point.x < 2.9
    assert geometry['depth_surface_pixels'] > 500
