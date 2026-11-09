import numpy as np
import pytest

from triffid_ugv.detectors.base import Detection
from triffid_ugv.geometry import (Intrinsics, intrinsics_from_camera_info, rigid, rotation_from_quaternion,
                                  robust_cluster, sample_points, static_optical_to_body, transform)


def test_optical_axes_map_to_body_axes():
    m = static_optical_to_body([0.4, 0.0, 0.3], [0, 0, 0])
    # 2 m straight ahead along the optical axis -> 2 m forward of the camera
    assert transform([[0, 0, 2.0]], m)[0] == pytest.approx([2.4, 0.0, 0.3])
    # optical +x (right) -> body -y; optical +y (down) -> body -z
    assert transform([[1.0, 0, 0]], m)[0] == pytest.approx([0.4, -1.0, 0.3])
    assert transform([[0, 1.0, 0]], m)[0] == pytest.approx([0.4, 0.0, -0.7])


def test_static_matches_recorded_tf():
    """base_link -> rs_front_link -> rs_front_color_frame -> optical (from the PUI bag) ~ static default."""
    q = (-0.5, 0.5, -0.5, 0.5)       # color_frame -> color_optical_frame
    tf = rigid(np.eye(3), [0.392, 0.0, 0.279]) @ rigid(np.eye(3), [0, 0.015, 0]) @ rigid(rotation_from_quaternion(*q), [0, 0, 0])
    static = static_optical_to_body([0.392, 0.015, 0.279], [0, 0, 0])
    assert tf == pytest.approx(static, abs=1e-9)


def test_sampling_back_projects_mask_depth():
    intr = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480)
    depth = np.zeros((480, 640), np.float32)
    depth[200:280, 300:340] = 4.0
    mask = np.zeros((480, 640), bool)
    mask[200:280, 300:340] = True
    points = sample_points(Detection((300, 200, 340, 280), 0.9, 23, 'citizen', mask), depth, intr, 0.3, 20)
    assert points[:, 2] == pytest.approx(4.0)
    assert np.median(points[:, 0]) == pytest.approx(0.0, abs=0.05)


def test_sampling_without_mask_and_invalid_depth():
    intr = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480)
    depth = np.zeros((480, 640), np.float32)
    assert sample_points(Detection((300, 200, 340, 280), 0.9, 23, 'citizen'), depth, intr, 0.3, 20) is None
    depth[:] = 6.0
    assert len(sample_points(Detection((300, 200, 340, 280), 0.9, 23, 'citizen'), depth, intr, 0.3, 20)) > 10


def test_robust_cluster_drops_background_leak():
    near = np.column_stack([np.zeros(50), np.zeros(50), np.full(50, 3.0)])
    far = np.column_stack([np.zeros(10), np.zeros(10), np.full(10, 15.0)])
    kept = robust_cluster(np.vstack([near, far]))
    assert kept[:, 2].max() == pytest.approx(3.0)


def test_camera_info_and_scaling():
    intr = intrinsics_from_camera_info([600, 0, 320, 0, 600, 240, 0, 0, 1], 640, 480)
    half = intr.scaled_to(320, 240)
    assert (half.fx, half.cx) == (300.0, 160.0)
    with pytest.raises(ValueError):
        intrinsics_from_camera_info([0] * 9, 640, 480)
    with pytest.raises(ValueError, match='aspect ratio'):     # 4:3 calibration on a 16:9 image
        intr.scaled_to(1280, 720)
