import math

import pytest

from triffid_ugv.config import load_config
from triffid_ugv.position import PoseEstimator


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def setup(tmp_path):
    clock = Clock()
    config = load_config(None, ['position.sources=[status, fix, fixed]'], root=tmp_path).position
    return PoseEstimator(config, clock=clock), clock


def test_no_pose_without_heading(setup):
    estimator, _ = setup
    estimator.on_fix(37.9, 23.7, 100)
    pose, reason = estimator.pose()
    assert pose is None and 'heading' in reason


def test_priority_freshness_and_fallback(setup):
    estimator, clock = setup
    estimator.on_heading_deg(90)
    estimator.on_fix(37.90, 23.70, 120)
    estimator.on_status(37.91, 23.71, gnss_valid=True, heading_rad=0.0)
    pose, _ = estimator.pose()
    assert (pose.lat, pose.position_source, pose.heading_source) == (37.91, 'status', 'magnetometer')
    assert pose.alt == 120 and math.degrees(pose.heading_rad) == pytest.approx(90)
    clock.t += 5                       # everything stale
    assert estimator.pose()[0] is None
    estimator.on_fix(37.90, 23.70, 120)
    estimator.on_status(37.91, 23.71, gnss_valid=False, heading_rad=0.0)   # no RTK: ignored
    estimator.on_heading_deg(10)
    assert estimator.pose()[0].position_source == 'fix'


def test_status_heading_fallback_and_offset(tmp_path):
    clock = Clock()
    config = load_config(None, ['position.heading_offset_deg=5'], root=tmp_path).position
    estimator = PoseEstimator(config, clock=clock)
    estimator.on_status(37.91, 23.71, gnss_valid=True, heading_rad=math.radians(30))
    pose, _ = estimator.pose()
    assert pose.heading_source == 'status' and math.degrees(pose.heading_rad) == pytest.approx(35)


def test_fix_median_rejects_jump(setup):
    estimator, _ = setup
    estimator.on_heading_deg(0)
    for lat in (37.9, 37.9, 38.5, 37.9, 37.9):
        estimator.on_fix(lat, 23.7, 0)
    assert estimator.pose()[0].lat == 37.9


def test_fixed_fallback(tmp_path):
    config = load_config(None, ['position.fixed_lat=37.5', 'position.fixed_lon=23.5',
                                'position.fixed_heading_deg=180'], root=tmp_path).position
    pose, _ = PoseEstimator(config).pose()
    assert (pose.position_source, pose.heading_source) == ('fixed', 'fixed')
