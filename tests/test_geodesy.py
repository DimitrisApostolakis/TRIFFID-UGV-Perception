import math

import pytest

from triffid_ugv.geodesy import LocalFrame, Pose, compass_to_enu_yaw, haversine_m, valid_lat_lon

LAT, LON = 37.9838, 23.7275


@pytest.mark.parametrize('heading_deg, east, north', [(0, 0, 10), (90, 10, 0), (180, 0, -10), (270, -10, 0)])
def test_forward_point_follows_compass_heading(heading_deg, east, north):
    frame = LocalFrame(LAT, LON)
    pose = Pose(LAT, LON, 0.0, math.radians(heading_deg))
    enu = frame.body_to_enu([10.0, 0.0, 0.0], pose)[0]
    assert enu[0] == pytest.approx(east, abs=1e-9) and enu[1] == pytest.approx(north, abs=1e-9)


def test_left_is_west_when_facing_north():
    frame = LocalFrame(LAT, LON)
    enu = frame.body_to_enu([0.0, 5.0, 1.0], Pose(LAT, LON, 100.0, 0.0))[0]
    assert enu == pytest.approx([-5.0, 0.0, 101.0])   # origin altitude 0, robot at 100 m


def test_matches_live_branch_formula():
    """Same result as the original GeoJSON bridge (_body_to_enu) for an arbitrary heading."""
    heading, x, y = math.radians(37.0), 7.0, -3.0
    rel = math.atan2(-y, x)
    d = math.hypot(x, y)
    live = (d * math.sin(heading + rel), d * math.cos(heading + rel))
    enu = LocalFrame(LAT, LON).body_to_enu([x, y, 0], Pose(LAT, LON, 0, heading))[0]
    assert enu[:2] == pytest.approx(live)


def test_round_trip_and_distance():
    frame = LocalFrame(LAT, LON, 50)
    lat, lon, alt = frame.to_geodetic([30.0, -40.0, 2.0])
    assert frame.to_enu(lat, lon, alt) == pytest.approx([30.0, -40.0, 2.0], abs=1e-6)
    assert haversine_m(LAT, LON, lat, lon) == pytest.approx(50.0, rel=1e-3)


def test_enu_yaw_and_validity():
    assert compass_to_enu_yaw(0.0) == pytest.approx(math.pi / 2)
    assert not valid_lat_lon(0, 0) and not valid_lat_lon(91, 0) and valid_lat_lon(LAT, LON)
