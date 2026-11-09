"""Robot pose and conversions between the body frame, a local ENU plane and WGS-84.

Conventions
  body frame   x forward, y left, z up (REP-103, ``b2/base_link``)
  ENU          x east, y north, z up, metres from a fixed origin (the first valid robot fix)
  heading      compass: radians from true/magnetic north, clockwise positive (what the robot topics use)

The local plane is an equirectangular projection about the origin. Over the few hundred metres a UGV
covers in a mission its error is millimetres, far below GNSS and depth noise.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

EARTH_RADIUS_M = 6378137.0


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compass_to_enu_yaw(heading_rad: float) -> float:
    """Compass heading (0 = north, clockwise) -> ENU yaw (0 = east, counter-clockwise)."""
    return wrap_pi(math.pi / 2 - heading_rad)


def enu_yaw_to_compass(yaw_rad: float) -> float:
    return (math.pi / 2 - yaw_rad) % (2 * math.pi)


def valid_lat_lon(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            and not (abs(lat) < 1e-9 and abs(lon) < 1e-9))


@dataclass(frozen=True)
class Pose:
    """Robot position and compass heading at one instant."""
    lat: float
    lon: float
    alt: float
    heading_rad: float                    # compass heading, already corrected by heading_offset_deg
    position_source: str = ''
    heading_source: str = ''

    @property
    def yaw_enu(self) -> float:
        return compass_to_enu_yaw(self.heading_rad)


class LocalFrame:
    """ENU plane anchored at a WGS-84 origin."""

    def __init__(self, lat: float, lon: float, alt: float = 0.0):
        if not valid_lat_lon(lat, lon):
            raise ValueError(f'Invalid local-frame origin: {lat}, {lon}')
        self.lat0, self.lon0, self.alt0 = float(lat), float(lon), float(alt)
        self._m_per_deg_lat = math.pi / 180 * EARTH_RADIUS_M
        self._m_per_deg_lon = self._m_per_deg_lat * math.cos(math.radians(self.lat0))

    def to_enu(self, lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
        return np.array([(lon - self.lon0) * self._m_per_deg_lon,
                         (lat - self.lat0) * self._m_per_deg_lat, alt - self.alt0])

    def to_geodetic(self, enu) -> tuple[float, float, float]:
        """(lat, lon, alt) of an ENU point."""
        e, n, u = (float(v) for v in np.asarray(enu, dtype=float)[:3])
        return self.lat0 + n / self._m_per_deg_lat, self.lon0 + e / self._m_per_deg_lon, self.alt0 + u

    def lon_lat(self, enu) -> list[float]:
        """GeoJSON coordinate order: [longitude, latitude], 8 decimals (~1 mm)."""
        lat, lon, _ = self.to_geodetic(enu)
        return [round(lon, 8), round(lat, 8)]

    def body_to_enu(self, points_body, pose: Pose) -> np.ndarray:
        """Rotate body-frame points by the heading and translate to the robot's ENU position.

        Roll and pitch are ignored: the platform is treated as level, matching the live system.
        """
        points = np.atleast_2d(np.asarray(points_body, dtype=float))
        yaw = pose.yaw_enu
        c, s = math.cos(yaw), math.sin(yaw)
        robot = self.to_enu(pose.lat, pose.lon, pose.alt)
        east = robot[0] + c * points[:, 0] - s * points[:, 1]
        north = robot[1] + s * points[:, 0] + c * points[:, 1]
        up = robot[2] + points[:, 2]
        return np.column_stack([east, north, up])


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))
