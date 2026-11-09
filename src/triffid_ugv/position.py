"""Robot position and heading from several sources, in configured priority order.

Position sources (default priority as on the live system)
  status   ugv_status_interfaces/UnitreeStatus latitude/longitude while gnss_valid (RTK fix)
  api      HTTP JSON polled in the background ({"message": {"latitude", "longitude", "gnss_valid"}})
  fix      sensor_msgs/NavSatFix, median-filtered (a few-sample window rejects GNSS jumps)
  fixed    configured seed position (bench tests, static deployments)
Heading sources
  magnetometer   std_msgs/Float32 compass heading in degrees (0 north, clockwise)
  status         UnitreeStatus.heading in radians (GNSS converter; convention configurable)
  fixed          configured heading

A source is used only while fresh (received within its timeout). ROS-free: the node feeds messages in.
"""
from __future__ import annotations

from collections import deque
import json
import logging
import math
import statistics
import threading
import time
from urllib.request import Request, urlopen

from .config import PositionConfig
from .geodesy import Pose, enu_yaw_to_compass, valid_lat_lon

log = logging.getLogger(__name__)


class _Reading:
    __slots__ = ('value', 'received')

    def __init__(self):
        self.value, self.received = None, None

    def set(self, value, now=None):
        self.value, self.received = value, time.monotonic() if now is None else now

    def fresh(self, timeout_s, now=None):
        now = time.monotonic() if now is None else now
        return self.value is not None and self.received is not None and now - self.received <= timeout_s


class PoseEstimator:
    def __init__(self, config: PositionConfig, opener=urlopen, clock=time.monotonic):
        self.config = config
        self.clock = clock
        self._lock = threading.Lock()
        self._status, self._fix, self._api = _Reading(), _Reading(), _Reading()
        self._magnetometer, self._status_heading = _Reading(), _Reading()
        self._fix_window = deque(maxlen=max(1, int(config.fix_median_window)))
        self._last_sources = (None, None)
        self._opener = opener
        self._stop = threading.Event()
        self._api_thread = None

    # ---- inputs -------------------------------------------------------------------------------
    def on_status(self, lat, lon, gnss_valid: bool, heading_rad=None, alt=None):
        now = self.clock()
        with self._lock:
            if gnss_valid and valid_lat_lon(lat, lon):
                self._status.set((float(lat), float(lon), None if alt is None else float(alt)), now)
            if gnss_valid and heading_rad is not None and math.isfinite(float(heading_rad)):
                heading = float(heading_rad)
                if self.config.status_heading_convention == 'enu':
                    heading = enu_yaw_to_compass(heading)
                self._status_heading.set(heading % (2 * math.pi), now)

    def on_fix(self, lat, lon, alt, status_ok: bool = True):
        if not status_ok or not valid_lat_lon(lat, lon):
            return
        alt = float(alt) if alt is not None and math.isfinite(float(alt)) else 0.0
        with self._lock:
            self._fix_window.append((float(lat), float(lon), alt))
            lats, lons, alts = zip(*self._fix_window)
            self._fix.set((statistics.median(lats), statistics.median(lons), statistics.median(alts)), self.clock())

    def on_heading_deg(self, degrees):
        try:
            value = float(degrees)
        except (TypeError, ValueError):
            return
        if math.isfinite(value):
            with self._lock:
                self._magnetometer.set(math.radians(value % 360.0), self.clock())

    # ---- HTTP position fallback ---------------------------------------------------------------
    def start_api(self):
        if 'api' not in self.config.sources or not self.config.api_url or self._api_thread:
            return
        self._api_thread = threading.Thread(target=self._poll_api, name='position-api', daemon=True)
        self._api_thread.start()

    def _poll_api(self):
        failures = 0
        while not self._stop.is_set():
            try:
                request = Request(self.config.api_url, headers={'Accept': 'application/json'})
                with self._opener(request, timeout=self.config.api_timeout_s) as response:
                    payload = json.loads(response.read(1 << 16).decode('utf-8'))
                message = payload.get('message') or {}
                if message.get('gnss_valid', True) and valid_lat_lon(message.get('latitude'), message.get('longitude')):
                    with self._lock:
                        self._api.set((float(message['latitude']), float(message['longitude']), None), self.clock())
                failures = 0
            except Exception as error:   # network, JSON, schema: keep polling, log rarely
                failures += 1
                if failures in (1, 10) or failures % 100 == 0:
                    log.warning('Position API unavailable (%d failures): %s', failures, error)
            self._stop.wait(self.config.api_poll_s)

    def close(self):
        self._stop.set()

    # ---- output -------------------------------------------------------------------------------
    def _position(self, now):
        c = self.config
        fix_alt = self._fix.value[2] if self._fix.fresh(c.fix_timeout_s, now) else None
        for source in c.sources:
            if source == 'status' and self._status.fresh(c.status_timeout_s, now):
                lat, lon, alt = self._status.value
                return lat, lon, alt if alt is not None else (fix_alt or c.fixed_alt), source
            if source == 'fix' and self._fix.fresh(c.fix_timeout_s, now):
                return (*self._fix.value, source)
            if source == 'api' and self._api.fresh(max(c.status_timeout_s, 2.5 * c.api_poll_s), now):
                lat, lon, _ = self._api.value
                return lat, lon, fix_alt or c.fixed_alt, source
            if source == 'fixed' and c.fixed_lat is not None:
                return c.fixed_lat, c.fixed_lon, c.fixed_alt, source
        return None

    def _heading(self, now):
        c = self.config
        for source in c.heading_sources:
            if source == 'magnetometer' and self._magnetometer.fresh(c.heading_timeout_s, now):
                return self._magnetometer.value, source
            if source == 'status' and self._status_heading.fresh(c.heading_timeout_s, now):
                return self._status_heading.value, source
            if source == 'fixed' and c.fixed_heading_deg is not None:
                return math.radians(c.fixed_heading_deg), source
        return None

    def pose(self) -> tuple[Pose | None, str]:
        """Current pose, or (None, reason)."""
        now = self.clock()
        with self._lock:
            position, heading = self._position(now), self._heading(now)
        sources = (position[3] if position else None, heading[1] if heading else None)
        if sources != self._last_sources:
            log.info('Pose sources: position=%s heading=%s', *sources)
            self._last_sources = sources
        if position is None:
            return None, 'no fresh position (status/fix/api) and no fixed position'
        if heading is None:
            return None, 'no fresh heading (magnetometer/status) and no fixed heading'
        lat, lon, alt, position_source = position
        heading_rad = (heading[0] + math.radians(self.config.heading_offset_deg)) % (2 * math.pi)
        return Pose(lat, lon, float(alt or 0.0), heading_rad, position_source, heading[1]), ''

    def status(self) -> dict:
        now = self.clock()
        c = self.config
        with self._lock:
            return dict(status=self._status.fresh(c.status_timeout_s, now), fix=self._fix.fresh(c.fix_timeout_s, now),
                        api=self._api.fresh(max(c.status_timeout_s, 2.5 * c.api_poll_s), now),
                        magnetometer=self._magnetometer.fresh(c.heading_timeout_s, now),
                        status_heading=self._status_heading.fresh(c.heading_timeout_s, now))

