"""One processed frame: detect -> localize (RGB-D) -> geolocate -> associate -> outputs. No ROS here.

The ROS node (``triffid_ugv.ros.node``) only gathers a synchronized `Frame` once per period and calls
`Pipeline.process`; replay tools and tests can drive the same class directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time

import cv2
import numpy as np

from .config import Config
from .geodesy import LocalFrame, Pose
from .geometry import Intrinsics, robust_cluster, sample_points, transform
from .outputs.geojson import feature_collection, iso, object_feature
from .tracking.world import ObjectMap, Observation
from .visualize import draw

log = logging.getLogger(__name__)


@dataclass
class Frame:
    stamp_ns: int
    bgr: np.ndarray                      # HxWx3 uint8
    depth_m: np.ndarray                  # HxW float32 metres, aligned to the RGB pixels (0 / nan = invalid)
    intrinsics: Intrinsics
    optical_to_body: np.ndarray          # 4x4 body-from-optical transform
    pose: Pose | None = None
    pose_reason: str = ''
    thermal: np.ndarray | None = None    # BGR FLIR frame paired with this RGB stamp (debug recording only)


@dataclass
class FrameReport:
    stamp_ns: int
    detections: int
    localized: int
    geolocated: int
    seconds: float
    skipped_reason: str = ''


def session_name() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


class Pipeline:
    def __init__(self, config: Config, detector, appearance=None, writer=None, api=None, session: str | None = None,
                 debug=None):
        # appearance: image-space tracker (SiftTracker or MasaTracker) whose IDs are hints for the map
        # debug: DebugRecorder (outputs/debug.py) when debug.enabled
        self.config, self.detector, self.appearance = config, detector, appearance
        self.writer, self.api, self.debug = writer, api, debug
        self.session = session or session_name()
        self.map = ObjectMap(config.tracker, config.geometry, id_prefix=f'ugv-{self.session}')
        self.local: LocalFrame | None = None
        self.frames = self.geolocated_frames = 0

    # ---- per frame ----------------------------------------------------------------------------
    def process(self, frame: Frame) -> FrameReport:
        started = time.perf_counter()
        cam = self.config.camera
        h, w = frame.bgr.shape[:2]
        depth = frame.depth_m
        if depth.shape[:2] != (h, w):     # aligned depth at another resolution: resample onto RGB pixels
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        intr = frame.intrinsics.scaled_to(w, h)

        result = self.detector.detect(frame.bgr)
        t_detect = time.perf_counter()
        detections = result.detections
        hints: dict[int, str] = {}
        if self.appearance is not None:
            candidates = detections
            if getattr(self.appearance, 'uses_candidates', False) and result.candidates is not None:
                candidates = result.candidates
            tracked = self.appearance.update(candidates, frame.bgr, frame.stamp_ns)
            position = {id(d): i for i, d in enumerate(candidates)}
            by_candidate = {i: track for i, track in tracked}
            hints = {k: by_candidate[position[id(d)]] for k, d in enumerate(detections)
                     if id(d) in position and position[id(d)] in by_candidate}

        t_appearance = time.perf_counter()
        pose = frame.pose
        if pose is not None and self.local is None:
            self.local = LocalFrame(pose.lat, pose.lon, pose.alt)
            log.info('Local ENU origin: %.7f, %.7f (%s)', pose.lat, pose.lon, pose.position_source)

        observations, records, drawn = [], [], []
        for k, det in enumerate(detections):
            item = dict(bbox=det.bbox, class_name=det.class_name, score=det.score, mask=det.mask)
            drawn.append(item)
            record = dict(class_id=det.class_id, class_name=det.class_name, score=round(det.score, 4),
                          bbox=[round(v, 1) for v in det.bbox])
            if hints.get(k):
                record['track'] = hints[k]
            records.append(record)
            points = sample_points(det, depth, intr, cam.min_depth_m, cam.max_depth_m, cam.max_samples)
            if points is None or len(points) < 3:
                record['localized'] = False
                continue
            points_body = transform(robust_cluster(points), frame.optical_to_body)
            centre_body = np.median(points_body, axis=0)
            record.update(localized=True, body_m=[round(float(v), 3) for v in centre_body],
                          range_m=round(float(np.linalg.norm(centre_body[:2])), 2), depth_points=len(points_body))
            item['range_m'] = record['range_m']
            if pose is None:
                continue
            points_enu = self.local.body_to_enu(points_body, pose)
            centre_enu = self.local.body_to_enu(centre_body, pose)[0]
            lat, lon, alt = self.local.to_geodetic(centre_enu)
            record.update(lat=round(lat, 8), lon=round(lon, 8), alt_m=round(alt, 2))
            obs = Observation(det.class_id, det.class_name, det.score, det.bbox, centre_enu, points_enu,
                              centre_body, frame.stamp_ns, hint=hints.get(k))
            observations.append((obs, record, item))

        t_localize = time.perf_counter()
        if observations:
            self.map.update([o for o, _, _ in observations], frame.stamp_ns)
            for obs, record, item in observations:
                record['object_id'] = item['object_id'] = obs.object_id

        t_track = time.perf_counter()
        self.frames += 1
        self.geolocated_frames += pose is not None
        seconds = time.perf_counter() - started
        report = FrameReport(frame.stamp_ns, len(detections), sum(r.get('localized', False) for r in records),
                             len(observations), seconds, '' if pose else frame.pose_reason)
        annotated = None

        def overlay():
            nonlocal annotated
            if annotated is None:
                header = f'{iso(frame.stamp_ns)}  ' + (f'{pose.lat:.6f},{pose.lon:.6f} hdg {np.degrees(pose.heading_rad):.0f}'
                                                       if pose else f'NO POSE: {frame.pose_reason}')
                annotated = draw(frame.bgr, drawn, header)
            return annotated

        debug_files = None
        if self.debug is not None and self.debug.due(frame.stamp_ns, bool(detections)):
            debug_files = self.debug.submit(frame.stamp_ns, frame.bgr, depth,
                                            overlay() if self.debug.wants_overlay else None, frame.thermal)
        if self.writer is not None:
            image_file = None
            if self.config.outputs.frames.enabled and (detections or self.config.outputs.frames.save_empty):
                image_file = self.writer.save_frame(overlay(), frame.stamp_ns)
            if self.config.outputs.observations:
                self.writer.append_observation(dict(
                    stamp_ns=frame.stamp_ns, time=iso(frame.stamp_ns), processing_s=round(seconds, 3),
                    pose=None if pose is None else dict(lat=pose.lat, lon=pose.lon, alt=pose.alt,
                                                        heading_deg=round(float(np.degrees(pose.heading_rad)), 2),
                                                        position_source=pose.position_source,
                                                        heading_source=pose.heading_source),
                    pose_missing=frame.pose_reason if pose is None else None,
                    intrinsics=intr.source, image_file=image_file, debug_files=debug_files,
                    timing_s=dict(detect=round(t_detect - started, 3), appearance=round(t_appearance - t_detect, 3),
                                  localize=round(t_localize - t_appearance, 3),
                                  track=round(t_track - t_localize, 3), outputs=round(time.perf_counter() - t_track, 3)),
                    detections=records))
        return report

    # ---- map snapshot -------------------------------------------------------------------------
    def features(self) -> list[dict]:
        if self.local is None:
            return []
        tracker = self.config.tracker.backend
        return [object_feature(o, self.local, self.config.geometry, self.config.api.source, tracker)
                for o in sorted(self.map.confirmed(), key=lambda o: o.id)]

    def write_snapshot(self) -> int:
        """Rewrite the local GeoJSON of confirmed objects."""
        features = self.features()
        if self.writer is not None and self.config.outputs.geojson:
            self.writer.write_objects(feature_collection(features, session=self.session,
                                                         generated=iso(time.time_ns()), **self.map.stats()))
        return len(features)

    def upload(self) -> int:
        """Queue the confirmed objects for the map API (non-blocking)."""
        features = self.features()
        if self.api is not None and features:
            self.api.submit(features)
        return len(features)

    def close(self):
        self.write_snapshot()
        self.upload()
        if self.api is not None:
            self.api.close()
        if self.debug is not None:
            self.debug.close()
        if self.writer is not None:
            self.writer.close()
