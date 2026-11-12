import json
import math

import numpy as np

from triffid_ugv.detectors.dummy import DummyDetector
from triffid_ugv.geodesy import Pose
from triffid_ugv.geometry import Intrinsics, static_optical_to_body
from triffid_ugv.outputs import SessionWriter
from triffid_ugv.pipeline import Frame, Pipeline

LAT, LON = 37.9838, 23.7275
S = 1_000_000_000


def frame(t, depth_value=5.0, heading_deg=90.0, pose=True):
    bgr = np.full((480, 640, 3), 90, np.uint8)
    depth = np.full((480, 640), depth_value, np.float32)
    p = Pose(LAT, LON, 100.0, math.radians(heading_deg), 'fix', 'magnetometer') if pose else None
    return Frame(t * S, bgr, depth, Intrinsics(600, 600, 320, 240, 640, 480),
                 static_optical_to_body([0.4, 0.0, 0.3], [0, 0, 0]), p, '' if pose else 'no heading')


def test_end_to_end_geolocation(config, tmp_path):
    writer = SessionWriter(config.paths.outputs, 'test')
    pipeline = Pipeline(config, DummyDetector('fire truck'), writer=writer, session='test')
    for t in range(3):
        report = pipeline.process(frame(t))
        assert report.detections == 1 and report.geolocated == 1
    assert pipeline.write_snapshot() == 1
    pipeline.close()
    geo = json.loads((writer.dir / 'objects.geojson').read_text())
    feature = geo['features'][0]
    lon, lat = feature['geometry']['coordinates']
    # facing east, object 5 m in front of a camera 0.4 m ahead of base_link -> ~5.4 m east
    east = (lon - LON) * math.pi / 180 * 6378137 * math.cos(math.radians(LAT))
    north = (lat - LAT) * math.pi / 180 * 6378137
    assert abs(east - 5.4) < 0.15 and abs(north) < 0.15
    assert feature['properties']['class'] == 'fire truck' and feature['properties']['observations'] == 3
    lines = (writer.dir / 'observations.jsonl').read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])['detections'][0]['object_id'] == feature['id']
    assert (tmp_path / 'outputs' / 'ugv_detections.geojson').is_file()
    assert len(list((writer.dir / 'frames').glob('*.jpg'))) == 3


def test_no_pose_still_logs_but_does_not_map(config):
    writer = SessionWriter(config.paths.outputs, 'nopose')
    pipeline = Pipeline(config, DummyDetector('citizen'), writer=writer, session='nopose')
    report = pipeline.process(frame(0, pose=False))
    assert report.geolocated == 0 and report.localized == 1 and pipeline.features() == []
    record = json.loads((writer.dir / 'observations.jsonl').read_text())
    assert record['pose'] is None and record['pose_missing'] == 'no heading'
    assert record['detections'][0]['range_m'] > 5


def test_missing_depth_is_not_localized(config):
    pipeline = Pipeline(config, DummyDetector('citizen'), session='nodepth')
    report = pipeline.process(frame(0, depth_value=0.0))
    assert report.detections == 1 and report.localized == 0 and report.geolocated == 0


def test_sift_hints_reach_the_observation_log(config):
    from triffid_ugv.tracking.sift import SiftTracker
    writer = SessionWriter(config.paths.outputs, 'sift')
    appearance = SiftTracker(config.tracker.sift, config.tracker.class_groups)
    pipeline = Pipeline(config, DummyDetector('fire truck'), appearance=appearance, writer=writer, session='sift')
    rng = np.random.default_rng(3)
    texture = np.repeat(np.repeat(rng.integers(0, 256, (80, 107, 3), dtype=np.uint8), 6, 0), 6, 1)[:480, :640]
    for t in range(2):
        f = frame(t)
        f.bgr = np.ascontiguousarray(texture)
        pipeline.process(f)
    pipeline.close()
    records = [json.loads(line) for line in (writer.dir / 'observations.jsonl').read_text().splitlines()]
    tracks = [r['detections'][0].get('track') for r in records]
    assert tracks[0] and tracks[0] == tracks[1] and tracks[0].startswith('sift:')
    assert 'appearance' in records[0]['timing_s']
