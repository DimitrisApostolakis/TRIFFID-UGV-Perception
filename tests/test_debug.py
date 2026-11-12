import json

import cv2
import numpy as np
import pytest

from triffid_ugv.config import load_config
from triffid_ugv.detectors.dummy import DummyDetector
from triffid_ugv.outputs import DebugRecorder, SessionWriter
from triffid_ugv.outputs.debug import STREAMS, depth_to_millimetres
from triffid_ugv.pipeline import Pipeline
from test_pipeline import frame

S = 1_000_000_000


def make_config(tmp_path, *overrides):
    return load_config(None, ['detector.backend=dummy', 'tracker.min_hits=1', 'debug.enabled=true', *overrides],
                       root=tmp_path)


def test_depth_is_saved_losslessly_in_millimetres(tmp_path):
    depth = np.array([[0.0, 1.2345, np.nan], [70.0, 5.0, 0.3]], np.float32)
    mm = depth_to_millimetres(depth)
    assert mm.dtype == np.uint16 and mm.tolist() == [[0, 1234, 0], [65535, 5000, 300]]


def test_pipeline_writes_one_aligned_set_per_frame(tmp_path):
    config = make_config(tmp_path)
    writer = SessionWriter(config.paths.outputs, 'dbg')
    recorder = DebugRecorder(config.debug, writer.dir, config.camera.min_depth_m)
    pipeline = Pipeline(config, DummyDetector('citizen'), writer=writer, session='dbg', debug=recorder)
    for t in range(3):
        f = frame(t)
        f.thermal = np.full((240, 320, 3), 200, np.uint8)
        pipeline.process(f)
    pipeline.close()
    root = writer.dir / 'debug'
    names = {s: sorted(p.stem for p in (root / s).iterdir()) for s in STREAMS}
    assert all(len(v) == 3 for v in names.values()) and len({tuple(v) for v in names.values()}) == 1
    depth = cv2.imread(str(next((root / 'depth').iterdir())), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and depth.shape == (480, 640) and int(depth[240, 320]) == 5000
    record = json.loads((writer.dir / 'observations.jsonl').read_text().splitlines()[0])
    assert set(record['debug_files']) == set(STREAMS)
    assert (writer.dir / record['debug_files']['overlay']).is_file()
    assert recorder.status()['sets'] == 3


def test_streams_period_and_limits_are_configurable(tmp_path):
    config = make_config(tmp_path, 'debug.thermal=false', 'debug.depth_color=false', 'debug.period_s=2',
                         'debug.max_sets=2')
    writer = SessionWriter(config.paths.outputs, 'cfg')
    recorder = DebugRecorder(config.debug, writer.dir)
    pipeline = Pipeline(config, DummyDetector('citizen'), writer=writer, session='cfg', debug=recorder)
    for t in range(8):                       # due at t = 0, 2, 4, 6; max_sets stops after two
        pipeline.process(frame(t))
    pipeline.close()
    root = writer.dir / 'debug'
    assert sorted(p.name for p in root.iterdir()) == ['depth', 'overlay', 'rgb']
    assert len(list((root / 'rgb').iterdir())) == 2 and 'max_sets' in recorder.status()['stopped']


def test_low_disk_stops_recording(tmp_path):
    config = make_config(tmp_path, 'debug.min_free_gb=1000000000000')
    recorder = DebugRecorder(config.debug, tmp_path / 'session')
    recorder.submit(0, np.zeros((10, 10, 3), np.uint8))
    recorder.close()
    assert recorder.status()['sets'] == 0 and 'free' in recorder.status()['stopped']


def test_disabled_by_default_and_folder_validated(tmp_path):
    assert load_config(None, root=tmp_path).debug.enabled is False
    with pytest.raises(ValueError):
        load_config(None, ['debug.folder=../elsewhere'], root=tmp_path)
