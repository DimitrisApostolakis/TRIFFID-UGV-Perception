import pytest

from triffid_ugv.config import load_config


def test_default_yaml_matches_dataclass_defaults(tmp_path):
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    from_file = load_config(repo / 'configs/default.yaml', root=tmp_path).to_dict()
    defaults = load_config(None, root=tmp_path).to_dict()
    assert from_file == defaults


def test_overrides_and_paths(tmp_path):
    config = load_config(None, ['tracker.backend=masa', 'tracker.geo.gate_m=5', 'topics.camera_info=',
                                'detector.classes=[citizen, fence]'], root=tmp_path)
    assert config.tracker.backend == 'masa' and config.tracker.geo.gate_m == 5
    assert config.detector.classes == ['citizen', 'fence'] and config.topics.camera_info == ''
    assert config.paths.outputs == str(tmp_path / 'outputs')


def test_defaults_are_yoloe_and_sift(tmp_path):
    config = load_config(None, root=tmp_path)
    assert config.detector.backend == 'yoloe' and config.tracker.backend == 'sift'


@pytest.mark.parametrize('override', ['detector.backend=yolo', 'detector.backend=sam3', 'tracker.backend=geo',
                                      'tracker.backend=bytetrack', 'tracker.sift.ratio=1.5', 'nope.key=1',
                                      'position.fixed_lat=38.0', 'pipeline.rate_hz=0',
                                      'camera.calibrations={1280x720: [1, 2, 3]}'])
def test_invalid_values_rejected(tmp_path, override):
    with pytest.raises(ValueError):
        load_config(None, [override], root=tmp_path)
