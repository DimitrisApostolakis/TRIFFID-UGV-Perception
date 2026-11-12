import numpy as np
import pytest

from triffid_ugv.config import load_config
from triffid_ugv.detectors.base import Detection
from triffid_ugv.tracking.sift import SiftTracker

S = 1_000_000_000


def texture(seed, size=120):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (size // 6, size // 6), dtype=np.uint8)
    import cv2
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)


def scene(patches):
    """Grey 480x640 frame with textured square patches at (x, y)."""
    img = np.full((480, 640), 127, np.uint8)
    for patch, (x, y) in patches:
        img[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
    return np.dstack([img] * 3)


def det(x, y, name='fire truck', size=120):
    return Detection((x, y, x + size, y + size), 0.9, 0, name)


@pytest.fixture
def tracker(tmp_path):
    config = load_config(None, root=tmp_path)
    return SiftTracker(config.tracker.sift, config.tracker.class_groups)


def test_same_texture_keeps_track_when_it_moves(tracker):
    a = texture(1)
    first = dict(tracker.update([det(50, 50)], scene([(a, (50, 50))]), 0))
    second = dict(tracker.update([det(400, 300)], scene([(a, (400, 300))]), S))
    assert first[0] == second[0]


def test_different_textures_get_different_tracks(tracker):
    a, b = texture(1), texture(2)
    out = dict(tracker.update([det(50, 50), det(400, 300)], scene([(a, (50, 50)), (b, (400, 300))]), 0))
    assert out[0] != out[1]
    swapped = dict(tracker.update([det(50, 50), det(400, 300)], scene([(b, (50, 50)), (a, (400, 300))]), S))
    assert swapped[0] == out[1] and swapped[1] == out[0]


def test_class_group_gate_and_textureless(tracker):
    a = texture(1)
    first = dict(tracker.update([det(50, 50)], scene([(a, (50, 50))]), 0))
    other_class = dict(tracker.update([det(50, 50, 'cone')], scene([(a, (50, 50))]), S))
    assert other_class[0] != first[0]
    blank = tracker.update([det(50, 50)], scene([]), 2 * S)      # flat grey: no keypoints, no hint
    assert blank == []


def test_tracks_expire_after_memory(tracker):
    a = texture(1)
    first = dict(tracker.update([det(50, 50)], scene([(a, (50, 50))]), 0))
    later = dict(tracker.update([det(50, 50)], scene([(a, (50, 50))]), 120 * S))
    assert later[0] != first[0]
