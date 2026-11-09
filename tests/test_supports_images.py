from types import SimpleNamespace

import numpy as np
import pytest

from triffid_ugv.ontology import BY_NAME
from triffid_ugv.ros.images import depth_to_metres, image_to_bgr
from triffid_ugv.supports import SupportBank, resolve_detector_classes


def test_bank_skips_empty_slots(bank):
    b = SupportBank(bank)
    assert b.summary() == {'fence': 1, 'citizen': 2}
    assert [c.name for c in resolve_detector_classes('auto', b)] == ['fence', 'citizen']
    assert [c.name for c in resolve_detector_classes(['citizen'], b)] == ['citizen']
    assert [c.name for c in resolve_detector_classes('first_responder,23', None)] == ['first responder', 'citizen']
    rgb, mask = b.examples(BY_NAME['citizen'])[0].load()
    assert rgb.shape == (40, 60, 3) and mask.sum() == 400


def msg(encoding, array, step=None):
    data = array.tobytes()
    h, w = array.shape[:2]
    return SimpleNamespace(encoding=encoding, height=h, width=w, step=step or len(data) // h, data=data,
                           is_bigendian=False)


def test_color_and_depth_decoding():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[..., 0] = 255
    assert image_to_bgr(msg('rgb8', rgb))[0, 0].tolist() == [0, 0, 255]
    raw = np.array([[1000, 0], [65535, 2500]], np.uint16)
    depth = depth_to_metres(msg('16UC1', raw))
    assert depth.tolist() == [[1.0, 0.0], [0.0, 2.5]]


def test_row_padding_is_ignored():
    padded = np.zeros((2, 8), np.uint8)       # 2x2 bgr8 image, rows padded to 8 bytes
    padded[:, :6] = 7
    image = image_to_bgr(SimpleNamespace(encoding='bgr8', height=2, width=2, step=8, data=padded.tobytes(),
                                         is_bigendian=False))
    assert image.shape == (2, 2, 3) and (image == 7).all()


def test_yuyv_supported():
    assert image_to_bgr(msg('yuv422_yuy2', np.full((4, 4, 2), 128, np.uint8))).shape == (4, 4, 3)
    with pytest.raises(ValueError):
        image_to_bgr(msg('bayer_rggb8', np.zeros((2, 2), np.uint8)))


def test_cdr_stamp_parsing():
    import struct
    from triffid_ugv.ros.images import cdr_stamp_ns
    little = bytes([0, 1, 0, 0]) + struct.pack('<iI', 1792505602, 601764675) + b'frame'
    big = bytes([0, 0, 0, 0]) + struct.pack('>iI', 5, 7)
    assert cdr_stamp_ns(little) == 1792505602_601764675 and cdr_stamp_ns(big) == 5_000_000_007
