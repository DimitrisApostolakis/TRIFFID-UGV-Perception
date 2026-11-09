"""sensor_msgs/Image -> numpy without cv_bridge (works with any message-like object)."""
from __future__ import annotations

import struct

import cv2
import numpy as np

_COLOR = {'rgb8': (3, cv2.COLOR_RGB2BGR), 'bgr8': (3, None), 'rgba8': (4, cv2.COLOR_RGBA2BGR),
          'bgra8': (4, cv2.COLOR_BGRA2BGR), 'mono8': (1, cv2.COLOR_GRAY2BGR), '8uc1': (1, cv2.COLOR_GRAY2BGR),
          '8uc3': (3, None)}
_YUV = {'yuv422': cv2.COLOR_YUV2BGR_UYVY, 'uyvy': cv2.COLOR_YUV2BGR_UYVY,
        'yuv422_yuy2': cv2.COLOR_YUV2BGR_YUY2, 'yuyv': cv2.COLOR_YUV2BGR_YUY2}


def _rows(msg, dtype, channels):
    itemsize = np.dtype(dtype).itemsize
    buffer = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray, memoryview)) else msg.data,
                           dtype=np.uint8)
    step = msg.step or msg.width * channels * itemsize
    rows = buffer[:step * msg.height].reshape(msg.height, step)[:, :msg.width * channels * itemsize]
    wire = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<') if itemsize > 1 else np.dtype(dtype)
    array = np.ascontiguousarray(rows).view(wire).astype(dtype)
    return array.reshape(msg.height, msg.width, channels) if channels > 1 else array.reshape(msg.height, msg.width)


def image_to_bgr(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in _YUV:
        return cv2.cvtColor(_rows(msg, np.uint8, 2), _YUV[encoding])
    if encoding in ('mono16', '16uc1'):     # thermal cameras: stretch to 8 bits for viewing
        raw = _rows(msg, np.uint16, 1).astype(np.float32)
        lo, hi = np.percentile(raw, (1, 99)) if raw.size else (0, 1)
        scaled = np.clip((raw - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    if encoding not in _COLOR:
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')
    channels, conversion = _COLOR[encoding]
    image = _rows(msg, np.uint8, channels)
    return image.copy() if conversion is None else cv2.cvtColor(image, conversion)


def depth_to_metres(msg, depth_scale: float = 0.001) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in ('16uc1', 'mono16'):
        raw = _rows(msg, np.uint16, 1)
        depth = raw.astype(np.float32) * depth_scale
        depth[(raw == 0) | (raw == 65535)] = 0.0   # 0 = no return; 65535 = saturated (not a range)
        return depth
    if encoding == '32fc1':
        depth = _rows(msg, np.float32, 1).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return depth
    raise ValueError(f'Unsupported depth encoding: {msg.encoding} (expected 16UC1 or 32FC1)')


def cdr_stamp_ns(raw: bytes) -> int:
    """Header stamp of a serialized message that starts with std_msgs/Header (e.g. Image).

    CDR: 4-byte encapsulation header (byte 1: 1 = little endian), then int32 sec, uint32 nanosec.
    """
    order = '<' if raw[1] & 1 else '>'
    sec, nanosec = struct.unpack_from(order + 'iI', raw, 4)
    return sec * 1_000_000_000 + nanosec


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
