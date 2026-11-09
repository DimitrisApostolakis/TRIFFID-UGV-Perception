"""Camera model and RGB-D localization, independent of ROS and of the detector.

A detection's mask (or box) is sampled in the aligned depth image, back-projected with the pinhole
intrinsics into the optical frame (x right, y down, z forward) and transformed into the body frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Optical (x right, y down, z forward) -> camera body axes (x forward, y left, z up).
OPTICAL_TO_CAMERA_BODY = np.array([[0.0, 0.0, 1.0],
                                   [-1.0, 0.0, 0.0],
                                   [0.0, -1.0, 0.0]])


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = 'config'

    def scaled_to(self, width: int, height: int) -> 'Intrinsics':
        """The same camera at another resolution (a uniformly resized stream).

        A different aspect ratio means a different sensor mode or crop (e.g. RealSense 640x480 vs
        1280x720), which scaling cannot reproduce: refuse instead of producing wrong geometry.
        """
        if (width, height) == (self.width, self.height):
            return self
        if abs(width / height - self.width / self.height) > 0.01 * self.width / self.height:
            raise ValueError(f'Intrinsics are for {self.width}x{self.height} but the image is {width}x{height} '
                             f'(different aspect ratio): publish CameraInfo or set camera.fx/fy/cx/cy/width/height')
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height,
                          self.source + '+scaled')


def intrinsics_from_camera_info(k, width: int, height: int) -> Intrinsics:
    k = np.asarray(k, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('CameraInfo has invalid intrinsics')
    return Intrinsics(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), int(width), int(height),
                      'camera_info')


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rotation_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not norm or not math.isfinite(norm):
        raise ValueError('Invalid quaternion')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rigid(rotation, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def static_optical_to_body(translation_m, rpy_deg) -> np.ndarray:
    """Body-from-optical transform from a configured camera pose (camera body axes, x forward)."""
    mount = rotation_from_rpy(*(math.radians(v) for v in rpy_deg))
    return rigid(mount @ OPTICAL_TO_CAMERA_BODY, np.asarray(translation_m, dtype=float))


def transform(points, matrix) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def sample_points(detection, depth_m: np.ndarray, intr: Intrinsics, min_depth: float, max_depth: float,
                  max_samples: int = 500) -> np.ndarray | None:
    """Optical-frame 3D points (N, 3) under a detection's mask, or a grid inside its box."""
    h, w = depth_m.shape[:2]
    mask = detection.mask
    if mask is not None and mask.shape == (h, w):
        ys, xs = np.nonzero(mask)
    else:
        x1, y1, x2, y2 = detection.bbox
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, max(ix1 + 1, int(math.ceil(x2)))), min(h, max(iy1 + 1, int(math.ceil(y2))))
        # Central 60% of the box: its border is mostly background.
        mx, my = (ix2 - ix1) * 0.2, (iy2 - iy1) * 0.2
        gx = np.linspace(ix1 + mx, ix2 - 1 - mx, num=max(2, min(25, ix2 - ix1)))
        gy = np.linspace(iy1 + my, iy2 - 1 - my, num=max(2, min(25, iy2 - iy1)))
        xs, ys = (a.ravel().round().astype(int) for a in np.meshgrid(gx, gy))
    if len(xs) == 0:
        return None
    if len(xs) > max_samples:
        index = np.linspace(0, len(xs) - 1, max_samples).astype(int)
        xs, ys = xs[index], ys[index]
    xs, ys = np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)
    z = depth_m[ys, xs].astype(float)
    ok = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if not ok.any():
        return None
    xs, ys, z = xs[ok].astype(float), ys[ok].astype(float), z[ok]
    return np.column_stack([(xs - intr.cx) * z / intr.fx, (ys - intr.cy) * z / intr.fy, z])


def robust_cluster(points: np.ndarray) -> np.ndarray:
    """Drop background/foreground leakage: keep points near the median range (mask edges bleed)."""
    if len(points) < 8:
        return points
    rng = points[:, 2]
    median = np.median(rng)
    mad = np.median(np.abs(rng - median)) * 1.4826
    keep = np.abs(rng - median) <= max(3 * mad, 0.15 + 0.03 * median)
    return points[keep] if keep.sum() >= 3 else points
