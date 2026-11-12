"""SIFT appearance tracker (the default, `tracker.backend: sift`). CPU only, no extra weights.

Each detection's mask (or box, when there is no mask) is described with SIFT keypoints. A detection
continues a track of the same class group when enough of its keypoints match one of the track's recent
views: Lowe ratio test, one-to-one keypoints, then a RANSAC homography as a geometric check. One frame's
detections are assigned to tracks jointly (Hungarian, most verified matches first).

As with MASA, the track IDs are only *hints* for the world map (`tracking/world.py`): a hinted object
keeps its identity when its geo position is noisy, but only within `tracker.hint_gate_m`, and every
detection without a usable hint (too small, textureless, new) is still associated by distance on the map.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import itertools

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import SiftConfig


@dataclass
class View:
    points: np.ndarray                   # (N, 2) keypoint positions in crop pixels
    descriptors: np.ndarray              # (N, 128) float32


@dataclass
class _Track:
    id: int
    group: str
    last_ns: int
    views: deque = field(default_factory=deque)
    hits: int = 0


class SiftTracker:
    name = 'sift'
    uses_candidates = False              # only detections above detector.confidence are described

    def __init__(self, config: SiftConfig, class_groups=()):
        self.config = config
        self.sift = cv2.SIFT_create(nfeatures=int(config.max_features))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._group = {name: f'group{i}' for i, group in enumerate(class_groups) for name in group}
        self.tracks: dict[int, _Track] = {}
        self._ids = itertools.count(1)
        self.last_ns = None

    def metadata(self) -> dict:
        return dict(name='sift', opencv_version=cv2.__version__, **asdict(self.config))

    def group(self, class_name: str) -> str:
        return self._group.get(class_name, class_name)

    # ---- features -----------------------------------------------------------------------------
    def describe(self, gray: np.ndarray, det) -> View | None:
        """SIFT keypoints inside the detection; None when it is too small or has too little texture."""
        cfg = self.config
        h, w = gray.shape[:2]
        x1, y1 = max(0, int(np.floor(det.bbox[0]))), max(0, int(np.floor(det.bbox[1])))
        x2, y2 = min(w, int(np.ceil(det.bbox[2]))), min(h, int(np.ceil(det.bbox[3])))
        if min(x2 - x1, y2 - y1) < cfg.min_side_px:
            return None
        crop = gray[y1:y2, x1:x2]
        mask = None
        if det.mask is not None:
            mask = det.mask[y1:y2, x1:x2].astype(np.uint8) * 255
        scale = cfg.max_side_px / max(crop.shape)
        if scale < 1:
            size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        if mask is not None and not mask.any():
            mask = None
        keypoints, descriptors = self.sift.detectAndCompute(np.ascontiguousarray(crop), mask)
        if descriptors is None or len(keypoints) < cfg.min_keypoints:
            return None
        return View(np.float32([k.pt for k in keypoints]), descriptors)

    def matches(self, view: View, reference: View) -> int:
        """Verified keypoint matches between two views (0 when below `min_matches`)."""
        cfg = self.config
        if len(reference.descriptors) < 2:
            return 0
        best: dict[int, cv2.DMatch] = {}
        for pair in self.matcher.knnMatch(view.descriptors, reference.descriptors, k=2):
            if len(pair) == 2 and pair[0].distance < cfg.ratio * pair[1].distance:
                m = pair[0]
                if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                    best[m.trainIdx] = m
        good = list(best.values())
        if len(good) < cfg.min_matches:
            return 0
        count = len(good)
        if cfg.ransac_px > 0:
            src = view.points[[m.queryIdx for m in good]]
            dst = reference.points[[m.trainIdx for m in good]]
            _, inliers = cv2.findHomography(src, dst, cv2.RANSAC, float(cfg.ransac_px))
            count = 0 if inliers is None else int(inliers.sum())
        return count if count >= cfg.min_matches else 0

    def score(self, view: View, track: _Track) -> int:
        return max((self.matches(view, ref) for ref in track.views), default=0)

    # ---- tracking -----------------------------------------------------------------------------
    def update(self, detections, bgr: np.ndarray, stamp_ns: int) -> list[tuple[int, str]]:
        """Returns (index into `detections`, track id) for every detection with an appearance identity."""
        if self.last_ns is not None and stamp_ns <= self.last_ns:
            return []
        self.last_ns = stamp_ns
        horizon = self.config.memory_s * 1e9
        self.tracks = {k: t for k, t in self.tracks.items() if stamp_ns - t.last_ns <= horizon}
        if not detections:
            return []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        views = [self.describe(gray, d) for d in detections]
        rows = [i for i, v in enumerate(views) if v is not None]
        tracks = list(self.tracks.values())
        assigned: dict[int, _Track] = {}
        if rows and tracks:
            scores = np.zeros((len(rows), len(tracks)))
            for a, i in enumerate(rows):
                group = self.group(detections[i].class_name)
                for b, track in enumerate(tracks):
                    if track.group == group:
                        scores[a, b] = self.score(views[i], track)
            for a, b in zip(*linear_sum_assignment(-scores)):
                if scores[a, b] > 0:
                    assigned[rows[a]] = tracks[b]
        out = []
        for i in rows:
            track = assigned.get(i)
            if track is None:
                track = _Track(next(self._ids), self.group(detections[i].class_name), stamp_ns,
                               deque(maxlen=int(self.config.gallery_size)))
                self.tracks[track.id] = track
            track.views.append(views[i])
            track.last_ns, track.hits = stamp_ns, track.hits + 1
            out.append((i, f'sift:{track.id}'))
        return out
