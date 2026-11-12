"""World map of detected objects: geo-space association, fed by appearance hints (SIFT or MASA).

At one frame per second on a moving robot, image boxes of the same object barely overlap between
frames, so pixel trackers (IoU / Kalman in image space) fragment identities. Here every observation
is first placed in a local ENU plane (metres) and then matched to known objects:

* only class-compatible objects are candidates (same class, or the same `class_groups` entry, e.g.
  citizen / first responder / military personnel are one physical person with a voted role);
* distance is horizontal: point-to-point, point-to-segment for line classes (fence, wall), and
  point-to-polygon for polygon classes, gated by `gate_m` (`large_gate_m` for extended classes);
* one frame's observations are assigned jointly (Hungarian), so two people standing 1 m apart in
  the same frame never collapse into one object;
* an optional per-observation `hint` (an appearance-tracker ID from SIFT or MASA) is honoured first and
  remembered, as long as the observation lies within `hint_gate_m` of the hinted object, so appearance
  tracking and geo association are combined.

Objects are exported once they have `min_hits` sightings; they stop being *matched* after
`max_age_s` without a sighting but stay on the map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import GeometryConfig, TrackerConfig
from ..ontology import BY_NAME

_INFEASIBLE = 1e6


@dataclass
class Observation:
    class_id: int
    class_name: str
    score: float
    bbox: tuple[float, float, float, float]
    enu: np.ndarray                      # robust centre, ENU metres
    points: np.ndarray                   # (N, 3) ENU samples of the object surface
    body: np.ndarray                     # centre in the body frame (x forward, y left, z up)
    stamp_ns: int
    hint: str | None = None              # image-tracker identity, if any
    object_id: str | None = None         # filled in by the map


@dataclass
class WorldObject:
    id: str
    class_votes: dict[str, float]
    enu: np.ndarray
    first_seen_ns: int
    last_seen_ns: int
    hits: int = 1
    best_score: float = 0.0
    last_score: float = 0.0
    height_m: float = 0.0
    line: np.ndarray | None = None       # (2, 2) ENU endpoints for line classes
    hull: np.ndarray | None = None       # (K, 2) ENU convex hull for polygon classes
    revision: int = 0                    # bumps on every update (outputs use it to detect changes)
    hints: set[str] = field(default_factory=set)

    @property
    def class_name(self) -> str:
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def class_id(self) -> int:
        return BY_NAME[self.class_name].id


def _principal_line(xy: np.ndarray, max_length: float) -> np.ndarray:
    centre = xy.mean(0)
    if len(xy) < 2:
        return np.stack([centre, centre])
    _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
    direction = vt[0]
    t = (xy - centre) @ direction
    lo, hi = float(t.min()), float(t.max())
    if hi - lo > max_length:
        mid = (hi + lo) / 2
        lo, hi = mid - max_length / 2, mid + max_length / 2
    return np.stack([centre + lo * direction, centre + hi * direction])


def _point_segment(p: np.ndarray, seg: np.ndarray) -> float:
    a, b = seg
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else min(1.0, max(0.0, float((p - a) @ ab) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


def _point_polygon(p: np.ndarray, hull: np.ndarray) -> float:
    if len(hull) < 3:
        return float(np.min(np.linalg.norm(hull - p, axis=1)))
    signed = cv2.pointPolygonTest(hull.astype(np.float32).reshape(-1, 1, 2), (float(p[0]), float(p[1])), True)
    return max(0.0, -float(signed))


def _hull(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 3:
        return xy
    return cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(float)


class ObjectMap:
    def __init__(self, tracker: TrackerConfig, geometry: GeometryConfig, id_prefix: str):
        self.config, self.geometry = tracker, geometry
        self.objects: dict[str, WorldObject] = {}
        self._hint_to_object: dict[str, str] = {}
        self._ids = itertools.count(1)
        self.id_prefix = id_prefix
        self._group = {}
        for index, group in enumerate(tracker.class_groups):
            for name in group:
                self._group[name] = f'group{index}'
        self.use_geo = tracker.backend != 'masa' or tracker.masa.geo_merge

    # ---- association helpers ------------------------------------------------------------------
    def group(self, class_name: str) -> str:
        return self._group.get(class_name, class_name)

    def kind(self, class_name: str) -> str:
        if class_name in self.geometry.line_classes:
            return 'line'
        if class_name in self.geometry.polygon_classes:
            return 'polygon'
        return 'point'

    def gate(self, class_name: str) -> float:
        g = self.config.geo
        return g.large_gate_m if class_name in g.large_classes else g.gate_m

    def distance(self, obs: Observation, obj: WorldObject) -> float:
        p = obs.enu[:2]
        if obj.line is not None:
            return _point_segment(p, obj.line)
        if obj.hull is not None and len(obj.hull) >= 3:
            return _point_polygon(p, obj.hull)
        return float(np.linalg.norm(p - obj.enu[:2]))

    def active(self, now_ns: int):
        horizon = self.config.max_age_s * 1e9
        return [o for o in self.objects.values() if now_ns - o.last_seen_ns <= horizon]

    # ---- update -------------------------------------------------------------------------------
    def update(self, observations: list[Observation], now_ns: int) -> list[Observation]:
        """Assign every observation to an object (new or existing); returns them with object_id set."""
        pending, claimed = [], set()
        hint_gate = self.config.hint_gate_m
        for obs in observations:
            target = self._hint_to_object.get(obs.hint) if obs.hint else None
            obj = self.objects.get(target) if target else None
            if (obj is not None and obj.id not in claimed and self.group(obj.class_name) == self.group(obs.class_name)
                    and (hint_gate <= 0 or self.distance(obs, obj) <= hint_gate)):
                self._apply(obj, obs)
                claimed.add(obj.id)
            else:
                pending.append(obs)

        if pending and self.use_geo:
            candidates = [o for o in self.active(now_ns) if o.id not in claimed]
            if candidates:
                cost = np.full((len(pending), len(candidates)), _INFEASIBLE)
                for i, obs in enumerate(pending):
                    for j, obj in enumerate(candidates):
                        if self.group(obs.class_name) != self.group(obj.class_name):
                            continue
                        d = self.distance(obs, obj)
                        if d <= self.gate(obj.class_name):
                            cost[i, j] = d
                rows, cols = linear_sum_assignment(cost)
                matched = set()
                for i, j in zip(rows, cols):
                    if cost[i, j] < _INFEASIBLE:
                        self._apply(candidates[j], pending[i])
                        matched.add(i)
                pending = [obs for i, obs in enumerate(pending) if i not in matched]

        for obs in pending:
            self._create(obs)
        return observations

    def _create(self, obs: Observation):
        obj = WorldObject(id=f'{self.id_prefix}-{next(self._ids):05d}', class_votes={obs.class_name: obs.score},
                          enu=obs.enu.copy(), first_seen_ns=obs.stamp_ns, last_seen_ns=obs.stamp_ns,
                          best_score=obs.score, last_score=obs.score)
        self._update_shape(obj, obs, first=True)
        self.objects[obj.id] = obj
        self._link(obj, obs)

    def _apply(self, obj: WorldObject, obs: Observation):
        obj.hits += 1
        alpha = max(self.config.geo.position_alpha, 1.0 / obj.hits)
        obj.enu = (1 - alpha) * obj.enu + alpha * obs.enu
        obj.class_votes[obs.class_name] = obj.class_votes.get(obs.class_name, 0.0) + obs.score
        obj.best_score = max(obj.best_score, obs.score)
        obj.last_score = obs.score
        obj.last_seen_ns = max(obj.last_seen_ns, obs.stamp_ns)
        self._update_shape(obj, obs)
        self._link(obj, obs)

    def _link(self, obj: WorldObject, obs: Observation):
        obj.revision += 1
        obs.object_id = obj.id
        if obs.hint:
            obj.hints.add(obs.hint)
            self._hint_to_object[obs.hint] = obj.id

    def _update_shape(self, obj: WorldObject, obs: Observation, first: bool = False):
        z = obs.points[:, 2]
        height = float(np.percentile(z, 95) - np.percentile(z, 5)) if len(z) > 3 else 0.0
        obj.height_m = height if first else max(obj.height_m * 0.7 + height * 0.3, 0.0)
        kind = self.kind(obs.class_name)
        xy = obs.points[:, :2]
        if kind == 'line':
            merged = xy if obj.line is None else np.vstack([obj.line, xy])
            obj.line = _principal_line(merged, self.geometry.max_line_length_m)
        elif kind == 'polygon':
            merged = xy if obj.hull is None else np.vstack([obj.hull, xy])
            obj.hull = _hull(merged)

    # ---- export -------------------------------------------------------------------------------
    def confirmed(self) -> list[WorldObject]:
        return [o for o in self.objects.values() if o.hits >= self.config.min_hits]

    def stats(self) -> dict:
        return dict(objects=len(self.objects), confirmed=len(self.confirmed()))

