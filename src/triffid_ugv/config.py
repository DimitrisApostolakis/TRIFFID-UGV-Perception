"""Run configuration: typed sections loaded from YAML, overridable from the command line.

Every tunable lives here; modules receive the section they need instead of reading globals.
Override any value with ``--set section.key=value`` (YAML syntax for the value), e.g.
``--set tracker.backend=masa`` or ``--set topics.rgb=/camera/color/image_raw``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

_FRONT = '/b2/camera_front_435i/realsense_front_435i'


@dataclass
class PathsConfig:
    outputs: str = 'outputs'              # GeoJSON, per-frame JSONL, annotated frames
    state: str = 'state'                  # caches and the API id map (survive restarts)
    supports: str = 'supports'            # support bank folder containing bank.json


@dataclass
class TopicsConfig:
    rgb: str = f'{_FRONT}/color/image_raw_broadcasted'
    depth: str = f'{_FRONT}/aligned_depth_to_color/image_raw_broadcasted'
    camera_info: str = ''                 # '' = <rgb namespace>/camera_info; only read when present
    status: str = '/b2/ugv_status'        # ugv_status_interfaces/UnitreeStatus ('' disables)
    fix: str = '/fix'                     # sensor_msgs/NavSatFix ('' disables)
    heading: str = '/b2/nicla/magnetometer/heading_broadcasted'   # std_msgs/Float32, degrees ('' disables)
    thermal: str = '/b2/camera_flir/image_raw_broadcasted'        # subscribed only for debug.thermal
    image_qos: str = 'reliable'           # camera topics; small topics always use best_effort (accepts any publisher)


@dataclass
class PipelineConfig:
    rate_hz: float = 1.0                  # frames processed per second; the newest frame wins
    max_rgbd_dt_s: float = 1.0            # largest RGB/depth stamp gap (the broadcaster bundles them with 1 s slop)
    max_frame_age_s: float = 2.0          # skip frames older than this (wall clock vs receipt)


@dataclass
class CameraConfig:
    intrinsics: str = 'auto'              # auto: CameraInfo when received, else `calibrations` | camera_info | config
    # Pinhole [fx, fy, cx, cy] per image size "WxH" (the ground-station broadcaster sends no CameraInfo).
    # The entry matching the image size is used; another size with the same aspect ratio is rescaled.
    calibrations: dict[str, list[float]] = field(default_factory=lambda: {
        '1280x720': [910.3245849609375, 910.1997680664062, 645.7958984375, 366.5615539550781],  # robot CameraInfo
        '640x480': [606.883056640625, 606.7998657226562, 323.86395263671875, 244.37435913085938]})  # live broadcast
    depth_scale: float = 0.001            # 16UC1 depth units -> metres
    min_depth_m: float = 0.3
    max_depth_m: float = 20.0
    extrinsics: str = 'auto'              # auto: TF when available, else static | tf | static
    body_frame: str = 'b2/base_link'      # x forward, y left, z up
    # Static camera pose in the body frame (camera body axes: x forward, y left, z up). The optical-axis
    # swap is applied automatically. Defaults: base_link -> rs_front_link -> rs_front_color_frame (bag TF).
    static_translation_m: list[float] = field(default_factory=lambda: [0.392, 0.015, 0.279])
    static_rpy_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    max_samples: int = 500                # depth pixels sampled per detection


@dataclass
class YoloeConfig:
    checkpoint: str = '/models/yoloe-v8l-seg.pt'
    image_size: int = 640
    nms_iou: float = 0.5
    candidate_floor: float = 0.1          # lower-scored candidates kept for MASA track recovery
    cache: str = 'yoloe_cache'            # visual-prompt embeddings, relative to paths.state


@dataclass
class DetectorConfig:
    backend: str = 'yoloe'                # yoloe (YOLOE-v8-L, visual prompts) | dummy (wiring tests)
    device: str = 'cuda'
    classes: str | list[str] = 'auto'     # auto = every class with at least one support in the bank
    confidence: float = 0.35              # detections below this are not localized or exported
    min_mask_px: int = 30
    max_per_frame: int = 100
    yoloe: YoloeConfig = field(default_factory=YoloeConfig)


@dataclass
class GeoTrackerConfig:
    gate_m: float = 3.0                   # largest distance between an observation and its object
    large_gate_m: float = 6.0             # gate for large/extended classes (buildings, roads, ...)
    large_classes: list[str] = field(default_factory=lambda: [
        'building', 'destroyed building', 'road', 'dirt road', 'pavement', 'wall', 'fence',
        'green grass', 'dry grass', 'burnt grass', 'water', 'mud', 'debris', 'smoke'])
    position_alpha: float = 0.3           # weight of a new observation in the object's position (EMA)


@dataclass
class SiftConfig:
    max_features: int = 200               # SIFT keypoints kept per detection
    max_side_px: int = 320                # detection crops are downscaled to this longer side first
    min_side_px: int = 24                 # smaller detections get no appearance identity (geo only)
    min_keypoints: int = 12               # textureless detections below this get no appearance identity
    ratio: float = 0.75                   # Lowe ratio test
    min_matches: int = 10                 # verified matches needed to continue a track
    ransac_px: float = 6.0                # homography RANSAC threshold in crop pixels; 0 = ratio test only
    gallery_size: int = 4                 # most recent views kept per track
    memory_s: float = 60.0                # how long a lost track can be recovered by appearance


@dataclass
class MasaConfig:
    repo: str = '/opt/masa'               # provided by the `masa` Docker target
    checkpoint: str = '/models/masa_r50.pth'
    memory_s: int = 60                    # how long a lost track can be recovered by appearance
    init_score: float = 0.35
    match_score: float = 0.5
    geo_merge: bool = True                # also merge a new MASA track into a nearby known object


@dataclass
class TrackerConfig:
    backend: str = 'sift'                 # sift (default, CPU) | masa (fallback, `masa` Docker image)
    min_hits: int = 2                     # sightings before an object is exported
    max_age_s: float = 600.0              # objects unseen this long stop being matched (they stay on the map)
    class_groups: list[list[str]] = field(default_factory=lambda: [
        ['citizen', 'first responder', 'military personnel']])
    hint_gate_m: float = 10.0             # an appearance match is trusted only this close to its object (0 = any)
    geo: GeoTrackerConfig = field(default_factory=GeoTrackerConfig)
    sift: SiftConfig = field(default_factory=SiftConfig)
    masa: MasaConfig = field(default_factory=MasaConfig)


@dataclass
class PositionConfig:
    # Priority order, as on the live system: RTK status topic, the status API (same RTK-gated data), /fix.
    sources: list[str] = field(default_factory=lambda: ['status', 'api', 'fix', 'fixed'])
    status_timeout_s: float = 3.0
    fix_timeout_s: float = 3.0
    fix_median_window: int = 3            # small: a long median lags the position while driving
    api_url: str = 'https://crispres.com/wp-json/mqtt/v1/latest'
    api_poll_s: float = 2.0
    api_timeout_s: float = 3.0
    fixed_lat: float | None = None        # last-resort seed position
    fixed_lon: float | None = None
    fixed_alt: float = 0.0
    heading_sources: list[str] = field(default_factory=lambda: ['magnetometer', 'status', 'fixed'])
    heading_timeout_s: float = 3.0
    status_heading_convention: str = 'compass'   # compass: 0 north, clockwise | enu: 0 east, counter-clockwise
    heading_offset_deg: float = 0.0       # added to every heading: magnetic declination + mounting offset
    fixed_heading_deg: float | None = None


@dataclass
class GeometryConfig:
    line_classes: list[str] = field(default_factory=lambda: ['fence', 'wall'])
    polygon_classes: list[str] = field(default_factory=list)     # e.g. [building, water]; others are Points
    min_extent_m: float = 0.3
    max_line_length_m: float = 30.0


@dataclass
class FramesConfig:
    enabled: bool = True                  # annotated JPEGs of frames with detections
    period_s: float = 1.0
    save_empty: bool = False


@dataclass
class OutputsConfig:
    geojson: bool = True                  # outputs/<session>/objects.geojson, rewritten every flush
    observations: bool = True             # outputs/<session>/observations.jsonl, one line per frame
    flush_period_s: float = 5.0
    frames: FramesConfig = field(default_factory=FramesConfig)


@dataclass
class DebugConfig:
    # Live-test recording: the inputs of every processed frame, side by side, under
    # outputs/<session>/<folder>/{rgb,depth,depth_color,thermal,overlay}/<stamp>.<ext> (same stamp per set).
    enabled: bool = False
    folder: str = 'debug'                 # inside the session folder
    period_s: float = 0.0                 # 0 = every processed frame; else at most one set per period
    save_empty: bool = True               # also frames without detections
    rgb: bool = True                      # colour frame as processed (JPEG)
    depth: bool = True                    # aligned depth, lossless 16-bit PNG in millimetres
    depth_color: bool = True              # depth colourised over [min_depth_m, depth_color_max_m] (JPEG)
    depth_color_max_m: float = 10.0
    thermal: bool = True                  # newest FLIR frame near the RGB stamp (subscribes to topics.thermal)
    thermal_max_dt_s: float = 1.0         # thermal older/newer than this relative to RGB is not paired
    overlay: bool = True                  # detections drawn: masks, boxes, class, score, object, range, pose
    jpeg_quality: int = 90
    queue_size: int = 8                   # pending sets for the writer thread; extra sets are dropped, never block
    max_sets: int = 0                     # stop after this many sets (0 = unlimited)
    min_free_gb: float = 2.0              # stop when the outputs disk has less free space


@dataclass
class ApiConfig:
    enabled: bool = False                 # crispres map-manager upload
    url: str = 'https://crispres.com/wp-json/map-manager/v1/features'
    flush_period_s: float = 5.0
    timeout_s: float = 5.0
    dedup_radius_m: float = 3.0           # reuse a same-class remote feature this close instead of creating one
    update_distance_m: float = 1.0        # PATCH a known feature when it moved this far ...
    update_confidence_delta: float = 0.05 # ... or its confidence rose this much
    token_env: str = 'CRISPRES_API_TOKEN' # optional bearer token read from this environment variable
    source: str = 'ugv'


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    topics: TopicsConfig = field(default_factory=TopicsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    def validate(self) -> 'Config':
        def choice(value, allowed, name):
            if value not in allowed:
                raise ValueError(f'{name} must be one of {sorted(allowed)}, got {value!r}')
        choice(self.detector.backend, {'yoloe', 'dummy'}, 'detector.backend')
        choice(self.tracker.backend, {'sift', 'masa'}, 'tracker.backend')
        choice(self.camera.intrinsics, {'auto', 'camera_info', 'config'}, 'camera.intrinsics')
        choice(self.camera.extrinsics, {'auto', 'tf', 'static'}, 'camera.extrinsics')
        choice(self.topics.image_qos, {'best_effort', 'reliable'}, 'topics.image_qos')
        choice(self.position.status_heading_convention, {'compass', 'enu'}, 'position.status_heading_convention')
        for source in self.position.sources:
            choice(source, {'status', 'fix', 'api', 'fixed'}, 'position.sources[]')
        for source in self.position.heading_sources:
            choice(source, {'magnetometer', 'status', 'fixed'}, 'position.heading_sources[]')
        positive = {'pipeline.rate_hz': self.pipeline.rate_hz, 'pipeline.max_rgbd_dt_s': self.pipeline.max_rgbd_dt_s,
                    'camera.depth_scale': self.camera.depth_scale, 'tracker.geo.gate_m': self.tracker.geo.gate_m,
                    'outputs.flush_period_s': self.outputs.flush_period_s, 'api.flush_period_s': self.api.flush_period_s}
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f'{name} must be positive')
        if not 0 <= self.detector.confidence <= 1 or not 0 <= self.detector.yoloe.candidate_floor <= 1:
            raise ValueError('detector confidences must be in [0, 1]')
        if not 0 < self.tracker.geo.position_alpha <= 1:
            raise ValueError('tracker.geo.position_alpha must be in (0, 1]')
        if not 1 <= self.debug.jpeg_quality <= 100 or self.debug.queue_size < 1 or self.debug.period_s < 0:
            raise ValueError('debug.jpeg_quality must be 1-100, debug.queue_size >= 1, debug.period_s >= 0')
        if not self.debug.depth_color_max_m > self.camera.min_depth_m:
            raise ValueError('debug.depth_color_max_m must be larger than camera.min_depth_m')
        if not self.debug.folder or Path(self.debug.folder).is_absolute() or '..' in Path(self.debug.folder).parts:
            raise ValueError('debug.folder must be a relative folder name inside the session folder')
        if not 0 < self.tracker.sift.ratio < 1:
            raise ValueError('tracker.sift.ratio must be in (0, 1)')
        if self.tracker.sift.min_matches < 4 or self.tracker.sift.gallery_size < 1:
            raise ValueError('tracker.sift.min_matches must be at least 4 and gallery_size at least 1')
        if self.tracker.hint_gate_m < 0:
            raise ValueError('tracker.hint_gate_m must be >= 0')
        if self.tracker.min_hits < 1:
            raise ValueError('tracker.min_hits must be at least 1')
        for size, values in self.camera.calibrations.items():
            w, _, h = str(size).partition('x')
            if not (w.isdigit() and h.isdigit()) or len(values) != 4 or min(values[:2]) <= 0:
                raise ValueError(f'camera.calibrations entries must be "WxH": [fx, fy, cx, cy], got {size}: {values}')
        if len(self.camera.static_translation_m) != 3 or len(self.camera.static_rpy_deg) != 3:
            raise ValueError('camera.static_translation_m and static_rpy_deg need three values')
        if (self.position.fixed_lat is None) != (self.position.fixed_lon is None):
            raise ValueError('Set both position.fixed_lat and position.fixed_lon, or neither')
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def state_path(self, *parts: str) -> Path:
        return Path(self.paths.state, *parts)


def _build(cls, values: dict | None, where: str):
    values = dict(values or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f'Unknown config key(s) in {where or "root"}: {sorted(unknown)}')
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default):
            if not isinstance(value, dict):
                raise ValueError(f'{where}.{name} must be a section'.strip('.'))
            kwargs[name] = _build(type(default), value, f'{where}.{name}'.strip('.'))
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _apply_override(raw: dict, assignment: str) -> None:
    if '=' not in assignment:
        raise ValueError(f'Override must be section.key=value, got {assignment!r}')
    path, value = assignment.split('=', 1)
    keys = path.strip().split('.')
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f'Cannot override inside non-section {path!r}')
    node[keys[-1]] = yaml.safe_load(value) if value.strip() else ''


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                root: str | Path | None = None) -> Config:
    """Defaults <- YAML file <- overrides. Relative paths resolve against `root` (default: CWD)."""
    raw: dict[str, Any] = {}
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f'Config file must contain a mapping: {path}')
        raw = loaded
    for assignment in overrides or ():
        _apply_override(raw, assignment)
    config = _build(Config, raw, '').validate()
    base = Path(root or Path.cwd()).resolve()
    for f in fields(PathsConfig):
        value = getattr(config.paths, f.name)
        setattr(config.paths, f.name, value if Path(value).is_absolute() else str((base / value).resolve()))
    return config
