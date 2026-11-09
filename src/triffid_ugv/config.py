"""Run configuration: typed sections loaded from YAML, overridable from the command line.

Every tunable lives here; modules receive the section they need instead of reading globals.
Override any value with ``--set section.key=value`` (YAML syntax for the value), e.g.
``--set detector.backend=sam3`` or ``--set topics.rgb=/camera/color/image_raw``.
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
    thermal: str = '/b2/camera_flir/image_raw_broadcasted'        # used only when outputs.thermal.enabled
    image_qos: str = 'reliable'           # camera topics; small topics always use best_effort (accepts any publisher)


@dataclass
class PipelineConfig:
    rate_hz: float = 1.0                  # frames processed per second; the newest frame wins
    max_rgbd_dt_s: float = 0.25           # largest RGB/depth stamp difference accepted
    max_frame_age_s: float = 2.0          # skip frames older than this (wall clock vs receipt)


@dataclass
class CameraConfig:
    intrinsics: str = 'auto'              # auto: CameraInfo when received, else the values below | camera_info | config
    fx: float = 606.883056640625          # RealSense D435i colour, 640x480 (live branch values)
    fy: float = 606.7998657226562
    cx: float = 323.86395263671875
    cy: float = 244.37435913085938
    width: int = 640                      # rescaled for another size with the same aspect ratio only
    height: int = 480
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
class Sam3Config:
    repo: str = '/opt/fss-sam3'           # FSS-SAM3 source checkout (provided by the Docker image)
    checkpoint: str = '/models/sam3.pt'   # same files as the UAV perception image
    bpe: str = '/models/bpe_simple_vocab_16e6.txt.gz'
    prompt: str = 'text'                  # text: one image encoding, class names as prompts (fast)
                                          # visual | text+visual: few-shot support collage per class (slow)
    shots: int = 1                        # visual prompts: 1 or 5 supports per class
    precision: str = 'bf16'               # bf16 autocast (~2x faster, as in the SAM3 examples) | fp32
    resolution: int = 1008
    support_ratio_1shot: float = 0.6
    support_ratio_5shot: float = 0.3


@dataclass
class DetectorConfig:
    backend: str = 'yoloe'                # yoloe | sam3 | dummy
    device: str = 'cuda'
    classes: str | list[str] = 'auto'     # auto = every class with at least one support (yoloe / sam3 visual)
    confidence: float = 0.35              # detections below this are not localized or exported
    min_mask_px: int = 30
    max_per_frame: int = 100
    yoloe: YoloeConfig = field(default_factory=YoloeConfig)
    sam3: Sam3Config = field(default_factory=Sam3Config)


@dataclass
class GeoTrackerConfig:
    gate_m: float = 3.0                   # largest distance between an observation and its object
    large_gate_m: float = 6.0             # gate for large/extended classes (buildings, roads, ...)
    large_classes: list[str] = field(default_factory=lambda: [
        'building', 'destroyed building', 'road', 'dirt road', 'pavement', 'wall', 'fence',
        'green grass', 'dry grass', 'burnt grass', 'water', 'mud', 'debris', 'smoke'])
    position_alpha: float = 0.3           # weight of a new observation in the object's position (EMA)


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
    backend: str = 'geo'                  # geo | masa
    min_hits: int = 2                     # sightings before an object is exported
    max_age_s: float = 600.0              # objects unseen this long stop being matched (they stay on the map)
    class_groups: list[list[str]] = field(default_factory=lambda: [
        ['citizen', 'first responder', 'military personnel']])
    geo: GeoTrackerConfig = field(default_factory=GeoTrackerConfig)
    masa: MasaConfig = field(default_factory=MasaConfig)


@dataclass
class PositionConfig:
    sources: list[str] = field(default_factory=lambda: ['status', 'fix', 'api', 'fixed'])  # priority order
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
class ThermalConfig:
    enabled: bool = False                 # optional FLIR debug snapshots; never used for detection
    period_s: float = 1.0


@dataclass
class OutputsConfig:
    geojson: bool = True                  # outputs/<session>/objects.geojson, rewritten every flush
    observations: bool = True             # outputs/<session>/observations.jsonl, one line per frame
    flush_period_s: float = 5.0
    frames: FramesConfig = field(default_factory=FramesConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)


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
    api: ApiConfig = field(default_factory=ApiConfig)

    def validate(self) -> 'Config':
        def choice(value, allowed, name):
            if value not in allowed:
                raise ValueError(f'{name} must be one of {sorted(allowed)}, got {value!r}')
        choice(self.detector.backend, {'yoloe', 'sam3', 'dummy'}, 'detector.backend')
        choice(self.detector.sam3.prompt, {'text', 'visual', 'text+visual'}, 'detector.sam3.prompt')
        choice(self.detector.sam3.shots, {1, 5}, 'detector.sam3.shots')
        choice(self.detector.sam3.precision, {'bf16', 'fp32'}, 'detector.sam3.precision')
        choice(self.tracker.backend, {'geo', 'masa'}, 'tracker.backend')
        choice(self.camera.intrinsics, {'auto', 'camera_info', 'config'}, 'camera.intrinsics')
        choice(self.camera.extrinsics, {'auto', 'tf', 'static'}, 'camera.extrinsics')
        choice(self.topics.image_qos, {'best_effort', 'reliable'}, 'topics.image_qos')
        choice(self.position.status_heading_convention, {'compass', 'enu'}, 'position.status_heading_convention')
        for source in self.position.sources:
            choice(source, {'status', 'fix', 'api', 'fixed'}, 'position.sources[]')
        for source in self.position.heading_sources:
            choice(source, {'magnetometer', 'status', 'fixed'}, 'position.heading_sources[]')
        positive = {'pipeline.rate_hz': self.pipeline.rate_hz, 'pipeline.max_rgbd_dt_s': self.pipeline.max_rgbd_dt_s,
                    'camera.fx': self.camera.fx, 'camera.fy': self.camera.fy,
                    'camera.depth_scale': self.camera.depth_scale, 'tracker.geo.gate_m': self.tracker.geo.gate_m,
                    'outputs.flush_period_s': self.outputs.flush_period_s, 'api.flush_period_s': self.api.flush_period_s}
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f'{name} must be positive')
        if not 0 <= self.detector.confidence <= 1 or not 0 <= self.detector.yoloe.candidate_floor <= 1:
            raise ValueError('detector confidences must be in [0, 1]')
        if not 0 < self.tracker.geo.position_alpha <= 1:
            raise ValueError('tracker.geo.position_alpha must be in (0, 1]')
        if self.tracker.min_hits < 1:
            raise ValueError('tracker.min_hits must be at least 1')
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
