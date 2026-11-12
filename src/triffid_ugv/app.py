"""Build the pipeline and its outputs from a Config (no ROS)."""
from __future__ import annotations

import logging
import platform

from . import __version__
from .config import Config
from .detectors import create_detector
from .outputs import DebugRecorder, MapApiClient, SessionWriter
from .pipeline import Pipeline, session_name

log = logging.getLogger(__name__)


def build_pipeline(config: Config, session: str | None = None) -> Pipeline:
    session = session or session_name()
    detector = create_detector(config)
    log.info('Detector ready: %s', detector.metadata().get('name'))
    tracker = config.tracker
    if tracker.backend == 'masa':
        from .tracking.masa import MasaTracker
        floor = config.detector.yoloe.candidate_floor if config.detector.backend == 'yoloe' else config.detector.confidence
        appearance = MasaTracker(tracker.masa, tracker.class_groups, recovery_floor=floor, device=config.detector.device)
    else:
        from .tracking.sift import SiftTracker
        appearance = SiftTracker(tracker.sift, tracker.class_groups)
    log.info('Tracker ready: %s appearance hints + geo association', appearance.name)
    out = config.outputs
    writer = SessionWriter(config.paths.outputs, session, out.frames.period_s)
    debug = DebugRecorder(config.debug, writer.dir, config.camera.min_depth_m) if config.debug.enabled else None
    api = None
    if config.api.enabled:
        api = MapApiClient(config.api, config.state_path('map_api_ids.json'))
        api.start()
        log.info('Map API upload enabled: %s', config.api.url)
    pipeline = Pipeline(config, detector, appearance=appearance, writer=writer, api=api, session=session, debug=debug)
    writer.write_run(dict(session=session, version=__version__, host=platform.node(), config=config.to_dict(),
                          detector=detector.metadata(), tracker=appearance.metadata()))
    return pipeline
