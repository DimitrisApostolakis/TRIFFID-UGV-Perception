"""Build the pipeline and its outputs from a Config (no ROS)."""
from __future__ import annotations

import logging
import platform

from . import __version__
from .config import Config
from .detectors import create_detector
from .outputs import MapApiClient, SessionWriter
from .pipeline import Pipeline, session_name

log = logging.getLogger(__name__)


def build_pipeline(config: Config, session: str | None = None) -> Pipeline:
    session = session or session_name()
    detector = create_detector(config)
    log.info('Detector ready: %s', detector.metadata().get('name'))
    masa = None
    if config.tracker.backend == 'masa':
        from .tracking.masa import MasaTracker
        floor = config.detector.yoloe.candidate_floor if config.detector.backend == 'yoloe' else config.detector.confidence
        masa = MasaTracker(config.tracker.masa, recovery_floor=floor, device=config.detector.device)
        log.info('MASA tracker ready')
    out = config.outputs
    writer = SessionWriter(config.paths.outputs, session, out.frames.period_s, out.thermal.period_s)
    api = None
    if config.api.enabled:
        api = MapApiClient(config.api, config.state_path('map_api_ids.json'))
        api.start()
        log.info('Map API upload enabled: %s', config.api.url)
    pipeline = Pipeline(config, detector, masa=masa, writer=writer, api=api, session=session)
    writer.write_run(dict(session=session, version=__version__, host=platform.node(), config=config.to_dict(),
                          detector=detector.metadata(), tracker=masa.metadata() if masa else dict(name='geo')))
    return pipeline
