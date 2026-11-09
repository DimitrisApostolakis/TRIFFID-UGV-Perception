"""Detector backends, selected with `detector.backend`."""
from ..config import Config
from .base import Detection, DetectionResult, Detector


def create_detector(config: Config) -> Detector:
    backend = config.detector.backend
    if backend == 'yoloe':
        from .yoloe import YoloeDetector
        return YoloeDetector(config)
    if backend == 'sam3':
        from .sam3 import Sam3Detector
        return Sam3Detector(config)
    if backend == 'dummy':
        from .dummy import DummyDetector
        classes = config.detector.classes
        name = classes[0] if isinstance(classes, list) and classes else 'citizen'
        return DummyDetector(name)
    raise ValueError(f'Unknown detector backend: {backend}')


__all__ = ['Detection', 'DetectionResult', 'Detector', 'create_detector']
