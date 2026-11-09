"""Model-free detector for wiring tests: one box in the image centre."""
import numpy as np

from ..ontology import resolve_class
from .base import Detection, DetectionResult


class DummyDetector:
    name = 'dummy'

    def __init__(self, class_name: str = 'citizen', score: float = 0.9, box_fraction: float = 0.2):
        self.item = resolve_class(class_name)
        self.score, self.fraction = score, box_fraction

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        half_w, half_h = w * self.fraction / 2, h * self.fraction / 2
        box = (w / 2 - half_w, h / 2 - half_h, w / 2 + half_w, h / 2 + half_h)
        mask = np.zeros((h, w), bool)
        mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
        return DetectionResult([Detection(box, self.score, self.item.id, self.item.name, mask)])

    def metadata(self):
        return dict(name=self.name, class_name=self.item.name, score=self.score)
