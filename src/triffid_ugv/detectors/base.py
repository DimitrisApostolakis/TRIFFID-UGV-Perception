"""Detector contract. Boxes and masks are in original image pixels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 (max exclusive)
    score: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = None              # bool HxW, same size as the image
    extra: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    detections: list[Detection]                 # score >= detector.confidence: localized and exported
    candidates: list[Detection] | None = None   # also low-scored ones (tracker recovery); None = same list


class Detector(Protocol):
    name: str

    def detect(self, bgr: np.ndarray) -> DetectionResult: ...

    def metadata(self) -> dict: ...


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
