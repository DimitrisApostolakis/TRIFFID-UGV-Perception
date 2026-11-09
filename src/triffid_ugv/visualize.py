"""Annotated debug frames: mask tint, box, class, score, object id and range."""
from __future__ import annotations

import cv2
import numpy as np

from .ontology import class_color


def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip('#')
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def draw(bgr: np.ndarray, items: list[dict], header: str = '') -> np.ndarray:
    """items: dicts with bbox, class_name, score, optional mask, object_id, range_m."""
    canvas = bgr.copy()
    overlay = canvas.copy()
    for item in items:
        if item.get('mask') is not None:
            overlay[item['mask']] = _bgr(class_color(item['class_name']))
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    for item in items:
        color = _bgr(class_color(item['class_name']))
        x1, y1, x2, y2 = (int(round(v)) for v in item['bbox'])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{item['class_name']} {item['score']:.2f}"
        if item.get('object_id'):
            label += f" {item['object_id'].rsplit('-', 1)[-1]}"
        if item.get('range_m') is not None:
            label += f" {item['range_m']:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(th + 4, y1)
        cv2.rectangle(canvas, (x1, y - th - 4), (x1 + tw + 4, y), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if header:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(canvas, header, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
