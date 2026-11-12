"""Local outputs of one run (a "session"), under ``paths.outputs``::

    outputs/ugv_detections.geojson        confirmed objects of the newest session (stable path)
    outputs/<session>/objects.geojson     the same, per session
    outputs/<session>/observations.jsonl  one JSON line per processed frame: pose + geolocated detections
    outputs/<session>/frames/*.jpg        annotated frames (rate-limited)
    outputs/<session>/debug/...           with debug.enabled: raw inputs + overlays per frame (outputs/debug.py)
    outputs/<session>/run.json            resolved config, model/tracker metadata, counters
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False))
    os.replace(tmp, path)


class SessionWriter:
    def __init__(self, outputs_dir: str | Path, session: str, frames_period_s: float = 1.0):
        self.root = Path(outputs_dir)
        self.dir = self.root / session
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session = session
        self._observations = None
        self._frames_period_ns = int(frames_period_s * 1e9)
        self._last_frame_ns = None

    def write_run(self, info: dict) -> None:
        atomic_write_json(self.dir / 'run.json', info)

    def write_objects(self, collection: dict) -> None:
        atomic_write_json(self.dir / 'objects.geojson', collection)
        atomic_write_json(self.root / 'ugv_detections.geojson', collection)

    def append_observation(self, record: dict) -> None:
        if self._observations is None:
            self._observations = (self.dir / 'observations.jsonl').open('a', encoding='utf-8')
        self._observations.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
        self._observations.flush()

    @staticmethod
    def _stamp_name(stamp_ns: int) -> str:
        return f'{stamp_ns // 1_000_000_000}_{(stamp_ns // 1_000_000) % 1000:03d}.jpg'

    def _due(self, last, stamp_ns, period_ns):
        return last is None or stamp_ns - last >= period_ns or stamp_ns < last

    def save_frame(self, image: np.ndarray, stamp_ns: int) -> str | None:
        if not self._due(self._last_frame_ns, stamp_ns, self._frames_period_ns):
            return None
        self._last_frame_ns = stamp_ns
        path = self.dir / 'frames' / self._stamp_name(stamp_ns)
        path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(path.relative_to(self.dir))

    def close(self) -> None:
        if self._observations is not None:
            self._observations.close()
            self._observations = None
