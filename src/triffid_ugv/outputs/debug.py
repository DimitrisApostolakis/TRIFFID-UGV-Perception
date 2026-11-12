"""Debug recording for live tests (`debug.enabled`): the inputs of every processed frame, side by side.

One *set* per processed frame (at most one per `debug.period_s`), every image named by the RGB stamp so
the folders line up::

    outputs/<session>/debug/rgb/<stamp>.jpg          colour frame as processed
    outputs/<session>/debug/depth/<stamp>.png        aligned depth, uint16 millimetres (lossless)
    outputs/<session>/debug/depth_color/<stamp>.jpg  depth colourised for viewing
    outputs/<session>/debug/thermal/<stamp>.jpg      FLIR frame closest to the RGB stamp (when paired)
    outputs/<session>/debug/overlay/<stamp>.jpg      detections drawn on the frame

Encoding and writing run on one background thread behind a bounded queue: a slow disk drops sets
(counted in `counters`), it never delays inference. Recording stops by itself after `max_sets` or when
the disk has less than `min_free_gb` free.
"""
from __future__ import annotations

import logging
from pathlib import Path
import queue
import shutil
import threading

import cv2
import numpy as np

from ..config import DebugConfig
from ..visualize import colorize_depth

log = logging.getLogger(__name__)

STREAMS = ('rgb', 'depth', 'depth_color', 'thermal', 'overlay')
_DISK_CHECK_EVERY = 20                   # sets between free-space checks


def stamp_name(stamp_ns: int) -> str:
    """`<seconds>_<milliseconds>`: the same naming as the annotated frames."""
    return f'{stamp_ns // 1_000_000_000}_{(stamp_ns // 1_000_000) % 1000:03d}'


def depth_to_millimetres(depth_m: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)


class DebugRecorder:
    def __init__(self, config: DebugConfig, session_dir: str | Path, min_depth_m: float = 0.3):
        self.config, self.min_depth_m = config, float(min_depth_m)
        self.dir = Path(session_dir) / config.folder
        self.streams = [s for s in STREAMS if getattr(config, s)]
        for stream in self.streams:
            (self.dir / stream).mkdir(parents=True, exist_ok=True)
        self.counters = dict(sets=0, dropped=0, failed=0)   # sets: written by the thread
        self._submitted = 0                                  # accepted into the queue (caller side)
        self.stopped_reason = ''
        self._period_ns = int(config.period_s * 1e9)
        self._last_ns = None
        self._queue: queue.Queue = queue.Queue(maxsize=int(config.queue_size))
        self._thread = threading.Thread(target=self._run, name='debug-recorder', daemon=True)
        self._thread.start()
        log.info('Debug recording to %s: %s', self.dir, ', '.join(self.streams))

    @property
    def wants_thermal(self) -> bool:
        return self.config.thermal

    @property
    def wants_overlay(self) -> bool:
        return self.config.overlay

    def due(self, stamp_ns: int, has_detections: bool) -> bool:
        """Whether a set for this frame would be recorded (callers skip drawing the overlay otherwise)."""
        if self.stopped_reason or not (has_detections or self.config.save_empty):
            return False
        return self._last_ns is None or stamp_ns < self._last_ns or stamp_ns - self._last_ns >= self._period_ns

    def submit(self, stamp_ns: int, bgr: np.ndarray, depth_m: np.ndarray | None = None,
               overlay: np.ndarray | None = None, thermal: np.ndarray | None = None) -> dict[str, str]:
        """Queue one set. Returns the files it will write (relative to the session folder); {} if dropped.

        The arrays must not be modified afterwards (the pipeline never does).
        """
        if self.stopped_reason:
            return {}
        cfg = self.config
        if cfg.max_sets and self._submitted >= cfg.max_sets:
            self._stop(f'max_sets={cfg.max_sets} reached')
            return {}
        name = stamp_name(stamp_ns)
        images = {}
        if cfg.rgb:
            images['rgb'] = (bgr, '.jpg')
        if cfg.depth and depth_m is not None:
            images['depth'] = (depth_m, '.png')
        if cfg.depth_color and depth_m is not None:
            images['depth_color'] = (depth_m, '.jpg')
        if cfg.thermal and thermal is not None:
            images['thermal'] = (thermal, '.jpg')
        if cfg.overlay and overlay is not None:
            images['overlay'] = (overlay, '.jpg')
        try:
            self._queue.put_nowait((name, images))
        except queue.Full:
            self.counters['dropped'] += 1
            return {}
        self._last_ns = stamp_ns
        self._submitted += 1
        relative = self.dir.name
        return {stream: f'{relative}/{stream}/{name}{ext}' for stream, (_, ext) in images.items()}

    # ---- writer thread ------------------------------------------------------------------------
    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write(*item)
            except Exception as error:      # a bad frame or full disk must never stop the thread or the node
                self.counters['failed'] += 1
                log.warning('Debug set %s not written: %s', item[0], error)
            finally:
                self._queue.task_done()

    def _write(self, name: str, images: dict):
        if self.stopped_reason:
            return
        if self.counters['sets'] % _DISK_CHECK_EVERY == 0 and self.config.min_free_gb > 0:
            free_gb = shutil.disk_usage(self.dir).free / 1e9
            if free_gb < self.config.min_free_gb:
                self._stop(f'only {free_gb:.1f} GB free (< debug.min_free_gb={self.config.min_free_gb:g})')
                return
        quality = [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)]
        for stream, (image, ext) in images.items():
            if stream == 'depth':
                image = depth_to_millimetres(image)
            elif stream == 'depth_color':
                image = colorize_depth(image, self.min_depth_m, self.config.depth_color_max_m)
            path = self.dir / stream / f'{name}{ext}'
            if not cv2.imwrite(str(path), image, quality if ext == '.jpg' else []):
                raise OSError(f'cv2.imwrite failed for {path}')
        self.counters['sets'] += 1

    def _stop(self, reason: str):
        if not self.stopped_reason:
            self.stopped_reason = reason
            log.warning('Debug recording stopped: %s', reason)

    def status(self) -> dict:
        return dict(self.counters, queued=self._queue.qsize(), stopped=self.stopped_reason or None)

    def close(self, timeout_s: float = 10.0):
        """Write what is queued, then stop the thread."""
        if self._thread.is_alive():
            try:
                self._queue.put(None, timeout=timeout_s)
            except queue.Full:
                log.warning('Debug writer did not drain its queue; %d sets lost', self._queue.qsize())
                return
            self._thread.join(timeout_s)
