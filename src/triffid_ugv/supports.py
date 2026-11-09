"""Support bank: example images + masks per class, used as visual prompts.

Manifest (``bank.json``), paths relative to the manifest::

    {"schema_version": "triffid.ugv.support-bank.v1",
     "classes": {"citizen": [{"id": "task_5", "image": "images/a.jpg", "mask": "masks/a.png"}, ...]}}

The UAV schema (``triffid.uav.support-bank.v1``) is read the same way. Empty slots (missing or blank
image/mask) are skipped, so a class may hold any number of examples. Masks are single-channel;
foreground is any non-zero pixel, or ``foreground_value`` for indexed masks.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .ontology import BY_NAME, UgvClass, resolve_classes

SCHEMAS = ('triffid.ugv.support-bank.v1', 'triffid.uav.support-bank.v1')


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Support:
    class_name: str
    example_id: str
    image: Path
    mask: Path
    foreground_value: int | None = None

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 image and boolean foreground mask of the same size."""
        bgr = cv2.imread(str(self.image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.mask), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None:
            raise ValueError(f'{self.example_id}: unreadable image or mask')
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != bgr.shape[:2]:
            raise ValueError(f'{self.example_id}: mask size differs from image size')
        foreground = mask > 0 if self.foreground_value is None else mask == self.foreground_value
        if not foreground.any():
            raise ValueError(f'{self.example_id}: mask has no foreground')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), foreground

    def box(self) -> tuple[int, int, int, int]:
        """Tight xyxy box (max exclusive) around the mask foreground."""
        _, mask = self.load()
        ys, xs = np.nonzero(mask)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def fingerprint(self) -> dict:
        return dict(example_id=self.example_id, image_sha256=sha256_file(self.image),
                    mask_sha256=sha256_file(self.mask), foreground_value=self.foreground_value)


class SupportBank:
    def __init__(self, location: str | Path):
        path = Path(location)
        self.path = (path / 'bank.json' if path.is_dir() else path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Support bank manifest not found: {self.path}')
        data = json.loads(self.path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or data.get('schema_version') not in SCHEMAS:
            raise ValueError(f'Unsupported support bank schema in {self.path}')
        if not isinstance(data.get('classes'), dict):
            raise ValueError('Support bank must contain a "classes" object')
        unknown = sorted(set(data['classes']) - set(BY_NAME))
        if unknown:
            raise ValueError(f'Unknown support bank classes: {unknown}')
        self.data = data
        self.manifest_sha256 = sha256_file(self.path)

    def examples(self, item: UgvClass) -> list[Support]:
        """Usable examples of a class, in manifest order (files must exist)."""
        out = []
        for index, entry in enumerate(self.data['classes'].get(item.name) or [], start=1):
            if not isinstance(entry, dict):
                continue
            image, mask = entry.get('image'), entry.get('mask')
            if not (isinstance(image, str) and image.strip() and isinstance(mask, str) and mask.strip()):
                continue
            image_path, mask_path = (self.path.parent / image).resolve(), (self.path.parent / mask).resolve()
            if not (image_path.is_file() and mask_path.is_file()):
                continue
            value = entry.get('foreground_value')
            out.append(Support(item.name, f"{item.name}/{entry.get('id') or f'example_{index:02d}'}",
                               image_path, mask_path, value if isinstance(value, int) else None))
        return out

    def ready_classes(self, minimum: int = 1) -> tuple[UgvClass, ...]:
        """Classes with at least `minimum` usable examples, in ontology order."""
        return tuple(BY_NAME[name] for name in BY_NAME
                     if name in self.data['classes'] and len(self.examples(BY_NAME[name])) >= minimum)

    def select(self, classes, max_per_class: int | None = None) -> dict[UgvClass, list[Support]]:
        selection = {}
        for item in classes:
            examples = self.examples(item)
            if not examples:
                raise ValueError(f'{item.name}: no usable support examples in {self.path}')
            selection[item] = examples[:max_per_class] if max_per_class else examples
        return selection

    def summary(self) -> dict:
        return {item.name: len(self.examples(item)) for item in self.ready_classes()}


def resolve_detector_classes(requested, bank: SupportBank | None, minimum: int = 1) -> tuple[UgvClass, ...]:
    """`auto` = classes the bank supports; otherwise the requested names/IDs."""
    if requested in (None, '', 'auto'):
        if bank is None:
            raise ValueError("classes: auto needs a support bank; list the classes explicitly instead")
        classes = bank.ready_classes(minimum)
        if not classes:
            raise ValueError(f'The support bank {bank.path} has no class with {minimum}+ usable examples')
        return classes
    return resolve_classes(requested)
