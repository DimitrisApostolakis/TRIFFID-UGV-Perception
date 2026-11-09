"""YOLOE-v8-L with visual prompts from the support bank (the evaluated UGV method).

For every class, each support example is passed through YOLOE with its tight mask box as a visual
prompt; the returned prompt embeddings are averaged and L2-normalised into one class embedding.
The model is then fixed to those classes and runs as a plain open-vocabulary detector.
Embeddings are cached in ``paths.state``, keyed by checkpoint + support files + image size.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from ..config import Config
from ..supports import SupportBank, resolve_detector_classes, sha256_file
from .base import Detection, DetectionResult, mask_box

log = logging.getLogger(__name__)


class YoloeDetector:
    name = 'yoloe'

    def __init__(self, config: Config):
        import torch
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
        import ultralytics

        det, cfg = config.detector, config.detector.yoloe
        self.torch, self.device = torch, det.device
        self.size, self.confidence = int(cfg.image_size), float(det.confidence)
        # Low-score candidates only help MASA recover tracks; the geo tracker never sees them.
        self.floor = min(float(cfg.candidate_floor), self.confidence) if config.tracker.backend == 'masa' else self.confidence
        self.nms, self.min_mask_px, self.max_per_frame = float(cfg.nms_iou), det.min_mask_px, det.max_per_frame
        checkpoint = Path(cfg.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'YOLOE checkpoint not found: {checkpoint} (see models/README.md)')

        bank = SupportBank(config.paths.supports)
        self.classes = resolve_detector_classes(det.classes, bank)
        selection = bank.select(self.classes)
        identity = dict(schema='triffid.ugv.yoloe-visual-prompts.v1', checkpoint_sha256=sha256_file(checkpoint),
                        image_size=self.size, aggregation='normalized mean of per-example visual prompt embeddings',
                        prompt_box='tight box of the support mask',
                        supports={c.name: [s.fingerprint() for s in selection[c]] for c in self.classes})
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        cache_dir = config.state_path(cfg.cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f'{key}.pt'

        self.model = YOLOE(str(checkpoint))
        self.model.to(self.device)
        memory = None
        if cache.is_file():
            saved = torch.load(cache, map_location='cpu', weights_only=False)
            if saved.get('identity') == identity:
                memory = saved['memory']
        self.cache_hit = memory is not None
        if memory is None:
            vectors = []
            with torch.inference_mode():
                for item in self.classes:
                    examples = []
                    for support in selection[item]:
                        rgb, mask = support.load()
                        ys, xs = np.nonzero(mask)
                        box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)
                        self.model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=self.size, device=self.device,
                                           prompts=dict(bboxes=box, cls=np.array([0])),
                                           predictor=YOLOEVPSegPredictor, return_vpe=True,
                                           verbose=False, save=False, conf=0.99)
                        examples.append(self.model.predictor.vpe[0, 0].detach())
                    vectors.append(torch.nn.functional.normalize(torch.stack(examples).mean(0), dim=0))
                    log.info('YOLOE visual prompt: %s (%d examples)', item.name, len(examples))
            memory = torch.stack(vectors)[None].cpu()
            fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix='.tmp')
            os.close(fd)
            try:
                torch.save(dict(identity=identity, memory=memory), tmp)
                os.replace(tmp, cache)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        if memory.ndim != 3 or memory.shape[:2] != (1, len(self.classes)) or not torch.isfinite(memory).all():
            raise ValueError('Invalid YOLOE visual prompt embeddings; delete the cache and retry')
        self.model.set_classes([c.name for c in self.classes], memory.to(self.device))
        self.model.predictor = None
        self.info = dict(name=self.name, model='YOLOE-v8-L-seg', checkpoint=str(checkpoint),
                         checkpoint_sha256=identity['checkpoint_sha256'], ultralytics_version=ultralytics.__version__,
                         image_size=self.size, confidence=self.confidence, candidate_floor=self.floor,
                         nms_iou=self.nms, prompting='visual (support bank)', cache=str(cache),
                         cache_hit=self.cache_hit, support_bank=str(bank.path),
                         support_bank_sha256=bank.manifest_sha256,
                         classes={c.name: len(selection[c]) for c in self.classes})

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        result = self.model.predict(bgr, imgsz=self.size, device=self.device, conf=self.floor, iou=self.nms,
                                    verbose=False, save=False, retina_masks=True, max_det=self.max_per_frame)[0]
        candidates = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else None
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                item = self.classes[label]
                mask = masks[i] if masks is not None else None
                if mask is not None and mask.shape != bgr.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                if mask is not None and mask.sum() < self.min_mask_px:
                    mask = None
                bbox = tuple(float(v) for v in box)
                if mask is not None:
                    bbox = mask_box(mask) or bbox
                candidates.append(Detection(bbox, float(score), item.id, item.name, mask))
        return DetectionResult([d for d in candidates if d.score >= self.confidence], candidates)

    def metadata(self) -> dict:
        return dict(self.info)
