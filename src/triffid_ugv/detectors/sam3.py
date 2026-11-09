"""SAM3 detector (optional backend), using the FSS-SAM3 source checkout and the UAV's sam3.pt.

Prompt modes (``detector.sam3.prompt``)
  text         the frame is encoded once; every class name is a text prompt on the shared features.
               One backbone pass per frame, then a light decoder pass per class.
  visual       few-shot: one support/query collage per class (FSS-SAM3 layout), support boxes as prompts.
  text+visual  the collage plus the class name. Visual modes encode one canvas per class, so they are
               several times slower than `text`; restrict `detector.classes` when using them live.

Cost (RTX 3060, bf16, text prompts): ~0.3 s image encoding + ~0.08 s per class, 5.3 GB VRAM. All 29
bank classes take ~2.6 s per frame; ~8 classes fit the 1 Hz budget. Class text is encoded once.
Each returned SAM3 instance becomes one detection (mask, tight box, SAM3 presence score).
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from ..config import Config
from ..ontology import CLASSES
from ..supports import SupportBank, resolve_detector_classes
from .base import Detection, DetectionResult, mask_box

log = logging.getLogger(__name__)


def one_shot_canvas(support_rgb, support_mask, query_rgb, size=1008, ratio=.6):
    """Query on top, support below (FSS-SAM3 evaluate.py 1-shot layout)."""
    ys, xs = np.nonzero(support_mask)
    support_h = int(size * ratio)
    query_h = size - support_h
    canvas = Image.new('RGB', (size, size))
    canvas.paste(Image.fromarray(query_rgb).resize((size, query_h), Image.Resampling.BILINEAR), (0, 0))
    canvas.paste(Image.fromarray(support_rgb).resize((size, support_h), Image.Resampling.BILINEAR), (0, query_h))
    height, width = support_mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max() + 1), int(ys.min()), int(ys.max() + 1)
    box = [(x0 + x1) / 2 / width, (query_h + (y0 + y1) / 2 / height * support_h) / size,
           (x1 - x0) / width, (y1 - y0) / height * support_h / size]
    return canvas, [box], (0, 0, size, query_h)


def five_shot_canvas(references, query_rgb, size=1008, ratio=.3):
    """FSS-SAM3 l_shape: three supports across the top, two down the left, query bottom-right."""
    thick = int(round(size * ratio))
    target, top_w = size - thick, size // 3
    left_h = target // 2
    layouts = [(0, 0, top_w, thick), (top_w, 0, top_w, thick), (top_w * 2, 0, size - top_w * 2, thick),
               (0, thick, thick, left_h), (0, thick + left_h, thick, target - left_h)]
    canvas, boxes = Image.new('RGB', (size, size)), []
    for (rgb, mask), (x, y, w, h) in zip(references, layouts):
        ys, xs = np.nonzero(mask)
        canvas.paste(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR), (x, y))
        mh, mw = mask.shape
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        boxes.append([(x + (x0 + x1) / 2 / mw * w) / size, (y + (y0 + y1) / 2 / mh * h) / size,
                      (x1 - x0) / mw * w / size, (y1 - y0) / mh * h / size])
    canvas.paste(Image.fromarray(query_rgb).resize((target, target), Image.Resampling.BILINEAR), (thick, thick))
    return canvas, boxes, (thick, thick, target, target)


class Sam3Detector:
    name = 'sam3'

    def __init__(self, config: Config):
        import torch
        det, cfg = config.detector, config.detector.sam3
        self.torch, self.device, self.cfg = torch, det.device, cfg
        self.confidence, self.min_mask_px, self.max_per_frame = det.confidence, det.min_mask_px, det.max_per_frame
        repo, checkpoint, bpe = Path(cfg.repo), Path(cfg.checkpoint), Path(cfg.bpe)
        for path in (repo / 'sam3/model_builder.py', checkpoint, bpe):
            if not path.is_file():
                raise FileNotFoundError(f'Missing SAM3 asset: {path} (see models/README.md)')

        visual = cfg.prompt != 'text'
        bank = None
        if visual or det.classes in (None, '', 'auto'):
            try:
                bank = SupportBank(config.paths.supports)
            except FileNotFoundError:
                if visual:
                    raise
        if bank is None and det.classes in (None, '', 'auto'):
            self.classes = CLASSES          # text prompts need no examples
        else:
            self.classes = resolve_detector_classes(det.classes, bank, minimum=cfg.shots if visual else 1)
        self.references = {}
        if visual:
            selection = bank.select(self.classes, cfg.shots)
            short = [c.name for c in self.classes if len(selection[c]) < cfg.shots]
            if short:
                raise ValueError(f'Classes with fewer than {cfg.shots} supports: {short}')
            self.references = {c.id: [s.load() for s in selection[c]] for c in self.classes}

        sys.path.insert(0, str(repo.resolve()))
        import sam3
        from sam3.model.sam3_image_processor import Sam3Processor
        self.model = sam3.build_sam3_image_model(bpe_path=str(bpe), device=self.device,
                                                 checkpoint_path=str(checkpoint), load_from_HF=False,
                                                 eval_mode=True, compile=False)
        self.model.eval().requires_grad_(False)
        self.processor = Sam3Processor(self.model, resolution=cfg.resolution, device=self.device,
                                       confidence_threshold=self.confidence)
        # Class names never change: encode them once (the processor would re-encode every frame).
        self.text_features = {}
        if cfg.prompt in ('text', 'text+visual'):
            with torch.inference_mode(), self._autocast():
                for item in self.classes:
                    self.text_features[item.id] = self.model.backbone.forward_text([item.name], device=self.device)
        self.info = dict(name=self.name, model='SAM3', checkpoint=str(checkpoint), source=str(repo),
                         prompt=cfg.prompt, precision=cfg.precision, shots=cfg.shots if visual else 0, resolution=cfg.resolution,
                         confidence=self.confidence, classes=[c.name for c in self.classes])
        log.info('SAM3 ready: %s prompts, %d classes', cfg.prompt, len(self.classes))

    def _instances(self, state, crop, shape, item):
        x, y, w, h = crop
        out = []
        if not len(state['scores']):
            return out
        masks = state['masks'][:, 0, y:y + h, x:x + w].cpu().numpy()
        for mask, score in zip(masks, state['scores'].float().cpu().numpy()):
            if (w, h) != (shape[1], shape[0]):
                mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            if mask.sum() < self.min_mask_px:
                continue
            out.append(Detection(mask_box(mask), float(score), item.id, item.name, mask))
        return out

    def _ground(self, state, item):
        """Same as Sam3Processor.set_text_prompt, with the class text features cached."""
        self.processor.reset_all_prompts(state)
        state['backbone_out'].update(self.text_features[item.id])
        state['geometric_prompt'] = self.model._get_dummy_prompt()
        return self.processor._forward_grounding(state)

    def _autocast(self):
        enabled = self.cfg.precision == 'bf16' and str(self.device).startswith('cuda')
        return self.torch.autocast('cuda', dtype=self.torch.bfloat16, enabled=enabled)

    def detect(self, bgr: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        shape, detections = rgb.shape[:2], []
        with self.torch.inference_mode(), self._autocast():
            if self.cfg.prompt == 'text':
                state = self.processor.set_image(Image.fromarray(rgb))
                for item in self.classes:
                    state = self._ground(state, item)
                    detections += self._instances(state, (0, 0, shape[1], shape[0]), shape, item)
            else:
                for item in self.classes:
                    refs = self.references[item.id]
                    if len(refs) == 1:
                        canvas, boxes, crop = one_shot_canvas(*refs[0], rgb, self.cfg.resolution,
                                                              self.cfg.support_ratio_1shot)
                    else:
                        canvas, boxes, crop = five_shot_canvas(refs, rgb, self.cfg.resolution,
                                                               self.cfg.support_ratio_5shot)
                    state = self.processor.set_image(canvas)
                    self.processor.reset_all_prompts(state)
                    if self.cfg.prompt == 'text+visual':
                        state = self._ground(state, item)
                    for box in boxes:
                        state = self.processor.add_geometric_prompt(state=state, box=box, label=True)
                    detections += self._instances(state, crop, shape, item)
        detections.sort(key=lambda d: d.score, reverse=True)
        detections = detections[:self.max_per_frame]
        return DetectionResult([d for d in detections if d.score >= self.confidence])

    def metadata(self) -> dict:
        return dict(self.info)
