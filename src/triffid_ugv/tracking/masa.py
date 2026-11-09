"""MASA-R50 appearance tracker (optional, `tracker.backend: masa`; needs the `masa` Docker target).

MASA associates the detector's boxes by learned instance appearance, so an object can be re-found
after the robot turns away and back. Its track IDs are passed to the world map as *hints*: a hinted
object keeps its identity even when its geo position is noisy, and (with `masa.geo_merge`) a new MASA
track that appears on top of a known object of the same class is merged into it instead of creating
a duplicate.

Setup follows the evaluated live profile: plug-and-play R50 config, external detections, 1 fps,
category gate (patched in the Docker image), init score 0.35, recovery down to the detector's floor.
"""
from __future__ import annotations

from collections import defaultdict, deque
import importlib
import logging
from pathlib import Path
import sys
import types

import cv2
import numpy as np

from ..config import MasaConfig
from ..supports import sha256_file

log = logging.getLogger(__name__)
_PERSON_SENTINEL = 10000


class MasaTracker:
    name = 'masa'

    def __init__(self, config: MasaConfig, recovery_floor: float = 0.1, device: str = 'cuda'):
        import torch
        from mmengine.config import Config
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules

        repo, checkpoint = Path(config.repo).resolve(), Path(config.checkpoint).resolve()
        source = repo / 'masa/models/tracker/masa_tao_tracker.py'
        if not source.is_file():
            raise FileNotFoundError(f'MASA source not found at {repo}; build the `masa` Docker target')
        if 'TRIFFID: respect the category gate' not in source.read_text():
            raise RuntimeError('MASA source lacks the category-gate patch (docker/patch_masa.py)')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'MASA checkpoint not found: {checkpoint} (see models/README.md)')
        register_all_modules()
        # Import only the MASA modules the plug-and-play tracker needs (the package __init__ pulls in
        # optional detectors and their dependencies).
        for name, relative in [('masa', 'masa'), ('masa.models', 'masa/models'), ('masa.models.necks', 'masa/models/necks'),
                               ('masa.models.mot', 'masa/models/mot'), ('masa.models.tracker', 'masa/models/tracker')]:
            module = types.ModuleType(name)
            module.__path__ = [str(repo / relative)]
            sys.modules[name] = module
        for name in ['masa.models.losses', 'masa.models.necks.deform_fusion',
                     'masa.models.tracker.masa_tao_tracker', 'masa.models.mot.masa']:
            importlib.import_module(name)
        cfg = Config.fromfile(str(repo / 'configs/masa-one/masa_r50_plug_and_play.py'))
        cfg.model.tracker.update(fps=1, memo_tracklet_frames=int(config.memory_s), max_distance=-1, with_cats=True,
                                 init_score_thr=float(config.init_score), obj_score_thr=float(recovery_floor),
                                 match_score_thr=float(config.match_score))
        self.model = MODELS.build(cfg.model)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state.get('state_dict', state), strict=True)
        del state
        self.device = device
        self.model.to(device).eval().requires_grad_(False)
        self.torch, self.memory_s = torch, int(config.memory_s)
        self.first_ns = self.last_ns = None
        self.info = dict(name='masa_r50', checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                         memory_s=self.memory_s, init_score=config.init_score, recovery_floor=recovery_floor,
                         match_score=config.match_score, category_gate=True, geo_merge=config.geo_merge)

    def metadata(self) -> dict:
        return dict(self.info)

    def update(self, detections, bgr: np.ndarray, stamp_ns: int, groups=None) -> list[tuple[int, str]]:
        """Track candidate detections; returns (index into `detections`, track id) for tracked ones."""
        from mmdet.structures import DetDataSample, TrackDataSample
        torch = self.torch
        if self.last_ns is not None and stamp_ns <= self.last_ns:
            return []
        self.first_ns = stamp_ns if self.first_ns is None else self.first_ns
        self.last_ns = stamp_ns
        # MASA resets on frame_id == 0, so frame ids are whole seconds since start, from 1.
        tick = 1 + (stamp_ns - self.first_ns) // 1_000_000_000
        tracker = self.model.tracker
        tracker.tracks = {k: v for k, v in tracker.tracks.items() if tick - v['last_frame'] < self.memory_s}
        if not detections:
            return []
        h, w = bgr.shape[:2]
        scale = 1024 / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        rgb = cv2.resize(bgr, (nw, nh))[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().to(self.device)
        mean = tensor.new_tensor([123.675, 116.28, 103.53])[:, None, None]
        std = tensor.new_tensor([58.395, 57.12, 57.375])[:, None, None]
        tensor = torch.nn.functional.pad((tensor - mean) / std, (0, (-nw) % 32, 0, (-nh) % 32))
        sample = DetDataSample(metainfo=dict(frame_id=int(tick), ori_shape=(h, w), img_shape=(nh, nw),
                                             scale_factor=(nw / w, nh / h)))
        sample.det_bboxes = torch.tensor([[*d.bbox, d.score] for d in detections], device=self.device)
        labels = [(groups or {}).get(d.class_name, d.class_id) for d in detections]
        sample.det_labels = torch.tensor(labels, device=self.device)
        with torch.inference_mode():
            output = self.model.predict(tensor[None, None], [TrackDataSample(video_data_samples=[sample])])
        tracks = output[0][0].pred_track_instances

        def key(box, score, label):
            return int(label), float(np.float32(score)), tuple(np.asarray(box, dtype=np.float32).tolist())

        index = defaultdict(deque)
        for i, (d, label) in enumerate(zip(detections, labels)):
            index[key(d.bbox, d.score, label)].append(i)
        out = []
        for box, score, label, track_id in zip(tracks.bboxes.cpu().tolist(), tracks.scores.cpu().tolist(),
                                               tracks.labels.cpu().tolist(), tracks.instances_id.cpu().tolist()):
            source = index[key(box, score, label)]
            if source:
                out.append((source.popleft(), f'masa:{int(track_id)}'))
        return out


def masa_label_groups(class_groups) -> dict[str, int]:
    """Classes in one tracker group share a MASA label so a person can change role without a new track."""
    from ..ontology import BY_NAME
    mapping = {}
    for index, group in enumerate(class_groups):
        for name in group:
            mapping[name] = _PERSON_SENTINEL + index
    return {k: v for k, v in mapping.items() if k in BY_NAME}
