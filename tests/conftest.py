import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from triffid_ugv.config import load_config


@pytest.fixture
def config(tmp_path):
    return load_config(None, ['detector.backend=dummy', 'tracker.min_hits=1'], root=tmp_path)


@pytest.fixture
def bank(tmp_path) -> Path:
    """A two-class bank: citizen with 2 examples, fence with 1 and one empty slot."""
    root = tmp_path / 'bank'
    entries = {}
    for name, count in [('citizen', 2), ('fence', 1)]:
        entries[name] = []
        for i in range(count):
            image = np.full((40, 60, 3), 30 * (i + 1), np.uint8)
            mask = np.zeros((40, 60), np.uint8)
            mask[10:30, 20:40] = 255
            (root / name).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / name / f'{i}.png'), image)
            cv2.imwrite(str(root / name / f'{i}_m.png'), mask)
            entries[name].append({'id': f'e{i}', 'image': f'{name}/{i}.png', 'mask': f'{name}/{i}_m.png'})
    entries['fence'].append({'id': 'empty', 'image': '', 'mask': ''})
    entries['water'] = []
    (root / 'bank.json').write_text(json.dumps({'schema_version': 'triffid.ugv.support-bank.v1',
                                                'examples_per_class': 16, 'classes': entries}))
    return root
