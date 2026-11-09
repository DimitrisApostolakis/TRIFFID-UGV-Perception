import numpy as np
import pytest

from triffid_ugv.config import load_config
from triffid_ugv.ontology import BY_NAME
from triffid_ugv.tracking.world import ObjectMap, Observation

S = 1_000_000_000


def obs(name, e, n, t, score=0.8, hint=None, spread=0.2):
    centre = np.array([e, n, 0.0])
    pts = centre + np.array([[dx, dy, dz] for dx in (-spread, spread) for dy in (-spread, spread) for dz in (0, 1.7)])
    return Observation(BY_NAME[name].id, name, score, (0, 0, 1, 1), centre, pts, np.zeros(3), t * S, hint=hint)


@pytest.fixture
def world(tmp_path):
    config = load_config(None, ['tracker.min_hits=2'], root=tmp_path)
    return ObjectMap(config.tracker, config.geometry, 'ugv-test')


def test_same_object_across_frames_one_id(world):
    for t, (e, n) in enumerate([(10, 5), (10.6, 5.3), (9.7, 4.8), (10.2, 5.1)]):
        world.update([obs('fire truck', e, n, t)], t * S)
    assert len(world.objects) == 1
    obj = next(iter(world.objects.values()))
    assert obj.hits == 4 and obj.enu[:2] == pytest.approx([10.1, 5.05], abs=0.4)


def test_close_people_in_one_frame_stay_separate(world):
    world.update([obs('citizen', 0, 0, 0), obs('citizen', 1.2, 0, 0)], 0)
    world.update([obs('citizen', 1.3, 0.1, 1), obs('citizen', 0.1, 0.1, 1)], S)
    assert len(world.objects) == 2 and all(o.hits == 2 for o in world.objects.values())


def test_roles_share_one_person_and_vote(world):
    world.update([obs('citizen', 3, 3, 0, 0.5)], 0)
    world.update([obs('first responder', 3.4, 3, 1, 0.9)], S)
    world.update([obs('first responder', 3.2, 3.1, 2, 0.8)], 2 * S)
    assert len(world.objects) == 1
    assert next(iter(world.objects.values())).class_name == 'first responder'


def test_different_classes_and_far_objects_do_not_merge(world):
    world.update([obs('citizen', 0, 0, 0), obs('cone', 0.5, 0, 0)], 0)
    world.update([obs('citizen', 8, 0, 1)], S)
    assert len(world.objects) == 3


def test_confirmation_and_expiry(world):
    world.update([obs('cone', 0, 0, 0)], 0)
    assert world.confirmed() == []
    world.update([obs('cone', 0.2, 0, 1)], S)
    assert len(world.confirmed()) == 1
    later = 2000
    world.update([obs('cone', 0.1, 0, later)], later * S)   # past max_age_s: a new object
    assert len(world.objects) == 2


def test_hints_link_identity_beyond_gate(world):
    world.update([obs('citizen', 0, 0, 0, hint='masa:7')], 0)
    world.update([obs('citizen', 9, 0, 1, hint='masa:7')], S)
    assert len(world.objects) == 1


def test_fence_becomes_a_growing_line(world):
    world.update([obs('fence', 0, 0, 0, spread=1.0)], 0)
    world.update([obs('fence', 3.5, 0, 1, spread=1.0)], S)
    assert len(world.objects) == 1
    line = next(iter(world.objects.values())).line
    assert abs(line[1][0] - line[0][0]) == pytest.approx(5.5, abs=0.3)
