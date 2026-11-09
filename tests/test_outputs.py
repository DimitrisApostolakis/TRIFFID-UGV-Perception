import json

import numpy as np

from triffid_ugv.config import load_config
from triffid_ugv.geodesy import LocalFrame
from triffid_ugv.outputs.api import MapApiClient
from triffid_ugv.outputs.geojson import feature_centroid, object_feature
from triffid_ugv.tracking.world import WorldObject

LAT, LON = 37.9838, 23.7275


def make_object(name='citizen', e=0.0, n=0.0, score=0.8, oid='ugv-x-00001', line=None):
    return WorldObject(oid, {name: score}, np.array([e, n, 0.0]), 0, 10**9, hits=3, best_score=score,
                       last_score=score, height_m=1.7, line=line)


def test_point_feature_schema(tmp_path):
    config = load_config(None, root=tmp_path)
    f = object_feature(make_object(), LocalFrame(LAT, LON), config.geometry)
    assert f['geometry'] == {'type': 'Point', 'coordinates': [LON, LAT]}
    p = f['properties']
    for key in ('class', 'id', 'confidence', 'category', 'detection_type', 'source', 'local_frame', 'altitude_m',
                'height_m', 'marker-color', 'marker-size', 'marker-symbol'):
        assert key in p
    assert p['category'] == 'person' and p['local_frame'] is False


def test_line_and_polygon(tmp_path):
    config = load_config(None, ['geometry.polygon_classes=[building]'], root=tmp_path)
    frame = LocalFrame(LAT, LON)
    line = object_feature(make_object('fence', line=np.array([[0, 0], [10, 0]])), frame, config.geometry)
    assert line['geometry']['type'] == 'LineString' and 'stroke' in line['properties']
    poly = object_feature(make_object('building'), frame, config.geometry)
    ring = poly['geometry']['coordinates'][0]
    assert poly['geometry']['type'] == 'Polygon' and ring[0] == ring[-1] and 'fill' in poly['properties']
    assert feature_centroid(line)[0] > LON


class FakeServer:
    """Records requests; GET returns `remote`; PUT returns increasing ids."""

    def __init__(self, remote=()):
        self.remote, self.calls, self.next_id = list(remote), [], 100

    def __call__(self, request, timeout):
        body = json.loads(request.data) if request.data else None
        self.calls.append((request.get_method(), request.full_url, body))
        if request.get_method() == 'GET':
            payload = {'type': 'FeatureCollection', 'features': self.remote}
        elif request.get_method() == 'PUT':
            self.next_id += 1
            payload = {'id': self.next_id}
        else:
            payload = {'success': True}
        return _Response(json.dumps(payload).encode())


class _Response:
    status = 200

    def __init__(self, data):
        self.data = data

    def read(self, limit=-1):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def feature(oid='a', lon=LON, lat=LAT, cls='citizen', conf=0.8):
    return {'type': 'Feature', 'id': oid, 'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': {'class': cls, 'id': oid, 'confidence': conf}}


def test_api_creates_once_then_patches_on_change(tmp_path):
    config = load_config(None, root=tmp_path).api
    server = FakeServer()
    client = MapApiClient(config, tmp_path / 'ids.json', opener=server)
    client.sync([feature()])
    assert [c[0] for c in server.calls] == ['GET', 'PUT']
    client.sync([feature()])                                  # unchanged: nothing sent
    assert len(server.calls) == 2
    moved = feature(lat=LAT + 2 / 111_320)                    # ~2 m north
    client.sync([moved])
    assert server.calls[-1][0] == 'PATCH' and server.calls[-1][1].endswith('/101')
    assert set(server.calls[-1][2]) == {'geometry', 'properties'}
    # a restart keeps the id map: no duplicate creation
    again = MapApiClient(config, tmp_path / 'ids.json', opener=server)
    again.sync([moved])
    assert len(server.calls) == 3


def test_api_adopts_nearby_remote_feature(tmp_path):
    config = load_config(None, root=tmp_path).api
    remote = dict(feature('remote'), id=55)
    server = FakeServer([remote])
    client = MapApiClient(config, tmp_path / 'ids.json', opener=server)
    client.sync([feature('new', lat=LAT + 1 / 111_320)])     # 1 m away, same class
    assert [c[0] for c in server.calls] == ['GET', 'PATCH'] and server.calls[1][1].endswith('/55')


def test_api_get_failure_defers_creation(tmp_path):
    def failing(request, timeout):
        raise OSError('offline')
    client = MapApiClient(load_config(None, root=tmp_path).api, tmp_path / 'ids.json', opener=failing)
    client.sync([feature()])
    assert client.known == {} and client.counters['errors'] == 1
