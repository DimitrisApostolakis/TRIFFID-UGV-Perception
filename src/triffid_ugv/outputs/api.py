"""Upload confirmed objects to the crispres map manager (TELESTO), in a background thread.

Protocol (as used by the live UGV system):
  GET   {url}            -> FeatureCollection of the map's current features
  PUT   {url}            body {geometry, properties} -> creates a feature, response {"id": ...}
  PATCH {url}/{id}       body {geometry, properties} -> updates that feature

PUT always creates, so blind re-uploads would duplicate. Each local object is therefore created once
and remembered (object id -> remote id, persisted in ``paths.state`` so a restart does not duplicate).
Before creating, the remote map is read once per flush and a same-class feature within
`dedup_radius_m` is adopted instead. A known object is PATCHed only when it moved more than
`update_distance_m`, its confidence rose by `update_confidence_delta`, or its class/geometry changed.
Network failures never block the pipeline: the newest snapshot simply waits for the next flush.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..config import ApiConfig
from ..geodesy import haversine_m
from .geojson import feature_centroid

log = logging.getLogger(__name__)


class MapApiClient:
    def __init__(self, config: ApiConfig, state_file: str | Path, opener=urlopen):
        self.config, self.opener = config, opener
        self.state_file = Path(state_file)
        self.known: dict[str, dict] = {}
        if self.state_file.is_file():
            try:
                self.known = json.loads(self.state_file.read_text()).get('objects', {})
            except (OSError, ValueError) as error:
                log.warning('Ignoring unreadable API state %s: %s', self.state_file, error)
        token = os.environ.get(config.token_env) if config.token_env else None
        self.headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
        self._pending = None
        self._condition = threading.Condition()
        self._stopped = False
        self._thread = None
        self.counters = dict(created=0, adopted=0, updated=0, unchanged=0, errors=0)

    # ---- lifecycle ----------------------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name='map-api', daemon=True)
        self._thread.start()

    def submit(self, features: list[dict]):
        """Queue the newest snapshot; an older unsent snapshot is replaced (it is a subset)."""
        with self._condition:
            self._pending = list(features)
            self._condition.notify()

    def close(self, timeout_s: float = 10.0):
        with self._condition:
            self._stopped = True
            self._condition.notify()
        if self._thread:
            self._thread.join(timeout_s)

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopped:
                    self._condition.wait()
                features, self._pending = self._pending, None
                stopping = self._stopped
            if features:
                try:
                    self.sync(features)
                except Exception as error:      # never let the uploader thread die
                    self.counters['errors'] += 1
                    log.error('Map API sync failed: %s', error)
            if stopping:
                return

    # ---- HTTP ---------------------------------------------------------------------------------
    def _request(self, method: str, url: str, feature: dict | None = None):
        data = None
        if feature is not None:
            data = json.dumps({'geometry': feature['geometry'], 'properties': feature['properties']}).encode()
        request = Request(url, data=data, method=method, headers=self.headers)
        with self.opener(request, timeout=self.config.timeout_s) as response:
            status = getattr(response, 'status', 200)
            if status not in (200, 201):
                raise RuntimeError(f'{method} {url} returned HTTP {status}')
            raw = response.read(8 << 20)
        try:
            return json.loads(raw.decode('utf-8')) if raw else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    # ---- sync ---------------------------------------------------------------------------------
    def _needs_update(self, record: dict, feature: dict, centre) -> bool:
        props = feature['properties']
        if record.get('class') != props.get('class') or record.get('type') != feature['geometry']['type']:
            return True
        if centre and record.get('lon') is not None:
            if haversine_m(record['lat'], record['lon'], centre[1], centre[0]) > self.config.update_distance_m:
                return True
        return float(props.get('confidence', 0)) - float(record.get('confidence', 0)) >= self.config.update_confidence_delta

    def _remember(self, object_id, remote_id, feature, centre):
        self.known[object_id] = {
            'remote_id': str(remote_id), 'class': feature['properties'].get('class'),
            'type': feature['geometry']['type'], 'confidence': float(feature['properties'].get('confidence', 0)),
            'lon': centre[0] if centre else None, 'lat': centre[1] if centre else None}

    def sync(self, features: list[dict]):
        new = [f for f in features if f['id'] not in self.known]
        remote_index = None
        if new:
            try:
                remote = self._request('GET', self.config.url) or {}
                remote_index = []
                for rf in remote.get('features', []) if isinstance(remote, dict) else []:
                    centre = feature_centroid(rf)
                    if centre and rf.get('id') is not None:
                        remote_index.append((str((rf.get('properties') or {}).get('class', '')).lower(), centre, rf['id']))
            except Exception as error:
                self.counters['errors'] += 1
                log.warning('Map API GET failed; new objects wait for the next flush: %s', error)

        for feature in features:
            object_id, centre = feature['id'], feature_centroid(feature)
            record = self.known.get(object_id)
            try:
                if record is None:
                    if remote_index is None:
                        continue
                    name = str(feature['properties'].get('class', '')).lower()
                    match = None
                    if centre:
                        nearby = [(haversine_m(centre[1], centre[0], c[1], c[0]), rid)
                                  for cls, c, rid in remote_index if cls == name]
                        nearby = [x for x in nearby if x[0] <= self.config.dedup_radius_m]
                        match = min(nearby)[1] if nearby else None
                    if match is not None:
                        self._request('PATCH', f"{self.config.url.rstrip('/')}/{quote(str(match), safe='')}", feature)
                        self._remember(object_id, match, feature, centre)
                        self.counters['adopted'] += 1
                    else:
                        response = self._request('PUT', self.config.url, feature)
                        remote_id = response.get('id') if isinstance(response, dict) else None
                        if remote_id is None:
                            log.warning('Map API created %s without returning an id', object_id)
                        else:
                            self._remember(object_id, remote_id, feature, centre)
                            if centre:
                                remote_index.append((name, centre, remote_id))
                        self.counters['created'] += 1
                elif self._needs_update(record, feature, centre):
                    remote_id = record['remote_id']
                    self._request('PATCH', f"{self.config.url.rstrip('/')}/{quote(remote_id, safe='')}", feature)
                    self._remember(object_id, remote_id, feature, centre)
                    self.counters['updated'] += 1
                else:
                    self.counters['unchanged'] += 1
            except Exception as error:
                self.counters['errors'] += 1
                log.warning('Map API upload of %s failed: %s', object_id, error)
        self._save()
        log.info('Map API: %s', self.counters)

    def _save(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix('.tmp')
            tmp.write_text(json.dumps({'url': self.config.url, 'objects': self.known}, indent=1))
            os.replace(tmp, self.state_file)
        except OSError as error:
            log.warning('Cannot save API state %s: %s', self.state_file, error)
