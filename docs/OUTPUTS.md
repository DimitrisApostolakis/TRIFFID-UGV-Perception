# Outputs and the map API

## Files (`paths.outputs`)

| Path | Content |
|---|---|
| `ugv_detections.geojson` | confirmed objects of the newest run, rewritten every `outputs.flush_period_s` |
| `<session>/objects.geojson` | the same, kept per run (`<session>` = UTC start time) |
| `<session>/observations.jsonl` | one line per processed frame: stamp, pose (and sources), timings, every detection with box, score, body-frame position, range, lat/lon, `object_id`, appearance `track`, and `debug_files` when a debug set was recorded |
| `<session>/frames/*.jpg` | annotated frames: mask, class, score, object number, range |
| `<session>/debug/{rgb,depth,depth_color,thermal,overlay}/` | only with `debug.enabled`: every processed frame's inputs and overlay, same file stem per set (see [DEBUG.md](DEBUG.md)) |
| `<session>/run.json` | resolved config, detector/tracker metadata (checkpoint hashes, classes, support counts) |

## GeoJSON feature

Geometry is `Point` by default, `LineString` for `geometry.line_classes` (fence, wall), and `Polygon`
for `geometry.polygon_classes`. Coordinates are `[longitude, latitude]` (RFC 7946).

```json
{"type": "Feature", "id": "ugv-20261109T120000Z-00004",
 "geometry": {"type": "Point", "coordinates": [23.7275123, 37.9838456]},
 "properties": {"class": "first responder", "id": "ugv-20261109T120000Z-00004", "confidence": 0.81,
   "category": "person", "detection_type": "seg", "source": "ugv", "local_frame": false,
   "altitude_m": 104.2, "height_m": 1.71, "marker-color": "#1e90ff", "marker-size": "medium",
   "marker-symbol": "pitch", "observations": 12, "first_seen": "2026-11-09T12:00:04Z",
   "last_seen": "2026-11-09T12:00:31Z", "tracker": "geo"}}
```

The property names and styles are the ones the map manager already received from the earlier live UGV
system. `confidence` is the best detector score seen for the object. `observations` is the number of
frames it was seen in.

## Map manager API (`api.enabled`)

* `PUT {url}` creates a feature (body `{geometry, properties}`); the response `id` is remembered in
  `state/map_api_ids.json`, so a restart does not create duplicates.
* Before creating, `GET {url}` is read once per upload. A same-class remote feature within
  `dedup_radius_m` is adopted (PATCHed) instead of creating a new one.
* `PATCH {url}/{id}` updates an object only when it moved more than `update_distance_m`, its confidence
  rose by `update_confidence_delta`, or its class or geometry changed.
* Uploads run on a background thread. If the network is down, the newest snapshot waits for the next
  period, and detection keeps running.
* An optional bearer token is read from the environment variable named by `api.token_env`.
