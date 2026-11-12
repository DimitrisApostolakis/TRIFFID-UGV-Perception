# TRIFFID UGV Perception

Live perception for the TRIFFID UGV (Unitree B2W). Once per second the node takes the newest front
RealSense RGB-D frame, detects the TRIFFID classes, measures each object's range from aligned depth,
geolocates it from the robot's GNSS position and heading, keeps one identity per real-world object and
publishes the map as GeoJSON: to local files and, when enabled, to the crispres map manager.

```
 ROS 2 topics                       triffid_ugv (one container, one node)
 ─────────────                      ──────────────────────────────────────────────────────────────
 RGB  ─┐                            every 1 s: newest RGB + closest depth
 depth ┴─ synchronise ──────────▶  detector (YOLOE-v8-L) ─▶ RGB-D localisation (body frame)
 CameraInfo / TF (optional) ─────▶        │                            │
 ugv_status / fix / heading ─▶ pose ─────────────────────────▶ geolocation (ENU / WGS-84)
                                                                       │
                     tracker: appearance hints (SIFT default | MASA fallback) + geo association
                                                                       │
                                     outputs/ GeoJSON + JSONL + frames ┴ map manager API (PUT/PATCH)
```

It runs next to [TRIFFID-UAV-Perception](https://github.com/DimitrisApostolakis/TRIFFID-UAV-Perception)
on the same machine: separate container, same GPU.

## Quick start

```bash
cp .env.example .env                 # ROS domain, models/supports folders, API switch
# put yoloe-v8l-seg.pt in models/ (see models/README.md) and a support bank in supports/
docker compose build ugv
docker compose run --rm ugv check    # config, weights and support bank, without loading a model
docker compose up -d ugv             # live
docker compose logs -f ugv
```

Results: `outputs/ugv_detections.geojson` (newest map) and `outputs/<session>/` (see
[docs/OUTPUTS.md](docs/OUTPUTS.md)). Enable the map manager with `API_ENABLED=true` in `.env`.
For a live test that also saves every frame's RGB, depth, thermal and detection overlay, set
`UGV_CONFIG=configs/live_debug.yaml` (see [docs/DEBUG.md](docs/DEBUG.md)).

## Choosing the model and tracker

Everything is selected in [`configs/default.yaml`](configs/default.yaml) or with `--set` (e.g. in
`UGV_ARGS`). No code changes are needed to swap components.

| Setting | Options |
|---|---|
| `detector.backend` | **`yoloe`**: YOLOE-v8-L with visual prompts built from the support bank (29 classes in the PUI bank). Open-vocabulary, and new classes only need examples, not training. ~50–80 ms per 1280x720 frame on an RTX 3060, ~2 GB VRAM. (`dummy`: model-free wiring tests only) |
| `tracker.backend` | **`sift`** (default): SIFT keypoints on each detection's mask, matched to recent views of known tracks (ratio test + RANSAC). CPU only, no weights, ~50–250 ms per frame. `masa` (fallback): MASA-R50 learned appearance, ~200 ms of GPU per frame, separate image (`docker compose --profile masa up -d ugv-masa`) |

Both trackers only give *hints*. Identity on the map is decided in metres: at 1 frame/s on a moving
robot, boxes of the same object barely overlap between frames, so image-space IoU/Kalman trackers
fragment identities. Every geolocated detection is matched to known objects by ground distance. An
appearance match keeps an object together when its geo position is noisy (within `tracker.hint_gate_m`).
Detections that SIFT cannot describe (small or textureless) are still associated by distance. Switch
to MASA when SIFT keeps too few identities, e.g. with textureless objects or large viewpoint changes
(see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#comparing-trackers)).

## Geolocation

Position comes from `/b2/ugv_status` (RTK, while `gnss_valid`), the crispres status API, `/fix`
(median-filtered) or a fixed seed, in the configured priority order. Heading comes from the
magnetometer topic (degrees, compass) or `ugv_status.heading`. Each source is used only while fresh.
Set `position.heading_offset_deg` to the local magnetic declination (≈ +5° in Greece) when the
magnetometer is the heading source. The camera pose in `base_link` comes from TF when published,
otherwise from `camera.static_*` (defaults measured from the robot's TF tree). Without a fresh pose
the frame is still detected and logged, but nothing is placed on the map.

## Repository layout

```
configs/            default.yaml (every setting, commented), live_debug.yaml, replay_pui.yaml
src/triffid_ugv/
  config.py         typed config sections + YAML/--set loading
  detectors/        yoloe.py, dummy.py   (Detector contract in base.py)
  tracking/         world.py (geo association / object map), sift.py (default hints), masa.py (fallback)
  geometry.py       intrinsics, extrinsics, depth sampling and back-projection
  geodesy.py        body -> ENU -> WGS-84, heading conventions
  position.py       pose estimator (position + heading sources, freshness, fallbacks)
  outputs/          geojson.py (features), files.py (session files), debug.py (live-test recording), api.py (map manager)
  pipeline.py       one frame end to end, no ROS
  ros/              node.py (subscriptions, 1 Hz scheduler), images.py (msg -> numpy)
interfaces/         ugv_status_interfaces, gnss_converter_interfaces (built into the image)
docker/             entrypoint, MASA source patch
tests/              unit + pipeline tests (no GPU or ROS needed)
```

## Development

```bash
docker run --rm -v $PWD:/app -w /app --entrypoint python triffid-ugv-perception:latest -m pytest -q
```

Docs: [architecture](docs/ARCHITECTURE.md) · [configuration](docs/CONFIGURATION.md) ·
[outputs and API](docs/OUTPUTS.md) · [debug recording](docs/DEBUG.md) · [ground station + broadcaster](docs/GROUND_STATION.md) · [testing with a bag](docs/REPLAY.md)
