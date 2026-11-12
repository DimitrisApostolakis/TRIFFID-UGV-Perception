# TRIFFID UGV Perception

Live detection, tracking and geolocation for the TRIFFID UGV (Unitree B2W) with
[YOLOE](https://github.com/THU-MIG/yoloe):

- **Detection** of the TRIFFID classes on the front RealSense RGB frames, with visual prompts built
  from a support bank (example image/mask pairs, no training).
- **Geolocation** of every detection from aligned depth, the robot's GNSS position and compass heading.
- **One identity per real-world object**: association in metres on the map, helped by SIFT appearance
  matching (default) or MASA (fallback).

The node runs on the ground station, reads the robot's topics over ROS 2 and writes the object map as
GeoJSON: to local files and, when enabled, to the crispres map manager.

```
ROS 2 topics (ground station)                     triffid_ugv (one container, one node)
color / aligned depth  ──► newest RGB + closest depth (≤ 1.5 s) ──┐
ugv_status / API / fix ──► position ─┐                             │
magnetometer heading   ──► heading  ─┴─► pose                      ▼
                                          YOLOE-v8-L (visual prompts from supports/)
                                          ► SIFT | MASA appearance hints
                                          ► depth under each mask ► base_link ► ENU ► WGS-84
                                          ► object map (association in metres, ≤ 3 m gate)
                                                                   ▼
outputs/   ugv_detections.geojson · <session>/objects.geojson · observations.jsonl · frames/ · debug/
api        crispres map manager (PUT / PATCH, optional)
```

## Folders

| Folder | You provide | Notes |
|---|---|---|
| `supports/` | support bank (`bank.json` + images/masks) | Not in the repo; mount it with `SUPPORTS_DIR`. Every class with at least one example is detected. See [supports/README.md](supports/README.md) |
| model weights | `yoloe-v8l-seg.pt` (+ `masa_r50.pth` for the MASA fallback) | Mounted at `/models` with `MODELS_DIR`. See [models/README.md](models/README.md) |
| `outputs/` | – | Written by the run, one folder per session |
| `state/` | – | YOLOE visual-prompt cache and the map-API id map; keep it between runs |

Everything in these folders is git-ignored.

## Run with Docker (GPU)

Requires Docker with the NVIDIA Container Toolkit and an NVIDIA driver ≥ 570. The image uses PyTorch
built for CUDA 12.8 (RTX 20xx–50xx) on ROS 2 Jazzy, and runs with the host network so it sees the
robot's topics.

```bash
cp .env.example .env        # ROS domain, MODELS_DIR, SUPPORTS_DIR, UGV_UID/GID, API switch
docker compose build ugv

docker compose run --rm ugv check           # validate config, weights and support bank, no model, no ROS
docker compose up -d ugv                    # live (configs/default.yaml, or UGV_CONFIG in .env)
docker compose logs -f ugv
docker compose down
```

Common variants, all without code changes:

```bash
UGV_CONFIG=configs/live_debug.yaml docker compose up -d ugv          # also save rgb/depth/thermal/overlay per frame
UGV_ARGS="--set detector.confidence=0.25" docker compose up -d ugv   # any --set override
docker compose --profile masa up -d ugv-masa                         # MASA tracker (builds the `masa` image)
docker compose run --rm ugv ros2 topic list                          # ROS tools inside the image
```

On the ground station the node subscribes to the `*_broadcasted` topics of the topics broadcaster: see
[docs/GROUND_STATION.md](docs/GROUND_STATION.md). To test without the robot, play a bag:
[docs/REPLAY.md](docs/REPLAY.md). The image pins the YOLOE fork to commit `40cd606`. Weights and
supports are never baked into the image.

## Configuration

All parameters live in [`configs/default.yaml`](configs/default.yaml). Override any of them with
`--set section.key=value` (YAML value syntax, e.g. in `UGV_ARGS`) or pass another file with
`--config` (`UGV_CONFIG`). Full reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

| Key | Default | Meaning |
|---|---|---|
| `detector.backend` | `yoloe` | YOLOE-v8-L with visual prompts (`dummy` for wiring tests) |
| `detector.classes` | `auto` | `auto` = every class with examples in the bank, or a list of names |
| `detector.confidence` | `0.35` | Minimum score to localise and map a detection |
| `detector.yoloe.image_size` | `640` | Inference size (1280 can help small objects, slower) |
| `tracker.backend` | `sift` | `sift` (CPU, no weights) or `masa` (GPU, `masa` image) |
| `tracker.geo.gate_m`, `large_gate_m` | `3.0`, `6.0` | Association distance; the large gate is for buildings, roads, fences, … |
| `tracker.hint_gate_m` | `10.0` | An appearance match counts only this close to its object |
| `tracker.min_hits` | `2` | Sightings before an object is exported |
| `pipeline.rate_hz` | `1.0` | Frames processed per second (the newest frame wins) |
| `pipeline.max_rgbd_dt_s` | `1.5` | Largest RGB/depth stamp gap |
| `camera.calibrations` | 640x480, 1280x720 | Intrinsics per image size when no matching CameraInfo is published |
| `position.sources` | `[status, api, fix, fixed]` | Position priority; each source is used only while fresh |
| `position.heading_offset_deg` | `0.0` | Magnetic declination + mounting offset (≈ +5° in Greece) |
| `debug.enabled` | `false` | Save rgb, depth, colourised depth, thermal and overlay per frame |
| `api.enabled` | `false` | Upload to the crispres map manager (`API_ENABLED` in `.env`) |

## Outputs

| File | Contents |
|---|---|
| `ugv_detections.geojson` | Confirmed objects of the newest session, rewritten every 5 s |
| `<session>/objects.geojson` | The same, kept per session (`<session>` = UTC start time) |
| `<session>/observations.jsonl` | One line per processed frame: pose and its sources, timings, every detection with box, score, range, lat/lon, `object_id` and appearance `track` |
| `<session>/frames/` | Annotated frames with detections |
| `<session>/debug/` | With `debug.enabled`: `rgb`, `depth` (16-bit mm), `depth_color`, `thermal`, `overlay`, one set per frame under the same name |
| `<session>/run.json` | Resolved config, detector and tracker metadata (checkpoint hashes, classes, support counts) |

GeoJSON features are RFC 7946 Points (LineStrings for fences and walls) with the map manager's
properties. Details: [docs/OUTPUTS.md](docs/OUTPUTS.md) and [docs/DEBUG.md](docs/DEBUG.md).

## Code layout

```
src/triffid_ugv/
  cli.py              commands: run, check, version
  config.py           typed config (YAML + --set overrides)
  app.py              builds the pipeline from a config
  pipeline.py         one frame end to end: detect -> hints -> localise -> geolocate -> associate -> outputs (no ROS)
  ros/                node.py (subscriptions, 1 Hz scheduler, RGB/depth pairing), images.py (msg -> numpy)
  detectors/          yoloe.py, dummy.py (contract in base.py)
  tracking/           world.py (object map, association in metres), sift.py (default hints), masa.py (fallback)
  geometry.py         intrinsics, extrinsics, depth sampling and back-projection
  geodesy.py          body -> ENU -> WGS-84, heading conventions
  position.py         position and heading sources, freshness, fallbacks
  supports.py         support bank loading and validation
  outputs/            geojson.py, files.py (session files), debug.py (debug recording), api.py (map manager)
  ontology.py         TRIFFID classes, categories and map styles
interfaces/           ugv_status_interfaces, gnss_converter_interfaces (built into the image)
```

To add a detector, implement `detect(bgr) -> DetectionResult` and `metadata()` (see
`detectors/base.py`) and register it in `detectors/__init__.py`. A tracker implements
`update(detections, bgr, stamp_ns) -> [(index, track_id)]` and `metadata()`.

More: [architecture](docs/ARCHITECTURE.md) · [ground station](docs/GROUND_STATION.md) ·
[testing with a bag](docs/REPLAY.md).

## Tests

```bash
docker run --rm -v $PWD:/app -w /app --entrypoint python triffid-ugv-perception:latest -m pytest -q
```

The tests use synthetic frames, a synthetic support bank and a model-free detector, so they need no
GPU, weights or ROS.

## Third-party

YOLOE (an Ultralytics fork, AGPL-3.0) and MASA are fetched at build time and are subject to their own
licences. Their source code and weights are not redistributed here.

## Acknowledgements

- [TRIFFID-HUA-Perception](https://github.com/AnnaMi0/TRIFFID-HUA-Perception), the baseline this
  module was built from.
- [YOLOE](https://github.com/THU-MIG/yoloe), the open-vocabulary detector used here.
- [MASA](https://github.com/siyuanliii/masa), the fallback appearance tracker.
- [topics_broadcaster_module_v2](https://github.com/foteinosKonstantinos/topics_broadcaster_module_v2),
  which brings the robot's camera and heading topics to the ground station.
