# TRIFFID UGV Perception

Live perception for the TRIFFID UGV (Unitree B2W). Once per second the node takes the newest front
RealSense RGB-D frame, detects the TRIFFID classes, measures each object's range from aligned depth,
geolocates it from the robot's GNSS position and heading, keeps one identity per real-world object and
publishes the map as GeoJSON: to local files and, when enabled, to the crispres map manager.

```
 ROS 2 topics                       triffid_ugv (one container, one node)
 ─────────────                      ──────────────────────────────────────────────────────────────
 RGB  ─┐                            every 1 s: newest RGB + closest depth
 depth ┴─ synchronise ──────────▶  detector (YOLOE | SAM3) ─▶ RGB-D localisation (body frame)
 CameraInfo / TF (optional) ─────▶        │                            │
 ugv_status / fix / heading ─▶ pose ─────────────────────────▶ geolocation (ENU / WGS-84)
                                                                       │
                                      tracker: geo association (default) | MASA hints + geo
                                                                       │
                                     outputs/ GeoJSON + JSONL + frames ┴ map manager API (PUT/PATCH)
```

It runs next to [TRIFFID-UAV-Perception](https://github.com/DimitrisApostolakis/TRIFFID-UAV-Perception)
on the same machine: separate container, same GPU, same `MODELS_DIR` for the shared SAM3 weights.

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

## Choosing the model and tracker

Everything is selected in [`configs/default.yaml`](configs/default.yaml) or with `--set` (e.g. in
`UGV_ARGS`). No code changes are needed to swap components.

| Setting | Options |
|---|---|
| `detector.backend` | **`yoloe`** (default): YOLOE-v8-L with visual prompts built from the support bank (29 classes in the PUI bank). Open-vocabulary, and new classes only need examples, not training. ~50–80 ms per 1280x720 frame on an RTX 3060, ~2 GB VRAM. `sam3`: SAM3 with class-name text prompts (bf16: ~0.3 s + 0.08 s per class, 5.3 GB VRAM; keep the class list short for 1 Hz) or few-shot visual prompts. `dummy`: wiring tests |
| `tracker.backend` | **`geo`** (default): association in metres on the map. `masa`: MASA-R50 appearance tracking whose IDs are combined with the geo association (`docker compose --profile masa up -d ugv-masa`) |

Why geo association is the default: at 1 frame/s on a moving robot, boxes of the same object barely
overlap between frames, so IoU/Kalman trackers in image space fragment identities. Matching each
detection to known objects by distance on the ground is independent of the frame rate and of where
the camera looks. MASA adds appearance, which helps keep two nearby people apart and recover an
object after the robot turns away. Run both on the same bag and compare (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#comparing-trackers)).

## Geolocation

Position comes from `/b2/ugv_status` (RTK, while `gnss_valid`), `/fix` (median-filtered),
the crispres status API or a fixed seed, in the configured priority order. Heading comes from the
magnetometer topic (degrees, compass) or `ugv_status.heading`. Each source is used only while fresh.
Set `position.heading_offset_deg` to the local magnetic declination (≈ +5° in Greece) when the
magnetometer is the heading source. The camera pose in `base_link` comes from TF when published,
otherwise from `camera.static_*` (defaults measured from the robot's TF tree). Without a fresh pose
the frame is still detected and logged, but nothing is placed on the map.

## Repository layout

```
configs/            default.yaml (every setting, commented), replay_pui.yaml
src/triffid_ugv/
  config.py         typed config sections + YAML/--set loading
  detectors/        yoloe.py, sam3.py, dummy.py   (Detector contract in base.py)
  tracking/         world.py (geo association / object map), masa.py
  geometry.py       intrinsics, extrinsics, depth sampling and back-projection
  geodesy.py        body -> ENU -> WGS-84, heading conventions
  position.py       pose estimator (position + heading sources, freshness, fallbacks)
  outputs/          geojson.py (features), files.py (session files), api.py (map manager)
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
[outputs and API](docs/OUTPUTS.md) · [testing with a bag](docs/REPLAY.md)
