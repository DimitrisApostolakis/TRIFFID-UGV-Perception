# Configuration

All settings live in [`configs/default.yaml`](../configs/default.yaml), one commented line per key, and
are mirrored by typed dataclasses in `src/triffid_ugv/config.py`. Unknown keys and invalid values stop
the node at startup.

Order of precedence: built-in defaults ← YAML file (`--config`) ← `--set section.key=value`
(repeatable, YAML syntax). A YAML file may list only the keys it changes, e.g. `configs/replay_pui.yaml`.

With Docker Compose, put extra overrides in `.env`:

```bash
UGV_ARGS=--set tracker.sift.min_matches=12 --set position.heading_offset_deg=5.1
API_ENABLED=true
```

| Section | What it controls |
|---|---|
| `paths` | `outputs`, `state` (caches, API id map) and `supports` folders |
| `topics` | every input topic name and the camera-topic QoS. `''` disables an input |
| `pipeline` | processing rate (`rate_hz`), RGB/depth pairing tolerance, frame staleness |
| `camera` | intrinsics source and the per-resolution `calibrations` table, depth scale/range, extrinsics source and static mount |
| `detector` | backend (`yoloe`, or `dummy` for wiring tests), classes, confidence, YOLOE settings |
| `tracker` | appearance backend (`sift` default / `masa` fallback), confirmation hits, matching horizon, class groups, geo and hint gates, per-backend settings |
| `position` | position/heading source priority, timeouts, heading convention and offset, fixed fallbacks |
| `geometry` | which classes become LineString / Polygon, minimum extent |
| `outputs` | GeoJSON / JSONL / annotated frames and their periods |
| `debug` | live-test recording of RGB, depth (raw + colourised), thermal and overlays per frame, with disk limits ([DEBUG.md](DEBUG.md)) |
| `api` | map manager URL, upload period, de-duplication and update thresholds, optional token |

## Common changes

* **Different robot topics**: `topics.rgb`, `topics.depth`, `topics.heading`, ...
* **Calibrate the camera mount**: `camera.static_translation_m`, `camera.static_rpy_deg`
  (camera x forward). TF wins when it is published (`camera.extrinsics: auto`).
* **Magnetic declination**: `position.heading_offset_deg` (east positive).
* **Only some classes**: `detector.classes: [citizen, first responder, fire truck]`.
* **MASA fallback tracker**: `docker compose --profile masa up -d ugv-masa` (sets `tracker.backend=masa`;
  needs `masa_r50.pth` in `MODELS_DIR`). Stop `ugv` first.
* **SIFT too strict / too loose**: lower `tracker.sift.min_matches` to keep more identities, or raise
  it (and `ransac_px` down) if different objects get merged.
* **Record everything during a live test**: `UGV_CONFIG=configs/live_debug.yaml` (or
  `--set debug.enabled=true`); turn single streams off with e.g. `--set debug.thermal=false`.
* **Polygons for areas**: `geometry.polygon_classes: [building, water]`.
