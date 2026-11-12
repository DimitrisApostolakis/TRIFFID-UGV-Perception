# Debug recording for live tests

`debug.enabled` saves what the node actually saw and decided, once per processed frame. Use it when
testing on the robot or ground station to check camera, depth and thermal input, calibration and
detections offline afterwards. It is off by default. The map, GeoJSON and API behave the same with it
on or off.

```bash
# .env
UGV_CONFIG=configs/live_debug.yaml      # debug on, plain annotated frames off (overlay/ replaces them)
API_ENABLED=false                       # keep test results off the live map

docker compose up -d ugv && docker compose logs -f ugv   # "Ready: ... | DEBUG recording rgb, depth, ..."
```

Or add it to any config: `UGV_ARGS=--set debug.enabled=true`.

## What is written

Every set uses the RGB stamp as file stem, so the same name in each folder is one frame:

```
outputs/<session>/debug/
  rgb/<stamp>.jpg          colour frame exactly as passed to the detector
  depth/<stamp>.png        aligned depth as uint16 millimetres (lossless: cv2.imread(p, cv2.IMREAD_UNCHANGED) / 1000 = metres)
  depth_color/<stamp>.jpg  depth colourised (TURBO, camera.min_depth_m .. debug.depth_color_max_m), invalid = black, % valid in the header
  thermal/<stamp>.jpg      FLIR frame closest to the RGB stamp, if within debug.thermal_max_dt_s
  overlay/<stamp>.jpg      masks, boxes, class, score, object number, range; header with time + pose (or why there is none)
```

`<stamp>` is `<unix seconds>_<milliseconds>`. Each line of `observations.jsonl` lists the set's files
under `debug_files`, next to that frame's detections, pose and timings. That links every image to the
numbers.

## Settings (`debug:` in `configs/default.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | master switch |
| `rgb`, `depth`, `depth_color`, `thermal`, `overlay` | `true` | each stream on/off |
| `period_s` | `0` | 0 = every processed frame; otherwise at most one set per period (e.g. 5 on long runs) |
| `save_empty` | `true` | also record frames without detections |
| `depth_color_max_m` | `10.0` | far end of the depth colour scale |
| `thermal_max_dt_s` | `1.0` | largest RGB/thermal stamp gap that is still paired |
| `jpeg_quality` | `90` | JPEG quality of rgb / depth_color / thermal / overlay |
| `queue_size` | `8` | sets waiting for the writer thread; more are dropped (counted), inference never waits |
| `max_sets` | `0` | stop after this many sets (0 = unlimited) |
| `min_free_gb` | `2.0` | stop when the outputs disk has less free space |
| `folder` | `debug` | folder name inside the session folder |

## How it works

`outputs/debug.py` (`DebugRecorder`) is created by `app.build_pipeline` only when enabled. The pipeline
hands it the frame, the depth resampled onto the RGB pixels, the overlay (drawn once and shared with
`outputs.frames`) and the paired thermal frame. Encoding and writing happen on one background thread.
The ROS node subscribes to `topics.thermal` only when `debug.thermal` is on. Like RGB and depth,
thermal messages are stored serialized, and only the one paired with a recorded frame is decoded.

The heartbeat log line every 30 s shows `debug {sets, dropped, failed, queued, stopped}`. A non-zero
`dropped` means the disk cannot keep up: raise `period_s` or turn streams off.
