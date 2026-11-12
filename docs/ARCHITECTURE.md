# Architecture

## Per-frame flow

1. **Scheduling** (`ros/node.py`). Sensor callbacks only store the newest RGB image, the last 30
   depth images and the latest CameraInfo/pose inputs. A timer at `pipeline.rate_hz` (1 Hz) takes the
   newest RGB image not yet processed and the depth image with the closest stamp (within
   `max_rgbd_dt_s`). Slow inference therefore never builds a backlog.
2. **Calibration**. Intrinsics come from CameraInfo when published (`camera.intrinsics: auto`),
   otherwise from the configured RealSense values, scaled to the image size. Extrinsics come from TF
   (`body_frame` ← image `frame_id`) when available, otherwise from `camera.static_*`.
3. **Detection** (`detectors/`). The detector returns boxes, masks, scores and classes.
   With the MASA tracker, YOLOE also returns low-score candidates that only MASA uses, to recover tracks.
4. **Appearance hints** (`tracking/sift.py` or `tracking/masa.py`). Each detection gets an image-space
   track ID when its appearance matches a recent track. These IDs are hints for step 7.
5. **Localisation** (`geometry.py`). Depth is sampled under each mask (or in the central box region),
   saturated and zero pixels are dropped, mask-edge leakage is removed around the median range, and
   the points are back-projected and transformed into `base_link`.
6. **Geolocation** (`geodesy.py`, `position.py`). Body points are rotated by the compass heading and
   added to the robot position in a local ENU plane anchored at the first fix. Roll and pitch are
   ignored, as in the earlier live system.
7. **Association** (`tracking/world.py`). See below.
8. **Outputs** (`outputs/`). Every frame is appended to `observations.jsonl` and annotated frames are
   saved. Every `outputs.flush_period_s` the confirmed objects are written as GeoJSON, and every
   `api.flush_period_s` they are queued for the map API on a background thread.

## Association (tracker)

Identity is decided on the map. First, an observation whose appearance track already belongs to an
object goes straight to that object. This holds only if the object is the same class group and lies
within `tracker.hint_gate_m` (10 m), which stops two identical-looking objects far apart from merging.
The remaining observations are matched jointly (Hungarian) to the known objects:

* same class, or same `class_groups` entry (person roles are one physical person, label voted by score);
* horizontal distance ≤ `gate_m` (`large_gate_m` for extended classes). Line objects (fence, wall) use
  point-to-segment distance, polygon objects use point-to-polygon distance;
* positions are averaged (exact mean for the first sightings, then an EMA with `position_alpha`);
* an object is exported after `min_hits` sightings and stops being matched after `max_age_s` unseen.

Anything still unmatched becomes a new object.

Appearance hints come from one of two trackers:

`sift` (default, `tracking/sift.py`): up to 200 SIFT keypoints inside each detection's mask, on a crop
downscaled to 320 px. A detection continues a track of the same class group when at least `min_matches`
(10) keypoints survive the Lowe ratio test (0.75), one-to-one matching and a RANSAC homography against
one of the track's last `gallery_size` (4) views. Detections are assigned to tracks jointly (most
verified matches). Tracks are forgotten after `memory_s` (60 s) unseen. Detections under 24 px or with
fewer than 12 keypoints get no hint and rely on geo association. CPU only. Measured at about 140 ms
median for 10 detections with 36 live tracks (synthetic worst case, every detection new). The homography
check suits rigid, roughly planar views (vehicles, signs, buildings). Walking people often fall back to geo.

`masa` (fallback, `tracking/masa.py`, `masa` image): MASA-R50 tracks the detector's boxes (and YOLOE's
low-score candidates) by learned appearance at 1 fps (category-gated, 60 s memory). It handles
textureless objects and large viewpoint changes better than SIFT, at ~200 ms of GPU per frame. With
`masa.geo_merge` (default), new MASA tracks are geo-matched before creating objects, so a
lost-then-refound object does not become a duplicate.

## Comparing trackers

Record (or reuse) a bag and run it twice:

```bash
docker compose run --rm ugv run --config configs/replay_pui.yaml                                # sift
docker compose --profile masa run --rm ugv-masa run --config configs/replay_pui.yaml --set tracker.backend=masa
```

Earlier result, on the 69 s PUI bag (non-RTK `/fix`, odometry-derived heading, so object positions drift
by up to ~17 m), before SIFT existed: geo association alone produced ~100 objects (61 confirmed) and
`masa` 44 (29 confirmed), e.g. 3–4 cones instead of 12. Appearance kept identities together where the
pose could not. MASA hints were not distance-gated then. They now are (`hint_gate_m`), so some of those
merges become separate objects. MASA added ~200 ms per frame (YOLOE ~200 ms, total ~440 ms on an RTX
3060). SIFT has not yet been measured on a bag. Run both trackers on a bag with RTK and the magnetometer.

Compare `objects.geojson` (how many objects for the people/vehicles really present) and
`observations.jsonl` (`object_id` per detection over time). Fewer objects than reality means merges;
more means fragmentation. The annotated frames show the object number next to every detection.

## Shared machine with the UAV pipeline

The UAV container runs FSS-SAM3 in batches. This node keeps YOLOE-v8-L resident (about 2 GB VRAM).
The default SIFT tracker runs on the CPU. The MASA fallback also loads its R50 model onto the GPU.

Measured on an RTX 3060 with the PUI bag: YOLOE `detect` takes 48 ms back to back and 78 ms median at
1 Hz (the GPU clocks down between frames; `sudo nvidia-smi -lgc <min>,<max>` pins the clocks if latency
matters). The whole frame took 200 ms median in the node while a bag player shared the machine.
Camera topics are received serialized and only the chosen frame is deserialized, which keeps the
17 Hz image streams from competing with inference for Python's GIL.

## Known limits

* Depth beyond about 10 m from a D435i is noisy (the error grows with the square of the range).
  `camera.max_depth_m` caps it.
* Magnetometer heading is magnetic north, so set `position.heading_offset_deg`. Heading error dominates
  far-object placement (1° ≈ 17 cm at 10 m).
* `ugv_status.heading` is assumed compass-style (as the GNSS converter's `GNSSPosition` message
  documents). The recorded bags never had RTK, so this could not be checked against data. Set
  `position.status_heading_convention: enu` if it turns out to be ENU yaw.
