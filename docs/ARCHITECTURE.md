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
   YOLOE also returns low-score candidates that only MASA uses, for recovering tracks.
4. **Localisation** (`geometry.py`). Depth is sampled under each mask (or in the central box region),
   saturated and zero pixels are dropped, mask-edge leakage is removed around the median range, and
   the points are back-projected and transformed into `base_link`.
5. **Geolocation** (`geodesy.py`, `position.py`). Body points are rotated by the compass heading and
   added to the robot position in a local ENU plane anchored at the first fix. Roll and pitch are
   ignored, as in the earlier live system.
6. **Association** (`tracking/world.py`). See below.
7. **Outputs** (`outputs/`). Every frame is appended to `observations.jsonl` and annotated frames are
   saved. Every `outputs.flush_period_s` the confirmed objects are written as GeoJSON, and every
   `api.flush_period_s` they are queued for the map API on a background thread.

## Association (tracker)

`geo` (default): each frame's geolocated observations are matched jointly (Hungarian) to known objects:

* same class, or same `class_groups` entry (person roles are one physical person, label voted by score);
* horizontal distance ≤ `gate_m` (`large_gate_m` for extended classes). Line objects (fence, wall) use
  point-to-segment distance, polygon objects use point-to-polygon distance;
* positions are averaged (exact mean for the first sightings, then an EMA with `position_alpha`);
* an object is exported after `min_hits` sightings and stops being matched after `max_age_s` unseen.

`masa`: MASA-R50 tracks the detector's boxes by appearance at 1 fps (category-gated, 60 s memory).
Its track IDs are *hints*: an observation whose MASA track already belongs to an object goes straight
to it, even if its geo position is noisy. With `geo_merge` (default), new MASA tracks are geo-matched
before creating objects, so a lost-then-refound object does not become a duplicate.

## Comparing trackers

Record (or reuse) a bag and run it twice:

```bash
docker compose run --rm ugv run --config configs/replay_pui.yaml                                # geo
docker compose --profile masa run --rm ugv-masa run --config configs/replay_pui.yaml --set tracker.backend=masa
```

First result, on the 69 s PUI bag (non-RTK `/fix`, odometry-derived heading, so object positions drift
by up to ~17 m): `geo` produced ~100 objects (61 confirmed) and `masa` 44 (29 confirmed), e.g. 3–4 cones
instead of 12. Appearance kept identities together where the pose could not. There is no identity
ground truth for this bag, so fewer objects may also include wrong merges (MASA hints bypass the
distance gate). MASA added ~200 ms per frame (YOLOE ~200 ms, total ~440 ms on an RTX 3060).
Repeat the comparison on a bag with RTK and the magnetometer before choosing the live default.

Compare `objects.geojson` (how many objects for the people/vehicles really present) and
`observations.jsonl` (`object_id` per detection over time). Fewer objects than reality means merges;
more means fragmentation. The annotated frames show the object number next to every detection.

## Shared machine with the UAV pipeline

The UAV container runs FSS-SAM3 in batches. This node keeps YOLOE-v8-L resident (about 2 GB VRAM).
SAM3 as the UGV detector needs about 5.3 GB (measured, bf16), so on a 12 GB GPU avoid running SAM3 here
while a UAV job runs. Both containers can mount the same `MODELS_DIR`.

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
