# Testing with a recorded bag

A replay runs the real node on recorded topics. Use another ROS domain so it never mixes with a robot
on the network, and keep the map upload off.

```bash
# .env for a replay
ROS_DOMAIN_ID=77
API_ENABLED=false
UGV_CONFIG=configs/replay_pui.yaml
BAG_PATH=/path/to/paired_bag          # folder with metadata.yaml + .mcap/.db3

docker compose up -d ugv                                   # 1. the node (wait for "Ready:" in the logs)
docker compose run --rm -d --name heading ugv \
  python scripts/replay_heading_from_yaw.py                # 2. heading (the bags have no magnetometer)
docker compose --profile replay run --rm replay            # 3. play the bag
docker compose logs ugv | tail; cat outputs/ugv_detections.geojson   # 4. results
docker compose down; docker rm -f heading
```

## What `configs/replay_pui.yaml` changes

* camera topics without the `_broadcasted` suffix;
* the 1280x720 calibration of the recorded RealSense stream (the defaults are the live 640x480 stream);
* position from `/fix` and `ugv_status` only. The live position API would place recorded detections at
  the robot's *current* location.

## What a PUI replay can and cannot show

Verified with the 69 s PUI bag: topic wiring, RGB/depth pairing, detection, depth localisation,
geolocation, the object map, GeoJSON/JSONL/frames, and timing. YOLOE-v8-L took a median 200 ms per
frame in the node on an RTX 3060 (78 ms isolated at 1 Hz; 48 ms back to back). The rest is the bag
player sharing the CPU and the GPU clocking down between frames.

It cannot measure geo-tracking quality. These recordings never had an RTK fix (`gnss_valid` is false),
their `/fix` is plain GNSS at 1 Hz, and the only heading available is odometry yaw with an unknown
north. In the PUI bag a static cone's computed position drifts by ~17 m while the robot approaches it,
so one physical object becomes several map objects. On the robot, RTK `ugv_status` plus the
magnetometer avoid this. To evaluate tracking, record a bag with `ugv_status` (RTK) and the
`heading_broadcasted` topic, then compare `tracker.backend: sift` and `masa` on it
(see [ARCHITECTURE.md](ARCHITECTURE.md#comparing-trackers)).
