# Ground-station deployment with topics_broadcaster_module_v2

On the robot, the broadcaster's receiver bundles the front RGB, aligned depth, FLIR and magnetometer
heading, then sends them over TCP to the ground station. On the ground station its transmitter
republishes them as `*_broadcasted` topics. This node runs on the ground station and subscribes to
those topics. The defaults in `configs/default.yaml` already use these names.

| Broadcast topic (ground station) | Content as received | Used for |
|---|---|---|
| `/b2/camera_front_435i/realsense_front_435i/color/image_raw_broadcasted` | `rgb8`, 640x480 on the live system, JPEG q80 round-trip | detection |
| `/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw_broadcasted` | `16UC1` mm, normalised by frame max + JPEG (≈60 mm median error) | localisation |
| `/b2/nicla/magnetometer/heading_broadcasted` | `std_msgs/Float32`, compass degrees | heading |
| `/b2/camera_flir/image_raw_broadcasted` | `rgb8` | debug recording only (`debug.thermal`) |

Not broadcast, and how the node copes:

* **CameraInfo**: intrinsics come from `camera.calibrations` by image size. The live stream is 640x480
  (the same intrinsics as the earlier live node); 1280x720 is the recorded bags' calibration. A CameraInfo
  with another aspect ratio than the images (e.g. the robot driver's 1280x720) is ignored in `auto` mode.
* **TF**: the camera pose comes from `camera.static_*`, measured from the robot's TF tree.
* **Position**: `/b2/ugv_status` if it reaches the ground station, otherwise the crispres position API
  (both in `position.sources`). The API reports the robot's position *now*, while images arrive after
  the transfer, so a moving robot adds about speed × link delay of error. If the broadcaster also sends
  `/fix` (present but commented out in its code), set `topics.fix: /fix_broadcasted`.

Rate: the broadcaster sends at most 2 bundles/s, and fewer on a slow link, because the TCP transfer
waits for an acknowledgement after every 4 kB chunk. The node always processes the newest bundle, so
any rate works. RGB and depth of a bundle can be up to 1 s apart (synchroniser slop), which
`pipeline.max_rgbd_dt_s: 1.0` accepts.

Checklist on the ground station:

1. Same `ROS_DOMAIN_ID` and RMW as the transmitter (`.env`).
2. Run the container as the same user as the transmitter (`UGV_UID`/`UGV_GID`), so DDS shared memory
   works between them.
3. `docker compose run --rm ugv ros2 topic hz /b2/camera_front_435i/realsense_front_435i/color/image_raw_broadcasted`
   should show the bundle rate. The transmitter is built for ROS 2 Lyrical and this image uses Jazzy;
   `Image` and `Float32` are unchanged between them, but confirm it once.
4. On the robot, the receiver's RealSense YUYV conversion must be the active colour path. The "dummy
   test" path expects 3-channel images and fails on the camera's `yuv422_yuy2`.
