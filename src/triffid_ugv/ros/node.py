"""ROS 2 node: gathers one synchronized RGB-D frame + pose per period and runs the pipeline.

Sensor callbacks only store the newest messages (cheap); a timer at `pipeline.rate_hz` takes the
newest RGB image, pairs it with the depth image closest in time, resolves intrinsics/extrinsics and
the robot pose, and processes it. Inference therefore never queues up behind the camera rate.
"""
from __future__ import annotations

from collections import deque
import logging
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, NavSatFix
from std_msgs.msg import Float32

from ..config import Config
from ..geometry import intrinsics_for_size, intrinsics_from_camera_info, rigid, rotation_from_quaternion, static_optical_to_body
from ..pipeline import Frame
from ..position import PoseEstimator
from .images import cdr_stamp_ns, depth_to_metres, image_to_bgr

log = logging.getLogger('triffid_ugv')

try:
    from ugv_status_interfaces.msg import UnitreeStatus
except ImportError:      # the Docker image builds it; a bare ROS install may not have it
    UnitreeStatus = None


class UgvPerceptionNode(Node):
    def __init__(self, config: Config, pipeline):
        super().__init__('triffid_ugv_perception')
        self.config, self.pipeline = config, pipeline
        t = config.topics
        self.pose = PoseEstimator(config.position)
        self.pose.start_api()
        self.static_extrinsics = static_optical_to_body(config.camera.static_translation_m,
                                                        config.camera.static_rpy_deg)
        self.camera_info = None
        self.rgb = deque(maxlen=30)           # (stamp_ns, serialized Image, received monotonic)
        self.depth = deque(maxlen=30)         # (stamp_ns, serialized Image)
        self.thermal = deque(maxlen=30)       # (stamp_ns, serialized Image), only for debug recording
        self.last_stamp = None
        self.counters = dict(processed=0, no_depth=0, stale=0, failed=0, detections=0, geolocated=0, no_pose=0)
        self._warned = {}
        self._extrinsics_source = None

        self.tf_buffer = None
        if config.camera.extrinsics in ('auto', 'tf'):
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        sensors, processing = ReentrantCallbackGroup(), MutuallyExclusiveCallbackGroup()
        def qos(reliability, depth=5):
            return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth, reliability=reliability,
                              durability=DurabilityPolicy.VOLATILE)
        # Camera images: reliable by default (best-effort drops fragments of large messages). Small
        # topics: best-effort, which receives from both reliable and best-effort publishers.
        image_qos = qos(ReliabilityPolicy.RELIABLE if t.image_qos == 'reliable' else ReliabilityPolicy.BEST_EFFORT)
        small_qos = qos(ReliabilityPolicy.BEST_EFFORT, depth=10)
        sub = lambda kind, topic, callback, profile=small_qos, raw=False: self.create_subscription(
            kind, topic, callback, profile, callback_group=sensors, raw=raw)
        sub(Image, t.rgb, self._on_rgb, image_qos, raw=True)
        sub(Image, t.depth, self._on_depth, image_qos, raw=True)
        info_topic = t.camera_info or t.rgb.rsplit('/', 1)[0] + '/camera_info'
        if config.camera.intrinsics != 'config':
            sub(CameraInfo, info_topic, self._on_camera_info)
        if t.status:
            if UnitreeStatus is None:
                self.get_logger().warn('ugv_status_interfaces not importable; status position/heading disabled')
            else:
                sub(UnitreeStatus, t.status, self._on_status)
        if t.fix:
            sub(NavSatFix, t.fix, self._on_fix)
        if t.heading:
            sub(Float32, t.heading, lambda msg: self.pose.on_heading_deg(msg.data))
        debug = pipeline.debug
        if debug is not None and debug.wants_thermal and t.thermal:
            sub(Image, t.thermal, self._on_thermal, image_qos, raw=True)

        self.create_timer(1.0 / config.pipeline.rate_hz, self._tick, callback_group=processing)
        self.create_timer(config.outputs.flush_period_s, self.pipeline.write_snapshot, callback_group=processing)
        if pipeline.api is not None:
            self.create_timer(config.api.flush_period_s, self.pipeline.upload, callback_group=processing)
        self.create_timer(30.0, self._heartbeat, callback_group=processing)
        self.get_logger().info(
            f'Ready: {config.detector.backend} + {config.tracker.backend} tracker at {config.pipeline.rate_hz:g} Hz | '
            f'rgb={t.rgb} depth={t.depth} heading={t.heading or "-"} status={t.status or "-"} fix={t.fix or "-"} | '
            f'outputs={pipeline.writer.dir if pipeline.writer else "-"} api={"on" if pipeline.api else "off"}'
            + (f' | DEBUG recording {", ".join(debug.streams)} -> {debug.dir}' if debug is not None else ''))

    # ---- sensor callbacks (store only) --------------------------------------------------------
    # Camera topics arrive serialized (raw=True): deserializing 17 Hz of multi-MB images in Python
    # would hold the GIL and slow inference ~5x. Only the frame chosen each period is deserialized.
    def _on_rgb(self, raw: bytes):
        self.rgb.append((cdr_stamp_ns(raw), raw, time.monotonic()))

    def _on_depth(self, raw: bytes):
        self.depth.append((cdr_stamp_ns(raw), raw))

    def _on_camera_info(self, msg):
        self.camera_info = msg

    def _on_status(self, msg):
        lat, lon, valid = msg.latitude, msg.longitude, bool(msg.gnss_valid)
        if (not valid or not lat) and getattr(msg, 'has_global_position', False):
            g = msg.global_position
            lat, lon, valid = g.latitude, g.longitude, bool(g.gnss_valid)
        self.pose.on_status(lat, lon, valid, heading_rad=msg.heading)

    def _on_fix(self, msg):
        self.pose.on_fix(msg.latitude, msg.longitude, msg.altitude, status_ok=msg.status.status >= 0)

    def _on_thermal(self, raw: bytes):
        self.thermal.append((cdr_stamp_ns(raw), raw))

    # ---- helpers ------------------------------------------------------------------------------
    def _warn(self, key, text, period_s=10.0):
        now = time.monotonic()
        if now - self._warned.get(key, -1e9) >= period_s:
            self._warned[key] = now
            self.get_logger().warn(text)

    def _intrinsics(self, width, height):
        mode, info = self.config.camera.intrinsics, self.camera_info
        if mode != 'config' and info is not None:
            try:
                intrinsics = intrinsics_from_camera_info(info.k, info.width, info.height)
                # A CameraInfo for another sensor mode (e.g. the robot driver's 1280x720 while the broadcast
                # is 640x480) cannot be rescaled; in auto mode fall back to camera.calibrations instead.
                intrinsics.scaled_to(width, height)
                return intrinsics
            except ValueError as error:
                self._warn('camera_info', f'Ignoring CameraInfo for {width}x{height} images: {error}', 30.0)
        if mode == 'camera_info':
            return None
        try:
            return intrinsics_for_size(self.config.camera.calibrations, width, height)
        except ValueError as error:
            self._warn('intrinsics', str(error), 30.0)
            return None

    def _thermal_for(self, stamp: int):
        """BGR thermal frame closest to `stamp` (within debug.thermal_max_dt_s), or None."""
        best = min(self.thermal, key=lambda m: abs(m[0] - stamp), default=None)
        if best is None:
            self._warn('thermal', f'Debug: no thermal frames on {self.config.topics.thermal}', 30.0)
            return None
        if abs(best[0] - stamp) > self.config.debug.thermal_max_dt_s * 1e9:
            self._warn('thermal', f'Debug: thermal is {abs(best[0] - stamp) / 1e9:.1f}s away from RGB; not paired', 30.0)
            return None
        try:
            return image_to_bgr(deserialize_message(best[1], Image))
        except Exception as error:
            self._warn('thermal_decode', f'Debug: thermal frame not decoded: {error}')
            return None

    def _extrinsics(self, frame_id):
        mode = self.config.camera.extrinsics
        if self.tf_buffer is not None and frame_id:
            try:
                tf = self.tf_buffer.lookup_transform(self.config.camera.body_frame, frame_id, Time())
                q, p = tf.transform.rotation, tf.transform.translation
                source = 'tf'
                matrix = rigid(rotation_from_quaternion(q.x, q.y, q.z, q.w), [p.x, p.y, p.z])
                if source != self._extrinsics_source:
                    self.get_logger().info(f'Camera extrinsics from TF: {self.config.camera.body_frame} <- {frame_id}')
                    self._extrinsics_source = source
                return matrix
            except Exception as error:
                if mode == 'tf':
                    self._warn('tf', f'No TF {self.config.camera.body_frame} <- {frame_id}: {error}')
                    return None
        if self._extrinsics_source != 'static':
            self.get_logger().info('Camera extrinsics: static values from camera.static_*')
            self._extrinsics_source = 'static'
        return self.static_extrinsics

    # ---- processing ---------------------------------------------------------------------------
    def _tick(self):
        if not self.rgb:
            self._warn('rgb', f'Waiting for RGB on {self.config.topics.rgb}')
            return
        if time.monotonic() - self.rgb[-1][2] > self.config.pipeline.max_frame_age_s:
            self.counters['stale'] += 1
            self._warn('stale', 'Newest RGB frame is stale (camera stopped?)')
            return
        # Newest RGB frame that already has its depth partner (depth images are larger and often
        # arrive after the RGB image of the same instant).
        tolerance, chosen = self.config.pipeline.max_rgbd_dt_s * 1e9, None
        for stamp, raw_rgb, received in reversed(list(self.rgb)):
            if self.last_stamp is not None and stamp <= self.last_stamp:
                break
            if time.monotonic() - received > self.config.pipeline.max_frame_age_s:
                break
            best = min(self.depth, key=lambda d: abs(d[0] - stamp), default=None)
            if best is not None and abs(best[0] - stamp) <= tolerance:
                chosen = stamp, raw_rgb, best
                break
        if chosen is None:
            if self.last_stamp is None or self.rgb[-1][0] > self.last_stamp:
                self.counters['no_depth'] += 1
                self._warn('depth', f'No depth within {self.config.pipeline.max_rgbd_dt_s}s of recent RGB frames '
                                    f'on {self.config.topics.depth}')
            return
        stamp, raw_rgb, best = chosen
        self.last_stamp = stamp
        try:
            msg = deserialize_message(raw_rgb, Image)
            bgr = image_to_bgr(msg)
            depth = depth_to_metres(deserialize_message(best[1], Image), self.config.camera.depth_scale)
            intrinsics = self._intrinsics(bgr.shape[1], bgr.shape[0])
            extrinsics = self._extrinsics(msg.header.frame_id)
            if intrinsics is None or extrinsics is None:
                self._warn('calibration', 'Waiting for camera intrinsics/extrinsics')
                return
            pose, reason = self.pose.pose()
            if pose is None:
                self.counters['no_pose'] += 1
                self._warn('pose', f'Not geolocating: {reason}')
            debug, thermal = self.pipeline.debug, None
            if debug is not None and debug.wants_thermal and debug.due(stamp, True):
                thermal = self._thermal_for(stamp)      # decoded only when a debug set will be written
            report = self.pipeline.process(Frame(stamp, bgr, depth, intrinsics, extrinsics, pose, reason, thermal))
        except Exception as error:
            self.counters['failed'] += 1
            self.get_logger().error(f'Frame failed: {error!r}')
            return
        self.counters['processed'] += 1
        self.counters['detections'] += report.detections
        self.counters['geolocated'] += report.geolocated
        if report.detections:
            self.get_logger().info(f'{report.detections} detections, {report.localized} with depth, '
                                   f'{report.geolocated} geolocated ({report.seconds * 1000:.0f} ms)')

    def _heartbeat(self):
        self.get_logger().info(f'{self.counters} | map {self.pipeline.map.stats()} | sources {self.pose.status()}'
                               + (f' | api {self.pipeline.api.counters}' if self.pipeline.api else '')
                               + (f' | debug {self.pipeline.debug.status()}' if self.pipeline.debug else ''))

    def close(self):
        self.pose.close()
        self.pipeline.close()


def spin(config: Config, pipeline) -> None:
    rclpy.init()
    node = UgvPerceptionNode(config, pipeline)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info('Shutting down: writing final snapshot')
        node.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
