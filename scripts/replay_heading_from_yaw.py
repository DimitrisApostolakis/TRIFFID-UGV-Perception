"""Replay helper: publish a compass heading derived from UnitreeStatus.yaw (odometry).

The recorded bags have no magnetometer topic. Odometry yaw is relative to the start pose, so the
absolute direction is unknown (`--initial-heading-deg`, default 0 = north), but turns are correct,
which is what tracking consistency depends on. For bags only; never run this on the live robot.

    python scripts/replay_heading_from_yaw.py --initial-heading-deg 0
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32
from ugv_status_interfaces.msg import UnitreeStatus


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--status', default='/b2/ugv_status')
    parser.add_argument('--heading', default='/b2/nicla/magnetometer/heading_broadcasted')
    parser.add_argument('--initial-heading-deg', type=float, default=0.0)
    args = parser.parse_args()
    rclpy.init()
    node = Node('replay_heading_from_yaw')
    publisher = node.create_publisher(Float32, args.heading, 10)
    start = {}

    def on_status(msg):
        start.setdefault('yaw', msg.yaw)
        # ENU yaw grows counter-clockwise; compass heading grows clockwise.
        heading = (args.initial_heading_deg - math.degrees(msg.yaw - start['yaw'])) % 360.0
        publisher.publish(Float32(data=float(heading)))

    node.create_subscription(UnitreeStatus, args.status, on_status, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
