#!/bin/bash
# Source ROS 2 and the bundled robot message packages, then run the CLI (or any command).
set -e
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source /opt/triffid_ws/install/setup.bash
export HOME="${HOME:-/tmp}"
case "${1:-}" in
  run|check|replay-info|version) exec python -m triffid_ugv "$@" ;;
  *) exec "$@" ;;
esac
