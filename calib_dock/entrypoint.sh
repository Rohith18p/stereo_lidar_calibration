#!/bin/bash
set -e

source /opt/ros/$ROS_DISTRO/setup.bash

# Build workspace
cd $ROS_WS
# rosdep update
# rosdep install --from-paths src --ignore-src -r -y
# colcon build 
source install/setup.bash

exec "$@"