#!/bin/bash
xhost +local:docker

docker run -it \
    --name calib_container \
    --net=host \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/../calib_ws:/home/ros/ros_ws \
    calib_image:latest
