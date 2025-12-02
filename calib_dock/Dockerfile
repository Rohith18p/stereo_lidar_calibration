FROM ros:humble

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV ROS_WS=/home/ros/ros_ws 

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release sudo locales \
    build-essential git \
    python3-pip python3-colcon-common-extensions \
    libasio-dev \
    ros-humble-desktop-full \
    ros-humble-moveit \
    && rm -rf /var/lib/apt/lists/*

# Add OSRF GPG key and source for packages
RUN curl -sSL http://packages.osrfoundation.org/gazebo.key | gpg --dearmor > /usr/share/keyrings/gazebo-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/gazebo-stable.list

# Update and install Gazebo
RUN apt-get update && apt-get install -y \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-image \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p $ROS_WS/src
WORKDIR $ROS_WS

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Source ROS2 on shell startup
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
