# Stereo Camera and LiDAR Extrinsic Calibration on the Fly

This project provides a setup for performing **online (on-the-fly) extrinsic calibration** between a stereo camera and a LiDAR sensor using ROS 2 and Docker.

---

## Folder Structure

```text
.
├── calid_dock/        # Docker setup for ROS 2 environment
├── calib_ws/          # ROS 2 workspace
│   └── src/
│       └── odometry_pkg/   # Package handling odometry / calibration-related nodes
└── Makefile           # Convenience commands for building and running Docker containers


