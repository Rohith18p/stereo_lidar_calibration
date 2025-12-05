#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import message_filters
import csv
import os

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odom_node')
        
        # Declare parameters with default values from calibration
        # Camera topics
        self.declare_parameter('left_image_topic', '/cam_sync/cam0/image_raw/debayered')
        self.declare_parameter('right_image_topic', '/cam_sync/cam1/image_raw/debayered')
        
        # Camera 0 (left) intrinsics
        self.declare_parameter('cam0_fx', 1468.93007186)
        self.declare_parameter('cam0_fy', 1471.81884863)
        self.declare_parameter('cam0_cx', 631.85246618)
        self.declare_parameter('cam0_cy', 525.95317983)
        self.declare_parameter('cam0_distortion', [-0.22513578, 0.17282213, 0.00027845, -0.00043702])
        
        # Camera 1 (right) intrinsics
        self.declare_parameter('cam1_fx', 1470.15663378)
        self.declare_parameter('cam1_fy', 1473.07927852)
        self.declare_parameter('cam1_cx', 643.94572844)
        self.declare_parameter('cam1_cy', 507.7164811)
        self.declare_parameter('cam1_distortion', [-0.23775883, 0.20650801, 0.00021593, -0.00022204])
        
        # Baseline transformation T_1_0 (cam1 relative to cam0)
        self.declare_parameter('baseline_quat', [0.00208869, 0.00136061, -0.0097347, 0.99994951])
        self.declare_parameter('baseline_trans', [-0.22236458, -0.00353404, -0.00112897])
        
        # Algorithm parameters
        self.declare_parameter('publish_frequency', 10.0)
        self.declare_parameter('max_features', 2000)
        self.declare_parameter('min_matches', 50)
        self.declare_parameter('min_3d_points', 20)
        self.declare_parameter('visual_enable_csv', False)
        self.declare_parameter('visual_csv_path', '/ros_ws/src/data/store_v1/visual_odom.csv')
        
        # Get parameter values
        left_topic = self.get_parameter('left_image_topic').value
        right_topic = self.get_parameter('right_image_topic').value
        self.publish_freq = self.get_parameter('publish_frequency').value
        self.max_features = self.get_parameter('max_features').value
        self.min_matches = self.get_parameter('min_matches').value
        self.min_3d_points = self.get_parameter('min_3d_points').value
        self.visual_enable_csv = self.get_parameter('visual_enable_csv').value
        self.visual_csv_path = self.get_parameter('visual_csv_path').value
        
        # CSV Logging Setup
        if self.visual_enable_csv:
            self.csv_file_path = self.visual_csv_path
            self.csv_file = open(self.csv_file_path, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header if file is empty
            if os.path.getsize(self.csv_file_path) == 0:
                header = ['timestamp', 'frame_id', 'child_frame_id', 
                          'pos_x', 'pos_y', 'pos_z', 
                          'orient_x', 'orient_y', 'orient_z', 'orient_w',
                          'linear_x', 'linear_y', 'linear_z',
                          'angular_x', 'angular_y', 'angular_z']
                self.csv_writer.writerow(header)
                self.csv_file.flush()
            
            self.get_logger().info(f'CSV logging enabled: {self.csv_file_path}')
        
        # Camera 0 (left) parameters
        self.cam0_fx = self.get_parameter('cam0_fx').value
        self.cam0_fy = self.get_parameter('cam0_fy').value
        self.cam0_cx = self.get_parameter('cam0_cx').value
        self.cam0_cy = self.get_parameter('cam0_cy').value
        self.cam0_dist = np.array(self.get_parameter('cam0_distortion').value)
        
        self.K0 = np.array([[self.cam0_fx, 0, self.cam0_cx],
                           [0, self.cam0_fy, self.cam0_cy],
                           [0, 0, 1]])
        
        # Camera 1 (right) parameters
        self.cam1_fx = self.get_parameter('cam1_fx').value
        self.cam1_fy = self.get_parameter('cam1_fy').value
        self.cam1_cx = self.get_parameter('cam1_cx').value
        self.cam1_cy = self.get_parameter('cam1_cy').value
        self.cam1_dist = np.array(self.get_parameter('cam1_distortion').value)
        
        self.K1 = np.array([[self.cam1_fx, 0, self.cam1_cx],
                           [0, self.cam1_fy, self.cam1_cy],
                           [0, 0, 1]])
        
        # Baseline transformation
        baseline_quat = self.get_parameter('baseline_quat').value
        baseline_trans = self.get_parameter('baseline_trans').value
        
        # Convert quaternion to rotation matrix (quat is [x, y, z, w])
        self.R_1_0 = Rotation.from_quat(baseline_quat).as_matrix()
        self.t_1_0 = np.array(baseline_trans).reshape(3, 1)
        
        # Baseline magnitude (primarily in x-direction)
        self.baseline = np.linalg.norm(baseline_trans)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers with message filters for synchronization
        self.left_sub = message_filters.Subscriber(self, Image, left_topic)
        self.right_sub = message_filters.Subscriber(self, Image, right_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], 10, 0.1)
        self.ts.registerCallback(self.stereo_callback)
        
        # Publisher
        self.odom_pub = self.create_publisher(Odometry, 'visual_odometry', 10)
        
        # State
        self.prev_left = None
        self.prev_right = None
        self.prev_left_undist = None
        self.prev_right_undist = None
        self.prev_timestamp = None
        
        # Feature detector and matcher
        self.detector = cv2.ORB_create(nfeatures=self.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Stereo matcher
        self.stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
        
        self.get_logger().info('Visual Odometry Node initialized')
        self.get_logger().info(f'Left camera: fx={self.cam0_fx:.2f}, fy={self.cam0_fy:.2f}')
        self.get_logger().info(f'Right camera: fx={self.cam1_fx:.2f}, fy={self.cam1_fy:.2f}')
        self.get_logger().info(f'Baseline: {self.baseline:.4f} m')
        
    def stereo_callback(self, left_msg, right_msg):
        try:
            # Convert images
            left_img = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
            right_img = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
            
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Undistort images
            left_undist = cv2.undistort(left_gray, self.K0, self.cam0_dist)
            right_undist = cv2.undistort(right_gray, self.K1, self.cam1_dist)
            
            current_timestamp = left_msg.header.stamp
            
            # Need previous frame for odometry
            if self.prev_left_undist is not None:
                # Compute odometry between prev and current (t to t+1)
                transformation = self.compute_odometry(
                    self.prev_left_undist, self.prev_right_undist,
                    left_undist, right_undist
                )
                
                if transformation is not None:
                    # Publish relative odometry with timestamp of t+1
                    self.publish_odometry(transformation, current_timestamp)
                else:
                    self.get_logger().warn('Failed to compute transformation')
            
            # Store current as previous
            self.prev_left = left_gray
            self.prev_right = right_gray
            self.prev_left_undist = left_undist
            self.prev_right_undist = right_undist
            self.prev_timestamp = current_timestamp
            
        except Exception as e:
            self.get_logger().error(f'Error in stereo callback: {str(e)}')
    
    def compute_odometry(self, prev_left, prev_right, curr_left, curr_right):
        """Compute transformation from time t to t+1 using stereo VO"""
        # Detect and match features in left images (t and t+1)
        kp_prev, des_prev = self.detector.detectAndCompute(prev_left, None)
        kp_curr, des_curr = self.detector.detectAndCompute(curr_left, None)
        
        if des_prev is None or des_curr is None or len(kp_prev) < 10 or len(kp_curr) < 10:
            self.get_logger().warn('Insufficient features detected')
            return None
        
        # Match features
        matches = self.matcher.match(des_prev, des_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take best matches
        num_matches = min(len(matches), 200)
        matches = matches[:num_matches]
        
        if len(matches) < self.min_matches:
            self.get_logger().warn(f'Insufficient matches: {len(matches)} < {self.min_matches}')
            return None
        
        # Get matched keypoints
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])
        
        # Compute disparity for previous frame to get 3D points
        disparity_prev = self.stereo.compute(prev_left, prev_right)
        
        # Get 3D points from stereo at time t
        points_3d = []
        points_2d_curr = []
        
        for pt_prev, pt_curr in zip(pts_prev, pts_curr):
            x, y = int(pt_prev[0]), int(pt_prev[1])
            
            # Check bounds
            if 0 <= y < disparity_prev.shape[0] and 0 <= x < disparity_prev.shape[1]:
                d = disparity_prev[y, x] / 16.0  # Disparity is in fixed-point
                
                if d > 1.0:  # Valid disparity threshold
                    # Triangulate 3D point using left camera parameters
                    Z = (self.cam0_fx * self.baseline) / d
                    X = (x - self.cam0_cx) * Z / self.cam0_fx
                    Y = (y - self.cam0_cy) * Z / self.cam0_fy
                    
                    # Filter out points too far or too close
                    if 0.1 < Z < 100.0:
                        points_3d.append([X, Y, Z])
                        points_2d_curr.append(pt_curr)
        
        if len(points_3d) < self.min_3d_points:
            self.get_logger().warn(f'Insufficient 3D points: {len(points_3d)} < {self.min_3d_points}')
            return None
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d_curr = np.array(points_2d_curr, dtype=np.float32)
        
        # Solve PnP to get transformation from cam(t) to cam(t+1)
        # This gives us where the camera moved to
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d_curr, self.K0, None,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < self.min_3d_points:
            self.get_logger().warn(f'PnP failed or insufficient inliers: {len(inliers) if inliers is not None else 0}')
            return None
        
        # Convert to transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        self.get_logger().info(f'Computed odometry: {len(inliers)} inliers from {len(points_3d)} points')
        
        return T
    
    def publish_odometry(self, transformation, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Extract position and orientation from transformation (t to t+1)
        position = transformation[:3, 3]
        rotation_matrix = transformation[:3, :3]
        
        # Convert rotation matrix to quaternion
        rot = Rotation.from_matrix(rotation_matrix)
        quat = rot.as_quat()  # [x, y, z, w]
        
        # Set pose
        odom.pose.pose.position = Point(x=position[0], y=position[1], z=position[2])
        odom.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        
        # Set covariance (example values)
        odom.pose.covariance = [0.1] * 36
        odom.twist.covariance = [0.1] * 36
        
        self.odom_pub.publish(odom)

        # Write to CSV
        if self.visual_enable_csv:
            try:
                row = [
                    f"{odom.header.stamp.sec}.{odom.header.stamp.nanosec:09d}",
                    odom.header.frame_id,
                    odom.child_frame_id,
                    odom.pose.pose.position.x,
                    odom.pose.pose.position.y,
                    odom.pose.pose.position.z,
                    odom.pose.pose.orientation.x,
                    odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z,
                    odom.pose.pose.orientation.w,
                    odom.twist.twist.linear.x,
                    odom.twist.twist.linear.y,
                    odom.twist.twist.linear.z,
                    odom.twist.twist.angular.x,
                    odom.twist.twist.angular.y,
                    odom.twist.twist.angular.z
                ]
                self.csv_writer.writerow(row)
                self.csv_file.flush()
            except Exception as e:
                self.get_logger().error(f"Failed to write to CSV: {e}")
        self.get_logger().info(f'Published visual odometry: pos=[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]')

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()