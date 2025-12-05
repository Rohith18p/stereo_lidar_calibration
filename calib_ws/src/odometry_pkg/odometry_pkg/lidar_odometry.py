#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import math
import csv
import os

class LidarOdometryNode(Node):
    def __init__(self):
        super().__init__('lidar_odom_node')
        
        # Declare parameters with default values
        self.declare_parameter('pointcloud_topic', '/ouster/points')
        self.declare_parameter('publish_frequency', 10.0)
        self.declare_parameter('max_iterations', 30)
        self.declare_parameter('tolerance', 1e-5)
        self.declare_parameter('max_correspondence_distance', 1.0)
        self.declare_parameter('voxel_size', 0.5)
        self.declare_parameter('max_range', 100.0)
        self.declare_parameter('min_range', 1.0)
        self.declare_parameter('min_points', 100)
        self.declare_parameter('lidar_enable_csv', False)
        self.declare_parameter('lidar_csv_path', '/ros_ws/src/data/store_v1/lidar_odom.csv')
        
        # Get parameter values
        pc_topic = self.get_parameter('pointcloud_topic').value
        self.publish_freq = self.get_parameter('publish_frequency').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.tolerance = self.get_parameter('tolerance').value
        self.max_corr_dist = self.get_parameter('max_correspondence_distance').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.max_range = self.get_parameter('max_range').value
        self.min_range = self.get_parameter('min_range').value
        self.min_points = self.get_parameter('min_points').value
        self.lidar_enable_csv = self.get_parameter('lidar_enable_csv').value
        self.lidar_csv_path = self.get_parameter('lidar_csv_path').value

        print(f"max iterations: {self.max_iterations}")
        print(f"lidar_enable_csv: {self.lidar_enable_csv}")
        print(f"lidar_csv_path: {self.lidar_csv_path}")

        # CSV Logging Setup
        if self.lidar_enable_csv:
            self.lidar_csv_file_path = self.lidar_csv_path
            self.lidar_csv_file = open(self.lidar_csv_file_path, 'a', newline='')
            self.lidar_csv_writer = csv.writer(self.lidar_csv_file)
            
            # Write header if file is empty
            if os.path.getsize(self.lidar_csv_file_path) == 0:
                header = ['timestamp', 'frame_id', 'child_frame_id', 
                          'pos_x', 'pos_y', 'pos_z', 
                          'orient_x', 'orient_y', 'orient_z', 'orient_w',
                          'linear_x', 'linear_y', 'linear_z',
                          'angular_x', 'angular_y', 'angular_z']
                self.lidar_csv_writer.writerow(header)
                self.lidar_csv_file.flush()
            
            self.get_logger().info(f'CSV logging enabled: {self.lidar_csv_file_path}')
        
        # Subscriber
        self.pc_sub = self.create_subscription(
            PointCloud2,
            pc_topic,
            self.pointcloud_callback,
            10
        )
        
        # Publisher
        self.odom_pub = self.create_publisher(Odometry, 'lidar_odometry', 10)
        
        # State
        self.prev_cloud = None
        self.prev_timestamp = None
        
        self.get_logger().info('LiDAR Odometry Node initialized (GICP)')
        self.get_logger().info(f'Parameters: max_iter={self.max_iterations}, '
                              f'max_corr_dist={self.max_corr_dist}, voxel_size={self.voxel_size}')
    
    def pointcloud_callback(self, msg):
        try:
            # Convert PointCloud2 to Open3D PointCloud
            current_cloud = self.pointcloud2_to_o3d(msg)
            current_timestamp = msg.header.stamp
            
            if current_cloud is None or len(current_cloud.points) < self.min_points:
                self.get_logger().warn(f'Insufficient points in cloud: {len(current_cloud.points) if current_cloud is not None else 0}')
                return
            
            # Preprocess point cloud (downsample + normals)
            current_cloud = self.preprocess_cloud(current_cloud)
            
            if len(current_cloud.points) < self.min_points:
                self.get_logger().warn(f'Insufficient points after preprocessing: {len(current_cloud.points)}')
                return
            
            if self.prev_cloud is not None:
                # Compute GICP transformation (t to t+1)
                transformation = self.compute_gicp(current_cloud, self.prev_cloud)
                
                if transformation is not None:
                    # Publish relative odometry with timestamp t+1
                    self.publish_odometry(transformation, current_timestamp)
                else:
                    self.get_logger().warn('GICP failed to compute transformation')
            
            # Store current as previous
            self.prev_cloud = current_cloud
            self.prev_timestamp = current_timestamp
            
        except Exception as e:
            self.get_logger().error(f'Error in pointcloud callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def pointcloud2_to_o3d(self, cloud_msg):
        """Convert PointCloud2 message to Open3D PointCloud"""
        try:
            points = []
            for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                # Filter by range
                dist = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
                if self.min_range < dist < self.max_range:
                    points.append([point[0], point[1], point[2]])
            
            if len(points) == 0:
                return None
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
            return pcd
            
        except Exception as e:
            self.get_logger().error(f'Error converting pointcloud: {str(e)}')
            return None
    
    def preprocess_cloud(self, pcd):
        """Downsample and estimate normals for GICP"""
        
        # Voxel downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Estimate normals (required for GICP)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
        )
        
        return pcd_down
    
    def compute_gicp(self, source, target):
        """Compute transformation using Generalized ICP"""
        
        # Initial guess (identity)
        trans_init = np.eye(4)
        
        # GICP parameters
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=self.tolerance,
            relative_rmse=self.tolerance,
            max_iteration=self.max_iterations
        )
        
        try:
            reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
                source, target, self.max_corr_dist, trans_init,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                criteria
            )
            
            # Sanity check on translation
            T = reg_p2p.transformation
            translation = np.linalg.norm(T[:3, 3])
            
            if translation > 5.0: # Unreasonable jump
                 self.get_logger().warn(f'Unreasonable translation: {translation:.2f}m')
                 return None
                 
            self.get_logger().info(f'GICP converged. Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}')
            
            return T
            
        except Exception as e:
             self.get_logger().error(f'Error in GICP: {str(e)}')
             return None

    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
            
        return [x, y, z, w]

    def rotation_matrix_to_euler(self, R):
        """Convert 3x3 rotation matrix to euler angles (xyz) in degrees"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
            
        return np.degrees([x, y, z])

    def publish_odometry(self, transformation, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Extract position and orientation from transformation
        position = transformation[:3, 3]
        rotation_matrix = transformation[:3, :3]
        
        # Convert to quaternion using custom function
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        # Set pose (relative transformation)
        odom.pose.pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        odom.pose.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), 
                                               z=float(quat[2]), w=float(quat[3]))
        
        # Set covariance
        odom.pose.covariance = [0.1] * 36
        odom.twist.covariance = [0.1] * 36
        
        self.odom_pub.publish(odom)

        # Write to CSV
        if self.lidar_enable_csv:
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
                self.lidar_csv_writer.writerow(row)
                self.lidar_csv_file.flush()
            except Exception as e:
                self.get_logger().error(f"Failed to write to CSV: {e}")
        
        # Compute Euler angles for logging
        roll, pitch, yaw = self.rotation_matrix_to_euler(rotation_matrix)
        translation_norm = np.linalg.norm(position)
        
        self.get_logger().info(f'Published GICP odometry: '
                              f'trans=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]m ({translation_norm:.3f}m), '
                              f'rot=[{roll:.1f}, {pitch:.1f}, {yaw:.1f}]deg')

def main(args=None):
    rclpy.init(args=args)
    node = LidarOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()