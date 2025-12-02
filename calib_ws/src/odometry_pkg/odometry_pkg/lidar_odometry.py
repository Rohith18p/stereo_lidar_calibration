#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
import numpy as np
from scipy.spatial.transform import Rotation
import sensor_msgs_py.point_cloud2 as pc2

class LidarOdometryNode(Node):
    def __init__(self):
        super().__init__('lidar_odometry_node')
        
        # Declare parameters with default values
        self.declare_parameter('pointcloud_topic', '/ouster/points')
        self.declare_parameter('publish_frequency', 10.0)
        self.declare_parameter('max_iterations', 100)
        self.declare_parameter('tolerance', 1e-5)
        self.declare_parameter('max_correspondence_distance', 2.0)
        self.declare_parameter('voxel_size', 0.3)
        self.declare_parameter('max_range', 50.0)
        self.declare_parameter('min_range', 0.5)
        self.declare_parameter('min_points', 100)
        
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
        
        self.get_logger().info('LiDAR Odometry Node initialized')
        self.get_logger().info(f'Parameters: max_iter={self.max_iterations}, '
                              f'max_corr_dist={self.max_corr_dist}, voxel_size={self.voxel_size}')
    
    def pointcloud_callback(self, msg):
        try:
            # Convert PointCloud2 to numpy array
            current_cloud = self.pointcloud2_to_array(msg)
            current_timestamp = msg.header.stamp
            
            if current_cloud is None or len(current_cloud) < self.min_points:
                self.get_logger().warn(f'Insufficient points in cloud: {len(current_cloud) if current_cloud is not None else 0}')
                return
            
            # Preprocess point cloud
            current_cloud = self.preprocess_cloud(current_cloud)
            
            if len(current_cloud) < self.min_points:
                self.get_logger().warn(f'Insufficient points after preprocessing: {len(current_cloud)}')
                return
            
            if self.prev_cloud is not None:
                # Compute ICP transformation (t to t+1)
                transformation = self.icp(self.prev_cloud, current_cloud)
                
                if transformation is not None:
                    # Publish relative odometry with timestamp t+1
                    self.publish_odometry(transformation, current_timestamp)
                else:
                    self.get_logger().warn('ICP failed to compute transformation')
            
            # Store current as previous
            self.prev_cloud = current_cloud
            self.prev_timestamp = current_timestamp
            
        except Exception as e:
            self.get_logger().error(f'Error in pointcloud callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to Nx3 numpy array"""
        try:
            points = []
            for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                # Filter by range
                dist = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
                if self.min_range < dist < self.max_range:
                    points.append([point[0], point[1], point[2]])
            
            if len(points) == 0:
                return None
                
            return np.array(points, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f'Error converting pointcloud: {str(e)}')
            return None
    
    def preprocess_cloud(self, cloud):
        """Downsample and clean point cloud using voxel grid"""
        if len(cloud) < self.min_points:
            return cloud
        
        # Voxel grid downsampling
        voxel_dict = {}
        
        for point in cloud:
            # Compute voxel index
            voxel_idx = tuple(np.floor(point / self.voxel_size).astype(int))
            
            # Store points in each voxel
            if voxel_idx not in voxel_dict:
                voxel_dict[voxel_idx] = []
            voxel_dict[voxel_idx].append(point)
        
        # Take centroid of each voxel
        downsampled = []
        for voxel_points in voxel_dict.values():
            centroid = np.mean(voxel_points, axis=0)
            downsampled.append(centroid)
        
        downsampled = np.array(downsampled, dtype=np.float32)
        
        self.get_logger().info(f'Downsampled from {len(cloud)} to {len(downsampled)} points')
        
        return downsampled
    
    def icp(self, source, target):
        """Improved Iterative Closest Point algorithm"""
        
        self.get_logger().info(f'Starting ICP with {len(source)} source and {len(target)} target points')
        
        if len(source) < self.min_points or len(target) < self.min_points:
            self.get_logger().warn(f'Insufficient points for ICP: source={len(source)}, target={len(target)}')
            return None
        
        # Initialize transformation
        T = np.eye(4)
        prev_error = float('inf')
        
        # Build KD-Tree for target once
        from scipy.spatial import KDTree
        tree = KDTree(target)
        
        for iteration in range(self.max_iterations):
            # Transform source points
            source_homogeneous = np.hstack([source, np.ones((len(source), 1))])
            source_transformed = (T @ source_homogeneous.T).T[:, :3]
            
            # Find nearest neighbors
            distances, indices = tree.query(source_transformed, k=1, workers=-1)
            
            # Filter correspondences by distance threshold
            valid_mask = distances < self.max_corr_dist
            num_valid = np.sum(valid_mask)
            
            if num_valid < self.min_points:
                self.get_logger().warn(f'Iteration {iteration}: Insufficient valid correspondences: {num_valid}')
                return None
            
            # Get valid correspondences
            source_matched = source[valid_mask]
            target_matched = target[indices[valid_mask]]
            valid_distances = distances[valid_mask]
            
            # Compute mean error
            mean_error = np.mean(valid_distances)
            
            # Check convergence
            error_change = abs(prev_error - mean_error)
            
            if iteration > 0 and error_change < self.tolerance:
                self.get_logger().info(f'ICP converged at iteration {iteration+1} with error {mean_error:.6f}m')
                break
            
            # Compute transformation using matched points
            delta_T = self.compute_transformation(source_matched, target_matched)
            
            if delta_T is None:
                self.get_logger().warn(f'Failed to compute transformation at iteration {iteration}')
                return None
            
            # Update transformation
            T = delta_T @ T
            
            prev_error = mean_error
            
            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                self.get_logger().info(f'Iteration {iteration+1}: error={mean_error:.6f}m, '
                                      f'valid_corr={num_valid}, error_change={error_change:.8f}')
        
        # Final validation
        source_homogeneous = np.hstack([source, np.ones((len(source), 1))])
        source_final = (T @ source_homogeneous.T).T[:, :3]
        final_distances, _ = tree.query(source_final, k=1, workers=-1)
        final_valid = np.sum(final_distances < self.max_corr_dist)
        final_error = np.mean(final_distances[final_distances < self.max_corr_dist])
        
        self.get_logger().info(f'ICP finished: final_error={final_error:.6f}m, '
                              f'final_valid_corr={final_valid}/{len(source)}')
        
        # Check if transformation is reasonable
        translation = np.linalg.norm(T[:3, 3])
        if translation > 5.0:  # More than 5m movement seems unreasonable at 10Hz
            self.get_logger().warn(f'Unreasonable translation: {translation:.2f}m')
            return None
        
        return T
    
    def compute_transformation(self, source, target):
        """Compute transformation from source to target using SVD"""
        try:
            if len(source) < 3 or len(target) < 3:
                return None
            
            # Center the points
            centroid_source = np.mean(source, axis=0)
            centroid_target = np.mean(target, axis=0)
            
            source_centered = source - centroid_source
            target_centered = target - centroid_target
            
            # Compute cross-covariance matrix
            H = source_centered.T @ target_centered
            
            # SVD
            U, S, Vt = np.linalg.svd(H)
            
            # Compute rotation
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = centroid_target - R @ centroid_source
            
            # Build transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            
            return T
            
        except Exception as e:
            self.get_logger().error(f'Error in compute_transformation: {str(e)}')
            return None
    
    def publish_odometry(self, transformation, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Extract position and orientation from transformation (t to t+1)
        position = transformation[:3, 3]
        rotation_matrix = transformation[:3, :3]
        
        # Convert to quaternion
        rot = Rotation.from_matrix(rotation_matrix)
        quat = rot.as_quat()
        
        # Set pose (relative transformation)
        odom.pose.pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        odom.pose.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), 
                                               z=float(quat[2]), w=float(quat[3]))
        
        # Set covariance (can be tuned based on error metrics)
        odom.pose.covariance = [0.1] * 36
        odom.twist.covariance = [0.1] * 36
        
        self.odom_pub.publish(odom)
        
        # Compute Euler angles for logging
        roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
        translation_norm = np.linalg.norm(position)
        
        self.get_logger().info(f'Published lidar odometry: '
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