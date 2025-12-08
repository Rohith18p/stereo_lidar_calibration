#!/usr/bin/env python3
"""
Batch calibration script for KITTI dataset.
Processes multiple frames and saves calibration results to CSV.
"""

import numpy as np
import open3d as o3d
import csv
import os
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import copy
import importlib

# Import from calibrate_gicp
import img_pcd
import read_pcd

# Reload modules to ensure we get the latest code (important in Docker/development)
importlib.reload(img_pcd)
importlib.reload(read_pcd)

# --- Default Paths (Edit these for your dataset) ---
DEFAULT_LEFT_DIR = "/home/ros/ros_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data"
DEFAULT_RIGHT_DIR = "/home/ros/ros_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_03/data"
DEFAULT_LIDAR_DIR = "/home/ros/ros_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data"
DEFAULT_OUTPUT_CSV = "/home/ros/ros_ws/src/archive/rs_gicp_calibration_results_v1.csv"
DEFAULT_MAX_FRAMES = 25
# ---------------------------------------------------

# --- Configuration ---
VOXEL_SIZE = 0.5
MAX_CORRESPONDENCE_DISTANCE = 1.0
MAX_ITERATIONS = 100
TOLERANCE = 1e-6
MAX_DIST = 50.0
FOV_DEG = 60.0
USE_KNOWN_TRANSLATION = False
# ---------------------

def preprocess_pcd(pcd, voxel_size):
    """Downsamples the point cloud and estimates normals."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down

def filter_stereo_xy(pcd, initial_transform):
    """Filters stereo point cloud by XY bounds in LiDAR frame."""
    points = np.asarray(pcd.points)
    points_hom = np.hstack([points, np.ones((len(points), 1))])
    points_lidar = (initial_transform @ points_hom.T).T[:, :3]
    mask = (np.abs(points_lidar[:, 0]) <= 50.0) & (np.abs(points_lidar[:, 1]) <= 50.0)
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    return pcd_filtered

def get_initial_guess():
    """Returns the initial transformation matrix from Camera to LiDAR."""
    # Base rotation from camera to lidar frame
    T_base = np.eye(4)
    T_base[:3, :3] = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    
    # Apply rotations in LiDAR frame (180° around X and Z)
    r_x = R.from_euler('x', 180, degrees=True)
    r_z = R.from_euler('z', 180, degrees=True)
    r_combined = r_z * r_x
    
    T_rot = np.eye(4)
    T_rot[:3, :3] = r_combined.as_matrix()
    T_final = T_rot @ T_base
    
    if USE_KNOWN_TRANSLATION:
        known_translation = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
        T_final[:3, 3] = -known_translation
    
    return T_final

def execute_gicp(source, target):
    """Aligns source to target using GICP."""
    trans_init = get_initial_guess()
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=TOLERANCE,
        relative_rmse=TOLERANCE,
        max_iteration=MAX_ITERATIONS
    )
    
    reg_p2p = o3d.pipelines.registration.registration_generalized_icp(
        source, target, MAX_CORRESPONDENCE_DISTANCE, trans_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria
    )
    
    return reg_p2p.transformation

def transform_to_euler_and_translation(transform):
    """
    Converts transformation matrix to translation and Euler angles.
    Returns: (x, y, z, yaw, pitch, roll) in meters and degrees
    """
    # Extract translation
    translation = transform[:3, 3]
    
    # Extract rotation and convert to Euler angles (ZYX convention)
    rotation_matrix = transform[:3, :3].copy()
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('zyx', degrees=True)  # yaw, pitch, roll
    
    return translation[0], translation[1], translation[2], euler_angles[0], euler_angles[1], euler_angles[2]

def process_frame(left_img_path, right_img_path, lidar_bin_path, timestamp):
    """
    Process a single frame and return calibration results.
    Returns: (timestamp, x, y, z, yaw, pitch, roll) or None if failed
    """
    try:
        print(f"\nProcessing frame: {timestamp}")
        
        # Load point clouds
        pcd_stereo = img_pcd.compute_point_cloud(left_img_path, right_img_path, max_dist=MAX_DIST)
        pcd_lidar = read_pcd.read_bin(lidar_bin_path, max_dist=MAX_DIST, fov_deg=FOV_DEG)
        
        if pcd_stereo is None or pcd_lidar is None:
            print(f"  Failed to load point clouds for {timestamp}")
            return None
        
        # Get initial transform and filter stereo
        initial_transform = get_initial_guess()
        pcd_stereo = filter_stereo_xy(pcd_stereo, initial_transform)
        
        # Downsample
        pcd_stereo_down = preprocess_pcd(pcd_stereo, VOXEL_SIZE)
        pcd_lidar_down = preprocess_pcd(pcd_lidar, VOXEL_SIZE)
        
        # Run GICP
        final_transform = execute_gicp(pcd_stereo_down, pcd_lidar_down)
        
        # Convert to Euler angles and translation
        x, y, z, yaw, pitch, roll = transform_to_euler_and_translation(final_transform)
        
        print(f"  Success: x={x:.4f}, y={y:.4f}, z={z:.4f}, yaw={yaw:.2f}°, pitch={pitch:.2f}°, roll={roll:.2f}°")
        
        return (timestamp, x, y, z, yaw, pitch, roll)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Error processing {timestamp}: {e}")
        return None

def batch_calibrate(left_dir, right_dir, lidar_dir, output_csv, max_frames=100):
    """
    Batch process KITTI dataset frames.
    
    Args:
        left_dir: Directory containing left images
        right_dir: Directory containing right images
        lidar_dir: Directory containing LiDAR .bin files
        output_csv: Output CSV file path
        max_frames: Maximum number of frames to process
    """
    # Get sorted list of files
    left_files = sorted(Path(left_dir).glob('*.png'))[:max_frames]
    
    if len(left_files) == 0:
        print(f"No image files found in {left_dir}")
        return
    
    print(f"Found {len(left_files)} frames to process")
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'pos_x', 'pos_y', 'pos_z', 'yaw', 'pitch', 'roll'])
        
        successful = 0
        failed = 0
        
        for left_file in left_files:
            # Extract timestamp from filename (e.g., 000000.png -> 000000)
            timestamp = left_file.stem
            
            # Construct corresponding file paths
            right_file = Path(right_dir) / left_file.name
            lidar_file = Path(lidar_dir) / f"{timestamp}.bin"
            
            # Check if all files exist
            if not right_file.exists():
                print(f"Missing right image: {right_file}")
                failed += 1
                continue
            
            if not lidar_file.exists():
                print(f"Missing LiDAR file: {lidar_file}")
                failed += 1
                continue
            
            # Process frame
            result = process_frame(str(left_file), str(right_file), str(lidar_file), timestamp)
            
            if result is not None:
                writer.writerow(result)
                csvfile.flush()  # Ensure data is written immediately
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"Successful: {successful}/{len(left_files)}")
        print(f"Failed: {failed}/{len(left_files)}")
        print(f"Results saved to: {output_csv}")
        print(f"{'='*60}")

def main():
    # Use command line arguments if provided, otherwise use defaults
    if len(sys.argv) >= 5:
        left_dir = sys.argv[1]
        right_dir = sys.argv[2]
        lidar_dir = sys.argv[3]
        output_csv = sys.argv[4]
        max_frames = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_MAX_FRAMES
    else:
        print("No arguments provided, using default paths from configuration...")
        print("Edit the DEFAULT_* variables at the top of this file to set your paths.\n")
        left_dir = DEFAULT_LEFT_DIR
        right_dir = DEFAULT_RIGHT_DIR
        lidar_dir = DEFAULT_LIDAR_DIR
        output_csv = DEFAULT_OUTPUT_CSV
        max_frames = DEFAULT_MAX_FRAMES
    
    print(f"Configuration:")
    print(f"  Left images:  {left_dir}")
    print(f"  Right images: {right_dir}")
    print(f"  LiDAR data:   {lidar_dir}")
    print(f"  Output CSV:   {output_csv}")
    print(f"  Max frames:   {max_frames}")
    print()
    
    batch_calibrate(left_dir, right_dir, lidar_dir, output_csv, max_frames)

if __name__ == "__main__":
    main()
