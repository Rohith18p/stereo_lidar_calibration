import argparse
import cv2
import numpy as np
import open3d as o3d
from .config import Config
from .dataloader import KittiLoader
from .stereo_depth import StereoProcessor
from .lidar_process import LidarProcessor
from .optimizer import CalibrationOptimizer
from .semantic_segmentation import ImageSegmenter

def visualize_result(img, lidar_points, rvec, tvec, config):
    """
    Visualize lidar points projected onto the image.
    """
    processor = LidarProcessor(config)
    uv, valid_mask, _ = processor.project_to_image(lidar_points, rvec, tvec)
    
    img_vis = img.copy()
    for p in uv:
        cv2.circle(img_vis, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
        
    return img_vis

def process_frame(index, img_left_dir, img_right_dir, lidar_dir, config=None, use_semantic=False):
    """
    Performs projection, matching (optimization), calculates extrinsics, 
    overlays points, and prints extrinsic values for a specific frame index.
    """
    if config is None:
        config = Config()

    print(f"Processing Frame {index} (Semantic: {use_semantic})...")

    # 1. Load Data
    print("Loading data...")
    loader = KittiLoader(img_left_dir, img_right_dir, lidar_dir)
    try:
        data = loader.get_item(index)
    except IndexError:
        print(f"Error: Frame index {index} out of bounds.")
        return
        
    img_left = data['img_left']
    img_right = data['img_right']
    lidar_points = data['lidar_points']
    
    lidar_proc = LidarProcessor(config)
    optimizer = CalibrationOptimizer(config)
    
    if use_semantic:
        # --- Semantic Pipeline ---
        print("Running Semantic Segmentation...")
        segmenter = ImageSegmenter()
        car_mask = segmenter.get_car_mask(img_left)
        
        # Save mask for debug
        cv2.imwrite(f"debug_mask_frame_{index}.png", car_mask * 255)
        
        print("Processing Lidar (Clustering)...")
        lidar_clusters = lidar_proc.get_potential_cars(lidar_points)
        print(f"Found {len(lidar_clusters)} potential car clusters.")
        
        if len(lidar_clusters) == 0:
            print("No lidar clusters found. Skipping optimization.")
            return

        print("Optimizing extrinsics (Semantic)...")
        opt_rvec, opt_tvec = optimizer.optimize_semantic(
            lidar_clusters, 
            car_mask, 
            config.init_rvec, 
            config.init_tvec
        )
        
        # Visualize Semantic Result
        print("Visualizing Semantic Result...")
        # Project all clusters
        all_cluster_points = np.vstack(lidar_clusters)
        img_vis = visualize_result(img_left, all_cluster_points, opt_rvec, opt_tvec, config)
        
        # Overlay on mask (optional, maybe just save separate image)
        mask_vis = cv2.cvtColor(car_mask * 255, cv2.COLOR_GRAY2BGR)
        img_vis_mask = visualize_result(mask_vis, all_cluster_points, opt_rvec, opt_tvec, config)
        
        cv2.imwrite(f"semantic_result_frame_{index}.png", img_vis)
        cv2.imwrite(f"semantic_mask_overlay_frame_{index}.png", img_vis_mask)
        print(f"Saved semantic_result_frame_{index}.png and semantic_mask_overlay_frame_{index}.png")
        
    else:
        # --- Original Stereo Pipeline ---
        # 2. Compute Stereo Depth
        print("Computing stereo depth...")
        stereo_proc = StereoProcessor(config)
        disparity = stereo_proc.compute_disparity(img_left, img_right)
        depth = stereo_proc.disparity_to_depth(disparity)
        stereo_points, stereo_colors = stereo_proc.depth_to_pointcloud(depth, img_left)
        
        # 3. Process Lidar
        print("Processing lidar...")
        lidar_points_filtered = lidar_proc.filter_points(lidar_points)
        
        # 4. Optimize
        print("Optimizing extrinsics...")
        
        # Downsample for speed if needed
        if len(lidar_points_filtered) > 5000:
            indices = np.random.choice(len(lidar_points_filtered), 5000, replace=False)
            lidar_points_opt = lidar_points_filtered[indices]
        else:
            lidar_points_opt = lidar_points_filtered
            
        if len(stereo_points) > 10000:
            indices = np.random.choice(len(stereo_points), 10000, replace=False)
            stereo_points_opt = stereo_points[indices]
        else:
            stereo_points_opt = stereo_points
            
        opt_rvec, opt_tvec = optimizer.optimize(
            lidar_points_opt, 
            stereo_points_opt, 
            config.init_rvec, 
            config.init_tvec
        )
        
        # 5. Visualize
        print("Visualizing...")
        # Initial Projection
        img_init = visualize_result(img_left, lidar_points_filtered, config.init_rvec, config.init_tvec, config)
        cv2.imwrite(f"projection_initial_frame_{index}.png", img_init)
        
        # Optimized Projection
        img_opt = visualize_result(img_left, lidar_points_filtered, opt_rvec, opt_tvec, config)
        cv2.imwrite(f"projection_optimized_frame_{index}.png", img_opt)
        
        print(f"Saved projection_initial_frame_{index}.png and projection_optimized_frame_{index}.png")
    
    print("\n" + "="*30)
    print(f"Frame {index} Results:")
    print("Optimized Extrinsics:")
    print(f"Rotation Vector (rvec): {opt_rvec}")
    print(f"Translation Vector (tvec): {opt_tvec}")
    print("="*30 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Bidirectional Lidar-Camera Calibration")
    parser.add_argument("--img_left_dir", default="/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data", help="Path to left image directory")
    parser.add_argument("--img_right_dir", default="/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_03/data", help="Path to right image directory")
    parser.add_argument("--lidar_dir", default="/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data", help="Path to lidar directory")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use")
    parser.add_argument("--semantic", action="store_true", help="Use semantic calibration (SST-Calib style)")
    args = parser.parse_args()

    config = Config()
    process_frame(args.frame, args.img_left_dir, args.img_right_dir, args.lidar_dir, config, args.semantic)

if __name__ == "__main__":
    main()
