import argparse
import cv2
import numpy as np
import open3d as o3d
from .config import Config
from .dataloader import KittiLoader
from .stereo_depth import StereoProcessor

def main():
    parser = argparse.ArgumentParser(description="Debug Stereo 3D Reconstruction")
    parser.add_argument("--img_left_dir", default="/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data", help="Path to left image directory")
    parser.add_argument("--img_right_dir", default="/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_03/data", help="Path to right image directory")
    parser.add_argument("--lidar_dir", default="/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data", help="Path to lidar directory")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use")
    args = parser.parse_args()

    config = Config()
    
    print(f"Processing Frame {args.frame}...")

    # 1. Load Data
    print("Loading data...")
    loader = KittiLoader(args.img_left_dir, args.img_right_dir, args.lidar_dir)
    try:
        data = loader.get_item(args.frame)
    except IndexError:
        print(f"Error: Frame index {args.frame} out of bounds.")
        return
        
    img_left = data['img_left']
    img_right = data['img_right']
    
    # 2. Compute Stereo Depth
    print("Computing stereo depth...")
    stereo_proc = StereoProcessor(config)
    disparity = stereo_proc.compute_disparity(img_left, img_right)
    
    # Visualize Disparity
    disparity_vis = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    disparity_vis = (disparity_vis * 255).astype(np.uint8)
    disparity_vis = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
    cv2.imwrite(f"debug_disparity_frame_{args.frame}.png", disparity_vis)
    print(f"Saved debug_disparity_frame_{args.frame}.png")
    
    depth = stereo_proc.disparity_to_depth(disparity)
    stereo_points, stereo_colors = stereo_proc.depth_to_pointcloud(depth, img_left)
    
    print(f"Generated {len(stereo_points)} points.")
    
    # 3. Visualize Point Cloud
    print("Visualizing Point Cloud (Open3D)...")
    pcd = stereo_proc.create_o3d_cloud(stereo_points, stereo_colors)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # Save point cloud
    o3d.io.write_point_cloud(f"debug_stereo_cloud_frame_{args.frame}.ply", pcd)
    print(f"Saved debug_stereo_cloud_frame_{args.frame}.ply")
    
    # Attempt to visualize if display is available (might fail in headless env)
    try:
        o3d.visualization.draw_geometries([pcd, coord_frame], window_name=f"Stereo Cloud Frame {args.frame}")
    except Exception as e:
        print(f"Could not open visualization window (likely headless environment): {e}")
        print("Please download the .ply file to view locally.")

if __name__ == "__main__":
    main()
