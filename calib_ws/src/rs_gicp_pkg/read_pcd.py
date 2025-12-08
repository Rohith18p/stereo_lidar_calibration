import numpy as np
import open3d as o3d
import sys
import os

def read_bin(file_path, max_dist=50.0, fov_deg=60.0):
    """
    Reads a KITTI .bin file and returns an Open3D PointCloud.
    
    Args:
        file_path: Path to .bin file
        max_dist: Maximum distance threshold (meters)
        fov_deg: Half-angle of FOV in degrees (e.g., 60 = 120° total sweep)
                 Filters points within ±fov_deg around the forward axis (X-axis for LiDAR)
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        # Load binary point cloud
        # KITTI point clouds are stored as float32 binaries (x, y, z, intensity)
        scan = np.fromfile(file_path, dtype=np.float32)
        scan = scan.reshape((-1, 4)) # Reshape to N x 4 

        # Filter by distance
        points = scan[:, :3]
        dists = np.linalg.norm(points, axis=1)
        mask = dists < max_dist
        
        # Filter by FOV (angular around Z-axis, keeping points in front ±fov_deg)
        # LiDAR X-axis is forward, so we compute angle in XY plane
        angles = np.arctan2(points[:, 1], points[:, 0]) * 180.0 / np.pi  # Convert to degrees
        fov_mask = np.abs(angles) <= fov_deg
        
        # Combine masks
        mask = mask & fov_mask
        
        points = points[mask]
        intensities = scan[mask, 3]

        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Use intensity as color (grayscale)
        # Normalize intensity to 0-1 for visualization
        intensities = scan[:, 3]
        max_intensity = np.max(intensities)
        if max_intensity > 0:
            intensities = intensities / max_intensity
        
        # Create Nx3 color array (R=G=B=intensity)
        colors = np.stack([intensities, intensities, intensities], axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def visualize(pcd):
    """
    Visualizes the point cloud.
    """
    if pcd is None:
        return
    
    print("Visualizing point cloud... (Close window to exit)")
    # Default view point
    o3d.visualization.draw_geometries([pcd], 
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 read_pcl.py <path_to_bin_file>")
        return

    file_path = sys.argv[1]
    print(f"Reading {file_path}...")
    
    pcd = read_bin(file_path)
    
    if pcd is not None:
        print(f"Point cloud has {len(pcd.points)} points.")
        visualize(pcd)

if __name__ == "__main__":
    main()
