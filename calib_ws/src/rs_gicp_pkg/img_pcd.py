import cv2
import numpy as np
import open3d as o3d
import sys
import os

# --- User Configuration ---
# Camera Intrinsics (Please replace with your actual values)
FX = 718.856
FY = 718.856
CX = 607.1928
CY = 185.2157

# Stereo Baseline (in meters)
BASELINE = 0.5372

# Disparity Parameters (Tweak these for better matching)
MIN_DISPARITY = 0
NUM_DISPARITIES = 16 * 6  # Must be divisible by 16
BLOCK_SIZE = 11
# --------------------------

def compute_point_cloud(left_img_path, right_img_path, fx=FX, fy=FY, cx=CX, cy=CY, baseline=BASELINE, max_dist=50.0):
    """
    Computes a point cloud from a pair of stereo images.
    """
    if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
        print("Error: Image files not found.")
        return None

    # Load images
    imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    imgL_color = cv2.imread(left_img_path, cv2.IMREAD_COLOR) # For coloring the point cloud

    if imgL is None or imgR is None:
        print("Error: Could not read images.")
        return None

    # Compute Disparity
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    print("Computing disparity map...")
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # Q Matrix for Reprojection
    # Q = [[1, 0, 0, -cx],
    #      [0, 1, 0, -cy],
    #      [0, 0, 0, f], 
    #      [0, 0, -1/Tx, (cx - c'x)/Tx]] # Simplified for standard stereo
    
    # Using the provided parameters:
    # We assume images are rectified.
    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx], # Assuming FX approx FY
        [0, 0, -1.0/baseline, 0] # 
    ])

    print("Reprojecting to 3D...")
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    
    # Fix mirror image - flip X axis
    points_3d[:, :, 0] = -points_3d[:, :, 0]
    
    # Filter out invalid points (disparity <= 0 or too far)
    # Also filter by max_dist
    dists = np.linalg.norm(points_3d, axis=2)
    mask = (disparity > 0) & (dists < max_dist)
    
    # Extract valid points and colors
    output_points = points_3d[mask]
    output_colors = imgL_color[mask]
    
    # Convert BGR to RGB
    output_colors = cv2.cvtColor(output_colors.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    output_colors = output_colors / 255.0 # Normalize to 0-1

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points)
    pcd.colors = o3d.utility.Vector3dVector(output_colors)

    return pcd

def visualize(pcd):
    if pcd is None:
        return
    
    print("Visualizing point cloud... (Close window to exit)")
    
    # Coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, coord_frame],
                                      zoom=0.5,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 10],
                                      up=[0, -1, 0])

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 img_pcd.py <left_image_path> <right_image_path>")
        print("Example: python3 img_pcd.py data/left.png data/right.png")
        return

    left_path = sys.argv[1]
    right_path = sys.argv[2]
    
    print(f"Processing:\n Left: {left_path}\n Right: {right_path}")
    print(f"Intrinsics: fx={FX}, fy={FY}, cx={CX}, cy={CY}")
    print(f"Baseline: {BASELINE}m")

    pcd = compute_point_cloud(left_path, right_path)
    
    if pcd is not None:
        print(f"Generated point cloud with {len(pcd.points)} points.")
        visualize(pcd)

if __name__ == "__main__":
    main()
