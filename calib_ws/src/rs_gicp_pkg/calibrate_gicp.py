import open3d as o3d
import numpy as np
import sys
import copy
import img_pcd
import read_pcd

# --- Configuration ---
VOXEL_SIZE = 0.5  # Downsampling voxel size (meters)
MAX_CORRESPONDENCE_DISTANCE = 1.0 # Max distance for GICP correspondence
MAX_ITERATIONS = 50
TOLERANCE = 1e-6
MAX_DIST = 50.0 # Max distance for points (meters)
FOV_DEG = 60.0  # Half-angle FOV in degrees (60 = 120° total sweep)
# ---------------------

def preprocess_pcd(pcd, voxel_size):
    """
    Downsamples the point cloud and estimates normals.
    """
    print(f"Preprocessing point cloud (Voxel Size: {voxel_size})...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    print(f"Estimating normals (Radius: {radius_normal})...")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    return pcd_down

def crop_lidar(pcd):
    """
    Crops the LiDAR point cloud to keep only relevant points (Front facing).
    """
    print("Cropping LiDAR point cloud...")
    points = np.asarray(pcd.points)
    
    # Filter conditions:
    # 1. x > 0 (Front only)
    # 2. Distance > 2.0m (Remove self-occlusion/vehicle parts)
    # 3. Distance < 50.0m (Remove distant points)
    
    dists = np.linalg.norm(points, axis=1)
    
    mask = (points[:, 0] > 0) & (dists > 2.0) & (dists < 50.0)
    
    pcd_cropped = pcd.select_by_index(np.where(mask)[0])
    print(f"Cropped LiDAR points: {len(points)} -> {len(pcd_cropped.points)}")
    return pcd_cropped

def crop_stereo(pcd):
    """
    Crops stereo point cloud to remove noise/distant points.
    """
    print("Cropping Stereo point cloud...")
    points = np.asarray(pcd.points)
    
    # Filter conditions:
    # 1. z > 0 (Front only - Camera frame usually Z forward)
    # 2. Distance < 50.0m
    
    dists = np.linalg.norm(points, axis=1)
    mask = (points[:, 2] > 0) & (dists < 50.0)
    
    pcd_cropped = pcd.select_by_index(np.where(mask)[0])
    print(f"Cropped Stereo points: {len(points)} -> {len(pcd_cropped.points)}")
    return pcd_cropped

def get_initial_guess():
    """
    Returns the initial transformation matrix from Camera Optical Frame to LiDAR Frame.
    Corrected Mapping:
    Cam Z (Forward) -> Lidar X (Forward)
    Cam X (Right)   -> Lidar -Y (Right)
    Cam Y (Down)    -> Lidar -Z (Up)
    
    Plus user-specified rotations in LiDAR frame:
    - Rotate 180° around LiDAR X-axis
    - Rotate 180° around LiDAR Z-axis
    """
    from scipy.spatial.transform import Rotation as R
    
    # Base rotation from camera to lidar frame
    T_base = np.eye(4)
    T_base[:3, :3] = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    
    # Apply rotations in LiDAR frame (extrinsic)
    # Order: R_z(180) * R_x(180) * T_base
    r_x = R.from_euler('x', 180, degrees=True)
    r_z = R.from_euler('z', 180, degrees=True)
    
    # Combined rotation (apply X first, then Z)
    r_combined = r_z * r_x
    
    T_rot = np.eye(4)
    T_rot[:3, :3] = r_combined.as_matrix()
    
    # Final transform
    T_final = T_rot @ T_base
    
    return T_final

def execute_gicp(source, target, voxel_size):
    """
    Aligns source to target using GICP.
    """
    print("Running GICP...")
    
    # Initial Guess
    trans_init = get_initial_guess()
    print("Using Initial Guess:")
    print(trans_init)
    
    # GICP Parameters
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
    
    print(f"GICP Converged: {reg_p2p.fitness:.4f} fitness, {reg_p2p.inlier_rmse:.4f} RMSE")
    print("Transformation Matrix:")
    print(reg_p2p.transformation)
    
    return reg_p2p.transformation

def visualize_registration(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # Don't paint - use original colors from point clouds
    # source_temp.paint_uniform_color([1, 0.706, 0]) # Yellow (Stereo)
    # target_temp.paint_uniform_color([0, 0.651, 0.929]) # Blue (LiDAR)
    
    source_temp.transform(transformation)
    
    # Create Coordinate Frames
    # LiDAR Frame (Target) - Fixed at Origin
    # Standard RGB (Red=X, Green=Y, Blue=Z)
    lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # Camera Frame (Source) - Transformed
    # Make it smaller and change colors to distinguish
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    
    # Modify Camera Frame Colors to distinguish from LiDAR
    # Red (X) -> Purple (1, 0, 1)
    # Green (Y) -> Yellow (1, 1, 0)
    # Blue (Z) -> Cyan (0, 1, 1)
    colors = np.asarray(camera_frame.vertex_colors)
    
    # Find indices for Red, Green, Blue
    # Note: Colors might not be exactly 1.0 or 0.0 due to shading, but usually create_coordinate_frame uses pure colors.
    # We use a threshold.
    is_red = (colors[:, 0] > 0.8) & (colors[:, 1] < 0.2) & (colors[:, 2] < 0.2)
    is_green = (colors[:, 0] < 0.2) & (colors[:, 1] > 0.8) & (colors[:, 2] < 0.2)
    is_blue = (colors[:, 0] < 0.2) & (colors[:, 1] < 0.2) & (colors[:, 2] > 0.8)
    
    colors[is_red] = [1, 0, 1]   # Purple
    colors[is_green] = [1, 1, 0] # Yellow
    colors[is_blue] = [0, 1, 1]  # Cyan
    
    camera_frame.vertex_colors = o3d.utility.Vector3dVector(colors)
    camera_frame.transform(transformation)
    
    print("Visualizing alignment (Stereo=RGB from image, LiDAR=Intensity grayscale)...")
    print("Axes:")
    print(" - LiDAR (Large): Red=X, Green=Y, Blue=Z")
    print(" - Camera (Small): Purple=X, Yellow=Y, Cyan=Z")
    
    o3d.visualization.draw_geometries([source_temp, target_temp, lidar_frame, camera_frame],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4538],
                                      up=[-0.3402, -0.9189, -0.1996])

from scipy.spatial.transform import Rotation as R

def get_manual_transform():
    """
    Returns the transformation matrix based on user provided values.
    User provided: Velodyne to Camera (T_cam_velo)
    
    R: 7.533745e-03 -9.999714e-01 -6.166020e-04 
       1.480249e-02 7.280733e-04 -9.998902e-01 
       9.998621e-01 7.523790e-03 1.480755e-02
    T: -4.069766e-03 -7.631618e-02 -2.717806e-01
    """
    # Rotation Matrix (3x3)
    R_val = np.array([
        [7.533745e-03, -9.999714e-01, -6.166020e-04],
        [1.480249e-02, 7.280733e-04, -9.998902e-01],
        [9.998621e-01, 7.523790e-03, 1.480755e-02]
    ])
    
    # Translation Vector (3,)
    T_val = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
    
    # T_cam_velo (Velodyne -> Camera)
    T_cam_velo = np.eye(4)
    T_cam_velo[:3, :3] = R_val
    T_cam_velo[:3, 3] = T_val
    
    print("Provided Transform (Velodyne -> Camera):")
    print(T_cam_velo)
    
    # T_velo_cam (Camera -> Velodyne) to align Stereo (Source, Camera Frame) to Lidar (Target, Velodyne Frame)
    T_velo_cam = np.linalg.inv(T_cam_velo)
    
    print("Base Transform (Camera -> Velodyne):")
    print(T_velo_cam)
    
    # Apply additional rotations requested by user:
    # 1. Rotate with respect to Velodyne Y by -90 degrees
    # 2. Rotate with respect to Velodyne Z by 90 degrees
    # Since these are about the fixed Velodyne frame (Target), we left-multiply.
    # Order: Apply Y(-90) first, then Z(90).
    # T_final = R_z(90) * R_y(-90) * T_velo_cam
    
    r_y = R.from_euler('y', -90, degrees=True)
    r_z = R.from_euler('z', 90, degrees=True)
    
    # Combined rotation (Z * Y)
    r_extra = r_z * r_y
    T_extra = np.eye(4)
    T_extra[:3, :3] = r_extra.as_matrix()
    
    T_final = T_extra @ T_velo_cam
    
    print("Final Transform with Additional Rotations:")
    print(T_final)
    
    return T_final

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 calibrate_gicp.py <left_img> <right_img> <lidar_bin>")
        return

    left_img = sys.argv[1]
    right_img = sys.argv[2]
    lidar_bin = sys.argv[3]

    # 1. Load Point Clouds
    print("--- Loading Data ---")
    pcd_stereo = img_pcd.compute_point_cloud(left_img, right_img, max_dist=MAX_DIST)
    pcd_lidar = read_pcd.read_bin(lidar_bin, max_dist=MAX_DIST, fov_deg=FOV_DEG)

    if pcd_stereo is None or pcd_lidar is None:
        print("Failed to load point clouds.")
        return

    print(f"Loaded {len(pcd_stereo.points)} stereo points")
    print(f"Loaded {len(pcd_lidar.points)} LiDAR points")

    # 2. Downsample & Normals for GICP
    print("\n--- Preprocessing for GICP ---")
    pcd_stereo_down = preprocess_pcd(pcd_stereo, VOXEL_SIZE)
    pcd_lidar_down = preprocess_pcd(pcd_lidar, VOXEL_SIZE)

    # 3. Get initial guess (current orientation setup)
    print("\n--- Initial Guess ---")
    initial_transform = get_initial_guess()
    print("Initial Transform (Camera -> LiDAR):")
    print(initial_transform)

    # 4. Run GICP
    print("\n--- Running GICP ---")
    final_transform = execute_gicp(pcd_stereo_down, pcd_lidar_down, VOXEL_SIZE)

    # 5. Print Final Extrinsic Matrix
    print("\n" + "="*60)
    print("FINAL EXTRINSIC CALIBRATION MATRIX")
    print("="*60)
    
    print("\n[1] Camera -> LiDAR Transformation (T_lidar_camera)")
    print("(Transforms stereo points to LiDAR frame)")
    print("-"*60)
    print(final_transform)
    print("\nRotation Matrix:")
    print(final_transform[:3, :3])
    print("\nTranslation Vector (meters):")
    print(final_transform[:3, 3])
    
    # Compute and print inverse
    inverse_transform = np.linalg.inv(final_transform)
    print("\n" + "-"*60)
    print("\n[2] LiDAR -> Camera Transformation (T_camera_lidar)")
    print("(Transforms LiDAR points to camera frame)")
    print("-"*60)
    print(inverse_transform)
    print("\nRotation Matrix:")
    print(inverse_transform[:3, :3])
    print("\nTranslation Vector (meters):")
    print(inverse_transform[:3, 3])
    print("\n" + "="*60)

    # 6. Visualize final alignment
    print("\n--- Visualizing Final Alignment ---")
    visualize_registration(pcd_stereo, pcd_lidar, final_transform)

if __name__ == "__main__":
    main()
