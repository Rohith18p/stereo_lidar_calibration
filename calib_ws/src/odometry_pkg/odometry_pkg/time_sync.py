import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import os

def load_csv(file_path):
    """Loads CSV data into a pandas DataFrame."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    return pd.read_csv(file_path)

def get_pose_matrix(row):
    """Constructs a 4x4 homogeneous transformation matrix from a CSV row."""
    # Extract position
    t = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
    
    # Extract orientation (quaternion: x, y, z, w)
    # Note: scipy Rotation expects (x, y, z, w)
    quat = [row['orient_x'], row['orient_y'], row['orient_z'], row['orient_w']]
    r = R.from_quat(quat)
    rot_mat = r.as_matrix()
    
    # Construct 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = t
    return T

def compute_relative_transform(T1, T2):
    """Computes the relative transformation from T1 to T2 (T_rel = inv(T1) * T2)."""
    T1_inv = np.linalg.inv(T1)
    T_rel = np.dot(T1_inv, T2)
    return T_rel

def decompose_transform(T):
    """Decomposes a 4x4 matrix into rotation matrix (3x3) and translation vector (3x1)."""
    return T[:3, :3], T[:3, 3]

def main():
    # Define paths (using the paths from previous context)
    lidar_csv_path = '/home/ros/ros_ws/src/data/store_v1/lidar_odom.csv'
    visual_csv_path = '/home/ros/ros_ws/src/data/store_v1/visual_odom.csv'
    output_csv_path = '/home/ros/ros_ws/src/data/store_v1/time_synced.csv'

    print(f"Loading Lidar data from: {lidar_csv_path}")
    lidar_df = load_csv(lidar_csv_path)
    print(f"Loading Visual data from: {visual_csv_path}")
    visual_df = load_csv(visual_csv_path)

    if lidar_df is None or visual_df is None:
        return

    # Arrays to store synchronized poses (4x4 matrices)
    lidar_poses = []
    visual_poses = []
    timestamps = []

    print("Synchronizing data...")
    # Iterate through Lidar data and find closest Visual match
    # Assuming timestamps are in seconds (float) or similar comparable format
    # The CSV header implies timestamp is a float or string. Let's parse it if needed.
    # Based on previous code: f"{odom.header.stamp.sec}.{odom.header.stamp.nanosec:09d}" -> this is a string in CSV?
    # Let's check the CSV format or assume it's loadable as float if it looks like a number.
    # If it's a string "sec.nanosec", pandas might read it as object/string.
    
    # Helper to parse timestamp
    def parse_timestamp(val):
        try:
            return float(val)
        except:
            return 0.0

    lidar_times = lidar_df['timestamp'].apply(parse_timestamp).values
    visual_times = visual_df['timestamp'].apply(parse_timestamp).values

    for i, l_time in enumerate(lidar_times):
        # Find index of closest timestamp in visual data
        idx = (np.abs(visual_times - l_time)).argmin()
        v_time = visual_times[idx]
        
        # Optional: Define a max time difference threshold for valid sync
        time_diff = abs(l_time - v_time)
        if time_diff > 0.1: # 100ms threshold, adjust as needed
            continue

        lidar_row = lidar_df.iloc[i]
        visual_row = visual_df.iloc[idx]

        lidar_poses.append(get_pose_matrix(lidar_row))
        visual_poses.append(get_pose_matrix(visual_row))
        timestamps.append(l_time)

    print(f"Synchronized {len(lidar_poses)} poses.")

    if len(lidar_poses) < 2:
        print("Not enough synchronized data to compute extrinsics.")
        return

    # Compute relative motions (A and B)
    # A_i = inv(L_i) * L_{i+1}
    # B_i = inv(V_i) * V_{i+1}
    
    R_lidar_rel = []
    t_lidar_rel = []
    R_visual_rel = []
    t_visual_rel = []

    # We need pairs of poses to compute relative motion
    # If we have N poses, we have N-1 relative motions
    # For sliding window of size W (poses), we have W-1 relative motions.
    
    # Pre-compute all relative motions
    for i in range(len(lidar_poses) - 1):
        rel_lidar = compute_relative_transform(lidar_poses[i], lidar_poses[i+1])
        rel_visual = compute_relative_transform(visual_poses[i], visual_poses[i+1])
        
        Rl, tl = decompose_transform(rel_lidar)
        Rv, tv = decompose_transform(rel_visual)
        
        R_lidar_rel.append(Rl)
        t_lidar_rel.append(tl)
        R_visual_rel.append(Rv)
        t_visual_rel.append(tv)

    # Sliding Window Calibration
    window_size = 50 # Number of relative motions (so window of poses is size+1)
    
    results = []

    print(f"Starting sliding window calibration (Window Size: {window_size})...")

    for i in range(len(R_lidar_rel) - window_size + 1):
        # Extract window
        R_lidar_window = R_lidar_rel[i : i + window_size]
        t_lidar_window = t_lidar_rel[i : i + window_size]
        R_visual_window = R_visual_rel[i : i + window_size]
        t_visual_window = t_visual_rel[i : i + window_size]

        try:
            # Solve AX = XB
            # A corresponds to Lidar (or Visual? Depends on definition). 
            # Usually we want T_cam_lidar (Lidar in Camera frame) or T_lidar_cam.
            # cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam) -> R_cam2gripper, t_cam2gripper
            # Mapping:
            # gripper2base -> Lidar Relative (A)
            # target2cam -> Visual Relative (B)
            # Result -> T_visual_lidar (Lidar in Visual frame)
            
            R_calib, t_calib = cv2.calibrateHandEye(
                R_lidar_window, t_lidar_window,
                R_visual_window, t_visual_window,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Convert result to quaternion for storage
            r = R.from_matrix(R_calib)
            quat = r.as_quat() # x, y, z, w
            
            # Store result with timestamp (using the timestamp of the END of the window)
            window_end_time = timestamps[i + window_size]
            
            results.append({
                'timestamp': window_end_time,
                'trans_x': t_calib[0][0],
                'trans_y': t_calib[1][0],
                'trans_z': t_calib[2][0],
                'quat_x': quat[0],
                'quat_y': quat[1],
                'quat_z': quat[2],
                'quat_w': quat[3]
            })
            
        except Exception as e:
            print(f"Calibration failed at window {i}: {e}")

    # Write results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv_path, index=False)
        print(f"Calibration complete. Results written to {output_csv_path}")
        print(f"Total extrinsic estimates: {len(results)}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
