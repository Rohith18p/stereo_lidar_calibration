import numpy as np

class Config:
    # Camera Intrinsics (P2 matrix from KITTI calib)
    # User should replace these with actual values from their dataset
    # Format: fx, 0, cx, 0, fy, cy, 0, 0, 1
    K = np.array([
        [718.856, 0.000000e+00, 609.55],
        [0.000000e+00, 718.856, 172.85],
        [0.000000e+00, 0.000000e+00, 1.000000e+00]
    ])
    
    # Stereo Baseline (in meters)
    BASELINE = 0.5372
    
    # Initial Extrinsics (Lidar to Camera)
    # Rotation matrix (R) and Translation vector (T)
    # T_lidar_to_cam
    # Updated with coarse alignment results from Frame 18
    init_rvec = np.array([0.5321, -1.5549, -0.0062]) # Rotation vector (rx, ry, rz)
    init_tvec = np.array([-0.0618, 0.2701, 0.6479]) # Translation vector (tx, ty, tz)
    
    # Optimization Parameters
    LEARNING_RATE = 0.01
    NUM_ITERATIONS = 100
    
    # Image Parameters
    IMG_WIDTH = 1242
    IMG_HEIGHT = 375
    
    # Lidar Parameters
    MIN_DEPTH = 2.0
    MAX_DEPTH = 60.0
