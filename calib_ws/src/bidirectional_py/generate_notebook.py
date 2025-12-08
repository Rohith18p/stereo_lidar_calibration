import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Imports
text_1 = """\
# Step 1: Stereo Disparity & Point Cloud Debugging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os

# Display plots inline
%matplotlib inline
"""

# Cell 2: Config
text_2 = """\
# Configuration
class Config:
    # Camera Intrinsics (P2 matrix from KITTI calib)
    # Format: fx, 0, cx, 0, fy, cy, 0, 0, 1
    K = np.array([
        [718.856, 0.000000e+00, 609.55],
        [0.000000e+00, 718.856, 172.85],
        [0.000000e+00, 0.000000e+00, 1.000000e+00]
    ])
    
    # Stereo Baseline (in meters)
    BASELINE = 0.5372
    
    # Image Parameters
    IMG_WIDTH = 1242
    IMG_HEIGHT = 375
    
    # Lidar Parameters
    MIN_DEPTH = 2.0
    MAX_DEPTH = 80.0

config = Config()
print("Configuration loaded.")
"""

# Cell 3: Data Loading
text_3 = """\
# Data Loading
# Update these paths to your specific dataset location
img_left_dir = "/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data"
img_right_dir = "/home/nail/other_work/Ro/CL_Calib/calib_ws/src/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_03/data"

frame_index = 18

def load_images(idx):
    filenames = sorted([f for f in os.listdir(img_left_dir) if f.endswith('.png')])
    if idx >= len(filenames):
        print(f"Index {idx} out of bounds.")
        return None, None
        
    filename = filenames[idx]
    img_left_path = os.path.join(img_left_dir, filename)
    img_right_path = os.path.join(img_right_dir, filename)
    
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    
    return img_left, img_right

img_left, img_right = load_images(frame_index)

if img_left is not None:
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    plt.title("Left Image")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    plt.title("Right Image")
    plt.show()
else:
    print("Failed to load images.")
"""

# Cell 4: Stereo Matching (Tunable)
text_4 = """\
# Stereo Matching (SGBM)
# Tune these parameters to improve disparity map quality

# SGBM Parameters
min_disp = 0
num_disp = 16 * 6  # Needs to be divisible by 16
block_size = 11    # Matched block size. It must be an odd number >= 1

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size**2,
    P2=32 * 3 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute Disparity
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

# Visualize Disparity
plt.figure(figsize=(15, 5))
plt.imshow(disparity, cmap='jet')
plt.colorbar()
plt.title("Disparity Map")
plt.show()
"""

# Cell 5: Depth Calculation
text_5 = """\
# Disparity to Depth
def disparity_to_depth(disparity, config):
    # Depth = (f * b) / disparity
    mask = disparity > 0
    depth = np.zeros_like(disparity)
    depth[mask] = (config.K[0, 0] * config.BASELINE) / disparity[mask]
    return depth

depth = disparity_to_depth(disparity, config)

# Visualize Depth
plt.figure(figsize=(15, 5))
plt.imshow(depth, cmap='plasma', vmin=0, vmax=80)
plt.colorbar()
plt.title("Depth Map (m)")
plt.show()
"""

# Cell 6: Point Cloud Generation
text_6 = """\
# Depth to Point Cloud
def depth_to_pointcloud(depth, img_rgb, config):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    fx = config.K[0, 0]
    fy = config.K[1, 1]
    cx = config.K[0, 2]
    cy = config.K[1, 2]
    
    mask = (depth > config.MIN_DEPTH) & (depth < config.MAX_DEPTH)
    
    z = depth[mask]
    x = (u[mask] - cx) * z / fx
    y = (v[mask] - cy) * z / fy
    
    points = np.stack([x, y, z], axis=1)
    colors = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)[mask] / 255.0
    
    return points, colors

points, colors = depth_to_pointcloud(depth, img_left, config)
print(f"Generated {len(points)} points.")

# Save to PLY for visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud("debug_stereo_notebook.ply", pcd)
print("Saved point cloud to debug_stereo_notebook.ply")
"""

# Cell 7: Visualize Point Cloud
text_7 = """\
# Visualize Point Cloud
# Note: This will open a separate window. If running remotely, this might not work.
print("Visualizing point cloud...")
if os.path.exists("debug_stereo_notebook.ply"):
    pcd = o3d.io.read_point_cloud("debug_stereo_notebook.ply")
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord_frame], window_name="Stereo Point Cloud", width=1024, height=768)
else:
    print("Point cloud file not found. Please run the previous cell to generate it.")
"""

nb['cells'] = [
    nbf.v4.new_code_cell(text_1),
    nbf.v4.new_code_cell(text_2),
    nbf.v4.new_code_cell(text_3),
    nbf.v4.new_code_cell(text_4),
    nbf.v4.new_code_cell(text_5),
    nbf.v4.new_code_cell(text_6),
    nbf.v4.new_code_cell(text_7)
]

with open('/home/nail/other_work/Ro/CL_Calib/calib_ws/src/bidirectional_py/step1_stereo_debug.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
