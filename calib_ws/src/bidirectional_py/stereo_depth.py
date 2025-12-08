import cv2
import numpy as np
import open3d as o3d
from .config import Config

class StereoProcessor:
    def __init__(self, config=Config):
        self.config = config
        # SGBM Parameters (can be tuned)
        window_size = 3
        min_disp = 0
        num_disp = 16 * 10 # Must be divisible by 16
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=16,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def compute_disparity(self, img_left, img_right):
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        return disparity

    def disparity_to_depth(self, disparity):
        # Depth = (f * b) / disparity
        # Avoid division by zero
        mask = disparity > 0
        depth = np.zeros_like(disparity)
        depth[mask] = (self.config.K[0, 0] * self.config.BASELINE) / disparity[mask]
        return depth

    def depth_to_pointcloud(self, depth, img_rgb=None):
        h, w = depth.shape
        
        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Reproject to 3D
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        
        fx = self.config.K[0, 0]
        fy = self.config.K[1, 1]
        cx = self.config.K[0, 2]
        cy = self.config.K[1, 2]
        
        mask = (depth > self.config.MIN_DEPTH) & (depth < self.config.MAX_DEPTH)
        
        z = depth[mask]
        x = (u[mask] - cx) * z / fx
        y = (v[mask] - cy) * z / fy
        
        points = np.stack([x, y, z], axis=1)
        
        colors = None
        if img_rgb is not None:
            colors = img_rgb[mask] / 255.0 # Normalize to [0, 1]
            
        return points, colors

    def create_o3d_cloud(self, points, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
