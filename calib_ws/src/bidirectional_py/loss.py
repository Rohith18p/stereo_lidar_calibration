import numpy as np
import cv2
from scipy.spatial import cKDTree
from .config import Config

class LossCalculator:
    def __init__(self, config=Config):
        self.config = config

    def project_lidar_to_camera(self, lidar_points, rvec, tvec):
        """
        Project lidar points to camera frame.
        """
        R, _ = cv2.Rodrigues(rvec)
        points_cam = (R @ lidar_points.T).T + tvec
        return points_cam

    def bidirectional_loss(self, lidar_points, stereo_cloud, rvec, tvec):
        """
        Compute bidirectional loss between lidar points and stereo point cloud.
        """
        # 1. Lidar -> Stereo Loss (3D Point to Point)
        # Transform lidar points to camera frame
        lidar_cam = self.project_lidar_to_camera(lidar_points, rvec, tvec)
        
        # Build KDTree for stereo cloud for fast nearest neighbor search
        if len(stereo_cloud) == 0 or len(lidar_cam) == 0:
            return float('inf')
            
        tree = cKDTree(stereo_cloud)
        
        # Find nearest neighbors for each lidar point
        dists_l2s, _ = tree.query(lidar_cam)
        
        # Robust loss (e.g., Huber or truncated)
        # For simplicity, mean squared error of distances, truncated
        # Robust loss (e.g., Huber or truncated)
        # For simplicity, mean squared error of distances, truncated
        threshold = 10.0 # Increased threshold to capture larger misalignments
        
        # Debug prints
        # print(f"Mean L2S dist: {np.mean(dists_l2s):.4f}, Min: {np.min(dists_l2s):.4f}")
        
        dists_l2s = np.clip(dists_l2s, 0, threshold)
        loss_l2s = np.mean(dists_l2s**2)
        
        # 2. Stereo -> Lidar Loss (Inverse)
        # Build KDTree for transformed lidar points
        tree_lidar = cKDTree(lidar_cam)
        dists_s2l, _ = tree_lidar.query(stereo_cloud)
        dists_s2l = np.clip(dists_s2l, 0, threshold)
        loss_s2l = np.mean(dists_s2l**2)
        
        # Total Loss
        total_loss = loss_l2s + loss_s2l
        
        return total_loss

    def semantic_loss(self, lidar_clusters, car_mask, rvec, tvec):
        """
        Compute semantic loss: distance of projected lidar cluster points to car mask.
        """
        if not lidar_clusters:
            return 0.0
            
        # Combine all cluster points
        all_points = np.vstack(lidar_clusters)
        
        # Project to image
        uv, valid_mask, valid_mask_uv = self.project_to_image_loss(all_points, rvec, tvec)
        
        if len(uv) == 0:
            return 1000.0 # High penalty if no points project to image
            
        # Compute Distance Transform of the inverted mask
        # dist_transform[y, x] = distance to nearest zero pixel (background)
        # We want distance to nearest ONE pixel (car).
        # So we invert mask: 0=Car, 1=Background.
        # Then dist_transform gives distance to nearest Car.
        
        inv_mask = 1 - car_mask
        dist_transform = cv2.distanceTransform(inv_mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Sample distance transform at projected points
        # bilinear interpolation or nearest neighbor
        # For simplicity, nearest neighbor (integer coordinates)
        u = np.clip(uv[:, 0].astype(int), 0, self.config.IMG_WIDTH - 1)
        v = np.clip(uv[:, 1].astype(int), 0, self.config.IMG_HEIGHT - 1)
        
        dists = dist_transform[v, u]
        
        # Loss is mean distance
        loss = np.mean(dists)
        
        return loss

    def project_to_image_loss(self, points, rvec, tvec):
        """
        Helper to project points for loss calculation (duplicates logic but avoids circular import).
        """
        R, _ = cv2.Rodrigues(rvec)
        points_cam = (R @ points.T).T + tvec
        
        fx = self.config.K[0, 0]
        fy = self.config.K[1, 1]
        cx = self.config.K[0, 2]
        cy = self.config.K[1, 2]
        
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        
        valid_mask = z > 0
        
        if not np.any(valid_mask):
            return np.array([]), valid_mask, valid_mask
            
        u = (x[valid_mask] * fx / z[valid_mask]) + cx
        v = (y[valid_mask] * fy / z[valid_mask]) + cy
        
        h, w = self.config.IMG_HEIGHT, self.config.IMG_WIDTH
        valid_mask_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        return np.stack([u[valid_mask_uv], v[valid_mask_uv]], axis=1), valid_mask, valid_mask_uv
