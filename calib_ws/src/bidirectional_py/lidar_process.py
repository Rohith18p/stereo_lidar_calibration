import numpy as np
import cv2
from .config import Config

class LidarProcessor:
    def __init__(self, config=Config):
        self.config = config

    def filter_points(self, points):
        """
        Filter points based on depth and field of view.
        Assumes points are in Lidar frame.
        """
        # Filter by depth (x-coordinate in Velodyne frame usually points forward)
        # But we need to transform to camera frame to check FOV properly if we want to be precise.
        # For now, just simple distance filtering.
        
        dists = np.linalg.norm(points, axis=1)
        mask = (dists > self.config.MIN_DEPTH) & (dists < self.config.MAX_DEPTH)
        
        # Also filter points behind the lidar (x < 0) if we assume forward facing
        mask &= (points[:, 0] > 0)
        
        return points[mask]

    def project_to_image(self, points, rvec, tvec):
        """
        Project 3D points to 2D image plane using extrinsics and intrinsics.
        """
        # Convert rvec to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Transform to Camera Frame
        # P_cam = R * P_lidar + T
        points_cam = (R @ points.T).T + tvec
        
        # Project to Image
        # p_img = K * P_cam
        # We need to divide by Z
        
        fx = self.config.K[0, 0]
        fy = self.config.K[1, 1]
        cx = self.config.K[0, 2]
        cy = self.config.K[1, 2]
        
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        
        # Filter points behind camera
        valid_mask = z > 0
        
        u = (x[valid_mask] * fx / z[valid_mask]) + cx
        v = (y[valid_mask] * fy / z[valid_mask]) + cy
        
        # Filter points outside image bounds
        h, w = self.config.IMG_HEIGHT, self.config.IMG_WIDTH
        valid_mask_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        return np.stack([u[valid_mask_uv], v[valid_mask_uv]], axis=1), valid_mask, valid_mask_uv

    def segment_features(self, points):
        """
        Segment planar or edge features.
        For now, returns all points.
        """
        return points

    def remove_ground(self, points):
        """
        Remove ground plane using RANSAC.
        """
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Segment plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                                 ransac_n=3,
                                                 num_iterations=100)
        
        # Remove ground points (inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return np.asarray(outlier_cloud.points)

    def cluster_objects(self, points):
        """
        Cluster points using DBSCAN to find objects.
        Returns list of point clusters.
        """
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=20, print_progress=False))
        
        max_label = labels.max()
        clusters = []
        for i in range(max_label + 1):
            cluster_points = points[labels == i]
            # Filter by size (approximate car size)
            # Car dimensions approx: 4.5m x 1.8m x 1.5m
            # Simple bounding box check
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            dims = max_bound - min_bound
            
            # Loose constraints for "Car-like" objects
            if (dims[0] < 6.0 and dims[0] > 1.0 and 
                dims[1] < 3.0 and dims[1] > 0.5 and 
                dims[2] < 2.5 and dims[2] > 0.5):
                clusters.append(cluster_points)
                
        return clusters

    def get_potential_cars(self, points):
        """
        Pipeline to get potential car points.
        """
        non_ground = self.remove_ground(points)
        clusters = self.cluster_objects(non_ground)
        return clusters
