import os
import cv2
import numpy as np
import open3d as o3d

class KittiLoader:
    def __init__(self, img_left_dir, img_right_dir, lidar_dir):
        self.img_left_dir = img_left_dir
        self.img_right_dir = img_right_dir
        self.lidar_dir = lidar_dir
        
        if not os.path.exists(self.img_left_dir):
             raise FileNotFoundError(f"Left image directory not found: {self.img_left_dir}")
        
        self.filenames = sorted([f.split('.')[0] for f in os.listdir(self.img_left_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.filenames)
        
    def get_item(self, idx):
        filename = self.filenames[idx]
        
        # Load Images
        img_left_path = os.path.join(self.img_left_dir, f"{filename}.png")
        img_right_path = os.path.join(self.img_right_dir, f"{filename}.png")
        
        img_left = cv2.imread(img_left_path)
        img_right = cv2.imread(img_right_path)
        
        if img_left is None or img_right is None:
            raise FileNotFoundError(f"Image not found: {img_left_path} or {img_right_path}")
            
        # Load Lidar
        lidar_path = os.path.join(self.lidar_dir, f"{filename}.bin")
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        return {
            'img_left': img_left,
            'img_right': img_right,
            'lidar_points': point_cloud[:, :3], # XYZ only
            'filename': filename
        }

if __name__ == "__main__":
    # Test loader
    # Adjust path as needed
    loader = KittiLoader("/home/nail/other_work/Ro/CL_Calib/calib_ws/src/bidirectional_py/data/", "/home/nail/other_work/Ro/CL_Calib/calib_ws/src/bidirectional_py/data/", "/home/nail/other_work/Ro/CL_Calib/calib_ws/src/bidirectional_py/data/")
    print(f"Found {len(loader)} frames")
