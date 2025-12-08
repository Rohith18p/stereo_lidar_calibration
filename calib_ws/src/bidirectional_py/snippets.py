# ==========================================
# BLOCK 1: Image Segmentation (DeepLabV3)
# ==========================================
import torch
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_car_mask(image):
    # Load model
    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Preprocess
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Class 7 is 'car' in COCO
    car_mask = (output_predictions == 7).astype(np.uint8)
    return car_mask

# Usage
# img_left = cv2.imread("path/to/image.png")
# mask = get_car_mask(img_left)
# plt.imshow(mask, cmap='gray')
# plt.show()


# ==========================================
# BLOCK 2: Lidar Clustering (Open3D)
# ==========================================
import open3d as o3d

def get_lidar_clusters(lidar_points):
    # 1. Remove Ground
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                             ransac_n=3,
                                             num_iterations=100)
    
    non_ground = pcd.select_by_index(inliers, invert=True)
    
    # 2. Cluster
    labels = np.array(non_ground.cluster_dbscan(eps=0.5, min_points=20, print_progress=False))
    
    max_label = labels.max()
    clusters = []
    points_np = np.asarray(non_ground.points)
    
    for i in range(max_label + 1):
        cluster_points = points_np[labels == i]
        
        # Filter by size (Car dimensions approx)
        min_bound = cluster_points.min(axis=0)
        max_bound = cluster_points.max(axis=0)
        dims = max_bound - min_bound
        
        # Loose constraints: Length 1-6m, Width 0.5-3m, Height 0.5-2.5m
        if (dims[0] < 6.0 and dims[0] > 1.0 and 
            dims[1] < 3.0 and dims[1] > 0.5 and 
            dims[2] < 2.5 and dims[2] > 0.5):
            clusters.append(cluster_points)
            
    return clusters

# Usage
# clusters = get_lidar_clusters(lidar_points)
# print(f"Found {len(clusters)} clusters")


# ==========================================
# BLOCK 3: Project & Visualize
# ==========================================
def project_clusters(img, clusters, rvec, tvec, K):
    img_vis = img.copy()
    
    if not clusters:
        return img_vis
        
    all_points = np.vstack(clusters)
    
    # Project
    R, _ = cv2.Rodrigues(rvec)
    points_cam = (R @ all_points.T).T + tvec
    
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    
    valid = z > 0
    u = (x[valid] * K[0,0] / z[valid]) + K[0,2]
    v = (y[valid] * K[1,1] / z[valid]) + K[1,2]
    
    for i in range(len(u)):
        cv2.circle(img_vis, (int(u[i]), int(v[i])), 2, (0, 255, 0), -1)
        
    return img_vis

# Usage
# img_out = project_clusters(img_left, clusters, rvec, tvec, config.K)
# plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
# plt.show()
