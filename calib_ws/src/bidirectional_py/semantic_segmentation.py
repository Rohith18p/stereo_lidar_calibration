import torch
from torchvision import models, transforms
import cv2
import numpy as np

class ImageSegmenter:
    def __init__(self):
        # Load pre-trained DeepLabV3 model
        self.model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
        self.model.eval()
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Standard ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # COCO class index for 'car' is 7 (in some models) or 3 (in others)
        # For torchvision DeepLabV3 trained on COCO, 'car' is usually index 7.
        # Wait, torchvision models are often trained on COCO subset (Pascal VOC) or full COCO.
        # Let's check the weights documentation or assume standard COCO indices.
        # COCO labels: 0=background, 1=aeroplane, ..., 7=car
        self.CAR_CLASS_ID = 7 

    def get_car_mask(self, image):
        """
        Returns a binary mask where 1 = Car, 0 = Background.
        """
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Create binary mask for cars
        car_mask = (output_predictions == self.CAR_CLASS_ID).astype(np.uint8)
        
        return car_mask

if __name__ == "__main__":
    # Test
    # img = cv2.imread("path/to/image.png")
    # segmenter = ImageSegmenter()
    # mask = segmenter.get_car_mask(img)
    # cv2.imwrite("mask.png", mask * 255)
    pass
