from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
import numpy as np
import torchvision
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models

class CNNModel(nn.Module):
    def __init__(self, input_channels):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 1)  # Regression output
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(input_channels=3).to(device)
model.load_state_dict(torch.load('../models/cnn_regression_model.pth'))
target_layer = [model.conv2]

img_path = 'test.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize((256,256))
img = np.array(img)
img_float_np = np.float32(img)/255
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

input_tensor = transform(img)
input_tensor = input_tensor.to(device)

input_tensor = input_tensor.unsqueeze(0)

cam = GradCAM(model=model, target_layers=target_layer)
targets = None 
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  

grayscale_cam = grayscale_cam[0,:]
cam_image = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=False)
cv2.imwrite(f'test_output.jpg', cam_image)