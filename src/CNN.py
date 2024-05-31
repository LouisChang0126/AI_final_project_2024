from dataset import read_image
import pandas as pd
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models


print(torch.cuda.is_available())
# Load images and targets
train_dataset, test_dataset, dataset = read_image(gray_img=False)
train_images = list()
train_targets = list()
test_images = list()
test_targets = list()
for data in train_dataset:
    train_images.append(data[0])
    train_targets.append(data[1])

for data in test_dataset:
    test_images.append(data[0])
    test_targets.append(data[1])

train_images = np.array(train_images)
train_targets = np.array(train_targets)
test_images = np.array(test_images)
test_targets = np.array(test_targets)


class ImageDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets
train_dataset = ImageDataset(train_images, train_targets, transform=transform)
testdataset = ImageDataset(test_images, test_targets, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(testdataset, batch_size=32, shuffle=False)

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
    
class DenseModel(nn.Module):
    def __init__(self, input_channels):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(input_channels * 256 * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)  # Regression output
        
    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ResNet50Regression(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(ResNet50Regression, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        # Modify the input layer to accept the desired number of input channels
        self.resnet50.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the output layer for regression task
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 1)
        
    def forward(self, x):
        return self.resnet50(x)
    


# Detect if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Determine input channels (1 for grayscale, 3 for RGB)
input_channels = 1 if len(train_images[0].shape) == 2 else 3

# Instantiate the model, define the loss function and the optimizer
#model = CNNModel(input_channels=input_channels).to(device)
#model = DenseModel(input_channels=input_channels).to(device)
model = ResNet50Regression(input_channels=input_channels).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Training loop
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device).float(), targets.to(device).float().view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}")
    
    # Evaluation on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device).float(), targets.to(device).float().view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Testing  Loss: {val_loss/len(test_loader):.4f}")
    print()

# Save the trained model
torch.save(model.state_dict(), 'cnn_regression_model.pth')

# Load the model
#model = CNNModel(input_channels=input_channels).to(device)
#model = DenseModel(input_channels=input_channels).to(device)
model = ResNet50Regression(input_channels=input_channels).to(device)
model.load_state_dict(torch.load('cnn_regression_model.pth'))
model.eval()

def predict(image_path, model, transform):
    # Read and preprocess the image
    image = cv2.imread(r'D:\user\Downloads (except program)\AI_final_store\AI_final_project_2024\AI_final_project_2024-main' + '/data/1800~1970/{}'.format(image_path))
    if input_channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=2)  # Add channel dimension
    if image is None:
        print(f"Error reading image: {image_path}")
        return

    image = cv2.resize(image, (256, 256))
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict the year
    with torch.no_grad():
        output = model(image)
    
    return output.item()

# Example prediction
image_path = '5.jpg'
predicted_year = predict(image_path, model, transform)
print(f"Predicted Year: {predicted_year}")
print(train_targets[4])
print()

i = 0
diff = list()
for data in test_dataset:
    i += 1
    image = data[0]
    year = data[1]
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    print(i, "image")
    print("Predicted Year :", output.item())
    print("Actual Year :", year)
    print("Difference between predict and true :", year - output)
    diff.append(abs((year - output).item()))
    print()

print('Avg :', sum(diff) / i)