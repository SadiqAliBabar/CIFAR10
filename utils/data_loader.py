import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjust input size based on image dimensions
        self.fc2 = nn.Linear(512, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Apply conv1
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Apply conv2
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  # Apply conv3
        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = F.relu(self.fc1(x))  # Apply fully connected layer 1
        x = self.fc2(x)  # Apply fully connected layer 2
        return x
