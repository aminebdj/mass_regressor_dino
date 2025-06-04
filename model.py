import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. Load model
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14.eval().to(device)

# 4. Feature Extractor using DINO
@torch.no_grad()
def extract_features(images):
    return dinov2_vits14.forward_features(images)['x_norm_clstoken']

# 5. Regression Head (Trainable)
class DINORegressor(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, device=device):
        super(DINORegressor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, images):
        a, b = images.shape[0], images.shape[1]
        rest = images.shape[2:]
        with torch.no_grad():
            features = extract_features(images.view(a * b, *rest))
            
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        # x = x.view(a, b, x.shape[-1])
        return x.squeeze(1)
