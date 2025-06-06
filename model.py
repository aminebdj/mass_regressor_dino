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
class Regressor(nn.Module):
    def __init__(self, feature_extractor='dino', hidden_dim=256, device=device, tune_blocks = []):
        super(Regressor, self).__init__()
        self.device = device
        self.feature_extractor_name = feature_extractor
        train_encoder = False
        if feature_extractor == 'dino':
            self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)
            self.input_dim = 384  # For dinov2_vits14
        elif feature_extractor == 'clip':
            import clip
            self.feature_extractor, _ = clip.load("RN50", device=device)
            self.feature_extractor.eval()
            self.input_dim = 256  # CLIP ViT-B/32 output dim
            # Freeze all CLIP parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            # Unfreeze last transformer block of the visual encoder
            # ViT-B/32 has 12 layers; unfreeze the 12th (index -1)
            # Unfreeze only specified layers
            for name in tune_blocks:
                layer = getattr(self.feature_extractor.visual, name)
                for param in layer.parameters():
                    param.requires_grad = True

        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

        self.regressor = nn.Sequential(
        
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    @torch.no_grad()
    def extract_features(self, images):
        if self.feature_extractor_name == 'dino':
            return self.feature_extractor.forward_features(images)['x_norm_clstoken']
        elif self.feature_extractor_name == 'clip':
            images = transforms.Resize(224)(images)  # Ensure proper size
            images = transforms.CenterCrop(224)(images)
            return self.feature_extractor.encode_image(images)
    
    def forward(self, images):
        B, N, C, H, W = images.shape  # For batch processing of multiple images per sample
        images = images.view(B * N, C, H, W).to(self.device)

        features = self.extract_features(images)

        out = self.regressor(features.float())
        # out = out.view(B, N)  # Return per-sample predictions if needed
        return out.squeeze()