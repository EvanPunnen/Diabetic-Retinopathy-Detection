import torch
import torch.nn as nn
import torch.nn.functional as F

class ALA_Module(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ALA_Module, self).__init__()
        
        # Multi-scale convolution layers (for different lesion sizes)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        
        # Global context to generate adaptive weights
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, 3)  # 3 for 1x1, 3x3, 5x5 branches
        
        # Channel attention (like SE block)
        self.channel_fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.channel_fc2 = nn.Linear(in_channels // reduction, in_channels)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Multi-scale feature extraction
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv5x5(x)
        
        # Adaptive weighting based on global context
        g = self.global_pool(x).view(b, c)
        w = self.fc2(self.relu(self.fc1(g)))  # [B, 3]
        w = self.sigmoid(w)
        w = F.softmax(w, dim=1)  # Normalize weights
        
        # Weighted sum of scales
        f = w[:, 0].view(b, 1, 1, 1) * f1 + w[:, 1].view(b, 1, 1, 1) * f2 + w[:, 2].view(b, 1, 1, 1) * f3
        
        # Channel attention refinement
        ca = self.global_pool(f).view(b, c)
        ca = self.channel_fc2(self.relu(self.channel_fc1(ca)))
        ca = self.sigmoid(ca).view(b, c, 1, 1)
        f = f * ca
        
        # Residual connection
        out = f + x
        
        return out
