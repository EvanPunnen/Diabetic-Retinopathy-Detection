import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.clanet import CLANet_DenseNet

# Initialize model (skip pretrained for now)
model = CLANet_DenseNet(num_classes=5)
print("CLANet (ALA + CSCA) initialized successfully")

# Test with dummy input
x = torch.randn(2, 3, 224, 224)
out = model(x)

print("Output shape:", out.shape)
print(" Forward pass successful")

# Model stats
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
