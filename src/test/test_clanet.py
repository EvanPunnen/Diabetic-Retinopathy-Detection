import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.clanet import CLANet_DenseNet


model = CLANet_DenseNet(num_classes=5)
x = torch.randn(2, 3, 224, 224)
out = model(x)
print("Output shape:", out.shape)