import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from modules
from modules.ala import ALA_Module

ala = ALA_Module(in_channels=1024)

x = torch.randn(1, 1024, 14, 14)

out = ala(x)

print(out.shape)