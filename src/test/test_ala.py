import sys
import os
import torch

# Allow running as both script and module
try:
    # When running as a module with python -m src.test.test_ala
    from ..modules.ala import ALA_Module
except ImportError:
    # When running directly as python src/test/test_ala.py
    # Add the project root to the path to enable absolute imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.modules.ala import ALA_Module

ala = ALA_Module(in_channels=1024)

x = torch.randn(1, 1024, 14, 14)

out = ala(x)

print(out.shape)