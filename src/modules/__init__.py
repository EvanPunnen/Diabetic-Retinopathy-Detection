"""
Modules package for Diabetic-Retinopathy-Detection project
This package contains various modules used in the model architecture
"""

# Import common components for easier access from the package level
from .ala import ALA_Module

# Define what gets imported with "from modules import *"
__all__ = [
    'ALA_Module',
]
