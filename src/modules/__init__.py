"""
Module containing model components and architectures for the Diabetic Retinopathy Detection project.
"""

# Import common components for easier access from the package level
from .ala import ALA_Module

# Define what gets imported with "from modules import *"
__all__ = [
    'ALA_Module',
]