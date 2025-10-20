from torchvision import models
import torch.nn as nn
from modules.ala import ALA_Module  

class CLANet_DenseNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CLANet_DenseNet, self).__init__()
        
        base = models.densenet121(weights="IMAGENET1K_V1")
        self.features = base.features
        self.ala = ALA_Module(in_channels=1024)   
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.ala(x)  # 🔥 adaptive lesion-aware enhancement
        x = self.classifier(x)
        return x