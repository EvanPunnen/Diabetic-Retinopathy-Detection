from torchvision import models
import torch.nn as nn
import torch
import os
from modules.ala import ALA_Module
from modules.csca import CSCA_Module  # ✅ Added CSCA module

class CLANet_DenseNet(nn.Module):
    def __init__(self, num_classes=5, pretrained_weights=None):
        super(CLANet_DenseNet, self).__init__()
        
        # Base DenseNet backbone
        base = models.densenet121(weights="IMAGENET1K_V1")
        self.features = base.features
        
        # CLANet-specific attention modules
        self.ala = ALA_Module(in_channels=1024)   # ✅ Adaptive Lesion-Aware module
        self.csca = CSCA_Module(in_channels=1024) # ✅ Cross-Scale Context Attention module
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
        
        # Optional pretrained weight loading (DenseNet feature reuse)
        if pretrained_weights:
            if not os.path.exists(pretrained_weights):
                pretrained_weights = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'models', pretrained_weights
                )
            
            if not os.path.exists(pretrained_weights):
                raise FileNotFoundError(f"Cannot find pretrained weights at {pretrained_weights}")
                
            print(f"Loading pretrained weights from {pretrained_weights}")
            
            checkpoint = torch.load(pretrained_weights, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
                print(f"Loaded model checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                state_dict = checkpoint
                print("Loaded model weights directly")
            
            # Load DenseNet backbone features
            features_dict = {
                k.replace('features.', ''): v for k, v in state_dict.items() if k.startswith('features')
            }
            
            print("Loading feature weights...")
            self.features.load_state_dict(features_dict, strict=False)
            print("Features loaded successfully")
            
            # Handle classifier weights
            if 'classifier.weight' in state_dict and 'classifier.bias' in state_dict:
                cls_weight = state_dict['classifier.weight']
                cls_bias = state_dict['classifier.bias']
                
                print(f"Found classifier weights with shape: {cls_weight.shape}")
                print(f"Our classifier expects shape: {self.classifier[2].weight.shape}")
                
                if cls_weight.shape[1] == self.classifier[2].weight.shape[1]:
                    print("Direct mapping of classifier weights...")
                    if cls_weight.shape[0] == self.classifier[2].weight.shape[0]:
                        self.classifier[2].weight.data.copy_(cls_weight)
                        self.classifier[2].bias.data.copy_(cls_bias)
                    else:
                        min_classes = min(cls_weight.shape[0], self.classifier[2].weight.shape[0])
                        self.classifier[2].weight.data[:min_classes].copy_(cls_weight[:min_classes])
                        self.classifier[2].bias.data[:min_classes].copy_(cls_bias[:min_classes])
                        print(f"Copied weights for {min_classes} classes")
                else:
                    print("Incompatible classifier dimensions. Using only feature weights.")
            
            print("✅ Successfully loaded DenseNet weights into CLANet")

    def forward(self, x):
        """
        Forward pass:
        1 Extract features from DenseNet backbone
        2 Refine with ALA module (adaptive lesion attention)
        3 Enhance with CSCA module (cross-scale context)
        4 Global pooling + classification
        """
        x = self.features(x)
        x = self.ala(x)
        x = self.csca(x)
        x = self.classifier(x)
        return x
