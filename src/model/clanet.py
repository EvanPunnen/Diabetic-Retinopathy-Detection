from torchvision import models
import torch.nn as nn
import torch
import os
from modules.ala import ALA_Module  

class CLANet_DenseNet(nn.Module):
    def __init__(self, num_classes=5, pretrained_weights=None):
        super(CLANet_DenseNet, self).__init__()
        
        base = models.densenet121(weights="IMAGENET1K_V1")
        self.features = base.features
        self.ala = ALA_Module(in_channels=1024)   
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
        
        # Load pretrained weights if specified
        if pretrained_weights:
            if not os.path.exists(pretrained_weights):
                pretrained_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', pretrained_weights)
            
            if not os.path.exists(pretrained_weights):
                raise FileNotFoundError(f"Cannot find pretrained weights at {pretrained_weights}")
                
            print(f"Loading pretrained weights from {pretrained_weights}")
            
            # Load the state dict
            checkpoint = torch.load(pretrained_weights, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
                print(f"Loaded model checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                state_dict = checkpoint
                print("Loaded model weights directly")
            
            # Extract and load feature weights from DenseNet
            features_dict = {}
            for key, value in state_dict.items():
                if key.startswith('features'):
                    # Remove the 'features.' prefix for loading to self.features
                    features_dict[key.replace('features.', '')] = value
            
            # Load feature weights
            print("Loading feature weights...")
            self.features.load_state_dict(features_dict, strict=False)
            print("Features loaded successfully")
            
            # Check if classifier weights exist and have compatible shapes
            if 'classifier.weight' in state_dict and 'classifier.bias' in state_dict:
                cls_weight = state_dict['classifier.weight']
                cls_bias = state_dict['classifier.bias']
                
                print(f"Found classifier weights with shape: {cls_weight.shape}")
                print(f"Our classifier expects shape: {self.classifier[2].weight.shape}")
                
                # Check if dimensions are compatible
                if cls_weight.shape[1] == self.classifier[2].weight.shape[1]:
                    # Direct mapping possible (first dimension can be different)
                    print("Direct mapping of classifier weights...")
                    if cls_weight.shape[0] == self.classifier[2].weight.shape[0]:
                        self.classifier[2].weight.data.copy_(cls_weight)
                        self.classifier[2].bias.data.copy_(cls_bias)
                    else:
                        # If number of classes is different, copy just the common classes
                        min_classes = min(cls_weight.shape[0], self.classifier[2].weight.shape[0])
                        self.classifier[2].weight.data[:min_classes].copy_(cls_weight[:min_classes])
                        self.classifier[2].bias.data[:min_classes].copy_(cls_bias[:min_classes])
                        print(f"Copied weights for {min_classes} classes")
                else:
                    if cls_weight.shape[0] == self.classifier[2].weight.shape[1]:
                        print("Found transposed weights, reshaping...")
                        self.classifier[2].weight.data.copy_(cls_weight.t())
                        self.classifier[2].bias.data.copy_(cls_bias)
                    else:
                        print("Incompatible classifier dimensions. Using only feature weights.")
            
            print("Successfully loaded DenseNet weights to CLANet")

    def forward(self, x):
        x = self.features(x)
        x = self.ala(x)
        x = self.classifier(x)
        return x