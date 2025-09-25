# src/modelzoo.py
import torch
import torch.nn as nn
from torchvision import models

def build_classifier(backbone="densenet121", num_classes=5, pretrained=True):
    if backbone == "densenet121":
        net = models.densenet121(pretrained=pretrained)
        in_features = net.classifier.in_features
        net.classifier = nn.Linear(in_features, num_classes)
    elif backbone == "resnet50":
        net = models.resnet50(pretrained=pretrained)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("backbone not supported")
    return net

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    return model, ckpt.get('epoch', 0)
