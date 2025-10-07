import os
import torch
from torch.utils.data import DataLoader
from dataset import FundusDataset, get_transforms
from modelzoo import build_classifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path, test_dir, test_csv, device='cuda'):
    # Load model
    print(f"Loading model from {model_path}")
    model = build_classifier(backbone="densenet121", num_classes=5, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    # Handle checkpoint format from modelzoo.py
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Create test dataset
    test_ds = FundusDataset(test_dir, labels_csv=test_csv, transform=get_transforms(False))
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Print classification report
    print("\nClassification Report:")
    print("=====================")
    print(classification_report(all_labels, all_preds, 
                              target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']))

    # Calculate per-class accuracy
    class_accuracy = {}
    for i in range(5):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean()
            class_accuracy[i] = acc

    print("\nPer-Class Accuracy:")
    print("=================")
    for cls, acc in class_accuracy.items():
        print(f"Class {cls}: {acc*100:.2f}%")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', 
                xticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                yticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png')
    print("\nConfusion matrix saved to outputs/confusion_matrix.png")

    return {
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'class_accuracy': class_accuracy,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    model_path = "models/densenet121_best.pth"
    test_dir = "data/iDRID/B.%20Disease%20Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    test_csv = "data/iDRID/B.%20Disease%20Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    
    results = evaluate_model(model_path, test_dir, test_csv)