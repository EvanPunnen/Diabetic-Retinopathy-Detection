import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
import pandas as pd
from dataset import FundusDataset, get_transforms
from modelzoo import build_classifier
import torch.nn.functional as F

# Labels for visualization
LABEL_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

def denormalize(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def plot_prediction(img, true_label, pred_label, probabilities, title=None):
    """Plot a single image with its true and predicted labels"""
    # Convert tensor to numpy and denormalize
    if torch.is_tensor(img):
        img = denormalize(img)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot image with labels
    ax1.imshow(img)
    ax1.axis('off')
    status = "✓" if true_label == pred_label else "✗"
    ax1.set_title(f'{status} True: {LABEL_NAMES[true_label]}\nPred: {LABEL_NAMES[pred_label]}')
    
    # Plot probability bars
    bars = ax2.bar(range(len(probabilities)), probabilities)
    ax2.set_xticks(range(len(probabilities)))
    ax2.set_xticklabels(LABEL_NAMES.values(), rotation=45)
    ax2.set_ylim(0, 1)
    ax2.set_title('Class Probabilities')
    
    # Color the bars
    for idx, bar in enumerate(bars):
        if idx == true_label:
            bar.set_color('green')
        elif idx == pred_label:
            bar.set_color('red')
        else:
            bar.set_color('gray')
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig

def main():
    # Configuration
    val_dir = "data/iDRID/images/val"
    val_csv = "data/iDRID/images/val_labels.csv"
    model_path = "models/best_model.pth"  # Update with your model path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory for visualizations
    viz_dir = "outputs/predictions"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load model
    model = build_classifier("densenet121", num_classes=5)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Searching for latest model file...")
        model_files = list(os.path.join(os.path.dirname(model_path), f) 
                         for f in os.listdir(os.path.dirname(model_path))
                         if f.endswith('.pth'))
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Loading model from {latest_model}")
            checkpoint = torch.load(latest_model, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("No model files found in models directory")
    
    model.eval()
    model = model.to(device)
    
    # Create dataset
    val_dataset = FundusDataset(val_dir, val_csv, transform=get_transforms(train=False))
    
    # Process random samples
    num_samples = 6
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for idx in indices:
            img, label, filename = val_dataset[idx]
            img = img.unsqueeze(0).to(device)
            
            # Get predictions
            outputs = model(img)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_label = outputs.argmax(dim=1).cpu().numpy()[0]
            
            # Plot and save
            fig = plot_prediction(
                img.cpu().squeeze(),
                label,
                pred_label,
                probabilities,
                f"File: {filename}"
            )
            save_path = os.path.join(viz_dir, f"pred_{filename}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved prediction visualization to: {save_path}")

    # Print overall statistics
    print("\nProcessing entire validation set...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_files = []
    
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            img, label, filename = val_dataset[idx]
            img = img.unsqueeze(0).to(device)
            outputs = model(img)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = outputs.argmax(dim=1).cpu().numpy()[0]
            
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probabilities)
            all_files.append(filename)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate per-class metrics
    print("\nPer-class Performance:")
    print("-------------------")
    for class_idx in range(5):
        class_mask = (all_labels == class_idx)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_preds[class_mask] == all_labels[class_mask])
            print(f"{LABEL_NAMES[class_idx]}:")
            print(f"  Total samples: {np.sum(class_mask)}")
            print(f"  Accuracy: {class_acc:.2%}")
    
    # Show worst misclassifications
    print("\nWorst Misclassifications:")
    print("----------------------")
    errors = []
    for pred, true, probs, fname in zip(all_preds, all_labels, all_probs, all_files):
        if pred != true:
            conf = probs[pred]  # confidence in wrong prediction
            errors.append((conf, abs(pred - true), fname, true, pred))
    
    # Sort by confidence in wrong prediction and severity of error
    errors.sort(key=lambda x: (-x[0], -x[1]))
    
    for conf, severity, fname, true, pred in errors[:5]:
        print(f"File: {fname}")
        print(f"True: {LABEL_NAMES[true]}")
        print(f"Predicted: {LABEL_NAMES[pred]} (confidence: {conf:.2%})")
        print(f"Severity error: {severity} levels")
        print()

if __name__ == "__main__":
    main()