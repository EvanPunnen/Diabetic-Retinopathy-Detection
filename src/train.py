import os
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from dataset import FundusDataset, get_transforms
from modelzoo import build_classifier, save_checkpoint
from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np

def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    losses = []
    all_preds, all_labels = [], []
    optimizer.zero_grad()
    
    for i, (imgs, labels, filenames) in enumerate(tqdm(loader)):
        # Validate labels
        if torch.any((labels < 0) | (labels >= 5)):
            invalid_indices = torch.where((labels < 0) | (labels >= 5))[0]
            print(f"\nWarning: Invalid labels found: {labels[invalid_indices].tolist()}")
            print(f"Files with invalid labels: {[filenames[idx] for idx in invalid_indices]}")
            # Fix invalid labels to class 0
            labels[labels < 0] = 0
            labels[labels >= 5] = 0
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights after accumulation steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Record metrics
        losses.append(loss.item() * accumulation_steps)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Print GPU memory usage every 10 iterations
        if torch.cuda.is_available() and i % 10 == 0:
            print(f"\rGPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB", end="")
    
    # Calculate final metrics
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = np.mean(losses)
    
    print(f"\nTraining - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    return avg_loss, acc

def val_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_labels = [], []
    all_files = []  # Track filenames
    
    print("\nStarting validation...")
    with torch.no_grad():
        for batch_idx, (imgs, labels, filenames) in enumerate(tqdm(loader)):
            # Validate labels
            if torch.any((labels < 0) | (labels >= 5)):
                invalid_indices = torch.where((labels < 0) | (labels >= 5))[0]
                print(f"\nWarning: Invalid labels found: {labels[invalid_indices].tolist()}")
                print(f"Files with invalid labels: {[filenames[idx] for idx in invalid_indices]}")
                labels[labels < 0] = 0
                labels[labels >= 5] = 0
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            # Debug output for first few batches
            if batch_idx < 2:  # Show first 2 batches
                print(f"\nBatch {batch_idx+1} predictions:")
                for i, (pred, prob) in enumerate(zip(preds, probs.cpu().numpy())):
                    print(f"Image: {filenames[i]}")
                    print(f"True label: {labels[i].item()}")
                    print(f"Predicted: {pred}")
                    print("Class probabilities:", {i: f"{p:.3f}" for i, p in enumerate(prob)})
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_files.extend(filenames)
    
    # Detailed analysis of predictions and labels
    print("\n" + "="*50)
    print("VALIDATION ANALYSIS")
    print("="*50)
    
    # Overall statistics
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    
    print("\nPrediction distribution:")
    for label, count in zip(unique_preds, pred_counts):
        print(f"Class {label}: {count} predictions ({count/len(all_preds)*100:.1f}%)")
    
    print("\nTrue label distribution:")
    for label, count in zip(unique_labels, label_counts):
        print(f"Class {label}: {count} samples ({count/len(all_labels)*100:.1f}%)")
    
    # Check for potential issues
    if len(unique_preds) == 1:
        print("\nWARNING: Model is predicting only one class!")
        pred_class = unique_preds[0]
        print(f"All predictions are class {pred_class}")
        
    if len(unique_labels) == 1:
        print("\nWARNING: Validation set contains only one class!")
        label_class = unique_labels[0]
        print(f"All true labels are class {label_class}")
    
    # Print some misclassifications if any
    if not np.array_equal(all_preds, all_labels):
        print("\nMisclassified examples:")
        misclassified = [(f, t, p) for f, t, p in zip(all_files, all_labels, all_preds) if t != p]
        for filename, true, pred in misclassified[:5]:
            print(f"File: {filename}, True: {true}, Predicted: {pred}")
    
    acc = accuracy_score(all_labels, all_preds)
    try:
        # Force kappa calculation with all possible labels
        kappa = cohen_kappa_score(
            all_labels, all_preds,
            weights='quadratic',
            labels=[0, 1, 2, 3, 4]
        )
    except Exception as e:
        print(f"\nWarning: Could not compute kappa score: {str(e)}")
        print("Label counts:", np.bincount(all_labels.astype(int), minlength=5))
        print("Prediction counts:", np.bincount(all_preds.astype(int), minlength=5))
        kappa = float('nan')
    
    return np.mean(losses), acc, kappa

def main():
    import gc
    gc.collect()  # Collect garbage first
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Set environment variables before importing torch
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    # Initialize CUDA settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Conservative memory settings
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Set to use only 50% of available GPU memory
        torch.cuda.set_per_process_memory_fraction(0.5)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--backbone", default="densenet121")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)  # Reduced batch size
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", default="./models")
    parser.add_argument("--accumulation_steps", type=int, default=2)  # Gradient accumulation
    args = parser.parse_args()

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("-" * 50)
        print("ERROR: CUDA is not available!")
        print("Training on CPU is disabled. Please ensure you have:")
        print("1. An NVIDIA GPU")
        print("2. CUDA drivers installed")
        print("3. PyTorch installed with CUDA support")
        print("-" * 50)
        return

    device = "cuda"
    print("-" * 50)
    print("Training Setup Information:")
    print("-" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    print(f"Batch Size: {args.batch}")
    print(f"Number of Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Backbone: {args.backbone}")
    print("-" * 50)

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = FundusDataset(args.train_dir, labels_csv=args.train_csv, transform=get_transforms(True))
    val_ds = FundusDataset(args.val_dir, labels_csv=args.val_csv, transform=get_transforms(False))

    # Conservative DataLoader settings
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,  # No multiprocessing initially
        pin_memory=False,  # Disable pin_memory
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,  # No multiprocessing initially
        pin_memory=False,  # Disable pin_memory
        persistent_workers=False
    )

    model = build_classifier(backbone=args.backbone, num_classes=5, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_kappa = -1
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args.accumulation_steps)
        print(f"\nTrain loss {train_loss:.4f} acc {train_acc:.4f}")
        val_loss, val_acc, val_kappa = val_epoch(model, val_loader, criterion, device)
        print(f"Val loss {val_loss:.4f} acc {val_acc:.4f} kappa {val_kappa:.4f}")

        ckpt_path = os.path.join(args.out_dir, f"{args.backbone}_epoch{epoch}.pth")
        save_checkpoint(model, optimizer, epoch, ckpt_path)

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_path = os.path.join(args.out_dir, f"{args.backbone}_best.pth")
            save_checkpoint(model, optimizer, epoch, best_path)
            print("Saved best model:", best_path)

if __name__ == "__main__":
    main()

