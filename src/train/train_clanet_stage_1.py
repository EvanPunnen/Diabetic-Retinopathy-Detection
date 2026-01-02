"""
STAGE-1: Representation learning for CLANet
Dataset: DDR + Messidor (combined)
IDRiD must NOT be used here
ALA and CSCA are FROZEN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import json
from datetime import datetime

from model.clanet import CLANet_DenseNet


# ===============================
# CONFIG
# ===============================
CONFIG = {
    "data_root": Path("data/processed/combined"),  # DDR + Messidor combined
    "model_dir": Path("models"),
    "num_classes": 5,
    "batch_size": 16,
    "epochs": 40,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "image_size": 224,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "early_stopping": 10
}


# ===============================
# DATASET
# ===============================
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image"]
        label = int(row["label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ===============================
# TRANSFORMS
# ===============================
def get_train_transforms(size):
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ===============================
# TRAIN / VALIDATE
# ===============================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), 100. * correct / total


# ===============================
# MAIN
# ===============================
def main():
    print("\n===== STAGE-1: CLANet Training (DDR + Messidor) =====")
    print(f"Device: {CONFIG['device']}")

    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = DRDataset(
        CONFIG["data_root"] / "train_labels.csv",
        CONFIG["data_root"] / "train",
        get_train_transforms(CONFIG["image_size"])
    )

    val_dataset = DRDataset(
        CONFIG["data_root"] / "val_labels.csv",
        CONFIG["data_root"] / "val",
        get_val_transforms(CONFIG["image_size"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    # Model
    model = CLANet_DenseNet(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])

    # FREEZE ALA + CSCA (Stage-1 requirement)
    for p in model.ala.parameters():
        p.requires_grad = False
    for p in model.csca.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    best_acc, patience = 0.0, 0
    history = []

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG["device"]
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG["device"]
        )

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        history.append({
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), CONFIG["model_dir"] / "clanet_stage1.pth")
            print("✓ Saved best model")
        else:
            patience += 1

        if patience >= CONFIG["early_stopping"]:
            print("Early stopping triggered")
            break

    with open(CONFIG["model_dir"] / "history_stage1.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nStage-1 complete. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
