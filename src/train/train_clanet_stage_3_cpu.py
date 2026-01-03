# ============================================================
# HARD EXECUTION PROOF (MUST BE FIRST LINE)
# ============================================================
print(">>> FILE EXECUTED: train_clanet_stage_3_cpu.py <<<", flush=True)

# ============================================================
# IMPORTS
# ============================================================
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ============================================================
# PATH SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(">>> PROJECT ROOT:", PROJECT_ROOT, flush=True)
print(">>> PYTHON EXECUTABLE:", sys.executable, flush=True)
print(">>> CURRENT WORKING DIR:", os.getcwd(), flush=True)

from model.clanet import CLANet_DenseNet

# ============================================================
# CONFIG (CPU ONLY)
# ============================================================
CONFIG = {
    "data_root": PROJECT_ROOT / "data/iDRID/images",
    "model_dir": PROJECT_ROOT / "models",
    "stage2_ckpt": PROJECT_ROOT / "models/clanet_stage2.pth",
    "num_classes": 5,
    "batch_size": 1,
    "epochs": 10,
    "lr": 5e-6,
    "image_size": 224,
    "device": "cpu",
    "early_stopping": 4
}

print(">>> CONFIG LOADED", flush=True)

# ============================================================
# SAFETY CHECKS
# ============================================================
assert CONFIG["data_root"].exists(), "IDRiD dataset path NOT FOUND"
assert CONFIG["stage2_ckpt"].exists(), "Stage-2 checkpoint NOT FOUND"

# ============================================================
# DATASET
# ============================================================
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["image"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label

# ============================================================
# TRANSFORMS
# ============================================================
def train_tf(size):
    return transforms.Compose([
        transforms.Resize((size + 16, size + 16)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

def val_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce = self.ce(inputs, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce)

# ============================================================
# TRAIN / VALIDATE
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    correct, total = 0, 0

    for x, y in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100 * correct / total

def validate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n===== STAGE-3 CPU TRAINING STARTED =====", flush=True)
    print("Device:", CONFIG["device"], flush=True)

    train_ds = DRDataset(
        CONFIG["data_root"] / "train_labels.csv",
        CONFIG["data_root"] / "train",
        train_tf(CONFIG["image_size"])
    )

    val_ds = DRDataset(
        CONFIG["data_root"] / "val_labels.csv",
        CONFIG["data_root"] / "val",
        val_tf(CONFIG["image_size"])
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = CLANet_DenseNet(num_classes=CONFIG["num_classes"])
    model.load_state_dict(torch.load(CONFIG["stage2_ckpt"], map_location="cpu"))

    # FREEZE EVERYTHING
    for p in model.parameters():
        p.requires_grad = False

    # TRAIN ONLY ALA + CLASSIFIER
    for p in model.ala.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"]
    )

    best, patience = 0, 0

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}", flush=True)
        tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        va_acc = validate(model, val_loader)

        print(f"Train Acc: {tr_acc:.2f}% | Val Acc: {va_acc:.2f}%", flush=True)

        if va_acc > best:
            best = va_acc
            patience = 0
            torch.save(
                model.state_dict(),
                CONFIG["model_dir"] / "clanet_stage3_cpu.pth"
            )
            print("✓ SAVED clanet_stage3_cpu.pth", flush=True)
        else:
            patience += 1
            if patience >= CONFIG["early_stopping"]:
                print("EARLY STOPPING", flush=True)
                break

    print(f"\nCPU STAGE-3 COMPLETE — Best Val Acc: {best:.2f}%", flush=True)

# ============================================================
# ENTRY POINT
# ============================================================
print(">>> BOTTOM OF FILE REACHED <<<", flush=True)

if __name__ == "__main__":
    print(">>> ENTERING MAIN() <<<", flush=True)
    main()
