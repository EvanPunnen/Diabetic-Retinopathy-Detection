"""
STAGE-3: Clinical fine-tuning for CLANet (RTX 3050 SAFE)
Dataset: IDRiD ONLY
Loads Stage-2 weights
Trains ALA + classifier ONLY
"""

# ===============================
# EXECUTION SAFETY
# ===============================
print(">>> STAGE-3 SCRIPT STARTED <<<")

import sys
from pathlib import Path
import torch

torch.multiprocessing.set_start_method("spawn", force=True)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# ===============================
# IMPORTS
# ===============================
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from model.clanet import CLANet_DenseNet

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# ===============================
# CONFIG
# ===============================
CONFIG = {
    "data_root": project_root / "data/iDRID/images",
    "model_dir": project_root / "models",
    "stage2_ckpt": project_root / "models/clanet_stage2.pth",
    "num_classes": 5,
    "batch_size": 1,
    "accum_steps": 2,        # SAFE
    "epochs": 25,
    "lr": 5e-6,
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "early_stopping": 6
}

assert CONFIG["data_root"].exists()
assert CONFIG["stage2_ckpt"].exists()

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
        img = Image.open(self.img_dir / row["image"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label

# ===============================
# TRANSFORMS (LOW AUG)
# ===============================
def train_tf(size):
    return transforms.Compose([
        transforms.Resize((size + 16, size + 16)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

def val_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

# ===============================
# FOCAL LOSS
# ===============================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        ce = self.ce(inputs, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ===============================
# TRAIN / VALIDATE
# ===============================
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    optimizer.zero_grad()
    correct, total = 0, 0

    for step, (x, y) in enumerate(tqdm(loader, desc="Training")):
        x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])

        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out, y) / CONFIG["accum_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % CONFIG["accum_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100 * correct / total

def validate(model, loader, criterion):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# ===============================
# MAIN
# ===============================
def main():
    print("\n===== STAGE-3: IDRiD Fine-Tuning =====")
    print("Device:", CONFIG["device"])

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

    model = CLANet_DenseNet(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["stage2_ckpt"], map_location=CONFIG["device"]))

    # 🔒 FREEZE EVERYTHING
    for p in model.parameters():
        p.requires_grad = False

    # 🔓 TRAIN ONLY ALA + CLASSIFIER
    for p in model.ala.parameters(): p.requires_grad = True
    for p in model.classifier.parameters(): p.requires_grad = True

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"]
    )
    scaler = torch.amp.GradScaler("cuda")

    best, patience = 0, 0

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        va_acc = validate(model, val_loader, criterion)

        print(f"Train Acc: {tr_acc:.2f}% | Val Acc: {va_acc:.2f}%")

        if va_acc > best:
            best = va_acc
            patience = 0
            torch.save(model.state_dict(),
                       CONFIG["model_dir"] / "clanet_stage3_final.pth")
            print("✓ Saved FINAL model")
        else:
            patience += 1
            if patience >= CONFIG["early_stopping"]:
                print("Early stopping")
                break

    print(f"\nFINAL STAGE COMPLETE — Best Val Acc: {best:.2f}%")

# ===============================
# ENTRY
# ===============================
if __name__ == "__main__":
    main()
