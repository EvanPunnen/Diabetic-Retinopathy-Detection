"""
STAGE-2: Attention learning for CLANet
Dataset: DDR + Messidor (combined)
Loads Stage-1 weights
ALA and CSCA are UNFROZEN
RTX 3050 (4GB) SAFE
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from model.clanet import CLANet_DenseNet

# ===============================
# CUDA SAFETY
# ===============================
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# ===============================
# CONFIG (STAGE-2)
# ===============================
CONFIG = {
    "data_root": project_root / "data/processed/combined",
    "model_dir": project_root / "models",
    "stage1_ckpt": project_root / "models/clanet_stage1.pth",
    "num_classes": 5,
    "batch_size": 2,
    "epochs": 20,
    "lr": 1e-5,               # lower LR for attention
    "weight_decay": 1e-4,
    "image_size": 224,
    "num_workers": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "early_stopping": 7
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
        img = Image.open(self.img_dir / row["image"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label

# ===============================
# TRANSFORMS
# ===============================
def train_tf(size):
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def val_tf(size):
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

# ===============================
# TRAIN / VALIDATE (AMP)
# ===============================
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    loss_sum, correct, total = 0, 0, 0

    for x,y in tqdm(loader, desc="Training"):
        x,y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum/len(loader), 100*correct/total

def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():
        for x,y in tqdm(loader, desc="Validating"):
            x,y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
            out = model(x)
            loss = criterion(out,y)
            loss_sum += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum/len(loader), 100*correct/total

# ===============================
# MAIN
# ===============================
def main():
    print("\n===== STAGE-2: CLANet Attention Training =====")
    print("Device:", CONFIG["device"])

    assert CONFIG["stage1_ckpt"].exists(), "❌ Stage-1 checkpoint not found"
    CONFIG["model_dir"].mkdir(exist_ok=True)

    train_ds = DRDataset(CONFIG["data_root"] / "train_labels.csv",
                         CONFIG["data_root"] / "train",
                         train_tf(CONFIG["image_size"]))
    val_ds = DRDataset(CONFIG["data_root"] / "val_labels.csv",
                       CONFIG["data_root"] / "val",
                       val_tf(CONFIG["image_size"]))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"],
                              pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=CONFIG["num_workers"],
                            pin_memory=False)

    model = CLANet_DenseNet(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["stage1_ckpt"], map_location=CONFIG["device"]))

    # 🔓 UNFREEZE ALA + CSCA
    for p in model.ala.parameters(): p.requires_grad = True
    for p in model.csca.parameters(): p.requires_grad = True

    # 🔒 Keep BatchNorm stable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()

    best, patience = 0, 0

    for e in range(CONFIG["epochs"]):
        print(f"\nEpoch {e+1}/{CONFIG['epochs']}")
        tr_l,tr_a = train_epoch(model,train_loader,criterion,optimizer,scaler)
        va_l,va_a = validate(model,val_loader,criterion)

        print(f"Train Acc: {tr_a:.2f}% | Val Acc: {va_a:.2f}%")

        if va_a > best:
            best = va_a
            patience = 0
            torch.save(model.state_dict(), CONFIG["model_dir"]/ "clanet_stage2.pth")
            print("✓ Saved best model")
        else:
            patience += 1
            if patience >= CONFIG["early_stopping"]:
                print("Early stopping")
                break

    print(f"\nStage-2 done. Best Val Acc: {best:.2f}%")

if __name__ == "__main__":
    main()
