# ============================================================
# DATASET-WISE ACCURACY EVALUATION (CLANet)
# ============================================================

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

# ============================================================
# PATH SETUP (MATCHES YOUR REPO)
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model.clanet import CLANet_DenseNet

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = PROJECT_ROOT / "models" / "clanet_stage2.pth"

DATASETS = {
    "DDR": PROJECT_ROOT / "data" / "processed" / "ddr",
    "Messidor": PROJECT_ROOT / "data" / "processed" / "messidor",
    "IDRiD": PROJECT_ROOT / "data" / "iDRID" / "images"
}

CLASS_NAMES = [
    "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"
]

# ============================================================
# DATASET CLASS
# ============================================================
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(self.img_dir / r["image"]).convert("RGB")
        return self.tf(img), int(r["label"])

# ============================================================
# LOAD MODEL
# ============================================================
model = CLANet_DenseNet(num_classes=5).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("\n========== DATASET-WISE PERFORMANCE ==========")

# ============================================================
# EVALUATION LOOP
# ============================================================
for name, path in DATASETS.items():
    csv_file = path / "val_labels.csv"
    img_dir = path / "val"

    if not csv_file.exists():
        print(f"\n{name}: SKIPPED (validation set missing)")
        continue

    dataset = DRDataset(csv_file, img_dir)
    loader = DataLoader(dataset, batch_size=1)

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            y_pred.append(out.argmax(1).item())
            y_true.append(y.item())

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    print(f"\n{name}")
    print(f"Accuracy          : {acc*100:.2f}%")
    print(f"Balanced Accuracy : {bal_acc*100:.2f}%")
    print(f"Quadratic Kappa   : {kappa:.4f}")
