import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score
)

from model.clanet import CLANet_DenseNet

# ==========================
# CONFIG
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/clanet_stage2.pth"   # Stage-2 model
DATA_ROOT = "data/processed/combined"     # DDR + Messidor
NUM_CLASSES = 5
BATCH_SIZE = 4

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# ==========================
# DATASET
# ==========================
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

# ==========================
# TRANSFORMS
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# LOAD DATA
# ==========================
val_dataset = DRDataset(
    csv_file=f"{DATA_ROOT}/val_labels.csv",
    img_dir=f"{DATA_ROOT}/val",
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# LOAD MODEL
# ==========================
model = CLANet_DenseNet(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==========================
# INFERENCE
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==========================
# METRICS
# ==========================
acc = accuracy_score(all_labels, all_preds)
bal_acc = balanced_accuracy_score(all_labels, all_preds)
kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
cm = confusion_matrix(all_labels, all_preds)

print("\n================ STAGE-2 METRICS =================")
print(f"Overall Accuracy      : {acc*100:.2f}%")
print(f"Balanced Accuracy     : {bal_acc*100:.2f}%")
print(f"Cohen's Kappa (Quad)  : {kappa:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES,
    digits=4
))

print("\n============== CONFUSION MATRIX ==============")
print(cm)

# ==========================
# PER-CLASS ACCURACY
# ==========================
print("\n========== PER-CLASS ACCURACY ==========")
for i, name in enumerate(CLASS_NAMES):
    class_idx = np.where(all_labels == i)[0]
    if len(class_idx) == 0:
        print(f"{name}: N/A")
    else:
        class_acc = (all_preds[class_idx] == i).mean()
        print(f"{name}: {class_acc*100:.2f}%")