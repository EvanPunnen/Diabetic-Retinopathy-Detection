# ============================================================
# OVERALL COMBINED ACCURACY (FINAL REPORTED METRIC)
# ============================================================

import sys
from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model.clanet import CLANet_DenseNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = PROJECT_ROOT / "models" / "clanet_stage2.pth"
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "combined"

class DRDataset(Dataset):
    def __init__(self, csv, img_dir):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.img_dir / r["image"]).convert("RGB")
        return self.tf(img), int(r["label"])

dataset = DRDataset(
    DATA_ROOT / "val_labels.csv",
    DATA_ROOT / "val"
)

loader = DataLoader(dataset, batch_size=1)

model = CLANet_DenseNet(num_classes=5).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x)
        y_pred.append(out.argmax(1).item())
        y_true.append(y.item())

acc = accuracy_score(y_true, y_pred)

print("\n========== FINAL SYSTEM PERFORMANCE ==========")
print(f"Overall Combined Accuracy: {acc*100:.2f}%")
