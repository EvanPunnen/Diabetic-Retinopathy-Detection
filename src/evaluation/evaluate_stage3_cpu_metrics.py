print(">>> METRICS SCRIPT EXECUTED <<<", flush=True)

import sys
print("Python executable:", sys.executable, flush=True)

from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
print("Project root:", PROJECT_ROOT, flush=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model.clanet import CLANet_DenseNet

MODEL_PATH = PROJECT_ROOT / "models" / "clanet_stage3_cpu.pth"
DATA_ROOT = PROJECT_ROOT / "data" / "iDRID" / "images"

print("Model exists:", MODEL_PATH.exists(), flush=True)
print("Data exists:", DATA_ROOT.exists(), flush=True)

class DRDataset(Dataset):
    def __init__(self, csv, img_dir, tf):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.img_dir / r["image"]).convert("RGB")
        return self.tf(img), int(r["label"])

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

ds = DRDataset(
    DATA_ROOT / "val_labels.csv",
    DATA_ROOT / "val",
    tf
)

dl = DataLoader(ds, batch_size=1)

model = CLANet_DenseNet(num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

preds, labels = [], []

with torch.no_grad():
    for x,y in dl:
        o = model(x)
        preds.append(o.argmax(1).item())
        labels.append(y.item())

acc = accuracy_score(labels, preds)
print("FINAL ACCURACY:", acc*100, flush=True)
