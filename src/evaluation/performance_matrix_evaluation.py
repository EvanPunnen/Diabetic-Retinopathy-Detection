# ============================================================
# PERFORMANCE MATRIX + FIGURE GENERATION (FIXED)
# ============================================================

print(">>> GENERATING RESULT FIGURES <<<", flush=True)

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# ============================================================
# PATH SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model.clanet import CLANet_DenseNet

RESULTS_DIR = PROJECT_ROOT / "src" / "results" / "final_stage"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Results will be saved to:", RESULTS_DIR, flush=True)

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = PROJECT_ROOT / "models" / "clanet_stage2.pth"  # CHANGE if needed
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "combined"

CLASS_NAMES = [
    "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"
]

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_ROOT / "val_labels.csv")

images_dir = DATA_ROOT / "val"

# ============================================================
# LOAD MODEL
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLANet_DenseNet(num_classes=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================================================
# INFERENCE
# ============================================================
y_true, y_pred = [], []

from torchvision import transforms
from PIL import Image

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

with torch.no_grad():
    for _, row in df.iterrows():
        img = Image.open(images_dir / row["image"]).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        out = model(x)
        y_pred.append(out.argmax(1).item())
        y_true.append(int(row["label"]))

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

cm_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ Saved:", cm_path)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
report = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, digits=4
)

report_path = RESULTS_DIR / "classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)

print("✔ Saved:", report_path)

# ============================================================
# PER-CLASS ACCURACY BAR PLOT
# ============================================================
per_class_acc = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(8,5))
plt.bar(CLASS_NAMES, per_class_acc)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.xticks(rotation=30)

bar_path = RESULTS_DIR / "per_class_accuracy.png"
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()

print("✔ Saved:", bar_path)

print("\n>>> ALL RESULTS GENERATED SUCCESSFULLY <<<")
