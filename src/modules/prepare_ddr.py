import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_DIR = "data/DDR/DR_grading/DR_grading"
CSV_PATH = "data/DDR/DR_grading.csv"
OUT_DIR = "data/DDR/images"

os.makedirs(f"{OUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/val", exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Expected columns: image_name, label
df.columns = ["image", "label"]

valid = []
for _, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, row["image"])
    if os.path.exists(img_path):
        valid.append({"image": row["image"], "label": int(row["label"]), "path": img_path})

df = pd.DataFrame(valid)

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

for _, r in train_df.iterrows():
    shutil.copy(r["path"], f"{OUT_DIR}/train/{r['image']}")

for _, r in val_df.iterrows():
    shutil.copy(r["path"], f"{OUT_DIR}/val/{r['image']}")

train_df[["image","label"]].to_csv(f"{OUT_DIR}/train_labels.csv", index=False)
val_df[["image","label"]].to_csv(f"{OUT_DIR}/val_labels.csv", index=False)

print("✅ DDR dataset prepared")
