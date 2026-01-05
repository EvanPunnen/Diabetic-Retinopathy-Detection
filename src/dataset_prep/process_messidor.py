# ============================================================
# PROCESS MESSIDOR DATASET (TRAIN / VAL SPLIT)
# ============================================================

import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------- PATHS ----------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_MESSIDOR = PROJECT_ROOT / "data" / "raw" / "messidor"
OUT_MESSIDOR = PROJECT_ROOT / "data" / "processed" / "messidor"

IMG_DIR = RAW_MESSIDOR / "images"
LABELS_CSV = RAW_MESSIDOR / "labels.csv"  # columns: image, label

# ---------------- SETUP ----------------
OUT_MESSIDOR.mkdir(parents=True, exist_ok=True)
(OUT_MESSIDOR / "train").mkdir(exist_ok=True)
(OUT_MESSIDOR / "val").mkdir(exist_ok=True)

# ---------------- LOAD LABELS ----------------
df = pd.read_csv(LABELS_CSV)

assert {"image", "label"}.issubset(df.columns), "Messidor labels.csv must have image,label"

# ---------------- STRATIFIED SPLIT ----------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# ---------------- COPY FILES ----------------
def copy_images(split_df, split_name):
    for _, row in split_df.iterrows():
        src = IMG_DIR / row["image"]
        dst = OUT_MESSIDOR / split_name / row["image"]
        if src.exists():
            shutil.copy(src, dst)

copy_images(train_df, "train")
copy_images(val_df, "val")

# ---------------- SAVE CSVs ----------------
train_df.to_csv(OUT_MESSIDOR / "train_labels.csv", index=False)
val_df.to_csv(OUT_MESSIDOR / "val_labels.csv", index=False)

print("✅ Messidor dataset processed successfully")
print("Train samples:", len(train_df))
print("Val samples:", len(val_df))
