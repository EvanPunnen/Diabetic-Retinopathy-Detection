from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data" / "DDR"

IMG_DIR = DATA / "images"
TRAIN_CSV = DATA / "train_labels.csv"
VAL_CSV = DATA / "val_labels.csv"

OUT = ROOT / "data" / "processed" / "ddr"
OUT.mkdir(parents=True, exist_ok=True)

def process_split(csv_file, split):
    df = pd.read_csv(csv_file)
    out_img = OUT / split
    out_img.mkdir(exist_ok=True)

    records = []
    for _, row in df.iterrows():
        src = IMG_DIR / row["image"]
        dst = out_img / src.name
        shutil.copy(src, dst)
        records.append({"image": dst.name, "label": row["label"]})

    pd.DataFrame(records).to_csv(OUT / f"{split}_labels.csv", index=False)

process_split(TRAIN_CSV, "train")
process_split(VAL_CSV, "val")

print("✅ DDR processed correctly")
