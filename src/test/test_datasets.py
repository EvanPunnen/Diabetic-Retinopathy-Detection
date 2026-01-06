import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

def verify_dataset(root):
    print(f"\n🔍 Verifying dataset at: {root}")
    root = Path(root)

    if not root.exists():
        print(f"❌ Root path does not exist: {root}")
        return False

    for split in ["train", "val"]:
        csv_path = root / f"{split}_labels.csv"
        img_dir = root / split

        if not csv_path.exists():
            print(f"❌ Missing {csv_path}")
            return False
        if not img_dir.exists():
            print(f"❌ Missing {img_dir}")
            return False

        df = pd.read_csv(csv_path)

        # Column check
        if list(df.columns) != ["image", "label"]:
            print(f"❌ CSV columns incorrect: {list(df.columns)}")
            return False

        # Label check - use numpy integer types too
        labels = df["label"].tolist()
        if not all(isinstance(x, (int, np.integer)) for x in labels):
            print("❌ Non-integer labels")
            return False
        if not set(labels).issubset({0, 1, 2, 3, 4}):
            print(f"❌ Invalid label values: {set(labels)}")
            return False

        # Class distribution
        print(f"\n📊 {split.upper()} class distribution:")
        dist = Counter(labels)
        for i in range(5):
            print(f"Class {i}: {dist.get(i, 0)}")
            if dist.get(i, 0) == 0:
                print(f"⚠️ Warning: Class {i} missing!")

        # Image checks
        for _, row in df.iterrows():
            img_path = img_dir / row["image"]
            if not img_path.exists():
                print(f"❌ Missing image: {img_path}")
                return False

            img = Image.open(img_path)
            if img.mode != "RGB":
                print(f"❌ Non-RGB image: {img_path}")
                return False
            w, h = img.size
            if w < 224 or h < 224:
                print(f"❌ Image too small: {img_path}")
                return False

        print(f"✅ {split} split verified successfully")

    print("\n🎯 DATASET VERIFICATION PASSED")
    return True

if __name__ == "__main__":
    # Use absolute paths based on project root
    project_root = Path(__file__).parent.parent.parent  # Go up from src/test to project root
    
    verify_dataset(project_root / "data/processed/combined")
    verify_dataset(project_root / "data/iDRID/images")
