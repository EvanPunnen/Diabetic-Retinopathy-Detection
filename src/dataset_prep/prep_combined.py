from pathlib import Path
import shutil

# Project root
project_root = Path(__file__).parent.parent.parent
BASE = project_root / "data/processed/combined"

def flatten(split):
    split_dir = BASE / split
    if not split_dir.exists():
        print(f"⚠️ {split} does not exist, skipping")
        return

    moved = 0
    for item in split_dir.iterdir():
        if item.is_dir():  # class folder
            for img in item.iterdir():
                target = split_dir / img.name
                if target.exists():
                    print(f"⚠️ Skipping existing: {img.name}")
                    continue
                shutil.move(str(img), str(target))
                moved += 1
            item.rmdir()

    if moved > 0:
        print(f"✅ {split} flattened ({moved} images moved)")
    else:
        print(f"ℹ️ {split} already flat")

# Flatten only val and test
flatten("val")
flatten("test")
