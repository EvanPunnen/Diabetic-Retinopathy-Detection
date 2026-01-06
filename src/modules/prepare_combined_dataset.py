"""
Prepare combined MESSIDOR-2 and DDR datasets for training.
Creates a unified dataset structure with train/val/test splits.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

random.seed(42)

# Paths
MESSIDOR_DIR = Path("data/messidor-2")
DDR_DIR = Path("data/ddr")
OUTPUT_DIR = Path("data/processed/combined")

# DR Classes (0-4 scale)
DR_CLASSES = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate", 
    3: "Severe",
    4: "Proliferative_DR"
}


def prepare_messidor():
    """Load and prepare MESSIDOR-2 dataset."""
    print("\n--- Preparing MESSIDOR-2 ---")
    
    # Find label file
    label_files = list(MESSIDOR_DIR.glob("*.csv"))
    if not label_files:
        print("No CSV label file found for MESSIDOR-2")
        return []
    
    df = pd.read_csv(label_files[0])
    print(f"Loaded {len(df)} records from {label_files[0].name}")
    
    # Identify columns
    image_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if "id_code" in col_lower or "image" in col_lower or "file" in col_lower:
            image_col = col
        elif "diagnosis" in col_lower or "grade" in col_lower or "label" in col_lower:
            label_col = col
    
    if not image_col or not label_col:
        print(f"Could not identify columns. Available: {df.columns.tolist()}")
        return []
    
    print(f"Using columns: image='{image_col}', label='{label_col}'")
    
    # Find images
    image_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
    image_dict = {}
    for ext in image_extensions:
        for img in MESSIDOR_DIR.rglob(ext):
            image_dict[img.stem] = img
            # Also try with extension
            image_dict[img.name] = img
    
    print(f"Found {len(image_dict)} images")
    
    # Match images with labels
    data = []
    for _, row in df.iterrows():
        img_name = str(row[image_col])
        img_stem = Path(img_name).stem
        
        img_path = image_dict.get(img_name) or image_dict.get(img_stem)
        
        if img_path and img_path.exists():
            label = int(row[label_col])
            if 0 <= label <= 4:  # Valid DR grade
                data.append({
                    "image_path": img_path,
                    "label": label,
                    "source": "messidor"
                })
    
    print(f"Matched {len(data)} images with valid labels")
    return data


def prepare_ddr():
    """Load and prepare DDR dataset."""
    print("\n--- Preparing DDR ---")
    
    if not DDR_DIR.exists():
        print(f"DDR directory not found: {DDR_DIR}")
        return []
    
    data = []
    image_extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    
    # Check for folder-based structure (train/val/test with class subfolders)
    for split in ["train", "test", "val", "valid", "training", "testing", "validation"]:
        split_dir = DDR_DIR / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    try:
                        label = int(class_dir.name)
                        if 0 <= label <= 4:
                            for img_path in class_dir.iterdir():
                                if img_path.suffix.lower() in image_extensions:
                                    data.append({
                                        "image_path": img_path,
                                        "label": label,
                                        "source": "ddr"
                                    })
                    except ValueError:
                        # Folder name is not a number, try mapping
                        label_map = {
                            "no_dr": 0, "nodr": 0, "normal": 0, "0": 0,
                            "mild": 1, "1": 1,
                            "moderate": 2, "2": 2,
                            "severe": 3, "3": 3,
                            "proliferative": 4, "pdr": 4, "4": 4
                        }
                        label = label_map.get(class_dir.name.lower())
                        if label is not None:
                            for img_path in class_dir.iterdir():
                                if img_path.suffix.lower() in image_extensions:
                                    data.append({
                                        "image_path": img_path,
                                        "label": label,
                                        "source": "ddr"
                                    })
    
    # Check for CSV-based labels if no folder structure found
    if not data:
        label_files = list(DDR_DIR.rglob("*.csv")) + list(DDR_DIR.rglob("*.txt"))
        for label_file in label_files:
            try:
                if label_file.suffix == ".csv":
                    df = pd.read_csv(label_file)
                else:
                    df = pd.read_csv(label_file, sep=None, engine='python')
                
                # Try to find image and label columns
                image_col = None
                label_col = None
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if "image" in col_lower or "file" in col_lower or "name" in col_lower:
                        image_col = col
                    elif "label" in col_lower or "grade" in col_lower or "diagnosis" in col_lower:
                        label_col = col
                
                if image_col and label_col:
                    image_dict = {}
                    for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
                        for img in DDR_DIR.rglob(ext):
                            image_dict[img.stem] = img
                            image_dict[img.name] = img
                    
                    for _, row in df.iterrows():
                        img_name = str(row[image_col])
                        img_stem = Path(img_name).stem
                        img_path = image_dict.get(img_name) or image_dict.get(img_stem)
                        
                        if img_path:
                            label = int(row[label_col])
                            if 0 <= label <= 4:
                                data.append({
                                    "image_path": img_path,
                                    "label": label,
                                    "source": "ddr"
                                })
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
    
    print(f"Found {len(data)} DDR images with valid labels")
    return data


def create_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create stratified train/val/test splits."""
    # Group by label for stratified split
    by_label = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
    
    return train_data, val_data, test_data


def copy_files(data, split_name):
    """Copy files to output directory."""
    for item in data:
        label_dir = OUTPUT_DIR / split_name / str(item["label"])
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Add source prefix to avoid name conflicts
        new_name = f"{item['source']}_{item['image_path'].name}"
        dest = label_dir / new_name
        
        if not dest.exists():
            shutil.copy2(item["image_path"], dest)


def print_statistics(train_data, val_data, test_data):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        print(f"\n{split_name}: {len(split_data)} images")
        
        # Count by label
        label_counts = defaultdict(int)
        source_counts = defaultdict(int)
        for item in split_data:
            label_counts[item["label"]] += 1
            source_counts[item["source"]] += 1
        
        print("  By class:")
        for label in sorted(label_counts.keys()):
            print(f"    {label} ({DR_CLASSES.get(label, 'Unknown')}): {label_counts[label]}")
        
        print("  By source:")
        for source, count in source_counts.items():
            print(f"    {source}: {count}")


def prepare_combined_dataset():
    """Main function to prepare combined dataset."""
    print("=" * 60)
    print("Preparing Combined MESSIDOR-2 + DDR Dataset")
    print("=" * 60)
    
    # Clear output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # Prepare individual datasets
    messidor_data = prepare_messidor()
    ddr_data = prepare_ddr()
    
    # Combine
    all_data = messidor_data + ddr_data
    print(f"\nTotal combined: {len(all_data)} images")
    
    if len(all_data) == 0:
        raise ValueError("No data found! Check your dataset paths.")
    
    # Create splits
    train_data, val_data, test_data = create_splits(all_data)
    
    # Print statistics
    print_statistics(train_data, val_data, test_data)
    
    # Copy files
    print("\nCopying files...")
    copy_files(train_data, "train")
    copy_files(val_data, "val")
    copy_files(test_data, "test")
    
    # Save metadata
    metadata = {
        "train": [{"image": f"{item['source']}_{item['image_path'].name}", "label": item["label"]} for item in train_data],
        "val": [{"image": f"{item['source']}_{item['image_path'].name}", "label": item["label"]} for item in val_data],
        "test": [{"image": f"{item['source']}_{item['image_path'].name}", "label": item["label"]} for item in test_data],
    }
    
    for split_name, split_meta in metadata.items():
        df = pd.DataFrame(split_meta)
        df.to_csv(OUTPUT_DIR / f"{split_name}_labels.csv", index=False)
    
    print(f"\nDataset prepared at: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_combined_dataset()