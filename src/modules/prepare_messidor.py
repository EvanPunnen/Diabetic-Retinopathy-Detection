"""
Prepare MESSIDOR-2 dataset for diabetic retinopathy detection.
Handles various label file formats and organizes images into train/val/test splits.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path("data/messidor-2")
OUTPUT_DIR = Path("data/processed/messidor-2")

# Possible label file names in MESSIDOR-2 dataset
POSSIBLE_LABEL_FILES = [
    "messidor-2.csv",
    "labels.csv",
    "messidor_data.csv",
    "Annotation_Base11.csv",
    "Annotation_Base12.csv",
    "Annotation_Base13.csv",
    "Annotation_Base14.csv",
    "Annotation_Base21.csv",
    "Annotation_Base22.csv",
    "Annotation_Base23.csv",
    "Annotation_Base24.csv",
    "Annotation_Base31.csv",
    "Annotation_Base32.csv",
    "Annotation_Base33.csv",
    "Annotation_Base34.csv",
]

POSSIBLE_EXCEL_FILES = [
    "messidor-2.xls",
    "messidor-2.xlsx",
    "messidor_annotation.xls",
]


def find_label_file():
    """Find the label file in the dataset directory."""
    # Check for CSV files
    for filename in POSSIBLE_LABEL_FILES:
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"Found label file: {filepath}")
            return filepath, "csv"
    
    # Check for Excel files
    for filename in POSSIBLE_EXCEL_FILES:
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"Found label file: {filepath}")
            return filepath, "excel"
    
    # Search for any CSV or Excel file
    for ext in ["*.csv", "*.xls", "*.xlsx"]:
        files = list(DATA_DIR.glob(ext))
        if files:
            print(f"Found potential label file: {files[0]}")
            ext_type = "csv" if ext == "*.csv" else "excel"
            return files[0], ext_type
    
    return None, None


def load_labels():
    """Load labels from the dataset."""
    label_file, file_type = find_label_file()
    
    if label_file is None:
        # List all files in directory for debugging
        print(f"\nFiles in {DATA_DIR}:")
        if DATA_DIR.exists():
            for f in DATA_DIR.iterdir():
                print(f"  - {f.name}")
        else:
            print(f"  Directory does not exist!")
        raise FileNotFoundError(
            f"No label file found in {DATA_DIR}. "
            "Please ensure the MESSIDOR-2 dataset is properly downloaded."
        )
    
    if file_type == "csv":
        df = pd.read_csv(label_file)
    else:
        df = pd.read_excel(label_file)
    
    print(f"Loaded {len(df)} records from {label_file.name}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def normalize_labels(df):
    """Normalize column names and label values."""
    # Standardize column names (case-insensitive matching)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if "image" in col_lower or "file" in col_lower or "id_code" in col_lower or col_lower == "id":
            col_mapping[col] = "image"
        elif "diagnosis" in col_lower or "retino" in col_lower or "grade" in col_lower or "label" in col_lower or "dr" in col_lower:
            col_mapping[col] = "label"
    
    if col_mapping:
        df = df.rename(columns=col_mapping)
    
    # Ensure required columns exist
    if "image" not in df.columns or "label" not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        raise ValueError("Could not identify 'image' and 'label' columns")
    
    return df[["image", "label"]]


def find_images():
    """Find all image files in the dataset directory."""
    image_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
    images = []
    
    for ext in image_extensions:
        images.extend(DATA_DIR.rglob(ext))
    
    print(f"Found {len(images)} images")
    return {img.stem: img for img in images}


def prepare_dataset():
    """Prepare the MESSIDOR-2 dataset."""
    print("=" * 50)
    print("Preparing MESSIDOR-2 Dataset")
    print("=" * 50)
    
    # Load and normalize labels
    df = load_labels()
    df = normalize_labels(df)
    
    # Find images
    image_dict = find_images()
    
    # Match images with labels
    matched = []
    for _, row in df.iterrows():
        img_name = Path(row["image"]).stem
        if img_name in image_dict:
            matched.append({
                "image": image_dict[img_name],
                "label": int(row["label"])
            })
    
    print(f"Matched {len(matched)} images with labels")
    
    if len(matched) == 0:
        raise ValueError("No images matched with labels")
    
    # Create train/val/test splits
    train_data, temp_data = train_test_split(matched, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create output directories and copy files
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for item in split_data:
            label_dir = OUTPUT_DIR / split_name / str(item["label"])
            label_dir.mkdir(parents=True, exist_ok=True)
            
            dest = label_dir / item["image"].name
            shutil.copy2(item["image"], dest)
    
    print(f"\nDataset prepared at: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    prepare_dataset()
