"""
Analyze MESSIDOR-2 and DDR datasets to understand their structure.
"""

import os
from pathlib import Path
import pandas as pd

def analyze_messidor():
    """Analyze MESSIDOR-2 dataset structure."""
    print("=" * 60)
    print("MESSIDOR-2 Dataset Analysis")
    print("=" * 60)
    
    data_dir = Path("data/messidor-2")
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return None
    
    # Find CSV/Excel files
    label_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xls*"))
    print(f"\nLabel files found: {[f.name for f in label_files]}")
    
    # Load and display label file info
    for label_file in label_files:
        print(f"\n--- {label_file.name} ---")
        if label_file.suffix == ".csv":
            df = pd.read_csv(label_file)
        else:
            df = pd.read_excel(label_file)
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First 3 rows:\n{df.head(3)}")
        
        # Find label column
        for col in df.columns:
            if "diagnosis" in col.lower() or "grade" in col.lower() or "label" in col.lower():
                print(f"\nLabel distribution ({col}):")
                print(df[col].value_counts().sort_index())
    
    # Find images
    image_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
    images = []
    for ext in image_extensions:
        images.extend(data_dir.rglob(ext))
    
    print(f"\nTotal images found: {len(images)}")
    if images:
        print(f"Sample image names: {[img.name for img in images[:3]]}")
    
    return df


def analyze_ddr():
    """Analyze DDR dataset structure."""
    print("\n" + "=" * 60)
    print("DDR Dataset Analysis")
    print("=" * 60)
    
    data_dir = Path("data/ddr")
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return None
    
    # List all subdirectories
    print(f"\nSubdirectories:")
    for item in data_dir.iterdir():
        if item.is_dir():
            print(f"  - {item.name}")
    
    # Find CSV/Excel files
    label_files = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.xls*")) + list(data_dir.rglob("*.txt"))
    print(f"\nLabel files found: {[str(f.relative_to(data_dir)) for f in label_files]}")
    
    # Load and display label file info
    for label_file in label_files[:3]:  # Limit to first 3
        print(f"\n--- {label_file.name} ---")
        try:
            if label_file.suffix == ".csv":
                df = pd.read_csv(label_file)
            elif label_file.suffix == ".txt":
                # Try different delimiters
                try:
                    df = pd.read_csv(label_file, sep="\t")
                except:
                    df = pd.read_csv(label_file, sep=" ", header=None)
            else:
                df = pd.read_excel(label_file)
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First 3 rows:\n{df.head(3)}")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    # Find images by folder structure (DDR often uses folder-based labels)
    image_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
    
    # Check for train/test/val folders
    for split in ["train", "test", "val", "valid", "training", "testing", "validation"]:
        split_dir = data_dir / split
        if split_dir.exists():
            print(f"\n{split.upper()} folder:")
            
            # Check for class subfolders
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    images = []
                    for ext in image_extensions:
                        images.extend(class_dir.glob(ext))
                    print(f"  Class '{class_dir.name}': {len(images)} images")
    
    # Count total images
    all_images = []
    for ext in image_extensions:
        all_images.extend(data_dir.rglob(ext))
    print(f"\nTotal images found: {len(all_images)}")
    
    return None


if __name__ == "__main__":
    analyze_messidor()
    analyze_ddr()
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)