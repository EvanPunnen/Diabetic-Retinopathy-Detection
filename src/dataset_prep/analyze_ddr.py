import os
import pandas as pd
from collections import Counter

def analyze_ddr():
    """Analyze DDR dataset structure and contents."""
    
    ddr_dir = "data/DDR"
    csv_path = os.path.join(ddr_dir, "DR_grading.csv")
    images_dir = os.path.join(ddr_dir, "DR_grading")
    
    print("=" * 60)
    print("DDR DATASET ANALYSIS")
    print("=" * 60)

    if not os.path.exists(ddr_dir):
        print(f"❌ DDR directory not found at: {ddr_dir}")
        return
    
    print(f"\n📁 DDR directory found: {ddr_dir}")
    print(f"   Contents: {os.listdir(ddr_dir)}")

    print(f"\n{'=' * 60}")
    print("CSV FILE ANALYSIS")
    print("=" * 60)
    
    # Search for any CSV files
    csv_files = []
    for root, dirs, files in os.walk(ddr_dir):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    
    print(f"\nCSV files found: {csv_files}")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        print(f"\n📄 Main CSV: {csv_path}")
        print(f"   Total rows: {len(df)}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"\n   Data types:\n{df.dtypes}")
        print(f"\n   First 10 rows:\n{df.head(10)}")
        print(f"\n   Last 5 rows:\n{df.tail(5)}")
        
        # Check for missing values
        print(f"\n   Missing values:\n{df.isnull().sum()}")
        
        # Find potential label columns
        label_keywords = ['grade', 'label', 'level', 'dr', 'class', 'diagnosis']
        potential_label_cols = [c for c in df.columns if any(k in c.lower() for k in label_keywords)]
        
        print(f"\n   Potential label columns: {potential_label_cols}")
        
        # Show distribution for each potential label column
        for col in potential_label_cols:
            print(f"\n   Distribution of '{col}':")
            print(df[col].value_counts().sort_index())
        
        # If no label columns found, show all column distributions
        if not potential_label_cols:
            print("\n   No obvious label column found. Showing all column value counts:")
            for col in df.columns:
                unique_count = df[col].nunique()
                if unique_count <= 10:  # Only show if few unique values
                    print(f"\n   '{col}' ({unique_count} unique values):")
                    print(df[col].value_counts().sort_index())
        
        # Find potential image columns
        img_keywords = ['image', 'file', 'name', 'id', 'path', 'img']
        potential_img_cols = [c for c in df.columns if any(k in c.lower() for k in img_keywords)]
        print(f"\n   Potential image columns: {potential_img_cols}")
        
        if potential_img_cols:
            sample_col = potential_img_cols[0]
            print(f"\n   Sample values from '{sample_col}':")
            print(df[sample_col].head(10).tolist())
    else:
        print(f"❌ CSV not found at: {csv_path}")
        
        # Try to read any CSV found
        for csv_file in csv_files:
            print(f"\n📄 Analyzing: {csv_file}")
            df = pd.read_csv(csv_file)
            print(f"   Columns: {df.columns.tolist()}")
            print(f"   Rows: {len(df)}")
            print(f"   First 5 rows:\n{df.head()}")

    print(f"\n{'=' * 60}")
    print("IMAGES DIRECTORY ANALYSIS")
    print("=" * 60)
    
    # Find all potential image directories
    image_dirs_found = []
    for root, dirs, files in os.walk(ddr_dir):
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))]
        if img_files:
            image_dirs_found.append((root, len(img_files)))
    
    print(f"\nDirectories containing images:")
    for dir_path, count in image_dirs_found:
        print(f"   {dir_path}: {count} images")
    
    if os.path.exists(images_dir):
        print(f"\n📁 Main images directory: {images_dir}")
        
        # Check for subdirectories
        subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        
        if subdirs:
            print(f"\n   Subdirectories found: {subdirs}")
            for subdir in subdirs:
                subdir_path = os.path.join(images_dir, subdir)
                files = os.listdir(subdir_path)
                img_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
                print(f"   📂 {subdir}: {img_count} images")
                
                # Show sample filenames
                sample_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))][:3]
                if sample_files:
                    print(f"      Sample: {sample_files}")
        
        # Count all images
        extensions = Counter()
        all_images = []
        
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
                    extensions[ext] += 1
                    all_images.append(f)
        
        print(f"\n   Total images: {len(all_images)}")
        print(f"   File extensions: {dict(extensions)}")
        print(f"\n   Sample filenames: {all_images[:10]}")
        
        # Analyze filename patterns
        if all_images:
            print(f"\n   Filename length range: {min(len(f) for f in all_images)} - {max(len(f) for f in all_images)}")
            
            # Check if filenames have common patterns
            separators = Counter()
            for f in all_images[:100]:
                if '_' in f:
                    separators['underscore'] += 1
                if '-' in f:
                    separators['hyphen'] += 1
                if ' ' in f:
                    separators['space'] += 1
            print(f"   Filename separators (sample of 100): {dict(separators)}")
    else:
        print(f"❌ Images directory not found at: {images_dir}")
  
    print(f"\n{'=' * 60}")
    print("CROSS-REFERENCE CHECK")
    print("=" * 60)
    
    if os.path.exists(csv_path) and all_images:
        df = pd.read_csv(csv_path)
        
        # Try to match CSV entries with actual files
        for col in df.columns:
            sample_val = str(df[col].iloc[0])
            # Check if this column value matches any filename
            matches = [img for img in all_images[:100] if sample_val in img or img.replace('.jpg', '').replace('.png', '') in sample_val]
            if matches:
                print(f"\n   Column '{col}' might contain image references")
                print(f"   Sample value: {sample_val}")
                print(f"   Matching files: {matches[:3]}")
    
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    analyze_ddr()