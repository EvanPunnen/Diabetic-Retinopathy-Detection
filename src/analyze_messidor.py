import os
import pandas as pd
from collections import Counter

def analyze_messidor():
    """Analyze Messidor-2 dataset structure and contents."""
    
    messidor_dir = "data/messidor-2"
    
    print("=" * 60)
    print("MESSIDOR-2 DATASET ANALYSIS")
    print("=" * 60)
    
    if not os.path.exists(messidor_dir):
        print(f"❌ Messidor-2 directory not found at: {messidor_dir}")
        return
    
    print(f"\n📁 Messidor-2 directory found: {messidor_dir}")
    contents = os.listdir(messidor_dir)
    print(f"   Top-level contents ({len(contents)} items):")
    for item in contents[:20]:
        item_path = os.path.join(messidor_dir, item)
        if os.path.isdir(item_path):
            print(f"   📂 {item}/")
        else:
            print(f"   📄 {item}")
    if len(contents) > 20:
        print(f"   ... and {len(contents) - 20} more items")

    print(f"\n{'=' * 60}")
    print("LABEL FILES ANALYSIS")
    print("=" * 60)
    
    # Search for CSV, Excel, and text files that might contain labels
    label_files = []
    for root, dirs, files in os.walk(messidor_dir):
        for f in files:
            if f.endswith(('.csv', '.xls', '.xlsx', '.txt', '.xml')):
                label_files.append(os.path.join(root, f))
    
    print(f"\nPotential label files found: {len(label_files)}")
    for lf in label_files:
        print(f"   {lf}")
    
    # Analyze each label file
    for label_file in label_files:
        print(f"\n{'─' * 40}")
        print(f"📄 Analyzing: {os.path.basename(label_file)}")
        print(f"   Path: {label_file}")
        
        try:
            if label_file.endswith('.csv'):
                df = pd.read_csv(label_file)
            elif label_file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(label_file)
            elif label_file.endswith('.txt'):
                # Try different separators
                try:
                    df = pd.read_csv(label_file, sep='\t')
                except:
                    df = pd.read_csv(label_file, sep=' ')
            else:
                print(f"   ⚠️ Skipping unsupported format")
                continue
            
            print(f"   Total rows: {len(df)}")
            print(f"   Columns: {df.columns.tolist()}")
            print(f"\n   Data types:\n{df.dtypes}")
            print(f"\n   First 10 rows:\n{df.head(10)}")
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                print(f"\n   Missing values:\n{missing[missing > 0]}")
            
            # Find potential label columns
            label_keywords = ['grade', 'label', 'level', 'dr', 'class', 'diagnosis', 'retinopathy', 'risk']
            potential_label_cols = [c for c in df.columns if any(k in str(c).lower() for k in label_keywords)]
            
            print(f"\n   Potential label columns: {potential_label_cols}")
            
            # Show distribution for numeric columns with few unique values
            for col in df.columns:
                try:
                    unique_count = df[col].nunique()
                    if unique_count <= 10:
                        print(f"\n   Distribution of '{col}' ({unique_count} unique):")
                        print(df[col].value_counts().sort_index())
                except:
                    pass
            
            # Find potential image columns
            img_keywords = ['image', 'file', 'name', 'id', 'path', 'img']
            potential_img_cols = [c for c in df.columns if any(k in str(c).lower() for k in img_keywords)]
            print(f"\n   Potential image columns: {potential_img_cols}")
            
            if potential_img_cols:
                for col in potential_img_cols[:2]:
                    print(f"\n   Sample values from '{col}':")
                    print(df[col].head(10).tolist())
                    
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n{'=' * 60}")
    print("IMAGES ANALYSIS")
    print("=" * 60)
    
    # Find all image files
    extensions = Counter()
    all_images = []
    image_locations = {}
    
    for root, dirs, files in os.walk(messidor_dir):
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'))]
        if img_files:
            rel_path = os.path.relpath(root, messidor_dir)
            image_locations[rel_path] = len(img_files)
            
            for f in img_files:
                ext = os.path.splitext(f)[1].lower()
                extensions[ext] += 1
                all_images.append((f, root))
    
    print(f"\nTotal images found: {len(all_images)}")
    print(f"File extensions: {dict(extensions)}")
    
    print(f"\nImages by location:")
    for loc, count in sorted(image_locations.items(), key=lambda x: -x[1]):
        print(f"   {loc}: {count} images")
    
    # Sample filenames
    if all_images:
        print(f"\nSample filenames:")
        for img, path in all_images[:15]:
            rel_path = os.path.relpath(path, messidor_dir)
            print(f"   [{rel_path}] {img}")
        
        # Analyze filename patterns
        filenames = [img for img, _ in all_images]
        print(f"\nFilename patterns:")
        print(f"   Length range: {min(len(f) for f in filenames)} - {max(len(f) for f in filenames)}")
        
        # Check separators
        separators = Counter()
        for f in filenames[:100]:
            if '_' in f:
                separators['underscore'] += 1
            if '-' in f:
                separators['hyphen'] += 1
            if ' ' in f:
                separators['space'] += 1
        print(f"   Separators (sample): {dict(separators)}")
        
        # Check for common prefixes
        prefixes = Counter()
        for f in filenames[:100]:
            prefix = f.split('_')[0] if '_' in f else f.split('-')[0] if '-' in f else f[:3]
            prefixes[prefix] += 1
        print(f"\n   Common prefixes: {prefixes.most_common(10)}")
    

    print(f"\n{'=' * 60}")
    print("CROSS-REFERENCE CHECK")
    print("=" * 60)
    
    if label_files and all_images:
        # Try to find matching between labels and images
        image_names = set(img for img, _ in all_images)
        image_names_no_ext = set(os.path.splitext(img)[0] for img, _ in all_images)
        
        for label_file in label_files:
            try:
                if label_file.endswith('.csv'):
                    df = pd.read_csv(label_file)
                elif label_file.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(label_file)
                else:
                    continue
                
                print(f"\n📄 Checking: {os.path.basename(label_file)}")
                
                for col in df.columns:
                    # Sample values from column
                    sample_values = df[col].dropna().astype(str).head(20).tolist()
                    
                    matches = 0
                    for val in sample_values:
                        # Check direct match
                        if val in image_names or val in image_names_no_ext:
                            matches += 1
                        # Check with common extensions
                        elif f"{val}.jpg" in image_names or f"{val}.png" in image_names or f"{val}.tif" in image_names:
                            matches += 1
                    
                    if matches > 0:
                        print(f"   Column '{col}': {matches}/{len(sample_values)} values match image names")
                        print(f"   Sample: {sample_values[:3]}")
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"\n📊 Total images: {len(all_images)}")
    print(f"📄 Label files: {len(label_files)}")
    print(f"📁 Image locations: {len(image_locations)}")
    
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    analyze_messidor()