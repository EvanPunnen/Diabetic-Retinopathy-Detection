import os
import pandas as pd
from collections import Counter

def analyze_idrid():
    """Analyze iDRID dataset structure and contents."""
    
    idrid_dir = "data/iDRID"
    
    print("=" * 60)
    print("iDRID DATASET ANALYSIS")
    print("=" * 60)
    
    
    if not os.path.exists(idrid_dir):
        print(f"❌ iDRID directory not found at: {idrid_dir}")
        return
    
    print(f"\n📁 iDRID directory found: {idrid_dir}")
    contents = os.listdir(idrid_dir)
    print(f"   Top-level contents ({len(contents)} items):")
    for item in contents:
        item_path = os.path.join(idrid_dir, item)
        if os.path.isdir(item_path):
            print(f"   📂 {item}/")
        else:
            print(f"   📄 {item}")
    

    print(f"\n{'=' * 60}")
    print("ORIGINAL DATA STRUCTURE (B. Disease Grading)")
    print("=" * 60)
    
    disease_grading_dir = os.path.join(idrid_dir, "B._20Disease_20Grading", "B. Disease Grading")
    
    if os.path.exists(disease_grading_dir):
        print(f"\n📁 Disease Grading directory found")
        
        # Check Original Images
        orig_images_dir = os.path.join(disease_grading_dir, "1. Original Images")
        if os.path.exists(orig_images_dir):
            print(f"\n   📂 1. Original Images/")
            for subset in ["a. Training Set", "b. Testing Set"]:
                subset_path = os.path.join(orig_images_dir, subset)
                if os.path.exists(subset_path):
                    img_count = len([f for f in os.listdir(subset_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
                    print(f"      📂 {subset}: {img_count} images")
                    
                    # Sample filenames
                    sample = [f for f in os.listdir(subset_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))][:3]
                    print(f"         Sample: {sample}")
        
        # Check Groundtruths
        groundtruths_dir = os.path.join(disease_grading_dir, "2. Groundtruths")
        if os.path.exists(groundtruths_dir):
            print(f"\n   📂 2. Groundtruths/")
            for f in os.listdir(groundtruths_dir):
                f_path = os.path.join(groundtruths_dir, f)
                if f.endswith('.csv'):
                    df = pd.read_csv(f_path)
                    print(f"\n      📄 {f}")
                    print(f"         Rows: {len(df)}")
                    print(f"         Columns: {df.columns.tolist()}")
                    print(f"\n         First 5 rows:\n{df.head()}")
                    
                    # Show label distribution
                    label_cols = [c for c in df.columns if 'grade' in c.lower() or 'risk' in c.lower() or 'label' in c.lower()]
                    for col in label_cols:
                        print(f"\n         Distribution of '{col}':")
                        print(df[col].value_counts().sort_index())
    else:
        print(f"   ⚠️ Original disease grading directory not found")
        print(f"   Looking for: {disease_grading_dir}")
    

    print(f"\n{'=' * 60}")
    print("PREPARED DATA STRUCTURE (images/)")
    print("=" * 60)
    
    prepared_dir = os.path.join(idrid_dir, "images")
    
    if os.path.exists(prepared_dir):
        print(f"\n📁 Prepared images directory found: {prepared_dir}")
        print(f"   Contents: {os.listdir(prepared_dir)}")
        
        # Analyze train folder
        train_dir = os.path.join(prepared_dir, "train")
        if os.path.exists(train_dir):
            train_images = [f for f in os.listdir(train_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            print(f"\n   📂 train/: {len(train_images)} images")
            print(f"      Sample: {train_images[:5]}")
            
            # Check extensions
            ext_counter = Counter(os.path.splitext(f)[1].lower() for f in train_images)
            print(f"      Extensions: {dict(ext_counter)}")
        
        # Analyze val folder
        val_dir = os.path.join(prepared_dir, "val")
        if os.path.exists(val_dir):
            val_images = [f for f in os.listdir(val_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            print(f"\n   📂 val/: {len(val_images)} images")
            print(f"      Sample: {val_images[:5]}")
        
        # Analyze train_labels.csv
        train_csv = os.path.join(prepared_dir, "train_labels.csv")
        if os.path.exists(train_csv):
            df_train = pd.read_csv(train_csv)
            print(f"\n   📄 train_labels.csv")
            print(f"      Rows: {len(df_train)}")
            print(f"      Columns: {df_train.columns.tolist()}")
            print(f"\n      First 5 rows:\n{df_train.head()}")
            
            if 'label' in df_train.columns:
                print(f"\n      Label distribution:")
                dist = df_train['label'].value_counts().sort_index()
                print(dist)
                
                # Calculate percentages
                print(f"\n      Percentages:")
                for label, count in dist.items():
                    pct = count / len(df_train) * 100
                    label_name = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}.get(label, str(label))
                    print(f"         {label} ({label_name}): {count} ({pct:.1f}%)")
        
        # Analyze val_labels.csv
        val_csv = os.path.join(prepared_dir, "val_labels.csv")
        if os.path.exists(val_csv):
            df_val = pd.read_csv(val_csv)
            print(f"\n   📄 val_labels.csv")
            print(f"      Rows: {len(df_val)}")
            print(f"      Columns: {df_val.columns.tolist()}")
            
            if 'label' in df_val.columns:
                print(f"\n      Label distribution:")
                dist = df_val['label'].value_counts().sort_index()
                print(dist)
                
                print(f"\n      Percentages:")
                for label, count in dist.items():
                    pct = count / len(df_val) * 100
                    label_name = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}.get(label, str(label))
                    print(f"         {label} ({label_name}): {count} ({pct:.1f}%)")
    else:
        print(f"\n   ⚠️ Prepared images directory not found")
        print(f"   Run prepare_idrid.py to create train/val splits")
    

    print(f"\n{'=' * 60}")
    print("DATA INTEGRITY CHECK")
    print("=" * 60)
    
    if os.path.exists(prepared_dir):
        issues_found = False
        
        # Check train set
        train_dir = os.path.join(prepared_dir, "train")
        train_csv = os.path.join(prepared_dir, "train_labels.csv")
        
        if os.path.exists(train_dir) and os.path.exists(train_csv):
            train_images = set(f for f in os.listdir(train_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')))
            df_train = pd.read_csv(train_csv)
            csv_images = set(df_train['image'].tolist()) if 'image' in df_train.columns else set()
            
            # Images in folder but not in CSV
            missing_in_csv = train_images - csv_images
            if missing_in_csv:
                print(f"\n   ⚠️ Train images in folder but not in CSV: {len(missing_in_csv)}")
                print(f"      Sample: {list(missing_in_csv)[:5]}")
                issues_found = True
            
            # Images in CSV but not in folder
            missing_in_folder = csv_images - train_images
            if missing_in_folder:
                print(f"\n   ⚠️ Train images in CSV but not in folder: {len(missing_in_folder)}")
                print(f"      Sample: {list(missing_in_folder)[:5]}")
                issues_found = True
            
            if not missing_in_csv and not missing_in_folder:
                print(f"\n   ✅ Train set: All {len(train_images)} images match CSV entries")
        
        # Check val set
        val_dir = os.path.join(prepared_dir, "val")
        val_csv = os.path.join(prepared_dir, "val_labels.csv")
        
        if os.path.exists(val_dir) and os.path.exists(val_csv):
            val_images = set(f for f in os.listdir(val_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')))
            df_val = pd.read_csv(val_csv)
            csv_images = set(df_val['image'].tolist()) if 'image' in df_val.columns else set()
            
            missing_in_csv = val_images - csv_images
            if missing_in_csv:
                print(f"\n   ⚠️ Val images in folder but not in CSV: {len(missing_in_csv)}")
                issues_found = True
            
            missing_in_folder = csv_images - val_images
            if missing_in_folder:
                print(f"\n   ⚠️ Val images in CSV but not in folder: {len(missing_in_folder)}")
                issues_found = True
            
            if not missing_in_csv and not missing_in_folder:
                print(f"\n   ✅ Val set: All {len(val_images)} images match CSV entries")
        
        # Check for overlap between train and val
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            train_images = set(os.listdir(train_dir))
            val_images = set(os.listdir(val_dir))
            overlap = train_images & val_images
            
            if overlap:
                print(f"\n   ❌ OVERLAP between train and val: {len(overlap)} images!")
                print(f"      Sample: {list(overlap)[:5]}")
                issues_found = True
            else:
                print(f"\n   ✅ No overlap between train and val sets")
        
        if not issues_found:
            print(f"\n   ✅ All integrity checks passed!")
    
  
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    total_original = 0
    total_prepared = 0
    
    # Count original images
    for subset in ["a. Training Set", "b. Testing Set"]:
        subset_path = os.path.join(disease_grading_dir, "1. Original Images", subset) if os.path.exists(disease_grading_dir) else ""
        if os.path.exists(subset_path):
            count = len([f for f in os.listdir(subset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
            total_original += count
            print(f"   Original {subset}: {count} images")
    
    # Count prepared images
    if os.path.exists(prepared_dir):
        train_count = len([f for f in os.listdir(os.path.join(prepared_dir, "train")) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]) if os.path.exists(os.path.join(prepared_dir, "train")) else 0
        val_count = len([f for f in os.listdir(os.path.join(prepared_dir, "val")) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]) if os.path.exists(os.path.join(prepared_dir, "val")) else 0
        total_prepared = train_count + val_count
        
        print(f"\n   Prepared train set: {train_count} images")
        print(f"   Prepared val set: {val_count} images")
        print(f"   Total prepared: {total_prepared} images")
        print(f"   Train/Val split: {train_count/(total_prepared)*100:.1f}% / {val_count/(total_prepared)*100:.1f}%")
    
    # Class imbalance warning
    if os.path.exists(os.path.join(prepared_dir, "train_labels.csv")):
        df = pd.read_csv(os.path.join(prepared_dir, "train_labels.csv"))
        if 'label' in df.columns:
            dist = df['label'].value_counts().sort_index()
            min_class = dist.min()
            max_class = dist.max()
            imbalance_ratio = max_class / min_class
            
            print(f"\n   ⚠️ Class imbalance ratio: {imbalance_ratio:.1f}x (max/min)")
            if imbalance_ratio > 5:
                print(f"      Consider using class weights or oversampling for minority classes")
    
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    analyze_idrid()