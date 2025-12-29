import os
import pandas as pd
from collections import Counter

def count_images_in_dir(directory, extensions=('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
    """Count images in a directory recursively."""
    count = 0
    if not os.path.exists(directory):
        return 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extensions):
                count += 1
    return count

def analyze_idrid():
    """Analyze iDRID dataset and return class distribution."""
    idrid_dir = "data/iDRID"
    distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Check prepared data first
    prepared_csv = os.path.join(idrid_dir, "images", "train_labels.csv")
    val_csv = os.path.join(idrid_dir, "images", "val_labels.csv")
    
    if os.path.exists(prepared_csv):
        df = pd.read_csv(prepared_csv)
        if 'label' in df.columns:
            for label in df['label']:
                if label in distribution:
                    distribution[label] += 1
    
    if os.path.exists(val_csv):
        df = pd.read_csv(val_csv)
        if 'label' in df.columns:
            for label in df['label']:
                if label in distribution:
                    distribution[label] += 1
    
    # If no prepared data, check original groundtruths
    if sum(distribution.values()) == 0:
        groundtruths_dir = os.path.join(idrid_dir, "B._20Disease_20Grading", "B. Disease Grading", "2. Groundtruths")
        if os.path.exists(groundtruths_dir):
            for f in os.listdir(groundtruths_dir):
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(groundtruths_dir, f))
                    # Find the DR grade column
                    grade_cols = [c for c in df.columns if 'grade' in c.lower() and 'retinopathy' in c.lower()]
                    if grade_cols:
                        for label in df[grade_cols[0]]:
                            if label in distribution:
                                distribution[label] += 1
    
    return distribution

def analyze_ddr():
    """Analyze DDR dataset and return class distribution."""
    ddr_dir = "data/DDR"
    distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Try to find and read CSV file
    csv_candidates = [
        os.path.join(ddr_dir, "DR_grading.csv"),
        os.path.join(ddr_dir, "labels.csv"),
        os.path.join(ddr_dir, "train.csv"),
    ]
    
    # Search for any CSV
    for root, dirs, files in os.walk(ddr_dir):
        for f in files:
            if f.endswith('.csv'):
                csv_candidates.append(os.path.join(root, f))
    
    for csv_path in csv_candidates:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                # Find label column
                label_cols = [c for c in df.columns if any(k in c.lower() for k in ['grade', 'label', 'level', 'dr', 'class'])]
                
                if label_cols:
                    label_col = label_cols[0]
                    for label in df[label_col]:
                        if label in distribution:
                            distribution[label] += 1
                    
                    if sum(distribution.values()) > 0:
                        break
            except Exception as e:
                continue
    
    # If CSV approach failed, check for folder-based structure (images organized by class)
    if sum(distribution.values()) == 0:
        for class_label in range(5):
            class_dirs = [
                os.path.join(ddr_dir, str(class_label)),
                os.path.join(ddr_dir, "DR_grading", str(class_label)),
                os.path.join(ddr_dir, "images", str(class_label)),
            ]
            for class_dir in class_dirs:
                if os.path.exists(class_dir):
                    count = count_images_in_dir(class_dir)
                    distribution[class_label] += count
    
    return distribution

def analyze_messidor():
    """Analyze Messidor-2 dataset and return class distribution."""
    messidor_dir = "data/messidor-2"
    distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Search for label files
    label_files = []
    for root, dirs, files in os.walk(messidor_dir):
        for f in files:
            if f.endswith(('.csv', '.xls', '.xlsx')):
                label_files.append(os.path.join(root, f))
    
    for label_file in label_files:
        try:
            if label_file.endswith('.csv'):
                df = pd.read_csv(label_file)
            else:
                df = pd.read_excel(label_file)
            
            # Find label column - Messidor uses different naming
            label_cols = [c for c in df.columns if any(k in str(c).lower() for k in 
                         ['grade', 'label', 'level', 'dr', 'retinopathy', 'adjudicated'])]
            
            if label_cols:
                label_col = label_cols[0]
                for label in df[label_col]:
                    try:
                        label_int = int(label)
                        if label_int in distribution:
                            distribution[label_int] += 1
                    except (ValueError, TypeError):
                        continue
                
                if sum(distribution.values()) > 0:
                    break
        except Exception as e:
            continue
    
    return distribution

def analyze_aptos():
    """Analyze APTOS dataset and return class distribution (if exists)."""
    aptos_dir = "data/aptos"
    aptos_alt_dirs = ["data/APTOS", "data/aptos2019"]
    distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Check all possible directories
    dirs_to_check = [aptos_dir] + aptos_alt_dirs
    
    for check_dir in dirs_to_check:
        if os.path.exists(check_dir):
            # Search for CSV
            for root, dirs, files in os.walk(check_dir):
                for f in files:
                    if f.endswith('.csv'):
                        try:
                            df = pd.read_csv(os.path.join(root, f))
                            label_cols = [c for c in df.columns if any(k in c.lower() for k in 
                                         ['diagnosis', 'grade', 'label', 'level'])]
                            
                            if label_cols:
                                for label in df[label_cols[0]]:
                                    if label in distribution:
                                        distribution[label] += 1
                                
                                if sum(distribution.values()) > 0:
                                    return distribution
                        except:
                            continue
    
    return distribution

def generate_summary_table():
    """Generate summary table for all datasets."""
    
    print("=" * 80)
    print("ANALYZING ALL DATASETS...")
    print("=" * 80)
    
    # Analyze each dataset
    print("\n📊 Analyzing iDRID...")
    idrid_dist = analyze_idrid()
    print(f"   Found: {sum(idrid_dist.values())} images")
    
    print("\n📊 Analyzing DDR...")
    ddr_dist = analyze_ddr()
    print(f"   Found: {sum(ddr_dist.values())} images")
    
    print("\n📊 Analyzing Messidor-2...")
    messidor_dist = analyze_messidor()
    print(f"   Found: {sum(messidor_dist.values())} images")
    
    print("\n📊 Analyzing APTOS...")
    aptos_dist = analyze_aptos()
    print(f"   Found: {sum(aptos_dist.values())} images")
    
    # DR Grade descriptions
    grade_descriptions = {
        0: "No Diabetic Retinopathy",
        1: "Mild Non-Proliferative DR",
        2: "Moderate Non-Proliferative DR",
        3: "Severe Non-Proliferative DR",
        4: "Proliferative DR"
    }
    
    # Build table data
    table_data = []
    for grade in range(5):
        row = {
            'DR Grade': grade,
            'Clinical Description': grade_descriptions[grade],
            'IDRiD': idrid_dist[grade] if idrid_dist[grade] > 0 else '-',
            'DDR': ddr_dist[grade] if ddr_dist[grade] > 0 else '-',
            'Messidor': messidor_dist[grade] if messidor_dist[grade] > 0 else '-',
            'APTOS': aptos_dist[grade] if aptos_dist[grade] > 0 else '-'
        }
        table_data.append(row)
    
    # Add totals row
    totals_row = {
        'DR Grade': '—',
        'Clinical Description': 'Total Images',
        'IDRiD': sum(idrid_dist.values()) if sum(idrid_dist.values()) > 0 else '-',
        'DDR': sum(ddr_dist.values()) if sum(ddr_dist.values()) > 0 else '-',
        'Messidor': sum(messidor_dist.values()) if sum(messidor_dist.values()) > 0 else '-',
        'APTOS': sum(aptos_dist.values()) if sum(aptos_dist.values()) > 0 else '-'
    }
    table_data.append(totals_row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Print formatted table
    print("\n" + "=" * 100)
    print("DATASET SUMMARY TABLE")
    print("=" * 100)
    
    # Print header
    print(f"\n{'DR Grade':<10} {'Clinical Description':<35} {'IDRiD':>10} {'DDR':>10} {'Messidor':>10} {'APTOS':>10}")
    print("-" * 95)
    
    # Print rows
    for _, row in df.iterrows():
        print(f"{str(row['DR Grade']):<10} {row['Clinical Description']:<35} {str(row['IDRiD']):>10} {str(row['DDR']):>10} {str(row['Messidor']):>10} {str(row['APTOS']):>10}")
    
    print("-" * 95)
    
    # Save to CSV
    output_path = "data/dataset_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Summary saved to: {output_path}")
    
    # Generate Markdown table
    print("\n" + "=" * 100)
    print("MARKDOWN TABLE (copy-paste ready)")
    print("=" * 100)
    
    print("\n| DR Grade | Clinical Description | IDRiD | DDR | Messidor | APTOS |")
    print("|----------|---------------------|-------|-----|----------|-------|")
    for _, row in df.iterrows():
        print(f"| {row['DR Grade']} | {row['Clinical Description']} | {row['IDRiD']} | {row['DDR']} | {row['Messidor']} | {row['APTOS']} |")
    
    # Print detailed stats
    print("\n" + "=" * 100)
    print("DETAILED STATISTICS")
    print("=" * 100)
    
    datasets = {
        'IDRiD': idrid_dist,
        'DDR': ddr_dist,
        'Messidor': messidor_dist,
        'APTOS': aptos_dist
    }
    
    for name, dist in datasets.items():
        total = sum(dist.values())
        if total > 0:
            print(f"\n📊 {name}:")
            print(f"   Total images: {total}")
            print(f"   Distribution:")
            for grade, count in dist.items():
                pct = (count / total * 100) if total > 0 else 0
                bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
                print(f"      Grade {grade}: {count:>5} ({pct:>5.1f}%) {bar}")
            
            # Class imbalance ratio
            non_zero = [v for v in dist.values() if v > 0]
            if len(non_zero) > 1:
                imbalance = max(non_zero) / min(non_zero)
                print(f"   Imbalance ratio: {imbalance:.1f}x")
        else:
            print(f"\n📊 {name}: No data found or dataset not present")
    
    # Combined statistics
    print("\n" + "=" * 100)
    print("COMBINED STATISTICS")
    print("=" * 100)
    
    combined_dist = {grade: 0 for grade in range(5)}
    for dist in datasets.values():
        for grade, count in dist.items():
            combined_dist[grade] += count
    
    total_combined = sum(combined_dist.values())
    if total_combined > 0:
        print(f"\n📊 All datasets combined:")
        print(f"   Total images: {total_combined}")
        for grade, count in combined_dist.items():
            pct = (count / total_combined * 100)
            print(f"      Grade {grade} ({grade_descriptions[grade]}): {count} ({pct:.1f}%)")
    
    return df

if __name__ == "__main__":
    generate_summary_table()