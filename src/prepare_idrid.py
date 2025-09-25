import os, shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
train_images_dir = os.path.join("data", "iDRID", "B.%20Disease%20Grading", "B. Disease Grading", "1. Original Images", "a. Training Set")
train_labels_csv = os.path.join("data", "iDRID", "B.%20Disease%20Grading", "B. Disease Grading", "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
out_dir = os.path.join("data", "iDRID", "images")

# Create output directories
os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)

# Read and process CSV
print("\nReading labels from CSV...")
df = pd.read_csv(train_labels_csv)
if "Image name" in df.columns and "Retinopathy grade" in df.columns:
    df = df.rename(columns={"Image name": "image", "Retinopathy grade": "label"})

print("\nInitial class distribution:")
print(df['label'].value_counts().sort_index())

# Find valid images
valid_rows = []

for _, row in df.iterrows():
    fname, label = row["image"], int(row["label"])
    
    # Try all common extensions
    found = None
    if os.path.exists(os.path.join(train_images_dir, fname)):
        found = os.path.join(train_images_dir, fname)
    else:
        for ext in [".jpg", ".JPG", ".jpeg", ".png", ".tif", ".tiff"]:
            path = os.path.join(train_images_dir, fname + ext)
            if os.path.exists(path):
                found = path
                break
    
    if not found:
        print("⚠️ Image not found:", fname)
        continue
        
    valid_rows.append({
        'image': os.path.basename(found),
        'label': label,
        'path': found
    })

# Convert to DataFrame for splitting
valid_df = pd.DataFrame(valid_rows)
print("\nValid images found:", len(valid_df))
print("\nValid images class distribution:")
print(valid_df['label'].value_counts().sort_index())

# Stratified split
train_df, val_df = train_test_split(
    valid_df,
    test_size=0.2,
    stratify=valid_df['label'],
    random_state=42
)

# Copy files and save CSVs
print("\nCopying files and creating CSV files...")

# Training set
for _, row in train_df.iterrows():
    dst = os.path.join(out_dir, "train", row['image'])
    shutil.copy(row['path'], dst)

# Validation set
for _, row in val_df.iterrows():
    dst = os.path.join(out_dir, "val", row['image'])
    shutil.copy(row['path'], dst)

# Save CSVs (without the path column)
train_df[['image', 'label']].to_csv(os.path.join(out_dir, "train_labels.csv"), index=False)
val_df[['image', 'label']].to_csv(os.path.join(out_dir, "val_labels.csv"), index=False)

print("\n✅ Split complete!")
print("\nTraining set class distribution:")
print(train_df['label'].value_counts().sort_index())
print(f"\nTotal training images: {len(train_df)}")

print("\nValidation set class distribution:")
print(val_df['label'].value_counts().sort_index())
print(f"\nTotal validation images: {len(val_df)}")
