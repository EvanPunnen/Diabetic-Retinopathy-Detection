import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ICDRS_LABELS = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

def get_transforms(train=True, size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

class FundusDataset(Dataset):
    def __init__(self, images_dir, labels_csv=None, transform=None):
        """
        images_dir: path with images (jpg/png/tiff)
        labels_csv: optional pandas CSV with columns ['image','label'] where label is 0-4
        transform: torchvision transforms
        """
        import glob, pandas as pd
        self.images = []
        self.transform = transform
        self.labels_map = {}
        
        # First read labels if provided
        if labels_csv is not None:
            print(f"\nLoading labels from: {labels_csv}")
            if not os.path.exists(labels_csv):
                raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
            
            df = pd.read_csv(labels_csv)
            print(f"CSV columns: {df.columns.tolist()}")
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Image name': 'image',
                'Retinopathy grade': 'label'
            })
            
            print("\nLabel distribution in CSV:")
            print(df['label'].value_counts().sort_index())
            
            # Only look for images that are in the CSV
            for _, row in df.iterrows():
                img_name = row['image']
                found = None
                # Try exact filename first
                if os.path.exists(os.path.join(images_dir, img_name)):
                    found = os.path.join(images_dir, img_name)
                else:
                    # Try all common extensions
                    basename = os.path.splitext(img_name)[0]
                    for ext in [".jpg", ".JPG", ".jpeg", ".png", ".tif", ".tiff"]:
                        path = os.path.join(images_dir, basename + ext)
                        if os.path.exists(path):
                            found = path
                            break
                
                if not found:
                    print(f"\nWARNING: Could not find image {img_name} in {images_dir}")
                else:
                    self.images.append(found)
                    self.labels_map[os.path.basename(found)] = int(row['label'])
                    
            print(f"\nFound {len(self.images)} valid images out of {len(df)} labels")
            
        else:
            # If no CSV provided, just load all images without labels
            for ext in ("*.jpg","*.jpeg","*.png","*.tif","*.tiff"):
                self.images.extend(glob.glob(os.path.join(images_dir, ext)))
            self.images.sort()
            
            for _, row in df.iterrows():
                self.labels_map[row['image']] = int(row['label'])
            
            print("\nSuccessfully loaded labels for all images!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        p = self.images[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        filename = os.path.basename(p)
        
        if filename not in self.labels_map:
            raise KeyError(f"No label found for image: {filename}")
        
        label = self.labels_map[filename]
        if label < 0 or label >= 5:
            raise ValueError(f"Invalid label {label} for {filename} (must be 0-4)")
            
        return img, label, filename
