from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.idrid_loader import IDRiDDataset, get_transforms

train_dataset = IDRiDDataset(
    csv_path="data/iDRID/images/train_labels.csv",
    img_dir="data/iDRID/images/train",
    transform=get_transforms(train=True)
)

val_dataset = IDRiDDataset(
    csv_path="data/iDRID/images/val_labels.csv",
    img_dir="data/iDRID/images/val",
    transform=get_transforms(train=False)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print(f"Total train samples: {len(train_dataset)}")
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}, Labels: {labels}")
    break
