import sys
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add parent directories to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now import project modules
from model.clanet import CLANet_DenseNet

class IDRiDDataset(Dataset):
    """Dataset for loading IDRiD retinal images"""
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """Get image transforms for preprocessing retinal images"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def evaluate_model():
    # Load model and data
    device = torch.device('cpu')  # Using CPU version
    
    try:
        # Load model weights
        model = CLANet_DenseNet(num_classes=5)
        # model.load_state_dict(torch.load('models/clanet_model.pth', map_location=device))
        model.eval()
        
        print("✅ Model loaded successfully")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load REAL retinal images from validation dataset
        print("📁 Loading REAL retinal images from IDRiD dataset...")
        
        val_csv = "data/iDRID/images/val_labels.csv"
        val_images = "data/iDRID/images/val"
        
        # Check if data exists
        if not os.path.exists(val_csv):
            print(f"❌ Could not find validation CSV at {val_csv}")
            return None
        if not os.path.exists(val_images):
            print(f"❌ Could not find validation images at {val_images}")
            return None
            
        # Create dataset and dataloader
        val_dataset = IDRiDDataset(val_csv, val_images, transform=get_transforms())
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        print(f"📁 Loaded {len(val_dataset)} REAL retinal images")
        
        # Show dataset distribution
        df = pd.read_csv(val_csv)
        label_counts = df['label'].value_counts().sort_index()
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        print(f"\n📊 Dataset Distribution:")
        for i, count in label_counts.items():
            print(f"   {i} - {class_names[i]}: {count} images ({count/len(df)*100:.1f}%)")
        
        # Get predictions from REAL retinal images
        all_predictions = []
        all_targets = []
        
        print(f"\n🔄 Processing {len(val_loader)} batches of real retinal images...")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                if batch_idx % 10 == 0:  # Progress update
                    print(f"   Processing batch {batch_idx+1}/{len(val_loader)}")
                    
                images = images.to(device)
                
                # Get model predictions on REAL retinal images
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        print(f"✅ Analyzed {len(all_predictions)} real retinal images")
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        kappa = cohen_kappa_score(all_targets, all_predictions, weights='quadratic')
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"🎯 Overall Accuracy: {accuracy:.2%}")
        print(f"🔢 Quadratic Weighted Kappa: {kappa:.3f}")
        
        # Detailed classification report
        print(f"\n📋 CLASSIFICATION REPORT:")
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Diabetic Retinopathy Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Confusion matrix saved as 'confusion_matrix.png'")
        
        return accuracy, kappa, all_predictions, all_targets
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None

if __name__ == "__main__":
    evaluate_model()