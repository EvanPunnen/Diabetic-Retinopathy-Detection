import os
import sys
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from tqdm import tqdm

    from model.clanet import CLANet_DenseNet  # your full model (ALA + CSCA)
    from data.idrid_loader import IDRiDDataset, get_transforms
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nMake sure you have activated your Python environment with the required packages.")
    print("Try running: ")
    print("    venv\\Scripts\\activate  # On Windows")
    print("    source venv/bin/activate  # On Linux/Mac")
    print("    pip install -r requirements.txt  # To install required packages")
    sys.exit(1)

# === CONFIG ===
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to save the model
def save_model(model):
    # Get the path to the root models directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(root_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "clanet_idrid.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining Complete — Model saved as {model_path}")

# Function to train the model
def main():
    # === DATA ===
    train_dataset = IDRiDDataset(
        "data/iDRID/images/train_labels.csv",
        "data/iDRID/images/train",
        transform=get_transforms(train=True)
    )
    val_dataset = IDRiDDataset(
        "data/iDRID/images/val_labels.csv",
        "data/iDRID/images/val",
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # === MODEL ===
    model = CLANet_DenseNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # === TRAIN LOOP ===
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # === VALIDATION ===
        model.eval()
        correct, total = 0, 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f} | Validation Accuracy: {val_acc*100:.2f}%")

    # Save the model
    save_model(model)

# This is the required protection for multiprocessing with PyTorch DataLoader
if __name__ == "__main__":
    # For Windows support
    import multiprocessing
    multiprocessing.freeze_support()
    main()
