import torch
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.clanet import CLANet_DenseNet
from utils import clahe_enhance_pil

# Define labels
LABEL_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate", 
    3: "Severe",
    4: "Proliferative"
}

def get_transforms(size=224):
    """Get transforms for test image preprocessing"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def denormalize(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def show_prediction(img_tensor, probabilities, pred_label, title=None, save_path=None):
    """Display image with prediction and class probabilities"""
    # Convert tensor to numpy and denormalize for display
    if torch.is_tensor(img_tensor):
        img = denormalize(img_tensor)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot image with prediction
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f"Prediction: {LABEL_NAMES[pred_label]}")
    
    # Plot probability bars
    bars = ax2.bar(range(len(probabilities)), probabilities)
    ax2.set_xticks(range(len(probabilities)))
    ax2.set_xticklabels([f"{i}: {LABEL_NAMES[i]}" for i in range(len(probabilities))], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.set_title('Class Probabilities')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Color the bars
    for idx, bar in enumerate(bars):
        if idx == pred_label:
            bar.set_color('green')
        else:
            bar.set_color('grey')
        ax2.text(idx, bar.get_height() + 0.02, f'{probabilities[idx]:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        # Try to show the plot interactively
        try:
            plt.show()
        except:
            print("Warning: Could not display the plot interactively. "
                  "Provide a save_path to save the visualization to a file.")

# Path to the pretrained weights and test image
pretrained_weights = "densenet121_best.pth"  # Will be found in the models folder
test_image_path = "data/test/images.jpg"

# Check if image exists
if not os.path.exists(test_image_path):
    print(f"Error: Test image not found at {test_image_path}")
    sys.exit(1)

# Load the model
print("Loading model with pretrained weights...")
model = CLANet_DenseNet(num_classes=5, pretrained_weights=pretrained_weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load and preprocess the image
print(f"Loading and processing test image: {test_image_path}")
image = Image.open(test_image_path).convert("RGB")
enhanced_image = clahe_enhance_pil(image)  # Apply CLAHE enhancement

# Transform image
transform = get_transforms()
img_tensor = transform(enhanced_image).unsqueeze(0).to(device)

# Get prediction
with torch.no_grad():
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
    predicted_class = output.argmax(dim=1).item()

# Print results
print("\n" + "="*50)
print("PREDICTION RESULTS")
print("="*50)
print(f"Predicted class: {predicted_class} - {LABEL_NAMES[predicted_class]}")
print("\nClass probabilities:")
for i in range(len(probabilities)):
    print(f"  Class {i} ({LABEL_NAMES[i]}): {probabilities[i]:.4f}")

# Create output directory
output_dir = "outputs/test"
os.makedirs(output_dir, exist_ok=True)

# Display and save visualization
save_path = os.path.join(output_dir, "classification_result.png")
show_prediction(
    img_tensor.cpu().squeeze(), 
    probabilities, 
    predicted_class, 
    title=f"Test Image: Predicted as {LABEL_NAMES[predicted_class]}",
    save_path=save_path
)

print("\nDone! The classification visualization has been created.")