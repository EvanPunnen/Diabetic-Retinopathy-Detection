import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
import torch.nn.functional as F

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import the model
from model.clanet import CLANet_DenseNet

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
MODEL_PATH = "models/clanet_idrid.pth"
TEST_IMAGE_DIR = "data/test"
OUTPUT_DIR = "outputs/test_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_clahe(img):
    """Apply CLAHE enhancement to improve feature visibility"""
    # Convert PIL image to numpy array and then to LAB color space
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel_enhanced = clahe.apply(l_channel)
    
    # Merge the enhanced L channel with the original A and B channels
    lab_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Convert numpy array back to PIL Image
    return Image.fromarray(enhanced)

def preprocess_image(image_path):
    """Load and preprocess an image for the model"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Apply CLAHE enhancement
    img = apply_clahe(img)
    
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img, img_tensor

def load_model(model_path):
    """Load the trained CLANet model"""
    model = CLANet_DenseNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model = model.to(DEVICE)
    return model

def predict(model, img_tensor):
    """Generate predictions with the model"""
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    return {
        'class_id': predicted_class,
        'class_name': CLASS_NAMES[predicted_class],
        'probabilities': probabilities.cpu().numpy()
    }

def visualize_prediction(image, prediction, output_path):
    """Create a visualization of the prediction results"""
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the image
    ax1.imshow(image)
    ax1.set_title(f"Prediction: {prediction['class_name']}")
    ax1.axis('off')
    
    # Display probability distribution
    probs = prediction['probabilities']
    ax2.bar(range(len(CLASS_NAMES)), probs, color='skyblue')
    ax2.set_xticks(range(len(CLASS_NAMES)))
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax2.set_ylabel('Probability')
    ax2.set_title('Class Probabilities')
    
    # Add probability values on top of bars
    for i, prob in enumerate(probs):
        ax2.text(i, prob + 0.01, f"{prob:.2%}", ha='center')
    
    # Highlight the predicted class
    ax2.get_children()[prediction['class_id']].set_color('orange')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def main():
    print(f"Using device: {DEVICE}")
    
    # Find test images
    test_images = [f for f in os.listdir(TEST_IMAGE_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_images:
        print(f"No images found in {TEST_IMAGE_DIR}")
        return
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # Process each test image
    for img_file in test_images:
        img_path = os.path.join(TEST_IMAGE_DIR, img_file)
        print(f"\nProcessing: {img_path}")
        
        # Preprocess image
        original_img, img_tensor = preprocess_image(img_path)
        
        # Generate prediction
        prediction = predict(model, img_tensor)
        print(f"Prediction: {prediction['class_name']} (Class {prediction['class_id']})")
        print(f"Confidence: {prediction['probabilities'][prediction['class_id']]:.2%}")
        
        # Visualize
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_file)[0]}_result.png")
        visualize_prediction(original_img, prediction, output_path)

if __name__ == "__main__":
    main()