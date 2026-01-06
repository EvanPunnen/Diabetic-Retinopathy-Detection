import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from modelzoo import build_classifier
import torch.nn.functional as F

# Labels for visualization
LABEL_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

def get_transforms(size=224):
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

def plot_prediction(img, pred_label, probabilities, title=None, save_path=None):
    """Plot a single image with its prediction and probability distribution"""
    # Convert tensor to numpy and denormalize
    if torch.is_tensor(img):
        img = denormalize(img)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot image with prediction
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Prediction: {LABEL_NAMES[pred_label]}\nConfidence: {probabilities[pred_label]:.1%}')
    
    # Plot probability bars
    bars = ax2.bar(range(len(probabilities)), probabilities)
    ax2.set_xticks(range(len(probabilities)))
    ax2.set_xticklabels([f"{LABEL_NAMES[i]}\n{prob:.1%}" for i, prob in enumerate(probabilities)], rotation=45)
    ax2.set_ylim(0, 1)
    ax2.set_title('Class Probabilities')
    
    # Color the predicted class
    bars[pred_label].set_color('red')
    
    if title:
        fig.suptitle(title, y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def main():
    # Configuration
    test_dir = "data/test"
    model_path = "models/densenet121_best.pth"
    output_dir = "outputs/test_predictions"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = build_classifier("densenet121", num_classes=5)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)
    
    # Get list of test images
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_images.sort()
    
    print(f"\nProcessing {len(test_images)} test images...")
    transform = get_transforms()
    
    # Store results for summary
    results = []
    
    with torch.no_grad():
        for img_name in test_images:
            # Load and preprocess image
            img_path = os.path.join(test_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_label = outputs.argmax(dim=1).cpu().numpy()[0]
            
            # Save visualization
            save_path = os.path.join(output_dir, f"pred_{os.path.splitext(img_name)[0]}.png")
            plot_prediction(
                img_tensor.cpu().squeeze(),
                pred_label,
                probabilities,
                f"File: {img_name}",
                save_path
            )
            
            # Store results
            results.append({
                'file': img_name,
                'prediction': LABEL_NAMES[pred_label],
                'confidence': probabilities[pred_label],
                'probabilities': probabilities
            })
    
    # Print summary
    print("\nPrediction Summary:")
    print("="*50)
    for result in results:
        print(f"\nFile: {result['file']}")
        print(f"Input Image Location: {os.path.join(test_dir, result['file'])}")
        print(f"Visualization Saved: {os.path.join(output_dir, f'pred_{os.path.splitext(result['file'])[0]}.png')}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("Class Probabilities:")
        for i, prob in enumerate(result['probabilities']):
            print(f"  {LABEL_NAMES[i]}: {prob:.1%}")
    
    print(f"\nSummary:")
    print(f"Input Directory: {os.path.abspath(test_dir)}")
    print(f"Prediction Visualizations Directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()