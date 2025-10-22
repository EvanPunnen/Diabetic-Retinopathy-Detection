import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import the model and dataset
from model.clanet import CLANet_DenseNet
from data.idrid_loader import IDRiDDataset, get_transforms

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
MODEL_PATH = "models/clanet_idrid.pth"
BATCH_SIZE = 16
OUTPUT_DIR = "outputs/model_evaluation"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_path):
    """Load the trained CLANet model"""
    model = CLANet_DenseNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model = model.to(DEVICE)
    return model

def evaluate_model(model, dataloader):
    """Evaluate model on test data"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating model"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            # Get predicted class and probabilities
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert lists to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return all_labels, all_preds, all_probs

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix as a heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_probs):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # Convert to one-hot encoding for multi-class ROC
    y_true_onehot = np.zeros((len(y_true), NUM_CLASSES))
    for i, val in enumerate(y_true):
        y_true_onehot[i, val] = 1
    
    # Plot ROC curve for each class
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_per_class(class_report):
    """Plot precision, recall, and F1 score for each class"""
    metrics = ['precision', 'recall', 'f1-score']
    
    # Extract data from classification report
    data = []
    for i, class_name in enumerate(CLASS_NAMES):
        # The classification_report uses class_names as keys
        if class_name in class_report:
            class_data = class_report[class_name]
        else:
            # Fall back to index if class_name is not a key
            class_data = class_report[str(i)]
        
        data.append([class_data[metric] for metric in metrics])
    
    # Create DataFrame
    df = pd.DataFrame(data, index=CLASS_NAMES, columns=metrics)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', ax=plt.gca())
    plt.title('Precision, Recall, and F1-Score by Class')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_html_report(accuracy, class_report, cm):
    """Generate an HTML report of the model evaluation"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CLANet Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .metric-value {{ font-weight: bold; color: #007bff; }}
            .container {{ margin-bottom: 30px; }}
            .images {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
            .image-container {{ margin: 10px; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>CLANet Model Evaluation Report</h1>
        <div class="container">
            <h2>Overall Performance</h2>
            <p>Accuracy: <span class="metric-value">{accuracy:.4f}</span></p>
            <p>Weighted Precision: <span class="metric-value">{class_report['weighted avg']['precision']:.4f}</span></p>
            <p>Weighted Recall: <span class="metric-value">{class_report['weighted avg']['recall']:.4f}</span></p>
            <p>Weighted F1-Score: <span class="metric-value">{class_report['weighted avg']['f1-score']:.4f}</span></p>
        </div>

        <div class="container">
            <h2>Class-wise Performance</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
    """

    # Add row for each class
    for i, class_name in enumerate(CLASS_NAMES):
        # Get the appropriate key for the class report
        if class_name in class_report:
            key = class_name
        else:
            key = str(i)
            
        html += f"""
                <tr>
                    <td>{CLASS_NAMES[i]} (Class {i})</td>
                    <td>{class_report[key]['precision']:.4f}</td>
                    <td>{class_report[key]['recall']:.4f}</td>
                    <td>{class_report[key]['f1-score']:.4f}</td>
                    <td>{class_report[key]['support']}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="container">
            <h2>Confusion Matrix</h2>
            <div class="image-container">
                <img src="confusion_matrix.png" alt="Confusion Matrix" style="max-width:100%; height:auto;">
            </div>
        </div>

        <div class="container">
            <h2>ROC Curves</h2>
            <div class="image-container">
                <img src="roc_curves.png" alt="ROC Curves" style="max-width:100%; height:auto;">
            </div>
        </div>

        <div class="container">
            <h2>Precision, Recall, and F1-Score by Class</h2>
            <div class="image-container">
                <img src="precision_recall_f1.png" alt="Precision, Recall, and F1-Score" style="max-width:100%; height:auto;">
            </div>
        </div>

        <div class="container">
            <h3>Model Information</h3>
            <p>Model: CLANet_DenseNet</p>
            <p>Model Path: {MODEL_PATH}</p>
            <p>Number of Classes: {NUM_CLASSES}</p>
            <p>Evaluation Date: <script>document.write(new Date().toLocaleDateString())</script></p>
        </div>
    </body>
    </html>
    """

    # Write HTML to file
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.html'), 'w') as f:
        f.write(html)

def main():
    print(f"Using device: {DEVICE}")
    
    # Load test dataset
    print("Loading validation dataset...")
    val_dataset = IDRiDDataset(
        "data/iDRID/images/val_labels.csv",
        "data/iDRID/images/val",
        transform=get_transforms(train=False)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # Evaluate model
    print("Evaluating model performance...")
    y_true, y_pred, y_probs = evaluate_model(model, val_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_report = classification_report(
        y_true, y_pred, 
        target_names=CLASS_NAMES,
        output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot ROC curves
    plot_roc_curves(y_true, y_probs)
    
    # Plot precision, recall, F1 score per class
    plot_precision_recall_per_class(class_report)
    
    # Generate HTML report
    generate_html_report(accuracy, class_report, cm)
    
    print(f"\nEvaluation complete. Results saved in {OUTPUT_DIR}/")
    print(f"Open {OUTPUT_DIR}/evaluation_report.html in a web browser to view the complete report")

if __name__ == "__main__":
    main()