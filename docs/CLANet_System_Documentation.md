# CLANet System Documentation
## Comprehensive Guide to Cross-scale Lesion Attention Network for Diabetic Retinopathy Detection

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Module 1: ALA (Adaptive Lesion Attention)](#module-1-ala-adaptive-lesion-attention)
4. [Module 2: CSCA (Cross-Scale Context Attention)](#module-2-csca-cross-scale-context-attention)
5. [Complete Model Architecture](#complete-model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Data Flow](#data-flow)
8. [Key Design Decisions](#key-design-decisions)
9. [Summary](#summary)

---

## Overview

CLANet (Cross-scale Lesion Attention Network) is a deep learning architecture designed specifically for **Diabetic Retinopathy (DR) grading** from retinal fundus images. The system addresses a key challenge in DR detection: **lesions appear at multiple scales** (microaneurysms are tiny, hemorrhages can be large), requiring the model to capture multi-scale features effectively.

### Key Features

- **Multi-scale Feature Extraction**: Captures lesions of varying sizes
- **Adaptive Attention**: Dynamically weights features based on image content
- **Long-range Dependencies**: Models spatial relationships across the entire image
- **Transfer Learning**: Leverages pretrained DenseNet-121 backbone

### DR Grading Scale

| Grade | Name | Description |
|-------|------|-------------|
| 0 | No DR | No visible signs of diabetic retinopathy |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than just microaneurysms |
| 3 | Severe | Extensive intraretinal hemorrhages |
| 4 | Proliferative | Neovascularization and/or vitreous hemorrhage |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLANet Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input Image (224x224x3)                                                   │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────┐                                                       │
│   │   DenseNet-121  │  ◄── Pretrained Backbone (Feature Extractor)         │
│   │   (Backbone)    │                                                       │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │   ALA Module    │  ◄── Adaptive Lesion Attention                       │
│   │ (Multi-scale +  │      • 1x1, 3x3, 5x5 convolutions                     │
│   │  Channel Attn)  │      • Adaptive scale weighting                       │
│   └────────┬────────┘      • Channel attention (SE-like)                    │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │   CSCA Module   │  ◄── Cross-Scale Context Attention                   │
│   │ (Self-Attention │      • Spatial self-attention                         │
│   │   Mechanism)    │      • Long-range dependency modeling                 │
│   └────────┬────────┘      • Learnable residual scaling                     │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │ Global Average  │                                                       │
│   │    Pooling      │                                                       │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │ Fully Connected │  ◄── Classification Head                             │
│   │   (5 classes)   │      Output: [No DR, Mild, Moderate, Severe, PDR]    │
│   └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: ALA (Adaptive Lesion Attention)

### Purpose

The ALA module addresses the **multi-scale nature of DR lesions**:

| Lesion Type | Size | Detection Requirement |
|-------------|------|----------------------|
| Microaneurysms | < 125μm | Fine-grained features (1x1 conv) |
| Hemorrhages | Variable | Medium receptive fields (3x3 conv) |
| Hard/Soft Exudates | Large regions | Wider context (5x5 conv) |

### Architecture Components

#### 1. Multi-Scale Feature Extraction

```python
# Three parallel convolution branches with different kernel sizes:

self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
# • Captures fine-grained, point-wise features
# • Best for detecting tiny microaneurysms
# • No spatial context, pure channel mixing

self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
# • Captures local spatial patterns
# • Good for small hemorrhages, early lesions
# • 3x3 receptive field

self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
# • Captures wider spatial context
# • Good for larger exudates, confluent lesions
# • 5x5 receptive field
```

#### 2. Adaptive Scale Weighting

```python
# Learns to weight each scale based on image content:

self.global_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze spatial dims to 1x1
self.fc1 = nn.Linear(in_channels, in_channels // reduction)  # Compress
self.fc2 = nn.Linear(in_channels // reduction, 3)  # Output 3 weights

# The network learns:
# - For images with microaneurysms → higher weight on 1x1 branch
# - For images with large hemorrhages → higher weight on 5x5 branch
# - Weights are normalized via softmax to sum to 1
```

#### 3. Channel Attention (SE-Block Style)

```python
# Recalibrates channel importance after scale fusion:

self.channel_fc1 = nn.Linear(in_channels, in_channels // reduction)
self.channel_fc2 = nn.Linear(in_channels // reduction, in_channels)

# This learns which feature channels are most important for DR detection
# Some channels might detect blood vessels, others detect lesions
```

### Forward Pass Visualization

```
Input Feature Map x: [B, C, H, W]
         │
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              │
    ┌─────────┐    ┌─────────┐    ┌─────────┐        │
    │ Conv1x1 │    │ Conv3x3 │    │ Conv5x5 │        │
    └────┬────┘    └────┬────┘    └────┬────┘        │
         │              │              │              │
         ▼              ▼              ▼              │
        f1             f2             f3             │
         │              │              │              │
         └──────────────┴──────────────┘              │
                        │                             │
                        ▼                             │
              ┌─────────────────┐                     │
              │ Global Avg Pool │ ◄───────────────────┘
              │   + FC layers   │
              └────────┬────────┘
                       │
                       ▼
              w = [w1, w2, w3]  (softmax normalized)
                       │
                       ▼
         f = w1*f1 + w2*f2 + w3*f3  (weighted fusion)
                       │
                       ▼
              ┌─────────────────┐
              │ Channel Attn   │
              │ (SE-block)     │
              └────────┬────────┘
                       │
                       ▼
                 f = f * ca  (channel recalibration)
                       │
                       ▼
                out = f + x  (residual connection)
```

### Mathematical Formulation

Given input `x ∈ ℝ^(B×C×H×W)`:

1. **Multi-scale extraction:**
   ```
   f₁ = Conv1×1(x)
   f₂ = Conv3×3(x)
   f₃ = Conv5×5(x)
   ```

2. **Global context:**
   ```
   g = GlobalAvgPool(x) ∈ ℝ^(B×C)
   ```

3. **Adaptive weights:**
   ```
   w = Softmax(FC₂(ReLU(FC₁(g)))) ∈ ℝ^(B×3)
   ```

4. **Weighted fusion:**
   ```
   f = w₁·f₁ + w₂·f₂ + w₃·f₃
   ```

5. **Channel attention:**
   ```
   ca = Sigmoid(FC₄(ReLU(FC₃(GAP(f))))) ∈ ℝ^(B×C×1×1)
   f = f ⊙ ca  (element-wise multiplication)
   ```

6. **Residual:**
   ```
   output = f + x
   ```

### Complete ALA Module Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ALA_Module(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ALA_Module, self).__init__()
        
        # Multi-scale convolution layers (for different lesion sizes)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        
        # Global context to generate adaptive weights
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, 3)  # 3 for 1x1, 3x3, 5x5 branches
        
        # Channel attention (like SE block)
        self.channel_fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.channel_fc2 = nn.Linear(in_channels // reduction, in_channels)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Multi-scale feature extraction
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv5x5(x)
        
        # Adaptive weighting based on global context
        g = self.global_pool(x).view(b, c)
        w = self.fc2(self.relu(self.fc1(g)))  # [B, 3]
        w = self.sigmoid(w)
        w = F.softmax(w, dim=1)  # Normalize weights
        
        # Weighted sum of scales
        f = w[:, 0].view(b, 1, 1, 1) * f1 + \
            w[:, 1].view(b, 1, 1, 1) * f2 + \
            w[:, 2].view(b, 1, 1, 1) * f3
        
        # Channel attention refinement
        ca = self.global_pool(f).view(b, c)
        ca = self.channel_fc2(self.relu(self.channel_fc1(ca)))
        ca = self.sigmoid(ca).view(b, c, 1, 1)
        f = f * ca
        
        # Residual connection
        out = f + x
        
        return out
```

---

## Module 2: CSCA (Cross-Scale Context Attention)

### Purpose

The CSCA module captures **long-range spatial dependencies** between different regions of the fundus image. This is crucial because:

- Lesion severity often depends on **spatial distribution** (not just count)
- Features in one area may relate to features in distant areas
- Standard convolutions have limited receptive fields

### Architecture Components

#### Self-Attention Mechanism

```python
# Based on the non-local neural networks / transformer attention

# Query, Key, Value projections (1x1 convolutions for efficiency)
self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
self.key_conv   = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

# Query: "What am I looking for?"
# Key: "What do I contain?"
# Value: "What information do I provide?"
```

#### Learnable Scaling Parameter

```python
self.gamma = nn.Parameter(torch.zeros(1))

# Initialized to 0, so initially output = x (identity)
# Network gradually learns to incorporate attention
# Prevents destabilizing training in early epochs
```

### Forward Pass Visualization

```
Input x: [B, C, H, W]
    │
    ├─────────────────┬─────────────────┬─────────────────┐
    ▼                 ▼                 ▼                 │
┌────────┐       ┌────────┐       ┌────────┐             │
│ Query  │       │  Key   │       │ Value  │             │
│ Conv   │       │ Conv   │       │ Conv   │             │
└───┬────┘       └───┬────┘       └───┬────┘             │
    │                │                │                   │
    ▼                ▼                ▼                   │
[B,C',H,W]      [B,C',H,W]       [B,C,H,W]               │
    │                │                │                   │
    ▼                ▼                ▼                   │
 Reshape          Reshape          Reshape               │
[B,N,C']         [B,C',N]         [B,C,N]                │
    │                │                │                   │
    └───────┬────────┘                │                   │
            ▼                         │                   │
    ┌───────────────┐                 │                   │
    │  Q × K^T      │                 │                   │
    │ [B, N, N]     │ ◄── Attention   │                   │
    │               │     Matrix      │                   │
    └───────┬───────┘                 │                   │
            ▼                         │                   │
    ┌───────────────┐                 │                   │
    │   Softmax     │                 │                   │
    └───────┬───────┘                 │                   │
            │                         │                   │
            └────────────┬────────────┘                   │
                         ▼                                │
                ┌───────────────┐                         │
                │  V × Attn^T   │                         │
                │  [B, C, N]    │                         │
                └───────┬───────┘                         │
                        │                                 │
                        ▼                                 │
                    Reshape                               │
                   [B,C,H,W]                              │
                        │                                 │
                        ▼                                 │
                 out = γ * out + x  ◄─────────────────────┘
                        │              (residual with learnable γ)
                        ▼
                    Output
```

### Attention Matrix Interpretation

```
Attention Matrix A ∈ ℝ^(N×N) where N = H × W

         Position 1  Position 2  ...  Position N
        ┌──────────┬──────────┬─────┬──────────┐
Pos 1   │   a₁₁    │   a₁₂    │ ... │   a₁ₙ    │
        ├──────────┼──────────┼─────┼──────────┤
Pos 2   │   a₂₁    │   a₂₂    │ ... │   a₂ₙ    │
        ├──────────┼──────────┼─────┼──────────┤
  ...   │   ...    │   ...    │ ... │   ...    │
        ├──────────┼──────────┼─────┼──────────┤
Pos N   │   aₙ₁    │   aₙ₂    │ ... │   aₙₙ    │
        └──────────┴──────────┴─────┴──────────┘

aᵢⱼ = How much position j attends to position i
    = Softmax(Qᵢ · Kⱼ)

Example interpretation:
- If aᵢⱼ is high: Feature at position j is relevant to position i
- The network learns that lesions in macula relate to lesions near optic disc
- Captures global context that CNNs miss
```

### Complete CSCA Module Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CSCA_Module(nn.Module):
    """
    Cross-Scale Context Attention (CSCA)
    - Captures contextual relationships between multiscale lesion features.
    - Learns how features from different receptive fields influence each other.
    """

    def __init__(self, in_channels, reduction=16):
        super(CSCA_Module, self).__init__()

        # 1x1 conv to reduce dimensionality
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: enhanced feature map [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Flatten spatial dimensions
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, N, C']
        proj_key   = self.key_conv(x).view(B, -1, H * W)                     # [B, C', N]
        energy = torch.bmm(proj_query, proj_key)                             # [B, N, N]
        attention = self.softmax(energy)                                     # spatial attention map

        proj_value = self.value_conv(x).view(B, -1, H * W)                   # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))              # [B, C, N]
        out = out.view(B, C, H, W)

        # Residual connection with learnable scaling
        out = self.gamma * out + x
        return out
```

---

## Complete Model Architecture

### CLANet with DenseNet-121 Backbone

```python
import torch
import torch.nn as nn
from torchvision import models

from modules.ala import ALA_Module
from modules.csca import CSCA_Module


class CLANet_DenseNet(nn.Module):
    """
    CLANet: Cross-scale Lesion Attention Network
    
    Architecture:
        DenseNet-121 (backbone) → ALA → CSCA → Global Pool → FC → Output
    
    Input: RGB fundus image [B, 3, 224, 224]
    Output: DR grade logits [B, 5]
    """
    
    def __init__(self, num_classes=5, pretrained=True):
        super(CLANet_DenseNet, self).__init__()
        
        # ════════════════════════════════════════════════════════════
        # BACKBONE: DenseNet-121
        # ════════════════════════════════════════════════════════════
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        self.feature_dim = 1024
        
        # ════════════════════════════════════════════════════════════
        # ATTENTION MODULES
        # ════════════════════════════════════════════════════════════
        self.ala = ALA_Module(in_channels=self.feature_dim, reduction=16)
        self.csca = CSCA_Module(in_channels=self.feature_dim, reduction=16)
        
        # ════════════════════════════════════════════════════════════
        # CLASSIFICATION HEAD
        # ════════════════════════════════════════════════════════════
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x):
        # Step 1: Extract features with DenseNet backbone
        features = self.features(x)
        features = torch.relu(features)
        
        # Step 2: Apply Adaptive Lesion Attention
        features = self.ala(features)
        
        # Step 3: Apply Cross-Scale Context Attention
        features = self.csca(features)
        
        # Step 4: Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Step 5: Dropout + Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
```

### Why DenseNet-121?

| Feature | Benefit |
|---------|---------|
| **Dense Connections** | Each layer receives feature maps from all preceding layers |
| **Feature Reuse** | Reduces parameters while maintaining representational power |
| **Gradient Flow** | Direct connections improve gradient propagation |
| **Pretrained Weights** | ImageNet pretraining provides strong initialization |
| **Medical Imaging** | Proven effective for medical image analysis tasks |

---

## Training Pipeline

### Configuration

```python
# HYPERPARAMETERS
BATCH_SIZE = 16      # Number of images per training step
EPOCHS = 50          # Total training iterations over dataset
LR = 1e-4            # Learning rate (0.0001) - conservative for fine-tuning
NUM_CLASSES = 5      # DR grades: 0-No DR, 1-Mild, 2-Moderate, 3-Severe, 4-PDR

# DEVICE SELECTION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CSV Files:                                                     │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │ train_labels.csv │    │  val_labels.csv  │                  │
│  │ image_id, grade  │    │ image_id, grade  │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  IDRiDDataset    │    │  IDRiDDataset    │                  │
│  │  (Train)         │    │  (Validation)    │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │   Transforms     │    │   Transforms     │                  │
│  │ • Resize(224)    │    │ • Resize(224)    │                  │
│  │ • Augmentation   │    │ • ToTensor       │                  │
│  │ • ToTensor       │    │ • Normalize      │                  │
│  │ • Normalize      │    │                  │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │   DataLoader     │    │   DataLoader     │                  │
│  │ batch_size=16    │    │ batch_size=16    │                  │
│  │ shuffle=True     │    │ shuffle=False    │                  │
│  │ num_workers=4    │    │ num_workers=4    │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Loop

```python
for epoch in range(EPOCHS):
    
    # ═══════════════════════════════════════════════════════════
    # TRAINING PHASE
    # ═══════════════════════════════════════════════════════════
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Step 1: Zero gradients
        optimizer.zero_grad()
        
        # Step 2: Forward pass
        outputs = model(images)
        
        # Step 3: Compute loss
        loss = criterion(outputs, labels)
        
        # Step 4: Backward pass
        loss.backward()
        
        # Step 5: Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    # ═══════════════════════════════════════════════════════════
    # VALIDATION PHASE
    # ═══════════════════════════════════════════════════════════
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_acc = correct / total
```

### Loss Function: Cross-Entropy

```
CrossEntropyLoss for multi-class classification:

Given:
- Output logits z = [z₀, z₁, z₂, z₃, z₄] for 5 classes
- True label y (e.g., y = 2 for Moderate DR)

Step 1: Softmax
  p = softmax(z) = [e^z₀/Σ, e^z₁/Σ, e^z₂/Σ, e^z₃/Σ, e^z₄/Σ]

Step 2: Negative Log Likelihood
  Loss = -log(p_y)

Example:
  If p = [0.1, 0.1, 0.6, 0.1, 0.1] and y = 2
  Loss = -log(0.6) ≈ 0.51 (correct prediction, low loss)

  If p = [0.1, 0.1, 0.1, 0.6, 0.1] and y = 2
  Loss = -log(0.1) ≈ 2.30 (wrong prediction, high loss)
```

### Optimizer: Adam

```
Adam (Adaptive Moment Estimation):

For each parameter w:
  m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        # Momentum
  v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       # Velocity
  
  m̂_t = m_t / (1 - β₁^t)                     # Bias correction
  v̂_t = v_t / (1 - β₂^t)
  
  w_t = w_{t-1} - lr * m̂_t / (√v̂_t + ε)     # Update

Default: β₁=0.9, β₂=0.999, ε=1e-8
LR: 1e-4 (0.0001)
```

### Complete Training Script

```python
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from model.clanet import CLANet_DenseNet
from data.idrid_loader import IDRiDDataset, get_transforms

# === CONFIG ===
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(root_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "clanet_idrid.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining Complete — Model saved as {model_path}")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=4)

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
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

    save_model(model)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
```

---

## Data Flow

### Complete Data Flow Through CLANet

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW THROUGH CLANET                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw Fundus Image                                                           │
│  ┌─────────────┐                                                            │
│  │             │  2048 × 1536 × 3 (typical raw size)                       │
│  │   👁️ 🔴     │                                                            │
│  │             │                                                            │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼  Preprocessing (resize, normalize)                               │
│  ┌─────────────┐                                                            │
│  │             │  224 × 224 × 3                                            │
│  │   👁️        │  Normalized: mean=[0.485,0.456,0.406]                     │
│  │             │              std=[0.229,0.224,0.225]                      │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼  DenseNet-121 Backbone                                           │
│  ┌─────────────┐                                                            │
│  │ ░░░░░░░░░░░ │  7 × 7 × 1024                                             │
│  │ ░░░░░░░░░░░ │  High-level features                                      │
│  │ ░░░░░░░░░░░ │  (edges, textures, lesion patterns)                       │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼  ALA Module                                                       │
│  ┌─────────────┐                                                            │
│  │ ▓▓▓▓▓▓▓▓▓▓▓ │  7 × 7 × 1024                                             │
│  │ ▓▓▓▓▓▓▓▓▓▓▓ │  Multi-scale enhanced                                     │
│  │ ▓▓▓▓▓▓▓▓▓▓▓ │  (lesion-focused features)                                │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼  CSCA Module                                                      │
│  ┌─────────────┐                                                            │
│  │ ████████████ │  7 × 7 × 1024                                            │
│  │ ████████████ │  Context-aware features                                  │
│  │ ████████████ │  (global spatial relationships)                          │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼  Global Average Pooling                                          │
│  ┌───┐                                                                      │
│  │ █ │  1 × 1 × 1024 → Flattened to [1024]                                │
│  └───┘                                                                      │
│         │                                                                   │
│         ▼  Fully Connected + Softmax                                       │
│  ┌─────────────────────────────────┐                                        │
│  │  0    1    2    3    4          │                                        │
│  │ 0.85 0.08 0.04 0.02 0.01       │  Probabilities for each DR grade      │
│  └─────────────────────────────────┘                                        │
│         │                                                                   │
│         ▼  Prediction                                                       │
│  ┌─────────────────────────────────┐                                        │
│  │  Grade 0: No Diabetic          │                                        │
│  │           Retinopathy          │  Final output                          │
│  └─────────────────────────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tensor Shapes Through Network

| Layer | Output Shape | Description |
|-------|--------------|-------------|
| Input | [B, 3, 224, 224] | RGB fundus image |
| DenseNet Features | [B, 1024, 7, 7] | High-level feature maps |
| ALA Output | [B, 1024, 7, 7] | Multi-scale attention enhanced |
| CSCA Output | [B, 1024, 7, 7] | Context-aware features |
| Global Pool | [B, 1024, 1, 1] | Spatial aggregation |
| Flatten | [B, 1024] | Feature vector |
| Classifier | [B, 5] | Class logits |
| Softmax | [B, 5] | Class probabilities |

---

## Key Design Decisions

| Component | Design Choice | Rationale |
|-----------|---------------|-----------|
| **Backbone** | DenseNet-121 | Dense connections improve gradient flow; pretrained weights provide strong initialization |
| **ALA Scales** | 1×1, 3×3, 5×5 | Cover microaneurysms (tiny), hemorrhages (medium), exudates (large) |
| **Reduction** | 16 | Balance between expressiveness and computational cost |
| **CSCA γ init** | 0 | Start with identity mapping, gradually learn attention |
| **Loss** | CrossEntropy | Standard for multi-class classification |
| **Optimizer** | Adam @ 1e-4 | Stable convergence for fine-tuning pretrained models |
| **Batch Size** | 16 | Balance GPU memory and gradient stability |
| **Dropout** | 0.5 | Regularization to prevent overfitting |
| **Input Size** | 224×224 | Standard size for pretrained models |

---

## Summary

### Model Overview

| Aspect | Details |
|--------|---------|
| **Task** | 5-class DR grading (No DR, Mild, Moderate, Severe, Proliferative) |
| **Input** | 224×224 RGB fundus images |
| **Backbone** | DenseNet-121 (pretrained on ImageNet) |
| **Key Innovation** | ALA (multi-scale adaptive attention) + CSCA (self-attention) |
| **Parameters** | ~8M total (DenseNet-121 base + attention modules) |
| **Training** | 50 epochs, Adam optimizer, lr=1e-4, batch=16 |
| **Output** | 5-dimensional logits → softmax → class probabilities |
| **Saved Model** | `models/clanet_idrid.pth` |

### Key Innovations

1. **ALA Module**: Addresses multi-scale nature of DR lesions by adaptively weighting features from different receptive fields
2. **CSCA Module**: Captures long-range spatial dependencies that standard CNNs miss
3. **End-to-End Training**: Both attention modules are trained jointly with the backbone

### Performance Considerations

- **GPU Recommended**: CUDA-enabled GPU for faster training
- **Memory**: ~4-6GB GPU memory for batch size 16
- **Training Time**: ~2-4 hours on modern GPU for 50 epochs

---

## File Structure

```
Diabetic-Retinopathy-Detection/
├── src/
│   ├── modules/
│   │   ├── ala.py          # Adaptive Lesion Attention module
│   │   └── csca.py         # Cross-Scale Context Attention module
│   ├── model/
│   │   └── clanet.py       # Complete CLANet model
│   ├── train/
│   │   └── train_clanet.py # Training script
│   └── data/
│       └── idrid_loader.py # Data loading utilities
├── models/
│   └── clanet_idrid.pth    # Saved model weights
├── data/
│   └── iDRID/
│       └── images/
│           ├── train/
│           ├── val/
│           ├── train_labels.csv
│           └── val_labels.csv
└── docs/
    └── CLANet_System_Documentation.md  # This file
```

---

## References

1. **DenseNet**: Huang, G., et al. "Densely Connected Convolutional Networks." CVPR 2017.
2. **Squeeze-and-Excitation**: Hu, J., et al. "Squeeze-and-Excitation Networks." CVPR 2018.
3. **Non-local Neural Networks**: Wang, X., et al. "Non-local Neural Networks." CVPR 2018.
4. **IDRiD Dataset**: Porwal, P., et al. "Indian Diabetic Retinopathy Image Dataset (IDRiD)." 2018.

---

*Document generated for the Diabetic Retinopathy Detection project*
*Last updated: December 2024*