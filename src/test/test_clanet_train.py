import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.clanet import CLANet_DenseNet 

# Dummy input (batch_size=4, channels=3, 224x224)
x = torch.randn(4, 3, 224, 224)

# Dummy labels (5 classes)
y = torch.randint(0, 5, (4,))

# Initialize model
model = CLANet_DenseNet(num_classes=5)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training sanity check (3 mini-epochs)
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

print("✅ Training sanity check complete — if loss decreases, model is ready!")
