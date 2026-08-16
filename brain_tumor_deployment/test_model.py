import json
import torch
import torch.nn as nn
from torchvision import models

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

print("Classes:", class_names)

# Create EfficientNet-B0
model = models.efficientnet_b0(weights=None)

# Get classifier input features
in_features = model.classifier[1].in_features

# Recreate the same classifier used during training
model.classifier = nn.Sequential(
    nn.Dropout(0.3, inplace=True),
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(256),
    nn.Dropout(0.15),
    nn.Linear(256, len(class_names)),
)

# Load trained model
model.load_state_dict(
    torch.load(
        "brain_tumor_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully!")
print("Number of classes:", len(class_names))