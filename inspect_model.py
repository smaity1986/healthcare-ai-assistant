# inspect_model.py

import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(1280,2)

state = torch.load(
    "backend/models/best_model.pth",
    map_location="cpu"
)

model.load_state_dict(state)

print("Model Loaded Successfully")