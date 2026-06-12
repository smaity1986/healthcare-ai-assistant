import torch
from torchvision.models import efficientnet_b0

device = "cuda"

print("Loading model...")

model = efficientnet_b0(weights="DEFAULT")
model = model.to(device)
model.eval()

print("Creating input...")

dummy = torch.randn(1, 3, 224, 224, device=device)

print("Running inference...")

with torch.no_grad():
    output = model(dummy)

print("Output shape:", output.shape)
print("Output device:", output.device)

print("\nSUCCESS")