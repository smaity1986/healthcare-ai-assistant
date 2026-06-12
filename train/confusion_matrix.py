import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision.models import efficientnet_b0

device = "cuda"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    "dataset/chest_xray/test",
    transform=transform
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 2)

model.load_state_dict(
    torch.load(
        "backend/models/best_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

cm = [[0,0],[0,0]]

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1).cpu()

        for p, l in zip(preds, labels):
            cm[l][p] += 1

print("Classes:", dataset.classes)
print()
print("Confusion Matrix")
print(cm)