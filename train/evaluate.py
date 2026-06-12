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

test_dataset = datasets.ImageFolder(
    "dataset/chest_xray/test",
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
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

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")