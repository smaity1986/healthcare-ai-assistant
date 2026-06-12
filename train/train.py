import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision.models import efficientnet_b0

device = torch.device("cuda")

print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    "dataset/chest_xray/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset/chest_xray/val",
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4
)

model = efficientnet_b0(weights="DEFAULT")

for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(1280, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

best_acc = 0

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss={running_loss:.4f} "
        f"ValAcc={accuracy:.2f}%"
    )

    if accuracy > best_acc:

        best_acc = accuracy

        torch.save(
            model.state_dict(),
            "../backend/models/best_model.pth"
        )

        print("Model Saved")

print("Training Complete")
print("Best Accuracy:", best_acc)