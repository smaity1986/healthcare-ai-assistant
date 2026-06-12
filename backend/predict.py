import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(1280,2)

model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()

classes = [
    "Normal",
    "Pneumonia"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image):

    image = transform(image).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        prob = torch.softmax(output, dim=1)

        confidence, pred = torch.max(prob,1)

    return {
        "prediction": classes[pred.item()],
        "confidence": round(confidence.item()*100,2)
    }