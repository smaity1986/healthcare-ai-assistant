import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

classes = [
    "Normal",
    "Pneumonia"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def predict_image(image):

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        print("Raw Output:", output)

        probs = torch.softmax(output, dim=1)

        print("Probabilities:", probs)

        confidence, prediction = torch.max(
            probs,
            dim=1
        )

    return {
        "prediction": classes[prediction.item()],
        "confidence": round(confidence.item()*100,2),
        "normal_probability":
            round(probs[0][0].item()*100,2),
        "pneumonia_probability":
            round(probs[0][1].item()*100,2)
    }