# inspect_predictions.py

from PIL import Image
from backend.predict import predict_image

print("Testing NORMAL image...")

img = Image.open(
    "train/dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
).convert("RGB")

print(predict_image(img))

print("\nTesting PNEUMONIA image...")

img = Image.open(
    "train/dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
).convert("RGB")

print(predict_image(img))