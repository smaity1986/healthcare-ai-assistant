from torchvision import datasets
import os

print(os.getcwd())

dataset = datasets.ImageFolder("dataset/train")

print("Classes:", dataset.classes)
print("Images:", len(dataset))