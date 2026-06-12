from pathlib import Path
from torchvision import datasets

dataset_path = Path("dataset/train")

print("Exists:", dataset_path.exists())
print("Absolute:", dataset_path.resolve())

print("\nContents:")

for item in dataset_path.iterdir():
    print(item)

dataset = datasets.ImageFolder(dataset_path)

print("\nClasses:", dataset.classes)
print("Images:", len(dataset))