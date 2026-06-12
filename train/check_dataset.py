from torchvision import datasets

dataset = datasets.ImageFolder(
    "dataset/chest_xray/train"
)

print(dataset.classes)
print(len(dataset))