
from pathlib import Path
from torchvision import datasets, transforms

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)


transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root=str(data_dir),
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=str(data_dir),
    train=False,
    download=True,
    transform=transform
)

print("Fashion-MNIST Dataset Downloaded Successfully!")
print("Training Images:", len(train_dataset))
print("Testing Images:", len(test_dataset))
print("Number of Classes:", len(train_dataset.classes))
print("Classes:", train_dataset.classes)
print("Image Shape:", train_dataset[0][0].shape)
0