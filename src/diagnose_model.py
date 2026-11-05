
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import create_model

torch.set_num_threads(4)

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=transform
)

indices = torch.randperm(
    len(dataset),
    generator=torch.Generator().manual_seed(42)
).tolist()

train_loader = DataLoader(
    Subset(dataset, indices[:2000]),
    batch_size=32,
    shuffle=False
)

val_loader = DataLoader(
    Subset(dataset, indices[54000:54500]),
    batch_size=32,
    shuffle=False
)

model = create_model()
model.load_state_dict(
    torch.load(
        "models/mobilenetv3_pilot.pth",
        map_location="cpu",
        weights_only=True
    )
)

def evaluate(loader, mode):
    if mode == "eval":
        model.eval()
    else:
        model.train()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

print("Train Accuracy (Eval Mode):",
      f"{evaluate(train_loader, 'eval'):.2f}%")

print("Validation Accuracy (Eval Mode):",
      f"{evaluate(val_loader, 'eval'):.2f}%")

print("Validation Accuracy (Train Mode):",
      f"{evaluate(val_loader, 'train'):.2f}%")
