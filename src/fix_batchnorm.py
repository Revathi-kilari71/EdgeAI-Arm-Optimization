
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

def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            predictions = model(images).argmax(1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

print(f"Before Calibration: {evaluate():.2f}%")

for module in model.modules():
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.reset_running_stats()

model.train()

with torch.no_grad():
    for images, _ in train_loader:
        model(images)

print(f"After Calibration: {evaluate():.2f}%")

torch.save(
    model.state_dict(),
    "models/mobilenetv3_calibrated.pth"
)

print("BatchNorm Calibration Completed!")
