
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import create_model

torch.manual_seed(42)
torch.set_num_threads(4)

Path("models").mkdir(exist_ok=True)

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

generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(dataset), generator=generator).tolist()

train_indices = indices[:54000]
val_indices = indices[54000:]

train_dataset = Subset(dataset, train_indices[:2000])
val_dataset = Subset(dataset, val_indices[:500])

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0
)

device = torch.device("cpu")
model = create_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

epochs = 2

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            val_correct += (
                outputs.argmax(1) == labels.to(device)
            ).sum().item()
            val_total += labels.size(0)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {running_loss/total:.4f} | "
        f"Train Accuracy: {100*correct/total:.2f}% | "
        f"Validation Accuracy: {100*val_correct/val_total:.2f}%"
    )

torch.save(model.state_dict(), "models/mobilenetv3_pilot.pth")
print("Pilot Training Completed Successfully!")
print("Model saved: models/mobilenetv3_pilot.pth")
