
from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
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

indices = torch.randperm(
    len(dataset),
    generator=torch.Generator().manual_seed(42)
).tolist()

train_dataset = Subset(dataset, indices[:10000])
val_dataset = Subset(dataset, indices[54000:55000])

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

device = torch.device("cpu")
model = create_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.0005)

best_accuracy = 0.0
epochs = 5

for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(1) == labels.to(device)
        ).sum().item()
        train_total += labels.size(0)

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

    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {train_loss/train_total:.4f} | "
        f"Train Accuracy: {train_acc:.2f}% | "
        f"Validation Accuracy: {val_acc:.2f}%",
        flush=True
    )

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(
            model.state_dict(),
            "models/mobilenetv3_best.pth"
        )

print("Improved Training Completed!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("Best Model Saved Successfully!")
