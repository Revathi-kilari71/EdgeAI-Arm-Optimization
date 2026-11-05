
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from model import create_model

torch.set_num_threads(4)

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

model = create_model()
model.load_state_dict(
    torch.load(
        "models/mobilenetv3_best.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

actual = []
predicted = []

with torch.inference_mode():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        actual.extend(labels.tolist())
        predicted.extend(predictions.tolist())

print("\nTEST MODEL PERFORMANCE")
print("-" * 40)

print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
print(f"Precision: {precision_score(actual, predicted, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(actual, predicted, average='weighted', zero_division=0):.4f}")
print(f"F1 Score: {f1_score(actual, predicted, average='weighted', zero_division=0):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predicted,
        target_names=test_dataset.classes,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))

print("\nModel Evaluation Completed!")
