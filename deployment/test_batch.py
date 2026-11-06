
import numpy as np
import onnxruntime as ort
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

project_dir = Path(__file__).resolve().parent.parent

model_path = project_dir / "deployment" / "mobilenetv3_static_int8.onnx"

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataset = datasets.FashionMNIST(
    root=str(project_dir / "data"),
    train=False,
    download=False,
    transform=transform
)

test_dataset = dataset

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

actual = []
predicted = []

for images, labels in test_loader:
    inputs = images.numpy().astype(np.float32)

    outputs = session.run(
        None,
        {input_name: inputs}
    )[0]

    predictions = np.argmax(outputs, axis=1)

    actual.extend(labels.numpy().tolist())
    predicted.extend(predictions.tolist())

accuracy = accuracy_score(actual, predicted)
f1 = f1_score(actual, predicted, average="weighted", zero_division=0)

print("\nDEPLOYMENT BATCH TEST RESULTS")
print("-" * 40)
print("Total Images Tested:", len(actual))
print("Correct Predictions:", sum(np.array(actual) == np.array(predicted)))
print("Incorrect Predictions:", sum(np.array(actual) != np.array(predicted)))
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")

print("\nDeployment Batch Testing Completed Successfully!")
