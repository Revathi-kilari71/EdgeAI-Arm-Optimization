
import numpy as np
import pandas as pd
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

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
    batch_size=1,
    shuffle=False,
    num_workers=0
)

models = {
    "FP32": "models/mobilenetv3_fp32.onnx",
    "Dynamic INT8": "models/mobilenetv3_int8.onnx",
    "Static INT8": "models/mobilenetv3_static_int8.onnx"
}

results = []

for name, path in models.items():
    session = ort.InferenceSession(
        path,
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

        actual.extend(labels.numpy())
        predicted.extend(predictions)

    accuracy = accuracy_score(actual, predicted)
    f1 = f1_score(actual, predicted, average="weighted")

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "F1 Score": round(f1, 4),
        "Size (MB)": round(
            Path(path).stat().st_size / (1024 * 1024), 2
        )
    })

df = pd.DataFrame(results)

print("\nTHREE MODEL ACCURACY COMPARISON")
print(df.to_string(index=False))

Path("results").mkdir(exist_ok=True)
df.to_csv("results/three_model_comparison.csv", index=False)

print("\nComparison Saved Successfully!")
