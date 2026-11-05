
import numpy as np
import onnxruntime as ort
from pathlib import Path
from torchvision import datasets, transforms

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

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"]
)

image, label = dataset[0]
input_name = session.get_inputs()[0].name

outputs = session.run(
    None,
    {input_name: image.unsqueeze(0).numpy().astype(np.float32)}
)[0]

prediction = int(np.argmax(outputs, axis=1)[0])

print("Actual Label:", dataset.classes[label])
print("Predicted Label:", dataset.classes[prediction])
print("Prediction Correct:", prediction == label)
print("Deployment Inference Test Completed Successfully!")
