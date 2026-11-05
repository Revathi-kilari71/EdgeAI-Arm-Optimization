
import json
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort

project_dir = Path(__file__).resolve().parents[2]
deployment_dir = project_dir / "deployment"
source_model = project_dir / "models" / "mobilenetv3_static_int8.onnx"
target_model = deployment_dir / "mobilenetv3_static_int8.onnx"

deployment_dir.mkdir(exist_ok=True)

if not source_model.exists():
    raise FileNotFoundError(f"Model not found: {source_model}")

shutil.copy2(source_model, target_model)

model = onnx.load(str(target_model))
onnx.checker.check_model(model)

session = ort.InferenceSession(
    str(target_model),
    providers=["CPUExecutionProvider"]
)

inputs = [
    {
        "name": item.name,
        "shape": item.shape,
        "type": item.type
    }
    for item in session.get_inputs()
]

outputs = [
    {
        "name": item.name,
        "shape": item.shape,
        "type": item.type
    }
    for item in session.get_outputs()
]

metadata = {
    "project": "EdgeAI-Arm-Optimization",
    "architecture": "MobileNetV3",
    "dataset": "Fashion-MNIST",
    "model_format": "ONNX",
    "quantization": "Static INT8",
    "input_preprocessing": {
        "resize": [96, 96],
        "channels": 3,
        "normalization_mean": [0.5, 0.5, 0.5],
        "normalization_std": [0.5, 0.5, 0.5]
    },
    "inputs": inputs,
    "outputs": outputs,
    "class_names": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ],
    "target_platform": "ARM CPU",
    "validated_platform": "Windows CPU",
    "arm_hardware_tested": False
}

metadata_path = deployment_dir / "model_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

requirements_path = deployment_dir / "requirements.txt"

requirements_path.write_text(
    "numpy\n"
    "onnxruntime\n"
    "Pillow\n",
    encoding="utf-8"
)

print("\nDEPLOYMENT PACKAGE PREPARATION")
print("Model Copied Successfully!")
print("ONNX Model Structure: Valid")
print("ONNX Runtime Session: Created")
print("Model Size:", round(target_model.stat().st_size / (1024 * 1024), 2), "MB")
print("Input Details:", inputs)
print("Output Details:", outputs)
print("Metadata Saved Successfully!")
print("Requirements Saved Successfully!")
print("\nDeployment Preparation Completed Successfully!")
