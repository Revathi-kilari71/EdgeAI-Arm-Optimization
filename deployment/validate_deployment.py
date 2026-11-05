
import json
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

deployment_dir = Path(__file__).resolve().parent

model_path = deployment_dir / "mobilenetv3_static_int8.onnx"
metadata_path = deployment_dir / "model_metadata.json"
requirements_path = deployment_dir / "requirements.txt"

print("\nFINAL DEPLOYMENT VALIDATION")
print("-" * 45)

files = {
    "ONNX Model": model_path,
    "Metadata": metadata_path,
    "Requirements": requirements_path
}

for name, path in files.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    print(f"{name}: Found")

model = onnx.load(str(model_path))
onnx.checker.check_model(model)

print("ONNX Model Structure: Valid")

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print("ONNX Runtime Session: Created")
print("Input Shape:", input_info.shape)
print("Output Shape:", output_info.shape)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Metadata JSON: Valid")

requirements = requirements_path.read_text(encoding="utf-8")

if not requirements.strip():
    raise ValueError("Requirements file is empty")

print("Requirements File: Valid")

if input_info.shape != [1, 3, 96, 96]:
    raise ValueError("Unexpected input shape")

if output_info.shape != [1, 10]:
    raise ValueError("Unexpected output shape")

sample = np.zeros((1, 3, 96, 96), dtype=np.float32)

input_name = input_info.name

output = session.run(
    None,
    {input_name: sample}
)[0]

if output.shape != (1, 10):
    raise ValueError("Invalid inference output shape")

if not np.isfinite(output).all():
    raise ValueError("Inference output contains invalid values")

prediction = int(np.argmax(output, axis=1)[0])

print("Test Prediction Class:", prediction)
print("Inference Output: Valid")
print("Model Size (MB):", round(model_path.stat().st_size / (1024 * 1024), 2))

print("\nFinal Deployment Validation Completed Successfully!")
