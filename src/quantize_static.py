
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat
)
from torchvision import datasets, transforms
from pathlib import Path

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

class FashionCalibrationReader(CalibrationDataReader):
    def __init__(self, model_path, dataset, num_samples=200):
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = session.get_inputs()[0].name
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.index = 0

    def get_next(self):
        if self.index >= self.num_samples:
            return None

        image, _ = self.dataset[self.index]
        self.index += 1

        return {
            self.input_name: image.unsqueeze(0).numpy().astype(np.float32)
        }

    def rewind(self):
        self.index = 0

input_model = "models/mobilenetv3_fp32.onnx"
output_model = "models/mobilenetv3_static_int8.onnx"

reader = FashionCalibrationReader(
    input_model,
    dataset,
    num_samples=200
)

quantize_static(
    model_input=input_model,
    model_output=output_model,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=False
)

session = ort.InferenceSession(
    output_model,
    providers=["CPUExecutionProvider"]
)

print("Static INT8 Model Created Successfully!")
print("Model Size:",
      round(Path(output_model).stat().st_size / (1024 * 1024), 2),
      "MB")
print("ONNX Runtime Validation Successful!")
