
from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from model import create_model

torch.set_num_threads(4)

Path("models").mkdir(exist_ok=True)

print("Available Quantization Engines:",
      torch.backends.quantized.supported_engines)

available = torch.backends.quantized.supported_engines

if "x86" in available:
    engine = "x86"
elif "fbgemm" in available:
    engine = "fbgemm"
elif "qnnpack" in available:
    engine = "qnnpack"
else:
    raise RuntimeError(
        "No supported INT8 quantization backend found."
    )

torch.backends.quantized.engine = engine

print("Selected Quantization Engine:", engine)

model = create_model()

model.load_state_dict(
    torch.load(
        "models/mobilenetv3_best.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

quantized_model = quantize_dynamic(
    copy.deepcopy(model),
    {nn.Linear},
    dtype=torch.qint8
)

quantized_model.eval()

torch.save(
    quantized_model,
    "models/mobilenetv3_dynamic_int8.pth"
)

original_size = Path(
    "models/mobilenetv3_best.pth"
).stat().st_size / (1024 * 1024)

quantized_size = Path(
    "models/mobilenetv3_dynamic_int8.pth"
).stat().st_size / (1024 * 1024)

print(f"Original Model Size: {original_size:.2f} MB")
print(f"Quantized Model Size: {quantized_size:.2f} MB")

print(
    f"Size Reduction: "
    f"{(1 - quantized_size/original_size)*100:.2f}%"
)

print("Dynamic INT8 Quantization Completed!")
