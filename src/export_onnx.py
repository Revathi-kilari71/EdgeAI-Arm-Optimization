
from pathlib import Path
import torch
from model import create_model

torch.set_num_threads(4)

Path("models").mkdir(exist_ok=True)

model = create_model()

model.load_state_dict(
    torch.load(
        "models/mobilenetv3_best.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

dummy_input = torch.randn(1, 3, 96, 96)

torch.onnx.export(
    model,
    dummy_input,
    "models/mobilenetv3_fp32.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamo=False
)

print("ONNX Model Exported Successfully!")

print(
    "ONNX Model Size:",
    f"{Path('models/mobilenetv3_fp32.onnx').stat().st_size / (1024 * 1024):.2f} MB"
)
