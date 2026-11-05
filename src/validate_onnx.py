
import numpy as np
import torch
import onnx
import onnxruntime as ort

from model import create_model

torch.set_num_threads(4)

onnx_path = "models/mobilenetv3_fp32.onnx"

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print("ONNX Model Structure: Valid")

session = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

print("ONNX Runtime Session: Created")

model = create_model()

model.load_state_dict(
    torch.load(
        "models/mobilenetv3_best.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

torch.manual_seed(42)
sample_input = torch.randn(1, 3, 96, 96)

with torch.inference_mode():
    pytorch_output = model(sample_input).numpy()

onnx_output = session.run(
    None,
    {"input": sample_input.numpy()}
)[0]

max_difference = np.max(
    np.abs(pytorch_output - onnx_output)
)

pytorch_prediction = np.argmax(pytorch_output, axis=1)
onnx_prediction = np.argmax(onnx_output, axis=1)

print("PyTorch Prediction:", pytorch_prediction)
print("ONNX Prediction:", onnx_prediction)
print("Maximum Output Difference:", max_difference)

np.testing.assert_allclose(
    pytorch_output,
    onnx_output,
    rtol=1e-3,
    atol=1e-4
)

print("PyTorch and ONNX Outputs Match!")
print("ONNX Validation Completed Successfully!")
