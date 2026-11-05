
import numpy as np
import onnx
import onnxruntime as ort

fp32_path = "models/mobilenetv3_fp32.onnx"
int8_path = "models/mobilenetv3_int8.onnx"

onnx.checker.check_model(int8_path)
print("INT8 ONNX Model Structure: Valid")

fp32_session = ort.InferenceSession(
    fp32_path,
    providers=["CPUExecutionProvider"]
)

int8_session = ort.InferenceSession(
    int8_path,
    providers=["CPUExecutionProvider"]
)

input_data = np.random.default_rng(42).random(
    (1, 3, 96, 96)
).astype(np.float32)

fp32_name = fp32_session.get_inputs()[0].name
int8_name = int8_session.get_inputs()[0].name

fp32_output = fp32_session.run(
    None, {fp32_name: input_data}
)[0]

int8_output = int8_session.run(
    None, {int8_name: input_data}
)[0]

print("FP32 Prediction:", np.argmax(fp32_output, axis=1))
print("INT8 Prediction:", np.argmax(int8_output, axis=1))

difference = np.max(np.abs(fp32_output - int8_output))

print("Maximum Output Difference:", difference)
print("INT8 ONNX Validation Completed Successfully!")
