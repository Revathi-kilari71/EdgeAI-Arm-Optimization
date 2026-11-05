
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = Path("models/mobilenetv3_fp32.onnx")
output_model = Path("models/mobilenetv3_int8.onnx")

if not input_model.exists():
    raise FileNotFoundError("FP32 ONNX model not found!")

quantize_dynamic(
    model_input=str(input_model),
    model_output=str(output_model),
    weight_type=QuantType.QInt8,
    per_channel=False
)

original_size = input_model.stat().st_size / (1024 * 1024)
quantized_size = output_model.stat().st_size / (1024 * 1024)

reduction = (1 - quantized_size / original_size) * 100

print("ONNX Dynamic Quantization Completed!")
print(f"Original Model Size: {original_size:.2f} MB")
print(f"Quantized Model Size: {quantized_size:.2f} MB")
print(f"Model Size Reduction: {reduction:.2f}%")
print("Quantized Model Saved:", output_model)
