
import time
import numpy as np
import onnxruntime as ort

model_path = "deployment/mobilenetv3_static_int8.onnx"
input_path = "cpp/test_image.bin"

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    model_path,
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

image = np.fromfile(input_path, dtype=np.float32)
image = image.reshape(1, 3, 96, 96)

for _ in range(10):
    session.run(None, {input_name: image})

latencies = []

for _ in range(100):
    start = time.perf_counter()
    session.run(None, {input_name: image})
    end = time.perf_counter()
    latencies.append((end - start) * 1000)

average = np.mean(latencies)

print("EDGEAI PYTHON INFERENCE BENCHMARK")
print("Warm-up runs: 10")
print("Benchmark runs: 100")
print(f"Average latency: {average:.4f} ms")
print(f"Minimum latency: {np.min(latencies):.4f} ms")
print(f"Maximum latency: {np.max(latencies):.4f} ms")
print(f"Throughput: {1000 / average:.4f} images/sec")
print("Python Benchmark Completed Successfully!")
