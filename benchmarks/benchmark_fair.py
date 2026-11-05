
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

models = {
    "FP32": "models/mobilenetv3_fp32.onnx",
    "INT8": "models/mobilenetv3_int8.onnx"
}

results = []

options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

input_data = np.random.randn(1, 3, 96, 96).astype(np.float32)

for name, path in models.items():
    session = ort.InferenceSession(
        path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    for _ in range(50):
        session.run(None, {input_name: input_data})

    times = []

    for _ in range(200):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        times.append((time.perf_counter() - start) * 1000)

    results.append({
        "Model": name,
        "Mean Latency (ms)": round(np.mean(times), 3),
        "Median Latency (ms)": round(np.median(times), 3),
        "P95 Latency (ms)": round(np.percentile(times, 95), 3),
        "Throughput (FPS)": round(1000 / np.mean(times), 2)
    })

df = pd.DataFrame(results)

print("\nCONTROLLED CPU BENCHMARK")
print(df.to_string(index=False))

Path("results").mkdir(exist_ok=True)
df.to_csv("results/fair_benchmark.csv", index=False)

print("\nBenchmark Completed Successfully!")
