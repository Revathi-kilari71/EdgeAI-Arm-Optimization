
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

models = {
    "FP32": "models/mobilenetv3_fp32.onnx",
    "Dynamic INT8": "models/mobilenetv3_int8.onnx",
    "Static INT8": "models/mobilenetv3_static_int8.onnx"
}

results = []

np.random.seed(42)
sample = np.random.randn(1, 3, 96, 96).astype(np.float32)

for name, path in models.items():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    for _ in range(30):
        session.run(None, {input_name: sample})

    latencies = []

    for _ in range(200):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    mean_latency = np.mean(latencies)

    results.append({
        "Model": name,
        "Mean Latency (ms)": round(mean_latency, 3),
        "Median Latency (ms)": round(np.median(latencies), 3),
        "P95 Latency (ms)": round(np.percentile(latencies, 95), 3),
        "Throughput (FPS)": round(1000 / mean_latency, 2),
        "Size (MB)": round(
            Path(path).stat().st_size / (1024 * 1024), 2
        )
    })

df = pd.DataFrame(results)

print("\nTHREE MODEL CPU LATENCY BENCHMARK")
print(df.to_string(index=False))

Path("results").mkdir(exist_ok=True)
df.to_csv("results/three_model_latency.csv", index=False)

print("\nLatency Benchmark Completed Successfully!")
