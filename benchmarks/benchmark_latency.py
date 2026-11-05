
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

for name, path in models.items():
    session = ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )

    input_info = session.get_inputs()[0]
    input_name = input_info.name

    input_data = np.random.randn(
        1, 3, 96, 96
    ).astype(np.float32)

    for _ in range(20):
        session.run(None, {input_name: input_data})

    times = []

    for _ in range(100):
        start = time.perf_counter()

        session.run(None, {input_name: input_data})

        end = time.perf_counter()

        times.append((end - start) * 1000)

    results.append({
        "Model": name,
        "Mean Latency (ms)": round(np.mean(times), 3),
        "Median Latency (ms)": round(np.median(times), 3),
        "P95 Latency (ms)": round(np.percentile(times, 95), 3),
        "Throughput (FPS)": round(1000 / np.mean(times), 2),
        "Size (MB)": round(
            Path(path).stat().st_size / (1024 * 1024), 2
        )
    })

df = pd.DataFrame(results)

print("\nFP32 VS INT8 LATENCY BENCHMARK")
print(df.to_string(index=False))

Path("results").mkdir(exist_ok=True)
df.to_csv("results/latency_comparison.csv", index=False)

print("\nLatency Benchmark Completed Successfully!")
