
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

np.random.seed(42)
sample = np.random.randn(1, 3, 96, 96).astype(np.float32)

results = []

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

    for round_num in range(1, 6):
        for _ in range(30):
            session.run(None, {input_name: sample})

        latencies = []

        for _ in range(200):
            start = time.perf_counter()
            session.run(None, {input_name: sample})
            latencies.append(
                (time.perf_counter() - start) * 1000
            )

        results.append({
            "Model": name,
            "Round": round_num,
            "Mean Latency (ms)": np.mean(latencies),
            "Median Latency (ms)": np.median(latencies),
            "P95 Latency (ms)": np.percentile(latencies, 95)
        })

df = pd.DataFrame(results)

summary = df.groupby("Model", as_index=False).agg(
    Mean_Latency=("Mean Latency (ms)", "mean"),
    Std_Latency=("Mean Latency (ms)", "std"),
    Median_Latency=("Median Latency (ms)", "mean"),
    P95_Latency=("P95 Latency (ms)", "mean")
)

summary = summary.round(4)

print("\nREPEATED CPU BENCHMARK RESULTS")
print(summary.to_string(index=False))

Path("results").mkdir(exist_ok=True)

df.to_csv("results/repeated_benchmark_raw.csv", index=False)
summary.to_csv("results/repeated_benchmark_summary.csv", index=False)

print("\nRepeated Benchmark Completed Successfully!")
