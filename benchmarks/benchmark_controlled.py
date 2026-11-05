
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

options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sessions = {}

for name, path in models.items():
    sessions[name] = ort.InferenceSession(
        path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

for round_num in range(1, 6):
    for name in models:
        session = sessions[name]
        input_name = session.get_inputs()[0].name
        input_data = {input_name: sample}

        for _ in range(30):
            session.run(None, input_data)

        latencies = []

        for _ in range(200):
            start = time.perf_counter()
            session.run(None, input_data)
            latencies.append(
                (time.perf_counter() - start) * 1000
            )

        mean_latency = np.mean(latencies)

        results.append({
            "Model": name,
            "Round": round_num,
            "Mean Latency (ms)": mean_latency,
            "Median Latency (ms)": np.median(latencies),
            "P95 Latency (ms)": np.percentile(latencies, 95),
            "Throughput (FPS)": 1000 / mean_latency
        })

df = pd.DataFrame(results)

summary = df.groupby("Model", as_index=False).agg(
    Mean_Latency=("Mean Latency (ms)", "mean"),
    Std_Latency=("Mean Latency (ms)", "std"),
    Median_Latency=("Median Latency (ms)", "median"),
    P95_Latency=("P95 Latency (ms)", "mean")
)

summary["Throughput_FPS"] = 1000 / summary["Mean_Latency"]

summary["Size_MB"] = summary["Model"].map({
    name: Path(path).stat().st_size / (1024 * 1024)
    for name, path in models.items()
})

summary = summary.round(4)

print("\nCONTROLLED CPU BENCHMARK RESULTS")
print(summary.to_string(index=False))

Path("results").mkdir(exist_ok=True)

df.to_csv("results/controlled_benchmark_raw.csv", index=False)
summary.to_csv("results/controlled_benchmark_summary.csv", index=False)

print("\nControlled CPU Benchmark Completed Successfully!")
