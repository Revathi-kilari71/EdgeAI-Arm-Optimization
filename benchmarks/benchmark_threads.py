
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

thread_counts = [1, 2, 4, 8]
rounds = 5
warmup_runs = 30
test_runs = 200

rng = np.random.default_rng(42)
sample = rng.standard_normal(
    (1, 3, 96, 96)
).astype(np.float32)

results = []

for threads in thread_counts:
    for name, path in models.items():

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(
            path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name
        input_data = {input_name: sample}

        for round_num in range(1, rounds + 1):

            for _ in range(warmup_runs):
                session.run(None, input_data)

            latencies = []

            for _ in range(test_runs):
                start = time.perf_counter()
                session.run(None, input_data)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

            results.append({
                "Model": name,
                "Threads": threads,
                "Round": round_num,
                "Mean_Latency": np.mean(latencies),
                "Median_Latency": np.median(latencies),
                "P95_Latency": np.percentile(latencies, 95)
            })

        print(f"Completed: {name} | Threads: {threads}")

df = pd.DataFrame(results)

summary = df.groupby(
    ["Model", "Threads"],
    as_index=False
).agg(
    Mean_Latency=("Mean_Latency", "mean"),
    Std_Latency=("Mean_Latency", "std"),
    Median_Latency=("Median_Latency", "median"),
    P95_Latency=("P95_Latency", "mean")
)

summary["Throughput_FPS"] = (
    1000 / summary["Mean_Latency"]
)

summary = summary.round(4)

print("\nCPU THREAD OPTIMIZATION RESULTS")
print(summary.to_string(index=False))

best = summary.loc[
    summary.groupby("Model")["Mean_Latency"].idxmin()
]

print("\nBEST THREAD CONFIGURATION FOR EACH MODEL")
print(best.to_string(index=False))

Path("results").mkdir(exist_ok=True)

df.to_csv(
    "results/thread_benchmark_raw.csv",
    index=False
)

summary.to_csv(
    "results/thread_benchmark_summary.csv",
    index=False
)

best.to_csv(
    "results/best_thread_configuration.csv",
    index=False
)

print("\nCPU Thread Optimization Completed Successfully!")
