
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

models = {
    "FP32": "models/mobilenetv3_fp32.onnx",
    "Static INT8": "models/mobilenetv3_static_int8.onnx"
}

threads = 2
rounds = 10
warmup_runs = 50
test_runs = 300

rng = np.random.default_rng(42)
sample = rng.standard_normal(
    (1, 3, 96, 96)
).astype(np.float32)

options = ort.SessionOptions()
options.intra_op_num_threads = threads
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

sessions = {}

for name, path in models.items():
    sessions[name] = ort.InferenceSession(
        path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

results = []

for round_num in range(1, rounds + 1):

    order = list(models.keys())

    if round_num % 2 == 0:
        order.reverse()

    for name in order:
        session = sessions[name]
        input_name = session.get_inputs()[0].name
        input_data = {input_name: sample}

        for _ in range(warmup_runs):
            session.run(None, input_data)

        latencies = []

        for _ in range(test_runs):
            start = time.perf_counter()
            session.run(None, input_data)
            elapsed = (
                time.perf_counter() - start
            ) * 1000

            latencies.append(elapsed)

        results.append({
            "Model": name,
            "Round": round_num,
            "Mean_Latency": np.mean(latencies),
            "Median_Latency": np.median(latencies),
            "P95_Latency": np.percentile(latencies, 95)
        })

        print(
            f"Round {round_num} completed: {name}"
        )

df = pd.DataFrame(results)

summary = df.groupby(
    "Model",
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

summary["Size_MB"] = summary["Model"].map({
    name: Path(path).stat().st_size / (1024 * 1024)
    for name, path in models.items()
})

summary = summary.round(4)

print("\nFINAL CPU BENCHMARK RESULTS")
print(summary.to_string(index=False))

fp32 = summary.loc[
    summary["Model"] == "FP32",
    "Mean_Latency"
].iloc[0]

int8 = summary.loc[
    summary["Model"] == "Static INT8",
    "Mean_Latency"
].iloc[0]

speedup = fp32 / int8

improvement = (
    (fp32 - int8) / fp32
) * 100

print(f"\nStatic INT8 Speedup: {speedup:.3f}x")
print(f"Latency Reduction: {improvement:.2f}%")

Path("results").mkdir(exist_ok=True)

df.to_csv(
    "results/final_benchmark_raw.csv",
    index=False
)

summary.to_csv(
    "results/final_benchmark_summary.csv",
    index=False
)

print("\nFinal Benchmark Validation Completed Successfully!")

