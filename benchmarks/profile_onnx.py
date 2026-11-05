
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

Path("results").mkdir(exist_ok=True)

models = {
    "FP32": "models/mobilenetv3_fp32.onnx",
    "INT8": "models/mobilenetv3_int8.onnx"
}

for name, path in models.items():
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.intra_op_num_threads = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    input_data = np.random.randn(1, 3, 96, 96).astype(np.float32)

    for _ in range(10):
        session.run(None, {input_name: input_data})

    for _ in range(100):
        session.run(None, {input_name: input_data})

    profile_path = session.end_profiling()

    with open(profile_path, "r") as f:
        events = json.load(f)

    rows = []

    for event in events:
        if event.get("cat") == "Node":
            rows.append({
                "Operation": event.get("args", {}).get(
                    "op_name", "Unknown"
                ),
                "Duration (ms)": event.get("dur", 0) / 1000
            })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("Operation", as_index=False)
        ["Duration (ms)"]
        .sum()
        .sort_values("Duration (ms)", ascending=False)
    )

    print(f"\n{name} OPERATOR PROFILING")
    print(summary.head(15).to_string(index=False))

    summary.to_csv(
        f"results/{name.lower()}_operator_profile.csv",
        index=False
    )

print("\nONNX Profiling Completed Successfully!")
