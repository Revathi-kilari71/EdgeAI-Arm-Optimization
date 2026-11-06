
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/performance_comparison.csv")

Path("results").mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))
plt.bar(df["Runtime"], df["Average_Latency_ms"])
plt.title("Python vs C++ Inference Latency")
plt.xlabel("Runtime")
plt.ylabel("Average Latency (ms)")
plt.tight_layout()
plt.savefig("results/latency_comparison.png", dpi=300)


plt.figure(figsize=(8, 5))
plt.bar(df["Runtime"], df["Throughput_images_sec"])
plt.title("Python vs C++ Inference Throughput")
plt.xlabel("Runtime")
plt.ylabel("Images per Second")
plt.tight_layout()
plt.savefig("results/throughput_comparison.png", dpi=300)


python_latency = df.loc[df["Runtime"] == "Python", "Average_Latency_ms"].iloc[0]
cpp_latency = df.loc[df["Runtime"] == "C++", "Average_Latency_ms"].iloc[0]

speedup = python_latency / cpp_latency

print("Python vs C++ Performance Comparison")
print(df.to_string(index=False))
print(f"\nC++ Speedup: {speedup:.2f}x")
print("Graphs saved successfully!")

