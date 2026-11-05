
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)

accuracy = pd.read_csv("results/three_model_comparison.csv")
latency = pd.read_csv("results/three_model_latency.csv")

df = accuracy.merge(latency, on="Model", suffixes=("", "_latency"))

metrics = [
    ("Accuracy (%)", "Accuracy (%)", "accuracy_comparison.png"),
    ("F1 Score", "F1 Score", "f1_comparison.png"),
    ("Size (MB)", "Model Size (MB)", "model_size_comparison.png"),
    ("Mean Latency (ms)", "Mean Latency (ms)", "latency_comparison.png"),
    ("Throughput (FPS)", "Throughput (FPS)", "throughput_comparison.png")
]

for column, ylabel, filename in metrics:
    plt.figure(figsize=(8, 5))

    bars = plt.bar(df["Model"], df[column])

    plt.title(ylabel + " Comparison")
    plt.ylabel(ylabel)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    plt.ylim(0, df[column].max() * 1.15)
    plt.tight_layout()
    plt.savefig("results/" + filename, dpi=300)
    plt.close()

print("All Five Comparison Graphs Saved Successfully!")
