import os
import pandas as pd
import matplotlib.pyplot as plt

onnx_path = "results/onnx_benchmark.csv"
if not os.path.exists(onnx_path):
    print(f"File {onnx_path} not found. Run benchmark_onnx.py first.")
    exit()

onnx_df = pd.read_csv(onnx_path)

plt.figure(figsize=(8, 5))
plt.bar(onnx_df["model"], onnx_df["inference_time"], color="skyblue")
plt.xlabel("Model")
plt.ylabel("Inference Time (seconds)")
plt.title("ONNX Inference Benchmark")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("results/onnx_benchmark_plot.png")
plt.show()
