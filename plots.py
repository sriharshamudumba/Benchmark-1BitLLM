import matplotlib.pyplot as plt
import numpy as np
import os

# Simulated data for plotting
batch_sizes = [1, 2, 4, 8, 16]
onnx_times = [0.0142, 0.0161, 0.0215, 0.0313, 0.0588]
torch_times = [0.0345, 0.0402, 0.0551, 0.0784, 0.1210]

# Calculate speedup
speedups = [t / o for t, o in zip(torch_times, onnx_times)]

# Create plots
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, onnx_times, marker='o', label='ONNX Inference Time')
plt.plot(batch_sizes, torch_times, marker='s', label='PyTorch Inference Time')
plt.xlabel("Batch Size")
plt.ylabel("Average Inference Time (s)")
plt.title("Inference Time vs Batch Size")
plt.legend()
plt.grid(True)
onnx_plot_path = "results/inference_time_comparison.png"
os.makedirs("results", exist_ok=True)
plt.savefig(onnx_plot_path)

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, speedups, marker='^', color='green')
plt.xlabel("Batch Size")
plt.ylabel("Speedup (Torch / ONNX)")
plt.title("Speedup of ONNX over PyTorch")
plt.grid(True)
speedup_plot_path = "results/speedup_plot.png"
plt.savefig(speedup_plot_path)

import ace_tools as tools; tools.display_dataframe_to_user(name="Simulated Benchmark Results", dataframe={
    "Batch Size": batch_sizes,
    "ONNX Time (s)": onnx_times,
    "PyTorch Time (s)": torch_times,
    "Speedup (x)": speedups
})
