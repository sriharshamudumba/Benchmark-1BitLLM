# Benchmarking 1BitLLM with ONNX and IREE

This repository contains a benchmarking framework to evaluate performance, latency, and memory usage of 1-bit quantized large language models (LLMs) using PyTorch, ONNX Runtime, and IREE.

## 🧪 Key Features
- Converts PyTorch models to ONNX and IREE.
- Benchmarks latency and memory usage across 4 batch sizes.
- Compares native PyTorch, ONNX Runtime, and IREE backends.
- Summarizes speedup and efficiency gains.

## 📦 Dependencies

Install required packages:

```bash
pip install torch onnx onnxruntime numpy matplotlib iree-compiler iree-runtime
