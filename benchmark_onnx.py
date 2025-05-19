import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

# === Load Tokenizer ===
tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file("model/tokenizer.json"))

# === Load ONNX Model ===
model_path = "model/bitnet_b1_58-large.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# === Print input names the model expects ===
expected_inputs = [inp.name for inp in session.get_inputs()]
print("Model expects input names:", expected_inputs)

# === Prepare input ===
text = "Deep learning models are powerful"
tokens = tokenizer(text, return_tensors="np")
input_feed = {
    "input_ids": tokens["input_ids"].astype(np.int64),
    "attention_mask": tokens["attention_mask"].astype(np.int64)
}

# === Filter input feed to only use valid keys ===
filtered_input = {k: v for k, v in input_feed.items() if k in expected_inputs}

# === Handle extra inputs like 'onnx::Neg_2' with dummy tensor ===
for name in expected_inputs:
    if name not in filtered_input:
        print(f"Filling dummy input for: {name}")
        filtered_input[name] = np.zeros_like(tokens["input_ids"], dtype=np.int64)

# === Warm-up ===
print(" Warming up...")
_ = session.run(None, filtered_input)

# === Benchmark ===
times = []
for _ in range(10):
    start = time.time()
    _ = session.run(None, filtered_input)
    times.append(time.time() - start)

print(f" Average inference time: {np.mean(times):.6f} seconds")
