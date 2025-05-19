# quantize_export.py
import os
import torch
import requests
from transformers import AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

model_id = "1bitLLM/bitnet_b1_58-large"
onnx_path = "model/bitnet_b1_58-large.onnx"
tokenizer_path = "model/tokenizer.json"

# Create model directory
os.makedirs("model", exist_ok=True)

# Download tokenizer if not already present
if not os.path.exists(tokenizer_path):
    print("Downloading tokenizer...")
    url = f"https://huggingface.co/{model_id}/resolve/main/tokenizer.json"
    r = requests.get(url)
    r.raise_for_status()
    with open(tokenizer_path, "wb") as f:
        f.write(r.content)
    print("Tokenizer saved.")

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path))

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.eval()

# Dummy input
text = "Hello, my name is"
inputs = tokenizer(text, return_tensors="pt")

# Export to ONNX
print("Exporting to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"}
        },
        opset_version=14
    )

print(f"Exported ONNX model to {onnx_path}")
