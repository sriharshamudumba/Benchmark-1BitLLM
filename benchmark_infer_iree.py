import os
import time
import pandas as pd
import iree.runtime as ireert
from transformers import AutoTokenizer

device = "local-task"
module_path = "model/bitnet_b1_58-3B.vmfb"
if not os.path.exists(module_path):
    raise FileNotFoundError("Run benchmark_iree.py first to compile IREE model")

config = ireert.Config(device)
vmfb = ireert.VmModule.mmap(config.vm_instance, module_path)
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vmfb)
iree_func = ctx.modules.module.main

os.makedirs("results", exist_ok=True)
with open("results/iree_benchmark.csv", "w") as f:
    f.write("model,batch_size,inference_time\n")
    tokenizer = AutoTokenizer.from_pretrained("1bitLLM/bitnet_b1_58-3B")
    for batch_size in [1, 2, 4, 8]:
        inputs = tokenizer(["This is a test sentence."] * batch_size, return_tensors="pt", padding=True, truncation=True)
        times = []
        for _ in range(10):
            start = time.time()
            _ = iree_func(inputs["input_ids"])
            times.append(time.time() - start)
        avg_time = sum(times) / len(times)
        f.write(f"bitnet_b1_58-3B,{batch_size},{avg_time:.6f}\n")
        print(f"Batch size {batch_size} - IREE avg inference time: {avg_time:.6f}s")
