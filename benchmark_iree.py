import os
import subprocess

onnx_path = "model/bitnet_b1_58-3B.onnx"
if not os.path.exists(onnx_path):
    raise FileNotFoundError("Run quantize_export.py first to export ONNX model")

output_path = "model/bitnet_b1_58-3B.vmfb"
cmd = [
    "iree-compile",
    onnx_path,
    "--iree-input-type=onnx",
    "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
    "--iree-hal-target-backends=llvm-cpu",
    f"-o={output_path}"
]

print("Compiling to IREE VMFB...")
subprocess.run(cmd, check=True)
print(f"Compiled to IREE VMFB: {output_path}")
