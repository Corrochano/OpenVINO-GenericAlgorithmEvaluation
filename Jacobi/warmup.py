"""
Copyright 2026 Álvaro Corrochano López

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import torch
import openvino as ov

def main(device):
    print("[INFO] Warming up...")

    print("[INFO] Loading warmup model...")
    device_names = {"cpu": "CPU", "gpu": "GPU", "ane": "AUTO", "all": "AUTO"}
    device_name = device_names.get(device.lower())
    if device_name is None:
        raise ValueError("device must be one of: cpu, gpu, ane, all")
    core = ov.Core()
    wu_model = core.compile_model(
        "jacobi1024_model_fp16.xml", device_name=device_name
    )

    print("[INFO] Creating Jacobi inputs...")
    matrix_a = (torch.rand(1024, 1024, dtype=torch.float16)).numpy()
    matrix_b = (torch.rand(1024, 1024, dtype=torch.float16)).numpy()

    print("[INFO] Running warmup...")
    for _ in range(100):
        wu_model([matrix_a, matrix_b])

    print("+++ Warmup Done +++")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="python warmup.py device")

    # Add arguments
    parser.add_argument("device_input", type=str, help="device: cpu/gpu/ane/all")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are correct (not necessary here as argparse will handle type checks)
    if args.device_input is None:
        print("Error: You must provide all the arguments.")
        parser.print_usage()  # Shows the usage message
    else:
        # Call the main function with provided arguments
        main(args.device_input)
