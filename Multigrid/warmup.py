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

import numpy as np
import openvino as ov

from executeOpenVINO import make_inputs


def main(grid_size, datatype, device, runs):
    numpy_dtype = {"fp32": np.float32, "fp16": np.float16}[datatype]
    model_path = f"multigrid{grid_size}_model_{datatype}.xml"
    inputs = make_inputs(grid_size, numpy_dtype)

    core = ov.Core()
    model = core.read_model(model_path)
    if len(model.inputs) != len(inputs):
        raise RuntimeError(
            f"Model expects {len(model.inputs)} inputs, but Multigrid supplies {len(inputs)}"
        )
    model.reshape({
        model_input: list(input_array.shape)
        for model_input, input_array in zip(model.inputs, inputs)
    })
    device_names = {"cpu": "CPU", "gpu": "GPU", "npu": "NPU", "all": "AUTO"}
    compiled_model = core.compile_model(model, device_name=device_names[device])

    print(f"[INFO] Warming up Multigrid {grid_size} ({datatype}) on {device}...")
    for _ in range(runs):
        compiled_model(inputs)
    print("+++ Warmup Done +++")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm up a compiled Multigrid model.")
    parser.add_argument("device_input", choices=("cpu", "gpu", "npu", "all"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--datatype", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.size <= 1 or args.runs <= 0:
        parser.error("size must exceed 1 and runs must be positive")
    main(args.size, args.datatype, args.device_input, args.runs)
