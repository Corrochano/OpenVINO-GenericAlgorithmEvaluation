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
import math
import os
import time

import openvino as ov
import torch

def main(size, runtimes, datatype, device, iterations):
   if size.is_integer():
      ssize = str(int(size))
      grid_size = int(size)
   else:
      ssize = str(size)
      grid_size = int(size)

   if datatype == "fp32":
      torch_dtype = torch.float32
   elif datatype == "fp16":
      torch_dtype = torch.float16
   else:
      print("Error: datatype not available! Use 'fp32' or 'fp16'.")
      return

   model_filename = f"matmul{ssize}_model_{datatype}.xml"
   device_names = {
      "cpu": "CPU",
      "gpu": "GPU",
      "npu": "NPU",
      "all": "AUTO",
   }
   device_name = device_names.get(device.lower())
   if device_name is None:
      print("Error device not found: cpu/gpu/npu/ane/all")
      return

   print("[INFO] Loading the model...")
   core = ov.Core()
   
   model = core.read_model(model_filename)
   model.reshape({inp: [grid_size, grid_size] for inp in model.inputs})

   compiled_model = core.compile_model(model, device_name=device_name)

   print("[INFO] Creating GEMM inputs...")
   matrix_a = (torch.rand(grid_size, grid_size, dtype=torch_dtype)).numpy()
   matrix_b = (torch.rand(grid_size, grid_size, dtype=torch_dtype)).numpy()

   print("[INFO] Running inference...")
   minTime = math.inf
   start_time = time.perf_counter()
   for _ in range(runtimes):
      start_local_time = time.perf_counter()
      compiled_model([matrix_a, matrix_b])
      local_elapsed_time = time.perf_counter() - start_local_time
      minTime = min(minTime, local_elapsed_time)
   elapsed_time = (time.perf_counter() - start_time) / runtimes

   print("****************************************************************************************************************************")
   print(f"GEMM of size {grid_size}x{grid_size} with {iterations} iterations in {datatype} took {elapsed_time:.4f} seconds.")
   print(f"Max Performance took {minTime:.4f} seconds")
   print("****************************************************************************************************************************")

   log_filename = f"gemm_{grid_size}x{grid_size}_{device}.csv"
   if os.path.exists(log_filename):
      log_content = f"{datatype};{elapsed_time:.4f};{minTime:.4f};"
   else:
      log_content = (
         f"datatype;MeanTime;PeakTime;datatype;MeanTime;PeakTime;"
         f"Datatype;TotalMeanEnergy;cpuMeanEnergy;gpuMeanEnergy;aneMeanEnergy;"
         f"Datatype;TotalMaxEnergy;cpuMaxEnergy;gpuMaxEnergy;aneMaxEnergy;"
         f"Datatype;TotalMeanEnergy;cpuMeanEnergy;gpuMeanEnergy;aneMeanEnergy;"
         f"Datatype;TotalMaxEnergy;cpuMaxEnergy;gpuMaxEnergy;aneMaxEnergy;\n"
         f"{datatype};{elapsed_time:.4f};{minTime:.4f};"
      )

   with open(log_filename, "a") as output_file:
      output_file.write(log_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpenVINO GEMM model.")
    parser.add_argument("float_input", type=float, help="matrix size")
    parser.add_argument("iteration_input", type=int, help="iteration count")
    parser.add_argument("int_input", type=int, help="number of inference runs")
    parser.add_argument("datatype_input", type=str, help="datatype: fp32/fp16")
    parser.add_argument("device_input", type=str, help="device: cpu/gpu/npu/all")

    args = parser.parse_args()
    if args.float_input <= 0 or args.int_input <= 0:
        parser.error("matrix size and inference run count must be positive")
    main(
        args.float_input,
        args.int_input,
        args.datatype_input,
        args.device_input,
        args.iteration_input,
    )