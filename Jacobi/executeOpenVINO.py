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
import time

import openvino as ov
import numpy as np

def main(grid_size, iterations, runtimes, datatype, device):
   if datatype == "fp32":
      numpy_dtype = np.float32
   elif datatype == "fp16":
      numpy_dtype = np.float16
   else:
      print("Error: datatype not available! Use 'fp32' or 'fp16'.")
      return

   model_filename = f"jacobi{grid_size}_model_{datatype}.xml"
   device_names = {
      "cpu": "CPU",
      "gpu": "GPU",
      "npu": "NPU",
      "all": "AUTO",
   }
   device_name = device_names.get(device.lower())
   if device_name is None:
      print("Error device not found: cpu/gpu/npu/all")
      return

   print("[INFO] Loading the model...")
   core = ov.Core()
   model = core.read_model(model_filename)
   compiled_model = core.compile_model(model, device_name=device_name)

   print("[INFO] Creating Jacobi inputs...")
   coordinates = np.linspace(0, 1, grid_size, dtype=numpy_dtype)
   x_coordinates, y_coordinates = np.meshgrid(coordinates, coordinates, indexing="ij")
   initial = np.exp(
      -50 * ((x_coordinates - 0.5) ** 2 + (y_coordinates - 0.5) ** 2)
   ).astype(numpy_dtype)[None, None, :, :]
   mask = np.ones_like(initial)
   mask[:, :, 0, :] = 0
   mask[:, :, -1, :] = 0
   mask[:, :, :, 0] = 0
   mask[:, :, :, -1] = 0
   inputs = [initial, initial.copy(), mask]

   print("[INFO] Running inference...")
   minTime = math.inf
   start_time = time.perf_counter()
   for _ in range(runtimes):
      start_local_time = time.perf_counter()
      compiled_model(inputs)
      local_elapsed_time = time.perf_counter() - start_local_time
      minTime = min(minTime, local_elapsed_time)
   elapsed_time = (time.perf_counter() - start_time) / runtimes

   print("****************************************************************************************************************************")
   print(f"Jacobi of size {grid_size}x{grid_size} with {iterations} iterations in {datatype} took {elapsed_time:.4f} seconds.")
   print(f"Max Performance took {minTime:.4f} seconds")
   print("****************************************************************************************************************************")

   log_filename = f"jacobi_{grid_size}x{grid_size}_{device}.csv"
   with open(log_filename, "a") as output_file:
      if output_file.tell() == 0:
         output_file.write("datatype;iterations;MeanTime;PeakTime\n")
      output_file.write(f"{datatype};{iterations};{elapsed_time:.4f};{minTime:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpenVINO Jacobi model.")
    parser.add_argument("grid_input", type=int, help="grid size")
    parser.add_argument("iteration_input", type=int, help="Jacobi iterations in the model")
    parser.add_argument("int_input", type=int, help="number of inference runs")
    parser.add_argument("datatype_input", type=str, help="datatype: fp32/fp16")
    parser.add_argument("device_input", type=str, help="device: cpu/gpu/npu/all")

    args = parser.parse_args()
    if args.grid_input <= 0 or args.iteration_input <= 0 or args.int_input <= 0:
        parser.error("grid size, iteration count, and inference run count must be positive")
    main(
        args.grid_input,
        args.iteration_input,
        args.int_input,
        args.datatype_input,
        args.device_input,
    )