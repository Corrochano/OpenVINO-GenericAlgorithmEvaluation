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
import json
import math
import os
import subprocess
import time
from pathlib import Path

import openvino as ov
import torch


def read_rapl_counters():
   powercap_path = Path("/sys/class/powercap")
   domain_paths = []
   for package_path in powercap_path.glob("intel-rapl:*"):
      domain_paths.extend([package_path, *package_path.glob("intel-rapl:*")])

   counter_files = [
      domain_path / field
      for domain_path in domain_paths
      for field in ("name", "energy_uj", "max_energy_range_uj")
      if all(
         (domain_path / required_field).is_file()
         for required_field in ("name", "energy_uj", "max_energy_range_uj")
      )
   ]
   if not counter_files:
      return None

   try:
      values = [counter_file.read_text().strip() for counter_file in counter_files]
   except OSError:
      try:
         result = subprocess.run(
            ["sudo", "-n", "cat", *(str(counter_file) for counter_file in counter_files)],
            capture_output=True,
            check=True,
            text=True,
         )
      except (OSError, subprocess.CalledProcessError):
         return None
      values = result.stdout.splitlines()

   if len(values) != len(counter_files):
      return None

   counters = {}
   for index in range(0, len(values), 3):
      try:
         name, energy_uj, max_energy_uj = values[index:index + 3]
         counters[f"{name}_{counter_files[index].parent.name}"] = (
            int(energy_uj),
            int(max_energy_uj),
         )
      except ValueError:
         continue
   return counters or None


def rapl_energy_delta(start_counters, end_counters):
   if not start_counters or not end_counters:
      return None

   energy_joules = {}
   for domain, (start_uj, max_energy_uj) in start_counters.items():
      if domain not in end_counters:
         continue
      end_uj, end_max_energy_uj = end_counters[domain]
      if max_energy_uj != end_max_energy_uj:
         continue
      energy_joules[domain] = round(
         ((end_uj - start_uj) % max_energy_uj) / 1_000_000,
         6,
      )
   return energy_joules or None


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
   energy_start = read_rapl_counters()
   start_time = time.perf_counter()
   for _ in range(runtimes):
      start_local_time = time.perf_counter()
      compiled_model([matrix_a, matrix_b])
      local_elapsed_time = time.perf_counter() - start_local_time
      minTime = min(minTime, local_elapsed_time)
   total_elapsed_time = time.perf_counter() - start_time
   elapsed_time = total_elapsed_time / runtimes
   energy_by_domain = rapl_energy_delta(energy_start, read_rapl_counters())
   power_by_domain_mw = {
      domain: round(energy_joules * 1000 / total_elapsed_time, 3)
      for domain, energy_joules in energy_by_domain.items()
   } if energy_by_domain else None

   print("****************************************************************************************************************************")
   print(f"GEMM of size {grid_size}x{grid_size} with {iterations} iterations in {datatype} took {elapsed_time:.4f} seconds.")
   print(f"Max Performance took {minTime:.4f} seconds")
   if energy_by_domain:
      print(f"RAPL energy for {runtimes} inference run(s): {json.dumps(energy_by_domain, sort_keys=True)} J")
      print(f"Average RAPL power: {json.dumps(power_by_domain_mw, sort_keys=True)} mW")
   else:
      print("RAPL energy unavailable; check sudo authorization and Linux RAPL availability.")
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

   if energy_by_domain:
      energy_filename = f"gemm_energy_{grid_size}x{grid_size}_{device}.csv"
      with open(energy_filename, "a") as output_file:
         if output_file.tell() == 0:
            output_file.write("datatype;iterations;runtimes;EnergyByDomain_J\n")
         output_file.write(
            f"{datatype};{iterations};{runtimes};"
            f"{json.dumps(energy_by_domain, sort_keys=True)}\n"
         )

      power_filename = f"gemm_power_{grid_size}x{grid_size}_{device}.csv"
      with open(power_filename, "a") as output_file:
         if output_file.tell() == 0:
            output_file.write("datatype;iterations;runtimes;AveragePowerByDomain_mW\n")
         output_file.write(
            f"{datatype};{iterations};{runtimes};"
            f"{json.dumps(power_by_domain_mw, sort_keys=True)}\n"
         )

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