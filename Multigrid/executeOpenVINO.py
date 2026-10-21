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
import subprocess
import time
from pathlib import Path

import numpy as np
import openvino as ov


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


def make_inputs(grid_size, numpy_dtype):
    coordinates = np.linspace(0, 1, grid_size, dtype=numpy_dtype)
    x_coordinates, y_coordinates = np.meshgrid(coordinates, coordinates, indexing="ij")

    masks = []
    level_size = grid_size
    for _ in range(10):
        mask = np.ones((1, 1, level_size, level_size), dtype=numpy_dtype)
        mask[:, :, 0, :] = 0
        mask[:, :, -1, :] = 0
        mask[:, :, :, 0] = 0
        mask[:, :, :, -1] = 0
        masks.append(mask)
        level_size = max(1, level_size // 2)

    return [x_coordinates, y_coordinates, *masks]


def main(grid_size, iterations, runtimes, datatype, device):
    if datatype == "fp32":
        numpy_dtype = np.float32
    elif datatype == "fp16":
        numpy_dtype = np.float16
    else:
        raise ValueError("datatype must be fp32 or fp16")

    device_names = {"cpu": "CPU", "gpu": "GPU", "npu": "NPU", "all": "AUTO"}
    device_name = device_names.get(device.lower())
    if device_name is None:
        raise ValueError("device must be cpu, gpu, npu, or all")

    model_filename = f"multigrid{grid_size}_model_{datatype}.xml"
    print("[INFO] Loading the Multigrid model...")
    core = ov.Core()
    model = core.read_model(model_filename)

    inputs = make_inputs(grid_size, numpy_dtype)
    if len(model.inputs) != len(inputs):
        raise RuntimeError(
            f"Model expects {len(model.inputs)} inputs, but Multigrid supplies {len(inputs)}"
        )
    model.reshape({
        model_input: list(input_array.shape)
        for model_input, input_array in zip(model.inputs, inputs)
    })
    compiled_model = core.compile_model(model, device_name=device_name)

    print("[INFO] Running inference...")
    min_time = math.inf
    energy_start = read_rapl_counters()
    start_time = time.perf_counter()
    for _ in range(runtimes):
        start_local_time = time.perf_counter()
        compiled_model(inputs)
        min_time = min(min_time, time.perf_counter() - start_local_time)
    total_elapsed_time = time.perf_counter() - start_time
    mean_time = total_elapsed_time / runtimes
    energy_by_domain = rapl_energy_delta(energy_start, read_rapl_counters())
    power_by_domain_mw = {
        domain: round(energy_joules * 1000 / total_elapsed_time, 3)
        for domain, energy_joules in energy_by_domain.items()
    } if energy_by_domain else None

    print("****************************************************************************************************************************")
    print(
        f"Multigrid {grid_size}x{grid_size}, {iterations} iterations, {datatype}: "
        f"{mean_time:.4f} s mean; {min_time:.4f} s best"
    )
    if energy_by_domain:
        print(f"RAPL energy for {runtimes} inference run(s): {json.dumps(energy_by_domain, sort_keys=True)} J")
        print(f"Average RAPL power: {json.dumps(power_by_domain_mw, sort_keys=True)} mW")
    else:
        print("RAPL energy unavailable; check sudo authorization and Linux RAPL availability.")
    print("****************************************************************************************************************************")

    timing_filename = f"multigrid_{grid_size}x{grid_size}_{device}.csv"
    with open(timing_filename, "a") as output_file:
        if output_file.tell() == 0:
            output_file.write("datatype;iterations;MeanTime;PeakTime\n")
        output_file.write(f"{datatype};{iterations};{mean_time:.4f};{min_time:.4f}\n")

    if energy_by_domain:
        energy_filename = f"multigrid_energy_{grid_size}x{grid_size}_{device}.csv"
        with open(energy_filename, "a") as output_file:
            if output_file.tell() == 0:
                output_file.write("datatype;iterations;runtimes;EnergyByDomain_J\n")
            output_file.write(
                f"{datatype};{iterations};{runtimes};"
                f"{json.dumps(energy_by_domain, sort_keys=True)}\n"
            )

        power_filename = f"multigrid_power_{grid_size}x{grid_size}_{device}.csv"
        with open(power_filename, "a") as output_file:
            if output_file.tell() == 0:
                output_file.write("datatype;iterations;runtimes;AveragePowerByDomain_mW\n")
            output_file.write(
                f"{datatype};{iterations};{runtimes};"
                f"{json.dumps(power_by_domain_mw, sort_keys=True)}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a compiled OpenVINO Multigrid model.")
    parser.add_argument("grid_input", type=int, help="grid size")
    parser.add_argument("iteration_input", type=int, help="iterations compiled into the model")
    parser.add_argument("runs_input", type=int, help="number of inference runs")
    parser.add_argument("datatype_input", choices=("fp32", "fp16"))
    parser.add_argument("device_input", choices=("cpu", "gpu", "npu", "all"))
    args = parser.parse_args()
    if args.grid_input <= 1 or args.iteration_input <= 0 or args.runs_input <= 0:
        parser.error("grid size must exceed 1; iterations and runs must be positive")
    main(
        args.grid_input,
        args.iteration_input,
        args.runs_input,
        args.datatype_input,
        args.device_input,
    )
