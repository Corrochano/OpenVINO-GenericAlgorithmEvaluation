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
import torch
from torch import nn


class MyMachine(nn.Module):

  def __init__(self):
    super(MyMachine, self).__init__()

  def forward(self, A, B):
    x = torch.matmul(A, B)
    return x


def main(matrix_size, datatype):
  print("Starting the test for matrix multiplication with OpenVINO and PyTorch...")
  model = MyMachine()
  model.eval()

  if datatype == "fp32":
    torchfloat = torch.float32
    compress_fp16 = False
  elif datatype == "fp16":
    torchfloat = torch.float16
    compress_fp16 = True
  else:
    print("Error: datatype not available! Use 'fp32' or 'fp16'.")
    return

  print("--------------------------")
  print("Creating matrix...")
  print("--------------------------\n")
  example_A = (torch.rand(matrix_size, matrix_size, dtype=torchfloat)).numpy()
  example_B = (torch.rand(matrix_size, matrix_size, dtype=torchfloat)).numpy()

  print("--------------------------")
  print("Testing the PyTorch model:")
  output = model(example_A, example_B)
  print("--------------------------\n")

  print("--------------------------")
  print("Exporting the model...")
  print("--------------------------\n")
  # OpenVINO converts PyTorch models directly using example inputs
  ov_model = ov.convert_model(model, example_input=(example_A, example_B))

  print("--------------------------")
  print("Saving the model...")
  print("--------------------------\n")
  model_path = f"matmul{matrix_size}_model_{datatype}.xml"
  # Save to OpenVINO Intermediate Representation (.xml and .bin)
  ov.save_model(ov_model, model_path, compress_to_fp16=compress_fp16)

  print("--------------------------")
  print("Loading and compiling the model...")
  print("--------------------------\n")
  core = ov.Core()
  # Device can be "CPU", "GPU", or "AUTO"
  compiled_model = core.compile_model(model_path, device_name="AUTO")

  print("--------------------------")
  print("Testing the OpenVINO model...")
  print("--------------------------\n")
  # OpenVINO accepts PyTorch tensors or NumPy arrays directly
  results = compiled_model([example_A, example_B])

  print("+++ OK +++")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="python matmul.py matrix_size datatype."
  )

  parser.add_argument(
      "matrix_input", type=int, help="matrix size in integer"
  )
  parser.add_argument(
      "datatype_input", type=str, help="datatype: fp32/fp16"
  )

  args = parser.parse_args()

  if args.matrix_input is None or args.datatype_input is None:
    print("Error: You must provide all the arguments.")
    parser.print_usage()
  else:
    main(args.matrix_input, args.datatype_input)