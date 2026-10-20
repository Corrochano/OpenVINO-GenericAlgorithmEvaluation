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
from torch.nn import functional as F


class MyMachine(nn.Module):
    def __init__(self, nt=1000, datatype=torch.float32):
        super(MyMachine, self).__init__()
        self.datatype=datatype
        self.nt = torch.tensor(nt, dtype=self.datatype)

    def forward(self, X, X_prev, Mask):
      '''    
      x = torch.exp(torch.mul( # If I cast everything, there are still operations on INT32 idk why and the performance and consume increase a lot
                        -50, 
                        torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))
                    ))
      x = x.unsqueeze(0).unsqueeze(0) # Channel and batch size. Necessary for conv layer
      x_prev = x.clone()
      
      mask = torch.ones_like(x)
      mask[:, :, 0, :] = 0        # Top boundary
      mask[:, :, -1, :] = 0       # Bottom boundary
      mask[:, :, :, 0] = 0        # Left boundary
      mask[:, :, :, -1] = 0       # Right boundary      
      '''
      
      x = X
      x_prev = X_prev
      mask = Mask
      
      # Define the 3x3 kernel
      kernel = torch.tensor([[0.0, 0.25, 0.0],
                            [0.25, 0.0, 0.25],
                            [0.0, 0.25, 0.0]], dtype=self.datatype).view(1, 1, 3, 3)
                            
      i = torch.tensor(0, dtype=self.datatype)
      
      while torch.ne(i, self.nt):
          x_prev = x.clone()

          x_next = F.conv2d(x_prev, kernel, padding=1)
                    
          x = x_next * mask

          # Check for convergence: max difference between u and u_prev
          diff = torch.max(torch.abs(x - x_prev))
          
          i = torch.add(i, 1)
      
      return x


def main(grid_size, iterations, dataType):
  print("Starting the test for jacobi with OpenVINO and PyTorch...")
  nt = iterations
  datatype = dataType

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
  print("Creating inputs...")
  print("--------------------------\n")
  jacobiModel = MyMachine(nt,torchfloat)
  jacobiModel.eval()

  nx=grid_size
  ny=grid_size
  dt=0.001
  alpha=0.01 

  dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)  # spatial step sizes

  x = torch.linspace(0, 1, nx, dtype=torchfloat)
  y = torch.linspace(0, 1, ny, dtype=torchfloat)
  X, Y = torch.meshgrid(x, y)

  '''
  X = X.float()
  Y = Y.float()
  '''
  
  # TRYING TO CREATE THE MASK OUTSIDE JACOBI
  
  x = torch.exp(torch.mul( # If I cast everything, there are still operations on INT32 idk why and the performance and consume increase a lot
                  -50, 
                  torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))
              ))
  x = x.unsqueeze(0).unsqueeze(0) # Channel and batch size. Necessary for conv layer
  x_prev = x.clone()

  mask = torch.ones_like(x)
  mask[:, :, 0, :] = 0        # Top boundary
  mask[:, :, -1, :] = 0       # Bottom boundary
  mask[:, :, :, 0] = 0        # Left boundary
  mask[:, :, :, -1] = 0       # Right boundary     

  print("--------------------------")
  print("Testing the PyTorch model:")
  output = jacobiModel(x, x_prev, mask)
  print("--------------------------\n")

  print("--------------------------")
  print("Exporting the model...")
  print("--------------------------\n")
  # OpenVINO converts PyTorch models directly using example inputs
  ov_model = ov.convert_model(jacobiModel, example_input=(x, x_prev, mask))

  print("--------------------------")
  print("Saving the model...")
  print("--------------------------\n")
  model_path = f"jacobi{grid_size}_model_{datatype}.xml"
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
  results = compiled_model([x, x_prev, mask])

  print("+++ OK +++")


if __name__ == "__main__":
  # Create the parser
  parser = argparse.ArgumentParser(description="python jacobi_conv.py grid_size number_of_iterations datatype.")
  
  # Add arguments
  parser.add_argument("grid_input", type=int, help="grids size in integer")
  parser.add_argument("iteration_input", type=int, help="Number of iterations on the jacobi")
  parser.add_argument("datatype_input", type=str, help="datatype: fp32/fp16")
  
  # Parse the arguments
  args = parser.parse_args()

  # Check if the arguments are correct (not necessary here as argparse will handle type checks)
  if args.grid_input is None or args.iteration_input is None or args.datatype_input is None:
      print("Error: You must provide all the arguments.")
      parser.print_usage()  # Shows the usage message
  else:
      # Call the main function with provided arguments
      main(args.grid_input, args.iteration_input, args.datatype_input)  