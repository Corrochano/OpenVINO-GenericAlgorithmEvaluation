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
    def __init__(self, nt=1000, num_levels=9, datatype=torch.float32):
        super(MyMachine, self).__init__()
        self.datatype=datatype
        self.num_levels = num_levels
        self.nt = torch.tensor(nt, dtype=self.datatype)

    def restriction(self, residual): # Applied when we want to convert from finegrid to coarsergrid
        kernel = torch.full(
            (1, 1, 2, 2),
            0.25,
            dtype=residual.dtype,
            device=residual.device,
        )
        return F.conv2d(residual, kernel, stride=2)
    
    def interpolate(self, f, target_size): # Applied when we want to convert from coarsergrid to finegrid
        return F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)

    def jacobi(self, Z, mask): # Jacobi method   
      x = Z
      x_prev = x.clone()
      
      '''
      mask = torch.ones_like(x)
      
      mask[:, :, 0, :] = 0
      mask[:, :, -1, :] = 0
      mask[:, :, :, 0] = 0
      mask[:, :, :, -1] = 0
      '''
              
      # Define the 3x3 kernel
      kernel = torch.tensor([[0.0, 0.25, 0.0],
                            [0.25, 0.0, 0.25],
                            [0.0, 0.25, 0.0]], dtype=self.datatype).view(1, 1, 3, 3)
                            
      i = torch.tensor(0, dtype=self.datatype)
      
      while torch.ne(i, self.nt):
          x_prev = x.clone()
          x_next = F.conv2d(x_prev, kernel, padding=1)            
          x = x_next * mask
          
          i = torch.add(i, 1)
      
      return x

    def forward(self, X, Y, Mask1, Mask2, Mask3, Mask4, Mask5, Mask6, Mask7, Mask8, Mask9, Mask10):
        Masks = [Mask1,Mask2,Mask3,Mask4,Mask5,Mask6,Mask7,Mask8,Mask9,Mask10]
        x = torch.exp(torch.mul( # Take the value of X
                        -50, 
                        torch.add(torch.pow((X - 0.5), 2), torch.pow((Y - 0.5), 2))
                    )).to(self.datatype)
        x = x.unsqueeze(0).unsqueeze(0) # Channel and batch size. Necessary for conv layer

        '''
        mask = torch.ones_like(x, dtype=self.datatype)
        mask[:, :, 0, :] = 0
        mask[:, :, -1, :] = 0
        mask[:, :, :, 0] = 0
        mask[:, :, :, -1] = 0
        '''
        
        # Define the 3x3 kernel
        kernel = torch.tensor([[0.0, 0.25, 0.0],
                               [0.25, 0.0, 0.25],
                               [0.0, 0.25, 0.0]], dtype=self.datatype).view(1, 1, 3, 3)
        
        # Define num_levels
        num_levels = self.num_levels
        
        '''
        masks = [mask]
        
        for _ in range(num_levels):# precalculate masks
            masks.append(self.restriction(masks[-1]).to(self.datatype))
        
        '''
        grids = [x]
        
        current_mask = Masks[0]
        
        # Downward phase
        for level in range(1, num_levels):
            masked_input = torch.mul(grids[level - 1] , (Masks[level - 1] > 0))
            unmasked_input = torch.mul(grids[level - 1] , (1 - Masks[level - 1]))

            masked_output = F.conv2d(masked_input, kernel, padding=1)
            unmasked_output = F.conv2d(unmasked_input, kernel, padding=1)

            residual = masked_output + unmasked_output
            
            coarse_residual = self.restriction(residual)  # Restrict to coarser grid
        
            grids.append(coarse_residual)  # Store
        
        # Solve on the coarsest grid
        coarse_solution = grids[-1]
        coarse_solution = self.jacobi(coarse_solution, Masks[num_levels - 1])  # Solve on the coarsest grid
        grids[-1] = coarse_solution  # Update store solution     
               
        # Upward phase
        for level in range(num_levels - 2, -1, -1):

            target_size = grids[level].shape[-2:]
          
            fine_solution = self.interpolate(grids[level + 1], target_size=target_size)  # Interpolate to finer gridclear
                        
            fine_solution = torch.add( fine_solution, grids[level] ) # += grids[level]  # Add correction to finer grid    
            
            masked_input = torch.mul(fine_solution , (Masks[level] > 0))
            unmasked_input = torch.mul(fine_solution , (1 - Masks[level]))

            masked_output = F.conv2d(masked_input, kernel, padding=1)
            unmasked_output = F.conv2d(unmasked_input, kernel, padding=1)

            fine_solution = masked_output + unmasked_output
            grids[level] = fine_solution
            
        # Final refinement on the finest grid        
        final_solution = grids[0]
        final_solution = self.jacobi(final_solution, Mask1)  # Final Jacobi iterations
    
        return final_solution


def main(grid_size, iterations, dataType):
  print("Starting the test for multigrid with OpenVINO and PyTorch...")
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
  num_levels = min(10, grid_size.bit_length())
  multigrid_model = MyMachine(
      nt=iterations,
      num_levels=num_levels,
      datatype=torchfloat,
  )
  multigrid_model.eval()

  coordinates = torch.linspace(0, 1, grid_size, dtype=torchfloat)
  X, Y = torch.meshgrid(coordinates, coordinates, indexing="ij")

  masks = []
  level_size = grid_size
  for _ in range(10):
    mask = torch.ones((1, 1, level_size, level_size), dtype=torchfloat)
    mask[:, :, 0, :] = 0
    mask[:, :, -1, :] = 0
    mask[:, :, :, 0] = 0
    mask[:, :, :, -1] = 0
    masks.append(mask)
    level_size = max(1, level_size // 2)

  model_inputs = (X, Y, *masks)

  print("--------------------------")
  print("Testing the PyTorch model:")
  output = multigrid_model(*model_inputs)
  print("--------------------------\n")

  print("--------------------------")
  print("Exporting the model...")
  print("--------------------------\n")
  ov_model = ov.convert_model(multigrid_model, example_input=model_inputs)
  ov_model.reshape({
      model_input: list(example_input.shape)
      for model_input, example_input in zip(ov_model.inputs, model_inputs)
  })

  print("--------------------------")
  print("Saving the model...")
  print("--------------------------\n")
  model_path = f"multigrid{grid_size}_model_{datatype}.xml"
  ov.save_model(ov_model, model_path, compress_to_fp16=compress_fp16)

  print("--------------------------")
  print("Loading and compiling the model...")
  print("--------------------------\n")
  core = ov.Core()
  compiled_model = core.compile_model(model_path, device_name="AUTO")

  print("--------------------------")
  print("Testing the OpenVINO model...")
  print("--------------------------\n")
  results = compiled_model(model_inputs)

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