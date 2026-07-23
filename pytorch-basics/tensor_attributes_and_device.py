import torch
import numpy as np

tensor = torch.rand(3,4)

print("shape of tensor: ", tensor.shape)
print("datatype of tensor: ", tensor.dtype)
print("device tensor is stored on: ", tensor.device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("starting on: ", tensor.device)
tensor = tensor.to(device)
print("device tensor is stored on: ", tensor.device)