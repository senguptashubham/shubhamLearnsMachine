import torch
import numpy as np
print(torch.__version__)

a = torch.tensor([1,2])
print(a)

empty_tensor = torch.empty((2,3))
print(empty_tensor)

vec = torch.tensor([1,2,3])
print(vec)

mat = torch.tensor([[1,2,3],
                    [4,5,6]])
print(mat)

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)
print(x_data.dtype)

data = ((1.1, 2.3),(3.2, 4.7))
y_data = torch.tensor(data)
print(y_data)
print(y_data.dtype)

z_data = x_data + y_data
print(z_data)
print(z_data.dtype)

identity_2 = torch.eye(2)
identity_5 = torch.eye(5)
print(identity_2)
print(identity_5)

ones = torch.ones((2,3))
zeros = torch.zeros((3,3))
twos = ones * 2
print("ones matrix:\n", ones)
print("zeros matrix:\n", zeros)
print("twos matrix:\n", twos)

x_random1 = torch.rand(10)
x_random2 = torch.rand((2,3))
print(x_random1)
print(x_random2)

x = torch.arange(start=0, end=20, step=2, dtype=torch.float32)
print(x)
y = torch.arange(start=0.5, end=10.5, step=0.5, dtype=torch.float32)
print(y)

x = torch.linspace(start=0, end=10, steps=5)
print(x)