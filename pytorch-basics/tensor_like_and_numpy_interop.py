import torch
import numpy as np
print(torch.__version__)

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

zeros = torch.zeros((3,3))

x_ones = torch.ones_like(zeros, dtype=torch.int16)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float32)
print(x_rand)

t = torch.ones((2,3))
print(t)
print(type(t))

n = t.numpy()
print(n)
print(type(n))

n = np.ones(5)
t = torch.from_numpy(n)
print(t)
print(type(t))
