import torch

tensor = torch.rand(4, 4)
print(tensor)
print("\nfirst row: ", tensor[0])
print("\nfirst column", tensor[:, 0])
print("\nlast column", tensor[:, -1])

tensor[:, 1] = 0
print("\n", tensor)

tensor1 = torch.ones(4, 4)
tensor2 = torch.zeros(4, 4)
tensor3 = torch.cat([tensor1, tensor2], dim=1)
print("\nconcatenated tensor with dim 1\n", tensor3)
tensor4 = torch.cat([tensor1, tensor2], dim=0)
print("\nconcatenated tensor with dim 0\n", tensor4)

shape = (2,3)
ones_tensor = torch.ones(shape)
rand_tensor = torch.rand(shape)
m1 = torch.matmul(ones_tensor, rand_tensor.T)
print("\nresult of ones_tensor x rand_tensor^T: ", m1.shape, "\n", m1)
m2 = torch.matmul(ones_tensor.T, rand_tensor)
print("\nresult of ones_tensor^T x rand_tensor: ", m2.shape, "\n", m2)

agg = m1.sum()
print(agg)
agg_item = agg.item()
print(agg_item, type(agg_item))

x = torch.ones((2,3))
y = torch.ones(3)
print(x+y)

x = torch.rand(5, 1, 4, 1)
y = torch.rand(1, 3, 1, 1) 
print((x+y).size()) #either dimensions same or one of them should be 1

x = torch.ones(1)
y = torch.ones(3,1,7)
print((x+y).size())

try:
  x = torch.ones(5, 2, 4, 1)
  y = torch.ones(   3, 1, 1)
  print((x+y).size())
except RuntimeError:
  print("for broadcasting size of tensor a must match the size of tensor b at non-singleton dimension")

x = torch.ones(5, 1, 4, 1)
y = torch.ones(   3, 1, 1)
print((x+y).size())