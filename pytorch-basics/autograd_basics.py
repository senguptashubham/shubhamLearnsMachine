import torch

# y = 1/3 (sum from i=1 to 3)[(xi + 2)^2 + 3] let us find gradient dy/dx for x = [0,1,2]

X = torch.arange(3, dtype=torch.float32, requires_grad=True)

print("X", X)

#build computational graph step by step
a = X + 2
b = a ** 2
c = b + 3
y = c.mean()

print("Y", y)

# dy/dx = dy/dc dc/db db/da da/dx <- chain rule
y.backward() #claculate gradient by going back along the computational graph
print("gradient of y wrt X: ", X.grad)

# a.grad or other intermediate gradient not stored by pytorch
