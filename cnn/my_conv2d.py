import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv2d(nn.Module):
  
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
    super().__init__()
    #store the parameters so forward can use
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.padding = padding
    self.stride = stride
    #initialize weights and biases and register them to parameters
    self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
    self.bias = nn.Parameter(torch.randn(out_channels))

  def forward(self, X):

    # F.unfold takes your (N, 3, 32, 32) input and extracts every 3×3×3 sliding patch, flattens each patch into a single vector of length 3*3*3=27, and stacks all patches side by side. Output shape: (N, 27, 1024)
    patches = F.unfold(X, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    
    # need to flatten the last 3 dims of weight (32, 3, 3, 3) —> (32, 27)

    weight_flat = self.weight.flatten(1) # flattens everything from dim 1 onward
    
    # Multiplying weight_flat @ patches (per batch item) is now literally ordinary 2D matrix multiplication: (32, 27) @ (27, L) = (32, L)
    weighted_sum = torch.matmul(weight_flat.unsqueeze(0), patches)    #torch.matmul broadcasts over the batch dim automatically if you unsqueeze weight_flat to (1, 32, 27) results in (N, 32, L)
    
    #adding bias reshaped to (1, 32, 1) so it broadcasts
    weighted_sum_biased = weighted_sum + self.bias.reshape(1, self.out_channels, 1)
    
    #calculate H_out and W_out for reshaping output
    H_out = ((X.shape[2] + (2*self.padding) - self.kernel_size) // self.stride) + 1
    W_out = ((X.shape[3] + (2*self.padding) - self.kernel_size) // self.stride) + 1

    #reshape the output tensor as per H_out and W_out and return
    weighted_sum_biased  = weighted_sum_biased.reshape(weighted_sum_biased.shape[0], self.out_channels, H_out, W_out)
    return weighted_sum_biased
  
#test whether this matches the conv2d output for same given data
# torch.manual_seed(0)
# x = torch.randn(2, 3, 8, 8)
# my_conv = MyConv2d(3, 4, kernel_size=3, padding=1)
# ref_conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

# ref_conv.weight.data = my_conv.weight.data.clone()
# ref_conv.bias.data = my_conv.bias.data.clone()

# out_mine = my_conv(x)
# out_ref = ref_conv(x)

# print(out_mine.shape, out_ref.shape)
# print(torch.allclose(out_mine, out_ref, atol=1e-5)) 