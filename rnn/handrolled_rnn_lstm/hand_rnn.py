import torch
import torch.nn as nn
import torch.nn.functional as F

class HandRNNCell(nn.Module):
# B=batch, D=input_dim, H=hidden_dim:
# x_t     : (B, D)
# h_prev  : (B, H)
# Wxh     : (H, D)      # so x_t @ Wxh.T -> (B, H)
# Whh     : (H, H)      # so h_prev @ Whh.T -> (B, H)
# bh      : (H,)         # broadcasts over B
# h_t     : (B, H)

  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.Wxh = nn.Parameter(torch.randn(hidden_dim, input_dim))
    self.Whh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
    self.bh = nn.Parameter(torch.randn(hidden_dim))

  def forward(self, x_t, h_prev):
    x_t = x_t @ self.Wxh.T
    h = h_prev @ self.Whh.T
    h_t = F.tanh(x_t + h + self.bh)
    return h_t


B, D, H = 2, 3, 4
cell = HandRNNCell(input_dim=D, hidden_dim=H)
res = cell.forward(torch.randn(B, D), torch.randn(B, H))
print(res.shape)

rnn_cell = nn.RNNCell(D, H)
for name, param in rnn_cell.named_parameters():
  print(name, param.shape)

cell.Wxh.data = rnn_cell.weight_ih.data
cell.Whh.data = rnn_cell.weight_hh.data
cell.bh.data = rnn_cell.bias_ih + rnn_cell.bias_hh
torch.manual_seed(42)
x_t = torch.randn(2, 3)
h_prev = torch.randn(2, 4)

my_h_t = cell.forward(x_t, h_prev)
torch_h_t = rnn_cell.forward(x_t, h_prev)
print(torch.allclose(my_h_t, torch_h_t, atol=1e-6))