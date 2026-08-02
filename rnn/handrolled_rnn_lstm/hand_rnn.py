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


if __name__ == "__main__":
  B, D, H = 2, 3, 4
  #make a HandRNNCell with custom param and check h_t shape after forward
  cell = HandRNNCell(input_dim=D, hidden_dim=H)
  res = cell.forward(torch.randn(B, D), torch.randn(B, H))
  print("h_t shape:", res.shape)  # expect (B, H)

  # check shape of wight_ih (input) and weight_hh (hidden state) weight matrices for nn.RNNCell
  rnn_cell = nn.RNNCell(D, H)
  for name, param in rnn_cell.named_parameters():
    print(name, param.shape)

  # assign same weights and bias to HandRNNCell, taking value from nn.RNNCell
  cell.Wxh.data = rnn_cell.weight_ih.data
  cell.Whh.data = rnn_cell.weight_hh.data
  cell.bh.data = rnn_cell.bias_ih + rnn_cell.bias_hh

  # with manual seed creating random matrices for input and hidden state
  torch.manual_seed(42)
  x_t = torch.randn(2, 3)
  h_prev = torch.randn(2, 4)

  #compare between HandRNNCell and nn.RNNCell after capturing the output of forward
  my_h_t = cell.forward(x_t, h_prev)
  torch_h_t = rnn_cell.forward(x_t, h_prev)
  print("HandRNNCell matches nn.RNNCell (allclose):", torch.allclose(my_h_t, torch_h_t, atol=1e-6))

  # investigate vanishing gradient: unroll HandRNNCell over many steps, keep every h_t
  # on the autograd graph (retain_grad, since only leaf tensors like Wxh/Whh/bh keep
  # .grad by default), backward() ONCE from the final h_t only, then compare .grad.norm()
  # at early vs late timesteps.
  #
  # h_prev = h_t inside the loop is what actually builds the T-step chain -- without
  # it every step is independent (same starting h_prev reused), nothing links back
  # to earlier steps, and backward() from the last h_t leaves earlier h_t.grad as None
  # (not an ancestor of the loss) instead of a real vanished-to-zero gradient.
  #
  # backward() must be called ONCE, after the loop, from the final h_t -- calling it
  # inside the loop breaks, since each backward() frees the graph buffers it just used,
  # and the next step's backward() would need to walk back through those freed buffers.
  h_prev = torch.randn(2, 4)
  h_history = []
  for roll in range(500):
    x_t = torch.randn(2, 3)
    h_t = cell.forward(x_t, h_prev)
    h_t.retain_grad()
    h_history.append(h_t)
    h_prev = h_t
  loss = h_history[-1].sum()
  loss.backward()

  print("gradient norm, early timesteps (far from loss):")
  for i in range(0, 500, 100):
    print(" ", i, h_history[i].grad.norm())

  print("gradient norm, late timesteps (close to loss):")
  for i in list(range(480, 500, 5)) + [499]:
    print(" ", i, h_history[i].grad.norm())

  # expected pattern: ~0.0 (underflowed) far from the loss, nonzero close to it --
  # the vanishing gradient problem, observed directly rather than just asserted.
