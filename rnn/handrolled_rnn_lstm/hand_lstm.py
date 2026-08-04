import torch
import torch.nn as nn
import torch.nn.functional as F

class HandLSTMCell(nn.Module):
# ----------- x ------------------------ + ------------------------------------> long term memory / cell state / c_t
#             |                          |                             |
#  [% longterm to remember] [% potential longterm memory]           [tanh]
#             |                          |                     [POT SHORTTERM]
#             |           [% POT MEMORY] X [POT LONGTERM]           X ----------> updated short term memory / h_t
#      [ FORGET GATE ]     [INPUT GATE]   [CANDIDATE GATE]  [% POT SHORTTERM]
#            f_t                i_t             g_t               o_t
# __________|___|______________|___|___________|___|_____________|___|
# short term|memory/ hidden sta|te / h_prev    |                 |
# ----------------------------------------------------------------
#   imput / x_t


  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.Wf_x = nn.Parameter(torch.empty(hidden_dim, input_dim))
    nn.init.xavier_uniform_(self.Wf_x)
    self.Wi_x = nn.Parameter(torch.empty(hidden_dim, input_dim))
    nn.init.xavier_uniform_(self.Wi_x)
    self.Wg_x = nn.Parameter(torch.empty(hidden_dim, input_dim))
    nn.init.xavier_uniform_(self.Wg_x)
    self.Wo_x = nn.Parameter(torch.empty(hidden_dim, input_dim))
    nn.init.xavier_uniform_(self.Wo_x)
    self.Wf_h = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    nn.init.xavier_uniform_(self.Wf_h)
    self.Wi_h = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    nn.init.xavier_uniform_(self.Wi_h)
    self.Wg_h = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    nn.init.xavier_uniform_(self.Wg_h)
    self.Wo_h = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    nn.init.xavier_uniform_(self.Wo_h)
    self.bf = nn.Parameter(torch.zeros(hidden_dim))
    self.bi = nn.Parameter(torch.zeros(hidden_dim))
    self.bg = nn.Parameter(torch.zeros(hidden_dim))
    self.bo = nn.Parameter(torch.zeros(hidden_dim))

  def forward(self, x_t, h_prev, c_prev):
    f_t = F.sigmoid(x_t @ self.Wf_x.T + h_prev @ self.Wf_h.T + self.bf)
    c_t = c_prev * f_t #element wise multiplication
    i_t = F.sigmoid(x_t @ self.Wi_x.T + h_prev @ self.Wi_h.T + self.bi)
    g_t = F.tanh(x_t @ self.Wg_x.T + h_prev @ self.Wg_h.T + self.bg)
    c_t = c_t + (i_t * g_t) # same shape -> elementwise muliplication
    o_t = F.sigmoid(x_t @ self.Wo_x.T + h_prev @ self.Wo_h.T + self.bo)
    h_t = F.tanh(c_t) * o_t # same shape -> elementwise muliplication
    return h_t, c_t 


if __name__ == "__main__":
  #test
  B, D, H = 2, 3, 4
  #make a HandLSTMCell with custom param and check h_t(short term memory), c_t(long term memory) shape after forward
  my_lstm = HandLSTMCell(input_dim=D, hidden_dim=H)
  torch_lstm = nn.LSTMCell(input_size=D, hidden_size=H)
  hidden_state, cell_state = my_lstm.forward(torch.randn(B, D), torch.randn(B, H), torch.randn(B, H))
  print("h_t shape:", hidden_state.shape, "| c_t shape:", cell_state.shape)  # expect (B, H) each

  # assign same weights and bias to HandLSTMCell, taking value from nn.LSTMCell (keep in mind it is stored in i->f->g->o order in nn.LSTMCell, we have to unpack it as per given hidden dimension)
  my_lstm.Wi_x.data = torch_lstm.weight_ih[:H].data
  my_lstm.Wf_x.data = torch_lstm.weight_ih[H:2*H].data
  my_lstm.Wg_x.data = torch_lstm.weight_ih[2*H:3*H].data
  my_lstm.Wo_x.data = torch_lstm.weight_ih[3*H:].data

  my_lstm.Wi_h.data = torch_lstm.weight_hh[:H].data
  my_lstm.Wf_h.data = torch_lstm.weight_hh[H:2*H].data
  my_lstm.Wg_h.data = torch_lstm.weight_hh[2*H:3*H].data
  my_lstm.Wo_h.data = torch_lstm.weight_hh[3*H:].data

  my_lstm.bi.data = torch_lstm.bias_ih[:H].data + torch_lstm.bias_hh[:H].data
  my_lstm.bf.data = torch_lstm.bias_ih[H:2*H].data + torch_lstm.bias_hh[H:2*H].data
  my_lstm.bg.data = torch_lstm.bias_ih[2*H:3*H].data + torch_lstm.bias_hh[2*H:3*H].data
  my_lstm.bo.data = torch_lstm.bias_ih[3*H:].data + torch_lstm.bias_hh[3*H:].data

  # with manual seed creating random matrices for input and previous hidden state, previous cell state to pass to forward
  torch.manual_seed(42)
  x_t = torch.randn(B, D)
  h_prev = torch.randn(B, H)
  c_prev = torch.randn(B, H)

  #compare between HandLSTMCell and nn.LSTMCell after capturing the hidden state and cell state of forward
  my_hidden, my_cell = my_lstm.forward(x_t=x_t, h_prev=h_prev, c_prev=c_prev)
  torch_hidden, torch_cell = torch_lstm.forward(input=x_t, hx=(h_prev, c_prev))
  print("HandLSTMCell h_t matches nn.LSTMCell (allclose):", torch.allclose(my_hidden, torch_hidden))
  print("HandLSTMCell c_t matches nn.LSTMCell (allclose):", torch.allclose(my_cell, torch_cell))

  # unroll forward 500 times, forward-only, no gradient: unlike HandRNNCell's h_t,
  # c_t has NO squashing applied to it (only h_t = o_t * tanh(c_t) is bounded) --
  # it's a pure additive accumulation, so it genuinely CAN grow unbounded over a
  # long rollout. This check is a real smoke test here, not a vacuous one.
  for roll in range(500):
    x_t = torch.randn(B, D)
    h_prev, c_prev = my_lstm.forward(x_t=x_t, h_prev=h_prev, c_prev=c_prev)

  print("500-step rollout, h_t finite:", torch.isfinite(h_prev).all().item(), h_prev.shape)
  print("500-step rollout, c_t finite:", torch.isfinite(c_prev).all().item(), c_prev.shape)

  # investigate vanishing gradient, same pattern as HandRNNCell but tracking BOTH
  # state paths (h_t and c_t) since LSTM has two of them, to compare how much each
  # one decays at the same depth.
  #
  # calling .backward() separately from two losses that share most of the same
  #     500-step graph fails on the second call (the first backward() frees buffers
  #     the second one still needs) -- combine into ONE scalar loss, ONE backward().
  h_prev = torch.randn(B, H)
  c_prev = torch.randn(B, H)
  long_memory = []   # c_t history -- the "highway", no squashing, expected to decay slower
  short_memory = []  # h_t history -- the exposed output, additionally gated/squashed every step
  for roll in range(500):
    x_t = torch.randn(B, D)
    h_t, c_t = my_lstm.forward(x_t=x_t, h_prev=h_prev, c_prev=c_prev)
    c_t.retain_grad()
    h_t.retain_grad()
    long_memory.append(c_t)
    short_memory.append(h_t)
    c_prev = c_t
    h_prev = h_t

  loss = long_memory[-1].sum() + short_memory[-1].sum()
  loss.backward()

  print("gradient norm, early/mid timesteps (far from loss):")
  for i in range(0, 500, 50):
    print(f"  {i}th long memory (c_t): {long_memory[i].grad.norm()} \t{i}th short memory (h_t): {short_memory[i].grad.norm()}")

  print("gradient norm, late timesteps (close to loss):")
  for i in range(450, 500, 10):
    print(f"  {i}th long memory (c_t): {long_memory[i].grad.norm()} \t{i}th short memory (h_t): {short_memory[i].grad.norm()}")

  # expected pattern: ~0.0 (underflowed) far from the loss for both, transitioning
  # to nonzero somewhere before the end -- and c_t (long_memory) consistently a bit
  # larger than h_t (short_memory) at the same index, since c_t's backward path is
  # a plain elementwise scale by f_t while h_t additionally passes through the
  # output gate and the full Wh matrices every step.
