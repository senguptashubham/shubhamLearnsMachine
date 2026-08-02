import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from hand_rnn import HandRNNCell
from hand_lstm import HandLSTMCell

#setting up shared config - Batch, Input Dimension, Hidden State Dimension, T: no. of unrolling
B, D, H, T = 2, 3, 4, 500

#creating my rnn and assigning weights of torch's rnn cell default weights
my_rnn = HandRNNCell(input_dim=D, hidden_dim=H)
torch_rnn = nn.RNNCell(input_size=D, hidden_size=H)
my_rnn.Wxh.data = torch_rnn.weight_ih.data
my_rnn.Whh.data = torch_rnn.weight_hh.data
my_rnn.bh.data = torch_rnn.bias_ih.data + torch_rnn.bias_hh.data

#creating my lstm and assigning weights of torch's lstm cell default weights
my_lstm = HandLSTMCell(input_dim=D, hidden_dim=H)
torch_lstm = nn.LSTMCell(input_size=D, hidden_size=H)
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

#cast both models to double to push the underflow
my_rnn = my_rnn.double()
my_lstm = my_lstm.double()

#initialize states to 0
h0_rnn = torch.zeros(B, H, dtype=torch.float64)
h0_lstm = torch.zeros(B, H, dtype=torch.float64)
c0_lstm = torch.zeros(B, H, dtype=torch.float64)

#build shared input sequence
torch.manual_seed(42)
input_seq = []
for t in range(T):
  input_seq.append(torch.randn(B, D, dtype=torch.float64))

#roll forward
h_prev_rnn = h0_rnn
h_prev_lstm, c_prev_lstm = h0_lstm, c0_lstm
h_rnn_hist = []
h_lstm_hist, c_lstm_hist = [], []

for i in range(len(input_seq)):
  x_t = input_seq[i]
  h_t_rnn = my_rnn.forward(x_t=x_t, h_prev=h_prev_rnn)
  h_t_rnn.retain_grad()
  h_rnn_hist.append(h_t_rnn)
  h_prev_rnn = h_t_rnn
  h_t_lstm, c_t_lstm = my_lstm.forward(x_t=x_t, h_prev=h_prev_lstm, c_prev=c_prev_lstm)
  h_t_lstm.retain_grad()
  c_t_lstm.retain_grad()
  h_lstm_hist.append(h_t_lstm)
  c_lstm_hist.append(c_t_lstm)
  h_prev_lstm = h_t_lstm
  c_prev_lstm = c_t_lstm

#backpropagate the loss for both seperately
h_rnn_hist[-1].sum().backward()
(h_lstm_hist[-1].sum() + c_lstm_hist[-1].sum()).backward()

#extract grad norms
rnn_h_t, lstm_h_t, lstm_c_t = [], [], []
for g in range(T):
  rnn_h_t.append(h_rnn_hist[g].grad.norm())
  lstm_h_t.append(h_lstm_hist[g].grad.norm())
  lstm_c_t.append(c_lstm_hist[g].grad.norm())

#stack into 1D tensors and clamp to a tiny floor so exact-0.0 (underflowed)
#entries still land somewhere on a log-scale axis instead of being dropped
#EPS = 1e-40
rnn_h_t = torch.stack(rnn_h_t).numpy()
lstm_h_t = torch.stack(lstm_h_t).numpy()
lstm_c_t = torch.stack(lstm_c_t).numpy()

#first timestep where RNN's gradient is representable at all (nonzero even in
#float64) -- everything before this underflowed to exact 0.0
rnn_underflow_idx = int((rnn_h_t > 0).argmax())

#plot gradient norm vs timestep on a log y-axis: flat at the floor = fully
#vanished, the point where each line lifts off the floor is the vanishing point
SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
# first three categorical slots, dark-surface steps -- validated together as colorblind-distinguishable
COLOR_RNN_H = "#3987e5"   # blue
COLOR_LSTM_H = "#d95926"  # orange
COLOR_LSTM_C = "#199e70"  # aqua

timesteps = range(T)
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(timesteps, rnn_h_t, label="RNN h_t", color=COLOR_RNN_H, linewidth=2)
ax.plot(timesteps, lstm_h_t, label="LSTM h_t", color=COLOR_LSTM_H, linewidth=2)
ax.plot(timesteps, lstm_c_t, label="LSTM c_t (highway)", color=COLOR_LSTM_C, linewidth=2)
ax.set_yscale("log")

# mark exactly where RNN's gradient stops being representable (float64 underflow)
# -- y in axes-fraction coords so the label sits just above the x-axis regardless
# of the actual data range on this log scale
ax.axvline(x=rnn_underflow_idx, color=COLOR_RNN_H, linestyle=":", linewidth=1.5, alpha=0.7, zorder=1)
ax.text(rnn_underflow_idx + 6, 0.03, f"RNN underflows float64 at t={rnn_underflow_idx}",
        transform=ax.get_xaxis_transform(), color=COLOR_RNN_H, fontsize=8.5, va="bottom", ha="left")

# recessive chrome: hairline gridlines, no top/right border, muted ticks
ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
  ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
  ax.spines[side].set_color(BASELINE)
ax.tick_params(colors=INK_MUTED, labelsize=9)

ax.set_xlabel("timestep (0 = start of sequence, far from the loss)", color=INK_SECONDARY, fontsize=10)
ax.set_ylabel(f"gradient norm, log scale", color=INK_SECONDARY, fontsize=10)
ax.set_title(f"Vanishing gradient: RNN vs LSTM, same input sequence, T={T}",
             color=INK_PRIMARY, fontsize=13, fontweight="bold", pad=14)

legend = ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=10)

fig.tight_layout()

save_path = os.path.join(os.path.dirname(__file__), "results", "vanishing_gradient_rnn_vs_lstm.png")
fig.savefig(save_path, dpi=150, facecolor=SURFACE)
print("saved plot to:", save_path)

plt.show()

