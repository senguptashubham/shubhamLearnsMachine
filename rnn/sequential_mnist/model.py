# Shared shell for Phase 2: loop a hand-rolled recurrent cell over a flattened
# pixel sequence, then classify from the FINAL state only. RNN and LSTM get
# separate classes (not one merged class) since LSTM threads two states (h, c)
# and RNN threads one -- forcing that into a single class would need branching
# for no real benefit.
#
# hidden_dim (H) must be identical between the RNN and LSTM classes when
# actually running the Phase 2 comparison -- otherwise an accuracy difference
# could come from capacity, not architecture, which is the thing being tested.

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "handrolled_rnn_lstm"))
import torch
import torch.nn as nn
from hand_rnn import HandRNNCell
from hand_lstm import HandLSTMCell


class SequentialMNISTRNN(nn.Module):
  # x_t     : (B, D)
  # h_prev  : (B, H)
  # Wxh     : (H, D)      # so x_t @ Wxh.T -> (B, H)
  # Whh     : (H, H)      # so h_prev @ Whh.T -> (B, H)
  # bh      : (H,)        # broadcasts over B
  # h_t     : (B, H)

  def __init__(self, input_dim, hidden_dim, num_classes=10):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.rnn_cell = HandRNNCell(input_dim=input_dim, hidden_dim=hidden_dim)
    self.classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)

  def forward(self, x):  # x: (B, T), one flattened pixel sequence per row
    B, T = x.shape[0], x.shape[1]
    h_prev = torch.zeros(B, self.hidden_dim)  # fresh memory per batch, not per weight
    for t in range(T):
      x_t = x[:, t].unsqueeze(1)  # (B,) -> (B, 1) = (B, input_dim)
      h_t = self.rnn_cell.forward(x_t=x_t, h_prev=h_prev)
      h_prev = h_t
    logits = self.classifier.forward(h_prev)  # only the FINAL h_t reaches the head
    return logits


class SequentialMNISTLSTM(nn.Module):
  # x_t     : (B, D)
  # h_prev  : (B, H)      -- exposed/public state
  # c_prev  : (B, H)      -- internal memory highway
  # h_t, c_t: (B, H)

  def __init__(self, input_dim, hidden_dim, num_classes=10):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.lstm_cell = HandLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)
    self.classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)

  def forward(self, x):  # x: (B, T), one flattened pixel sequence per row
    B, T = x.shape[0], x.shape[1]
    h_prev = torch.zeros(B, self.hidden_dim)
    c_prev = torch.zeros(B, self.hidden_dim)
    for t in range(T):
      x_t = x[:, t].unsqueeze(1)  # (B,) -> (B, 1) = (B, input_dim)
      h_t, c_t = self.lstm_cell.forward(x_t=x_t, h_prev=h_prev, c_prev=c_prev)
      h_prev, c_prev = h_t, c_t
    # only h_t reaches the classifier, same as the RNN class -- c_t is the
    # internal ledger, h_t is the gate-filtered "what's relevant to expose"
    # signal (h_t = o_t * tanh(c_t)); stacking c_t in would also give the LSTM
    # classifier a bigger input than the RNN's, breaking the capacity-matched
    # comparison the two models are supposed to have.
    logits = self.classifier.forward(h_prev)
    return logits


if __name__ == "__main__":
  B, D, H, T = 2, 1, 4, 5
  mnist_rnn = SequentialMNISTRNN(input_dim=D, hidden_dim=H, num_classes=6)
  logits = mnist_rnn.forward(torch.randn(B, T))
  print("SequentialMNISTRNN logits shape:", logits.shape)  # expect (B, num_classes)

  mnist_lstm = SequentialMNISTLSTM(input_dim=D, hidden_dim=H, num_classes=6)
  logits = mnist_lstm.forward(torch.randn(B, T))
  print("SequentialMNISTLSTM logits shape:", logits.shape)  # expect (B, num_classes)

