import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "handrolled_rnn_lstm"))
import torch
import torch.nn as nn
from hand_rnn import HandRNNCell
from hand_lstm import HandLSTMCell

# Phase 3's real architectural difference from Phase 2: many-to-many, not many-to-one. Phase 2 classified from the FINAL h_t only; here every timestep needs its own prediction (predict the NEXT char after each position), so we collect h_t at every step and classify all of them, not just the last.
# x: (B, T) integer character indices -- NOT input_dim=1 raw scalars like Phase 2's pixels. A character index has no meaningful magnitude (index 47 isn't "one more" than 46), so nn.Embedding maps each index to a learned dense embed_dim-length vector before it ever reaches the recurrent cell.
# The cell's input_dim is therefore always embed_dim -- not a free choice, so it's not exposed as a separate constructor argument (that would let the embedding output and the cell's expected input silently disagree).

class CharRNN(nn.Module):

  def __init__(self, hidden_dim, vocab_size, embed_dim):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    self.rnn_cell = HandRNNCell(input_dim=embed_dim, hidden_dim=hidden_dim)
    self.classifier = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

  def forward(self, x):  # x: (B, T) character indices
    B, T = x.shape[0], x.shape[1]
    h_list = []  # will hold T tensors, each (B, H) -- one per timestep
    h_prev = torch.zeros(B, self.hidden_dim, device=x.device)

    x_embed = self.embedding(x)              # (B, T) -> (B, T, embed_dim)
    x_proj = self.rnn_cell.project_input(x_embed)  # (B, T, embed_dim) -> (B, T, H)
    # only the input-side term is precomputed here -- Whh @ h_prev is the
    # genuinely recurrent part and has to stay inside the loop
    for t in range(T):
      h_t = self.rnn_cell.step(x_proj_t=x_proj[:, t], h_prev=h_prev)  # (B, H)
      h_list.append(h_t)
      h_prev = h_t

    # stack, not slice -- h_t never had a T axis, it's created here by combining T separate (B, H) tensors into one (B, T, H) tensor
    h_all = torch.stack(h_list, dim=1)  # (B, T, H)
    # nn.Linear operates on the LAST axis only -- B and T both ride along as "batch," so one call classifies every timestep at once, no loop needed
    logits = self.classifier(h_all)  # (B, T, H) -> (B, T, vocab_size)
    return logits


class CharLSTM(nn.Module):

  def __init__(self, hidden_dim, vocab_size, embed_dim):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    self.lstm_cell = HandLSTMCell(input_dim=embed_dim, hidden_dim=hidden_dim)
    self.classifier = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

  def forward(self, x):
    B, T = x.shape[0], x.shape[1]
    h_list= []
    h_prev = torch.zeros(B, self.hidden_dim, device=x.device)
    c_prev = torch.zeros(B, self.hidden_dim, device=x.device)
    x_embed = self.embedding(x)
    x_dict = self.lstm_cell.project_input(x_embed)
    for t in range(T):
      h_t, c_t = self.lstm_cell.step(xf=x_dict['x_f'][:,t], xi=x_dict['x_i'][:,t], xg=x_dict['x_g'][:,t], xo=x_dict['x_o'][:,t], h_prev=h_prev, c_prev=c_prev)
      h_list.append(h_t)
      h_prev, c_prev = h_t, c_t
    h_all = torch.stack(h_list, dim=1)
    logits = self.classifier(h_all)
    return logits


if __name__ == "__main__":
  B, T, H, vocab_size, embed_dim = 2, 5, 8, 12, 4
  model = CharRNN(hidden_dim=H, vocab_size=vocab_size, embed_dim=embed_dim)
  x = torch.randint(0, vocab_size, (B, T))  # random fake character indices
  logits = model.forward(x)
  print("CharRNN logits shape:", logits.shape)  # expect (B, T, vocab_size)

  lstm_model = CharLSTM(hidden_dim=H, vocab_size=vocab_size, embed_dim=embed_dim)
  logits = lstm_model.forward(x)
  print("CharLSTM logits shape:", logits.shape)  # expect (B, T, vocab_size)