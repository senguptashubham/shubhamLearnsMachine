# One-off benchmark: does precomputing the input-side projection (project_input +
# step, see hand_rnn.py/hand_lstm.py) actually speed up training vs the original
# per-timestep forward() loop, and does torch.compile help further? RNN only, at
# T=196 -- the config the optimization targets (many tiny kernel launches from the
# per-step matmul, not FLOPs, was the identified bottleneck).
#
# OldSequentialMNISTRNN below is a throwaway copy of SequentialMNISTRNN.forward as
# it existed BEFORE the optimization -- HandRNNCell.forward itself is untouched,
# still the validated reference implementation both paths are checked against
# (see the allclose check added to hand_rnn.py's __main__ block).
#
# Result (2026-08-06, RTX-class GPU, H=64, batch_size=64, 5 epochs): new path is
# ~1.1-1.3x faster than old -- real but modest, since the recurrent Whh @ h_prev
# matmul and the per-iteration Python loop overhead are unavoidable and dominate
# more than the one matmul removed. torch.compile failed outright: no working
# Triton install on this Windows setup (Inductor's default backend needs it) --
# not pursued further, since getting Triton working on Windows is a disproportionate
# side-quest for the remaining upside.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import torch.nn as nn
import torch.optim as optim

from data import build_dataloader
from model import SequentialMNISTRNN
from hand_rnn import HandRNNCell
from train import train_one_epoch


class OldSequentialMNISTRNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_classes=10):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.rnn_cell = HandRNNCell(input_dim=input_dim, hidden_dim=hidden_dim)
    self.classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)

  def forward(self, x):  # x: (B, T), pre-optimization per-step loop
    B, T = x.shape[0], x.shape[1]
    h_prev = torch.zeros(B, self.hidden_dim, device=x.device)
    for t in range(T):
      x_t = x[:, t].unsqueeze(1)
      h_t = self.rnn_cell.forward(x_t=x_t, h_prev=h_prev)
      h_prev = h_t
    logits = self.classifier.forward(h_prev)
    return logits


def time_training(model, train_loader, device, epochs=5, lr=1e-3, clip_norm=5.0):
  # times the FULL multi-epoch run, not just one epoch -- for the compiled model,
  # epoch 1 eats the one-time trace+compile cost, epochs 2-5 run on the already-
  # compiled graph, so the total gives a fairer read on whether that cost
  # actually amortizes away rather than penalizing compile for a single epoch.
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()

  avg_loss, accuracy = None, None
  start = time.time()
  for epoch in range(epochs):
    avg_loss, accuracy = train_one_epoch(
      model=model, dataloader=train_loader, optimizer=optimizer,
      criterion=criterion, device=device, clip_norm=clip_norm
    )
    print(f"  epoch {epoch + 1}/{epochs} | loss {avg_loss:.4f} | acc {accuracy:.2f}%")
  elapsed = time.time() - start
  return elapsed, avg_loss, accuracy


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device:", device)

  H, batch_size, epochs = 64, 64, 5
  dataloaders = build_dataloader(batch_size=batch_size)
  train_loader = dataloaders["crop14"][0]  # T=196

  torch.manual_seed(42)
  old_model = OldSequentialMNISTRNN(input_dim=1, hidden_dim=H, num_classes=10)
  torch.manual_seed(42)
  new_model = SequentialMNISTRNN(input_dim=1, hidden_dim=H, num_classes=10)

  print("--- old (per-step forward) ---")
  old_time, old_loss, old_acc = time_training(old_model, train_loader, device, epochs=epochs)
  print(f"total elapsed {old_time:.2f}s | final loss {old_loss:.4f} | final acc {old_acc:.2f}%")

  print("--- new (project_input + step) ---")
  new_time, new_loss, new_acc = time_training(new_model, train_loader, device, epochs=epochs)
  print(f"total elapsed {new_time:.2f}s | final loss {new_loss:.4f} | final acc {new_acc:.2f}%")

  # torch.compile: traces forward() into a graph and fuses ops via the Inductor/
  # Triton backend -- targets the same per-step kernel-launch overhead directly.
  # Windows + Triton support is spotty; this may error out or silently no-op
  # rather than speed anything up, which is itself useful information.
  torch.manual_seed(42)
  compiled_model = torch.compile(SequentialMNISTRNN(input_dim=1, hidden_dim=H, num_classes=10))

  print("--- compiled (torch.compile on new path) ---")
  try:
    compiled_time, compiled_loss, compiled_acc = time_training(compiled_model, train_loader, device, epochs=epochs)
    print(f"total elapsed {compiled_time:.2f}s | final loss {compiled_loss:.4f} | final acc {compiled_acc:.2f}%")
  except Exception as e:
    compiled_time = None
    print(f"torch.compile failed: {e}")

  print(f"\nspeedup (new vs old): {old_time / new_time:.2f}x")
  if compiled_time is not None:
    print(f"speedup (compiled vs old): {old_time / compiled_time:.2f}x")
    print(f"speedup (compiled vs new): {new_time / compiled_time:.2f}x")
