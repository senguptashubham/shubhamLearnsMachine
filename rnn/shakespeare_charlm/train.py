import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import math
import os
from model import CharRNN, CharLSTM
from shakespeare_data import build_dataloader

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

def train_one_apoch(model, dataloader, optimizer, criterion, device, clip_norm=0.5):
  model.train()
  # scalar accumulator kept ON DEVICE, not a Python float -- avoids a GPU->CPU
  # sync (.item()) every single batch, only paying that cost once at the end
  total_loss = torch.tensor(0.0, device=device)

  for inputs, targets in dataloader:
    inputs, targets = inputs.to(device), targets.to(device)
    # zero-grad -> forward pass -> loss -> backward -> clip -> step
    optimizer.zero_grad()
    logits = model(inputs)  # (B, T, vocab_size)
    # CrossEntropyLoss wants (N, C) predictions and (N,) targets -- flatten B
    # and T together into one N=B*T axis, since every (batch, timestep) pair
    # is an independent "predict the next char" example. Order doesn't matter
    # for the resulting loss value, only that both reshape identically so
    # row i of one still lines up with row i of the other.
    loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    # clip AFTER backward, BEFORE step -- same reasoning as Phase 2: caps
    # gradient magnitude before it updates weights
    clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    optimizer.step()
    # .detach() strips the autograd graph reference before accumulating --
    # without it, total_loss would chain onto every batch's (buffer-freed but
    # structurally intact) graph for the whole epoch, growing memory for no
    # reason since we never call .backward() on total_loss itself
    total_loss += loss.detach()

  avg_loss = total_loss.item() / len(dataloader)  # only sync point, once per epoch
  # perplexity = exp(avg cross-entropy loss) -- roughly "the model is as
  # uncertain as choosing uniformly among this many characters." Ranges from
  # 1 (perfect) up toward vocab_size (no better than random guessing); NOT a
  # percentage, don't format it with a %.
  perplexity = math.exp(avg_loss)

  return avg_loss, perplexity


def evaluate(model, dataloader, criterion, device):
  model.eval()
  total_loss = torch.tensor(0.0, device=device)
  # no zero_grad/backward/step here -- eval only measures the loss the
  # current weights already produce, it never updates them
  with torch.no_grad():
    for inputs, targets in dataloader:
      inputs, targets = inputs.to(device), targets.to(device)
      logits = model(inputs)
      loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
      total_loss += loss.detach()

  avg_loss = total_loss.item() / len(dataloader)
  perplexity = math.exp(avg_loss)

  return avg_loss, perplexity


def train(model, train_loader, val_loader, epochs, device, mappings, lr=1e-3, clip_norm=5.0):

  model = model.to(device)
  # optimizer/criterion created ONCE, outside the epoch loop -- Adam carries
  # momentum state across epochs, recreating it per epoch would silently
  # reset that state every time
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()
  train_loss_history, train_perplexity_history, val_loss_history, val_perplexity_history = [], [], [], []
  best_val_loss = float('inf')
  for epoch in range(epochs):
    train_loss, train_perplexity = train_one_apoch(model, train_loader, optimizer, criterion, device, clip_norm)
    train_loss_history.append(train_loss)
    train_perplexity_history.append(train_perplexity)
    val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(model, mappings)
    val_loss_history.append(val_loss)
    val_perplexity_history.append(val_perplexity)
    print(f"epoch {epoch + 1}/{epochs} | train loss {train_loss:.4f} perplexity {train_perplexity:.2f} "f"| val loss {val_loss:.4f} perplexity {val_perplexity:.2f}")

  history = {'Training Loss History': train_loss_history,
               'Training Perplexity History': train_perplexity_history,
               'Validation Loss History': val_loss_history,
               'Validation Perplexity History': val_perplexity_history}
  return model, history

def save_checkpoint(model, mappings):
  checkpoint = {'model_state_dict':model.state_dict(),
                'hidden_dim':model.hidden_dim,
                'embed_dim':model.embed_dim,
                'vocab_size':model.vocab_size,
                'char2idx':mappings['char2idx'],
                'idx2char':mappings['idx2char']
                }
  checkpoint_name = model.__class__.__name__.lower()
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)
  checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{checkpoint_name}.pt')
  torch.save(obj=checkpoint, f=checkpoint_path)


if __name__ == "__main__":
  # smoke test: quick end-to-end run (data -> model -> train -> test eval) for
  # BOTH cells -- confirms the training loop itself is healthy before a real,
  # longer training run
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device:", device)
  # vocab_size/mappings (char2idx, idx2char) come from build_dataloader, not
  # hardcoded -- train/val/test all share ONE vocabulary built from the full
  # text before splitting, so there's no risk of them disagreeing
  dataloaders, vocab_size, mappings = build_dataloader(seq_len=100, batch_size=64)
  criterion = nn.CrossEntropyLoss()
  for label, model_cls in [("RNN", CharRNN), ("LSTM", CharLSTM)]:
    print(f"--- smoke test: {label} ---")
    # hidden_dim/embed_dim must stay IDENTICAL between RNN and LSTM here --
    # same capacity-matched-comparison principle as Phase 2, otherwise a
    # perplexity gap could come from capacity, not architecture
    model = model_cls(hidden_dim=128, vocab_size=vocab_size, embed_dim=32)
    model, history = train(model, dataloaders['train_loader'], dataloaders['val_loader'], epochs=2, device=device, mappings=mappings)
    test_loss, test_perplexity = evaluate(model, dataloaders['test_loader'], criterion, device=device)
    print(f"{label} smoke test done | test loss {test_loss:.4f} perplexity {test_perplexity:.2f}\n")