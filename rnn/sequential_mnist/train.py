import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from model import SequentialMNISTRNN, SequentialMNISTLSTM
from data import build_dataloader


def train_one_epoch(model, dataloader, optimizer, criterion, clip_norm=5.0):
  model.train()
  train_loss = 0
  epoch_correct = 0
  epoch_total = 0

  for images, labels in dataloader:
    # zero-grad -> forward pass -> loss -> backward -> clip -> step
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    # clip AFTER backward, BEFORE step -- caps the gradient's magnitude before
    # it's used to update weights. Matters most for the RNN at long T, where
    # exploding gradients (not just vanishing) are a real risk.
    clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    optimizer.step()

    # accumulate raw counts/sums here, divide once at the end -- averaging
    # per-batch accuracies directly would be slightly wrong if the last
    # batch is a different size than the rest
    train_loss += loss.item()
    epoch_total += labels.size(0)
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum()
    epoch_correct += correct.item()

  avg_loss = train_loss / len(dataloader)
  accuracy = 100 * epoch_correct / epoch_total
  return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
  model.eval()
  eval_loss = 0
  eval_total = 0
  eval_correct = 0

  with torch.no_grad():
    for images, labels in dataloader:
      logits = model(images)
      loss = criterion(logits, labels)

      eval_loss += loss.item()
      eval_total += labels.size(0)
      preds = logits.argmax(dim=1)
      correct = (preds == labels).sum()
      eval_correct += correct.item()

  avg_loss = eval_loss / len(dataloader)
  accuracy = 100 * eval_correct / eval_total
  return avg_loss, accuracy


def train(model, train_loader, val_loader, epochs, lr=1e-3, clip_norm=5.0):
  # optimizer/criterion created ONCE, outside the epoch loop -- optimizer
  # carries momentum/Adam state across epochs, recreating it per epoch would
  # silently reset that state every time
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()
  train_loss_history = []
  train_accuracy_history = []
  val_loss_history = []
  val_accuracy_history = []

  for epoch in range(epochs):
    train_loss, train_accuracy = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, clip_norm=clip_norm)
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    val_loss, val_accuracy = evaluate(model=model, dataloader=val_loader, criterion=criterion)
    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)

    print(f"epoch {epoch + 1}/{epochs} | train loss {train_loss:.4f} acc {train_accuracy:.2f}% "
          f"| val loss {val_loss:.4f} acc {val_accuracy:.2f}%")

  history = {'Training Loss History': train_loss_history,
             'Training Accuracy History': train_accuracy_history,
             'Validation Loss History': val_loss_history,
             'Validation Accuracy History': val_accuracy_history}
  return model, history


if __name__ == "__main__":
  # smoke test: quick end-to-end run (data -> model -> train -> test eval) for
  # BOTH cells, on the fastest (T=64) variant -- confirms the training loop
  # itself is healthy (init fix included) for each architecture independently,
  # before run_experiments.py runs the full 6-config grid across all 3 T's.
  H = 64
  dataloaders = build_dataloader(batch_size=64)
  train_loader, val_loader, test_loader = dataloaders['crop8']  # T=64, fastest to smoke-test
  criterion = nn.CrossEntropyLoss()

  for label, model_cls in [("RNN", SequentialMNISTRNN), ("LSTM", SequentialMNISTLSTM)]:
    print(f"--- smoke test: {label} ---")
    model = model_cls(input_dim=1, hidden_dim=H, num_classes=10)
    model, history = train(model, train_loader, val_loader, epochs=2)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print(f"{label} smoke test done | test loss {test_loss:.4f} acc {test_accuracy:.2f}%\n")
