import time
import torch
import copy

from cifar_model import CNNcifar10


def train_one_epoch(model, loader, criterion, optimizer, device):
  """Runs one training epoch. Returns (avg_loss, accuracy_pct)."""
  model.train()  # set training mode
  epoch_loss = 0
  epoch_correct = 0
  epoch_total = 0

  for images, labels in loader:
    #move images labels to device
    images, labels = images.to(device), labels.to(device)

    #initialize params -> forward pass -> calculate loss -> backpropagate -> update params
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    #update epoch totals by adding current batch values
    epoch_loss += loss.item()
    predicted = outputs.argmax(dim=1)
    epoch_correct += (predicted == labels).sum().item()
    epoch_total += labels.size(0)

  avg_loss = epoch_loss / len(loader)
  accuracy = 100 * epoch_correct / epoch_total
  return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
  """model.eval() + no_grad pass. Returns (avg_loss, accuracy_pct). Used for both val and test."""
  model.eval()  # set evaluation mode so no param updates
  total_loss = 0
  total_correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in loader:
      #move images labels to device
      images, labels = images.to(device), labels.to(device)

      # just forward pass and loss calculation
      outputs = model(images)
      loss = criterion(outputs, labels)

      #accumulate loss, update total correct and total
      total_loss += loss.item()
      predicted = outputs.argmax(dim=1)
      total_correct += (predicted == labels).sum().item()
      total += labels.size(0)

  avg_loss = total_loss / len(loader)
  accuracy = 100 * total_correct / total
  return avg_loss, accuracy


def fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5, verbose=True, patience=4):
  """Trains for num_epochs, running a validation pass each epoch.
  Returns a history dict: {"train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"}
  each a list with one entry per epoch."""
  history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
  #tracking best epoch in terms of validation loss
  best_val_loss = float('inf')
  best_state_dict = None
  epochs_no_improve = 0

  for epoch in range(num_epochs):
    start = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    epoch_time = time.time() - start
    
    #tracking best epoch
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_state_dict = copy.deepcopy(model.state_dict())
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["epoch_time"].append(epoch_time)

    if verbose:
      print(f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% "
            f"({epoch_time:.1f}s)")
      
    # checking epochs_no_improve and breaking if it reached patience
    if epochs_no_improve >= patience:
      print(f"Early stopping as there were no improvements for {epochs_no_improve} epochs")
      break

  model.load_state_dict(best_state_dict)
  return history


def save_checkpoint(model, path, dropout_rate, **extra_metadata):
  """Saves state_dict + metadata (including fetched dropout_rate) to one file."""
  metadata = {'dropout_rate': dropout_rate, **extra_metadata}
  torch.save({'model_state_dict': model.state_dict(), 'metadata': metadata}, path)


def load_checkpoint(path, device):
  """Reconstructs CNNcifar10 from metadata['dropout_rate'], loads weights,
  returns (model, metadata). Does not run any training code."""
  checkpoint = torch.load(path, map_location=device)
  metadata = checkpoint['metadata']
  model = CNNcifar10(dropout_rate=metadata['dropout_rate'])
  model.load_state_dict(checkpoint['model_state_dict'])
  model = model.to(device)
  model.eval()
  return model, metadata
