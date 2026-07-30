import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import math
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from cifar_data import build_dataloaders
from cifar_model import CNNcifar10
from train_engine import fit, evaluate, save_checkpoint

RUN_ID = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RESULTS_DIR = 'results'
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
RESULTS_FILE = os.path.join(RESULTS_DIR, f'hparam_search_{RUN_ID}.csv')

SEARCH_EPOCHS = 15  # ceiling; patience will cut this short
FINAL_EPOCHS = 40   # ceiling; patience will cut this short
SEARCH_PATIENCE = 2  # aggressive — cut off unpromising trials fast, search is meant to be cheap
FINAL_PATIENCE = 5  # more lenient — be thorough about confirming the true peak for the winning config



def sample_config():
  log_lr = random.uniform(math.log10(1e-4), math.log10(1e-2))
  return {
    'lr': 10 ** log_lr,
    'batch_size': random.choice([32, 64, 128]),
    'dropout_rate': random.uniform(0.1, 0.5),
  }


def run_trial(config, device, num_epochs=SEARCH_EPOCHS):
  train_loader, val_loader, _ = build_dataloaders(root='./data', batch_size=config['batch_size'])
  model = CNNcifar10(dropout_rate=config['dropout_rate']).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=config['lr'])

  history = fit(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=num_epochs, verbose=False, patience=SEARCH_PATIENCE)

  best_epoch = history['val_loss'].index(min(history['val_loss']))
  return {
    'best_epoch': best_epoch + 1,
    'val_loss': history['val_loss'][best_epoch],
    'val_acc': history['val_acc'][best_epoch],
  }


def append_result(result):
  file_exists = os.path.isfile(RESULTS_FILE)
  with open(RESULTS_FILE, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['trial', 'lr', 'batch_size', 'dropout_rate',
                                            'best_epoch', 'val_loss', 'val_acc'])
    if not file_exists:
      writer.writeheader()
    writer.writerow(result)


def random_search(n_trials, device):
  results = []
  for trial in range(n_trials):
    config = sample_config()
    print(f"Trial {trial + 1}/{n_trials}: lr={config['lr']:.5f}, "
          f"batch_size={config['batch_size']}, dropout_rate={config['dropout_rate']:.3f}")
    trial_result = run_trial(config, device)
    row = {'trial': trial, **config, **trial_result}
    print(f"  -> best_epoch={row['best_epoch']}, val_loss={row['val_loss']:.4f}, val_acc={row['val_acc']:.2f}%")
    append_result(row)
    results.append(row)
  return results


def main():
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Using device: {device}")

  results = random_search(n_trials=10, device=device)
  #take result from results such that result['val_loss'] is minimum
  winner = min(results, key=lambda r: r['val_loss'])
  print(f"\nBest config: lr={winner['lr']:.5f}, batch_size={winner['batch_size']}, "
        f"dropout_rate={winner['dropout_rate']:.3f} (val_loss={winner['val_loss']:.4f})")

  print(f"Retraining winner for {FINAL_EPOCHS} epochs...")
  train_loader, val_loader, test_loader = build_dataloaders(root='./data', batch_size=winner['batch_size'])
  model = CNNcifar10(dropout_rate=winner['dropout_rate']).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=winner['lr'])

  fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE)
  test_loss, test_acc = evaluate(model, test_loader, criterion, device)
  print(f"Winner Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

  checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{RUN_ID}.pt")
  save_checkpoint(model, checkpoint_path, dropout_rate=winner['dropout_rate'], lr=winner['lr'], batch_size=winner['batch_size'],num_epochs=FINAL_EPOCHS, test_acc=test_acc)
  print(f"Saved winning checkpoint to {checkpoint_path}")
  print(f"Trial log: {RESULTS_FILE}")


if __name__ == '__main__':
  main()
