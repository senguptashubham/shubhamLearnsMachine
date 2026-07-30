import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from cifar_data import build_dataloaders, visualize_image, CIFAR10_CLASSES
from cifar_model import CNNcifar10
from train_engine import fit, evaluate, save_checkpoint

RESULTS_DIR = 'results'
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')


def plot_history(history, out_path):
  epochs = range(1, len(history["train_loss"]) + 1)
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor='#fcfcfb')

  axes[0].plot(epochs, history["train_loss"], color='#2a78d6', marker='o', markersize=4, linewidth=2, label='Train')
  axes[0].plot(epochs, history["val_loss"], color='#e34948', marker='o', markersize=4, linewidth=2, label='Val')
  axes[0].set_title('Loss per epoch', color='#0b0b0b')
  axes[0].set_xlabel('Epoch', color='#52514e')
  axes[0].set_ylabel('Loss', color='#52514e')

  axes[1].plot(epochs, history["train_acc"], color='#2a78d6', marker='o', markersize=4, linewidth=2, label='Train')
  axes[1].plot(epochs, history["val_acc"], color='#e34948', marker='o', markersize=4, linewidth=2, label='Val')
  axes[1].set_title('Accuracy per epoch', color='#0b0b0b')
  axes[1].set_xlabel('Epoch', color='#52514e')
  axes[1].set_ylabel('Accuracy (%)', color='#52514e')

  for ax in axes:
    ax.set_facecolor('#fcfcfb')
    ax.grid(True, color='#e1e0d9', linewidth=0.7)
    for side in ('top', 'right'):
      ax.spines[side].set_visible(False)
    ax.tick_params(colors='#898781')
    ax.legend(frameon=False, labelcolor='#52514e')

  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.show()


def main():
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Using device: {device}")

  train_loader, val_loader, test_loader = build_dataloaders(root='./data', batch_size=64)

  images, labels = next(iter(train_loader))
  visualize_image(images, labels, num_images=5, class_names=CIFAR10_CLASSES)

  dropout_rate = 0.2
  lr = 0.001
  num_epochs = 10

  model = CNNcifar10(dropout_rate=dropout_rate).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)

  history = fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

  test_loss, test_acc = evaluate(model, test_loader, criterion, device)
  print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

  run_id = f"lr{lr}_bs64_do{dropout_rate}"
  checkpoint_path = os.path.join(CHECKPOINT_DIR, f"baseline_{run_id}.pt")
  save_checkpoint(model, checkpoint_path, dropout_rate=dropout_rate, lr=lr, batch_size=64, num_epochs=num_epochs, test_acc=test_acc)
  print(f"Saved checkpoint to {checkpoint_path}")

  os.makedirs(RESULTS_DIR, exist_ok=True)
  plot_history(history, out_path=os.path.join(RESULTS_DIR, f"baseline_history_{run_id}.png"))


if __name__ == '__main__':
  main()
