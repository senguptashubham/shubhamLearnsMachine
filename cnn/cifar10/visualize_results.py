import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from cifar_data import build_dataloaders, CIFAR10_CLASSES
from train_engine import load_checkpoint

RESULTS_DIR = 'results'
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

MEAN = 0.5
STD = 0.5


def latest_checkpoint():
  files = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pt'))
  if not files:
    raise FileNotFoundError(f"No checkpoints found under {CHECKPOINT_DIR}")
  return max(files, key=os.path.getmtime)


def run_id_from_checkpoint(checkpoint_path):
  return os.path.splitext(os.path.basename(checkpoint_path))[0]


def unnormalize(image):
  image = image * STD + MEAN
  image = image.permute(1, 2, 0).numpy()
  return image.clip(0, 1)


def collect_predictions(model, test_loader, device):
  """One pass over test_loader. Returns (true_labels, pred_labels) as numpy
  arrays, and a list of (image_tensor, true_label, pred_label) for every
  misclassified example."""
  model.eval()
  all_true = []
  all_pred = []
  misclassified = []

  with torch.no_grad():
    for images, labels in test_loader:
      images_dev, labels_dev = images.to(device), labels.to(device)
      outputs = model(images_dev)
      predicted = outputs.argmax(dim=1)

      all_true.append(labels.numpy())
      all_pred.append(predicted.cpu().numpy())

      wrong_mask = predicted.cpu() != labels
      for image, true_label, pred_label in zip(images[wrong_mask], labels[wrong_mask], predicted.cpu()[wrong_mask]):
        misclassified.append((image, true_label.item(), pred_label.item()))

  true_labels = np.concatenate(all_true)
  pred_labels = np.concatenate(all_pred)
  return true_labels, pred_labels, misclassified


def plot_confusion_matrix(true_labels, pred_labels, class_names, out_path):
  n_classes = len(class_names)
  cm = np.zeros((n_classes, n_classes), dtype=int)
  np.add.at(cm, (true_labels, pred_labels), 1)

  fig, ax = plt.subplots(figsize=(8, 7), facecolor='#fcfcfb')
  im = ax.imshow(cm, cmap='Blues')
  fig.colorbar(im, ax=ax, label='Count')

  ax.set_xticks(range(n_classes))
  ax.set_yticks(range(n_classes))
  ax.set_xticklabels(class_names, rotation=45, ha='right', color='#52514e')
  ax.set_yticklabels(class_names, color='#52514e')
  ax.set_xlabel('Predicted', color='#52514e')
  ax.set_ylabel('True', color='#52514e')
  ax.set_title('Confusion Matrix', color='#0b0b0b')

  # annotate counts, using light/dark text depending on cell darkness for readability
  threshold = cm.max() / 2
  for i in range(n_classes):
    for j in range(n_classes):
      color = 'white' if cm[i, j] > threshold else '#0b0b0b'
      ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=8)

  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.show()


def plot_misclassified_gallery(misclassified, class_names, out_path, per_class_cap=3, grid_size=25, seed=42):
  """Buckets misclassified examples by true class and samples a capped number
  per bucket, so the gallery isn't dominated by whichever class is most-confused."""
  rng = random.Random(seed)
  by_class = {}
  for image, true_label, pred_label in misclassified:
    by_class.setdefault(true_label, []).append((image, true_label, pred_label))

  sampled = []
  for true_label in sorted(by_class):
    bucket = by_class[true_label]
    rng.shuffle(bucket)
    sampled.extend(bucket[:per_class_cap])
  sampled = sampled[:grid_size]

  cols = 5
  rows = (len(sampled) + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), facecolor='#fcfcfb')
  axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

  for ax, (image, true_label, pred_label) in zip(axes, sampled):
    ax.imshow(unnormalize(image))
    ax.set_title(f"true: {class_names[true_label]}\npred: {class_names[pred_label]}", fontsize=8)
    ax.axis('off')
  for ax in axes[len(sampled):]:
    ax.axis('off')

  fig.suptitle('Misclassified examples (sampled per class)', color='#0b0b0b')
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.show()


def main(checkpoint_path=None):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  checkpoint_path = checkpoint_path or latest_checkpoint()
  print(f"Loading checkpoint: {checkpoint_path}")

  model, metadata = load_checkpoint(checkpoint_path, device)
  print(f"Checkpoint metadata: {metadata}")

  _, _, test_loader = build_dataloaders(root='./data', batch_size=metadata.get('batch_size', 64))
  true_labels, pred_labels, misclassified = collect_predictions(model, test_loader, device)

  accuracy = 100 * (true_labels == pred_labels).sum() / len(true_labels)
  print(f"Test accuracy: {accuracy:.2f}% ({len(misclassified)} misclassified out of {len(true_labels)})")

  run_id = run_id_from_checkpoint(checkpoint_path)
  os.makedirs(RESULTS_DIR, exist_ok=True)
  plot_confusion_matrix(true_labels, pred_labels, CIFAR10_CLASSES,
                         out_path=os.path.join(RESULTS_DIR, f'confusion_matrix_{run_id}.png'))
  plot_misclassified_gallery(misclassified, CIFAR10_CLASSES,
                              out_path=os.path.join(RESULTS_DIR, f'misclassified_gallery_{run_id}.png'))


if __name__ == '__main__':
  main()
