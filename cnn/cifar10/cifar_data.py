import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torch.utils.data import Subset

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

EVAL_TRANSFORM = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#data augmentation for training data, so model don't memorize and learn better
TRAIN_TRANSFORM = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  #transforms.RandomRotation(degrees=10), #experimenting to check if accuracy increases 
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CustomCIFAR10(Dataset):
  def __init__(self, root, split='train', train_frac=0.8, transform=None, seed=42):
    """
    Custom Dataset for CIFAR10, with a real train/val split carved out of
    the official training set. The official test set is never touched by
    the split -- it stays held out for final evaluation only.
    Parameters:
      - root: str, the root directory where the data is stored
      - split: 'train' | 'val' | 'test'
      - train_frac: float, fraction of the official training set used for `train`
        (the remainder becomes `val`)
      - transform: callable, a function/transform to apply to the images
      - seed: int, must match between the 'train' and 'val' instances so the two
        splits are complementary (same seed + same sizes => same random_split
        permutation => no overlap, no gaps)
    """
    is_train_file = split in ('train', 'val')
    base = datasets.CIFAR10(root=root, train=is_train_file, download=True, transform=transform)

    if split == 'test':
      self.dataset = base
    else:
      n_train = int(len(base) * train_frac)
      n_val = len(base) - n_train
      generator = torch.Generator().manual_seed(seed)
      train_subset, val_subset = random_split(base, [n_train, n_val], generator=generator)
      self.dataset = train_subset if split == 'train' else val_subset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    image, label = self.dataset[index] # type: ignore[assignment]
    return image, label


def build_dataloaders(root='./data', batch_size=64, train_frac=0.8, seed=42):
  """Builds train/val/test DataLoaders. train_frac splits the official
  training set into train+val; the official test set is untouched."""

  train_dataset = CustomCIFAR10(root=root, split='train', train_frac=train_frac, transform=TRAIN_TRANSFORM, seed=seed)
  val_dataset = CustomCIFAR10(root=root, split='val', train_frac=train_frac, transform=EVAL_TRANSFORM, seed=seed)
  test_dataset = CustomCIFAR10(root=root, split='test', transform=EVAL_TRANSFORM)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader


def visualize_image(images, labels, num_images=5, class_names=None):
  import matplotlib.pyplot as plt
  mean = 0.5
  std = 0.5
  plt.figure(figsize=(12, 12))
  for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    image = images[i] * std + mean  #undo normalization, back to [0,1]
    image = image.permute(1, 2, 0).numpy()  # (c, h, w) -> (h, w, c)
    image = image.clip(0, 1)  #guard against float rounding outside [0,1]
    plt.imshow(image)
    label_idx = labels[i].item()
    title = class_names[label_idx] if class_names else str(label_idx)
    plt.title(title)
    plt.axis('off')
  plt.show()


if __name__ == '__main__':
  train_loader, val_loader, test_loader = build_dataloaders()

  # sanity check: train/val subsets must be disjoint (same seed + same sizes => complementary split)
  train_dataset = train_loader.dataset
  val_dataset = val_loader.dataset
  assert isinstance(train_dataset, CustomCIFAR10) and isinstance(train_dataset.dataset, Subset)
  assert isinstance(val_dataset, CustomCIFAR10) and isinstance(val_dataset.dataset, Subset)
  train_indices = set(train_dataset.dataset.indices)
  val_indices = set(val_dataset.dataset.indices)
  overlap = train_indices & val_indices
  print(f"train size={len(train_indices)}, val size={len(val_indices)}, overlap={len(overlap)}")
  assert len(overlap) == 0, "train/val split overlaps -- seed or sizes mismatch"

  for images, labels in train_loader:
    visualize_image(images, labels, num_images=5, class_names=CIFAR10_CLASSES)
    break
