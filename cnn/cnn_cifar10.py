import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


#define custom dataset class
class CustomCIFAR10(Dataset):
  def __init__(self, root, train=True, train_frac=0.8, transform=None):
    """
    Custom Dataset for CIFAR10.
    Parameters:
      - root: str, the root directory where the data is stored
      - train: bool, if True, it loads the training set, otherwise the test set
      - transform: callable, a function/transform to apply to the images
    """
    #load cifar 10 dataset using torchvision
    self.cifar10 = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

  def __len__(self):
    """Return the number of samples in the dataset."""
    return len(self.cifar10)
  
  def __getitem__(self, index):
    """
    Args:
    - index: int, index to fetch the sample
    Returns:
    - image: tensor, the image data after applying the transform
    - label: tensor, the label of the image
    """
    #retrieve image and label
    image, label = self.cifar10[index]
    return image, label
  

#define transformation that will be applied
transform = transforms.Compose([
  transforms.ToTensor(),  #convert images to pytorch tensors
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #normalize images to [-1,1]
])

#instantiate custom dataset
train_dataset = CustomCIFAR10(root='./data', train=True, transform=transform)
test_dataset = CustomCIFAR10(root='./data', train=False, transform=transform)

#create dataloader for batching
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataset = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(type(train_loader))

#visualize the image
def visualize_image(images, labels, num_images=5):
  mean = 0.5
  std = 0.5
  plt.figure(figsize=(12, 12))
  for i in range(num_images):
    plt.subplot(1, num_images, i+1)
    image = images[i] * std + mean #undo normalization, back to [0,1]
    image = image.permute(1, 2, 0).numpy() # (c, h, w) -> (h, w, c)
    image = image.clip(0, 1) #guard against float runding outside [0,1]
    plt.imshow(image)
    plt.title(labels[i].item())
    plt.axis('off')
  plt.show()

for images, labels in train_loader:
  visualize_image(images, labels, num_images=5)
  break


#define custm cnn model
# class CNNcifar10(nn.Module):

#   def __init__(self):
#     super(CNNcifar10, self).__init__()
#     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)