import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


#transform to build three sequence variants: CenterCrop(8)/CenterCrop(14)/no-crop
transform_nocrop = transforms.Compose([
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x.flatten())
])
transform_crop14 = transforms.Compose([
  transforms.CenterCrop(14),
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x.flatten())
])
transform_crop8 = transforms.Compose([
  transforms.CenterCrop(8),
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x.flatten())
])


def build_dataloader(root='./data', batch_size=64, train_frac=0.8, seed=42):
  dataloaders = {'nocrop':[], 'crop14':[], 'crop8':[]}
  for key in dataloaders.keys():
    transform = transform_nocrop if key == 'nocrop' else transform_crop14 if key == 'crop14' else transform_crop8
    base_train = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    n_train = int(len(base_train) * train_frac)
    n_val = len(base_train) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(base_train, [n_train, n_val], generator=generator)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    dataloaders[key] = [train_loader, val_loader, test_loader]
  return dataloaders

  