import torch
import torch.nn as nn


class CNNcifar10(nn.Module):

  def __init__(self, dropout_rate=0.2):

    super(CNNcifar10, self).__init__()

    #define convolutional layers
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    # recall -> H_out = floor((H_in + 2·padding - kernel_size) / stride) + 1
    # after conv1 -> (32 + 2*1 - 3)/1 + 1 = 32, i.e. 32x32 in => 32x32 out
    self.batchnorm1 = nn.BatchNorm2d(num_features=32) # num_features must match the previous output
    # normalizes activations across the batch to 0 mean and 1 variance, applies small learnable scale + shift on top -> stabilizes training, gives more accuracy
    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    # after pooling -> floor((H_in + 2·0 - 2) / 2) + 1 = H_in // 2 so, 32x32 in => 16x16 out
    # output dim = batch * 32 * 16 * 16
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.batchnorm2 = nn.BatchNorm2d(num_features=64)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    # output dim = batch * 64 * 8 * 8
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.batchnorm3 = nn.BatchNorm2d(num_features=128)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    # output dim = batch * 128 * 4 * 4 = batch * 2048

    #define fully connected layers
    self.fcn1 = nn.Linear(in_features=2048, out_features=512)
    self.fcn2 = nn.Linear(in_features=512, out_features=128)
    self.fcn3 = nn.Linear(in_features=128, out_features=10)

    #declare dropout layer
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):

    #forward pass through each conv → batchnorm → relu → maxpool
    x = self.batchnorm1(self.conv1(x))
    x = torch.relu(x)
    x = self.maxpool1(x)
    x = self.batchnorm2(self.conv2(x))
    x = torch.relu(x)
    x = self.maxpool2(x)
    x = self.batchnorm3(self.conv3(x))
    x = torch.relu(x)
    x = self.maxpool3(x)

    #flattening tensor before passing to fully connected layer
    x = torch.flatten(x, 1)

    #forward pass through fully connected network
    x = torch.relu(self.fcn1(x))
    x = self.dropout(x)
    x = torch.relu(self.fcn2(x))
    x = self.dropout(x)
    x = self.fcn3(x)

    return x
