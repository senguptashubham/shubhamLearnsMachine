import os
from urllib import request
import torch
from torch.utils.data import random_split, DataLoader, Dataset

# anchored to this file's own folder, not the process's current working
# directory -- './data' alone resolves against whatever cwd happens to be
# when the script runs (e.g. repo root if launched from there), silently
# creating a second, duplicate download in the wrong place
DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), 'data')

class TSDataset(Dataset):
  def __init__(self, root=DEFAULT_ROOT, filename='tinyshakespeare.txt', seq_len=20):
    self.root = root
    self.filename = filename
    self.seq_len = seq_len
    self.text = self.load_text(root, filename)
    self.vocab_size, self.char2idx, self.idx2char = self.build_vocab(self.text)
    self.encoded_text = self.encode(self.text, self.char2idx)

  def __len__(self):
    # -1 before the floor division: guarantees start + seq_len + 1 (the target
    # chunk's upper bound) never runs past the end of encoded_text, same
    # boundary condition as chunk()'s old range(0, length-seq_len, seq_len)
    return (len(self.encoded_text) - 1) // self.seq_len

  def __getitem__(self, idx):
    start = idx * self.seq_len
    input_chunk = self.encoded_text[start : start + self.seq_len]
    target_chunk = self.encoded_text[start + 1 : start + self.seq_len + 1]
    return input_chunk, target_chunk

  def load_text(self, root=DEFAULT_ROOT, filename='tinyshakespeare.txt'):
    #download once if it doesn't already exist
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)
    if not os.path.exists(file_path):
      request.urlretrieve(url, file_path)
    text = open(file_path).read()
    return text

  def build_vocab(self, text):
    chars = sorted(set(text))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, char2idx, idx2char

  def encode(self, text, char2idx):
    encoded = [char2idx[ch] for ch in text]
    return torch.LongTensor(encoded)


def build_dataloader(root=DEFAULT_ROOT, filename='tinyshakespeare.txt', seq_len=20, batch_size=64, train_frac=0.8, val_frac=0.1, seed=42):
  dataset = TSDataset(root, filename, seq_len)
  n_total = len(dataset)
  n_train = int(n_total * train_frac)
  n_val = int(n_total * val_frac)
  n_test = n_total - n_train - n_val  # remainder, avoids rounding error dropping/adding a chunk
  generator = torch.Generator().manual_seed(seed)
  # random_split, not slicing -- Dataset objects don't support base_train[:n] the
  # way a list does; this is the same tool Phase 2's data.py used for its split.
  train_dataset, val_dataset, test_dataset = random_split(
    dataset, [n_train, n_val, n_test], generator=generator
  )
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
  dataloaders = {'train_loader':train_loader, 'val_loader':val_loader, 'test_loader':test_loader}
  mappings = {'char2idx':dataset.char2idx, 'idx2char':dataset.idx2char}
  return dataloaders, dataset.vocab_size, mappings


if __name__ == "__main__":
  dataset = TSDataset(seq_len=10)
  print("vocab_size:", dataset.vocab_size)
  print("num chunks:", len(dataset))
  input_chunk, target_chunk = dataset[0]
  print("input_chunk:", input_chunk)
  print("target_chunk:", target_chunk)
  dataloaders, vocab_size, mappings = build_dataloader(seq_len=10, batch_size=4)
  train_loader, val_loader, test_loader = dataloaders['train_loader'], dataloaders['val_loader'], dataloaders['test_loader']
  batch_inputs, batch_targets = next(iter(train_loader))
  print("batch_inputs shape:", batch_inputs.shape)   # expect (batch_size, seq_len)
  print("batch_targets shape:", batch_targets.shape)  # expect (batch_size, seq_len)