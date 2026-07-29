import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import matplotlib.pyplot as plt
import csv
import os
from my_conv2d import MyConv2d
from datetime import datetime

RUN_ID = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RESULTS_FILE = f'results/benchmark_{RUN_ID}.csv'

def append_result(result):
  file_exists = os.path.isfile(RESULTS_FILE)
  with open(RESULTS_FILE, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['device', 'size', 'mine', 'ref', 'error'])
    if not file_exists:
      writer.writeheader()
    writer.writerow(result)

def make_conv_pair(in_channels, out_channels, kernel_size, device, padding=0, stride=1, batch=2, H=8, W=8):
  x = torch.randn(batch, in_channels, H, W, device=device)
  my_conv = MyConv2d(in_channels, out_channels, kernel_size, padding, stride).to(device)
  ref_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride).to(device)
  ref_conv.weight.data = my_conv.weight.data.clone()
  ref_conv.bias.data = my_conv.bias.data.clone()
  return x, my_conv, ref_conv

def time_one_config(device, size, repeats=20):
  try:
    x, my_conv, ref_conv = make_conv_pair(in_channels=3, out_channels=32, kernel_size=3, device=device, padding=1, stride=1, batch=2, H=size, W=size)
    timer_mine = benchmark.Timer(stmt='my_conv(x)', globals={'my_conv':my_conv, 'x':x})
    timer_ref = benchmark.Timer(stmt='ref_conv(x)', globals={'ref_conv':ref_conv, 'x':x})
    mean_mine = timer_mine.timeit(repeats).mean
    mean_ref = timer_ref.timeit(repeats).mean
    return {'device':device, 'size':size, 'mine':mean_mine, 'ref':mean_ref, 'error':None}
  except RuntimeError as e:
    return {'device':device, 'size':size, 'mine':None, 'ref':None, 'error':str(e)}
  
def main():
  print("Smoke test (size=8, cpu)...")
  print(time_one_config('cpu', 8))
  devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
  sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
  for device in devices:
    for size in sizes:
      result = time_one_config(device, size)
      print(result)
      append_result(result)
  print("Done. Results in", RESULTS_FILE)

if __name__ == '__main__':
  main()
