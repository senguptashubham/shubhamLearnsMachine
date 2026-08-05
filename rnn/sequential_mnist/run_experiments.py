import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from data import build_dataloader
from model import SequentialMNISTRNN, SequentialMNISTLSTM
from train import train, evaluate
import torch.nn as nn
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

#fixed shared config to be used across all 6 runs
H, lr, batch_size, clip_norm = 64, 1e-3, 64, 5.0
#epochs starting with 5
epochs = 10

#build dataloaders for all 3 crops -> 'nocrop':[], 'crop14':[], 'crop8':[]
# each containing [train_loader, val_loader, test_loader]
dataloaders = build_dataloader(batch_size=batch_size)

#define all 6 configs as list of tuples
configs = [("RNN", SequentialMNISTRNN, "crop8", 64),
           ("RNN", SequentialMNISTRNN, "crop14", 196),
           ("RNN", SequentialMNISTRNN, "nocrop", 784),
           ("LSTM", SequentialMNISTLSTM, "crop8", 64),
           ("LSTM", SequentialMNISTLSTM, "crop14", 196),
           ("LSTM", SequentialMNISTLSTM, "nocrop", 784)]

#run experiment looping through all configs and collect results
results = []
criterion = nn.CrossEntropyLoss()
for label, model_cls, crop, T in configs:
  print(f"running experiment for {label} model with crop: {crop} and T: {T}")
  model = model_cls(input_dim=1, hidden_dim=H, num_classes=10)
  train_loader = dataloaders[crop][0]
  val_loader = dataloaders[crop][1]
  model, history = train(model=model, train_loader=train_loader, val_loader=val_loader, epochs=epochs, lr=lr, clip_norm=clip_norm, device=device)
  test_loader = dataloaders[crop][2]
  test_loss, test_accuracy = evaluate(model=model, dataloader=test_loader, criterion=criterion, device=device)
  #update results
  results.append({
    "label": label,
    "T": T,
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "train_loss_history": history["Training Loss History"],
    "train_accuracy_history": history["Training Accuracy History"],
    "val_loss_history": history["Validation Loss History"],
    "val_accuracy_history": history["Validation Accuracy History"],
  })

#save the results to json
os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
save_path = os.path.join(os.path.dirname(__file__), "results", "experiment_results.json")
with open(save_path, "w") as f:
  json.dump(results, f, indent=2)
print("saved results to:", save_path)
print("run visualize_results.py to plot -- no need to retrain to iterate on the chart")