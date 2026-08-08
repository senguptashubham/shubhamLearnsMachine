import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from shakespeare_data import build_dataloader
from model import CharRNN, CharLSTM
from train import train, evaluate
import torch.nn as nn
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# fixed shared config, used for BOTH runs below -- H/embed_dim must stay
# identical between RNN and LSTM (capacity-matched comparison, same principle
# as Phase 2), and centralizing everything here means changing one value
# actually changes the run, instead of a stale literal elsewhere disagreeing
H, embed_dim, lr, seq_len, batch_size, clip_norm = 128, 32, 1e-3, 100, 64, 5.0
epochs = 25

# vocab_size/mappings come from the data, not hardcoded -- train/val/test all
# share ONE vocabulary built from the full text before splitting
dataloaders, vocab_size, mappings = build_dataloader(seq_len=seq_len, batch_size=batch_size)

configs = [("RNN", CharRNN), ("LSTM", CharLSTM)]

results = []
criterion = nn.CrossEntropyLoss()  # only for the final test-set evaluate() calls below --
                                    # train() creates its own internal criterion for training

for label, model_cls in configs:
  print(f"running experiment for {label} model")
  model = model_cls(hidden_dim=H, embed_dim=embed_dim, vocab_size=vocab_size)
  # train() also saves a best-val-loss checkpoint internally (see train.py's
  # save_checkpoint), named after the model class -- charrnn.pt / charlstm.pt,
  # so the two runs' checkpoints never collide
  model, history = train(model, dataloaders['train_loader'], dataloaders['val_loader'], epochs=epochs, device=device, mappings=mappings, lr=lr, clip_norm=clip_norm)
  # deliberately a SEPARATE call from the per-epoch val evaluation inside
  # train() -- this is the one-time, final test-set number, not something
  # monitored during training
  test_loss, test_perplexity = evaluate(model, dataloaders['test_loader'], criterion, device=device)
  print(f"{label} training completed | test loss {test_loss:.4f} perplexity {test_perplexity:.2f}\n")
  results.append({
      "label": label,
      "test_loss": test_loss,
      "test_perplexity": test_perplexity,
      "train_loss_history": history["Training Loss History"],
      "train_perplexity_history": history["Training Perplexity History"],
      "val_loss_history": history["Validation Loss History"],
      "val_perplexity_history": history["Validation Perplexity History"],
    })

#save the results to json
os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
save_path = os.path.join(os.path.dirname(__file__), "results", "experiment_results.json")
with open(save_path, "w") as f:
  json.dump(results, f, indent=2)
print("saved results to:", save_path)