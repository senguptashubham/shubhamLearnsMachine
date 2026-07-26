import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import time
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

#step 0: creating a custom multi layer perceptron based on nn.module
class CustomMLP(nn.Module):
  
  def __init__(self, input_size, hidden_sizes, output_size, activation='relu', task_type='classification'):
    
    super(CustomMLP, self).__init__()
    self.activation = activation
    self.task_type = task_type
    self.layers = nn.ModuleList()
    
    #input layer -> first hidden layer
    self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

    #hidden layers
    for i in range(len(hidden_sizes) - 1):
      self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
    
    #last hidden layer -> output layer
    self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

  def forward(self, X):
    for i in range(len(self.layers) - 1):
      if self.activation == 'relu':
        X = F.relu(self.layers[i](X))
      else:
        X = F.tanh(self.layers[i](X))
    if self.task_type == 'classification':
      # binary case -> sigmoid gives a probability
      # multiclass case -> return raw logits; nn.CrossEntropyLoss applies
      # log_softmax internally, so softmax-ing here would apply it twice
      # and flatten the gradients during training
      if self.layers[-1].out_features == 1:
        X = F.sigmoid(self.layers[-1](X))
      else:
        X = self.layers[-1](X)
    else:
      # No activation on output layer for regression tasks
      X = self.layers[-1](X)
    return X

#step 1: configure model  
input_size = 784
hidden_sizes = [200, 150]  # two hidden layers with 20 and 15 neurons
output_size = 10
activation = 'relu'
task_type = 'classification'

model = CustomMLP(input_size, hidden_sizes, output_size, activation, task_type)
print(model)

#step 2: set input
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]).astype('float32') / 255.0
X_train_tensor = torch.from_numpy(X_train)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).astype('float32') / 255.0
X_test_tensor = torch.from_numpy(X_test)
y_train = to_categorical(y_train, num_classes=10)
y_train_tensor = torch.from_numpy(y_train)
y_test = to_categorical(y_test, num_classes=10)
y_test_tensor = torch.from_numpy(y_test)
print(X_train_tensor.shape, y_train_tensor.shape, X_test_tensor.shape, y_test_tensor.shape)

#step 3: define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#step 4: perform a manual pass: zero_grad → forward → loss → backward → step
optimizer.zero_grad()
y_pred_tensor = model(X_train_tensor)
print('shape of output tensor:', y_pred_tensor.shape)
loss = criterion(y_pred_tensor, y_train_tensor)
loss.backward()
optimizer.step()
print("Loss after one pass", loss)

#step 5: wrapping in epoch loop to check how the loss and accuracy changes
for epoch in range (3):
  optimizer.zero_grad()
  y_pred_tensor = model(X_train_tensor)
  loss = criterion(y_pred_tensor, y_train_tensor)
  loss.backward()
  optimizer.step()
  print(f"Loss after epoch {epoch} is {loss}")
  predicted = torch.argmax(y_pred_tensor, dim=1)
  true_labels = torch.argmax(y_train_tensor, dim=1)
  correct = (predicted == true_labels).sum().item()
  total = y_train_tensor.size(0)
  print(f"Accuracy after epoch {epoch} is {100 * correct / total:.2f}%")

#-----------------------------------------------------------------------------
# Step 6: Full-batch vs Mini-batch vs Stochastic Gradient Descent
#
# Only `batch_size` changes between the three runs below - everything else (architecture, optimizer type, learning rate, data) stays identical, so any difference in the curves comes from batch size alone.
#
# Full-batch does 1 weight update per epoch; mini-batch and stochastic (bs=1) do many more. That makes "epoch" an unfair x-axis on its own, so alongside it we track cumulative wall-clock time and plot both, using the same color per run in every panel - that's what ties the three comparisons together.
#-----------------------------------------------------------------------------

def train_one_config(batch_size, num_epochs, X, y, lr=0.001, X_val=None, y_val=None):
  model = CustomMLP(input_size, hidden_sizes, output_size, activation, task_type) #getting from global vars, can be parameterized
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()
  n_samples = X.size(0)
  train_loss_history = []
  train_acc_history = []
  time_history = []
  val_loss_history = []
  val_acc_history = []
  start_time = time.time()
  for epoch in range(num_epochs):
    permutation = torch.randperm(n_samples)
    total_loss = 0
    total_correct = 0
    for start in range(0, n_samples, batch_size):
      batch = permutation[start : start + batch_size]
      X_batch = X[batch]
      y_batch = y[batch]
      # zero_grad → forward → loss → backward → step
      optimizer.zero_grad()
      y_pred_batch = model(X_batch)
      loss_batch = criterion(y_pred_batch, y_batch)
      total_loss += loss_batch.item() * X_batch.size(0)
      loss_batch.backward()
      optimizer.step()
      predicted_batch = torch.argmax(y_pred_batch, dim=1)
      true_labels_batch = torch.argmax(y_batch, dim=1)
      correct_batch = (predicted_batch == true_labels_batch).sum().item()
      total_correct += correct_batch
    total_loss /= n_samples
    print(f"Training Loss after epoch {epoch} is {total_loss}")
    train_loss_history.append(round(total_loss, 4))
    total_accuracy = round(100 * total_correct / n_samples, 2)
    print(f"Training Accuracy after epoch {epoch} is {total_accuracy}%")
    train_acc_history.append(total_accuracy)
    time_history.append(round(time.time() - start_time, 2))
    #switching to evaluation
    model.eval()
    with torch.no_grad():
      y_val_pred = model(X_val)
      val_loss = criterion(y_val_pred, y_val).item()
      print(f"Testing Loss after epoch {epoch} is {val_loss}")
      val_loss_history.append(round(val_loss, 4))
      predicted_val = torch.argmax(y_val_pred, dim=1)
      true_labels_val = torch.argmax(y_val, dim=1)
      correct_val = (predicted_val == true_labels_val).sum().item()
      val_accuracy = round(100 * correct_val / y_val.size(0), 2)
      print(f"Testing Accuracy after epoch {epoch} is {val_accuracy}%")
      val_acc_history.append(val_accuracy)
    #switching back to train for next epoch
    model.train()
  return train_loss_history, train_acc_history, time_history, val_loss_history, val_acc_history

#comparison time: full batch (batch_size = total samples) vs mini batch (batch_size = 64) vs stochastic (batch_size = 1)
results_path = "gd_comparison_results.json"
if os.path.exists(results_path):
  # reuse a previous run instead of retraining (stochastic alone takes ~6 min)
  # delete the json file if you actually want to retrain with new settings
  print(f"Loading cached results from {results_path}")
  with open(results_path) as f:
    results = json.load(f)
else:
  epoch_batch_config = {"Full-batch": (X_train_tensor.size(0), 40), "Mini-batch": (64, 20), "Stochastic": (1, 5)}
  results = {"Full-batch":[], "Mini-batch":[], "Stochastic":[]}
  for batch_type, config in epoch_batch_config.items():
    results[batch_type] = train_one_config(config[0], config[1], X_train_tensor, y_train_tensor, 0.001, X_test_tensor, y_test_tensor)
    print(f"For the Batch type: {batch_type}")
    print(f"Training loss data: {results[batch_type][0]}\nTraining accuracy data:{results[batch_type][1]}\nTraining time data: {results[batch_type][2]}")
    print(f"Testing loss data: {results[batch_type][3]}\nTesting accuracy data:{results[batch_type][4]}\n")

  with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

#plot time
palette = {
    "Full-batch": "#2a78d6",   # blue
    "Mini-batch": "#008300",   # green
    "Stochastic": "#eb6834",   # orange
}

fig, ((loss_vs_epoch, loss_vs_time), (accuracy_vs_epoch, accuracy_vs_time)) = plt.subplots(2, 2, figsize=(12,8))
loss_vs_epoch.set(title="Loss vs Epoch", xlabel="epoch", ylabel="loss")
loss_vs_epoch.set_xscale('log')
loss_vs_epoch.set_yscale('log')
loss_vs_time.set(title="Loss vs Time", xlabel="time", ylabel="loss")
loss_vs_time.set_xscale('log')
loss_vs_time.set_yscale('log')
accuracy_vs_epoch.set(title="Accuracy vs Epoch", xlabel="epoch", ylabel="accuracy")
accuracy_vs_epoch.set_xscale('log')
accuracy_vs_time.set(title="Accuracy vs Time", xlabel="time", ylabel="accuracy")
accuracy_vs_time.set_xscale('log')
for config_name, data in results.items():
  color = palette[config_name]
  loss_vs_epoch.plot(range(1, len(data[0])+1), data[0], color=color, label=config_name)
  loss_vs_time.plot(data[2], data[0], color=color, label=config_name)
  accuracy_vs_epoch.plot(range(1, len(data[0])+1), data[1], color=color, label=config_name)
  accuracy_vs_time.plot(data[2], data[1], color=color, label=config_name)

for ax in (loss_vs_epoch, loss_vs_time, accuracy_vs_epoch, accuracy_vs_time):
  ax.grid(True, which="both", ls="--", alpha=0.3)

handles, labels = loss_vs_epoch.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
plt.savefig("gd_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

#-----------------------------------------------------------------------------
# Chart B: Train vs Validation per config - checking for overfitting
#
# Each config gets its own column here (same epoch count for train/val within
# a column), so unlike Chart A there's no cross-config epoch mismatch - a
# plain linear epoch axis is fine. Color now encodes "train vs validation",
# not batch-size config, since config identity is carried by the column title.
#-----------------------------------------------------------------------------
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
train_color = "#2a78d6"  # blue
val_color = "#e34948"    # red

for col, (config_name, data) in enumerate(results.items()):
  train_loss, train_acc, time_hist, val_loss, val_acc = data
  epochs = range(1, len(train_loss) + 1)
  # the epoch where validation loss was lowest - i.e. where early stopping
  # would have stopped training, before the model started overfitting
  best_epoch = val_loss.index(min(val_loss)) + 1

  ax_loss = axes2[0, col]
  ax_loss.plot(epochs, train_loss, color=train_color, label="Train")
  ax_loss.plot(epochs, val_loss, color=val_color, label="Validation")
  ax_loss.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
  ax_loss.set_yscale('log')
  ax_loss.set(title=config_name, xlabel="epoch", ylabel="loss")
  ax_loss.grid(True, which="both", ls="--", alpha=0.3)
  ax_loss.text(best_epoch, ax_loss.get_ylim()[1], f" early stop: epoch {best_epoch}",
               rotation=90, va="top", ha="right", fontsize=8, color="gray")

  ax_acc = axes2[1, col]
  ax_acc.plot(epochs, train_acc, color=train_color, label="Train")
  ax_acc.plot(epochs, val_acc, color=val_color, label="Validation")
  ax_acc.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
  ax_acc.set(xlabel="epoch", ylabel="accuracy")
  ax_acc.grid(True, ls="--", alpha=0.3)

handles, labels = axes2[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
fig2.suptitle("Train vs Validation — Overfitting Check", y=1.08, fontsize=14)
fig2.tight_layout()
plt.savefig("gd_overfitting_check.png", dpi=150, bbox_inches="tight")
plt.show()